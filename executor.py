"""
Trade Executor – the glue between strategy signals and the broker.
Handles the full lifecycle: scan → signal → size → order → track.
"""

from __future__ import annotations

import datetime as dt

import config
from broker import AlpacaBroker
from entry_filters import EarningsCalendar
from indicators import compute_all, compute_weekly_trend, realized_volatility
from pdt_guard import PDTGuard
from risk_manager import RiskManager
from screener import Screener
from strategy import (
    check_entry,
    check_exit,
    classify_market_regime,
    compute_momentum,
)
from logger import get_logger

log = get_logger("executor")


class TradeExecutor:
    """Orchestrates a single scan-and-act cycle."""

    def __init__(self) -> None:
        self.broker = AlpacaBroker()
        self.pdt = PDTGuard()
        self.screener = Screener(self.broker)
        self.earnings_calendar = EarningsCalendar()
        self._init_risk_manager()
        self._last_regime = "bull"
        self._sector_counts: dict[str, int] = {}
        self._rebuild_sector_counts()

    def _init_risk_manager(self) -> None:
        equity = self.broker.get_equity()
        n_positions = len(self.broker.get_positions())
        self.risk = RiskManager(equity, n_positions)
        log.info(
            "Account: $%.2f  (%s mode, max %d positions, %.0f%% per position)",
            equity,
            "small" if equity < config.SMALL_ACCOUNT_THRESHOLD else "normal",
            config.get_max_positions(equity),
            config.get_position_pct(equity) * 100,
        )

    def refresh(self) -> None:
        """Refresh equity / position count before each cycle."""
        self._init_risk_manager()
        self._rebuild_sector_counts()

    def _rebuild_sector_counts(self) -> None:
        """Rebuild sector exposure counts from current positions."""
        self._sector_counts = {}
        for pos in self.broker.get_positions():
            sector = config.SECTOR_MAP.get(pos.symbol, "Other")
            self._sector_counts[sector] = self._sector_counts.get(sector, 0) + 1

    def _get_vol_regime_scale(self) -> float:
        """Compute position-size scale factor from SPY realized vol."""
        if not config.VOL_REGIME_ENABLED:
            return 1.0
        try:
            spy_df = self.broker.get_bars(config.MARKET_REGIME_SYMBOL, limit=50)
            if spy_df is None or len(spy_df) < config.REALIZED_VOL_WINDOW + 1:
                return 1.0
            vol = realized_volatility(spy_df, window=config.REALIZED_VOL_WINDOW)
            if vol > config.HIGH_VOL_THRESHOLD:
                return config.HIGH_VOL_SIZE_SCALE
            if vol < config.LOW_VOL_THRESHOLD:
                return config.LOW_VOL_SIZE_SCALE
        except Exception as exc:
            log.warning("Vol regime check failed: %s", exc)
        return 1.0

    def _get_dynamic_threshold(self, spy_df=None) -> int:
        """Adjust entry score threshold based on market quality."""
        base = config.ENTRY_SCORE_THRESHOLD
        if not config.DYNAMIC_THRESHOLD_ENABLED:
            return base
        try:
            if spy_df is None:
                spy_df = self.broker.get_bars(config.MARKET_REGIME_SYMBOL, limit=250)
            if spy_df is None or len(spy_df) < config.EMA_TREND + 5:
                return base
            spy_df = compute_all(spy_df)
            row = spy_df.iloc[-1]
            spy_close = row["close"]
            spy_ema50 = row.get("ema_trend", None)
            if spy_ema50 is None:
                return base
            if len(spy_df) >= config.EMA_SLOPE_PERIOD + 1:
                ema50_ago = spy_df.iloc[-(config.EMA_SLOPE_PERIOD + 1)].get("ema_trend", None)
                if ema50_ago is not None:
                    if spy_close > spy_ema50 and spy_ema50 > ema50_ago:
                        return base - 1  # strong market: lower bar
        except Exception as exc:
            log.warning("Dynamic threshold check failed: %s", exc)
        return base

    def _get_market_regime(self, spy_df=None) -> str:
        """Return current market regime: bull or bear (with hysteresis)."""
        if not config.MARKET_REGIME_ENABLED:
            return "bull"
        try:
            if spy_df is None:
                spy_df = self.broker.get_bars(config.MARKET_REGIME_SYMBOL, limit=300)
            if spy_df is None or len(spy_df) < config.EMA_LONG + 5:
                return self._last_regime
            spy_df = compute_all(spy_df)
            self._last_regime = classify_market_regime(spy_df, last_regime=self._last_regime)
            return self._last_regime
        except Exception as exc:
            log.warning("Market regime check failed: %s", exc)
            return self._last_regime

    def _passes_breadth_filter(self, regime: str) -> bool:
        """Apply optional market breadth gate before scanning entries."""
        if not config.BREADTH_FILTER_ENABLED:
            return True

        breadth = self.screener.compute_breadth(config.WATCHLIST)
        if breadth is None:
            log.info("Breadth filter skipped (insufficient usable symbols)")
            return True

        if regime == "bull":
            if breadth < config.BREADTH_MIN_BULL:
                log.info(
                    "Breadth gate blocked bull entries: breadth=%.1f%% < %.1f%%",
                    breadth * 100,
                    config.BREADTH_MIN_BULL * 100,
                )
                return False
            return True

        if regime == "bear":
            if breadth > config.BREADTH_MAX_BEAR:
                log.info(
                    "Breadth gate blocked bear entries: breadth=%.1f%% > %.1f%%",
                    breadth * 100,
                    config.BREADTH_MAX_BEAR * 100,
                )
                return False
            return True

        return False

    # ─────────────────────────────────────────────────────────
    # EXIT SCAN – check existing positions for exit signals
    # ─────────────────────────────────────────────────────────
    def scan_exits(self) -> int:
        """
        Iterate over open positions, check exit signals, and sell
        where appropriate (respecting PDT guard).
        Returns the number of positions closed.
        """
        positions = self.broker.get_positions()
        active_symbols = {p.symbol for p in positions}
        self.pdt.cleanup_stale(active_symbols)
        market_regime = self._get_market_regime()

        closed = 0
        for pos in positions:
            symbol = pos.symbol
            qty = int(pos.qty)
            hold_days = self.pdt.days_held(symbol) or 0

            # PDT check – can we sell today?
            if not self.pdt.can_sell_today(symbol):
                days = self.pdt.days_held(symbol)
                log.info(
                    "Skipping exit check for %s – held %s day(s), PDT blocked",
                    symbol,
                    days,
                )
                continue

            # Fetch fresh data and compute indicators
            try:
                df = self.broker.get_bars(symbol)
                if df is None or len(df) < config.EMA_TREND + 5:
                    continue
                df = compute_all(df)
            except Exception as exc:
                log.warning("Data error for %s: %s", symbol, exc)
                continue

            entry_price = float(pos.avg_entry_price)
            signal = check_exit(
                df,
                entry_price=entry_price,
                hold_days=hold_days,
                regime=market_regime,
            )

            if signal is not None:
                log.info(
                    "EXIT  %s  qty=%d  entry=%.2f  now=%.2f  reasons=%s",
                    symbol,
                    qty,
                    entry_price,
                    signal["price"],
                    signal["reasons"],
                )
                try:
                    # Cancel any existing orders for the symbol first
                    for order in self.broker.get_open_orders(symbol):
                        self.broker.api.cancel_order(order.id)

                    self.broker.submit_market_sell(symbol, qty)
                    self.pdt.record_sell(symbol)
                    closed += 1
                except Exception as exc:
                    log.error("Sell order failed for %s: %s", symbol, exc)
            else:
                # If position is profitable, consider adding trailing stop
                current_price = float(pos.current_price)
                unrealised_pct = (current_price - entry_price) / entry_price

                # Determine which trailing stop level applies
                if unrealised_pct >= config.TRAILING_STOP_TIGHT_ACTIVATE:
                    trail_pct = config.TRAILING_STOP_TIGHT_PCT
                elif unrealised_pct >= config.TRAILING_STOP_ACTIVATE_PCT:
                    trail_pct = config.TRAILING_STOP_PCT
                else:
                    trail_pct = None

                if trail_pct is not None:
                    # Check if trailing stop already exists
                    open_orders = self.broker.get_open_orders(symbol)
                    has_trailing = any(
                        o.type == "trailing_stop" for o in open_orders
                    )
                    if not has_trailing:
                        log.info(
                            "Adding trailing stop for %s (%.1f%% profit, trail=%.1f%%)",
                            symbol,
                            unrealised_pct * 100,
                            trail_pct * 100,
                        )
                        try:
                            self.broker.submit_trailing_stop(
                                symbol, qty, trail_pct
                            )
                        except Exception as exc:
                            log.warning("Trailing stop failed for %s: %s", symbol, exc)

        log.info("Exit scan complete – closed %d position(s)", closed)
        return closed

    # ─────────────────────────────────────────────────────────
    # ENTRY SCAN – look for new swing-trade setups
    # ─────────────────────────────────────────────────────────
    def scan_entries(self) -> int:
        """
        Screen the watchlist, evaluate entry signals, size positions,
        and submit buy orders. Uses v4 advanced filters.
        Returns the number of new positions opened.
        """
        self.refresh()

        if not self.risk.can_open_new_position():
            log.info("Max positions reached - skipping entry scan")
            return 0

        # Market regime classification
        spy_df = None
        if config.MARKET_REGIME_ENABLED:
            try:
                spy_df = self.broker.get_bars(config.MARKET_REGIME_SYMBOL, limit=300)
            except Exception:
                spy_df = None

        regime = self._get_market_regime(spy_df)
        if regime == "risk_off":
            log.info("RISK-OFF regime detected - skipping new entries")
            return 0

        if regime == "bear" and not config.BEAR_STRATEGY_ENABLED:
            log.info("BEAR MARKET detected, bear strategy disabled - skipping entries")
            return 0

        if not self._passes_breadth_filter(regime):
            return 0

        # Advanced features: vol regime + dynamic threshold
        vol_scale = self._get_vol_regime_scale() if regime == "bull" else 1.0
        dyn_threshold = self._get_dynamic_threshold(spy_df) if regime == "bull" else config.BEAR_ENTRY_SCORE_THRESHOLD

        if regime == "bull" and dyn_threshold != config.ENTRY_SCORE_THRESHOLD:
            log.info("Dynamic threshold: %d (base %d)", dyn_threshold, config.ENTRY_SCORE_THRESHOLD)

        entry_symbols = config.BEAR_WATCHLIST if regime == "bear" else config.WATCHLIST
        if regime == "bear":
            current_positions = len(self.broker.get_positions())
            if current_positions >= config.BEAR_MAX_POSITIONS_CAP:
                log.info("Bear cap reached (%d/%d) - skipping new entries", current_positions, config.BEAR_MAX_POSITIONS_CAP)
                return 0
            log.info("Regime=%s, scanning %d bear symbols", regime, len(entry_symbols))
        else:
            log.info("Regime=%s, scanning %d symbols", regime, len(entry_symbols))

        # Symbols we already hold - skip them
        held = {p.symbol for p in self.broker.get_positions()}
        # Symbols with pending buy orders - skip them too
        pending = {o.symbol for o in self.broker.get_open_orders() if o.side == "buy"}

        candidates = self.screener.screen(entry_symbols)
        opened = 0
        today = dt.date.today()

        for c in candidates:
            symbol = c["symbol"]
            if symbol in held or symbol in pending:
                continue

            if not self.risk.can_open_new_position():
                break

            if regime == "bear" and (len(held) + opened) >= config.BEAR_MAX_POSITIONS_CAP:
                break

            if not self.pdt.can_buy_today(symbol):
                continue

            if self.earnings_calendar.is_blocked(symbol, today):
                log.info("Skipping %s - in earnings blackout window", symbol)
                continue

            # Sector exposure limit
            sector = config.SECTOR_MAP.get(symbol, "Other")
            if self._sector_counts.get(sector, 0) >= config.MAX_PER_SECTOR:
                log.info("Sector %s full (%d/%d) - skipping %s",
                         sector, self._sector_counts.get(sector, 0),
                         config.MAX_PER_SECTOR, symbol)
                continue

            # Weekly trend check
            weekly_bull = True
            if regime == "bull" and config.WEEKLY_TREND_ENABLED:
                try:
                    df_full = c.get("df")
                    if df_full is not None and len(df_full) > config.WEEKLY_EMA_SLOW * 5:
                        wt = compute_weekly_trend(df_full)
                        weekly_bull = wt["bullish"]
                except Exception:
                    pass

            momentum = compute_momentum(c["df"])
            signal = check_entry(
                c["df"],
                weekly_bullish=weekly_bull,
                regime=regime,
                momentum=momentum,
                score_threshold=dyn_threshold,
            )
            if signal is None:
                continue

            entry_price = signal["price"]
            atr = signal["atr"]

            atr_pct = atr / entry_price if entry_price > 0 else float("inf")
            max_atr_pct = config.MAX_ATR_PCT_BEAR if regime == "bear" else config.MAX_ATR_PCT_BULL
            if atr_pct > max_atr_pct:
                continue

            if regime == "bear":
                stop_loss = round(entry_price - atr * config.BEAR_ATR_STOP_MULTIPLIER, 2)
                take_profit = round(entry_price + atr * config.BEAR_ATR_PROFIT_MULTIPLIER, 2)
            else:
                stop_loss = self.risk.compute_stop_loss(entry_price, atr)
                take_profit = self.risk.compute_take_profit(entry_price, atr)

            risk_per_share = entry_price - stop_loss
            reward_per_share = take_profit - entry_price
            if risk_per_share <= 0 or reward_per_share <= 0:
                continue
            rr_ratio = reward_per_share / risk_per_share
            min_rr = config.MIN_RR_RATIO_BEAR if regime == "bear" else config.MIN_RR_RATIO_BULL
            if rr_ratio < min_rr:
                continue

            qty = self.risk.calculate_position_size(
                entry_price=entry_price,
                stop_price=stop_loss,
                buying_power=self.broker.get_buying_power(),
            )

            # Apply volatility regime scaling
            if vol_scale != 1.0 and qty > 0:
                qty = round(qty * vol_scale, 3) if config.FRACTIONAL_SHARES else int(qty * vol_scale)
                log.info("Vol regime scale: %.2f -> qty adjusted to %.3f", vol_scale, qty)

            if regime == "bear" and qty > 0:
                qty = round(qty * config.BEAR_MAX_POSITION_SCALE, 3) if config.FRACTIONAL_SHARES else int(qty * config.BEAR_MAX_POSITION_SCALE)

            if qty == 0:
                log.info("Position size = 0 for %s - skipping", symbol)
                continue

            log.info(
                "ENTRY %s  regime=%s  qty=%.3f  price=%.2f  SL=%.2f  TP=%.2f  [%s]",
                symbol,
                regime,
                qty,
                entry_price,
                stop_loss,
                take_profit,
                signal["reason"],
            )

            try:
                self.broker.submit_market_buy(
                    symbol, qty, stop_loss=stop_loss, take_profit=take_profit
                )
                self.pdt.record_buy(symbol)
                opened += 1
                # Update counter so risk manager knows
                self.risk.open_positions += 1
                # Track sector
                self._sector_counts[sector] = self._sector_counts.get(sector, 0) + 1
            except Exception as exc:
                log.error("Buy order failed for %s: %s", symbol, exc)

        log.info("Entry scan complete - opened %d position(s)", opened)
        return opened

    # ─────────────────────────────────────────────────────────
    # FULL CYCLE
    # ─────────────────────────────────────────────────────────
    def run_cycle(self) -> None:
        """Execute one full scan cycle: exits first, then entries."""
        if not self.broker.is_market_open():
            log.info("Market is closed – skipping cycle")
            return

        log.info("=" * 60)
        log.info("CYCLE START  equity=$%.2f  positions=%d",
                 self.broker.get_equity(),
                 len(self.broker.get_positions()))
        log.info("=" * 60)

        self.scan_exits()
        self.scan_entries()

        # Summary
        positions = self.broker.get_positions()
        equity = self.broker.get_equity()
        log.info(
            "CYCLE END  equity=$%.2f  positions=%d  symbols=%s",
            equity,
            len(positions),
            [p.symbol for p in positions],
        )
