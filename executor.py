"""
Trade Executor – the glue between strategy signals and the broker.
Handles the full lifecycle: scan → signal → size → order → track.
"""

from __future__ import annotations

import datetime as dt

import config
from broker import AlpacaBroker
from indicators import compute_all, compute_weekly_trend, realized_volatility
from pdt_guard import PDTGuard
from risk_manager import RiskManager
from screener import Screener
from strategy import check_entry, check_exit
from logger import get_logger

log = get_logger("executor")


class TradeExecutor:
    """Orchestrates a single scan-and-act cycle."""

    def __init__(self) -> None:
        self.broker = AlpacaBroker()
        self.pdt = PDTGuard(broker=self.broker)
        self.screener = Screener(self.broker)
        self._init_risk_manager()
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

        closed = 0
        for pos in positions:
            symbol = pos.symbol
            qty = float(pos.qty)

            # PDT check – can we sell today (respects day trade count)?
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
            signal = check_exit(df, entry_price)

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

        # Market regime filter: don't open new longs in a bear market
        spy_df = None
        if config.MARKET_REGIME_ENABLED:
            try:
                spy_df = self.broker.get_bars(config.MARKET_REGIME_SYMBOL, limit=250)
                if spy_df is not None and len(spy_df) >= 200:
                    from indicators import compute_all as _compute
                    spy_df = _compute(spy_df)
                    spy_row = spy_df.iloc[-1]
                    spy_ema200 = spy_row.get("ema_200", None)
                    if spy_ema200 is not None and spy_row["close"] < spy_ema200:
                        log.info("BEAR MARKET - %s below EMA-200, skipping entries",
                                 config.MARKET_REGIME_SYMBOL)
                        return 0
            except Exception as exc:
                log.warning("Market regime check failed: %s", exc)

        # Advanced features: vol regime + dynamic threshold
        vol_scale = self._get_vol_regime_scale()
        dyn_threshold = self._get_dynamic_threshold(spy_df)

        if dyn_threshold != config.ENTRY_SCORE_THRESHOLD:
            log.info("Dynamic threshold: %d (base %d)", dyn_threshold, config.ENTRY_SCORE_THRESHOLD)

        # Symbols we already hold - skip them
        held = {p.symbol for p in self.broker.get_positions()}
        # Symbols with pending buy orders - skip them too
        pending = {o.symbol for o in self.broker.get_open_orders() if o.side == "buy"}

        candidates = self.screener.screen()
        opened = 0

        for c in candidates:
            symbol = c["symbol"]
            if symbol in held or symbol in pending:
                continue

            if not self.risk.can_open_new_position():
                break

            if not self.pdt.can_buy_today(symbol):
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
            if config.WEEKLY_TREND_ENABLED:
                try:
                    df_full = c.get("df")
                    if df_full is not None and len(df_full) > config.WEEKLY_EMA_SLOW * 5:
                        wt = compute_weekly_trend(df_full)
                        weekly_bull = wt["bullish"]
                except Exception:
                    pass

            signal = check_entry(c["df"], weekly_bullish=weekly_bull)
            if signal is None:
                continue

            # Apply dynamic threshold manually (check_entry uses config default)
            if signal["score"] < dyn_threshold:
                continue

            entry_price = signal["price"]
            atr = signal["atr"]
            stop_loss = self.risk.compute_stop_loss(entry_price, atr)
            take_profit = self.risk.compute_take_profit(entry_price, atr)

            qty = self.risk.calculate_position_size(
                entry_price=entry_price,
                stop_price=stop_loss,
                buying_power=self.broker.get_buying_power(),
            )

            # Apply volatility regime scaling
            if vol_scale != 1.0 and qty > 0:
                qty = round(qty * vol_scale, 3) if config.FRACTIONAL_SHARES else int(qty * vol_scale)
                log.info("Vol regime scale: %.2f -> qty adjusted to %.3f", vol_scale, qty)

            if qty == 0:
                log.info("Position size = 0 for %s - skipping", symbol)
                continue

            # Alpaca minimum order value is $1 — skip if notional is too small
            MIN_ORDER_NOTIONAL = 1.0
            if qty * entry_price < MIN_ORDER_NOTIONAL:
                log.info(
                    "Skipping %s – order value $%.2f below minimum $%.2f",
                    symbol, qty * entry_price, MIN_ORDER_NOTIONAL,
                )
                continue

            log.info(
                "ENTRY %s  qty=%.3f  price=%.2f  SL=%.2f  TP=%.2f  [%s]",
                symbol,
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
    # MORNING TASKS – run once per day shortly after market open
    # ─────────────────────────────────────────────────────────
    def morning_tasks(self) -> None:
        """
        Housekeeping that should run once each trading day after market open:

        1. Resubmit stop-loss orders that expired at yesterday's close
           (fractional stop orders use time_in_force="day" and expire daily).
        2. Clean up stale PDT ledger entries.
        3. Log the day-trade budget remaining.
        """
        log.info("--- Morning tasks start ---")

        # 1. Stop-loss resubmission
        n = self.broker.resubmit_stop_losses(self.pdt)
        if n:
            log.info("Morning tasks: resubmitted %d stop-loss order(s)", n)

        # 2. Ledger hygiene
        active = {p.symbol for p in self.broker.get_positions()}
        self.pdt.cleanup_stale(active)

        # 3. PDT budget report
        used = self.pdt._rolling_day_trade_count()
        remaining = max(0, config.MAX_DAY_TRADES_ALLOWED - used)
        log.info(
            "Morning tasks: day-trade budget — %d used / %d allowed / %d remaining (rolling %d-day window)",
            used, config.MAX_DAY_TRADES_ALLOWED, remaining, config.PDT_LOOKBACK_DAYS,
        )

        log.info("--- Morning tasks done ---")

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
