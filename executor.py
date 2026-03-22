"""
Trade Executor – the glue between strategy signals and the broker.
Handles the full lifecycle: scan → signal → size → order → track.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

import config
from broker import AlpacaBroker
from indicators import compute_all, compute_weekly_trend, realized_volatility
from pdt_guard import PDTGuard
from risk_manager import RiskManager
from screener import Screener
from strategy import check_entry, check_exit, check_short_entry, check_short_exit, check_inverse_entry
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
    # MARKET REGIME DETECTION
    # ─────────────────────────────────────────────────────────
    def _detect_regime(self) -> tuple:
        """
        Robustly detect bull vs bear market regime.

        Returns (bear_market: bool, spy_df: DataFrame | None).

        Detection logic:
          1. Fetch SPY daily bars (up to 250).
          2. If we get >= 200 bars, use EMA-200 as the regime line.
          3. If we get 50-199 bars, fall back to EMA-50 (degraded but still useful).
          4. If we get < 50 bars, log a warning and default to BULL
             (safe — avoids opening shorts when we can't confirm the regime).

        A bear market is declared when SPY closes below the chosen EMA.
        """
        bear_market = False
        spy_df = None

        if not config.MARKET_REGIME_ENABLED:
            log.debug("Market regime filter disabled — defaulting to bull mode")
            return bear_market, spy_df

        try:
            spy_df = self.broker.get_bars(config.MARKET_REGIME_SYMBOL, limit=250)
        except Exception as exc:
            log.warning(
                "Failed to fetch %s bars for regime detection: %s — defaulting to bull mode",
                config.MARKET_REGIME_SYMBOL, exc,
            )
            return bear_market, spy_df

        if spy_df is None or len(spy_df) < 50:
            log.warning(
                "Regime detection: only got %d %s bars (need >= 50) — "
                "defaulting to bull mode (not enough data to detect bear regime)",
                len(spy_df) if spy_df is not None else 0,
                config.MARKET_REGIME_SYMBOL,
            )
            return bear_market, spy_df

        try:
            spy_df = compute_all(spy_df)
        except Exception as exc:
            log.warning("Regime detection: indicator computation failed: %s — defaulting to bull", exc)
            return bear_market, spy_df

        spy_row = spy_df.iloc[-1]
        spy_close = spy_row["close"]

        # Use live quote price for regime detection so we react to intraday
        # drops immediately, rather than waiting for the daily bar to close.
        try:
            live_price = self.broker.get_latest_price(config.MARKET_REGIME_SYMBOL)
            if live_price and live_price > 0:
                log.debug(
                    "Regime: using live %s price $%.2f (last bar close $%.2f)",
                    config.MARKET_REGIME_SYMBOL, live_price, spy_close,
                )
                spy_close = live_price
        except Exception as exc:
            log.debug("Could not fetch live %s price, using bar close: %s",
                      config.MARKET_REGIME_SYMBOL, exc)

        # Primary: use EMA-200 if we have enough bars
        spy_ema200 = spy_row.get("ema_200", None)
        if spy_ema200 is not None and not (hasattr(spy_ema200, '__class__') and spy_ema200 != spy_ema200):
            # spy_ema200 != spy_ema200 is a NaN check for numpy
            if not pd.isna(spy_ema200):
                if spy_close < spy_ema200:
                    bear_market = True
                    log.info(
                        "REGIME: BEAR — %s close $%.2f < EMA-200 $%.2f  (bars=%d)",
                        config.MARKET_REGIME_SYMBOL, spy_close, spy_ema200, len(spy_df),
                    )
                else:
                    log.info(
                        "REGIME: BULL — %s close $%.2f >= EMA-200 $%.2f  (bars=%d)",
                        config.MARKET_REGIME_SYMBOL, spy_close, spy_ema200, len(spy_df),
                    )
                return bear_market, spy_df

        # Fallback: use EMA-50 if EMA-200 is unavailable (not enough bars)
        spy_ema50 = spy_row.get("ema_trend", None)
        if spy_ema50 is not None and not pd.isna(spy_ema50):
            if spy_close < spy_ema50:
                bear_market = True
                log.warning(
                    "REGIME: BEAR (degraded) — %s close $%.2f < EMA-50 $%.2f  "
                    "(only %d bars; EMA-200 unavailable — using EMA-50 fallback)",
                    config.MARKET_REGIME_SYMBOL, spy_close, spy_ema50, len(spy_df),
                )
            else:
                log.info(
                    "REGIME: BULL (degraded) — %s close $%.2f >= EMA-50 $%.2f  "
                    "(only %d bars; EMA-200 unavailable — using EMA-50 fallback)",
                    config.MARKET_REGIME_SYMBOL, spy_close, spy_ema50, len(spy_df),
                )
            return bear_market, spy_df

        log.warning(
            "Regime detection: could not compute any EMA for %s (%d bars) — defaulting to bull",
            config.MARKET_REGIME_SYMBOL, len(spy_df),
        )
        return bear_market, spy_df

    def _is_short_position(self, pos) -> bool:
        """Return True if the position is a short (negative qty)."""
        qty = float(pos.qty)
        return qty < 0

    # ─────────────────────────────────────────────────────────
    # EXIT SCAN – check existing positions for exit signals
    # ─────────────────────────────────────────────────────────
    def scan_exits(self) -> int:
        """
        Iterate over open positions, check exit signals, and sell/cover
        where appropriate (respecting PDT guard).
        Handles both LONG positions (sell) and SHORT positions (buy-to-cover).
        Returns the number of positions closed.
        """
        positions = self.broker.get_positions()
        active_symbols = {p.symbol for p in positions}
        self.pdt.cleanup_stale(active_symbols)

        closed = 0
        for pos in positions:
            symbol = pos.symbol
            qty = float(pos.qty)
            is_short = qty < 0
            abs_qty = abs(qty)

            # PDT check – can we sell/cover today?
            if not self.pdt.can_sell_today(symbol):
                days = self.pdt.days_held(symbol)
                log.info(
                    "Skipping exit check for %s – held %s day(s), PDT blocked",
                    symbol, days,
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
            current_price = float(pos.current_price)
            hold_days = self.pdt.days_held(symbol) or 0

            # Use the appropriate exit checker for long vs short
            if is_short:
                signal = check_short_exit(df, entry_price, hold_days=hold_days, explain=True)
            else:
                signal = check_exit(df, entry_price, hold_days=hold_days, explain=True)

            if signal.get("should_exit"):
                action_word = "COVER" if is_short else "EXIT"
                log.info(
                    "%s  %s  qty=%s  entry=%.2f  now=%.2f  reasons=%s",
                    action_word, symbol, abs_qty, entry_price,
                    signal["price"], signal["reasons"],
                )
                try:
                    # Cancel any existing orders for the symbol first
                    open_orders = self.broker.get_open_orders(symbol)
                    for order in open_orders:
                        self.broker.api.cancel_order(order.id)

                    # Re-fetch position to avoid over-selling/covering
                    fresh_pos = self.broker.get_position(symbol)
                    if fresh_pos is None:
                        log.info("Position %s already closed (stop filled?) – skipping", symbol)
                        closed += 1
                        self.pdt.record_sell(symbol)
                        continue
                    abs_qty = abs(float(fresh_pos.qty))

                    if is_short:
                        self.broker.submit_market_cover(symbol, abs_qty)
                    else:
                        self.broker.submit_market_sell(symbol, abs_qty)
                    self.pdt.record_sell(symbol)
                    closed += 1
                except Exception as exc:
                    log.error("%s order failed for %s: %s",
                              "Cover" if is_short else "Sell", symbol, exc)
            else:
                if is_short:
                    # For short positions: profit when price drops
                    unrealised_pct = (entry_price - current_price) / entry_price
                else:
                    unrealised_pct = (current_price - entry_price) / entry_price

                log.info(
                    "HOLD  %s  qty=%s  side=%s  entry=%.2f  now=%.2f  held=%dd  "
                    "reason=%s  (hard=%s, soft=%s)",
                    symbol, qty, "SHORT" if is_short else "LONG",
                    entry_price, current_price, hold_days,
                    signal.get("hold_reason", "None given"),
                    signal.get("hard_reasons", []),
                    signal.get("soft_reasons", []),
                )

                # Trailing stop logic — only for LONG positions
                # (Short trailing stops use buy-to-cover stops and are
                #  managed separately via the stop-loss resubmission logic)
                if not is_short:
                    self._manage_trailing_stop(pos, unrealised_pct, current_price, abs_qty)

        log.info("Exit scan complete – closed %d position(s)", closed)
        return closed

    def _manage_trailing_stop(self, pos, unrealised_pct: float,
                              current_price: float, qty: float) -> None:
        """Manage trailing stop orders for a LONG position (Chandelier Exit).

        Uses ATR-based dynamic stops: stop = high − (ATR × mult).
        Falls back to static percentage if ATR is unavailable.
        """
        symbol = pos.symbol

        # Determine which trailing stop tier applies
        if unrealised_pct >= config.TRAILING_STOP_TIGHT_ACTIVATE:
            fallback_pct = config.TRAILING_STOP_TIGHT_PCT
            atr_mult = getattr(config, "ATR_TRAILING_STOP_TIGHT_MULT", 1.0)
        elif unrealised_pct >= config.TRAILING_STOP_ACTIVATE_PCT:
            fallback_pct = config.TRAILING_STOP_PCT
            atr_mult = getattr(config, "ATR_TRAILING_STOP_MULT", 1.5)
        else:
            return  # not profitable enough for a trailing stop

        # Check if trailing stop already exists
        open_orders = self.broker.get_open_orders(symbol)

        existing_stop = None
        has_native_trailing = False
        for o in open_orders:
            if o.side == "sell" and o.type == "trailing_stop":
                has_native_trailing = True
                break
            if o.side == "sell" and o.type in ("stop", "stop_limit"):
                existing_stop = o

        if has_native_trailing:
            return

        # Compute ATR-based Chandelier stop; fall back to static %
        atr_val = 0.0
        try:
            df = self.broker.get_bars(symbol)
            if df is not None and len(df) >= 15:
                df = compute_all(df)
                atr_val = float(df.iloc[-1].get("atr", 0) or 0)
        except Exception as exc:
            log.debug("ATR lookup failed for %s trailing stop: %s", symbol, exc)

        if atr_val > 0:
            ideal_stop = round(current_price - atr_mult * atr_val, 2)
            trail_label = f"ATR×{atr_mult}"
        else:
            ideal_stop = round(current_price * (1 - fallback_pct), 2)
            trail_label = f"{fallback_pct*100:.1f}%"

        if existing_stop is not None:
            current_stop = float(existing_stop.stop_price)
            if ideal_stop > current_stop:
                log.info(
                    "Ratcheting trailing stop for %s: %.2f → %.2f "
                    "(price=%.2f, trail=%s, ATR=%.2f)",
                    symbol, current_stop, ideal_stop, current_price,
                    trail_label, atr_val,
                )
                try:
                    self.broker.api.cancel_order(existing_stop.id)
                    self.broker.submit_trailing_stop(
                        symbol, qty, fallback_pct,
                        stop_price=ideal_stop,
                        trail_amount=round(atr_mult * atr_val, 2) if atr_val > 0 else None,
                    )
                except Exception as exc:
                    log.warning("Trailing stop ratchet failed for %s: %s", symbol, exc)
        else:
            log.info(
                "Adding trailing stop for %s (%.1f%% profit, trail=%s, ATR=%.2f)",
                symbol, unrealised_pct * 100, trail_label, atr_val,
            )
            try:
                self.broker.submit_trailing_stop(
                    symbol, qty, fallback_pct,
                    stop_price=ideal_stop,
                    trail_amount=round(atr_mult * atr_val, 2) if atr_val > 0 else None,
                )
            except Exception as exc:
                log.warning("Trailing stop failed for %s: %s", symbol, exc)

    # ─────────────────────────────────────────────────────────
    # ENTRY SCAN – look for new swing-trade setups
    # ─────────────────────────────────────────────────────────
    def scan_entries(self) -> int:
        """
        Screen the watchlist, evaluate entry signals, size positions,
        and submit buy (bull) or short-sell (bear) orders.
        Returns the number of new positions opened.
        """
        self.refresh()

        if not self.risk.can_open_new_position():
            log.info("Max positions reached - skipping entry scan")
            return 0

        # ── Open-market delay ────────────────────────────────
        if config.MARKET_OPEN_DELAY_MINUTES > 0:
            try:
                clock = self.broker.get_clock()
                if clock.is_open:
                    from zoneinfo import ZoneInfo
                    eastern = ZoneInfo("America/New_York")
                    now_et = clock.timestamp.astimezone(eastern)
                    market_open_today = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
                    minutes_since_open = (now_et - market_open_today).total_seconds() / 60
                    if minutes_since_open < config.MARKET_OPEN_DELAY_MINUTES:
                        remaining = config.MARKET_OPEN_DELAY_MINUTES - minutes_since_open
                        log.info(
                            "Market open delay active – %.0f min since open, "
                            "waiting until %d min mark (%.0f min remaining). "
                            "Skipping entries.",
                            minutes_since_open,
                            config.MARKET_OPEN_DELAY_MINUTES,
                            remaining,
                        )
                        return 0
            except Exception as exc:
                log.warning("Could not check market open delay: %s – proceeding with entries", exc)

        # ── Market regime detection ──────────────────────────
        bear_market, spy_df = self._detect_regime()

        # In bear mode, decide action based on equity and config
        use_inverse_etfs = False
        if bear_market:
            equity = self.broker.get_equity()
            short_min_equity = getattr(config, "SHORT_MIN_EQUITY", 2000.0)

            if equity >= short_min_equity and config.BEAR_SHORT_MODE_ENABLED:
                log.info(
                    "BEAR MARKET — equity $%.2f >= $%.2f — "
                    "short-selling mode active. Will scan watchlist for short entries.",
                    equity, short_min_equity,
                )
            elif config.INVERSE_WATCHLIST:
                use_inverse_etfs = True
                log.info(
                    "BEAR MARKET — equity $%.2f < $%.2f (short-sell minimum) — "
                    "switching to INVERSE ETF mode. Will buy inverse ETFs instead of shorting.",
                    equity, short_min_equity,
                )
            else:
                log.info(
                    "BEAR MARKET — equity $%.2f < $%.2f and no INVERSE_WATCHLIST configured "
                    "— skipping all entries (cannot short or buy inverse ETFs).",
                    equity, short_min_equity,
                )
                return 0

        # Load VIXY for ML macro context (if ML enabled)
        vixy_df = None
        if config.ML_ENABLED:
            try:
                from indicators import compute_all as _compute
                if spy_df is None:
                    spy_df = self.broker.get_bars(config.MARKET_REGIME_SYMBOL, limit=250)
                    if spy_df is not None:
                        spy_df = _compute(spy_df)
                vixy_df = self.broker.get_bars(config.VIX_SYMBOL, limit=50)
                if vixy_df is not None:
                    vixy_df = _compute(vixy_df)
            except Exception as exc:
                log.warning("ML macro data load failed: %s", exc)

        # ── VIX fear filter ──────────────────────────────────
        vix_size_scale = 1.0
        if config.VIX_FILTER_ENABLED:
            try:
                _vixy = vixy_df
                if _vixy is None:
                    from indicators import compute_all as _compute
                    _vixy = self.broker.get_bars(config.VIX_SYMBOL, limit=50)
                    if _vixy is not None:
                        _vixy = _compute(_vixy)
                if _vixy is not None and len(_vixy) > 0:
                    vixy_price = float(_vixy.iloc[-1]["close"])
                    if not bear_market and vixy_price >= config.VIX_HALT_THRESHOLD:
                        log.info(
                            "VIX HALT — %s at %.2f (>= %.2f). Blocking ALL new long entries.",
                            config.VIX_SYMBOL, vixy_price, config.VIX_HALT_THRESHOLD,
                        )
                        return 0
                    if vixy_price >= config.VIX_REDUCE_THRESHOLD:
                        vix_size_scale = config.VIX_SIZE_SCALE
                        log.info(
                            "VIX elevated — %s at %.2f (>= %.2f). Reducing size to %.0f%%.",
                            config.VIX_SYMBOL, vixy_price, config.VIX_REDUCE_THRESHOLD,
                            vix_size_scale * 100,
                        )
            except Exception as exc:
                log.warning("VIX filter check failed: %s", exc)

        # Advanced features: vol regime + dynamic threshold
        vol_scale = self._get_vol_regime_scale()
        dyn_threshold = self._get_dynamic_threshold(spy_df)

        if dyn_threshold != config.ENTRY_SCORE_THRESHOLD:
            log.info("Dynamic threshold: %d (base %d)", dyn_threshold, config.ENTRY_SCORE_THRESHOLD)

        # Symbols we already hold (long or short) - skip them
        held = {p.symbol for p in self.broker.get_positions()}
        # Symbols with pending orders - skip them too
        pending = {o.symbol for o in self.broker.get_open_orders()
                   if o.side in ("buy", "sell")}

        # Portfolio risk gate
        positions = self.broker.get_positions()
        if positions:
            portfolio_risk = self.risk.portfolio_at_risk(positions)
            if portfolio_risk >= config.MAX_PORTFOLIO_RISK_PCT:
                log.info(
                    "Portfolio risk %.2f%% >= max %.2f%% — skipping new entries",
                    portfolio_risk * 100, config.MAX_PORTFOLIO_RISK_PCT * 100,
                )
                return 0

        candidates = self.screener.screen(
            symbols=config.INVERSE_WATCHLIST if use_inverse_etfs else None
        )
        opened = 0

        for c in candidates:
            symbol = c["symbol"]
            if symbol in held or symbol in pending:
                continue

            if not self.risk.can_open_new_position():
                break

            if not self.pdt.can_buy_today(symbol):
                continue

            # In bull mode: skip if symbol is in SHORT_BLACKLIST (irrelevant)
            # In bear mode: skip if symbol is in SHORT_BLACKLIST
            if bear_market and not use_inverse_etfs and symbol in config.SHORT_BLACKLIST:
                log.info("SKIP %s – on SHORT_BLACKLIST (not shortable)", symbol)
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

            # ── Route to bull (long) or bear (short / inverse ETF) entry logic ──
            if bear_market and use_inverse_etfs:
                opened += self._try_inverse_etf_entry(
                    c, weekly_bull, spy_df, vixy_df,
                    dyn_threshold, vol_scale, vix_size_scale, sector,
                )
            elif bear_market:
                opened += self._try_short_entry(
                    c, weekly_bull, spy_df, vixy_df,
                    dyn_threshold, vol_scale, vix_size_scale, sector,
                )
            else:
                opened += self._try_long_entry(
                    c, weekly_bull, spy_df, vixy_df,
                    dyn_threshold, vol_scale, vix_size_scale, sector,
                )

        if use_inverse_etfs:
            mode_str = "INVERSE_ETF"
        elif bear_market:
            mode_str = "SHORT"
        else:
            mode_str = "LONG"
        log.info("Entry scan complete (%s mode) - opened %d position(s)", mode_str, opened)
        return opened

    # ─────────────────────────────────────────────────────────
    # LONG ENTRY (bull mode)
    # ─────────────────────────────────────────────────────────
    def _try_long_entry(self, c: dict, weekly_bull: bool,
                        spy_df, vixy_df,
                        dyn_threshold: int, vol_scale: float,
                        vix_size_scale: float, sector: str) -> int:
        """Evaluate and submit a LONG entry. Returns 1 if opened, 0 otherwise."""
        symbol = c["symbol"]

        signal = check_entry(c["df"], weekly_bullish=weekly_bull,
                            spy_df=spy_df, vixy_df=vixy_df)
        if signal is None:
            return 0

        if signal["score"] < dyn_threshold:
            return 0

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

        # Apply VIX fear-filter sizing
        if vix_size_scale != 1.0 and qty > 0:
            qty = round(qty * vix_size_scale, 3) if config.FRACTIONAL_SHARES else int(qty * vix_size_scale)
            log.info("VIX size scale: %.2f -> qty adjusted to %.3f", vix_size_scale, qty)

        if qty == 0:
            log.info("Position size = 0 for %s - skipping", symbol)
            return 0

        MIN_ORDER_NOTIONAL = 1.0
        if qty * entry_price < MIN_ORDER_NOTIONAL:
            log.info("Skipping %s – order value $%.2f below minimum $%.2f",
                     symbol, qty * entry_price, MIN_ORDER_NOTIONAL)
            return 0

        log.info(
            "ENTRY LONG %s  qty=%.3f  price=%.2f  SL=%.2f  TP=%.2f  [%s]",
            symbol, qty, entry_price, stop_loss, take_profit, signal["reason"],
        )

        try:
            self.broker.submit_market_buy(
                symbol, qty, stop_loss=stop_loss, take_profit=take_profit
            )
            self.pdt.record_buy(symbol)
            self.risk.open_positions += 1
            self._sector_counts[sector] = self._sector_counts.get(sector, 0) + 1
            return 1
        except Exception as exc:
            log.error("Buy order failed for %s: %s", symbol, exc)
            return 0

    # ─────────────────────────────────────────────────────────
    # INVERSE ETF ENTRY (bear mode, equity < SHORT_MIN_EQUITY)
    # ─────────────────────────────────────────────────────────
    def _try_inverse_etf_entry(self, c: dict, weekly_bull: bool,
                                spy_df, vixy_df,
                                dyn_threshold: int, vol_scale: float,
                                vix_size_scale: float, sector: str) -> int:
        """
        Evaluate and submit a LONG buy on an inverse ETF as a bear-market
        alternative to short-selling (used when equity < SHORT_MIN_EQUITY).

        Inverse ETFs naturally rise when the market falls, so we use the
        standard long-entry scoring logic — the ETF's own technicals
        (RSI, MACD, EMAs) will reflect the bullish trend *of the ETF*
        which corresponds to a bearish move in the underlying index.

        Returns 1 if opened, 0 otherwise.
        """
        symbol = c["symbol"]

        # Use the dedicated inverse ETF model for scoring
        signal = check_inverse_entry(c["df"], weekly_bullish=weekly_bull,
                                     spy_df=spy_df, vixy_df=vixy_df)
        if signal is None:
            return 0

        if signal["score"] < dyn_threshold:
            return 0

        entry_price = signal["price"]
        atr = signal["atr"]
        stop_loss = self.risk.compute_stop_loss(entry_price, atr)
        take_profit = self.risk.compute_take_profit(entry_price, atr)

        qty = self.risk.calculate_position_size(
            entry_price=entry_price,
            stop_price=stop_loss,
            buying_power=self.broker.get_buying_power(),
        )

        # Apply inverse-ETF size scaling (same risk profile as shorts)
        inverse_scale = getattr(config, "INVERSE_ETF_SIZE_SCALE", 0.60)
        if inverse_scale != 1.0 and qty > 0:
            qty = round(qty * inverse_scale, 3) if config.FRACTIONAL_SHARES else int(qty * inverse_scale)
            log.info("Inverse ETF size scale: %.2f -> qty adjusted to %.3f",
                     inverse_scale, qty)

        # Apply volatility regime scaling
        if vol_scale != 1.0 and qty > 0:
            qty = round(qty * vol_scale, 3) if config.FRACTIONAL_SHARES else int(qty * vol_scale)
            log.info("Vol regime scale: %.2f -> qty adjusted to %.3f", vol_scale, qty)

        # Apply VIX sizing (elevated VIX helps inverse ETFs, but still respect scaling)
        if vix_size_scale != 1.0 and qty > 0:
            qty = round(qty * vix_size_scale, 3) if config.FRACTIONAL_SHARES else int(qty * vix_size_scale)
            log.info("VIX size scale: %.2f -> qty adjusted to %.3f", vix_size_scale, qty)

        if qty == 0:
            log.info("Position size = 0 for inverse ETF %s - skipping", symbol)
            return 0

        MIN_ORDER_NOTIONAL = 1.0
        if qty * entry_price < MIN_ORDER_NOTIONAL:
            log.info("Skipping inverse ETF %s – order value $%.2f below minimum $%.2f",
                     symbol, qty * entry_price, MIN_ORDER_NOTIONAL)
            return 0

        log.info(
            "ENTRY INVERSE-ETF (LONG) %s  qty=%.3f  price=%.2f  SL=%.2f  TP=%.2f  [%s]",
            symbol, qty, entry_price, stop_loss, take_profit, signal["reason"],
        )

        try:
            self.broker.submit_market_buy(
                symbol, qty, stop_loss=stop_loss, take_profit=take_profit
            )
            self.pdt.record_buy(symbol)
            self.risk.open_positions += 1
            self._sector_counts[sector] = self._sector_counts.get(sector, 0) + 1
            return 1
        except Exception as exc:
            log.error("Inverse ETF buy order failed for %s: %s", symbol, exc)
            return 0

    # ─────────────────────────────────────────────────────────
    # SHORT ENTRY (bear mode)
    # ─────────────────────────────────────────────────────────
    def _try_short_entry(self, c: dict, weekly_bull: bool,
                         spy_df, vixy_df,
                         dyn_threshold: int, vol_scale: float,
                         vix_size_scale: float, sector: str) -> int:
        """Evaluate and submit a SHORT entry. Returns 1 if opened, 0 otherwise."""
        symbol = c["symbol"]

        signal = check_short_entry(c["df"], weekly_bullish=weekly_bull,
                                   spy_df=spy_df, vixy_df=vixy_df)
        if signal is None:
            return 0

        if signal["score"] < dyn_threshold:
            return 0

        entry_price = signal["price"]
        atr = signal["atr"]
        # For shorts: stop is ABOVE entry, target is BELOW entry
        stop_loss = self.risk.compute_short_stop_loss(entry_price, atr)
        take_profit = self.risk.compute_short_take_profit(entry_price, atr)

        qty = self.risk.calculate_short_position_size(
            entry_price=entry_price,
            stop_price=stop_loss,
            buying_power=self.broker.get_buying_power(),
        )

        # Apply bear-mode size scaling (shorts are riskier)
        if config.BEAR_SHORT_SIZE_SCALE != 1.0 and qty > 0:
            qty = round(qty * config.BEAR_SHORT_SIZE_SCALE, 3) if config.FRACTIONAL_SHARES else int(qty * config.BEAR_SHORT_SIZE_SCALE)
            log.info("Bear short size scale: %.2f -> qty adjusted to %.3f",
                     config.BEAR_SHORT_SIZE_SCALE, qty)

        # Apply volatility regime scaling
        if vol_scale != 1.0 and qty > 0:
            qty = round(qty * vol_scale, 3) if config.FRACTIONAL_SHARES else int(qty * vol_scale)
            log.info("Vol regime scale: %.2f -> qty adjusted to %.3f", vol_scale, qty)

        # Apply VIX sizing (elevated VIX actually helps shorts, but still reduce size)
        if vix_size_scale != 1.0 and qty > 0:
            qty = round(qty * vix_size_scale, 3) if config.FRACTIONAL_SHARES else int(qty * vix_size_scale)
            log.info("VIX size scale: %.2f -> qty adjusted to %.3f", vix_size_scale, qty)

        if qty == 0:
            log.info("Short position size = 0 for %s - skipping", symbol)
            return 0

        MIN_ORDER_NOTIONAL = 1.0
        if qty * entry_price < MIN_ORDER_NOTIONAL:
            log.info("Skipping short %s – order value $%.2f below minimum $%.2f",
                     symbol, qty * entry_price, MIN_ORDER_NOTIONAL)
            return 0

        log.info(
            "ENTRY SHORT %s  qty=%.3f  price=%.2f  SL=%.2f  TP=%.2f  [%s]",
            symbol, qty, entry_price, stop_loss, take_profit, signal["reason"],
        )

        try:
            self.broker.submit_short_sell(
                symbol, qty, stop_loss=stop_loss, take_profit=take_profit
            )
            self.pdt.record_buy(symbol)
            self.risk.open_positions += 1
            self._sector_counts[sector] = self._sector_counts.get(sector, 0) + 1
            return 1
        except Exception as exc:
            log.error("Short sell order failed for %s: %s", symbol, exc)
            return 0

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
        try:
            if not self.broker.is_market_open():
                log.info("Market is closed – skipping cycle")
                return
        except Exception as exc:
            log.warning("Network error while checking market clock: %s – skipping cycle", exc)
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
