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
from time_utils import parse_iso_datetime
from trade_journal import TradeJournal
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
        self.journal = TradeJournal()

    def _get_recent_fill(self, symbol: str, side: str) -> tuple[dt.datetime | None, float | None]:
        """Return the most recent filled order time/price for *symbol* and *side*."""
        try:
            recent = self.broker.api.list_orders(
                status="closed",
                symbols=[symbol],
                limit=20,
                direction="desc",
            )
            for order in recent:
                if order.side != side or not getattr(order, "filled_at", None):
                    continue
                filled_at = getattr(order, "filled_at")
                if isinstance(filled_at, str):
                    filled_at = parse_iso_datetime(filled_at)
                fill_price = float(getattr(order, "filled_avg_price", 0) or 0)
                return filled_at, (fill_price if fill_price > 0 else None)
        except Exception as exc:
            log.debug("Could not fetch recent fill for %s (%s): %s", symbol, side, exc)
        return None, None

    def _reconcile_open_positions(self, positions: list | None = None) -> None:
        """Backfill PDT/journal state for positions that filled outside the direct entry path."""
        try:
            positions = positions if positions is not None else self.broker.get_positions()
            tracked_symbols = set(self.pdt.open_symbols())
            journal_symbols = {row["symbol"] for row in self.journal.get_open_trades()}
        except Exception as exc:
            log.warning("Open-position reconcile failed to gather state: %s", exc)
            return

        for pos in positions:
            symbol = pos.symbol
            qty = float(pos.qty)
            abs_qty = abs(qty)
            is_short = qty < 0
            side = "short" if is_short else ("inverse" if symbol in config.INVERSE_WATCHLIST else "long")
            order_side = "sell" if is_short else "buy"
            fill_dt, fill_price = self._get_recent_fill(symbol, order_side)
            entry_price = fill_price or float(pos.avg_entry_price)
            regime = "bear" if side in ("short", "inverse") else "bull"

            if symbol not in tracked_symbols:
                self.pdt.record_buy(symbol, fill_date=fill_dt.date() if fill_dt else None)
                if fill_dt is not None:
                    self.pdt._buy_times[symbol] = fill_dt.isoformat()
                    self.pdt._save()
                log.warning("Reconcile: backfilled PDT entry for %s %s", side.upper(), symbol)

            if symbol not in journal_symbols:
                self.journal.record_entry(
                    symbol,
                    side,
                    entry_price,
                    abs_qty,
                    0.0,
                    0.0,
                    {"price": entry_price, "score": 0, "reason": "reconciled_live_fill"},
                    regime=regime,
                    vwap_used=False,
                    order_type="reconciled",
                )
                log.warning("Reconcile: backfilled journal entry for %s %s", side.upper(), symbol)

    def _finalize_entry(
        self,
        *,
        symbol: str,
        sector: str,
        side: str,
        requested_entry_price: float,
        qty: float,
        stop_loss: float,
        take_profit: float,
        signal: dict,
        regime: str,
        order_type: str,
        vwap_used: bool,
        order=None,
    ) -> int:
        """Record PDT/journal state only after a new position is actually visible."""
        fill_side = "sell" if side == "short" else "buy"
        filled_price = self._poll_fill_price(order) if order is not None else None

        if filled_price is None:
            try:
                position = self.broker.get_position(symbol)
            except Exception:
                position = None
            if position is not None:
                filled_price = float(position.avg_entry_price)

        if filled_price is None:
            log.info(
                "ENTRY PENDING %s %s — order accepted but not filled yet. "
                "Deferring PDT/journal accounting until reconciliation.",
                side.upper(), symbol,
            )
            return 0

        fill_dt, recent_fill_price = self._get_recent_fill(symbol, fill_side)
        actual_entry_price = recent_fill_price or filled_price or requested_entry_price

        self.pdt.record_buy(symbol, fill_date=fill_dt.date() if fill_dt else None)
        if fill_dt is not None:
            self.pdt._buy_times[symbol] = fill_dt.isoformat()
            self.pdt._save()
        self.risk.open_positions += 1
        self._sector_counts[sector] = self._sector_counts.get(sector, 0) + 1
        self.journal.record_entry(
            symbol,
            side,
            actual_entry_price,
            qty,
            stop_loss,
            take_profit,
            signal,
            regime=regime,
            vwap_used=vwap_used,
            order_type=order_type,
        )
        return 1

    def _finalize_exit(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        requested_exit_price: float,
        hold_days: int,
        exit_reason: str,
        order=None,
    ) -> int:
        """Record PDT/journal state only after the position is actually closed."""
        fill_side = "buy" if side == "short" else "sell"
        filled_price = self._poll_fill_price(order) if order is not None else None

        try:
            position = self.broker.get_position(symbol)
        except Exception:
            position = None

        if position is not None and abs(float(position.qty)) > 0:
            log.info(
                "EXIT PENDING %s %s — order accepted but position remains open. "
                "Deferring PDT/journal accounting until reconciliation.",
                side.upper(), symbol,
            )
            return 0

        fill_dt, recent_fill_price = self._get_recent_fill(symbol, fill_side)
        actual_exit_price = recent_fill_price or filled_price or requested_exit_price

        self.pdt.record_sell(symbol)
        if side == "short":
            trade_pnl = (entry_price - actual_exit_price) * qty
        else:
            trade_pnl = (actual_exit_price - entry_price) * qty

        self.journal.record_exit(
            symbol,
            exit_price=requested_exit_price,
            pnl=trade_pnl,
            hold_days=hold_days,
            exit_reason=exit_reason,
            filled_price=actual_exit_price,
        )
        return 1

    def _poll_fill_price(self, order, timeout_secs: float = 5.0) -> float | None:
        """Poll an order for its filled_avg_price (market orders fill fast)."""
        if order is None:
            return None
        import time
        deadline = time.monotonic() + timeout_secs
        order_id = getattr(order, "id", None)
        if not order_id:
            return None
        while time.monotonic() < deadline:
            try:
                refreshed = self.broker.api.get_order(order_id)
                status = getattr(refreshed, "status", "")
                fp = float(getattr(refreshed, "filled_avg_price", 0) or 0)
                if fp > 0:
                    return fp
                if status in ("filled", "canceled", "expired", "rejected"):
                    return fp if fp > 0 else None
            except Exception:
                pass
            time.sleep(0.5)
        return None

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
        self._reconcile_open_positions()

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
                        adj = getattr(config, "DYNAMIC_THRESHOLD_ADJUSTMENT", 1)
                        return base - adj  # strong market: lower bar
        except Exception as exc:
            log.warning("Dynamic threshold check failed: %s", exc)
        return base

    # ─────────────────────────────────────────────────────────
    # MARKET REGIME DETECTION
    # ─────────────────────────────────────────────────────────
    def _detect_regime(self) -> tuple:
        """
        Detect bull vs bear (vs chop) market regime.

        Returns (bear_market: bool, spy_df: DataFrame | None).

        Detection layers (in priority order):
          1. HMM regime model (fast probabilistic detection) — if trained and enabled
          2. EMA-200 fallback (classic lagging indicator)
          3. EMA-50 degraded fallback
          4. Default to BULL if insufficient data

        The HMM also detects a "CHOP" regime which is stored on the instance
        as self._hmm_chop so the entry scanner can reduce position sizing.
        """
        bear_market = False
        spy_df = None
        self._hmm_regime = None    # raw HMM result dict
        self._hmm_chop = False     # True when HMM detects chop regime
        self._hmm_bear_raw = False # True when HMM reports BEAR *before* overrides

        if not config.MARKET_REGIME_ENABLED:
            log.debug("Market regime filter disabled — defaulting to bull mode")
            return bear_market, spy_df

        # Fetch SPY bars (needed by both HMM and EMA fallback)
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

        # ── Layer 1: HMM regime detection ────────────────────
        hmm_decided = False
        if getattr(config, "HMM_REGIME_ENABLED", False):
            try:
                from hmm_model import predict_regime, is_available as hmm_available

                if hmm_available():
                    hmm_result = predict_regime(
                        spy_df,
                        lookback=getattr(config, "HMM_LOOKBACK", 30),
                    )
                    if hmm_result is not None:
                        self._hmm_regime = hmm_result
                        probs = hmm_result["probabilities"]
                        state = hmm_result["state"]

                        bear_prob = probs.get("BEAR", 0.0)
                        chop_prob = probs.get("CHOP", 0.0)
                        bull_prob = probs.get("BULL", 0.0)

                        bear_thresh = getattr(config, "HMM_BEAR_THRESHOLD", 0.60)
                        chop_thresh = getattr(config, "HMM_CHOP_THRESHOLD", 0.50)

                        if bear_prob >= bear_thresh:
                            # ── Price confirmation: require SPY below EMA-200 ──
                            # HMM can lag during fast recoveries. If SPY is already
                            # above its 200-day EMA, don't declare a full BEAR
                            # (which would trigger inverse ETF buying). Downgrade
                            # to CHOP instead — reduced sizing, no inverse ETFs.
                            hmm_called_bear = True
                            self._hmm_bear_raw = True
                            try:
                                _spy_indicators = compute_all(spy_df)
                                _spy_row = _spy_indicators.iloc[-1]
                                _spy_close = float(_spy_row["close"])
                                _spy_ema200 = float(_spy_row.get("ema_200", 0) or 0)

                                # ── Momentum override ──────────────────────────
                                # If SPY has recovered strongly in recent days,
                                # the HMM is lagging — downgrade BEAR to CHOP to
                                # prevent inverse ETF entries during a recovery.
                                _mom_days = getattr(config, "HMM_MOMENTUM_OVERRIDE_DAYS", 3)
                                _mom_pct  = getattr(config, "HMM_MOMENTUM_OVERRIDE_PCT", 0.015)
                                _override_enabled = getattr(config, "HMM_MOMENTUM_OVERRIDE_ENABLED", True)
                                _recent_return = 0.0
                                if len(_spy_indicators) > _mom_days:
                                    _past_close = float(_spy_indicators["close"].iloc[-(_mom_days + 1)])
                                    if _past_close > 0:
                                        _recent_return = (_spy_close - _past_close) / _past_close
                                momentum_override = _override_enabled and _recent_return > _mom_pct

                                # Both the momentum override and the EMA-200 override are
                                # controlled by the single HMM_MOMENTUM_OVERRIDE_ENABLED flag
                                # so the dashboard "Override Enabled" toggle disables both.

                                if momentum_override:
                                    self._hmm_chop = True
                                    bear_market = False
                                    hmm_decided = True
                                    spy_df = _spy_indicators
                                    log.warning(
                                        "REGIME [HMM→CHOP override]: HMM P(BEAR)=%.2f but "
                                        "SPY %d-day return=+%.1f%% (threshold +%.1f%%) — "
                                        "market is recovering. No inverse ETFs, reduced sizing.",
                                        bear_prob, _mom_days, _recent_return * 100, _mom_pct * 100,
                                    )
                                elif _override_enabled and _spy_ema200 > 0 and _spy_close > _spy_ema200:
                                    # HMM says BEAR but price is above EMA-200 — contradiction.
                                    # Only fires when HMM_EMA200_OVERRIDE_ENABLED = True.
                                    self._hmm_chop = True
                                    bear_market = False
                                    hmm_decided = True
                                    spy_df = _spy_indicators
                                    log.warning(
                                        "REGIME [HMM→CHOP override]: HMM P(BEAR)=%.2f but "
                                        "SPY $%.2f > EMA-200 $%.2f — price action disagrees. "
                                        "Downgrading to CHOP (no inverse ETFs, reduced sizing).",
                                        bear_prob, _spy_close, _spy_ema200,
                                    )
                                else:
                                    bear_market = True
                                    hmm_decided = True
                                    spy_df = _spy_indicators
                                    log.info(
                                        "REGIME [HMM]: BEAR — P(BEAR)=%.2f  P(CHOP)=%.2f  P(BULL)=%.2f  "
                                        "(threshold=%.2f, SPY $%.2f < EMA-200 $%.2f confirmed, "
                                        "%d-day return=%.1f%%)",
                                        bear_prob, chop_prob, bull_prob, bear_thresh,
                                        _spy_close, _spy_ema200,
                                        _mom_days, _recent_return * 100,
                                    )
                            except Exception as _exc:
                                log.debug("EMA-200 confirmation check failed: %s — accepting HMM BEAR", _exc)
                                bear_market = True
                                hmm_decided = True
                                log.info(
                                    "REGIME [HMM]: BEAR — P(BEAR)=%.2f  P(CHOP)=%.2f  P(BULL)=%.2f  "
                                    "(threshold=%.2f)",
                                    bear_prob, chop_prob, bull_prob, bear_thresh,
                                )
                        elif chop_prob >= chop_thresh:
                            self._hmm_chop = True
                            hmm_decided = True
                            log.info(
                                "REGIME [HMM]: CHOP — P(CHOP)=%.2f  P(BEAR)=%.2f  P(BULL)=%.2f  "
                                "(threshold=%.2f). Reducing position sizes.",
                                chop_prob, bear_prob, bull_prob, chop_thresh,
                            )
                        else:
                            hmm_decided = True
                            log.info(
                                "REGIME [HMM]: BULL — P(BULL)=%.2f  P(BEAR)=%.2f  P(CHOP)=%.2f",
                                bull_prob, bear_prob, chop_prob,
                            )

                        if hmm_decided:
                            # Still compute indicators on spy_df for downstream use
                            try:
                                spy_df = compute_all(spy_df)
                            except Exception:
                                pass
                            return bear_market, spy_df
                else:
                    log.debug("HMM model not available — falling back to EMA regime detection")
            except ImportError:
                log.debug("hmmlearn not installed — falling back to EMA regime detection")
            except Exception as exc:
                log.warning("HMM regime detection failed: %s — falling back to EMA", exc)

        # ── Layer 2: EMA-based regime detection (fallback) ───
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
                        "REGIME [EMA]: BEAR — %s close $%.2f < EMA-200 $%.2f  (bars=%d)",
                        config.MARKET_REGIME_SYMBOL, spy_close, spy_ema200, len(spy_df),
                    )
                else:
                    log.info(
                        "REGIME [EMA]: BULL — %s close $%.2f >= EMA-200 $%.2f  (bars=%d)",
                        config.MARKET_REGIME_SYMBOL, spy_close, spy_ema200, len(spy_df),
                    )
                return bear_market, spy_df

        # Fallback: use EMA-50 if EMA-200 is unavailable (not enough bars)
        spy_ema50 = spy_row.get("ema_trend", None)
        if spy_ema50 is not None and not pd.isna(spy_ema50):
            if spy_close < spy_ema50:
                bear_market = True
                log.warning(
                    "REGIME [EMA]: BEAR (degraded) — %s close $%.2f < EMA-50 $%.2f  "
                    "(only %d bars; EMA-200 unavailable — using EMA-50 fallback)",
                    config.MARKET_REGIME_SYMBOL, spy_close, spy_ema50, len(spy_df),
                )
            else:
                log.info(
                    "REGIME [EMA]: BULL (degraded) — %s close $%.2f >= EMA-50 $%.2f  "
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

    def _get_hold_days(self, symbol: str) -> int:
        """Return calendar days held, using PDT ledger with Alpaca order fallback."""
        days = self.pdt.days_held(symbol)
        if days is not None:
            return days
        # PDT ledger missing entry — fallback to Alpaca order history
        try:
            side = "buy"
            recent = self.broker.api.list_orders(
                status="closed", symbols=[symbol], limit=20, direction="desc"
            )
            for order in recent:
                if order.side == side and getattr(order, "filled_at", None):
                    filled_at = getattr(order, "filled_at")
                    if isinstance(filled_at, str):
                        filled_at = parse_iso_datetime(filled_at)
                    fill_date = filled_at.date()
                    # Backfill PDT ledger so future lookups succeed
                    self.pdt.record_buy(symbol, fill_date=fill_date)
                    log.info("Backfilled PDT ledger for %s from Alpaca order (filled %s)", symbol, fill_date)
                    return (dt.date.today() - fill_date).days
        except Exception as exc:
            log.warning("Could not fetch order history for %s hold days: %s", symbol, exc)
        return 0

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
        self._reconcile_open_positions(positions)
        active_symbols = {p.symbol for p in positions}
        self.pdt.cleanup_stale(active_symbols)

        # Reconcile any stop/TP fills that happened between cycles so the
        # journal is up-to-date before we evaluate current positions.
        self._reconcile_closed_trades()

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
            hold_days = self._get_hold_days(symbol)

            # Use the appropriate exit checker for long vs short
            if is_short:
                signal = check_short_exit(df, entry_price, hold_days=hold_days, explain=True)
            else:
                signal = check_exit(df, entry_price, hold_days=hold_days, explain=True, symbol=symbol)

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
                        # Record the stop-loss/TP exit in the journal so it
                        # shows up without requiring a manual backfill.
                        try:
                            fill_price = current_price
                            intended_stop = current_price  # fallback
                            side_needed = "buy" if is_short else "sell"
                            recent_orders = self.broker.api.list_orders(
                                status="closed", symbols=[symbol], limit=10, direction="desc"
                            )
                            for _ro in recent_orders:
                                if _ro.side == side_needed and getattr(_ro, "filled_at", None):
                                    _fp = float(getattr(_ro, "filled_avg_price", 0) or 0)
                                    if _fp > 0:
                                        fill_price = _fp
                                        # Get intended stop price for slippage calc
                                        _sp = float(getattr(_ro, "stop_price", 0) or 0)
                                        if _sp > 0:
                                            intended_stop = _sp
                                        break
                            _sl_pnl = (entry_price - fill_price) * abs_qty if is_short else (fill_price - entry_price) * abs_qty
                            self.journal.record_exit(
                                symbol,
                                exit_price=intended_stop,
                                pnl=round(_sl_pnl, 4),
                                hold_days=hold_days,
                                exit_reason="stop_loss",
                                filled_price=fill_price,
                            )
                        except Exception as _exc:
                            log.warning("Could not record stop-loss exit for %s in journal: %s", symbol, _exc)
                        continue
                    abs_qty = abs(float(fresh_pos.qty))

                    if is_short:
                        _exit_order = self.broker.submit_market_cover(symbol, abs_qty)
                    else:
                        _exit_order = self.broker.submit_market_sell(symbol, abs_qty)
                    closed += self._finalize_exit(
                        symbol=symbol,
                        side="short" if is_short else "long",
                        qty=abs_qty,
                        entry_price=entry_price,
                        requested_exit_price=current_price,
                        hold_days=hold_days,
                        exit_reason=", ".join(signal.get("reasons", [])),
                        order=_exit_order,
                    )
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

        A profit-locking floor guarantees the stop never gives back more
        than a configured fraction of the unrealised gain, regardless of
        where the ATR-based Chandelier stop lands.
        """
        symbol = pos.symbol
        entry_price = float(pos.avg_entry_price)

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

        # De-scale ATR for leveraged ETFs (3× ETF has ~3× ATR of underlying)
        atr_val = config.descale_atr(atr_val, symbol)

        if atr_val > 0:
            ideal_stop = round(current_price - atr_mult * atr_val, 2)
            trail_label = f"ATR×{atr_mult}"
        else:
            ideal_stop = round(current_price * (1 - fallback_pct), 2)
            trail_label = f"{fallback_pct*100:.1f}%"

        # ── Profit-locking floor ─────────────────────────────
        # Never let the stop give back more than half the unrealised gain.
        # This prevents the ATR-based stop from sitting far below when a
        # stock has run up significantly (e.g. 13% gain but stop 9% below).
        unrealised_gain = current_price - entry_price
        if unrealised_gain > 0:
            # Lock in at least 50% of gain (break-even minimum)
            profit_floor = round(entry_price + unrealised_gain * 0.50, 2)
            if ideal_stop < profit_floor:
                log.debug(
                    "%s: ATR stop %.2f below profit floor %.2f "
                    "(entry=%.2f, gain=%.2f) — raising",
                    symbol, ideal_stop, profit_floor,
                    entry_price, unrealised_gain,
                )
                ideal_stop = profit_floor
                trail_label += "+floor"

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
                    market_open_today = now_et.replace(hour=9, minute=15, second=0, microsecond=0)
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

        # ── HMM BEAR override guard ────────────────────────
        # When the HMM reported BEAR but overrides downgraded to CHOP,
        # do NOT switch to long mode.  The overrides prevent inverse/short
        # entries (price action disagrees), but we must also block long
        # entries because the HMM still says the market is bearish.
        # Sit on cash until the regime resolves.
        if getattr(self, '_hmm_bear_raw', False) and not bear_market:
            log.warning(
                "HMM detected BEAR regime (overridden to CHOP) — "
                "blocking ALL new entries. Will not open longs while "
                "HMM signals BEAR. Waiting for regime to resolve."
            )
            return 0

        # ── Portfolio conflict guard ──────────────────────────
        # Never hold inverse ETFs and regular longs simultaneously.
        inverse_set = set(getattr(config, 'INVERSE_WATCHLIST', []))
        if inverse_set:
            current_positions = self.broker.get_positions()
            held_inverse = [p.symbol for p in current_positions
                           if p.symbol in inverse_set]
            held_regular_long = [p.symbol for p in current_positions
                                if p.symbol not in inverse_set
                                and float(p.qty) > 0]

            if use_inverse_etfs and held_regular_long:
                log.warning(
                    "MODE CONFLICT: Inverse ETF mode active but still holding "
                    "regular long positions %s. Blocking new inverse entries "
                    "until long positions are closed.",
                    held_regular_long,
                )
                return 0

            if not bear_market and held_inverse:
                log.warning(
                    "MODE CONFLICT: Bull/long mode active but still holding "
                    "inverse ETF positions %s. Blocking new long entries "
                    "until inverse positions are closed.",
                    held_inverse,
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

        # HMM chop regime: reduce position sizing when market is choppy
        if getattr(self, "_hmm_chop", False):
            chop_scale = getattr(config, "HMM_CHOP_SIZE_SCALE", 0.65)
            vol_scale *= chop_scale
            log.info("HMM CHOP regime active — sizing scale reduced by %.0f%% (combined vol_scale=%.2f)",
                     (1 - chop_scale) * 100, vol_scale)

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

        # Pre-fetch close-price series for all open positions so that the
        # correlation guard can reuse the same data across every candidate
        # without making redundant broker calls.
        _pos_closes: dict[str, pd.Series] = {}
        if config.CORRELATION_GUARD_ENABLED and positions:
            for _pos in positions:
                try:
                    _df = self.broker.get_bars(
                        _pos.symbol,
                        limit=config.CORRELATION_LOOKBACK + 5,
                    )
                    if _df is not None and len(_df) >= 10:
                        _pos_closes[_pos.symbol] = _df["close"]
                except Exception as _exc:
                    log.debug(
                        "Correlation guard: could not fetch %s: %s",
                        _pos.symbol, _exc,
                    )

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

            # Portfolio correlation guard – reject if the candidate moves
            # too closely with an already-held position (cross-sector included).
            if config.CORRELATION_GUARD_ENABLED and _pos_closes:
                try:
                    cand_closes = c["df"]["close"].tail(config.CORRELATION_LOOKBACK)
                    blocked, reason = self.risk.check_correlation(cand_closes, _pos_closes)
                    if blocked:
                        log.info(
                            "SKIP %s – correlation >= %.0f%% with %s",
                            symbol, config.MAX_CORRELATION * 100, reason,
                        )
                        continue
                except Exception as _exc:
                    log.debug(
                        "Correlation guard check failed for %s: %s – proceeding",
                        symbol, _exc,
                    )

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
        stop_loss = self.risk.compute_stop_loss(entry_price, atr, symbol)
        take_profit = self.risk.compute_take_profit(entry_price, atr, symbol)

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

        # ── Route: VWAP slicing for high-conviction ML setups ──────────
        ml_prob = signal.get("ml_prob")
        use_vwap = (
            config.VWAP_EXECUTION_ENABLED
            and ml_prob is not None
            and ml_prob >= config.VWAP_ML_THRESHOLD
        )

        try:
            order = None
            if use_vwap:
                log.info(
                    "VWAP routing active for %s (ml_prob=%.3f >= %.2f) – "
                    "splitting into %d slices over %ds",
                    symbol, ml_prob, config.VWAP_ML_THRESHOLD,
                    config.VWAP_SLICES, config.VWAP_INTERVAL_SECONDS * config.VWAP_SLICES,
                )
                log.warning(
                    "VWAP execution for %s is asynchronous — entry accounting will be "
                    "deferred until fills are reconciled from live positions.",
                    symbol,
                )
                self.broker.submit_vwap_buy(
                    symbol, qty,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    slices=config.VWAP_SLICES,
                    interval_seconds=config.VWAP_INTERVAL_SECONDS,
                )
            else:
                order = self.broker.submit_market_buy(
                    symbol, qty, stop_loss=stop_loss, take_profit=take_profit
                )
            return self._finalize_entry(
                symbol=symbol,
                sector=sector,
                side="long",
                requested_entry_price=entry_price,
                qty=qty,
                stop_loss=stop_loss,
                take_profit=take_profit,
                signal=signal,
                regime="bull",
                order_type="vwap" if use_vwap else "market",
                vwap_used=use_vwap,
                order=order,
            )
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
        stop_loss = self.risk.compute_stop_loss(entry_price, atr, symbol)
        take_profit = self.risk.compute_take_profit(entry_price, atr, symbol)

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
            order = self.broker.submit_market_buy(
                symbol, qty, stop_loss=stop_loss, take_profit=take_profit
            )
            return self._finalize_entry(
                symbol=symbol,
                sector=sector,
                side="inverse",
                requested_entry_price=entry_price,
                qty=qty,
                stop_loss=stop_loss,
                take_profit=take_profit,
                signal=signal,
                regime="bear",
                order_type="market",
                vwap_used=False,
                order=order,
            )
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
        stop_loss = self.risk.compute_short_stop_loss(entry_price, atr, symbol)
        take_profit = self.risk.compute_short_take_profit(entry_price, atr, symbol)

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
            order = self.broker.submit_short_sell(
                symbol, qty, stop_loss=stop_loss, take_profit=take_profit
            )
            return self._finalize_entry(
                symbol=symbol,
                sector=sector,
                side="short",
                requested_entry_price=entry_price,
                qty=qty,
                stop_loss=stop_loss,
                take_profit=take_profit,
                signal=signal,
                regime="bear",
                order_type="market",
                vwap_used=False,
                order=order,
            )
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
        4. Reconcile any stop-loss / take-profit fills that happened
           outside of scan_exits (between cycles, overnight, etc.).
        """
        log.info("--- Morning tasks start ---")

        # 1. Reconcile bracket/stop fills that occurred outside of scan_exits
        self._reconcile_closed_trades()

        # 2. Stop-loss resubmission
        n = self.broker.resubmit_stop_losses(self.pdt)
        if n:
            log.info("Morning tasks: resubmitted %d stop-loss order(s)", n)

        # 3. Ledger hygiene
        active = {p.symbol for p in self.broker.get_positions()}
        self.pdt.cleanup_stale(active)

        # 4. PDT budget report
        used = self.pdt._rolling_day_trade_count()
        remaining = max(0, config.MAX_DAY_TRADES_ALLOWED - used)
        log.info(
            "Morning tasks: day-trade budget — %d used / %d allowed / %d remaining (rolling %d-day window)",
            used, config.MAX_DAY_TRADES_ALLOWED, remaining, config.PDT_LOOKBACK_DAYS,
        )

        log.info("--- Morning tasks done ---")

    def _reconcile_closed_trades(self) -> None:
        """
        Journal any stop-loss, take-profit, or broker-initiated fills that
        closed a position outside of scan_exits (e.g. between cycles or
        overnight).  Runs once per morning and at the start of each
        scan_exits to keep the journal current.

        For each open journal row where the position no longer exists in
        Alpaca, look up the most recent closed sell/buy-cover order for that
        symbol, compute P&L, and call record_exit().
        """
        try:
            open_rows = self.journal.get_open_trades()
        except Exception as exc:
            log.warning("Reconcile: could not fetch open journal rows: %s", exc)
            return

        if not open_rows:
            return

        held_symbols = {p.symbol for p in self.broker.get_positions()}

        for row in open_rows:
            symbol = row["symbol"]
            # Still holding — nothing to reconcile
            if symbol in held_symbols:
                continue

            entry_price = float(row["entry_price"] or 0)
            qty = float(row["qty"] or 0)
            side = row["side"]  # 'long', 'short', 'inverse'
            is_short = side == "short"

            # Find the most recent closed fill on the exit side
            exit_side = "buy" if is_short else "sell"
            fill_price: float | None = None
            exit_reason = "stop_or_tp"
            try:
                recent = self.broker.api.list_orders(
                    status="closed",
                    symbols=[symbol],
                    limit=10,
                    direction="desc",
                )
                for o in recent:
                    if o.side == exit_side and getattr(o, "filled_at", None):
                        fp = float(getattr(o, "filled_avg_price", 0) or 0)
                        if fp > 0:
                            fill_price = fp
                            # Try to infer exit reason from order type
                            o_type = getattr(o, "type", "")
                            if o_type in ("stop", "stop_limit"):
                                exit_reason = "stop_loss"
                            elif o_type in ("limit",):
                                exit_reason = "take_profit"
                            elif o_type == "trailing_stop":
                                exit_reason = "trailing_stop"
                            break
            except Exception as exc:
                log.warning("Reconcile: could not fetch orders for %s: %s", symbol, exc)
                continue

            if fill_price is None or fill_price <= 0:
                log.debug(
                    "Reconcile: %s has open journal row but no closed fill found — "
                    "skipping (may have been manually closed or data unavailable)",
                    symbol,
                )
                continue

            if is_short:
                pnl = (entry_price - fill_price) * qty
            else:
                pnl = (fill_price - entry_price) * qty

            entry_time = row.get("entry_time", "")
            entry_date = entry_time[:10] if entry_time else ""
            hold_days = 0
            try:
                if entry_date:
                    hold_days = (dt.date.today() - dt.date.fromisoformat(entry_date)).days
            except Exception:
                pass

            try:
                self.journal.record_exit(
                    symbol,
                    exit_price=fill_price,
                    pnl=round(pnl, 4),
                    hold_days=hold_days,
                    exit_reason=exit_reason,
                    filled_price=fill_price,
                )
                log.info(
                    "Reconcile: journaled %s exit for %s  fill=%.2f  pnl=%.4f  reason=%s",
                    side.upper(), symbol, fill_price, pnl, exit_reason,
                )
            except Exception as exc:
                log.warning("Reconcile: failed to journal exit for %s: %s", symbol, exc)

    # ─────────────────────────────────────────────────────────
    # FULL CYCLE
    # ─────────────────────────────────────────────────────────
    def run_cycle(self) -> None:
        """Execute one full scan cycle: exits first, then entries."""
        try:
            session = self.broker.get_trading_session()
            if session == "closed":
                log.info("Trading session is closed – skipping cycle")
                return
            if session != "regular" and not getattr(config, "EXTENDED_HOURS_TRADING", False):
                log.info(
                    "Extended-hours session detected (%s) but extended trading is disabled – skipping cycle",
                    session,
                )
                return
        except Exception as exc:
            log.warning("Network error while checking market clock: %s – skipping cycle", exc)
            return

        log.info("=" * 60)
        log.info("CYCLE START  session=%s  equity=$%.2f  positions=%d",
                 session,
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
