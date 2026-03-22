"""
Backtester – simulates the swing-trading strategy on historical data.

Features
--------
* Downloads historical daily bars via the Alpaca data API
* Replays bars day-by-day through the same strategy logic used live
* Enforces PDT rules (min 1-day hold) and the same risk-management
  position-sizing / ATR stops used in production
* Tracks equity curve, drawdown, per-trade log, and key statistics
* Prints a full report and optionally saves an equity-curve chart

Usage
-----
    python backtest.py                          # defaults: SPY, 1 year
    python backtest.py --symbols AAPL MSFT NVDA --months 6
    python backtest.py --all                    # full watchlist, 12 months
    python backtest.py --start 2024-06-01 --end 2025-06-01 --symbols QQQ
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import contextlib
import numpy as np
import pandas as pd

import config
from indicators import compute_all, compute_weekly_trend, realized_volatility
from strategy import check_entry as strategy_check_entry
from strategy import check_short_entry as strategy_check_short_entry
from strategy import check_short_exit as strategy_check_short_exit
from logger import get_logger


@contextlib.contextmanager
def _override_config(overrides: dict):
    """Temporarily patch the config module with the given key/value pairs."""
    if not overrides:
        yield
        return
    originals = {}
    for key, val in overrides.items():
        if hasattr(config, key):
            originals[key] = getattr(config, key)
            setattr(config, key, val)
    try:
        yield
    finally:
        for key, orig_val in originals.items():
            setattr(config, key, orig_val)

log = get_logger("backtest")

# Try to import matplotlib for charting; gracefully degrade if missing
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ═════════════════════════════════════════════════════════════
# Data classes
# ═════════════════════════════════════════════════════════════

@dataclass
class Trade:
    symbol: str
    side: str               # "BUY" or "SELL"
    date: dt.date
    price: float
    qty: float
    stop_loss: float = 0.0
    take_profit: float = 0.0
    reason: str = ""

@dataclass
class ClosedTrade:
    symbol: str
    entry_date: dt.date
    exit_date: dt.date
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_pct: float
    hold_days: int
    exit_reason: str
    side: str = "long"  # "long" or "short"

@dataclass
class Position:
    symbol: str
    entry_date: dt.date
    entry_price: float
    qty: float
    stop_loss: float
    take_profit: float
    highest_price: float = 0.0   # for trailing stop tracking

    def __post_init__(self):
        self.highest_price = self.entry_price

@dataclass
class ShortPosition:
    symbol: str
    entry_date: dt.date
    entry_price: float
    qty: float
    stop_loss: float
    take_profit: float
    lowest_price: float = 0.0   # for trailing stop tracking

    def __post_init__(self):
        self.lowest_price = self.entry_price


# ═════════════════════════════════════════════════════════════
# Backtester
# ═════════════════════════════════════════════════════════════

class Backtester:
    """
    Event-driven backtester that replays daily bars through the
    strategy and tracks portfolio performance.
    """

    def __init__(
        self,
        symbols: List[str],
        start_date: dt.date,
        end_date: dt.date,
        initial_capital: float = 300.0,
        param_overrides: Optional[dict] = None,
    ) -> None:
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.param_overrides = param_overrides or {}

        # Portfolio state
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.short_positions: Dict[str, ShortPosition] = {}
        self.equity_curve: List[Tuple[dt.date, float]] = []
        self.trade_log: List[ClosedTrade] = []
        self.all_trades: List[Trade] = []  # every buy/sell

        # Pre-fetched data: symbol → DataFrame
        self._data: Dict[str, pd.DataFrame] = {}

        # Market-regime data (SPY SMA-200)
        self._regime_data: Optional[pd.DataFrame] = None

        # VIX proxy data (VIXY)
        self._vix_data: Optional[pd.DataFrame] = None

        # Cooldown tracker: symbol → last exit date
        self._cooldowns: Dict[str, dt.date] = {}
        # Sector exposure tracker: sector -> count of open positions
        self._sector_counts: Dict[str, int] = {}
        # Commission model (Alpaca is commission-free, but slippage sim)
        self.slippage_pct = 0.0005  # 5 bps slippage per trade

    # ── data loading ─────────────────────────────────────────
    def _load_data(self) -> None:
        """Fetch historical bars for all symbols from Alpaca."""
        import alpaca_trade_api as tradeapi

        api = tradeapi.REST(
            key_id=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
            base_url=config.BASE_URL,
            api_version="v2",
        )

        # Fetch extra bars before start_date so indicators are warm
        warmup_start = self.start_date - dt.timedelta(days=200)

        for symbol in self.symbols:
            try:
                bars = api.get_bars(
                    symbol,
                    config.BAR_TIMEFRAME,
                    start=warmup_start.isoformat(),
                    end=self.end_date.isoformat(),
                    limit=10_000,
                    feed=config.DATA_FEED,
                )
                df = bars.df.copy()
                df.index = pd.to_datetime(df.index)
                df = df[["open", "high", "low", "close", "volume"]]

                if len(df) < config.EMA_TREND + 10:
                    log.warning("%s: not enough bars (%d), skipping", symbol, len(df))
                    continue

                # Compute all indicators once
                df = compute_all(df)
                self._data[symbol] = df
                log.info("Loaded %s: %d bars (%s to %s)", symbol, len(df),
                         df.index[0].date(), df.index[-1].date())
            except Exception as exc:
                log.warning("Failed to load %s: %s", symbol, exc)

        # Load market regime data (SPY for bull/bear filter)
        if config.MARKET_REGIME_ENABLED:
            regime_sym = config.MARKET_REGIME_SYMBOL
            if regime_sym in self._data:
                self._regime_data = self._data[regime_sym]
            else:
                try:
                    bars = api.get_bars(
                        regime_sym, config.BAR_TIMEFRAME,
                        start=warmup_start.isoformat(),
                        end=self.end_date.isoformat(),
                        limit=10_000,
                        feed=config.DATA_FEED,
                    )
                    rdf = bars.df.copy()
                    rdf.index = pd.to_datetime(rdf.index)
                    rdf = rdf[["open", "high", "low", "close", "volume"]]
                    rdf = compute_all(rdf)
                    self._regime_data = rdf
                    log.info("Loaded regime filter: %s (%d bars)", regime_sym, len(rdf))
                except Exception as exc:
                    log.warning("Failed to load regime data %s: %s", regime_sym, exc)

        # Load VIX proxy data (VIXY) for the fear/volatility filter
        if config.VIX_FILTER_ENABLED:
            vix_sym = config.VIX_SYMBOL
            if vix_sym in self._data:
                self._vix_data = self._data[vix_sym]
            else:
                try:
                    bars = api.get_bars(
                        vix_sym, config.BAR_TIMEFRAME,
                        start=warmup_start.isoformat(),
                        end=self.end_date.isoformat(),
                        limit=10_000,
                        feed=config.DATA_FEED,
                    )
                    vdf = bars.df.copy()
                    vdf.index = pd.to_datetime(vdf.index)
                    vdf = vdf[["open", "high", "low", "close", "volume"]]
                    vdf = compute_all(vdf)
                    self._vix_data = vdf
                    log.info("Loaded VIX proxy: %s (%d bars)", vix_sym, len(vdf))
                except Exception as exc:
                    log.warning("Failed to load VIX proxy %s: %s", vix_sym, exc)

        # Pre-load inverse ETFs if bear-mode is enabled (so they're ready to trade)
        if config.INVERSE_ETF_MODE_ENABLED:
            for sym in config.INVERSE_WATCHLIST:
                if sym not in self._data:
                    try:
                        bars = api.get_bars(
                            sym, config.BAR_TIMEFRAME,
                            start=warmup_start.isoformat(),
                            end=self.end_date.isoformat(),
                            limit=10_000,
                            feed=config.DATA_FEED,
                        )
                        idf = bars.df.copy()
                        idf.index = pd.to_datetime(idf.index)
                        idf = idf[["open", "high", "low", "close", "volume"]]
                        idf = compute_all(idf)
                        self._data[sym] = idf
                        log.info("Loaded inverse ETF: %s (%d bars)", sym, len(idf))
                    except Exception as exc:
                        log.warning("Failed to load inverse ETF %s: %s", sym, exc)

        if not self._data:
            log.error("No data loaded – cannot backtest")
            sys.exit(1)

    # ── helpers ──────────────────────────────────────────────
    def _portfolio_value(self, date: dt.date) -> float:
        """Total equity = cash + market value of all open positions."""
        value = self.cash
        for pos in self.positions.values():
            price = self._get_close(pos.symbol, date)
            if price:
                value += price * pos.qty
        for pos in self.short_positions.values():
            price = self._get_close(pos.symbol, date)
            if price:
                # Cash already includes short sale proceeds (entry_price * qty).
                # Subtract current cover cost to get true equity.
                value -= price * pos.qty
        return value

    def _get_close(self, symbol: str, date: dt.date) -> Optional[float]:
        df = self._data.get(symbol)
        if df is None:
            return None
        mask = df.index.date <= date
        if not mask.any():
            return None
        return float(df.loc[mask].iloc[-1]["close"])

    def _get_row(self, symbol: str, date: dt.date) -> Optional[pd.Series]:
        df = self._data.get(symbol)
        if df is None:
            return None
        mask = df.index.date == date
        rows = df.loc[mask]
        if rows.empty:
            return None
        return rows.iloc[-1]

    def _get_df_up_to(self, symbol: str, date: dt.date) -> Optional[pd.DataFrame]:
        """Return the indicator-enriched DF up to and including `date`."""
        df = self._data.get(symbol)
        if df is None:
            return None
        subset = df[df.index.date <= date]
        if len(subset) < config.EMA_TREND + 5:
            return None
        return subset

    def _apply_slippage(self, price: float, side: str) -> float:
        if side == "BUY":
            return price * (1 + self.slippage_pct)
        return price * (1 - self.slippage_pct)

    # ── position sizing (mirrors risk_manager.py) ────────────
    def _size_position(self, entry_price: float, stop_price: float, date: dt.date) -> float:
        equity = self._portfolio_value(date)
        if entry_price <= 0 or stop_price <= 0 or stop_price >= entry_price:
            return 0

        risk_per_share = entry_price - stop_price
        if risk_per_share <= 0:
            return 0

        # Dynamic parameters based on account size
        risk_pct = config.get_risk_per_trade(equity)
        pos_pct = config.get_position_pct(equity)

        max_risk = equity * risk_pct
        shares_by_risk = max_risk / risk_per_share

        max_pos_value = equity * pos_pct
        shares_by_value = max_pos_value / entry_price

        shares_by_cash = self.cash * 0.95 / entry_price

        qty_raw = min(shares_by_risk, shares_by_value, shares_by_cash)

        if config.FRACTIONAL_SHARES:
            return max(0.0, round(qty_raw, 3))
        return max(0, math.floor(qty_raw))

    def _size_short_position(self, entry_price: float, stop_price: float, date: dt.date) -> float:
        """Position sizing for SHORT trades (stop is ABOVE entry)."""
        equity = self._portfolio_value(date)
        if entry_price <= 0 or stop_price <= 0 or stop_price <= entry_price:
            return 0

        risk_per_share = stop_price - entry_price
        if risk_per_share <= 0:
            return 0

        risk_pct = config.get_risk_per_trade(equity)
        pos_pct = config.get_position_pct(equity)

        max_risk = equity * risk_pct
        shares_by_risk = max_risk / risk_per_share

        max_pos_value = equity * pos_pct
        shares_by_value = max_pos_value / entry_price

        shares_by_cash = self.cash * 0.95 / entry_price

        qty_raw = min(shares_by_risk, shares_by_value, shares_by_cash)

        # Apply bear-mode size scale (shorts are riskier)
        bear_scale = getattr(config, 'BEAR_SHORT_SIZE_SCALE', 0.6)
        qty_raw *= bear_scale

        if config.FRACTIONAL_SHARES:
            return max(0.0, round(qty_raw, 3))
        return max(0, math.floor(qty_raw))

    # ── market regime check ──────────────────────────────────
    def _is_bull_market(self, date: dt.date) -> bool:
        """Check if SPY is above its EMA-200 (matches executor.py regime detection)."""
        if not config.MARKET_REGIME_ENABLED or self._regime_data is None:
            return True  # default: allow trades
        mask = self._regime_data.index.date <= date
        if not mask.any():
            return True
        row = self._regime_data.loc[mask].iloc[-1]
        # Primary: use EMA-200 (matches executor._detect_regime)
        ema_200 = row.get("ema_200", None)
        if ema_200 is not None and not pd.isna(ema_200):
            return row["close"] > ema_200
        # Fallback: use SMA-200 if EMA unavailable
        sma_200 = row.get("sma_200", None)
        if sma_200 is not None and not pd.isna(sma_200):
            return row["close"] > sma_200
        return True

    # ── VIX fear-level helper ────────────────────────────────
    def _vix_size_scale(self, date: dt.date) -> float:
        """
        Returns a position-size scale factor based on the VIX proxy (VIXY):
          - VIXY > VIX_HALT_THRESHOLD  → 0.0  (no new entries at all)
          - VIXY > VIX_REDUCE_THRESHOLD → VIX_SIZE_SCALE  (cut size in half)
          - Otherwise                  → 1.0  (normal sizing)
        """
        if not config.VIX_FILTER_ENABLED or self._vix_data is None:
            return 1.0
        mask = self._vix_data.index.date <= date
        if not mask.any():
            return 1.0
        vixy_price = float(self._vix_data.loc[mask].iloc[-1]["close"])
        if vixy_price >= config.VIX_HALT_THRESHOLD:
            return 0.0   # panic mode — no new trades
        if vixy_price >= config.VIX_REDUCE_THRESHOLD:
            return config.VIX_SIZE_SCALE  # elevated fear — cut size
        return 1.0

    # ── momentum helper ──────────────────────────────────────
    def _compute_momentum(self, symbol: str, date: dt.date) -> float:
        """Compute the N-day rate of change for ranking."""
        df = self._data.get(symbol)
        if df is None:
            return 0.0
        subset = df[df.index.date <= date]
        lookback = config.MOMENTUM_LOOKBACK
        if len(subset) < lookback + 1:
            return 0.0
        cur_close = float(subset.iloc[-1]["close"])
        past_close = float(subset.iloc[-(lookback + 1)]["close"])
        if past_close <= 0 or pd.isna(past_close) or pd.isna(cur_close):
            return 0.0
        return (cur_close - past_close) / past_close

    # ── weekly trend helper ──────────────────────────────────
    def _is_weekly_bullish(self, symbol: str, date: dt.date) -> bool:
        """Check if the weekly EMA trend is bullish for a symbol."""
        if not config.WEEKLY_TREND_ENABLED:
            return True
        df = self._data.get(symbol)
        if df is None:
            return True
        subset = df[df.index.date <= date]
        if len(subset) < config.WEEKLY_EMA_SLOW * 5:
            return True
        info = compute_weekly_trend(subset)
        return info["bullish"]

    # ── volatility regime helper ─────────────────────────────
    def _vol_regime_scale(self, date: dt.date) -> float:
        """
        Compute position-size scale factor based on realized volatility
        of the regime symbol (SPY).  Returns 0.6 / 1.0 / 1.2.
        """
        if not config.VOL_REGIME_ENABLED:
            return 1.0
        regime_df = self._regime_data
        if regime_df is None:
            return 1.0
        subset = regime_df[regime_df.index.date <= date]
        vol = realized_volatility(subset, window=config.REALIZED_VOL_WINDOW)
        if vol > config.HIGH_VOL_THRESHOLD:
            return config.HIGH_VOL_SIZE_SCALE
        if vol < config.LOW_VOL_THRESHOLD:
            return config.LOW_VOL_SIZE_SCALE
        return 1.0

    # ── dynamic threshold helper ─────────────────────────────
    def _dynamic_score_threshold(self, date: dt.date) -> int:
        """
        Adjust the entry score threshold based on market quality.
        Strong market -> lower threshold (easier entries).
        Weak market -> higher threshold (more conviction needed).
        """
        base = config.ENTRY_SCORE_THRESHOLD
        if not config.DYNAMIC_THRESHOLD_ENABLED or self._regime_data is None:
            return base

        mask = self._regime_data.index.date <= date
        if not mask.any():
            return base
        regime_subset = self._regime_data.loc[mask]
        if len(regime_subset) < config.EMA_TREND + 5:
            return base

        row = regime_subset.iloc[-1]
        spy_close = row["close"]
        spy_ema50 = row.get("ema_trend", None)
        if spy_ema50 is None or pd.isna(spy_ema50):
            return base

        # Check if SPY EMA-50 is rising
        if len(regime_subset) >= config.EMA_SLOPE_PERIOD + 1:
            ema50_ago = regime_subset.iloc[-(config.EMA_SLOPE_PERIOD + 1)].get("ema_trend", None)
            if ema50_ago is not None and not pd.isna(ema50_ago):
                if spy_close > spy_ema50 and spy_ema50 > ema50_ago:
                    return base - 1  # strong market: lower bar (more entries)

        return base

    # ── sector limit helper ──────────────────────────────────
    def _sector_ok(self, symbol: str) -> bool:
        """Check if we can open a position in this symbol's sector."""
        sector = config.SECTOR_MAP.get(symbol, "Other")
        count = self._sector_counts.get(sector, 0)
        return count < config.MAX_PER_SECTOR

    def _sector_add(self, symbol: str) -> None:
        sector = config.SECTOR_MAP.get(symbol, "Other")
        self._sector_counts[sector] = self._sector_counts.get(sector, 0) + 1

    def _sector_remove(self, symbol: str) -> None:
        sector = config.SECTOR_MAP.get(symbol, "Other")
        count = self._sector_counts.get(sector, 0)
        self._sector_counts[sector] = max(0, count - 1)

    # ── strategy evaluation (v4 scoring + ML + advanced filters) ──
    def _check_entry(self, df: pd.DataFrame, momentum: float = 0.0,
                      weekly_bullish: bool = True,
                      score_threshold: int = 0,
                      inverse_mode: bool = False) -> Optional[dict]:
        """
        Entry check for backtesting.

        For normal (non-inverse) symbols: delegates to strategy.check_entry,
        which is the same function used in live/paper trading and already
        includes the full ML pipeline (GBM gate/replace modes).

        For inverse ETFs: uses the separate bearish scoring logic.
        """
        if inverse_mode:
            if len(df) < max(config.MOMENTUM_LOOKBACK + 1, config.EMA_SLOPE_PERIOD + 1, 3):
                return None
            cur = df.iloc[-1]
            prv = df.iloc[-2]
            return self._check_inverse_entry(df, cur, prv, momentum, score_threshold)

        # ── Normal symbols: use live strategy (includes ML) ──────────
        # Temporarily lower the config threshold if dynamic threshold says so,
        # then restore it. strategy.check_entry reads config.ENTRY_SCORE_THRESHOLD.
        original_threshold = config.ENTRY_SCORE_THRESHOLD
        if score_threshold > 0 and score_threshold != original_threshold:
            config.ENTRY_SCORE_THRESHOLD = score_threshold

        # Build the up-to-date SPY and VIXY slices for macro context
        # (mirrors how executor.py passes them in live mode)
        spy_df: pd.DataFrame | None = None
        vixy_df: pd.DataFrame | None = None
        if self._regime_data is not None and not self._regime_data.empty:
            bar_date = df.index[-1].date()
            spy_mask = self._regime_data.index.date <= bar_date
            if spy_mask.any():
                spy_df = self._regime_data.loc[spy_mask]
        if self._vix_data is not None and not self._vix_data.empty:
            bar_date = df.index[-1].date()
            vixy_mask = self._vix_data.index.date <= bar_date
            if vixy_mask.any():
                vixy_df = self._vix_data.loc[vixy_mask]

        try:
            signal = strategy_check_entry(
                df,
                weekly_bullish=weekly_bullish,
                spy_df=spy_df,
                vixy_df=vixy_df,
            )
        finally:
            config.ENTRY_SCORE_THRESHOLD = original_threshold

        if signal is None:
            return None

        # If a momentum bonus applies in the backtest, add it to the score
        # (live path doesn't compute momentum inline — that's the screener's job)
        if momentum > 0.05:
            signal["score"] = signal.get("score", 0) + config.MOMENTUM_SCORE_WEIGHT
            signal["reason"] += f", Momentum +{momentum*100:.0f}%"
        elif momentum > 0.02:
            signal["score"] = signal.get("score", 0) + 1
            signal["reason"] += f", Momentum +{momentum*100:.0f}%"

        return signal

    def _check_inverse_entry(self, df: pd.DataFrame, cur: pd.Series,
                              prv: pd.Series, momentum: float,
                              score_threshold: int) -> Optional[dict]:
        """
        Entry scoring for inverse ETFs (SQQQ, SPXS, SH, PSQ, SOXS).
        Bearish market conditions score positively here — inverse ETFs go UP
        when the market goes down, so we want to enter them when everything
        looks bearish.
        """
        price = cur["close"]
        ema_fast = cur["ema_fast"]
        ema_slow = cur["ema_slow"]
        ema_trend = cur["ema_trend"]
        ema_200 = cur.get("ema_200", None)
        rsi = cur["rsi"]
        macd_hist = cur["macd_hist"]
        adx = cur["adx"]
        atr = cur["atr"]
        vol_ratio = cur["vol_ratio"]
        bb_mid = cur.get("bb_mid", None)

        if pd.isna(rsi) or pd.isna(atr) or pd.isna(adx):
            return None

        # Gap-down filter: don't chase panic gaps DOWN in inverse ETFs
        # (means the underlying already crashed — likely to snap back)
        prev_close = prv["close"]
        today_open = cur["open"]
        if prev_close > 0 and today_open > 0:
            gap_pct = (today_open - prev_close) / prev_close
            if gap_pct < -config.GAP_UP_MAX_PCT:  # inverted: big gap down = skip
                return None

        score = 0
        factors = []

        # +2: Inverse ETF is in an uptrend (above its own EMA-50)
        # This means the bear move is sustained, not just a 1-day spike
        if price > ema_trend:
            score += 2
            factors.append("Inverse above EMA-50")

        # +2: Bearish EMA crossover on the inverse ETF itself
        if (prv["ema_fast"] <= prv["ema_slow"]) and (ema_fast > ema_slow):
            score += 2
            factors.append("EMA crossover (bear)")

        # +1: EMA-50 of the inverse ETF is rising (trend gaining strength)
        if len(df) >= config.EMA_SLOPE_PERIOD + 1:
            ema_now = cur["ema_trend"]
            ema_ago = df.iloc[-(config.EMA_SLOPE_PERIOD + 1)]["ema_trend"]
            if not pd.isna(ema_now) and not pd.isna(ema_ago) and ema_now > ema_ago:
                score += 1
                factors.append("EMA-50 rising")

        # +2: RSI in pullback zone on the inverse ETF (30-55) — it pulled back
        # within the broader uptrend, giving a better entry
        if 30 <= rsi <= 55:
            score += 2
            factors.append(f"RSI pullback ({rsi:.0f})")
        elif 55 < rsi <= 65:
            score += 1
            factors.append(f"RSI mid ({rsi:.0f})")

        # +1: MACD positive (inverse ETF gaining momentum)
        if macd_hist > 0 or (prv["macd_hist"] < 0 and macd_hist > prv["macd_hist"]):
            score += 1
            factors.append("MACD+")

        # +1: Volume above average (institutional selling = volume on inverse)
        if vol_ratio >= config.VOLUME_SURGE_FACTOR:
            score += 1
            factors.append(f"Vol {vol_ratio:.1f}x")

        # +1: ADX — bear trend has conviction
        if adx > 20:
            score += 1
            factors.append(f"ADX {adx:.0f}")

        # +1: Price near lower BB on the inverse ETF (pullback entry within uptrend)
        if bb_mid is not None and not pd.isna(bb_mid) and price <= bb_mid:
            score += 1
            factors.append("Near BB lower")

        # +2: Strong positive momentum on the inverse ETF itself
        if momentum > 0.05:
            score += config.MOMENTUM_SCORE_WEIGHT
            factors.append(f"Momentum +{momentum*100:.0f}%")
        elif momentum > 0.02:
            score += 1
            factors.append(f"Momentum +{momentum*100:.0f}%")

        threshold = score_threshold if score_threshold > 0 else config.ENTRY_SCORE_THRESHOLD
        if score < threshold:
            return None

        return {
            "price": price,
            "atr": atr,
            "score": score,
            "reason": f"[BEAR] Score {score}: {', '.join(factors)}",
        }

    def _check_exit(self, df: pd.DataFrame, entry_price: float = 0.0,
                    hold_days: int = 0) -> Optional[List[str]]:
        """Layered exit: hard exits fire immediately, soft exits need 2+ signals."""
        if len(df) < 4:
            return None

        cur = df.iloc[-1]
        prv = df.iloc[-2]

        rsi = cur["rsi"]
        macd_hist = cur["macd_hist"]
        ema_fast = cur["ema_fast"]
        ema_slow = cur["ema_slow"]
        ema_trend = cur["ema_trend"]
        ema_200 = cur.get("ema_200", None)
        price = cur["close"]

        if pd.isna(rsi):
            return None

        hard_reasons = []
        soft_reasons = []

        # HARD: price below BOTH EMA-50 and EMA-200
        if ema_200 is not None and not pd.isna(ema_200):
            if price < ema_trend and price < ema_200:
                hard_reasons.append("Below EMA-50 & EMA-200")

        # SOFT: RSI extremely overbought
        if rsi >= config.RSI_OVERBOUGHT:
            soft_reasons.append(f"RSI overbought ({rsi:.1f})")

        # SOFT: Bearish EMA crossover
        if (prv["ema_fast"] >= prv["ema_slow"]) and (ema_fast < ema_slow):
            soft_reasons.append("Bearish EMA crossover")

        # SOFT: MACD declining for 2+ bars
        if (prv["macd_hist"] < 0 and macd_hist < 0
                and macd_hist < prv["macd_hist"]):
            soft_reasons.append("MACD declining 2+ bars")

        # SOFT: price below EMA-50 (but still above 200)
        if price < ema_trend:
            if ema_200 is None or pd.isna(ema_200) or price >= ema_200:
                soft_reasons.append(f"Price below EMA-{config.EMA_TREND}")

        # SOFT: Dead money — position hasn't moved in N days
        if entry_price > 0 and hold_days >= config.DEAD_MONEY_DAYS:
            move_pct = abs(price - entry_price) / entry_price
            if move_pct < config.DEAD_MONEY_THRESHOLD:
                soft_reasons.append(
                    f"Dead money ({hold_days}d, {move_pct*100:.1f}% move)")

        # SOFT: Momentum decay — 20-day return is strongly negative + below EMA-50
        if len(df) >= config.MOMENTUM_LOOKBACK + 1:
            past_price = df.iloc[-(config.MOMENTUM_LOOKBACK + 1)]["close"]
            if not pd.isna(past_price) and past_price > 0:
                mom = (price - past_price) / past_price
                if mom < -0.05 and price < ema_trend:
                    soft_reasons.append(f"Momentum decay ({mom*100:.1f}%)")

        reasons = []
        if hard_reasons:
            reasons = hard_reasons
        elif len(soft_reasons) >= 2:
            reasons = soft_reasons

        return reasons if reasons else None

    # ── core loop ────────────────────────────────────────────
    def run(self) -> dict:
        """
        Run the backtest across the date range.
        Returns a summary statistics dict.
        """
        with _override_config(self.param_overrides):
            return self._run_inner()

    def _run_inner(self) -> dict:
        """Internal backtest loop (called inside the override context)."""
        log.info("=" * 60)
        log.info("BACKTEST START")
        log.info("  Symbols    : %s", ", ".join(self.symbols))
        log.info("  Period     : %s -> %s", self.start_date, self.end_date)
        log.info("  Capital    : $%s", f"{self.initial_capital:,.0f}")
        log.info("=" * 60)

        self._load_data()

        # Build a sorted list of all trading dates
        all_dates = set()
        for df in self._data.values():
            for d in df.index.date:
                if self.start_date <= d <= self.end_date:
                    all_dates.add(d)
        trading_dates = sorted(all_dates)

        if not trading_dates:
            log.error("No trading dates in range")
            return {}

        for date in trading_dates:
            self._process_day(date)

        # Close any remaining positions at last available price
        final_date = trading_dates[-1]
        for symbol in list(self.positions.keys()):
            self._close_position(symbol, final_date, "Backtest end")
        for symbol in list(self.short_positions.keys()):
            self._close_short_position(symbol, final_date, "Backtest end")

        stats = self._compute_stats(trading_dates)
        self._print_report(stats)
        return stats

    def _process_day(self, date: dt.date) -> None:
        """Process a single day: check stops -> check exits -> check entries."""

        # 1. Check stop-loss and take-profit on open positions
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            row = self._get_row(symbol, date)
            if row is None:
                continue

            low = row["low"]
            high = row["high"]
            close = row["close"]

            # Update trailing high
            pos.highest_price = max(pos.highest_price, high)

            # Stop-loss hit?
            if low <= pos.stop_loss:
                self._close_position(symbol, date, "Stop-loss hit",
                                     exit_price=pos.stop_loss)
                continue

            # Take-profit hit?
            if high >= pos.take_profit:
                self._close_position(symbol, date, "Take-profit hit",
                                     exit_price=pos.take_profit)
                continue

            # Adaptive trailing stop (Chandelier Exit): ATR-based, tighter as profit grows
            profit_pct = (pos.highest_price - pos.entry_price) / pos.entry_price
            if profit_pct >= config.TRAILING_STOP_TIGHT_ACTIVATE:
                atr_mult = getattr(config, "ATR_TRAILING_STOP_TIGHT_MULT", 1.0)
                fallback_pct = config.TRAILING_STOP_TIGHT_PCT
            elif profit_pct >= config.TRAILING_STOP_ACTIVATE_PCT:
                atr_mult = getattr(config, "ATR_TRAILING_STOP_MULT", 1.5)
                fallback_pct = config.TRAILING_STOP_PCT
            else:
                atr_mult = None
                fallback_pct = None

            if atr_mult is not None:
                # Use ATR-based Chandelier Exit when ATR available
                cur_atr = float(row.get("atr", 0) or 0)
                if cur_atr > 0:
                    trail_price = pos.highest_price - atr_mult * cur_atr
                else:
                    trail_price = pos.highest_price * (1 - fallback_pct)

                if low <= trail_price:
                    self._close_position(
                        symbol, date,
                        f"Chandelier stop (ATR×{atr_mult} from ${pos.highest_price:.2f})",
                        exit_price=trail_price,
                    )
                    continue

                # Ratchet up the hard stop to trailing level (never lower it)
                if trail_price > pos.stop_loss:
                    pos.stop_loss = trail_price

        # 1b. Check stop-loss and take-profit on SHORT positions
        for symbol in list(self.short_positions.keys()):
            pos = self.short_positions[symbol]
            row = self._get_row(symbol, date)
            if row is None:
                continue

            low = row["low"]
            high = row["high"]

            # Update trailing low (shorts profit when price drops)
            pos.lowest_price = min(pos.lowest_price, low)

            # Stop-loss hit? For shorts, stop is ABOVE entry → triggered by HIGH
            if high >= pos.stop_loss:
                self._close_short_position(symbol, date, "Stop-loss hit (short)",
                                           exit_price=pos.stop_loss)
                continue

            # Take-profit hit? For shorts, TP is BELOW entry → triggered by LOW
            if low <= pos.take_profit:
                self._close_short_position(symbol, date, "Take-profit hit (short)",
                                           exit_price=pos.take_profit)
                continue

            # Adaptive trailing stop for shorts (Chandelier Exit): ATR-based
            profit_pct = (pos.entry_price - pos.lowest_price) / pos.entry_price
            if profit_pct >= config.TRAILING_STOP_TIGHT_ACTIVATE:
                atr_mult = getattr(config, "ATR_TRAILING_STOP_TIGHT_MULT", 1.0)
                fallback_pct = config.TRAILING_STOP_TIGHT_PCT
            elif profit_pct >= config.TRAILING_STOP_ACTIVATE_PCT:
                atr_mult = getattr(config, "ATR_TRAILING_STOP_MULT", 1.5)
                fallback_pct = config.TRAILING_STOP_PCT
            else:
                atr_mult = None
                fallback_pct = None

            if atr_mult is not None:
                cur_atr = float(row.get("atr", 0) or 0)
                if cur_atr > 0:
                    trail_price = pos.lowest_price + atr_mult * cur_atr
                else:
                    trail_price = pos.lowest_price * (1 + fallback_pct)

                if high >= trail_price:
                    self._close_short_position(
                        symbol, date,
                        f"Chandelier stop short (ATR×{atr_mult} from ${pos.lowest_price:.2f})",
                        exit_price=trail_price,
                    )
                    continue

                # Ratchet DOWN the hard stop to trailing level (never raise it for shorts)
                if trail_price < pos.stop_loss:
                    pos.stop_loss = trail_price

        # 2. Check indicator-based exits (PDT: must have held >= MIN_HOLD days)
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            hold_days = (date - pos.entry_date).days
            if hold_days < config.MIN_HOLD_CALENDAR_DAYS:
                continue  # PDT protection

            df = self._get_df_up_to(symbol, date)
            if df is None:
                continue

            reasons = self._check_exit(df, entry_price=pos.entry_price,
                                       hold_days=hold_days)
            if reasons:
                self._close_position(symbol, date, "; ".join(reasons))

        # 2b. Check indicator-based exits for SHORT positions
        for symbol in list(self.short_positions.keys()):
            pos = self.short_positions[symbol]
            hold_days = (date - pos.entry_date).days
            if hold_days < config.MIN_HOLD_CALENDAR_DAYS:
                continue

            df = self._get_df_up_to(symbol, date)
            if df is None:
                continue

            signal = strategy_check_short_exit(
                df, entry_price=pos.entry_price,
                hold_days=hold_days, explain=True,
            )
            if signal and signal.get("should_exit"):
                reasons = signal.get("reasons", ["Short exit signal"])
                self._close_short_position(symbol, date, "; ".join(reasons))

        # 3. Scan for new entries (respect market regime + momentum ranking + v4 filters)
        bull_market = self._is_bull_market(date)
        equity = self._portfolio_value(date)
        max_positions = config.get_max_positions(equity)
        atr_stop_mult = config.get_atr_stop_mult(equity)
        atr_profit_mult = config.get_atr_profit_mult(equity)
        vol_scale = self._vol_regime_scale(date)
        vix_scale = self._vix_size_scale(date)
        dyn_threshold = self._dynamic_score_threshold(date)

        # VIX halt: if VIXY is in panic territory, block new LONG entries only.
        # Short entries and inverse ETF entries are still allowed.
        long_halted = (vix_scale == 0.0)
        bear_short_enabled = getattr(config, 'BEAR_SHORT_MODE_ENABLED', False)
        short_min_equity = getattr(config, 'SHORT_MIN_EQUITY', 2000.0)
        has_inverse_watchlist = bool(config.INVERSE_WATCHLIST)

        if long_halted and not has_inverse_watchlist and not bear_short_enabled:
            equity = self._portfolio_value(date)
            self.equity_curve.append((date, equity))
            return

        # Determine which mode to use based on market regime + equity
        # In bear regime:
        #   - If equity >= SHORT_MIN_EQUITY → short-sell (original behavior)
        #   - If equity <  SHORT_MIN_EQUITY → buy inverse ETFs (no margin needed)
        # The INVERSE_ETF_MODE_ENABLED static flag is now dynamically overridden
        # by the equity check, but still respected if explicitly set.
        bear_regime = not bull_market or long_halted

        if bear_regime and bear_short_enabled and equity >= short_min_equity:
            # Enough equity to short-sell
            inverse_mode = False
            short_mode = True
        elif bear_regime and has_inverse_watchlist:
            # Not enough equity (or shorts disabled) → use inverse ETFs
            inverse_mode = True
            short_mode = False
            if equity < short_min_equity:
                log.debug(
                    "Bear mode on %s — equity $%.2f < $%.2f short minimum — "
                    "using INVERSE ETF mode instead of shorting",
                    date, equity, short_min_equity,
                )
        elif bear_regime and bear_short_enabled:
            # Shorts enabled but no inverse watchlist and equity check passes above
            inverse_mode = False
            short_mode = True
        else:
            inverse_mode = False
            short_mode = False

        if inverse_mode:
            # Bear market: trade inverse ETFs only (no longs on regular stocks)
            scan_symbols = [s for s in config.INVERSE_WATCHLIST if s in self._data]
            vol_scale = vol_scale * config.INVERSE_ETF_SIZE_SCALE
            log.debug("Bear mode on %s (bull=%s, vix_halt=%s) — scanning inverse ETFs: %s",
                      date, bull_market, long_halted, scan_symbols)
        elif short_mode:
            # Bear market with short-selling: scan normal watchlist for shorts
            short_blacklist = getattr(config, 'SHORT_BLACKLIST', [])
            scan_symbols = [
                s for s in self.symbols
                if s not in config.INVERSE_WATCHLIST and s not in short_blacklist
            ]
            log.debug("Bear SHORT mode on %s (bull=%s, vix_halt=%s) — scanning %d symbols for shorts",
                      date, bull_market, long_halted, len(scan_symbols))
        else:
            # Bull market: trade the normal watchlist (exclude inverse ETFs)
            scan_symbols = [
                s for s in self.symbols
                if s not in config.INVERSE_WATCHLIST
            ]
            # Apply VIX reduce-scale to longs if elevated fear
            vix_scale = vix_scale  # already computed above (0.5 or 1.0)

        total_positions = len(self.positions) + len(self.short_positions)
        can_enter = bull_market or inverse_mode or short_mode
        if total_positions < max_positions and can_enter:
            # Phase 1: Gather candidates with momentum scores
            candidates = []
            for symbol in scan_symbols:
                if symbol in self.positions or symbol in self.short_positions:
                    continue

                # Sector exposure limit
                if not self._sector_ok(symbol):
                    continue

                # Re-entry cooldown
                last_exit = self._cooldowns.get(symbol)
                if last_exit and (date - last_exit).days < config.RE_ENTRY_COOLDOWN_DAYS:
                    continue

                df = self._get_df_up_to(symbol, date)
                if df is None:
                    continue

                row = self._get_row(symbol, date)
                if row is None:
                    continue

                # Price / volume pre-filter
                price = row["close"]
                if price < config.MIN_PRICE or price > config.MAX_PRICE:
                    continue

                vol_sma = row.get("vol_sma", 0)
                if pd.isna(vol_sma) or vol_sma < config.MIN_AVG_VOLUME:
                    continue

                # Compute momentum
                mom = self._compute_momentum(symbol, date)
                candidates.append((symbol, df, mom))

            if not candidates:
                equity = self._portfolio_value(date)
                self.equity_curve.append((date, equity))
                return

            # Phase 2: Rank by momentum, keep top half
            # For shorts: rank by WEAKEST momentum (most bearish)
            if short_mode:
                candidates.sort(key=lambda x: x[2], reverse=False)
            else:
                candidates.sort(key=lambda x: x[2], reverse=True)
            cutoff = max(1, int(len(candidates) * config.MOMENTUM_TOP_PCT))
            top_candidates = candidates[:cutoff]

            # Phase 3: Score and enter the best (with v4 advanced filters)
            for symbol, df, mom in top_candidates:
                if len(self.positions) + len(self.short_positions) >= max_positions:
                    break

                # Sector re-check (may have filled during this loop)
                if not self._sector_ok(symbol):
                    continue

                # Multi-timeframe: weekly trend check (hard filter)
                # Skip weekly filter for inverse ETFs and short entries
                weekly_bull = self._is_weekly_bullish(symbol, date)
                is_inverse = symbol in config.INVERSE_WATCHLIST
                if config.WEEKLY_TREND_ENABLED and not weekly_bull and not is_inverse and not short_mode:
                    continue  # skip long entry when weekly trend disagrees

                # ── Route to short entry or long entry ──────
                if short_mode:
                    # SHORT ENTRY path
                    spy_slice = None
                    vixy_slice = None
                    if self._regime_data is not None and not self._regime_data.empty:
                        spy_mask = self._regime_data.index.date <= date
                        if spy_mask.any():
                            spy_slice = self._regime_data.loc[spy_mask]
                    if self._vix_data is not None and not self._vix_data.empty:
                        vixy_mask = self._vix_data.index.date <= date
                        if vixy_mask.any():
                            vixy_slice = self._vix_data.loc[vixy_mask]

                    signal = strategy_check_short_entry(
                        df, weekly_bullish=weekly_bull,
                        spy_df=spy_slice, vixy_df=vixy_slice,
                    )
                    if signal is None:
                        continue

                    if signal.get("score", 0) < dyn_threshold:
                        continue

                    entry_price = self._apply_slippage(signal["price"], "SELL")
                    atr = signal["atr"]
                    # For shorts: stop is ABOVE entry, target is BELOW entry
                    stop_loss = round(entry_price + atr * atr_stop_mult, 2)
                    take_profit = round(entry_price - atr * atr_profit_mult, 2)

                    qty = self._size_short_position(entry_price, stop_loss, date)

                    # Apply combined scale: volatility regime + VIX
                    combined_scale = vol_scale * (vix_scale if vix_scale > 0 else 1.0)
                    if combined_scale != 1.0:
                        qty = round(qty * combined_scale, 3) if config.FRACTIONAL_SHARES else math.floor(qty * combined_scale)

                    if qty <= 0:
                        continue

                    cost = entry_price * qty
                    if cost > self.cash:
                        continue

                    # Execute short sell (receive cash proceeds)
                    self.cash += cost  # short sale proceeds credited
                    self.short_positions[symbol] = ShortPosition(
                        symbol=symbol,
                        entry_date=date,
                        entry_price=entry_price,
                        qty=qty,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                    )
                    self.all_trades.append(Trade(
                        symbol=symbol, side="SELL", date=date, price=entry_price,
                        qty=qty, stop_loss=stop_loss, take_profit=take_profit,
                        reason=signal.get("reason", "SHORT entry"),
                    ))
                    self._sector_add(symbol)
                    log.debug("SHORT ENTRY %s on %s  qty=%.3f  price=%.2f  SL=%.2f  TP=%.2f  [%s]",
                              symbol, date, qty, entry_price, stop_loss, take_profit,
                              signal.get("reason", ""))
                else:
                    # LONG ENTRY path (bull market or inverse ETF)
                    signal = self._check_entry(
                        df, momentum=mom, weekly_bullish=weekly_bull,
                        score_threshold=dyn_threshold,
                        inverse_mode=is_inverse,
                    )
                    if signal is None:
                        continue

                    # ── NLP Sentiment Check (Backtest) ──────────
                    if getattr(config, "NLP_SENTIMENT_ENABLED", False):
                        import sentiment
                        from broker import AlpacaBroker
                        
                        if not hasattr(self, "_alpaca_broker"):
                            self._alpaca_broker = AlpacaBroker()
                        if not hasattr(self, "_sentiment_cache"):
                            self._sentiment_cache = {}
                            
                        end_str = date.strftime("%Y-%m-%dT23:59:59Z")
                        cache_key = f"{symbol}_{end_str}"
                        
                        if cache_key not in self._sentiment_cache:
                            news_limit = getattr(config, "NLP_NEWS_LIMIT_PER_SYMBOL", 10)
                            headlines = self._alpaca_broker.get_news(symbol, limit=news_limit, end=end_str)
                            sentiment_score = sentiment.get_sentiment(headlines)
                            self._sentiment_cache[cache_key] = sentiment_score
                            log.info("BACKTEST NLP %s on %s: Score = %.2f", symbol, date, sentiment_score)
                        
                        sentiment_score = self._sentiment_cache[cache_key]
                        min_sentiment = getattr(config, "NLP_MIN_SENTIMENT", -0.20)
                        
                        if sentiment_score < min_sentiment:
                            log.debug(
                                "BACKTEST SKIP %s on %s – Negative news sentiment (%.2f < %.2f)",
                                symbol, date, sentiment_score, min_sentiment
                            )
                            continue

                    entry_price = self._apply_slippage(signal["price"], "BUY")
                    atr = signal["atr"]
                    stop_loss = round(entry_price - atr * atr_stop_mult, 2)
                    take_profit = round(entry_price + atr * atr_profit_mult, 2)

                    qty = self._size_position(entry_price, stop_loss, date)

                    # Apply combined scale: volatility regime + VIX fear filter
                    combined_scale = vol_scale * vix_scale
                    if combined_scale != 1.0:
                        qty = round(qty * combined_scale, 3) if config.FRACTIONAL_SHARES else math.floor(qty * combined_scale)

                    if qty <= 0:
                        continue

                    cost = entry_price * qty
                    if cost > self.cash:
                        continue

                    # Execute buy
                    self.cash -= cost
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        entry_date=date,
                        entry_price=entry_price,
                        qty=qty,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                    )
                    self.all_trades.append(Trade(
                        symbol=symbol, side="BUY", date=date, price=entry_price,
                        qty=qty, stop_loss=stop_loss, take_profit=take_profit,
                        reason=signal["reason"],
                    ))
                    self._sector_add(symbol)

        # Record equity at end of day
        equity = self._portfolio_value(date)
        self.equity_curve.append((date, equity))

    def _close_position(
        self,
        symbol: str,
        date: dt.date,
        reason: str,
        exit_price: Optional[float] = None,
    ) -> None:
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return

        if exit_price is None:
            exit_price = self._get_close(symbol, date) or pos.entry_price
        exit_price = self._apply_slippage(exit_price, "SELL")

        pnl = (exit_price - pos.entry_price) * pos.qty
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        hold_days = (date - pos.entry_date).days

        self.cash += exit_price * pos.qty

        # Track sector removal
        self._sector_remove(symbol)

        # Record cooldown so we don't re-enter this symbol too soon
        self._cooldowns[symbol] = date

        self.trade_log.append(ClosedTrade(
            symbol=symbol,
            entry_date=pos.entry_date,
            exit_date=date,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            qty=pos.qty,
            pnl=pnl,
            pnl_pct=pnl_pct,
            hold_days=hold_days,
            exit_reason=reason,
        ))
        self.all_trades.append(Trade(
            symbol=symbol, side="SELL", date=date, price=exit_price,
            qty=pos.qty, reason=reason,
        ))

    def _close_short_position(
        self,
        symbol: str,
        date: dt.date,
        reason: str,
        exit_price: Optional[float] = None,
    ) -> None:
        pos = self.short_positions.pop(symbol, None)
        if pos is None:
            return

        if exit_price is None:
            exit_price = self._get_close(symbol, date) or pos.entry_price
        exit_price = self._apply_slippage(exit_price, "BUY")

        pnl = (pos.entry_price - exit_price) * pos.qty
        pnl_pct = (pos.entry_price - exit_price) / pos.entry_price
        hold_days = (date - pos.entry_date).days

        # Buy-to-cover: debit the cover cost from cash.
        # (Short sale proceeds were credited at entry: +entry_price*qty)
        # Net effect on cash: entry_price*qty - exit_price*qty = pnl
        self.cash -= exit_price * pos.qty

        # Track sector removal
        self._sector_remove(symbol)

        # Record cooldown so we don't re-enter this symbol too soon
        self._cooldowns[symbol] = date

        self.trade_log.append(ClosedTrade(
            symbol=symbol,
            entry_date=pos.entry_date,
            exit_date=date,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            qty=pos.qty,
            pnl=pnl,
            pnl_pct=pnl_pct,
            hold_days=hold_days,
            exit_reason=reason,
            side="short",
        ))
        self.all_trades.append(Trade(
            symbol=symbol, side="BUY", date=date, price=exit_price,
            qty=pos.qty, reason=reason,
        ))

    # ── statistics ───────────────────────────────────────────
    def _compute_stats(self, trading_dates: List[dt.date]) -> dict:
        if not self.equity_curve:
            return {}

        equities = pd.Series(
            [e for _, e in self.equity_curve],
            index=[d for d, _ in self.equity_curve],
        )

        total_return = (equities.iloc[-1] - self.initial_capital) / self.initial_capital
        peak = equities.expanding().max()
        drawdown = (equities - peak) / peak
        max_drawdown = drawdown.min()

        # Daily returns
        daily_returns = equities.pct_change().dropna()
        sharpe = 0.0
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * math.sqrt(252)

        # Trade stats
        total_trades = len(self.trade_log)
        winners = [t for t in self.trade_log if t.pnl > 0]
        losers = [t for t in self.trade_log if t.pnl <= 0]
        win_rate = len(winners) / total_trades if total_trades > 0 else 0

        avg_win = np.mean([t.pnl_pct for t in winners]) if winners else 0
        avg_loss = np.mean([t.pnl_pct for t in losers]) if losers else 0
        profit_factor = (
            abs(sum(t.pnl for t in winners)) / abs(sum(t.pnl for t in losers))
            if losers and sum(t.pnl for t in losers) != 0
            else float("inf")
        )

        avg_hold = np.mean([t.hold_days for t in self.trade_log]) if self.trade_log else 0

        # Calmar ratio
        years = max((trading_dates[-1] - trading_dates[0]).days / 365.25, 0.01)
        annual_return = (1 + total_return) ** (1 / years) - 1
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            "start_date": str(trading_dates[0]),
            "end_date": str(trading_dates[-1]),
            "trading_days": len(trading_dates),
            "initial_capital": self.initial_capital,
            "final_equity": equities.iloc[-1],
            "total_return_pct": total_return * 100,
            "annual_return_pct": annual_return * 100,
            "max_drawdown_pct": max_drawdown * 100,
            "sharpe_ratio": sharpe,
            "calmar_ratio": calmar,
            "profit_factor": profit_factor,
            "total_trades": total_trades,
            "winners": len(winners),
            "losers": len(losers),
            "win_rate_pct": win_rate * 100,
            "avg_win_pct": avg_win * 100,
            "avg_loss_pct": avg_loss * 100,
            "avg_hold_days": avg_hold,
            "max_consecutive_wins": self._max_consecutive(True),
            "max_consecutive_losses": self._max_consecutive(False),
            "long_trades": len([t for t in self.trade_log if getattr(t, 'side', 'long') == 'long']),
            "short_trades": len([t for t in self.trade_log if getattr(t, 'side', 'long') == 'short']),
        }

    def _max_consecutive(self, winning: bool) -> int:
        """Max consecutive wins or losses."""
        best = current = 0
        for t in self.trade_log:
            if (t.pnl > 0) == winning:
                current += 1
                best = max(best, current)
            else:
                current = 0
        return best

    # ── reporting ────────────────────────────────────────────
    def _print_report(self, stats: dict) -> None:
        if not stats:
            print("\n  No statistics to report.\n")
            return

        print("\n" + "=" * 68)
        print("  BACKTEST REPORT")
        print("=" * 68)
        print(f"  Period          : {stats['start_date']}  ->  {stats['end_date']}")
        print(f"  Trading days    : {stats['trading_days']}")
        print(f"  Symbols tested  : {len(self.symbols)}")
        print("-" * 68)
        print(f"  Initial capital : ${stats['initial_capital']:>12,.2f}")
        print(f"  Final equity    : ${stats['final_equity']:>12,.2f}")
        print(f"  Total return    :  {stats['total_return_pct']:>+10.2f}%")
        print(f"  Annual return   :  {stats['annual_return_pct']:>+10.2f}%")
        print(f"  Max drawdown    :  {stats['max_drawdown_pct']:>10.2f}%")
        print("-" * 68)
        print(f"  Sharpe ratio    :  {stats['sharpe_ratio']:>10.2f}")
        print(f"  Calmar ratio    :  {stats['calmar_ratio']:>10.2f}")
        print(f"  Profit factor   :  {stats['profit_factor']:>10.2f}")
        print("-" * 68)
        print(f"  Total trades    :  {stats['total_trades']:>10d}")
        print(f"    Long trades   :  {stats.get('long_trades', stats['total_trades']):>10d}")
        print(f"    Short trades  :  {stats.get('short_trades', 0):>10d}")
        print(f"  Winners         :  {stats['winners']:>10d}")
        print(f"  Losers          :  {stats['losers']:>10d}")
        print(f"  Win rate        :  {stats['win_rate_pct']:>10.1f}%")
        print(f"  Avg win         :  {stats['avg_win_pct']:>+10.2f}%")
        print(f"  Avg loss        :  {stats['avg_loss_pct']:>+10.2f}%")
        print(f"  Avg hold (days) :  {stats['avg_hold_days']:>10.1f}")
        print(f"  Max consec wins :  {stats['max_consecutive_wins']:>10d}")
        print(f"  Max consec loss :  {stats['max_consecutive_losses']:>10d}")
        print("=" * 68)

        # Trade log
        if self.trade_log:
            print("\n  TRADE LOG (last 30)")
            print(f"  {'Symbol':<7} {'Side':<6} {'Entry':>10} {'Exit':>10} {'Entry$':>9} {'Exit$':>9}"
                  f" {'P&L':>9} {'P&L%':>7} {'Days':>5} {'Reason'}")
            print("  " + "-" * 100)
            for t in self.trade_log[-30:]:
                side_str = getattr(t, 'side', 'long').upper()
                print(
                    f"  {t.symbol:<7} {side_str:<6} {str(t.entry_date):>10} {str(t.exit_date):>10}"
                    f" {t.entry_price:>9.2f} {t.exit_price:>9.2f}"
                    f" {t.pnl:>+9.2f} {t.pnl_pct * 100:>+6.2f}%"
                    f" {t.hold_days:>5d}  {t.exit_reason}"
                )
        print()

    # ── charting ─────────────────────────────────────────────
    def save_chart(self, path: str = "logs/backtest_equity.png") -> None:
        """Save an equity-curve + drawdown chart to disk."""
        if not HAS_MPL:
            log.warning("matplotlib not installed – skipping chart")
            return
        if not self.equity_curve:
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)

        dates = [d for d, _ in self.equity_curve]
        equities = [e for _, e in self.equity_curve]
        eq_series = pd.Series(equities, index=dates)
        peak = eq_series.expanding().max()
        dd = ((eq_series - peak) / peak) * 100

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1]})

        # Equity curve
        ax1.plot(dates, equities, linewidth=1.5, color="#2196F3", label="Equity")
        ax1.axhline(y=self.initial_capital, color="gray", linestyle="--",
                     linewidth=0.8, label="Starting Capital")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.set_title("Backtest — Equity Curve", fontsize=14, fontweight="bold")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        # Mark trades
        for t in self.all_trades:
            if t.side == "BUY":
                ax1.axvline(x=t.date, color="green", alpha=0.15, linewidth=0.5)
            else:
                ax1.axvline(x=t.date, color="red", alpha=0.15, linewidth=0.5)

        # Drawdown
        ax2.fill_between(dates, dd, 0, color="#F44336", alpha=0.4, label="Drawdown")
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)

        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        fig.autofmt_xdate()

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("Equity chart saved to %s", path)
        print(f"\n  Chart saved -> {path}\n")


# ═════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest the swing-trading strategy on historical data",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=["SPY"],
        help="Symbols to backtest (default: SPY)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Use the full watchlist from config.py",
    )
    parser.add_argument(
        "--months", type=int, default=12,
        help="Lookback period in months (default: 12)",
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Start date YYYY-MM-DD (overrides --months)",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--capital", type=float, default=300,
        help="Starting capital (default: 300)",
    )
    parser.add_argument(
        "--chart", action="store_true", default=True,
        help="Save equity-curve chart (default: True)",
    )
    parser.add_argument(
        "--no-chart", action="store_true",
        help="Skip chart generation",
    )
    args = parser.parse_args()

    if not config.ALPACA_API_KEY or config.ALPACA_API_KEY == "your_api_key_here":
        print("\n  ERROR: Set your Alpaca API keys in .env (see .env.example)")
        print("  (Backtesting needs API access to download historical data)\n")
        sys.exit(1)

    symbols = config.WATCHLIST if args.all else args.symbols
    end_date = dt.date.fromisoformat(args.end) if args.end else dt.date.today()

    if args.start:
        start_date = dt.date.fromisoformat(args.start)
    else:
        start_date = end_date - dt.timedelta(days=args.months * 30)

    bt = Backtester(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
    )
    stats = bt.run()

    if not args.no_chart and stats:
        bt.save_chart()


if __name__ == "__main__":
    main()
