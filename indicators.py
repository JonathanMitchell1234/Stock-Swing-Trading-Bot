"""
Technical-analysis indicator calculations.
All functions take a pandas DataFrame with OHLCV columns and return
a new DataFrame (or Series) with the indicator values appended.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta

import config
from logger import get_logger

log = get_logger("indicators")


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute every indicator the strategy needs and return
    the enriched DataFrame.  Expects columns: open, high, low, close, volume.
    """
    df = df.copy()

    # ── Exponential Moving Averages ──────────────────────────
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=config.EMA_FAST)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=config.EMA_SLOW)
    df["ema_trend"] = ta.trend.ema_indicator(df["close"], window=config.EMA_TREND)
    df["ema_200"] = ta.trend.ema_indicator(df["close"], window=200)

    # ── Simple Moving Average (200-day, for regime filter) ───
    df["sma_200"] = df["close"].rolling(window=200).mean()

    # ── RSI ──────────────────────────────────────────────────
    df["rsi"] = ta.momentum.rsi(df["close"], window=config.RSI_PERIOD)

    # ── MACD ─────────────────────────────────────────────────
    macd = ta.trend.MACD(
        df["close"],
        window_slow=config.MACD_SLOW,
        window_fast=config.MACD_FAST,
        window_sign=config.MACD_SIGNAL,
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # ── ATR (Average True Range) ─────────────────────────────
    df["atr"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=config.ATR_PERIOD
    )

    # ── Volume SMA ───────────────────────────────────────────
    df["vol_sma"] = df["volume"].rolling(window=config.VOLUME_SMA_PERIOD).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma"]

    # ── Bollinger Bands (supplementary) ──────────────────────
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()

    # ── Stochastic Oscillator (supplementary) ────────────────
    stoch = ta.momentum.StochasticOscillator(
        df["high"], df["low"], df["close"], window=14, smooth_window=3
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # ── ADX (trend strength) ─────────────────────────────────
    df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)

    # ── Support / Resistance levels ──────────────────────────
    df = _add_support_resistance(df)

    log.debug("Indicators computed for %d bars", len(df))
    return df


def _add_support_resistance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect swing-high (resistance) and swing-low (support) using a
    rolling window.  Each bar gets the most recent S/R levels.
    """
    lookback = config.SR_LOOKBACK
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    support = np.full(n, np.nan)
    resistance = np.full(n, np.nan)

    for i in range(lookback, n):
        window_high = highs[i - lookback : i]
        window_low = lows[i - lookback : i]
        resistance[i] = float(np.max(window_high))
        support[i] = float(np.min(window_low))

    df["sr_support"] = support
    df["sr_resistance"] = resistance
    return df


def compute_weekly_trend(df: pd.DataFrame) -> dict:
    """
    Resample daily OHLCV to weekly and check EMA alignment.
    Returns dict with 'bullish' (bool) and 'weekly_ema_fast'/'weekly_ema_slow'.
    Expects a DatetimeIndex.
    """
    if len(df) < config.WEEKLY_EMA_SLOW * 5:
        return {"bullish": True, "weekly_ema_fast": None, "weekly_ema_slow": None}

    weekly = df["close"].resample("W").last().dropna()

    if len(weekly) < config.WEEKLY_EMA_SLOW + 1:
        return {"bullish": True, "weekly_ema_fast": None, "weekly_ema_slow": None}

    wf = ta.trend.ema_indicator(weekly, window=config.WEEKLY_EMA_FAST)
    ws = ta.trend.ema_indicator(weekly, window=config.WEEKLY_EMA_SLOW)

    fast_val = wf.iloc[-1] if not pd.isna(wf.iloc[-1]) else None
    slow_val = ws.iloc[-1] if not pd.isna(ws.iloc[-1]) else None

    bullish = True
    if fast_val is not None and slow_val is not None:
        bullish = fast_val > slow_val

    return {"bullish": bullish, "weekly_ema_fast": fast_val, "weekly_ema_slow": slow_val}


def realized_volatility(df: pd.DataFrame, window: int = 20) -> float:
    """
    Compute annualized realized volatility from daily returns.
    Returns a float (e.g. 0.20 = 20% annualized vol).
    """
    if len(df) < window + 1:
        return 0.15  # default moderate vol

    returns = df["close"].pct_change().dropna().iloc[-window:]
    if len(returns) < window:
        return 0.15

    daily_std = float(returns.std())
    return daily_std * np.sqrt(252)


def latest_row(df: pd.DataFrame) -> pd.Series:
    """Return the most recent fully-formed bar (last row)."""
    return df.iloc[-1]


def prev_row(df: pd.DataFrame) -> pd.Series:
    """Return the second-to-last bar (for crossover detection)."""
    return df.iloc[-2]
