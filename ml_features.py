"""
ML Feature Extraction – builds a fixed-width feature vector from an
indicator-enriched DataFrame.

Used by both the training pipeline (ml_trainer.py) and live inference
(ml_model.py).  Every feature is scale-invariant (ratios, not raw prices)
so the model generalises across price ranges.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from strategy import score_entry

# ─────────────────────────────────────────────────────────────
# Feature names (in order) — keep in sync with extract_row()
# ─────────────────────────────────────────────────────────────
FEATURE_NAMES: list[str] = [
    # Price-relative (distance to moving averages)
    "dist_ema_fast",        # close/ema9 - 1
    "dist_ema_slow",        # close/ema21 - 1
    "dist_ema_trend",       # close/ema50 - 1
    "dist_ema_200",         # close/ema200 - 1
    "ema_fast_slow_ratio",  # ema9/ema21 - 1
    "ema_slow_trend_ratio", # ema21/ema50 - 1
    # Oscillators
    "rsi",
    "rsi_delta",            # rsi change from prev bar
    "macd_hist",
    "macd_hist_delta",      # macd_hist change from prev bar
    "adx",
    "stoch_k",
    "stoch_d",
    "stoch_k_minus_d",
    # Oscillator velocity (3-day slopes — captures momentum direction)
    "rsi_slope_3d",         # rsi now − rsi 3 bars ago
    "macd_hist_slope_3d",   # macd_hist 3-bar change
    "adx_slope_3d",         # adx 3-bar change
    "stoch_k_slope_3d",     # stoch_k 3-bar change
    "vol_ratio_slope_3d",   # vol_ratio 3-bar change
    # Volatility
    "atr_ratio",            # atr / atr_sma50  (normalised volatility)
    "bb_width",             # (upper-lower) / mid
    "bb_pctb",              # (close-lower) / (upper-lower)
    "rvol_5",               # 5-day realized vol (annualised)
    "rvol_20",              # 20-day realized vol
    # Volume
    "vol_ratio",
    "vol_delta",            # volume / prev volume
    "vol_trend",            # vol SMA-5 / vol SMA-20
    # Returns
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    # Trend
    "ema50_slope",          # normalised 5-bar slope of EMA-50
    "ema200_slope",         # normalised 10-bar slope of EMA-200
    "higher_high_5",        # close > max(high[-6:-1])
    "higher_low_5",         # low > min(low[-6:-1])
    # Support / Resistance
    "dist_to_support",
    "dist_to_resistance",
    # Hand-crafted strategy score
    "entry_score",
    # Calendar
    "day_of_week",          # 0=Mon … 4=Fri
    # Macro context (replaces raw month — avoids calendar overfitting)
    "spy_sma200_dist",      # SPY close / SPY SMA-200 − 1
    "vixy_relative",        # VIXY close / VIXY 20-SMA − 1
]

NUM_FEATURES = len(FEATURE_NAMES)


# ─────────────────────────────────────────────────────────────
# Per-row extraction
# ─────────────────────────────────────────────────────────────

def _safe_div(a, b, default: float = 0.0) -> float:
    """Safe division returning *default* when b is 0, NaN, or None."""
    if b is None or b == 0:
        return default
    try:
        if np.isnan(b):
            return default
    except (TypeError, ValueError):
        pass
    return float(a) / float(b)


def _safe(val, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        if np.isnan(val):
            return default
    except (TypeError, ValueError):
        pass
    return float(val)


def extract_row(df: pd.DataFrame, idx: int = -1, weekly_bullish: bool = True,
                spy_df: pd.DataFrame | None = None,
                vixy_df: pd.DataFrame | None = None) -> list[float] | None:
    """
    Extract a feature vector from bar *idx* of an indicator-enriched DF.

    Parameters
    ----------
    spy_df  : indicator-enriched SPY DataFrame (for macro regime context)
    vixy_df : indicator-enriched VIXY DataFrame (for fear/vol context)

    Returns a list of floats (length == NUM_FEATURES), or None if there
    isn't enough data.
    """
    # Need at least 21 bars before the target row for look-back features
    abs_idx = idx if idx >= 0 else len(df) + idx
    if abs_idx < 25 or abs_idx >= len(df):
        return None

    cur = df.iloc[abs_idx]
    prv = df.iloc[abs_idx - 1]
    close = _safe(cur["close"])
    if close <= 0:
        return None

    # ── Price-relative ───────────────────────────────────────
    ema_fast  = _safe(cur.get("ema_fast"))
    ema_slow  = _safe(cur.get("ema_slow"))
    ema_trend = _safe(cur.get("ema_trend"))
    ema_200   = _safe(cur.get("ema_200"))

    dist_ema_fast  = _safe_div(close, ema_fast) - 1.0 if ema_fast else 0.0
    dist_ema_slow  = _safe_div(close, ema_slow) - 1.0 if ema_slow else 0.0
    dist_ema_trend = _safe_div(close, ema_trend) - 1.0 if ema_trend else 0.0
    dist_ema_200   = _safe_div(close, ema_200) - 1.0 if ema_200 else 0.0

    ema_fs_ratio = _safe_div(ema_fast, ema_slow) - 1.0 if ema_slow else 0.0
    ema_st_ratio = _safe_div(ema_slow, ema_trend) - 1.0 if ema_trend else 0.0

    # ── Oscillators ──────────────────────────────────────────
    rsi       = _safe(cur.get("rsi"), 50.0)
    rsi_prev  = _safe(prv.get("rsi"), 50.0)
    rsi_delta = rsi - rsi_prev

    macd_hist       = _safe(cur.get("macd_hist"))
    macd_hist_prev  = _safe(prv.get("macd_hist"))
    macd_hist_delta = macd_hist - macd_hist_prev

    adx     = _safe(cur.get("adx"), 20.0)
    stoch_k = _safe(cur.get("stoch_k"), 50.0)
    stoch_d = _safe(cur.get("stoch_d"), 50.0)
    stoch_kd = stoch_k - stoch_d

    # ── Oscillator velocity (3-day slopes) ───────────────────
    def _vel(col: str, n: int = 3, default: float = 0.0) -> float:
        """Value now minus value n bars ago."""
        back_idx = abs_idx - n
        if back_idx < 0:
            return default
        return _safe(df.iloc[abs_idx].get(col), default) - _safe(df.iloc[back_idx].get(col), default)

    rsi_slope_3d       = _vel("rsi", 3)
    macd_hist_slope_3d = _vel("macd_hist", 3)
    adx_slope_3d       = _vel("adx", 3)
    stoch_k_slope_3d   = _vel("stoch_k", 3)
    vol_ratio_slope_3d = _vel("vol_ratio", 3)

    # ── Volatility ───────────────────────────────────────────
    atr = _safe(cur.get("atr"))
    # Normalised volatility: ATR relative to its own 50-bar mean.
    # This democratises the watchlist — a blue-chip 2× its normal ATR
    # looks the same as a tech stock 2× its normal ATR.
    atr_slice = df["atr"].iloc[max(0, abs_idx - 49):abs_idx + 1]
    atr_sma50 = float(atr_slice.mean()) if len(atr_slice) >= 10 else float(atr) if atr > 0 else 1.0
    atr_ratio = _safe_div(atr, atr_sma50, default=1.0)

    bb_upper = _safe(cur.get("bb_upper"))
    bb_lower = _safe(cur.get("bb_lower"))
    bb_mid   = _safe(cur.get("bb_mid"))
    bb_range = bb_upper - bb_lower
    bb_width = _safe_div(bb_range, bb_mid)
    bb_pctb  = _safe_div(close - bb_lower, bb_range, 0.5)

    # Realised volatility (5-day and 20-day annualised)
    closes = df["close"].iloc[max(0, abs_idx - 20):abs_idx + 1]
    rets = closes.pct_change().dropna()
    rvol_5  = float(rets.iloc[-5:].std() * np.sqrt(252)) if len(rets) >= 5 else 0.15
    rvol_20 = float(rets.std() * np.sqrt(252)) if len(rets) >= 10 else 0.15

    # ── Volume ───────────────────────────────────────────────
    vol_ratio = _safe(cur.get("vol_ratio"), 1.0)
    prv_vol   = _safe(prv.get("volume"), 1)
    cur_vol   = _safe(cur.get("volume"), 1)
    vol_delta = _safe_div(cur_vol, prv_vol, 1.0)

    # Volume trend: SMA-5 / SMA-20 of volume
    vol_slice = df["volume"].iloc[max(0, abs_idx - 19):abs_idx + 1]
    vol_sma5  = float(vol_slice.iloc[-5:].mean()) if len(vol_slice) >= 5 else cur_vol
    vol_sma20 = float(vol_slice.mean()) if len(vol_slice) >= 10 else cur_vol
    vol_trend = _safe_div(vol_sma5, vol_sma20, 1.0)

    # ── Returns ──────────────────────────────────────────────
    def _ret(n: int) -> float:
        if abs_idx - n < 0:
            return 0.0
        past = _safe(df.iloc[abs_idx - n]["close"])
        return _safe_div(close - past, past)

    ret_1d  = _ret(1)
    ret_3d  = _ret(3)
    ret_5d  = _ret(5)
    ret_10d = _ret(10)
    ret_20d = _ret(20)

    # ── Trend ────────────────────────────────────────────────
    def _slope(col: str, period: int) -> float:
        if abs_idx - period < 0:
            return 0.0
        now  = _safe(df.iloc[abs_idx].get(col))
        ago  = _safe(df.iloc[abs_idx - period].get(col))
        return _safe_div(now - ago, ago)

    ema50_slope  = _slope("ema_trend", 5)
    ema200_slope = _slope("ema_200", 10)

    # Higher high / higher low (5-bar lookback)
    highs_5 = df["high"].iloc[max(0, abs_idx - 5):abs_idx].values
    lows_5  = df["low"].iloc[max(0, abs_idx - 5):abs_idx].values
    higher_high = 1.0 if len(highs_5) > 0 and close > float(np.max(highs_5)) else 0.0
    higher_low  = 1.0 if len(lows_5) > 0 and _safe(cur.get("low")) > float(np.min(lows_5)) else 0.0

    # ── Support / Resistance ─────────────────────────────────
    sr_sup = _safe(cur.get("sr_support"))
    sr_res = _safe(cur.get("sr_resistance"))
    dist_support    = _safe_div(close - sr_sup, close) if sr_sup > 0 else 0.5
    dist_resistance = _safe_div(sr_res - close, close) if sr_res > 0 else 0.5

    # ── Hand-crafted score ───────────────────────────────────
    # Use the sub-DataFrame up to this row for score_entry
    sub_df = df.iloc[max(0, abs_idx - 80):abs_idx + 1]
    try:
        entry_score, _ = score_entry(sub_df, weekly_bullish=weekly_bullish)
    except Exception:
        entry_score = 0

    # ── Calendar ─────────────────────────────────────────────
    dt_index = df.index[abs_idx]
    try:
        dow = float(dt_index.weekday())   # 0-4
    except Exception:
        dow = 2.0

    # ── Macro context (replaces raw month) ───────────────────
    spy_sma200_dist = 0.0
    vixy_relative   = 0.0

    try:
        bar_date = pd.Timestamp(dt_index).normalize()  # midnight
    except Exception:
        bar_date = None

    if bar_date is not None and spy_df is not None and not spy_df.empty:
        try:
            spy_idx = spy_df.index.searchsorted(bar_date, side="right") - 1
            if spy_idx >= 0:
                spy_row = spy_df.iloc[spy_idx]
                spy_close  = _safe(spy_row.get("close"))
                spy_sma200 = _safe(spy_row.get("sma_200"))
                if spy_sma200 > 0:
                    spy_sma200_dist = _safe_div(spy_close, spy_sma200) - 1.0
        except Exception:
            pass

    if bar_date is not None and vixy_df is not None and not vixy_df.empty:
        try:
            v_idx = vixy_df.index.searchsorted(bar_date, side="right") - 1
            if v_idx >= 0:
                vixy_close = _safe(vixy_df.iloc[v_idx].get("close"))
                # Normalise against 20-bar trailing mean
                v_start = max(0, v_idx - 19)
                vixy_slice = vixy_df["close"].iloc[v_start:v_idx + 1]
                vixy_sma20 = float(vixy_slice.mean()) if len(vixy_slice) >= 5 else vixy_close
                if vixy_sma20 > 0:
                    vixy_relative = _safe_div(vixy_close, vixy_sma20) - 1.0
        except Exception:
            pass

    # ── Assemble feature vector (must match FEATURE_NAMES order) ─
    return [
        dist_ema_fast, dist_ema_slow, dist_ema_trend, dist_ema_200,
        ema_fs_ratio, ema_st_ratio,
        rsi, rsi_delta, macd_hist, macd_hist_delta,
        adx, stoch_k, stoch_d, stoch_kd,
        rsi_slope_3d, macd_hist_slope_3d, adx_slope_3d,
        stoch_k_slope_3d, vol_ratio_slope_3d,
        atr_ratio, bb_width, bb_pctb, rvol_5, rvol_20,
        vol_ratio, vol_delta, vol_trend,
        ret_1d, ret_3d, ret_5d, ret_10d, ret_20d,
        ema50_slope, ema200_slope, higher_high, higher_low,
        dist_support, dist_resistance,
        float(entry_score),
        dow,
        spy_sma200_dist, vixy_relative,
    ]


# ─────────────────────────────────────────────────────────────
# Bulk extraction (for training)
# ─────────────────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    start_idx: int = 60,
    end_idx: int | None = None,
    spy_df: pd.DataFrame | None = None,
    vixy_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, list[int]]:
    """
    Extract features for every bar from *start_idx* to *end_idx*.
    Returns (X, valid_indices) where X is shape (N, NUM_FEATURES)
    and valid_indices maps each row back to the original df index position.
    """
    end_idx = end_idx or len(df)
    rows: list[list[float]] = []
    indices: list[int] = []

    for i in range(start_idx, end_idx):
        fv = extract_row(df, idx=i, spy_df=spy_df, vixy_df=vixy_df)
        if fv is not None:
            rows.append(fv)
            indices.append(i)

    if not rows:
        return np.empty((0, NUM_FEATURES)), []
    return np.array(rows, dtype=np.float32), indices


def generate_labels(
    df: pd.DataFrame,
    indices: list[int],
    forward_bars: int = 5,
    min_gain_pct: float = 0.03,
    **_kwargs,
) -> np.ndarray:
    """
    Generate binary labels for the training set.

    Simplified target: "Does the closing price reach ≥ *min_gain_pct*
    above the entry price within *forward_bars* bars?"

    This predicts immediate momentum — whether the setup sparks — and
    lets the bot's trailing stops / exit logic manage the rest of the
    trade.  Much easier for the model to learn than wide ATR brackets.

    Entry is assumed at the NEXT bar's open.
      - label = 1 if any close in [entry_bar .. entry_bar+forward_bars]
                  is ≥ entry_price × (1 + min_gain_pct)
      - label = 0 otherwise
    """
    labels = np.zeros(len(indices), dtype=np.int32)
    n = len(df)

    for i, bar_idx in enumerate(indices):
        entry_bar = bar_idx + 1  # enter on next bar's open
        if entry_bar >= n:
            continue

        entry_price = df.iloc[entry_bar]["open"]
        if pd.isna(entry_price) or entry_price <= 0:
            continue

        target = entry_price * (1.0 + min_gain_pct)

        # Walk forward — did any close hit our target?
        end = min(entry_bar + forward_bars, n)
        for j in range(entry_bar, end):
            if df.iloc[j]["close"] >= target:
                labels[i] = 1
                break

    return labels
