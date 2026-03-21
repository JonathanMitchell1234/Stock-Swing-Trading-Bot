"""
Swing Trading Strategy v4 – Advanced multi-factor scoring with smart exits.

Builds on v3 with six advanced features:
  1. Multi-timeframe confirmation — weekly trend must agree (bonus/penalty)
  2. Support/Resistance awareness — bonus near support, penalty near resistance
  3. Gap-up avoidance — skip entries after >3% gap-ups (exhaustion risk)
  4. Dynamic score threshold — adjusts by market quality
  5. Volatility regime awareness — strategy adapts to vol conditions
  6. Sector exposure limits — diversification enforcement

Entry scoring (max ~20 points):
  +2  Price > EMA-50  (uptrend)
  +1  Price > EMA-200  (bull market regime on the stock itself)
  +2  EMA-9 just crossed above EMA-21  (bullish crossover)
  +1  EMA-50 slope is rising (trend quality)
  +2  RSI in 30-50 zone  (pullback, not overbought)
  +1  MACD histogram positive or turning up
  +1  Volume >= average
  +1  ADX > 20  (trending)
  +1  Price near lower Bollinger Band (<= BB mid)
  +1  Stochastic %K crossed above %D from oversold
  +2  Top-quartile momentum (20-day return)
  +1  Weekly trend agrees (multi-TF confirmation)
  +1  Price near support level

  Threshold (configurable, dynamically adjusted): 5

Exit criteria  (layered - not hair-trigger):
  HARD exits (immediate):
    - Stop-loss / take-profit hit  (bracket order / ATR-based)
    - Price closes below EMA-200 AND below EMA-50  (trend destroyed)

  SOFT exits (need 2+ signals to trigger):
    - RSI >= 80 (truly overbought)
    - Bearish EMA-9/21 crossover
    - MACD histogram negative for 2+ bars (accelerating down)
    - Price below EMA-50 (but still above 200)
    - Dead money: held N days with < 2% total move
    - Momentum decay: -5% 20-day return + below EMA-50

  Trailing stop handles the rest - lets winners run.
"""

from __future__ import annotations

import pandas as pd

import config
from logger import get_logger

log = get_logger("strategy")

# Lazy import — ml_model is only loaded when ML is enabled
_ml_model = None

def _get_ml_model():
    """Lazy-load the ML model module to avoid import overhead when ML is disabled."""
    global _ml_model
    if _ml_model is None:
        import ml_model as _m
        _ml_model = _m
    return _ml_model


def _score_entry_details(df: pd.DataFrame, weekly_bullish: bool = True) -> dict:
    """Return full entry-score details, including any early block reasons."""
    details = {
        "score": 0,
        "factors": [],
        "block_reasons": [],
    }

    if len(df) < max(config.MOMENTUM_LOOKBACK + 1, config.EMA_SLOPE_PERIOD + 1, 3):
        details["block_reasons"].append(
            f"Not enough bars ({len(df)}) for scoring"
        )
        return details

    cur = df.iloc[-1]
    prv = df.iloc[-2]

    price = cur["close"]
    ema_fast = cur["ema_fast"]
    ema_slow = cur["ema_slow"]
    ema_trend = cur["ema_trend"]
    ema_200 = cur.get("ema_200", None)
    rsi = cur["rsi"]
    macd_hist = cur["macd_hist"]
    adx = cur["adx"]
    vol_ratio = cur["vol_ratio"]
    atr = cur["atr"]
    bb_mid = cur.get("bb_mid", None)
    stoch_k = cur.get("stoch_k", None)
    stoch_d = cur.get("stoch_d", None)

    if pd.isna(rsi) or pd.isna(atr) or pd.isna(adx):
        details["block_reasons"].append("Missing key indicators (RSI/ATR/ADX)")
        return details

    # Gap-up filter: skip entries after exhaustion gaps.
    prev_close = prv["close"]
    today_open = cur["open"]
    if prev_close > 0 and today_open > 0:
        gap_pct = (today_open - prev_close) / prev_close
        if gap_pct > config.GAP_UP_MAX_PCT:
            reason = f"Gap-up {gap_pct*100:.1f}% exceeds {config.GAP_UP_MAX_PCT*100:.1f}%"
            details["factors"].append(reason)
            details["block_reasons"].append(reason)
            return details

    score = 0
    factors: list[str] = []

    # +2: Price above 50-EMA (core uptrend)
    if price > ema_trend:
        score += 2
        factors.append("Above EMA-50")

    # +1: Price above 200-EMA (bull regime)
    if ema_200 is not None and not pd.isna(ema_200) and price > ema_200:
        score += 1
        factors.append("Above EMA-200")

    # +2: Bullish EMA crossover (9 crosses above 21)
    if (prv["ema_fast"] <= prv["ema_slow"]) and (ema_fast > ema_slow):
        score += 2
        factors.append("EMA crossover")

    # +1: Trend quality — EMA-50 slope is rising
    if len(df) >= config.EMA_SLOPE_PERIOD + 1:
        ema50_now = cur["ema_trend"]
        ema50_ago = df.iloc[-(config.EMA_SLOPE_PERIOD + 1)]["ema_trend"]
        if not pd.isna(ema50_now) and not pd.isna(ema50_ago) and ema50_now > ema50_ago:
            score += 1
            factors.append("EMA-50 rising")

    # +2: RSI in pullback zone (30-50)
    if config.RSI_OVERSOLD <= rsi <= 50:
        score += 2
        factors.append(f"RSI pullback ({rsi:.0f})")
    elif 50 < rsi <= 60:
        score += 1
        factors.append(f"RSI mid-range ({rsi:.0f})")

    # +1: MACD histogram positive or turning up
    macd_ok = macd_hist > 0 or (
        prv["macd_hist"] < 0 and macd_hist > prv["macd_hist"]
    )
    if macd_ok:
        score += 1
        factors.append("MACD positive/turning")

    # +1: Volume above average
    if vol_ratio >= config.VOLUME_SURGE_FACTOR:
        score += 1
        factors.append(f"Volume {vol_ratio:.1f}x")

    # +1: ADX showing trend
    if adx > 20:
        score += 1
        factors.append(f"ADX {adx:.0f}")

    # +1: Price near lower Bollinger Band
    if bb_mid is not None and not pd.isna(bb_mid) and price <= bb_mid:
        score += 1
        factors.append("Near BB lower")

    # +1: Stochastic bullish crossover from oversold
    if (stoch_k is not None and stoch_d is not None
            and not pd.isna(stoch_k) and not pd.isna(stoch_d)):
        prv_stoch_k = prv.get("stoch_k", None)
        prv_stoch_d = prv.get("stoch_d", None)
        if (prv_stoch_k is not None and prv_stoch_d is not None
                and not pd.isna(prv_stoch_k) and not pd.isna(prv_stoch_d)):
            if prv_stoch_k <= prv_stoch_d and stoch_k > stoch_d and stoch_k < 50:
                score += 1
                factors.append("Stoch bullish cross")

    # +1: Weekly trend agrees (multi-timeframe)
    if config.WEEKLY_TREND_ENABLED and weekly_bullish:
        score += config.WEEKLY_TREND_BONUS
        factors.append("Weekly trend OK")

    # +1: Price near support / -penalty near resistance (only below EMA-50)
    sr_support = cur.get("sr_support", None)
    sr_resistance = cur.get("sr_resistance", None)
    if sr_support is not None and not pd.isna(sr_support):
        dist_to_support = (price - sr_support) / price if price > 0 else 1.0
        if dist_to_support <= 0.03:
            score += config.SR_SUPPORT_BONUS
            factors.append("Near support")
    if sr_resistance is not None and not pd.isna(sr_resistance):
        dist_to_resistance = (sr_resistance - price) / price if price > 0 else 1.0
        if dist_to_resistance <= config.SR_RESISTANCE_BUFFER and price < ema_trend:
            score -= 1
            factors.append("Near resistance (-1)")

    details["score"] = score
    details["factors"] = factors
    return details


# ─────────────────────────────────────────────
# ENTRY  (scoring system)
# ─────────────────────────────────────────────
def score_entry(df: pd.DataFrame, weekly_bullish: bool = True) -> tuple[int, list[str]]:
    """
    Score the latest bar for entry quality.
    Returns (score, [list of contributing factors]).
    weekly_bullish: whether the weekly trend is aligned (from caller).
    """
    details = _score_entry_details(df, weekly_bullish=weekly_bullish)
    return details["score"], details["factors"]


def compute_momentum(df: pd.DataFrame) -> float:
    """
    Compute the momentum (rate of change) over the lookback period.
    Returns the % change as a decimal (e.g. 0.05 = +5%).
    """
    lookback = config.MOMENTUM_LOOKBACK
    if len(df) < lookback + 1:
        return 0.0

    cur_close = df.iloc[-1]["close"]
    past_close = df.iloc[-(lookback + 1)]["close"]

    if past_close <= 0 or pd.isna(past_close) or pd.isna(cur_close):
        return 0.0

    return (cur_close - past_close) / past_close


def check_entry(df: pd.DataFrame, weekly_bullish: bool = True,
                spy_df: pd.DataFrame | None = None,
                vixy_df: pd.DataFrame | None = None,
                explain: bool = False) -> dict | None:
    """
    Evaluate the latest bar using the scoring system + optional ML model.

    When ML_ENABLED is True:
      - "gate" mode: hand-crafted score must pass ML_MIN_SCORE AND
        GBM probability must be >= ML_ENTRY_THRESHOLD
      - "replace" mode: only GBM probability matters (score is still
        computed and logged but not used as a gate)

    When ML_ENABLED is False:
      - Falls back to the original threshold-only behaviour.

    Return a signal dict if score meets threshold, else None.
    """
    details = _score_entry_details(df, weekly_bullish=weekly_bullish)
    score = details["score"]
    factors = list(details["factors"])
    block_reasons = list(details["block_reasons"])

    diagnostics = {
        "eligible": False,
        "score": score,
        "threshold": config.ENTRY_SCORE_THRESHOLD,
        "factors": factors,
        "block_reasons": block_reasons,
        "ml_prob": None,
        "price": None,
        "atr": None,
        "reason": "",
        "signal": None,
    }

    if block_reasons:
        diagnostics["reason"] = "; ".join(block_reasons)
        return diagnostics if explain else None

    # ── ML-enhanced path ─────────────────────────────────────
    if config.ML_ENABLED:
        ml = _get_ml_model()
        ml_prob = None

        # In "gate" mode, require minimum hand-crafted score first
        if config.ML_BLEND_MODE == "gate" and score < config.ML_MIN_SCORE:
            block_reasons.append(
                f"Score {score} below ML gate minimum {config.ML_MIN_SCORE}"
            )
            diagnostics["block_reasons"] = block_reasons
            diagnostics["reason"] = block_reasons[-1]
            return diagnostics if explain else None

        # Get ML prediction
        if ml.is_available():
            ml_prob = ml.predict_entry_proba(
                df, idx=-1, weekly_bullish=weekly_bullish,
                spy_df=spy_df, vixy_df=vixy_df,
            )

        if ml_prob is not None:
            if ml_prob < config.ML_ENTRY_THRESHOLD:
                reason = (
                    f"ML prob {ml_prob:.3f} below threshold {config.ML_ENTRY_THRESHOLD:.3f}"
                )
                block_reasons.append(reason)
                diagnostics["block_reasons"] = block_reasons
                diagnostics["ml_prob"] = ml_prob
                diagnostics["reason"] = reason
                return diagnostics if explain else None
            factors.append(f"ML prob={ml_prob:.2f}")
            diagnostics["ml_prob"] = ml_prob
        else:
            # Model unavailable — fall back to hand-crafted threshold
            if score < config.ENTRY_SCORE_THRESHOLD:
                reason = (
                    f"Score {score} below threshold {config.ENTRY_SCORE_THRESHOLD} (ML unavailable fallback)"
                )
                block_reasons.append(reason)
                diagnostics["block_reasons"] = block_reasons
                diagnostics["reason"] = reason
                return diagnostics if explain else None
    else:
        # ── Original path (no ML) ───────────────────────────
        if score < config.ENTRY_SCORE_THRESHOLD:
            reason = f"Score {score} below threshold {config.ENTRY_SCORE_THRESHOLD}"
            block_reasons.append(reason)
            diagnostics["block_reasons"] = block_reasons
            diagnostics["reason"] = reason
            return diagnostics if explain else None

    cur = df.iloc[-1]
    price = cur["close"]
    atr = cur["atr"]

    reason = f"Score {score}: {', '.join(factors)}"

    signal = {
        "action": "BUY",
        "price": price,
        "atr": atr,
        "rsi": cur["rsi"],
        "macd_hist": cur["macd_hist"],
        "adx": cur["adx"],
        "vol_ratio": cur["vol_ratio"],
        "score": score,
        "reason": reason,
    }
    if config.ML_ENABLED and ml_prob is not None:
        signal["ml_prob"] = ml_prob
    diagnostics.update(
        {
            "eligible": True,
            "factors": factors,
            "price": price,
            "atr": atr,
            "reason": reason,
            "signal": signal,
        }
    )
    if explain:
        return diagnostics
    log.info("ENTRY signal: price=%.2f  %s", price, reason)
    return signal


# ─────────────────────────────────────────────
# EXIT  (layered – hard + soft)
# ─────────────────────────────────────────────
def check_exit(df: pd.DataFrame, entry_price: float = 0.0,
               hold_days: int = 0,
               explain: bool = False) -> dict | None:
    """
    Evaluate whether an existing position should be closed.

    HARD exits fire immediately (1 is enough).
    SOFT exits require 2+ simultaneous signals to avoid
    getting shaken out by normal volatility.
    """
    diagnostics = {
        "should_exit": False,
        "signal": None,
        "hard_reasons": [],
        "soft_reasons": [],
        "reasons": [],
        "hold_reason": "",
        "price": None,
        "rsi": None,
        "macd_hist": None,
        "entry_price": entry_price,
        "hold_days": hold_days,
    }

    if len(df) < 4:
        diagnostics["hold_reason"] = f"Not enough bars ({len(df)}/4)"
        return diagnostics if explain else None

    cur = df.iloc[-1]
    prv = df.iloc[-2]

    price = cur["close"]
    ema_fast = cur["ema_fast"]
    ema_slow = cur["ema_slow"]
    ema_trend = cur["ema_trend"]
    ema_200 = cur.get("ema_200", None)
    rsi = cur["rsi"]
    macd_hist = cur["macd_hist"]

    if pd.isna(rsi):
        diagnostics["hold_reason"] = "RSI unavailable"
        return diagnostics if explain else None

    hard_reasons = []
    soft_reasons = []

    # ── HARD: price below BOTH EMA-50 and EMA-200 (trend destroyed) ──
    if ema_200 is not None and not pd.isna(ema_200):
        if price < ema_trend and price < ema_200:
            hard_reasons.append("Below EMA-50 & EMA-200")

    # ── HARD: stale loser — held past dead-money window, price below
    #    entry AND below EMA-50. No second signal needed: capital is
    #    both stagnant/declining and technically broken. ────────────
    if config.STALE_LOSER_EXIT_ENABLED and entry_price > 0:
        if hold_days >= config.DEAD_MONEY_DAYS and price < entry_price and price < ema_trend:
            hard_reasons.append(
                f"Stale loser ({hold_days}d, {(price - entry_price) / entry_price * 100:.1f}%,"
                f" below EMA-{config.EMA_TREND})"
            )

    # ── SOFT: RSI extremely overbought ───────────────────────
    if rsi >= config.RSI_OVERBOUGHT:
        soft_reasons.append(f"RSI overbought ({rsi:.1f})")

    # ── SOFT: Bearish EMA crossover ──────────────────────────
    if (prv["ema_fast"] >= prv["ema_slow"]) and (ema_fast < ema_slow):
        soft_reasons.append("Bearish EMA crossover")

    # ── SOFT: MACD histogram declining for 2+ bars ───────────
    if (prv["macd_hist"] < 0 and macd_hist < 0
            and macd_hist < prv["macd_hist"]):
        soft_reasons.append("MACD declining 2+ bars")

    # ── SOFT: price below EMA-50 (but still above 200) ──────
    if price < ema_trend:
        if ema_200 is None or pd.isna(ema_200) or price >= ema_200:
            soft_reasons.append(f"Price below EMA-{config.EMA_TREND}")

    # ── SOFT: Dead money — position is flat after N days ─────
    if (entry_price > 0 and hold_days >= config.DEAD_MONEY_DAYS):
        move_pct = abs(price - entry_price) / entry_price
        if move_pct < config.DEAD_MONEY_THRESHOLD:
            soft_reasons.append(
                f"Dead money ({hold_days}d, {move_pct*100:.1f}% move)"
            )

    # ── SOFT: Momentum decay — 20-day return turned negative ─
    if len(df) >= config.MOMENTUM_LOOKBACK + 1:
        past_price = df.iloc[-(config.MOMENTUM_LOOKBACK + 1)]["close"]
        if not pd.isna(past_price) and past_price > 0:
            mom = (price - past_price) / past_price
            if mom < -0.05 and price < ema_trend:
                soft_reasons.append(f"Momentum decay ({mom*100:.1f}%)")

    # ── Decision ─────────────────────────────────────────────
    # Hard exits fire on 1 signal.
    # Soft exits normally require 2+ signals to avoid shakeouts.
    # Exception: once a position is past DEAD_MONEY_DAYS, lower the
    # gate to DEAD_MONEY_SOFT_GATES (default 1) so stale positions
    # don't need a full second signal to exit.
    soft_gate = (
        config.DEAD_MONEY_SOFT_GATES
        if hold_days >= config.DEAD_MONEY_DAYS
        else 2
    )

    reasons = []
    if hard_reasons:
        reasons = hard_reasons
    elif len(soft_reasons) >= soft_gate:
        reasons = soft_reasons

    diagnostics["price"] = price
    diagnostics["rsi"] = rsi
    diagnostics["macd_hist"] = macd_hist
    diagnostics["hard_reasons"] = hard_reasons
    diagnostics["soft_reasons"] = soft_reasons

    if not reasons:
        if soft_reasons:
            diagnostics["hold_reason"] = (
                f"Only {len(soft_reasons)}/2 soft exit signals"
            )
        else:
            diagnostics["hold_reason"] = "No hard or soft exit triggers"
        return diagnostics if explain else None

    signal = {
        "action": "SELL",
        "price": price,
        "rsi": rsi,
        "macd_hist": macd_hist,
        "reasons": reasons,
    }
    diagnostics.update(
        {
            "should_exit": True,
            "signal": signal,
            "reasons": reasons,
            "hold_reason": "",
        }
    )
    if explain:
        return diagnostics
    log.info("EXIT  signal: price=%.2f  reasons=%s", price, ", ".join(reasons))
    return signal


# ─────────────────────────────────────────────
# BEAR MODE – SHORT ENTRY (inverted scoring)
# ─────────────────────────────────────────────
def _score_short_entry_details(df: pd.DataFrame, weekly_bullish: bool = True) -> dict:
    """
    Mirror of _score_entry_details but for SHORT entries.
    Scores weakness signals that indicate a stock is ready to drop.
    """
    details = {
        "score": 0,
        "factors": [],
        "block_reasons": [],
    }

    if len(df) < max(config.MOMENTUM_LOOKBACK + 1, config.EMA_SLOPE_PERIOD + 1, 3):
        details["block_reasons"].append(
            f"Not enough bars ({len(df)}) for scoring"
        )
        return details

    cur = df.iloc[-1]
    prv = df.iloc[-2]

    price = cur["close"]
    ema_fast = cur["ema_fast"]
    ema_slow = cur["ema_slow"]
    ema_trend = cur["ema_trend"]
    ema_200 = cur.get("ema_200", None)
    rsi = cur["rsi"]
    macd_hist = cur["macd_hist"]
    adx = cur["adx"]
    vol_ratio = cur["vol_ratio"]
    atr = cur["atr"]
    bb_mid = cur.get("bb_mid", None)
    stoch_k = cur.get("stoch_k", None)
    stoch_d = cur.get("stoch_d", None)

    if pd.isna(rsi) or pd.isna(atr) or pd.isna(adx):
        details["block_reasons"].append("Missing key indicators (RSI/ATR/ADX)")
        return details

    # Gap-down filter: skip shorts after exhaustion gaps down
    prev_close = prv["close"]
    today_open = cur["open"]
    if prev_close > 0 and today_open > 0:
        gap_pct = (prev_close - today_open) / prev_close
        if gap_pct > config.GAP_DOWN_MAX_PCT:
            reason = f"Gap-down {gap_pct*100:.1f}% exceeds {config.GAP_DOWN_MAX_PCT*100:.1f}%"
            details["factors"].append(reason)
            details["block_reasons"].append(reason)
            return details

    score = 0
    factors: list[str] = []

    # +2: Price BELOW EMA-50 (downtrend — mirror of "Above EMA-50")
    if price < ema_trend:
        score += 2
        factors.append("Below EMA-50")

    # +1: Price BELOW EMA-200 (bear regime on the stock itself)
    if ema_200 is not None and not pd.isna(ema_200) and price < ema_200:
        score += 1
        factors.append("Below EMA-200")

    # +2: Bearish EMA crossover (EMA-9 crosses BELOW EMA-21)
    if (prv["ema_fast"] >= prv["ema_slow"]) and (ema_fast < ema_slow):
        score += 2
        factors.append("Bearish EMA crossover")

    # +1: Trend quality — EMA-50 slope is FALLING
    if len(df) >= config.EMA_SLOPE_PERIOD + 1:
        ema50_now = cur["ema_trend"]
        ema50_ago = df.iloc[-(config.EMA_SLOPE_PERIOD + 1)]["ema_trend"]
        if not pd.isna(ema50_now) and not pd.isna(ema50_ago) and ema50_now < ema50_ago:
            score += 1
            factors.append("EMA-50 falling")

    # +2: RSI in overbought zone (65-85 — stretched, ready to fall)
    if config.RSI_SHORT_ENTRY_MIN <= rsi <= config.RSI_SHORT_ENTRY_MAX:
        score += 2
        factors.append(f"RSI overbought ({rsi:.0f})")
    elif 55 <= rsi < config.RSI_SHORT_ENTRY_MIN:
        score += 1
        factors.append(f"RSI elevated ({rsi:.0f})")

    # +1: MACD histogram negative or turning down
    macd_ok = macd_hist < 0 or (
        prv["macd_hist"] > 0 and macd_hist < prv["macd_hist"]
    )
    if macd_ok:
        score += 1
        factors.append("MACD negative/turning down")

    # +1: Volume above average (confirms the move)
    if vol_ratio >= config.VOLUME_SURGE_FACTOR:
        score += 1
        factors.append(f"Volume {vol_ratio:.1f}x")

    # +1: ADX showing trend (strong trend in either direction)
    if adx > 20:
        score += 1
        factors.append(f"ADX {adx:.0f}")

    # +1: Price near UPPER Bollinger Band (overextended to upside)
    if bb_mid is not None and not pd.isna(bb_mid) and price >= bb_mid:
        score += 1
        factors.append("Near BB upper")

    # +1: Stochastic bearish crossover from overbought
    if (stoch_k is not None and stoch_d is not None
            and not pd.isna(stoch_k) and not pd.isna(stoch_d)):
        prv_stoch_k = prv.get("stoch_k", None)
        prv_stoch_d = prv.get("stoch_d", None)
        if (prv_stoch_k is not None and prv_stoch_d is not None
                and not pd.isna(prv_stoch_k) and not pd.isna(prv_stoch_d)):
            if prv_stoch_k >= prv_stoch_d and stoch_k < stoch_d and stoch_k > 50:
                score += 1
                factors.append("Stoch bearish cross")

    # +1: Weekly trend is bearish (multi-timeframe confirmation)
    if config.WEEKLY_TREND_ENABLED and not weekly_bullish:
        score += config.WEEKLY_TREND_BONUS
        factors.append("Weekly trend bearish")

    # +1: Price near resistance (about to reject) / -1 near support
    sr_support = cur.get("sr_support", None)
    sr_resistance = cur.get("sr_resistance", None)
    if sr_resistance is not None and not pd.isna(sr_resistance):
        dist_to_resistance = (sr_resistance - price) / price if price > 0 else 1.0
        if dist_to_resistance <= 0.03:
            score += 1
            factors.append("Near resistance")
    if sr_support is not None and not pd.isna(sr_support):
        dist_to_support = (price - sr_support) / price if price > 0 else 1.0
        if dist_to_support <= config.SR_RESISTANCE_BUFFER and price > ema_trend:
            score -= 1
            factors.append("Near support (-1)")

    details["score"] = score
    details["factors"] = factors
    return details


# Lazy import — short ML model is only loaded when bear-mode + ML is enabled
_ml_model_short = None

def _get_ml_model_short():
    """Lazy-load the SHORT ML model module."""
    global _ml_model_short
    if _ml_model_short is None:
        import ml_model_short as _m
        _ml_model_short = _m
    return _ml_model_short


def check_short_entry(df: pd.DataFrame, weekly_bullish: bool = True,
                      spy_df: pd.DataFrame | None = None,
                      vixy_df: pd.DataFrame | None = None,
                      explain: bool = False) -> dict | None:
    """
    Evaluate the latest bar for SHORT entry quality.
    Mirror of check_entry() but scores bearish weakness.

    When ML_SHORT_ENABLED is True and a trained short GBM model exists:
      - "gate" mode: hand-crafted short score must pass ML_SHORT_MIN_SCORE AND
        short GBM probability must be >= ML_SHORT_ENTRY_THRESHOLD
      - Falls back to hand-crafted threshold when short model is unavailable.

    Returns a signal dict if score meets threshold, else None.
    """
    details = _score_short_entry_details(df, weekly_bullish=weekly_bullish)
    score = details["score"]
    factors = list(details["factors"])
    block_reasons = list(details["block_reasons"])

    diagnostics = {
        "eligible": False,
        "score": score,
        "threshold": config.ENTRY_SCORE_THRESHOLD,
        "factors": factors,
        "block_reasons": block_reasons,
        "ml_prob": None,
        "price": None,
        "atr": None,
        "reason": "",
        "signal": None,
    }

    if block_reasons:
        diagnostics["reason"] = "; ".join(block_reasons)
        return diagnostics if explain else None

    # ── Short ML-enhanced path ───────────────────────────────
    ml_short_threshold = getattr(config, "ML_SHORT_ENTRY_THRESHOLD", config.ML_ENTRY_THRESHOLD)
    ml_short_min_score = getattr(config, "ML_SHORT_MIN_SCORE", config.ML_MIN_SCORE)
    ml_prob = None

    if getattr(config, "ML_SHORT_ENABLED", False):
        ml_short = _get_ml_model_short()

        # Gate mode: require minimum hand-crafted score first
        if config.ML_BLEND_MODE == "gate" and score < ml_short_min_score:
            block_reasons.append(
                f"Short score {score} below ML gate minimum {ml_short_min_score}"
            )
            diagnostics["block_reasons"] = block_reasons
            diagnostics["reason"] = block_reasons[-1]
            return diagnostics if explain else None

        # Get short ML prediction
        if ml_short.is_available():
            ml_prob = ml_short.predict_short_proba(
                df, idx=-1, weekly_bullish=weekly_bullish,
                spy_df=spy_df, vixy_df=vixy_df,
            )

        if ml_prob is not None:
            if ml_prob < ml_short_threshold:
                reason = (
                    f"Short ML prob {ml_prob:.3f} below threshold {ml_short_threshold:.3f}"
                )
                block_reasons.append(reason)
                diagnostics["block_reasons"] = block_reasons
                diagnostics["ml_prob"] = ml_prob
                diagnostics["reason"] = reason
                return diagnostics if explain else None
            factors.append(f"Short ML prob={ml_prob:.2f}")
            diagnostics["ml_prob"] = ml_prob
        else:
            # Short model unavailable — fall back to hand-crafted threshold
            if score < config.ENTRY_SCORE_THRESHOLD:
                reason = (
                    f"Short score {score} below threshold {config.ENTRY_SCORE_THRESHOLD} "
                    f"(short ML unavailable fallback)"
                )
                block_reasons.append(reason)
                diagnostics["block_reasons"] = block_reasons
                diagnostics["reason"] = reason
                return diagnostics if explain else None
    else:
        # ── No short ML — pure hand-crafted scoring ─────────
        if score < config.ENTRY_SCORE_THRESHOLD:
            reason = f"Short score {score} below threshold {config.ENTRY_SCORE_THRESHOLD}"
            block_reasons.append(reason)
            diagnostics["block_reasons"] = block_reasons
            diagnostics["reason"] = reason
            return diagnostics if explain else None

    cur = df.iloc[-1]
    price = cur["close"]
    atr = cur["atr"]

    reason = f"SHORT Score {score}: {', '.join(factors)}"

    signal = {
        "action": "SHORT",
        "price": price,
        "atr": atr,
        "rsi": cur["rsi"],
        "macd_hist": cur["macd_hist"],
        "adx": cur["adx"],
        "vol_ratio": cur["vol_ratio"],
        "score": score,
        "reason": reason,
    }
    if ml_prob is not None:
        signal["ml_prob"] = ml_prob

    diagnostics.update(
        {
            "eligible": True,
            "factors": factors,
            "price": price,
            "atr": atr,
            "reason": reason,
            "signal": signal,
        }
    )
    if explain:
        return diagnostics
    log.info("SHORT ENTRY signal: price=%.2f  %s", price, reason)
    return signal


# ─────────────────────────────────────────────
# INVERSE ETF ENTRY (bear mode, equity < SHORT_MIN_EQUITY)
# ─────────────────────────────────────────────

# Lazy import — inverse ML model is only loaded when inverse ETF mode + ML is enabled
_ml_model_inverse = None

def _get_ml_model_inverse():
    """Lazy-load the INVERSE ETF ML model module."""
    global _ml_model_inverse
    if _ml_model_inverse is None:
        import ml_model_inverse as _m
        _ml_model_inverse = _m
    return _ml_model_inverse


def check_inverse_entry(df: pd.DataFrame, weekly_bullish: bool = True,
                        spy_df: pd.DataFrame | None = None,
                        vixy_df: pd.DataFrame | None = None,
                        explain: bool = False) -> dict | None:
    """
    Evaluate the latest bar for INVERSE ETF entry quality.

    Uses the standard long entry scoring (score_entry) because inverse ETFs
    are bought long — their bullish technicals are the correct signal.

    When ML_INVERSE_ENABLED is True and a trained inverse GBM model exists:
      - "gate" mode: hand-crafted score must pass ML_INVERSE_MIN_SCORE AND
        inverse GBM probability must be >= ML_INVERSE_ENTRY_THRESHOLD
      - Falls back to hand-crafted threshold when inverse model is unavailable.

    Returns a signal dict if score meets threshold, else None.
    """
    details = _score_entry_details(df, weekly_bullish=weekly_bullish)
    score = details["score"]
    factors = list(details["factors"])
    block_reasons = list(details["block_reasons"])

    diagnostics = {
        "eligible": False,
        "score": score,
        "threshold": config.ENTRY_SCORE_THRESHOLD,
        "factors": factors,
        "block_reasons": block_reasons,
        "ml_prob": None,
        "price": None,
        "atr": None,
        "reason": "",
        "signal": None,
    }

    if block_reasons:
        diagnostics["reason"] = "; ".join(block_reasons)
        return diagnostics if explain else None

    # ── Inverse ML-enhanced path ─────────────────────────────
    ml_inv_threshold = getattr(config, "ML_INVERSE_ENTRY_THRESHOLD", config.ML_ENTRY_THRESHOLD)
    ml_inv_min_score = getattr(config, "ML_INVERSE_MIN_SCORE", config.ML_MIN_SCORE)
    ml_prob = None

    if getattr(config, "ML_INVERSE_ENABLED", False):
        ml_inv = _get_ml_model_inverse()

        # Gate mode: require minimum hand-crafted score first
        if config.ML_BLEND_MODE == "gate" and score < ml_inv_min_score:
            block_reasons.append(
                f"Inverse score {score} below ML gate minimum {ml_inv_min_score}"
            )
            diagnostics["block_reasons"] = block_reasons
            diagnostics["reason"] = block_reasons[-1]
            return diagnostics if explain else None

        # Get inverse ML prediction
        if ml_inv.is_available():
            ml_prob = ml_inv.predict_inverse_proba(
                df, idx=-1, weekly_bullish=weekly_bullish,
                spy_df=spy_df, vixy_df=vixy_df,
            )

        if ml_prob is not None:
            if ml_prob < ml_inv_threshold:
                reason = (
                    f"Inverse ML prob {ml_prob:.3f} below threshold {ml_inv_threshold:.3f}"
                )
                block_reasons.append(reason)
                diagnostics["block_reasons"] = block_reasons
                diagnostics["ml_prob"] = ml_prob
                diagnostics["reason"] = reason
                return diagnostics if explain else None
            factors.append(f"Inverse ML prob={ml_prob:.2f}")
            diagnostics["ml_prob"] = ml_prob
        else:
            # Inverse model unavailable — fall back to hand-crafted threshold
            if score < config.ENTRY_SCORE_THRESHOLD:
                reason = (
                    f"Inverse score {score} below threshold {config.ENTRY_SCORE_THRESHOLD} "
                    f"(inverse ML unavailable fallback)"
                )
                block_reasons.append(reason)
                diagnostics["block_reasons"] = block_reasons
                diagnostics["reason"] = reason
                return diagnostics if explain else None
    else:
        # ── No inverse ML — pure hand-crafted scoring ───────
        if score < config.ENTRY_SCORE_THRESHOLD:
            reason = f"Inverse score {score} below threshold {config.ENTRY_SCORE_THRESHOLD}"
            block_reasons.append(reason)
            diagnostics["block_reasons"] = block_reasons
            diagnostics["reason"] = reason
            return diagnostics if explain else None

    cur = df.iloc[-1]
    price = cur["close"]
    atr = cur["atr"]

    reason = f"INVERSE Score {score}: {', '.join(factors)}"

    signal = {
        "action": "BUY",  # buying inverse ETFs long
        "price": price,
        "atr": atr,
        "rsi": cur["rsi"],
        "macd_hist": cur["macd_hist"],
        "adx": cur["adx"],
        "vol_ratio": cur["vol_ratio"],
        "score": score,
        "reason": reason,
    }
    if ml_prob is not None:
        signal["ml_prob"] = ml_prob

    diagnostics.update(
        {
            "eligible": True,
            "factors": factors,
            "price": price,
            "atr": atr,
            "reason": reason,
            "signal": signal,
        }
    )
    if explain:
        return diagnostics
    log.info("INVERSE ETF ENTRY signal: price=%.2f  %s", price, reason)
    return signal


# ─────────────────────────────────────────────
# BEAR MODE – SHORT EXIT (cover) logic
# ─────────────────────────────────────────────
def check_short_exit(df: pd.DataFrame, entry_price: float = 0.0,
                     hold_days: int = 0,
                     explain: bool = False) -> dict | None:
    """
    Evaluate whether an existing SHORT position should be covered (bought back).
    Mirror of check_exit() but inverted for short positions.

    HARD exits (immediate cover):
      - Price above BOTH EMA-50 and EMA-200 (uptrend re-established)
      - Stale loser: held too long, price ABOVE entry AND above EMA-50

    SOFT exits (need 2+ to trigger):
      - RSI extremely oversold (bounce incoming)
      - Bullish EMA crossover (9 crosses above 21)
      - MACD histogram rising for 2+ bars
      - Price above EMA-50 (but still below 200)
      - Dead money (flat position)
      - Momentum reversal: +5% 20-day return + above EMA-50
    """
    diagnostics = {
        "should_exit": False,
        "signal": None,
        "hard_reasons": [],
        "soft_reasons": [],
        "reasons": [],
        "hold_reason": "",
        "price": None,
        "rsi": None,
        "macd_hist": None,
        "entry_price": entry_price,
        "hold_days": hold_days,
    }

    if len(df) < 4:
        diagnostics["hold_reason"] = f"Not enough bars ({len(df)}/4)"
        return diagnostics if explain else None

    cur = df.iloc[-1]
    prv = df.iloc[-2]

    price = cur["close"]
    ema_fast = cur["ema_fast"]
    ema_slow = cur["ema_slow"]
    ema_trend = cur["ema_trend"]
    ema_200 = cur.get("ema_200", None)
    rsi = cur["rsi"]
    macd_hist = cur["macd_hist"]

    if pd.isna(rsi):
        diagnostics["hold_reason"] = "RSI unavailable"
        return diagnostics if explain else None

    hard_reasons = []
    soft_reasons = []

    # ── HARD: price ABOVE both EMA-50 and EMA-200 (uptrend re-established) ──
    if ema_200 is not None and not pd.isna(ema_200):
        if price > ema_trend and price > ema_200:
            hard_reasons.append("Above EMA-50 & EMA-200")

    # ── HARD: stale loser — short held past dead-money window, price ABOVE
    #    entry AND above EMA-50 (trade going against us) ──
    if config.STALE_LOSER_EXIT_ENABLED and entry_price > 0:
        if hold_days >= config.DEAD_MONEY_DAYS and price > entry_price and price > ema_trend:
            hard_reasons.append(
                f"Stale short loser ({hold_days}d, {(price - entry_price) / entry_price * 100:+.1f}%,"
                f" above EMA-{config.EMA_TREND})"
            )

    # ── SOFT: RSI extremely oversold (bounce risk) ──
    if rsi <= config.RSI_SHORT_EXIT:
        soft_reasons.append(f"RSI oversold ({rsi:.1f})")

    # ── SOFT: Bullish EMA crossover (9 crosses above 21) ──
    if (prv["ema_fast"] <= prv["ema_slow"]) and (ema_fast > ema_slow):
        soft_reasons.append("Bullish EMA crossover")

    # ── SOFT: MACD histogram rising for 2+ bars ──
    if (prv["macd_hist"] > 0 and macd_hist > 0
            and macd_hist > prv["macd_hist"]):
        soft_reasons.append("MACD rising 2+ bars")

    # ── SOFT: price above EMA-50 (but still below 200) ──
    if price > ema_trend:
        if ema_200 is None or pd.isna(ema_200) or price <= ema_200:
            soft_reasons.append(f"Price above EMA-{config.EMA_TREND}")

    # ── SOFT: Dead money — short position is flat after N days ──
    if (entry_price > 0 and hold_days >= config.DEAD_MONEY_DAYS):
        move_pct = abs(price - entry_price) / entry_price
        if move_pct < config.DEAD_MONEY_THRESHOLD:
            soft_reasons.append(
                f"Dead money ({hold_days}d, {move_pct*100:.1f}% move)"
            )

    # ── SOFT: Momentum reversal — 20-day return turned positive ──
    if len(df) >= config.MOMENTUM_LOOKBACK + 1:
        past_price = df.iloc[-(config.MOMENTUM_LOOKBACK + 1)]["close"]
        if not pd.isna(past_price) and past_price > 0:
            mom = (price - past_price) / past_price
            if mom > 0.05 and price > ema_trend:
                soft_reasons.append(f"Momentum reversal ({mom*100:+.1f}%)")

    # ── Decision (same gating as long exits) ──
    soft_gate = (
        config.DEAD_MONEY_SOFT_GATES
        if hold_days >= config.DEAD_MONEY_DAYS
        else 2
    )

    reasons = []
    if hard_reasons:
        reasons = hard_reasons
    elif len(soft_reasons) >= soft_gate:
        reasons = soft_reasons

    diagnostics["price"] = price
    diagnostics["rsi"] = rsi
    diagnostics["macd_hist"] = macd_hist
    diagnostics["hard_reasons"] = hard_reasons
    diagnostics["soft_reasons"] = soft_reasons

    if not reasons:
        if soft_reasons:
            diagnostics["hold_reason"] = (
                f"Only {len(soft_reasons)}/2 soft exit signals"
            )
        else:
            diagnostics["hold_reason"] = "No hard or soft exit triggers"
        return diagnostics if explain else None

    signal = {
        "action": "COVER",
        "price": price,
        "rsi": rsi,
        "macd_hist": macd_hist,
        "reasons": reasons,
    }
    diagnostics.update(
        {
            "should_exit": True,
            "signal": signal,
            "reasons": reasons,
            "hold_reason": "",
        }
    )
    if explain:
        return diagnostics
    log.info("SHORT EXIT signal: price=%.2f  reasons=%s", price, ", ".join(reasons))
    return signal
