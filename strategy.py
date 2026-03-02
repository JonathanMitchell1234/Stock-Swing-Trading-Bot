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


def classify_market_regime(
    regime_df: pd.DataFrame,
    last_regime: str = "bull",
) -> str:
    """
    Classify market regime using SPY (or configured proxy) close vs EMA-200
    with multi-day confirmation to reduce whipsaws.

        Returns one of:
            - "bull": normal long strategy
            - "risk_off": no new entries (legacy bear filter behavior)
            - "bear": enable bear playbook (inverse/defensive focus)
    """
    if regime_df is None or len(regime_df) < max(config.EMA_LONG, config.EMA_TREND) + 5:
        return last_regime

    confirm = max(2, int(config.MARKET_REGIME_CONFIRM_DAYS))
    if len(regime_df) < confirm + 1:
        return last_regime

    subset = regime_df.iloc[-confirm:]
    if "ema_200" not in subset.columns or "ema_trend" not in subset.columns:
        return last_regime

    close = subset["close"]
    ema_200 = subset["ema_200"]
    ema_50 = subset["ema_trend"]

    if close.isna().any() or ema_200.isna().any() or ema_50.isna().any():
        return last_regime

    latest_close = float(close.iloc[-1])
    latest_ema200 = float(ema_200.iloc[-1])

    # Fast bull re-entry: once market proxy closes back above EMA-200,
    # revert to bull regime immediately to avoid missing recovery legs.
    if latest_close > latest_ema200:
        return "bull"

    buffer = float(config.MARKET_REGIME_EMA_BUFFER)
    ema200_rising = ema_200.iloc[-1] > ema_200.iloc[0]
    ema200_falling = ema_200.iloc[-1] < ema_200.iloc[0]

    dd_lookback = max(60, int(config.BEAR_REGIME_DRAWDOWN_LOOKBACK))
    recent = regime_df["close"].iloc[-dd_lookback:]
    recent_high = float(recent.max()) if len(recent) else float(close.iloc[-1])
    drawdown = ((latest_close - recent_high) / recent_high) if recent_high > 0 else 0.0
    deep_drawdown = drawdown <= -float(config.BEAR_REGIME_DRAWDOWN_TRIGGER)

    bear_confirmed = bool(
        (close < ema_200 * (1 - buffer)).all()
        and ema_50.iloc[-1] < ema_200.iloc[-1]
        and ema_50.iloc[-1] < ema_50.iloc[0]
        and ema200_falling
        and deep_drawdown
    )

    if bear_confirmed:
        return "bear"

    # Below EMA-200 but not a deep/prolonged bear: stay risk-off and
    # preserve legacy behavior (skip new entries rather than forcing shorts).
    return "risk_off"


# ─────────────────────────────────────────────
# ENTRY  (scoring system)
# ─────────────────────────────────────────────
def score_entry(df: pd.DataFrame, weekly_bullish: bool = True) -> tuple[int, list[str]]:
    """
    Score the latest bar for entry quality.
    Returns (score, [list of contributing factors]).
    weekly_bullish: whether the weekly trend is aligned (from caller).
    """
    if len(df) < max(config.MOMENTUM_LOOKBACK + 1, config.EMA_SLOPE_PERIOD + 1, 3):
        return 0, []

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
    bb_lower = cur.get("bb_lower", None)
    bb_mid = cur.get("bb_mid", None)
    stoch_k = cur.get("stoch_k", None)
    stoch_d = cur.get("stoch_d", None)

    if pd.isna(rsi) or pd.isna(atr) or pd.isna(adx):
        return 0, []

    # ── Gap-up filter: skip entries after exhaustion gaps ────
    prev_close = prv["close"]
    today_open = cur["open"]
    if prev_close > 0 and today_open > 0:
        gap_pct = (today_open - prev_close) / prev_close
        if gap_pct > config.GAP_UP_MAX_PCT:
            return 0, [f"Gap-up {gap_pct*100:.1f}% (skipped)"]

    score = 0
    factors = []

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
        if dist_to_support <= 0.03:  # within 3% of support
            score += config.SR_SUPPORT_BONUS
            factors.append("Near support")
    if sr_resistance is not None and not pd.isna(sr_resistance):
        dist_to_resistance = (sr_resistance - price) / price if price > 0 else 1.0
        # Only penalize near resistance if stock is BELOW EMA-50 (overhead resistance)
        if dist_to_resistance <= config.SR_RESISTANCE_BUFFER and price < ema_trend:
            score -= 1
            factors.append("Near resistance (-1)")

    return score, factors


def score_entry_bear(df: pd.DataFrame, momentum: float = 0.0) -> tuple[int, list[str]]:
    """
    Score entries for bear regime (long inverse/defensive instruments).
    Designed to be trend-following and faster to de-risk.
    """
    if len(df) < max(config.MOMENTUM_LOOKBACK + 1, config.EMA_SLOPE_PERIOD + 1, 3):
        return 0, []

    cur = df.iloc[-1]
    prv = df.iloc[-2]

    price = cur["close"]
    ema_fast = cur["ema_fast"]
    ema_slow = cur["ema_slow"]
    ema_trend = cur["ema_trend"]
    rsi = cur["rsi"]
    macd_hist = cur["macd_hist"]
    adx = cur["adx"]
    vol_ratio = cur["vol_ratio"]
    atr = cur["atr"]

    if pd.isna(rsi) or pd.isna(atr) or pd.isna(adx):
        return 0, []

    score = 0
    factors: list[str] = []

    if price > ema_trend:
        score += 2
        factors.append("Above EMA-50")

    if ema_fast > ema_slow:
        score += 1
        factors.append("EMA-9 > EMA-21")

    if (prv["ema_fast"] <= prv["ema_slow"]) and (ema_fast > ema_slow):
        score += 1
        factors.append("Bullish EMA cross")

    if len(df) >= config.EMA_SLOPE_PERIOD + 1:
        ema50_now = cur["ema_trend"]
        ema50_ago = df.iloc[-(config.EMA_SLOPE_PERIOD + 1)]["ema_trend"]
        if not pd.isna(ema50_now) and not pd.isna(ema50_ago) and ema50_now > ema50_ago:
            score += 1
            factors.append("EMA-50 rising")

    if 40 <= rsi <= 72:
        score += 1
        factors.append(f"RSI healthy ({rsi:.0f})")

    if macd_hist > 0:
        score += 1
        factors.append("MACD positive")

    if adx >= 18:
        score += 1
        factors.append(f"ADX {adx:.0f}")

    if vol_ratio >= config.VOLUME_SURGE_FACTOR:
        score += 1
        factors.append(f"Volume {vol_ratio:.1f}x")

    if momentum > 0.03:
        score += 2
        factors.append(f"Momentum +{momentum*100:.0f}%")
    elif momentum > 0:
        score += 1
        factors.append(f"Momentum +{momentum*100:.0f}%")

    return score, factors


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


def check_entry(
    df: pd.DataFrame,
    weekly_bullish: bool = True,
    regime: str = "bull",
    momentum: float = 0.0,
    score_threshold: int | None = None,
) -> dict | None:
    """
    Evaluate the latest bar using the scoring system.
    Return a signal dict if score meets threshold, else None.
    """
    use_bear_logic = regime == "bear"

    if use_bear_logic:
        score, factors = score_entry_bear(df, momentum=momentum)
        threshold = (
            score_threshold
            if score_threshold is not None
            else config.BEAR_ENTRY_SCORE_THRESHOLD
        )
    else:
        score, factors = score_entry(df, weekly_bullish=weekly_bullish)
        if momentum > 0.05:
            score += config.MOMENTUM_SCORE_WEIGHT
            factors.append(f"Momentum +{momentum*100:.0f}%")
        elif momentum > 0.02:
            score += 1
            factors.append(f"Momentum +{momentum*100:.0f}%")
        threshold = (
            score_threshold
            if score_threshold is not None
            else config.ENTRY_SCORE_THRESHOLD
        )

    if score < threshold:
        return None

    cur = df.iloc[-1]
    price = cur["close"]
    atr = cur["atr"]

    if price <= 0 or pd.isna(price) or pd.isna(atr):
        return None

    atr_pct = atr / price
    max_atr_pct = config.MAX_ATR_PCT_BEAR if use_bear_logic else config.MAX_ATR_PCT_BULL
    if atr_pct > max_atr_pct:
        return None

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
        "regime": regime,
        "reason": reason,
    }
    log.info("ENTRY signal: regime=%s price=%.2f  %s", regime, price, reason)
    return signal


# ─────────────────────────────────────────────
# EXIT  (layered – hard + soft)
# ─────────────────────────────────────────────
def _check_exit_bear(
    df: pd.DataFrame,
    entry_price: float = 0.0,
    hold_days: int = 0,
) -> dict | None:
    if len(df) < 4:
        return None

    cur = df.iloc[-1]
    prv = df.iloc[-2]

    price = cur["close"]
    ema_fast = cur["ema_fast"]
    ema_slow = cur["ema_slow"]
    ema_trend = cur["ema_trend"]
    rsi = cur["rsi"]
    macd_hist = cur["macd_hist"]

    if pd.isna(rsi):
        return None

    hard_reasons: list[str] = []
    soft_reasons: list[str] = []

    if config.TIME_STOP_ENABLED and entry_price > 0 and hold_days >= config.TIME_STOP_DAYS_BEAR:
        lookback = max(2, hold_days + 1)
        window = df.tail(min(len(df), lookback))
        if not window.empty:
            peak_price = float(window["high"].max())
            peak_gain = (peak_price - entry_price) / entry_price
            weak_structure = (price < ema_trend) or (ema_fast < ema_slow)
            if weak_structure and peak_gain < config.TIME_STOP_MIN_PEAK_PROFIT_BEAR:
                hard_reasons.append(
                    f"Time stop ({hold_days}d, peak {peak_gain*100:.1f}%)"
                )

    if price < ema_trend and ema_fast < ema_slow:
        hard_reasons.append("Bear-mode trend break")

    if (prv["ema_fast"] >= prv["ema_slow"]) and (ema_fast < ema_slow):
        soft_reasons.append("Bearish EMA crossover")

    if prv["macd_hist"] > 0 and macd_hist < 0:
        soft_reasons.append("MACD flipped negative")

    if rsi >= config.BEAR_RSI_OVERBOUGHT_EXIT:
        soft_reasons.append(f"RSI overbought ({rsi:.1f})")

    if entry_price > 0 and hold_days >= config.BEAR_DEAD_MONEY_DAYS:
        move_pct = abs(price - entry_price) / entry_price
        if move_pct < config.BEAR_DEAD_MONEY_THRESHOLD:
            soft_reasons.append(
                f"Dead money ({hold_days}d, {move_pct*100:.1f}% move)"
            )

    if len(df) >= config.MOMENTUM_LOOKBACK + 1:
        past_price = df.iloc[-(config.MOMENTUM_LOOKBACK + 1)]["close"]
        if not pd.isna(past_price) and past_price > 0:
            mom = (price - past_price) / past_price
            if mom < -0.03:
                soft_reasons.append(f"Momentum decay ({mom*100:.1f}%)")

    reasons: list[str] = []
    if hard_reasons:
        reasons = hard_reasons
    elif soft_reasons:
        reasons = soft_reasons

    if not reasons:
        return None

    return {
        "action": "SELL",
        "price": price,
        "rsi": rsi,
        "macd_hist": macd_hist,
        "reasons": reasons,
        "regime": "bear",
    }


def check_exit(
    df: pd.DataFrame,
    entry_price: float = 0.0,
    hold_days: int = 0,
    regime: str = "bull",
) -> dict | None:
    """
    Evaluate whether an existing position should be closed.

    HARD exits fire immediately (1 is enough).
    SOFT exits require 2+ simultaneous signals to avoid
    getting shaken out by normal volatility.
    """
    if regime == "bear":
        signal = _check_exit_bear(df, entry_price=entry_price, hold_days=hold_days)
        if signal is not None:
            log.info(
                "EXIT  signal: regime=bear price=%.2f  reasons=%s",
                signal["price"],
                ", ".join(signal["reasons"]),
            )
        return signal

    if len(df) < 4:
        return None

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
        return None

    hard_reasons = []
    soft_reasons = []

    if config.TIME_STOP_ENABLED and entry_price > 0 and hold_days >= config.TIME_STOP_DAYS_BULL:
        lookback = max(2, hold_days + 1)
        window = df.tail(min(len(df), lookback))
        if not window.empty:
            peak_price = float(window["high"].max())
            peak_gain = (peak_price - entry_price) / entry_price
            weak_structure = (price < ema_trend) or (ema_fast < ema_slow)
            if weak_structure and peak_gain < config.TIME_STOP_MIN_PEAK_PROFIT_BULL:
                hard_reasons.append(
                    f"Time stop ({hold_days}d, peak {peak_gain*100:.1f}%)"
                )

    # ── HARD: price below BOTH EMA-50 and EMA-200 (trend destroyed) ──
    if ema_200 is not None and not pd.isna(ema_200):
        if price < ema_trend and price < ema_200:
            hard_reasons.append("Below EMA-50 & EMA-200")

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

    # ── Decision: hard fires immediately, soft needs 2+ ──────
    reasons = []
    if hard_reasons:
        reasons = hard_reasons
    elif len(soft_reasons) >= 2:
        reasons = soft_reasons

    if not reasons:
        return None

    signal = {
        "action": "SELL",
        "price": price,
        "rsi": rsi,
        "macd_hist": macd_hist,
        "reasons": reasons,
        "regime": "bull",
    }
    log.info("EXIT  signal: price=%.2f  reasons=%s", price, ", ".join(reasons))
    return signal
