"""
Dashboard API server — reads live state from Alpaca and the log file.

Run with:
    uvicorn dashboard.server:app --host 127.0.0.1 --port 8000 --reload

Then open:  http://localhost:8000
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
# ── make sure the parent dir (Trader/) is importable ─────────
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Trader Dashboard", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR  = Path(__file__).parent / "static"
LOG_PATH    = Path(__file__).parent.parent / "logs" / "trader.log"
CHART_PATH  = Path(__file__).parent.parent / "logs" / "backtest_equity.png"

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _get_api():
    """Return a connected Alpaca REST client."""
    import alpaca_trade_api as tradeapi
    return tradeapi.REST(
        key_id=config.ALPACA_API_KEY,
        secret_key=config.ALPACA_SECRET_KEY,
        base_url=config.BASE_URL,
        api_version="v2",
    )


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _bars_to_df(bars, symbol: str) -> "pd.DataFrame":
    """Convert Alpaca bars to a flat OHLCV DataFrame, handling MultiIndex columns."""
    import pandas as pd
    df = bars.df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        if symbol in df.columns.get_level_values(0):
            df = df.xs(symbol, axis=1, level=0)
        else:
            df = df.droplevel(0, axis=1)
    df.index = pd.to_datetime(df.index)
    return df[["open", "high", "low", "close", "volume"]]


def _pct(val) -> float:
    return round(_safe_float(val) * 100, 2)


# ─────────────────────────────────────────────────────────────
# Routes — static files
# ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    index = STATIC_DIR / "index.html"
    return HTMLResponse(index.read_text())


@app.get("/chart")
async def serve_chart():
    if not CHART_PATH.exists():
        raise HTTPException(status_code=404, detail="Chart not found")
    return FileResponse(str(CHART_PATH), media_type="image/png")


# ─────────────────────────────────────────────────────────────
# Routes — live data
# ─────────────────────────────────────────────────────────────

@app.get("/api/account")
async def get_account():
    """Equity, cash, buying power, day-trade count."""
    try:
        api = _get_api()
        acct = api.get_account()
        equity        = _safe_float(acct.equity)
        cash          = _safe_float(acct.cash)
        buying_power  = _safe_float(acct.buying_power)
        last_equity   = _safe_float(getattr(acct, "last_equity", equity))
        day_pnl       = equity - last_equity
        day_pnl_pct   = (day_pnl / last_equity * 100) if last_equity else 0.0

        return {
            "equity":        round(equity, 2),
            "cash":          round(cash, 2),
            "buying_power":  round(buying_power, 2),
            "day_pnl":       round(day_pnl, 2),
            "day_pnl_pct":   round(day_pnl_pct, 2),
            "daytrade_count": int(getattr(acct, "daytrade_count", 0)),
            "mode":          config.TRADING_MODE.upper(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/positions")
async def get_positions():
    """All open positions with live P&L."""
    try:
        api = _get_api()
        raw = api.list_positions()
        positions = []
        for p in raw:
            entry  = _safe_float(p.avg_entry_price)
            cur    = _safe_float(p.current_price)
            qty    = _safe_float(p.qty)
            pnl    = _safe_float(p.unrealized_pl)
            pnl_pct = _pct(p.unrealized_plpc)
            mkt_val = _safe_float(p.market_value)
            positions.append({
                "symbol":      p.symbol,
                "qty":         qty,
                "entry_price": round(entry, 2),
                "current_price": round(cur, 2),
                "market_value":  round(mkt_val, 2),
                "pnl":         round(pnl, 2),
                "pnl_pct":     pnl_pct,
                "side":        p.side,
            })
        return positions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/positions_sentiment")
async def get_positions_sentiment():
    """FinBERT sentiment snapshot for all currently held symbols."""
    try:
        api = _get_api()
        raw_positions = api.list_positions()
        symbols = [p.symbol for p in raw_positions]

        if not symbols:
            return {
                "enabled": bool(getattr(config, "NLP_SENTIMENT_ENABLED", False)),
                "items": [],
            }

        if not getattr(config, "NLP_SENTIMENT_ENABLED", False):
            return {
                "enabled": False,
                "items": [
                    {
                        "symbol": sym,
                        "score": None,
                        "label": "DISABLED",
                        "headline_count": 0,
                    }
                    for sym in symbols
                ],
            }

        import sentiment

        limit = int(getattr(config, "NLP_NEWS_LIMIT_PER_SYMBOL", 10))
        items = []

        for sym in symbols:
            try:
                news_items = api.get_news(sym, limit=limit)
                headlines = [n.headline for n in news_items if hasattr(n, "headline")]
            except Exception:
                headlines = []

            score = sentiment.get_sentiment(headlines)
            if score >= 0.20:
                label = "POSITIVE"
            elif score <= -0.20:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"

            items.append(
                {
                    "symbol": sym,
                    "score": round(score, 3),
                    "label": label,
                    "headline_count": len(headlines),
                }
            )

        return {"enabled": True, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/orders")
async def get_orders():
    """Recent filled orders (trade history)."""
    try:
        api = _get_api()
        raw = api.list_orders(status="closed", limit=100, direction="desc")
        orders = []
        for o in raw:
            if o.filled_at is None:
                continue
            filled_price = _safe_float(getattr(o, "filled_avg_price", 0))
            qty          = _safe_float(getattr(o, "filled_qty", 0))
            orders.append({
                "id":           o.id,
                "symbol":       o.symbol,
                "side":         o.side,
                "qty":          qty,
                "filled_price": round(filled_price, 2),
                "filled_at":    str(o.filled_at)[:19].replace("T", " "),
                "order_type":   o.order_type,
            })
        return orders
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/clock")
async def get_clock():
    """Market open/close status."""
    try:
        api = _get_api()
        clock = api.get_clock()
        return {
            "is_open":    clock.is_open,
            "next_open":  str(clock.next_open)[:16].replace("T", " "),
            "next_close": str(clock.next_close)[:16].replace("T", " "),
            "timestamp":  str(clock.timestamp)[:19].replace("T", " "),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/regime")
async def get_regime():
    """
    Market regime: SPY vs SMA-200, VIXY level, bear mode flag.
    """
    try:
        from indicators import compute_all
        import pandas as pd
        api = _get_api()

        # SPY regime — need 250 daily bars for SMA-200; use start date not limit
        import datetime as _dt
        spy_start = (_dt.date.today() - _dt.timedelta(days=380)).isoformat()
        bars = api.get_bars("SPY", config.BAR_TIMEFRAME,
                            start=spy_start, limit=10_000,
                            feed=config.DATA_FEED)
        spy_df = _bars_to_df(bars, "SPY")
        spy_df = compute_all(spy_df)

        row       = spy_df.iloc[-1]
        spy_close = _safe_float(row["close"])
        sma_200   = _safe_float(row.get("sma_200", 0))
        ema_200   = _safe_float(row.get("ema_200", 0))
        ema_50    = _safe_float(row.get("ema_trend", 0))
        # Use EMA-200 for regime detection (matches executor.py logic)
        regime_ema = ema_200 if ema_200 > 0 else sma_200
        bull_market  = spy_close > regime_ema if regime_ema > 0 else True
        sma_pct_diff = ((spy_close - regime_ema) / regime_ema * 100) if regime_ema else 0

        # VIXY level
        vixy_level  = None
        long_halted = False
        vix_reduced = False
        if config.VIX_FILTER_ENABLED:
            try:
                vix_start = (_dt.date.today() - _dt.timedelta(days=5)).isoformat()
                vbars = api.get_bars(config.VIX_SYMBOL, config.BAR_TIMEFRAME,
                                     start=vix_start, limit=10_000,
                                     feed=config.DATA_FEED)
                vix_df = _bars_to_df(vbars, config.VIX_SYMBOL)
                vixy_level  = round(_safe_float(vix_df.iloc[-1]["close"]), 2)
                long_halted = vixy_level >= config.VIX_HALT_THRESHOLD
                vix_reduced = vixy_level >= config.VIX_REDUCE_THRESHOLD
            except Exception:
                pass

        return {
            "bull_market":    bull_market,
            "regime_label":   "BULL" if bull_market else "BEAR",
            "spy_close":      round(spy_close, 2),
            "sma_200":        round(sma_200, 2),
            "ema_200":        round(ema_200, 2),
            "ema_50":         round(ema_50, 2),
            "sma_pct_diff":   round(sma_pct_diff, 2),
            "vixy_level":     vixy_level,
            "long_halted":    long_halted,
            "vix_reduced":    vix_reduced,
            "vix_halt_threshold":   config.VIX_HALT_THRESHOLD,
            "vix_reduce_threshold": config.VIX_REDUCE_THRESHOLD,
            "inverse_mode_enabled": config.INVERSE_ETF_MODE_ENABLED,
            "bear_short_mode_enabled": getattr(config, "BEAR_SHORT_MODE_ENABLED", False),
            "short_min_equity": getattr(config, "SHORT_MIN_EQUITY", 2000.0),
            "bear_mode_note": (
                "Equity < $%.0f: inverse ETFs mode" % getattr(config, "SHORT_MIN_EQUITY", 2000.0)
                if not bull_market
                else "Bull market — normal long mode"
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/watchlist")
async def get_watchlist():
    """
    Top watchlist symbols with latest close, RSI, and entry score.
    Returns up to 30 symbols to keep the request fast.
    """
    try:
        from indicators import compute_all
        import pandas as pd
        api = _get_api()
        symbols = config.WATCHLIST[:30]
        results = []

        import datetime as _dt
        wl_start = (_dt.date.today() - _dt.timedelta(days=120)).isoformat()
        for sym in symbols:
            try:
                bars = api.get_bars(sym, config.BAR_TIMEFRAME,
                                    start=wl_start, limit=10_000,
                                    feed=config.DATA_FEED)
                df   = _bars_to_df(bars, sym)
                df   = compute_all(df)
                row  = df.iloc[-1]

                close    = _safe_float(row["close"])
                rsi      = _safe_float(row.get("rsi", 0))
                adx      = _safe_float(row.get("adx", 0))
                macd_h   = _safe_float(row.get("macd_hist", 0))
                ema_fast = _safe_float(row.get("ema_fast", 0))
                ema_slow = _safe_float(row.get("ema_slow", 0))
                ema_trend = _safe_float(row.get("ema_trend", 0))
                vol_ratio = _safe_float(row.get("vol_ratio", 0))
                atr      = _safe_float(row.get("atr", 0))

                # Quick score (subset of full entry scoring)
                score = 0
                if close > ema_trend:          score += 2
                ema_200 = _safe_float(row.get("ema_200", 0))
                if ema_200 > 0 and close > ema_200: score += 1
                if ema_fast > ema_slow:        score += 1
                if 30 <= rsi <= 50:            score += 2
                elif 50 < rsi <= 60:           score += 1
                if macd_h > 0:                 score += 1
                if vol_ratio >= 1.0:           score += 1
                if adx > 20:                   score += 1

                results.append({
                    "symbol":    sym,
                    "close":     round(close, 2),
                    "rsi":       round(rsi, 1),
                    "adx":       round(adx, 1),
                    "macd_hist": round(macd_h, 4),
                    "vol_ratio": round(vol_ratio, 2),
                    "atr":       round(atr, 2),
                    "score":     score,
                    "above_ema50": close > ema_trend,
                })
            except Exception:
                continue

        results.sort(key=lambda x: x["score"], reverse=True)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/equity_history")
async def get_equity_history():
    """
    30-day portfolio history from Alpaca account activities.
    Returns list of {date, equity} points for the equity curve.
    """
    try:
        api = _get_api()
        # Alpaca portfolio history endpoint
        ph = api.get_portfolio_history(
            period="1M",
            timeframe="1D",
            extended_hours=False,
        )
        dates    = ph.timestamp   # list of unix timestamps
        equities = ph.equity      # list of floats

        points = []
        for ts, eq in zip(dates, equities):
            if eq is None or eq == 0:
                continue
            d = dt.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
            points.append({"date": d, "equity": round(float(eq), 2)})
        return points
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logs")
async def get_logs(lines: int = 100):
    """Return the last N lines from the bot log file."""
    if not LOG_PATH.exists():
        return {"lines": []}
    try:
        with open(LOG_PATH, "r") as f:
            all_lines = f.readlines()
        tail = [l.rstrip() for l in all_lines[-lines:]]
        return {"lines": tail}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# Server-Sent Events — push updates every 30 s
# ─────────────────────────────────────────────────────────────

async def _sse_generator():
    """Yield SSE events with a combined snapshot every 30 seconds."""
    while True:
        try:
            api = _get_api()
            acct = api.get_account()
            equity = _safe_float(acct.equity)
            cash   = _safe_float(acct.cash)
            last_equity = _safe_float(getattr(acct, "last_equity", equity))
            day_pnl = equity - last_equity
            day_pnl_pct = (day_pnl / last_equity * 100) if last_equity else 0.0

            positions = api.list_positions()
            pos_list = [{
                "symbol":        p.symbol,
                "qty":           _safe_float(p.qty),
                "entry_price":   round(_safe_float(p.avg_entry_price), 2),
                "current_price": round(_safe_float(p.current_price), 2),
                "pnl":           round(_safe_float(p.unrealized_pl), 2),
                "pnl_pct":       _pct(p.unrealized_plpc),
            } for p in positions]

            clock = api.get_clock()

            payload = {
                "ts":            dt.datetime.now().strftime("%H:%M:%S"),
                "equity":        round(equity, 2),
                "cash":          round(cash, 2),
                "day_pnl":       round(day_pnl, 2),
                "day_pnl_pct":   round(day_pnl_pct, 2),
                "n_positions":   len(positions),
                "positions":     pos_list,
                "market_open":   clock.is_open,
                "daytrade_count": int(getattr(acct, "daytrade_count", 0)),
            }
            yield f"data: {json.dumps(payload)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        await asyncio.sleep(30)


@app.get("/api/stream")
async def stream():
    """SSE endpoint — the UI subscribes to this for live updates."""
    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────────────────────
# Config read / write endpoints
# ─────────────────────────────────────────────────────────────

import threading
import queue
import importlib
from pydantic import BaseModel

# Path where UI-driven overrides are persisted between restarts
CONFIG_OVERRIDES_PATH = Path(__file__).parent.parent / "config_overrides.json"

# Path where backtest-specific algorithm overrides are persisted
BACKTEST_OVERRIDES_PATH = Path(__file__).parent.parent / "backtest_config_overrides.json"

# All keys that the UI is allowed to read/write (never touch secrets or paths)
_CONFIG_EDITABLE_KEYS = {
    # Indicators
    "EMA_FAST", "EMA_SLOW", "EMA_TREND", "EMA_LONG",
    "RSI_PERIOD", "RSI_OVERSOLD", "RSI_OVERBOUGHT",
    "MACD_FAST", "MACD_SLOW", "MACD_SIGNAL",
    "ATR_PERIOD", "ATR_STOP_MULTIPLIER", "ATR_PROFIT_MULTIPLIER",
    "VOLUME_SMA_PERIOD", "VOLUME_SURGE_FACTOR",
    # Entry scoring
    "ENTRY_SCORE_THRESHOLD",
    "MOMENTUM_LOOKBACK", "MOMENTUM_TOP_PCT", "MOMENTUM_SCORE_WEIGHT",
    "EMA_SLOPE_PERIOD",
    "GAP_UP_MAX_PCT",
    "SR_LOOKBACK", "SR_RESISTANCE_BUFFER", "SR_SUPPORT_BONUS",
    "DYNAMIC_THRESHOLD_ENABLED", "DYNAMIC_THRESHOLD_ADJUSTMENT",
    # Trend quality / weekly trend
    "EMA_SLOPE_PERIOD",
    "WEEKLY_EMA_FAST", "WEEKLY_EMA_SLOW", "WEEKLY_TREND_BONUS",
    "WEEKLY_TREND_ENABLED",
    # Exit / hold
    "DEAD_MONEY_DAYS", "DEAD_MONEY_THRESHOLD",
    "RE_ENTRY_COOLDOWN_DAYS",
    "MIN_HOLD_CALENDAR_DAYS",
    # Risk
    "MAX_OPEN_POSITIONS", "MAX_POSITION_PCT", "MAX_PORTFOLIO_RISK_PCT",
    "MAX_LOSS_PER_TRADE_PCT", "MAX_PORTFOLIO_EXPOSURE_PCT",
    "TRAILING_STOP_ACTIVATE_PCT", "TRAILING_STOP_PCT",
    "TRAILING_STOP_TIGHT_ACTIVATE", "TRAILING_STOP_TIGHT_PCT",
    # Small account
    "SMALL_ACCOUNT_THRESHOLD", "FRACTIONAL_SHARES",
    "SMALL_MAX_OPEN_POSITIONS", "SMALL_MAX_POSITION_PCT",
    "SMALL_MAX_LOSS_PER_TRADE_PCT", "SMALL_ATR_STOP_MULTIPLIER",
    "SMALL_ATR_PROFIT_MULTIPLIER",
    # Regime / VIX / Inverse ETF
    "MARKET_REGIME_ENABLED", "MARKET_REGIME_SYMBOL",
    "VOL_REGIME_ENABLED", "REALIZED_VOL_WINDOW",
    "HIGH_VOL_THRESHOLD", "LOW_VOL_THRESHOLD",
    "HIGH_VOL_SIZE_SCALE", "LOW_VOL_SIZE_SCALE",
    "VIX_FILTER_ENABLED", "VIX_SYMBOL",
    "VIX_HALT_THRESHOLD", "VIX_REDUCE_THRESHOLD", "VIX_SIZE_SCALE",
    "INVERSE_ETF_MODE_ENABLED", "INVERSE_ETF_SIZE_SCALE",
    # Universe / data
    "MIN_PRICE", "MAX_PRICE", "MIN_AVG_VOLUME",
    "BARS_LOOKBACK",
    # Scheduling
    "SCAN_INTERVAL_MINUTES", "CHECK_EXITS_MINUTES",
    # Sector
    "MAX_PER_SECTOR",
    # PDT
    "MAX_DAY_TRADES_ALLOWED", "PDT_LOOKBACK_DAYS",
    # Machine Learning
    "ML_ENABLED", "ML_ENTRY_THRESHOLD", "ML_MIN_SCORE",
    "ML_BLEND_MODE", "ML_FORWARD_BARS",
    "ML_MIN_GAIN_PCT", "ML_TRAINING_MONTHS",
    # Data feed
    "DATA_FEED",
}


def _load_overrides() -> dict:
    """Load persisted overrides from disk (empty dict if none)."""
    if CONFIG_OVERRIDES_PATH.exists():
        try:
            return json.loads(CONFIG_OVERRIDES_PATH.read_text())
        except Exception:
            return {}
    return {}


def _apply_overrides(overrides: dict) -> None:
    """Apply a dict of key→value onto the live config module."""
    for key, val in overrides.items():
        if key in _CONFIG_EDITABLE_KEYS and hasattr(config, key):
            setattr(config, key, val)


def _save_and_apply(overrides: dict) -> None:
    """Persist overrides to disk and apply them live."""
    CONFIG_OVERRIDES_PATH.write_text(json.dumps(overrides, indent=2))
    _apply_overrides(overrides)


# Apply any saved overrides immediately at startup
_apply_overrides(_load_overrides())


@app.get("/api/config")
async def get_config():
    """Return all editable config values (current live values + metadata)."""
    result = {}
    for key in sorted(_CONFIG_EDITABLE_KEYS):
        val = getattr(config, key, None)
        result[key] = val
    return result


class ConfigPatchRequest(BaseModel):
    updates: Dict[str, Any]


@app.post("/api/config")
async def patch_config(req: ConfigPatchRequest):
    """
    Apply a partial update to the live config.
    Only keys in _CONFIG_EDITABLE_KEYS are accepted.
    Changes are persisted to config_overrides.json.
    """
    bad_keys = [k for k in req.updates if k not in _CONFIG_EDITABLE_KEYS]
    if bad_keys:
        raise HTTPException(status_code=400, detail=f"Non-editable keys: {bad_keys}")

    # Merge with existing overrides
    overrides = _load_overrides()
    for key, val in req.updates.items():
        expected_type = type(getattr(config, key, None))
        # Coerce type to match config (bool must come before int check)
        if expected_type is bool:
            val = bool(val)
        elif expected_type is int:
            val = int(val)
        elif expected_type is float:
            val = float(val)
        elif expected_type is str:
            val = str(val)
        overrides[key] = val

    _save_and_apply(overrides)
    return {"ok": True, "applied": list(req.updates.keys())}


@app.post("/api/config/reset")
async def reset_config():
    """Remove all overrides, reverting to defaults in config.py."""
    if CONFIG_OVERRIDES_PATH.exists():
        CONFIG_OVERRIDES_PATH.unlink()
    # Reload module from disk to restore defaults
    importlib.reload(config)
    return {"ok": True}


# ─────────────────────────────────────────────────────────────
# Backtest-specific config endpoints
# Independent parameters used only during backtests — never
# touch the live bot settings.
# ─────────────────────────────────────────────────────────────

def _load_backtest_overrides() -> dict:
    """Load persisted backtest-specific overrides (empty dict if none)."""
    if BACKTEST_OVERRIDES_PATH.exists():
        try:
            return json.loads(BACKTEST_OVERRIDES_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_backtest_overrides(overrides: dict) -> None:
    """Persist backtest-specific overrides to disk."""
    BACKTEST_OVERRIDES_PATH.write_text(json.dumps(overrides, indent=2))


@app.get("/api/backtest/config")
async def get_backtest_config():
    """
    Return all editable backtest config values.
    Falls back to the current live config value for any key that has
    no backtest-specific override yet.
    """
    bt_overrides = _load_backtest_overrides()
    result = {}
    for key in sorted(_CONFIG_EDITABLE_KEYS):
        # Prefer the backtest-specific override; fall back to live value
        if key in bt_overrides:
            result[key] = bt_overrides[key]
        else:
            result[key] = getattr(config, key, None)
    return result


class BacktestConfigPatchRequest(BaseModel):
    updates: Dict[str, Any]


@app.post("/api/backtest/config")
async def patch_backtest_config(req: BacktestConfigPatchRequest):
    """
    Apply a partial update to the backtest-specific config.
    Only keys in _CONFIG_EDITABLE_KEYS are accepted.
    Changes are persisted to backtest_config_overrides.json and do NOT
    affect the live running bot.
    """
    bad_keys = [k for k in req.updates if k not in _CONFIG_EDITABLE_KEYS]
    if bad_keys:
        raise HTTPException(status_code=400, detail=f"Non-editable keys: {bad_keys}")

    overrides = _load_backtest_overrides()
    for key, val in req.updates.items():
        expected_type = type(getattr(config, key, None))
        if expected_type is bool:
            val = bool(val)
        elif expected_type is int:
            val = int(val)
        elif expected_type is float:
            val = float(val)
        elif expected_type is str:
            val = str(val)
        overrides[key] = val

    _save_backtest_overrides(overrides)
    return {"ok": True, "applied": list(req.updates.keys())}


@app.post("/api/backtest/config/clone-live")
async def clone_live_to_backtest():
    """
    Copy all current live config values into the backtest-specific overrides,
    making the backtest parameters an exact clone of the live bot settings.
    """
    live_snapshot = {key: getattr(config, key, None) for key in _CONFIG_EDITABLE_KEYS}
    _save_backtest_overrides(live_snapshot)
    return {"ok": True, "cloned_keys": len(live_snapshot)}


@app.post("/api/backtest/config/reset")
async def reset_backtest_config():
    """
    Remove all backtest-specific overrides, so the next backtest will
    use defaults from config.py (same as the live bot baseline).
    """
    if BACKTEST_OVERRIDES_PATH.exists():
        BACKTEST_OVERRIDES_PATH.unlink()
    return {"ok": True}


# ─────────────────────────────────────────────────────────────
# Trading mode switch (paper ↔ live)
# ─────────────────────────────────────────────────────────────

_TRADING_MODE_KEY = "__TRADING_MODE__"


def _get_trading_mode() -> str:
    """Return the currently active trading mode ('paper' or 'live')."""
    overrides = _load_overrides()
    return overrides.get(_TRADING_MODE_KEY, config.TRADING_MODE)


def _set_trading_mode(mode: str) -> None:
    """Persist and apply a new trading mode — swaps URL and key pair."""
    overrides = _load_overrides()
    overrides[_TRADING_MODE_KEY] = mode
    CONFIG_OVERRIDES_PATH.write_text(json.dumps(overrides, indent=2))
    config.TRADING_MODE = mode
    config.BASE_URL = config.PAPER_BASE_URL if mode == "paper" else config.LIVE_BASE_URL
    # Swap the active key pair to match the selected account
    if mode == "paper":
        config.ALPACA_API_KEY    = config.PAPER_ALPACA_API_KEY
        config.ALPACA_SECRET_KEY = config.PAPER_ALPACA_SECRET_KEY
    else:
        config.ALPACA_API_KEY    = config.LIVE_ALPACA_API_KEY
        config.ALPACA_SECRET_KEY = config.LIVE_ALPACA_SECRET_KEY


# Apply persisted trading mode at startup
_stored_mode = _load_overrides().get(_TRADING_MODE_KEY)
if _stored_mode in ("paper", "live"):
    _set_trading_mode(_stored_mode)


@app.get("/api/trading-mode")
async def get_trading_mode():
    """Return the current trading mode and both endpoint URLs."""
    mode = _get_trading_mode()
    live_keys_configured = bool(
        config.LIVE_ALPACA_API_KEY and config.LIVE_ALPACA_SECRET_KEY
    )
    return {
        "mode":                  mode,
        "base_url":              config.BASE_URL,
        "paper_url":             config.PAPER_BASE_URL,
        "live_url":              config.LIVE_BASE_URL,
        "live_keys_configured":  live_keys_configured,
    }


class TradingModeRequest(BaseModel):
    mode: str           # "paper" or "live"
    confirm: str        # must equal "CONFIRM" to switch to live


@app.post("/api/trading-mode")
async def set_trading_mode(req: TradingModeRequest):
    """
    Switch between paper and live trading.
    Switching to live requires confirm="CONFIRM" as an explicit safety gate.
    """
    if req.mode not in ("paper", "live"):
        raise HTTPException(status_code=400, detail="mode must be 'paper' or 'live'")

    if req.mode == "live" and req.confirm != "CONFIRM":
        raise HTTPException(
            status_code=400,
            detail="Switching to live trading requires confirm='CONFIRM'",
        )

    _set_trading_mode(req.mode)
    return {
        "ok":      True,
        "mode":    req.mode,
        "base_url": config.BASE_URL,
    }


# ─────────────────────────────────────────────────────────────
# Watchlist management endpoints
# ─────────────────────────────────────────────────────────────

# Overrides file also stores __WATCHLIST__ and __INVERSE_WATCHLIST__
# as special keys (double-underscore) so they don't collide with
# the scalar config keys.

_WL_KEY  = "__WATCHLIST__"
_IWL_KEY = "__INVERSE_WATCHLIST__"


def _load_watchlists() -> tuple[list, list]:
    """
    Return (watchlist, inverse_watchlist) — either from persisted
    overrides or directly from the live config module.
    """
    overrides = _load_overrides()
    wl  = overrides.get(_WL_KEY,  list(config.WATCHLIST))
    iwl = overrides.get(_IWL_KEY, list(config.INVERSE_WATCHLIST))
    return wl, iwl


def _save_watchlists(wl: list, iwl: list) -> None:
    """Persist watchlist changes and apply them to the live config."""
    overrides = _load_overrides()
    overrides[_WL_KEY]  = wl
    overrides[_IWL_KEY] = iwl
    CONFIG_OVERRIDES_PATH.write_text(json.dumps(overrides, indent=2))
    config.WATCHLIST          = wl
    config.INVERSE_WATCHLIST  = iwl


# Apply any persisted watchlist overrides at startup
_wl_init, _iwl_init = _load_watchlists()
config.WATCHLIST         = _wl_init
config.INVERSE_WATCHLIST = _iwl_init


@app.get("/api/watchlists")
async def get_watchlists():
    """Return both the main watchlist and the inverse ETF watchlist."""
    wl, iwl = _load_watchlists()
    return {"watchlist": wl, "inverse_watchlist": iwl}


class WatchlistUpdateRequest(BaseModel):
    action: str         # "add" | "remove"
    symbol: str         # ticker, e.g. "TSLA"
    list_name: str      # "watchlist" | "inverse_watchlist"


@app.post("/api/watchlists")
async def update_watchlist(req: WatchlistUpdateRequest):
    """Add or remove a symbol from the main or inverse watchlist."""
    sym = req.symbol.strip().upper()
    if not sym:
        raise HTTPException(status_code=400, detail="Symbol cannot be empty")

    if req.list_name not in ("watchlist", "inverse_watchlist"):
        raise HTTPException(status_code=400, detail="list_name must be 'watchlist' or 'inverse_watchlist'")

    if req.action not in ("add", "remove"):
        raise HTTPException(status_code=400, detail="action must be 'add' or 'remove'")

    wl, iwl = _load_watchlists()
    target = wl if req.list_name == "watchlist" else iwl

    if req.action == "add":
        if sym not in target:
            target.append(sym)
    else:  # remove
        if sym not in target:
            raise HTTPException(status_code=404, detail=f"{sym} not found in {req.list_name}")
        target.remove(sym)

    if req.list_name == "watchlist":
        _save_watchlists(target, iwl)
    else:
        _save_watchlists(wl, target)

    return {"ok": True, "action": req.action, "symbol": sym, "list_name": req.list_name}


@app.post("/api/watchlists/reset")
async def reset_watchlists():
    """Remove watchlist overrides, restoring config.py defaults."""
    overrides = _load_overrides()
    overrides.pop(_WL_KEY,  None)
    overrides.pop(_IWL_KEY, None)
    if overrides:
        CONFIG_OVERRIDES_PATH.write_text(json.dumps(overrides, indent=2))
    elif CONFIG_OVERRIDES_PATH.exists():
        CONFIG_OVERRIDES_PATH.unlink()
    importlib.reload(config)
    return {"ok": True}


# ─────────────────────────────────────────────────────────────
# ML model endpoints  (train / status / feature importance)
# ─────────────────────────────────────────────────────────────

_ml_training = False
_ml_training_status: Optional[str] = None


@app.get("/api/ml/status")
async def ml_status():
    """
    Return the ML model status: loaded, training metrics, last trained, etc.
    """
    import ml_model
    meta = ml_model.get_meta()
    model_loaded = ml_model.is_available()
    return {
        "ml_enabled":     config.ML_ENABLED,
        "model_loaded":   model_loaded,
        "training":       _ml_training,
        "training_status": _ml_training_status,
        "meta":           meta,
    }


@app.get("/api/ml/features")
async def ml_features():
    """Return feature importance data for the loaded model."""
    import ml_model
    meta = ml_model.get_meta()
    if meta is None:
        raise HTTPException(status_code=404, detail="No trained model found")
    return {
        "feature_importance": meta.get("feature_importance", {}),
        "feature_names": meta.get("feature_names", []),
        "avg_metrics": meta.get("avg_metrics", {}),
    }


class MLTrainRequest(BaseModel):
    months: Optional[int] = None
    forward_bars: Optional[int] = None
    min_gain_pct: Optional[float] = None
    symbols: Optional[List[str]] = None


def _ml_train_background(req: MLTrainRequest) -> None:
    """Background thread that runs the full ML training pipeline."""
    global _ml_training, _ml_training_status
    try:
        from ml_trainer import run_training
        import ml_model

        _ml_training_status = "Loading data…"
        meta = run_training(
            symbols=req.symbols,
            months=req.months or config.ML_TRAINING_MONTHS,
            forward_bars=req.forward_bars or config.ML_FORWARD_BARS,
            min_gain_pct=req.min_gain_pct or config.ML_MIN_GAIN_PCT,
        )

        # Reload model in the inference cache
        ml_model.reload_model()

        avg = meta.get("avg_metrics", {})
        _ml_training_status = (
            f"Done — AUC={avg.get('auc', 0):.3f}  "
            f"F1={avg.get('f1', 0):.3f}  "
            f"({meta.get('n_samples', 0)} samples)"
        )
    except Exception as exc:
        _ml_training_status = f"Error: {exc}"
        import traceback; traceback.print_exc()
    finally:
        _ml_training = False


@app.post("/api/ml/train")
async def ml_train(req: MLTrainRequest = MLTrainRequest()):
    """
    Kick off ML model training in a background thread.
    Returns immediately — poll /api/ml/status to monitor.
    """
    global _ml_training, _ml_training_status

    if _ml_training:
        raise HTTPException(status_code=409, detail="Training already in progress")

    _ml_training = True
    _ml_training_status = "Starting…"

    t = threading.Thread(target=_ml_train_background, args=(req,), daemon=True)
    t.start()

    return {"ok": True, "message": "Training started in background"}


@app.post("/api/ml/reload")
async def ml_reload():
    """Reload the ML model from disk (after external training)."""
    import ml_model
    loaded = ml_model.reload_model()
    return {"ok": True, "model_loaded": loaded}


# ─────────────────────────────────────────────────────────────
# Bot process management  (start / stop / status)
# ─────────────────────────────────────────────────────────────

import subprocess
import signal

MAIN_PY = Path(__file__).parent.parent / "main.py"

# One subprocess slot per trading mode
_bot_procs: Dict[str, Optional[subprocess.Popen]] = {"paper": None, "live": None}
_bot_lock  = threading.Lock()
_bot_start_times: Dict[str, Optional[float]] = {"paper": None, "live": None}

# Per-mode log queues (last 200 lines kept in memory)
_bot_log_lines: Dict[str, List[str]] = {"paper": [], "live": []}
_bot_log_queues: Dict[str, "queue.Queue[str]"] = {
    "paper": queue.Queue(),
    "live":  queue.Queue(),
}
_BOT_LOG_MAX = 200


def _bot_reader_thread(mode: str, proc: subprocess.Popen) -> None:
    """Background thread: read stdout+stderr from a bot process and buffer it."""
    import select, io
    try:
        for raw in proc.stdout:  # type: ignore[union-attr]
            line = raw.rstrip("\n")
            with _bot_lock:
                buf = _bot_log_lines[mode]
                buf.append(line)
                if len(buf) > _BOT_LOG_MAX:
                    buf.pop(0)
            _bot_log_queues[mode].put(line)
    except Exception:
        pass


def _bot_state(mode: str) -> dict:
    """Return a status dict for one bot process."""
    with _bot_lock:
        proc  = _bot_procs[mode]
        start = _bot_start_times[mode]

    running = False
    pid     = None
    if proc is not None:
        rc = proc.poll()
        running = (rc is None)
        if running:
            pid = proc.pid
        else:
            # Process died — clean up slot
            with _bot_lock:
                _bot_procs[mode]       = None
                _bot_start_times[mode] = None
            start = None

    uptime_s = None
    if running and start is not None:
        import time as _t
        uptime_s = int(_t.time() - start)

    return {
        "mode":     mode,
        "running":  running,
        "pid":      pid,
        "uptime_s": uptime_s,
    }


@app.get("/api/bot/status")
async def bot_status():
    """Return running state for both paper and live bot processes."""
    return {
        "paper": _bot_state("paper"),
        "live":  _bot_state("live"),
    }


class BotStartRequest(BaseModel):
    mode: str   # "paper" | "live"


@app.post("/api/bot/start")
async def bot_start(req: BotStartRequest):
    """Launch main.py for the given mode (paper or live)."""
    mode = req.mode
    if mode not in ("paper", "live"):
        raise HTTPException(status_code=400, detail="mode must be 'paper' or 'live'")

    # Guard: live requires keys
    if mode == "live" and not (config.LIVE_ALPACA_API_KEY and config.LIVE_ALPACA_SECRET_KEY):
        raise HTTPException(status_code=400, detail="Live API keys are not configured in .env")

    with _bot_lock:
        proc = _bot_procs[mode]
        if proc is not None and proc.poll() is None:
            raise HTTPException(status_code=409, detail=f"{mode} bot is already running (pid {proc.pid})")

    # Build environment: inherit current env, override TRADING_MODE
    env = os.environ.copy()
    env["TRADING_MODE"] = mode
    if mode == "paper":
        env["ALPACA_API_KEY"]    = config.PAPER_ALPACA_API_KEY
        env["ALPACA_SECRET_KEY"] = config.PAPER_ALPACA_SECRET_KEY
    else:
        env["ALPACA_API_KEY"]    = config.LIVE_ALPACA_API_KEY
        env["ALPACA_SECRET_KEY"] = config.LIVE_ALPACA_SECRET_KEY

    # Inject all saved config overrides as CFG_OVERRIDE_<KEY>=<value> so
    # the subprocess config.py picks them up (it cannot read config_overrides.json
    # because that is only applied by the dashboard server process).
    saved_overrides = _load_overrides()
    for k, v in saved_overrides.items():
        if k in _CONFIG_EDITABLE_KEYS:          # skip __WATCHLIST__ etc.
            env[f"CFG_OVERRIDE_{k}"] = str(v)

    proc = subprocess.Popen(
        [sys.executable, str(MAIN_PY)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(MAIN_PY.parent),
    )

    import time as _t
    with _bot_lock:
        _bot_procs[mode]       = proc
        _bot_start_times[mode] = _t.time()
        _bot_log_lines[mode].clear()

    # Drain old SSE queue
    q = _bot_log_queues[mode]
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break

    # Start reader thread
    t = threading.Thread(target=_bot_reader_thread, args=(mode, proc), daemon=True)
    t.start()

    return {"ok": True, "pid": proc.pid, "mode": mode}


@app.post("/api/bot/stop")
async def bot_stop(req: BotStartRequest):
    """Gracefully stop the bot for the given mode."""
    mode = req.mode
    if mode not in ("paper", "live"):
        raise HTTPException(status_code=400, detail="mode must be 'paper' or 'live'")

    with _bot_lock:
        proc = _bot_procs[mode]

    if proc is None or proc.poll() is not None:
        raise HTTPException(status_code=404, detail=f"No running {mode} bot found")

    try:
        proc.send_signal(signal.SIGTERM)
    except Exception:
        pass

    with _bot_lock:
        _bot_procs[mode]       = None
        _bot_start_times[mode] = None

    return {"ok": True, "mode": mode}


@app.get("/api/bot/logs/{mode}")
async def bot_logs(mode: str, lines: int = 100):
    """Return buffered log lines for one bot process."""
    if mode not in ("paper", "live"):
        raise HTTPException(status_code=400, detail="mode must be 'paper' or 'live'")
    with _bot_lock:
        buf = list(_bot_log_lines[mode])
    return {"mode": mode, "lines": buf[-lines:]}


async def _bot_stream_generator(mode: str):
    """SSE generator — streams new log lines from a running bot."""
    q = _bot_log_queues[mode]
    while True:
        try:
            line = q.get_nowait()
            yield f"data: {json.dumps({'line': line, 'mode': mode})}\n\n"
        except queue.Empty:
            state = _bot_state(mode)
            yield f"data: {json.dumps({'line': None, 'running': state['running']})}\n\n"
            await asyncio.sleep(0.8)


@app.get("/api/bot/stream/{mode}")
async def bot_stream(mode: str):
    """SSE stream of live log output for paper or live bot."""
    if mode not in ("paper", "live"):
        raise HTTPException(status_code=400, detail="mode must be 'paper' or 'live'")
    return StreamingResponse(
        _bot_stream_generator(mode),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────
# Backtesting endpoints
# ─────────────────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    start: str           # YYYY-MM-DD
    end: str             # YYYY-MM-DD
    symbols: List[str]   # ["AAPL", "MSFT"] or ["__ALL__"]
    capital: float = 300.0

# In-memory job store (one job at a time for simplicity)
_bt_lock   = threading.Lock()
_bt_status = {"state": "idle", "progress": "", "pct": 0}   # state: idle|running|done|error
_bt_result: dict = {}
_bt_trades: list = []
_bt_equity: list = []   # [{date, equity}]
_bt_log_q: "queue.Queue[str]" = queue.Queue()


def _run_backtest_thread(req: BacktestRequest) -> None:
    """Execute backtest in a background thread; write progress to _bt_log_q."""
    global _bt_status, _bt_result, _bt_trades, _bt_equity

    def _progress(msg: str, pct: int = -1) -> None:
        _bt_log_q.put(msg)
        with _bt_lock:
            _bt_status["progress"] = msg
            if pct >= 0:
                _bt_status["pct"] = pct

    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))

        import datetime as _dt
        import config as _cfg
        from backtest import Backtester

        symbols = _cfg.WATCHLIST if req.symbols == ["__ALL__"] else req.symbols

        _progress(f"Loading data for {len(symbols)} symbol(s)…", 5)

        start = _dt.date.fromisoformat(req.start)
        end   = _dt.date.fromisoformat(req.end)

        # Load backtest-specific algorithm overrides (independent from live config)
        bt_overrides = _load_backtest_overrides()

        bt = Backtester(
            symbols=symbols,
            start_date=start,
            end_date=end,
            initial_capital=req.capital,
            param_overrides=bt_overrides,
        )

        # Patch _load_data to emit progress messages
        _orig_load = bt._load_data
        def _patched_load():
            _progress("Fetching historical bars from Alpaca…", 10)
            _orig_load()
            _progress(f"Loaded {len(bt._data)} symbols. Running simulation…", 30)
        bt._load_data = _patched_load

        # Patch _process_day to emit periodic progress
        _orig_process = bt._process_day
        _day_count = [0]
        _total_days = [(end - start).days or 1]
        def _patched_process(date):
            _orig_process(date)
            _day_count[0] += 1
            if _day_count[0] % 20 == 0:
                pct = 30 + int((_day_count[0] / _total_days[0]) * 60)
                _progress(f"Replaying {date} … ({_day_count[0]} days done)", min(pct, 89))
        bt._process_day = _patched_process

        _progress("Starting backtest run…", 30)
        stats = bt.run()

        if not stats:
            raise ValueError("Backtester returned no results (no trades / no data)")

        _progress("Computing statistics…", 92)

        # Equity curve for chart
        equity_points = [
            {"date": str(d), "equity": round(e, 2)}
            for d, e in bt.equity_curve
        ]

        # Trade log
        trades = [
            {
                "symbol":      t.symbol,
                "entry_date":  str(t.entry_date),
                "exit_date":   str(t.exit_date),
                "entry_price": round(t.entry_price, 2),
                "exit_price":  round(t.exit_price, 2),
                "qty":         round(t.qty, 3),
                "pnl":         round(t.pnl, 2),
                "pnl_pct":     round(t.pnl_pct * 100, 2),
                "hold_days":   t.hold_days,
                "exit_reason": t.exit_reason,
            }
            for t in bt.trade_log
        ]

        # Save chart
        try:
            bt.save_chart()
        except Exception:
            pass

        with _bt_lock:
            _bt_result.clear()
            _bt_result.update(stats)
            _bt_trades.clear()
            _bt_trades.extend(trades)
            _bt_equity.clear()
            _bt_equity.extend(equity_points)
            _bt_status["state"] = "done"
            _bt_status["pct"]   = 100

        _progress("✓ Backtest complete.", 100)

    except Exception as exc:
        with _bt_lock:
            _bt_status["state"] = "error"
            _bt_status["progress"] = f"Error: {exc}"
        _bt_log_q.put(f"ERROR: {exc}")


@app.post("/api/backtest/run")
async def start_backtest(req: BacktestRequest):
    """Launch a backtest job (one at a time)."""
    with _bt_lock:
        if _bt_status["state"] == "running":
            raise HTTPException(status_code=409, detail="A backtest is already running")
        _bt_status.update({"state": "running", "progress": "Starting…", "pct": 0})
        _bt_result.clear()
        _bt_trades.clear()
        _bt_equity.clear()

    # Drain old log messages
    while not _bt_log_q.empty():
        try:
            _bt_log_q.get_nowait()
        except queue.Empty:
            break

    t = threading.Thread(target=_run_backtest_thread, args=(req,), daemon=True)
    t.start()
    return {"ok": True}


@app.get("/api/backtest/status")
async def backtest_status():
    """Poll current backtest job state."""
    with _bt_lock:
        return dict(_bt_status)


@app.get("/api/backtest/result")
async def backtest_result():
    """Return final stats + trade log + equity curve once job is done."""
    with _bt_lock:
        if _bt_status["state"] not in ("done", "error"):
            raise HTTPException(status_code=425, detail="Backtest not finished yet")
        return {
            "stats":  dict(_bt_result),
            "trades": list(_bt_trades),
            "equity": list(_bt_equity),
        }


async def _bt_stream_generator():
    """SSE stream for live backtest log lines."""
    while True:
        try:
            msg = _bt_log_q.get_nowait()
            with _bt_lock:
                state = _bt_status["state"]
                pct   = _bt_status["pct"]
            payload = {"msg": msg, "state": state, "pct": pct}
            yield f"data: {json.dumps(payload)}\n\n"
        except queue.Empty:
            with _bt_lock:
                state = _bt_status["state"]
                pct   = _bt_status["pct"]
            yield f"data: {json.dumps({'msg': None, 'state': state, 'pct': pct})}\n\n"
            await asyncio.sleep(0.5)
            if state in ("done", "error"):
                break


@app.get("/api/backtest/stream")
async def backtest_stream():
    """SSE stream — pushes log lines while the backtest runs."""
    return StreamingResponse(
        _bt_stream_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
