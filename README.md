# Stock Swing Trading Bot (Alpaca API)

A fully automated **swing-trading bot** powered by the Alpaca brokerage API. It combines a multi-factor technical scoring system with a **Gradient Boosting Machine (GBM) ML model** to scan for setups, size positions, execute trades, and manage exits — while enforcing PDT rules, minimum hold times, and stagnation exits.

Start the dashboard:
python -m uvicorn dashboard.server:app --host 127.0.0.1 --port 8000 --reload

---

## Architecture

```
Watchlist (60+ liquid stocks/ETFs)
        │
        ▼
   ┌──────────────┐
   │   Screener   │  price, volume, and liquidity filters
   └──────┬───────┘
          │  candidates
          ▼
   ┌──────────────────────────────────────────────────┐
   │               Strategy (Two-Gate Entry)          │
   │                                                  │
   │  Gate 1 – Hand-crafted scoring system            │
   │    · EMA trend/crossover  · RSI pullback         │
   │    · MACD  · ADX  · Volume surge                 │
   │    · Bollinger Bands  · Stochastic               │
   │    · Momentum ranking  · S/R awareness           │
   │    · Market regime (SPY)  · VIX fear filter      │
   │                                                  │
   │  Gate 2 – GBM ML model (when ML_ENABLED=True)   │
   │    · Predicts probability of ≥3% gain in 5 bars │
   │    · Must exceed ML_ENTRY_THRESHOLD (default 40%)│
   └──────┬───────────────────────────────────────────┘
          │  BUY signals
          ▼
   ┌──────────────┐
   │  PDT Guard   │  enforces day-trade limits & minimum hold time
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │  Risk Manager│  ATR-based stops · 2% risk per trade · position sizing
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │   Executor   │  bracket orders (SL + TP) via Alpaca API
   └──────────────┘
```

---

## Entry Logic

### Hand-crafted Scoring (Gate 1)

Entry signals are **scored** out of ~14 base points. A trade is only considered if the score meets the configurable threshold (`ENTRY_SCORE_THRESHOLD`, default **5**).

| Points | Condition | Purpose |
|--------|-----------|---------|
| +2 | Price > EMA-50 | Core uptrend |
| +1 | Price > EMA-200 | Bull regime on the stock |
| +2 | EMA-9 crosses above EMA-21 | Bullish crossover timing |
| +1 | EMA-50 slope is rising | Trend quality |
| +2 | RSI in 30–50 zone | Pullback, not overbought |
| +1 | RSI in 50–60 zone | Mid-range entry |
| +1 | MACD histogram positive or turning up | Momentum |
| +1 | Volume ≥ 20-day average | Conviction |
| +1 | ADX > 20 | Trend strength |
| +1 | Price ≤ Bollinger Band midline | Near lower band |
| +1 | Stochastic %K crosses above %D from oversold | Fine timing |
| +2 | Top-quartile 20-day momentum | Buy the strongest |
| +1 | Weekly trend agrees (if enabled) | Multi-timeframe |
| +1 | Price near support level | S/R awareness |
| −1 | Price near resistance while below EMA-50 | S/R penalty |

**Pre-flight filters** (automatic disqualifiers before scoring):
- Gap-up > 8% since prior close → skip (exhaustion risk)
- SPY below its 200-day SMA → skip new longs (bear market regime)
- VIXY above 35 → halt all new longs (fear / volatility spike)
- VIXY above 22 → reduce position size to 50%

### ML Model (Gate 2)

When `ML_ENABLED = True`, a **Gradient Boosting classifier** acts as a second gate:

| Config | Default | Meaning |
|--------|---------|---------|
| `ML_ENABLED` | `True` | Enable/disable the ML model |
| `ML_BLEND_MODE` | `"gate"` | `"gate"` = score + ML both required; `"replace"` = ML only |
| `ML_MIN_SCORE` | `3` | Minimum hand-crafted score before consulting ML (gate mode) |
| `ML_ENTRY_THRESHOLD` | `0.40` | Minimum GBM win-probability to enter |
| `ML_FORWARD_BARS` | `5` | Forward bars used to define a winning trade for labels |
| `ML_MIN_GAIN_PCT` | `3%` | Min gain in forward bars to label a bar as a winner |
| `ML_TRAINING_MONTHS` | `24` | Months of history used to train the model |

**Gate mode (default) flow:**
1. Hand-crafted score must reach `ML_MIN_SCORE` (3 pts) — fast pre-filter
2. GBM probability must reach `ML_ENTRY_THRESHOLD` (40%) — learned pattern recognition
3. Both gates must pass for a BUY signal to fire

If the ML model file is missing or unavailable, the bot **automatically falls back** to the hand-crafted score-only path.

---

## Exit Logic

Exits are **layered** to avoid shaking out positions on normal volatility.

### Hard Exits (1 signal fires immediately)
| Condition | Reason |
|-----------|--------|
| Price below both EMA-50 **and** EMA-200 | Trend destroyed |
| ATR-based stop-loss hit | Risk management |
| ATR-based take-profit hit | Lock in gains |

### Soft Exits (2+ signals required to fire)
| Condition | Reason |
|-----------|--------|
| RSI ≥ 80 | Truly overbought |
| Bearish EMA-9/21 crossover | Trend reversing |
| MACD histogram negative for 2+ bars (accelerating) | Momentum fading |
| Price below EMA-50 (still above 200) | Trend weakening |
| Dead money: held ≥ 7 days with < 1.5% total move | Free up capital |
| Momentum decay: −5% 20-day return + below EMA-50 | Losing trade |

### Trailing Stop
- Activates after **+6% gain** — trails 4% below the peak price
- Tightens to **3%** once profit exceeds **+15%** (lock in large winners)

---

## PDT Protection

The bot maintains a **persistent JSON ledger** (`logs/pdt_ledger.json`) and enforces Pattern Day Trader rules.

| Config | Default | Meaning |
|--------|---------|---------|
| `MAX_DAY_TRADES_ALLOWED` | `3` | Max day trades allowed in the rolling window (0 = full lock) |
| `PDT_LOOKBACK_DAYS` | `5` | Rolling window size in business days |
| `MIN_HOLD_CALENDAR_DAYS` | `0` | Minimum calendar days before a non-day-trade sell is allowed |

**60-minute minimum hold**: Even when a day trade is permitted by the counter, positions must be held for at least **60 minutes** from the buy fill time. Sells are blocked until this minimum is met.

**Re-entry cooldown**: After selling a symbol, the bot will not re-buy it for `RE_ENTRY_COOLDOWN_DAYS` (**5 days**) to prevent churn and reverse-day-trade patterns.

**Startup reconciliation**: On every startup, the PDT guard fetches Alpaca's actual closed-order history and reconciles the local ledger — ensuring nothing is lost across bot restarts or crashes.

---

## Stagnation (Dead-Money) Exit

Positions that go nowhere tie up capital and are exited automatically:

| Config | Default | Meaning |
|--------|---------|---------|
| `DEAD_MONEY_DAYS` | `7` | Days after entry before stagnation is checked |
| `DEAD_MONEY_THRESHOLD` | `1.5%` | If total price move < 1.5% after 7 days → exit |

This counts as a soft-exit signal; one more soft signal (e.g. RSI overbought, MACD declining) is enough to trigger the sell.

---

## Market Regime & VIX Filter

| Feature | Config | Behaviour |
|---------|--------|-----------|
| Bull/Bear regime | `MARKET_REGIME_ENABLED = True` | SPY below 200-day SMA → no new longs |
| VIX halt | `VIX_HALT_THRESHOLD = 35.0` | VIXY > 35 → halt all entries |
| VIX size reduction | `VIX_REDUCE_THRESHOLD = 22.0` | VIXY > 22 → cut position size to 50% |
| Inverse ETF mode | `INVERSE_ETF_MODE_ENABLED = True` | Bear market → trade SQQQ/SPXS/SOXS/SH/PSQ |

VIXY (a VIX-futures ETF) is used as the fear gauge because Alpaca does not carry the `^VIX` index directly.

---

## Small-Account Mode

When equity falls below `SMALL_ACCOUNT_THRESHOLD` (**$2,000**), parameters auto-adjust:

| Parameter | Normal | Small-Account |
|-----------|--------|---------------|
| Max open positions | 10 | 3 |
| Max position size | 15% | 45% |
| Risk per trade | 2% | 3% |
| ATR stop multiplier | 3.0× | 2.0× |
| ATR profit multiplier | 7.0× | 4.0× |

---

## Project Structure

```
Stock-Swing-Trading-Bot/
├── main.py               # Entry point & scheduler
├── config.py             # All tunable parameters
├── broker.py             # Alpaca API wrapper
├── indicators.py         # Technical indicators (EMA, RSI, MACD, ATR, BB, Stoch, etc.)
├── strategy.py           # Entry & exit signal logic (scoring + ML integration)
├── screener.py           # Stock universe filtering
├── pdt_guard.py          # PDT protection, min hold, cooldown, reconciliation
├── risk_manager.py       # Position sizing & risk limits
├── executor.py           # Trade execution orchestrator
├── backtest.py           # Event-driven backtester (same strategy as live)
├── ml_model.py           # GBM model wrapper (train, save, predict)
├── ml_features.py        # Feature extraction for the ML model
├── logger.py             # Logging setup
├── requirements.txt      # Python dependencies
├── config_overrides.json # Dashboard/UI config overrides (auto-generated)
├── .env                  # API keys & trading mode (git-ignored)
├── dashboard/            # Optional web dashboard
│   ├── server.py
│   └── static/index.html
└── logs/
    ├── trader.log         # Runtime log
    ├── pdt_ledger.json    # PDT buy-date tracker, day-trade history, sell dates
    └── backtest_equity.png
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file:
```env
# Paper trading (default — always start here)
ALPACA_API_KEY=PKxxxxxxxxxxxxxxxxxxxx
ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TRADING_MODE=paper

# Live trading (optional — separate Alpaca live-account key pair)
LIVE_ALPACA_API_KEY=PKxxxxxxxxxxxxxxxxxxxx
LIVE_ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Get keys at: https://app.alpaca.markets/

### 3. (Optional) Train the ML model

```bash
python ml_model.py --train --symbols AAPL MSFT NVDA SPY QQQ --months 24
```

Downloads 24 months of history, builds features, trains the GBM classifier, and saves `ml_model.pkl`. If you skip this step the bot runs in score-only mode.

### 4. Run the bot

```bash
# Continuous mode (scans every 30 min during market hours)
python main.py

# Single scan cycle and exit
python main.py --once

# Print account & position status
python main.py --status
```

---

## Backtesting

The backtester uses the **exact same `strategy.check_entry` function** as live trading, including the ML model and SPY/VIXY macro context.

```bash
# Default: SPY, 1 year
python backtest.py

# Specific symbols & lookback
python backtest.py --symbols AAPL MSFT NVDA --months 6

# Custom date range
python backtest.py --start 2024-01-01 --end 2025-01-01 --symbols QQQ

# Full watchlist, 12 months
python backtest.py --all
```

**What the backtest reports:**
- Total return & annualised return
- Max drawdown
- Sharpe ratio
- Win rate & profit factor
- Average hold days
- Per-trade log
- Equity-curve chart saved to `logs/backtest_equity.png`

**ML in backtesting**: When `ML_ENABLED = True`, the backtester passes SPY and VIXY context slices to the ML model exactly as live trading does. Falls back to score-only if the model file is unavailable.

---

## Configuration Reference

All parameters are in [`config.py`](config.py). They can be overridden at runtime via:
- `config_overrides.json` (written by the dashboard UI)
- `CFG_OVERRIDE_<KEY>=<value>` environment variables (highest priority)

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRADING_MODE` | `"paper"` | `"paper"` or `"live"` |
| `ENTRY_SCORE_THRESHOLD` | `5` | Min hand-crafted score to enter |
| `ML_ENABLED` | `True` | Enable GBM second gate |
| `ML_ENTRY_THRESHOLD` | `0.40` | Min ML win-probability |
| `ML_MIN_SCORE` | `3` | Min score before consulting ML |
| `ML_BLEND_MODE` | `"gate"` | `"gate"` or `"replace"` |
| `ATR_STOP_MULTIPLIER` | `3.0` | Stop = entry − 3×ATR |
| `ATR_PROFIT_MULTIPLIER` | `7.0` | Target = entry + 7×ATR |
| `TRAILING_STOP_ACTIVATE_PCT` | `6%` | Activate trailing stop at +6% gain |
| `TRAILING_STOP_PCT` | `4%` | Trailing stop width |
| `TRAILING_STOP_TIGHT_ACTIVATE` | `15%` | Tighten at +15% gain |
| `TRAILING_STOP_TIGHT_PCT` | `3%` | Tightened trailing stop width |
| `MAX_OPEN_POSITIONS` | `10` | Max simultaneous positions |
| `MAX_POSITION_PCT` | `15%` | Max equity per position |
| `MAX_LOSS_PER_TRADE_PCT` | `2%` | Risk budget per trade |
| `MAX_DAY_TRADES_ALLOWED` | `3` | PDT day-trade limit (0 = disabled) |
| `PDT_LOOKBACK_DAYS` | `5` | Rolling PDT window (business days) |
| `MIN_HOLD_CALENDAR_DAYS` | `0` | Minimum calendar days before selling |
| `DEAD_MONEY_DAYS` | `7` | Days before stagnation exit check |
| `DEAD_MONEY_THRESHOLD` | `1.5%` | Min move to avoid stagnation exit |
| `RE_ENTRY_COOLDOWN_DAYS` | `5` | Days before re-buying a sold symbol |
| `MARKET_REGIME_ENABLED` | `True` | Skip longs in bear markets |
| `VIX_FILTER_ENABLED` | `True` | VIXY-based fear filter |
| `VIX_HALT_THRESHOLD` | `35.0` | VIXY > 35 → halt all entries |
| `VIX_REDUCE_THRESHOLD` | `22.0` | VIXY > 22 → half position size |
| `INVERSE_ETF_MODE_ENABLED` | `True` | Trade inverse ETFs in bear markets |
| `SCAN_INTERVAL_MINUTES` | `30` | Entry scan frequency |
| `CHECK_EXITS_MINUTES` | `15` | Exit check frequency |
| `MOMENTUM_LOOKBACK` | `20` | Days to measure momentum |
| `MOMENTUM_TOP_PCT` | `50%` | Only consider top 50% by momentum |
| `GAP_UP_MAX_PCT` | `8%` | Skip if gapped up > 8% |
| `DYNAMIC_THRESHOLD_ENABLED` | `False` | Lower score bar in strong markets |
| `SMALL_ACCOUNT_THRESHOLD` | `$2,000` | Below this → small-account rules apply |

---

## Risk Management Summary

| Rule | Value |
|------|-------|
| Risk per trade | ≤ 2% of equity |
| Max position size | 15% of equity |
| Max open positions | 10 |
| Max portfolio exposure | 95% of buying power |
| Max portfolio risk | 6% total at risk |
| Stop loss | Entry − 3×ATR |
| Take profit | Entry + 7×ATR |
| Trailing stop | Activates at +6%, trails 4% |
| Trailing stop (tight) | Tightens to 3% above +15% gain |
| PDT day trades | Up to 3 per 5-business-day window |
| Min intraday hold | 60 minutes |
| Stagnation exit | < 1.5% move after 7 days |
| Re-entry cooldown | 5 days after selling |

---

## Logs

| File | Contents |
|------|----------|
| `logs/trader.log` | Full runtime log (level configurable via `LOG_LEVEL`) |
| `logs/pdt_ledger.json` | Buy dates, day-trade history, sell dates |
| `logs/backtest_equity.png` | Equity curve from the last backtest |

---

## Disclaimer

This project is provided for **educational and research purposes only**. Trading stocks and ETFs involves substantial risk of loss. Backtest results do not guarantee future performance. Always paper-trade a strategy before risking real capital. Use at your own risk.
