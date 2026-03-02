# Swing Trading Bot (Alpaca API)

A fully automated **swing-trading bot** that uses the Alpaca brokerage API to scan for setups, size positions, execute trades, and manage exits — all while **guaranteeing zero day trades** so you never trigger the Pattern Day Trader rule.

The strategy is now **regime-adaptive** with three states:
- `bull`: normal long swing strategy
- `risk_off`: no new entries (legacy safety behavior)
- `bear`: dedicated bear playbook using inverse/defensive ETFs (still long-only)

---

## How It Works

```
Watchlist (60+ liquid stocks/ETFs)
        │
        ▼
   ┌──────────┐
   │ Screener │  price, volume, and liquidity filters
   └────┬─────┘
        │  candidates
        ▼
   ┌──────────┐
   │ Strategy │  EMA crossover · RSI pullback · MACD · ADX · Volume
   └────┬─────┘
        │  BUY / SELL signals
        ▼
   ┌──────────┐
   │ PDT Guard│  blocks any same-day round trip
   └────┬─────┘
        │
        ▼
     ┌──────────┐
     │ Risk Mgr │  ATR stops · dynamic position sizing · PDT-safe exits
     └────┬─────┘
        │
        ▼
   ┌──────────┐
   │ Executor │  bracket orders (SL + TP) via Alpaca API
   └──────────┘
```

### Entry Criteria
Entries use a **scoring model** (not a strict all-conditions gate). Typical factors include:
- Trend alignment (EMA-50/EMA-200), EMA crossover, and EMA slope
- RSI pullback quality, MACD behavior, ADX, and volume confirmation
- Momentum ranking and optional weekly trend alignment
- In `bear` regime, a dedicated bear score is used for inverse/defensive ETFs

### Exit Criteria
Exits are **layered**:
- Hard exits (immediate): stop-loss/take-profit or major trend break
- Soft exits (confirmation-based): overbought, bearish crossover, momentum decay, dead-money conditions
- Trailing stops tighten as gains increase (separate bull vs bear parameters)

### PDT Protection
The bot enforces a **zero day-trade policy**:
- A local ledger records every buy with its fill date
- Before any sell, the guard checks the position wasn't opened today
- Minimum hold period: **2 calendar days** (configurable)
- The PDT counter will always stay at **0**

---

## Project Structure

```
Trader/
├── backtest.py       # Historical simulator and performance report
├── main.py           # Entry point & scheduler
├── config.py         # All tunable parameters
├── broker.py         # Alpaca API wrapper
├── indicators.py     # Technical indicators (EMA, RSI, MACD, ATR, etc.)
├── strategy.py       # Entry & exit signal logic
├── screener.py       # Stock universe filtering
├── pdt_guard.py      # Pattern Day Trader protection
├── risk_manager.py   # Position sizing & risk limits
├── executor.py       # Trade execution orchestrator
├── run_validation.py # Benchmark validator (bear + bull windows)
├── logger.py         # Logging setup
├── requirements.txt  # Python dependencies
├── .env.example      # API key template
└── logs/             # Runtime logs & PDT ledger
```

---

## Quick Start

### 1. Install dependencies

```bash
cd Trader
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and add your Alpaca API keys:
```
ALPACA_API_KEY=PKxxxxxxxxxxxxxxxxxxxx
ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TRADING_MODE=paper
```

> **Start with `paper` mode** to test with simulated money before going live.

Get keys at: https://app.alpaca.markets/

### 3. Run the bot

```bash
# Continuous mode (runs every 30 min during market hours)
python main.py

# Single cycle (scan once and exit)
python main.py --once

# Check account status
python main.py --status

# Run benchmark validation windows (bear + bull)
python run_validation.py
```

---

## Configuration

All parameters are in [`config.py`](config.py). Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_OPEN_POSITIONS` | 10 | Max simultaneous positions |
| `MAX_POSITION_PCT` | 15% | Max equity per position |
| `MAX_LOSS_PER_TRADE_PCT` | 2% | Risk budget per trade |
| `ATR_STOP_MULTIPLIER` | 3.0 | Stop loss = Entry − ATR×multiplier |
| `ATR_PROFIT_MULTIPLIER` | 6.0 | Take profit = Entry + ATR×multiplier |
| `TRAILING_STOP_PCT` | 4% | Trailing stop on winners |
| `MIN_HOLD_CALENDAR_DAYS` | 2 | PDT safety — minimum hold |
| `SCAN_INTERVAL_MINUTES` | 30 | How often to scan for entries |
| `CHECK_EXITS_MINUTES` | 15 | How often to check exits |

### Market Regime Controls
Key bear-adaptation settings are in `config.py`:
- `MARKET_REGIME_ENABLED`, `MARKET_REGIME_SYMBOL`
- `MARKET_REGIME_CONFIRM_DAYS`, `MARKET_REGIME_EMA_BUFFER`
- `BEAR_REGIME_DRAWDOWN_LOOKBACK`, `BEAR_REGIME_DRAWDOWN_TRIGGER`
- `BEAR_STRATEGY_ENABLED`, `BEAR_WATCHLIST`, `BEAR_ENTRY_SCORE_THRESHOLD`
- `BEAR_ATR_STOP_MULTIPLIER`, `BEAR_ATR_PROFIT_MULTIPLIER`
- `BEAR_TRAILING_STOP_*` and `BEAR_MAX_POSITION_SCALE`

### Optional Robustness Filters
Additional generalization controls are available:
- `BREADTH_FILTER_ENABLED`, `BREADTH_MIN_BULL`, `BREADTH_MAX_BEAR`, `BREADTH_MIN_SYMBOLS`
- `EARNINGS_FILTER_ENABLED`, `EARNINGS_CALENDAR_CSV`, `EARNINGS_BLACKOUT_DAYS_BEFORE`, `EARNINGS_BLACKOUT_DAYS_AFTER`

If you enable earnings filtering, copy `data/earnings_calendar.csv.example` to `data/earnings_calendar.csv`
and keep it updated with `symbol,date` rows (YYYY-MM-DD).

### Watchlist
The default watchlist includes ~60 liquid mid/large-cap stocks and ETFs. Edit `WATCHLIST` in `config.py` to customize.

---

## Risk Management

- **Per-trade risk**: sized so a stop-loss hit only costs ≤ 2% of equity
- **Position cap**: no single position exceeds 15% of equity (auto-adjusts for small accounts)
- **Portfolio cap**: max 10 concurrent positions (auto-adjusts for small accounts), max 95% buying-power usage
- **Bracket orders**: every entry has an automatic stop-loss and take-profit
- **Trailing stops**: adaptive and regime-aware (bull vs bear settings)
- **PDT lock**: zero same-day round trips — ever

---

## Logs

Logs are written to both the console and `logs/trader.log`. The PDT ledger is stored at `logs/pdt_ledger.json`.

---

## Disclaimer

This bot is for **educational and paper-trading purposes**. Trading stocks involves risk of loss. Past performance of any strategy does not guarantee future results. Use at your own risk.
