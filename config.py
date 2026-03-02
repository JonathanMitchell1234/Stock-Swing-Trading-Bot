"""
Configuration for the Swing Trading Bot.
All tunable parameters in one place.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Alpaca API
# ─────────────────────────────────────────────
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
TRADING_MODE = os.getenv("TRADING_MODE", "paper")  # "paper" or "live"

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"

BASE_URL = PAPER_BASE_URL if TRADING_MODE == "paper" else LIVE_BASE_URL

# ─────────────────────────────────────────────
# Account Size & Fractional Shares
# ─────────────────────────────────────────────
FRACTIONAL_SHARES = True            # use fractional shares (Alpaca supports)
SMALL_ACCOUNT_THRESHOLD = 2_000     # below this $ → small-account rules kick in

# ─────────────────────────────────────────────
# PDT Protection  (Pattern Day Trader Guard)
# ─────────────────────────────────────────────
# Accounts under $25k are limited to 3 day trades per 5 rolling business days.
# We set a hard cap of 0 day trades — the bot NEVER opens and closes
# the same symbol on the same calendar day.
MAX_DAY_TRADES_ALLOWED = 0          # hard-lock: no same-day round trips
PDT_LOOKBACK_DAYS = 5               # rolling window (business days)
MIN_HOLD_CALENDAR_DAYS = 2          # minimum days to hold before selling

# ─────────────────────────────────────────────
# Strategy – Technical Indicators
# ─────────────────────────────────────────────
# Trend EMAs
EMA_FAST = 9
EMA_SLOW = 21
EMA_TREND = 50          # long-term trend filter
EMA_LONG = 200          # market regime / super-trend filter

# RSI
RSI_PERIOD = 14
RSI_OVERSOLD = 30       # buy zone  (widened for more entries)
RSI_OVERBOUGHT = 80     # sell zone (raised to let winners run)

# MACD (standard 12-26-9)
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# ATR (for stops and position sizing)
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 3.0    # stop loss = entry − ATR * mult (wide to survive noise)
ATR_PROFIT_MULTIPLIER = 6.0  # take profit = entry + ATR * mult (let winners run)

# Volume confirmation
VOLUME_SMA_PERIOD = 20
VOLUME_SURGE_FACTOR = 1.0  # 1.0 = normal volume ok (removed as hard gate)

# ── Scoring system – entry signals are scored, not binary ────
ENTRY_SCORE_THRESHOLD = 5      # minimum score (out of ~14 base) to trigger a buy

# ── Momentum ranking (buy the strongest stocks) ─────────────
MOMENTUM_LOOKBACK = 20          # days to measure momentum (rate of change)
MOMENTUM_TOP_PCT = 0.50         # only consider top 50% by momentum
MOMENTUM_SCORE_WEIGHT = 2       # bonus points for top-quartile momentum

# ── Trend quality (EMA slope must be rising) ────────────────
EMA_SLOPE_PERIOD = 5            # bars to measure EMA-50 slope direction

# ── Dead-money exit: sell stagnant positions ────────────────
DEAD_MONEY_DAYS = 10            # if position flat for this many trading days
DEAD_MONEY_THRESHOLD = 0.02     # has moved less than 2% total

# ── Time stop: if momentum never materializes, free capital ─
TIME_STOP_ENABLED = False
TIME_STOP_DAYS_BULL = 12
TIME_STOP_MIN_PEAK_PROFIT_BULL = 0.015
TIME_STOP_DAYS_BEAR = 7
TIME_STOP_MIN_PEAK_PROFIT_BEAR = 0.01

# ── Cooldown: don't re-enter a symbol within N days of exit ──
RE_ENTRY_COOLDOWN_DAYS = 5

# ── Market regime filter (SPY-based) ────────────────────────
MARKET_REGIME_ENABLED = True   # classify bull/bear regime from market proxy
MARKET_REGIME_SYMBOL = "SPY"   # index proxy
# Bull/Bear regime confirmation (reduces whipsaws)
MARKET_REGIME_CONFIRM_DAYS = 4
MARKET_REGIME_EMA_BUFFER = 0.005  # 0.5% buffer around EMA-200
BEAR_REGIME_DRAWDOWN_LOOKBACK = 252
BEAR_REGIME_DRAWDOWN_TRIGGER = 0.25  # require >=25% drawdown from recent high

# Bear-mode strategy (long-only, no shorting)
BEAR_STRATEGY_ENABLED = True
BEAR_ENTRY_SCORE_THRESHOLD = 6
BEAR_ATR_STOP_MULTIPLIER = 1.8
BEAR_ATR_PROFIT_MULTIPLIER = 3.2
BEAR_RSI_OVERBOUGHT_EXIT = 75
BEAR_DEAD_MONEY_DAYS = 6
BEAR_DEAD_MONEY_THRESHOLD = 0.015
BEAR_MAX_POSITION_SCALE = 0.85
BEAR_MAX_POSITIONS_CAP = 2

# Instruments used when the market is in bear regime.
# Prefer inverse/defensive ETFs so we can stay long-only and PDT-safe.
BEAR_WATCHLIST = [
    "SH", "SDS", "SPXU",  # inverse S&P 500
    "PSQ", "SQQQ",         # inverse Nasdaq 100
    "DOG",                   # inverse Dow
    "SOXS",                  # inverse semiconductors
    "XLE", "XLP", "XLU",  # defensive/relative-strength sectors
    "GLD", "TLT",           # risk-off assets
]

# ── Multi-timeframe confirmation (weekly trend) ────────────
WEEKLY_TREND_ENABLED = False    # disabled for small accounts (blocks momentum plays)
WEEKLY_EMA_FAST = 10            # ~10-week EMA (approximately 50-day)
WEEKLY_EMA_SLOW = 40            # ~40-week EMA (approximately 200-day)
WEEKLY_TREND_BONUS = 0          # no scoring bonus; used as hard filter instead

# ── Volatility regime (adaptive sizing via realized vol) ────
VOL_REGIME_ENABLED = False      # disabled for testing
REALIZED_VOL_WINDOW = 20        # days to measure realized volatility
HIGH_VOL_THRESHOLD = 0.30       # annualized vol above 30% = high vol
LOW_VOL_THRESHOLD = 0.12        # annualized vol below 12% = low vol
HIGH_VOL_SIZE_SCALE = 0.80      # reduce position to 80% in high vol
LOW_VOL_SIZE_SCALE = 1.15       # increase position to 115% in low vol

# ── Sector exposure limits ─────────────────────────────────
MAX_PER_SECTOR = 99             # effectively disabled for testing
SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "AMZN": "Tech",
    "META": "Tech", "NVDA": "Tech", "AMD": "Tech", "TSM": "Tech",
    "CRM": "Tech", "ADBE": "Tech", "NFLX": "Tech", "QCOM": "Tech",
    "INTC": "Tech", "AVGO": "Tech", "MU": "Tech",
    "PLTR": "Tech", "SOFI": "Fintech", "MARA": "Crypto", "HOOD": "Fintech",
    "SNAP": "Tech", "U": "Tech",
    "JPM": "Finance", "BAC": "Finance", "GS": "Finance", "MS": "Finance",
    "V": "Finance", "MA": "Finance", "AXP": "Finance",
    "C": "Finance", "SCHW": "Finance",
    "JNJ": "Health", "UNH": "Health", "PFE": "Health", "ABBV": "Health",
    "MRK": "Health", "LLY": "Health", "TMO": "Health",
    "WMT": "Consumer", "COST": "Consumer", "HD": "Consumer",
    "NKE": "Consumer", "SBUX": "Consumer", "MCD": "Consumer", "DIS": "Consumer",
    "F": "Consumer", "RIVN": "Consumer",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "SLB": "Energy", "EOG": "Energy",
    "CAT": "Industrial", "DE": "Industrial", "BA": "Industrial",
    "GE": "Industrial", "HON": "Industrial", "UNP": "Industrial",
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "XLF": "ETF",
    "XLE": "ETF", "XLK": "ETF", "TQQQ": "ETF", "SOXL": "ETF",
    "ARKK": "ETF", "VTI": "ETF", "VOO": "ETF", "DIA": "ETF",
}

# ── Support / Resistance ───────────────────────────────────
SR_LOOKBACK = 20                # bars to detect swing highs/lows
SR_RESISTANCE_BUFFER = 0.015    # penalize entries within 1.5% of resistance
SR_SUPPORT_BONUS = 0            # no scoring bonus from S/R (used for info only)

# ── Gap filter: avoid chasing exhaustion gaps ──────────────
GAP_UP_MAX_PCT = 0.08           # skip if today gapped up > 8%

# ── Entry quality / robustness filters ─────────────────────
# Require enough listing history to avoid unstable newly listed names.
MIN_BARS_HISTORY_ENTRY = 120
# Skip overly volatile names where ATR is too large relative to price.
MAX_ATR_PCT_BULL = 0.15
MAX_ATR_PCT_BEAR = 0.20
# Require a minimum reward:risk profile before entering.
MIN_RR_RATIO_BULL = 1.8
MIN_RR_RATIO_BEAR = 1.5

# ── Market breadth filter (optional) ───────────────────────
# Breadth = fraction of symbols trading above EMA-50.
BREADTH_FILTER_ENABLED = True
BREADTH_MIN_SYMBOLS = 20
BREADTH_MIN_BULL = 0.35
BREADTH_MAX_BEAR = 0.55

# ── Earnings blackout filter (optional) ────────────────────
# CSV columns expected: symbol,date  (date format: YYYY-MM-DD)
EARNINGS_FILTER_ENABLED = False
EARNINGS_CALENDAR_CSV = "data/earnings_calendar.csv"
EARNINGS_BLACKOUT_DAYS_BEFORE = 2
EARNINGS_BLACKOUT_DAYS_AFTER = 1

# ── Dynamic threshold: adjust score bar by market quality ──
DYNAMIC_THRESHOLD_ENABLED = False
DYNAMIC_THRESHOLD_ADJUSTMENT = 1  # +/- this amount
# In strong markets (SPY > EMA-50 and EMA-50 rising), lower threshold by 1
# In weak/choppy markets the threshold stays at base (no increase)

# ─────────────────────────────────────────────
# Risk Management
# ─────────────────────────────────────────────
MAX_OPEN_POSITIONS = 10           # hard cap (overridden for small accounts)
MAX_POSITION_PCT = 0.15           # max 15% of equity per position
MAX_PORTFOLIO_RISK_PCT = 0.06     # max 6% total portfolio at risk
MAX_LOSS_PER_TRADE_PCT = 0.02     # risk at most 2% of equity per trade
MAX_PORTFOLIO_EXPOSURE_PCT = 0.95  # never use more than 95% buying power
TRAILING_STOP_ACTIVATE_PCT = 0.06  # activate trailing stop after 6% gain
TRAILING_STOP_PCT = 0.04          # 4% trailing stop on winners

# ── Adaptive stop: tighten trailing stop as profit grows ─────
TRAILING_STOP_TIGHT_ACTIVATE = 0.15  # above 15% profit, tighten
TRAILING_STOP_TIGHT_PCT = 0.03       # to 3% trailing stop

# ── Bear-mode trailing stops (faster de-risking) ────────────
BEAR_TRAILING_STOP_ACTIVATE_PCT = 0.04
BEAR_TRAILING_STOP_PCT = 0.03
BEAR_TRAILING_STOP_TIGHT_ACTIVATE = 0.10
BEAR_TRAILING_STOP_TIGHT_PCT = 0.02

# ── Small-account overrides (auto-applied when equity < SMALL_ACCOUNT_THRESHOLD)
SMALL_MAX_OPEN_POSITIONS = 3       # concentrate with tiny capital
SMALL_MAX_POSITION_PCT = 0.45      # up to 45% per position (need size)
SMALL_MAX_LOSS_PER_TRADE_PCT = 0.03  # 3% risk (tolerate more to get in)
SMALL_ATR_STOP_MULTIPLIER = 2.0    # tighter stops to limit $ loss
SMALL_ATR_PROFIT_MULTIPLIER = 4.0  # corresponding tighter targets

# ─────────────────────────────────────────────
# Stock Universe / Screener Filters
# ─────────────────────────────────────────────
# Core watchlist – mixed price range for small + large accounts
WATCHLIST = [
    # Tech (mid-high price)
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "TSM",
    "CRM", "ADBE", "NFLX", "QCOM", "INTC", "AVGO", "MU",
    # Tech (affordable < $50)
    "PLTR", "SOFI", "MARA", "HOOD", "SNAP", "U",
    # Finance
    "JPM", "BAC", "GS", "MS", "V", "MA", "AXP",
    "C", "SCHW",  # affordable financials
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO",
    # Consumer / Retail
    "WMT", "COST", "HD", "NKE", "SBUX", "MCD", "DIS",
    "F", "RIVN",  # affordable consumer
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Industrial
    "CAT", "DE", "BA", "GE", "HON", "UNP",
    # ETFs (mixed price range)
    "SPY", "QQQ", "IWM", "XLF", "XLE", "XLK",
    "TQQQ", "SOXL", "ARKK", "VTI", "VOO", "DIA",
]

MIN_PRICE = 5.0           # lower floor for affordable stocks
MAX_PRICE = 1_500.0       # skip ultra-high-price names
MIN_AVG_VOLUME = 500_000  # relaxed for broader universe

# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────
BARS_LOOKBACK = 100       # how many daily bars to fetch for analysis
BAR_TIMEFRAME = "1Day"    # daily bars for swing trading

# ─────────────────────────────────────────────
# Scheduling
# ─────────────────────────────────────────────
SCAN_INTERVAL_MINUTES = 30    # re-scan for entries every N minutes
CHECK_EXITS_MINUTES = 15      # check exit signals every N minutes

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE = "logs/trader.log"


# ─────────────────────────────────────────────
# Dynamic helpers (adjust for account size)
# ─────────────────────────────────────────────
def get_max_positions(equity: float) -> int:
    """Scale max open positions to account size."""
    if equity < SMALL_ACCOUNT_THRESHOLD:
        return SMALL_MAX_OPEN_POSITIONS
    return MAX_OPEN_POSITIONS


def get_position_pct(equity: float) -> float:
    """Max position % scales up for smaller accounts."""
    if equity < SMALL_ACCOUNT_THRESHOLD:
        return SMALL_MAX_POSITION_PCT
    return MAX_POSITION_PCT


def get_risk_per_trade(equity: float) -> float:
    """Risk per trade % — slightly higher for small accounts."""
    if equity < SMALL_ACCOUNT_THRESHOLD:
        return SMALL_MAX_LOSS_PER_TRADE_PCT
    return MAX_LOSS_PER_TRADE_PCT


def get_atr_stop_mult(equity: float) -> float:
    """ATR stop multiplier — tighter for small accounts to limit $ loss."""
    if equity < SMALL_ACCOUNT_THRESHOLD:
        return SMALL_ATR_STOP_MULTIPLIER
    return ATR_STOP_MULTIPLIER


def get_atr_profit_mult(equity: float) -> float:
    """ATR profit multiplier — tighter targets for small accounts."""
    if equity < SMALL_ACCOUNT_THRESHOLD:
        return SMALL_ATR_PROFIT_MULTIPLIER
    return ATR_PROFIT_MULTIPLIER
