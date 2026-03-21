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
# Paper trading credentials (Alpaca paper account)
PAPER_ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY", "")
PAPER_ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

# Live trading credentials (separate key pair from Alpaca live account)
LIVE_ALPACA_API_KEY    = os.getenv("LIVE_ALPACA_API_KEY", "")
LIVE_ALPACA_SECRET_KEY = os.getenv("LIVE_ALPACA_SECRET_KEY", "")

TRADING_MODE = os.getenv("TRADING_MODE", "paper")  # "paper" or "live"

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL  = "https://api.alpaca.markets"

BASE_URL = PAPER_BASE_URL if TRADING_MODE == "paper" else LIVE_BASE_URL

# Active credentials — whichever mode is current
ALPACA_API_KEY    = PAPER_ALPACA_API_KEY    if TRADING_MODE == "paper" else LIVE_ALPACA_API_KEY
ALPACA_SECRET_KEY = PAPER_ALPACA_SECRET_KEY if TRADING_MODE == "paper" else LIVE_ALPACA_SECRET_KEY

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
MAX_DAY_TRADES_ALLOWED = 3          # allow up to 3 day trades per rolling window
PDT_LOOKBACK_DAYS = 5               # rolling window (business days)
MIN_HOLD_CALENDAR_DAYS = 0          # allow same-day sells (day trading enabled)

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
ATR_PROFIT_MULTIPLIER = 7.0  # raised from 6.0 — let winners run further (improves profit factor)

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
DEAD_MONEY_DAYS = 7             # reduced from 10 — free up capital faster
DEAD_MONEY_THRESHOLD = 0.015    # tightened from 2% — exit if < 1.5% move in 7 days
# After DEAD_MONEY_DAYS, how many soft signals are required to exit?
# Set to 1 so that a stale position only needs ONE additional soft trigger
# (e.g. below EMA-50, MACD declining) rather than two.
DEAD_MONEY_SOFT_GATES = 1
# Hard-exit a position that is BOTH stale (>= DEAD_MONEY_DAYS) AND losing
# (price < entry) AND below EMA-50 — no second signal needed at all.
STALE_LOSER_EXIT_ENABLED = True

# ── Cooldown: don't re-enter a symbol within N days of exit ──
RE_ENTRY_COOLDOWN_DAYS = 5

# ── Market regime filter (SPY SMA-200 based) ────────────────
MARKET_REGIME_ENABLED = True    # reject new long buys in bear markets
MARKET_REGIME_SYMBOL = "SPY"    # index proxy
# Bull = SPY above its 200-day SMA; Bear = below (uses SMA, not EMA,
# matching the industry-standard definition of a bull/bear market)

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

# ── VIX-based fear filter ─────────────────────────────────
# Uses VIXY (VIX proxy ETF) since Alpaca does not carry ^VIX directly.
# VIXY tracks short-term VIX futures and is a reliable fear gauge.
VIX_FILTER_ENABLED = True       # enable VIX-based position halting/sizing
VIX_SYMBOL = "VIXY"             # VIX proxy available on Alpaca
VIX_HALT_THRESHOLD = 35.0       # VIXY price above this → halt ALL new longs
VIX_REDUCE_THRESHOLD = 22.0     # VIXY price above this → cut position size in half
VIX_SIZE_SCALE = 0.50           # scale factor when VIXY > VIX_REDUCE_THRESHOLD

# ── Bear-market short-selling mode ─────────────────────────
# When the market regime turns bearish (SPY < EMA-200), the bot
# switches to short-selling stocks from the main WATCHLIST using
# an inverted scoring system (mirrors of the bull entry/exit logic).
# Alpaca supports short selling on margin accounts.
BEAR_SHORT_MODE_ENABLED = True    # enable short selling in bear markets
BEAR_SHORT_SIZE_SCALE = 0.60      # trade at 60% of normal size (shorts are riskier)
# Symbols that should NEVER be shorted (illiquid, hard to borrow, etc.)
SHORT_BLACKLIST = [
    "MARA", "RIVN",  # volatile / hard-to-borrow
]

# Minimum account equity required to short-sell (Alpaca margin requirement).
# If equity is below this threshold in a bear regime, the bot automatically
# switches to buying inverse ETFs from INVERSE_WATCHLIST instead of shorting.
SHORT_MIN_EQUITY = 2_000.0

# ── Inverse ETF mode (auto-activated when equity < SHORT_MIN_EQUITY) ──
# When enabled the bot buys inverse/leveraged-inverse ETFs in bear markets
# instead of short-selling, which requires no margin.
INVERSE_ETF_MODE_ENABLED = False   # static default; overridden at runtime by equity check
INVERSE_ETF_SIZE_SCALE = 0.60
INVERSE_WATCHLIST = [
    "SH",     # ProShares Short S&P 500
    "SDS",    # ProShares UltraShort S&P 500 (2×)
    "SPXU",   # ProShares UltraPro Short S&P 500 (3×)
    "PSQ",    # ProShares Short QQQ
    "QID",    # ProShares UltraShort QQQ (2×)
    "SQQQ",   # ProShares UltraPro Short QQQ (3×)
    "RWM",    # ProShares Short Russell 2000
    "SDOW",   # ProShares UltraPro Short Dow30 (3×)
    "SPXS",   # Direxion Daily S&P 500 Bear 3×
    "SOXS",   # Direxion Daily Semiconductor Bear 3×
]

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
    "VIXY": "ETF",   # VIX proxy used for the fear filter
    # Inverse ETFs (bear-mode alternatives to short-selling)
    "SH": "InverseETF", "SDS": "InverseETF", "SPXU": "InverseETF",
    "PSQ": "InverseETF", "QID": "InverseETF", "SQQQ": "InverseETF",
    "RWM": "InverseETF", "SDOW": "InverseETF", "SPXS": "InverseETF",
    "SOXS": "InverseETF",
}

# ── Bear-mode entry scoring overrides ─────────────────────
# These mirror the bull-mode thresholds but for short entries.
# RSI overbought zone for SHORT entry (stock is stretched, ready to drop)
RSI_SHORT_ENTRY_MIN = 65          # short entry zone: RSI 65-85
RSI_SHORT_ENTRY_MAX = 85
# RSI oversold zone triggers SHORT exit (cover — stock is bouncing)
RSI_SHORT_EXIT = 25               # cover if RSI drops below this
# Gap filter for shorts — skip if stock gapped DOWN too much (exhaustion)
GAP_DOWN_MAX_PCT = 0.08           # skip short if today gapped down > 8%

# ── Support / Resistance ───────────────────────────────────
SR_LOOKBACK = 20                # bars to detect swing highs/lows
SR_RESISTANCE_BUFFER = 0.015    # penalize entries within 1.5% of resistance
SR_SUPPORT_BONUS = 0            # no scoring bonus from S/R (used for info only)

# ── Gap filter: avoid chasing exhaustion gaps ──────────────
GAP_UP_MAX_PCT = 0.08           # skip if today gapped up > 8%

# ── Dynamic threshold: adjust score bar by market quality ──
DYNAMIC_THRESHOLD_ENABLED = False
DYNAMIC_THRESHOLD_ADJUSTMENT = 1  # +/- this amount
# In strong markets (SPY > EMA-50 and EMA-50 rising), lower threshold by 1
# In weak/choppy markets the threshold stays at base (no increase)

# ─────────────────────────────────────────────
# Order Execution
# ─────────────────────────────────────────────
# Use limit orders for entries instead of market orders.
# The limit price is set slightly above the last trade price so the
# order fills quickly while still guaranteeing a price ceiling.
USE_LIMIT_ORDERS = True             # True = limit buy, False = market buy
LIMIT_ORDER_OFFSET_PCT = 0.001      # limit price = last_price * (1 + offset), e.g. 0.1% above
LIMIT_ORDER_TIF = "day"             # time-in-force: "day" cancels at market close if unfilled

# Delay new entries for N minutes after market open to avoid the
# chaotic first-30-minute price discovery / spread-widening period.
MARKET_OPEN_DELAY_MINUTES = 30      # 0 = no delay (enter as soon as market opens)

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
DATA_FEED = "iex"         # "iex" (free) or "sip" (paid subscription)

# ─────────────────────────────────────────────
# Scheduling
# ─────────────────────────────────────────────
SCAN_INTERVAL_MINUTES = 30    # re-scan for entries every N minutes
CHECK_EXITS_MINUTES = 15      # check exit signals every N minutes

# ─────────────────────────────────────────────
# Machine Learning (GBM entry model)
# ─────────────────────────────────────────────
ML_ENABLED = True              # set True after training a model
ML_ENTRY_THRESHOLD = 0.40       # minimum GBM probability to enter (0-1)
ML_MIN_SCORE = 3                # minimum hand-crafted score before consulting GBM
ML_BLEND_MODE = "gate"          # "gate" = both score+ML must pass; "replace" = ML only
ML_FORWARD_BARS = 5             # forward bars for label generation
ML_MIN_GAIN_PCT = 0.03          # label = 1 if close ≥ entry × (1 + this) within forward bars
ML_TRAINING_MONTHS = 24         # months of history to use for training

# ── Recency weighting: give more importance to recent data ──
# Rows are up-weighted exponentially as they approach today.
# The half-life controls how fast the weight decays into the past:
#   e.g. 90 days → a bar from 90 days ago gets half the weight of today's bar.
# Set ML_RECENCY_WEIGHT_ENABLED = False to revert to uniform weights.
ML_RECENCY_WEIGHT_ENABLED = True
ML_RECENCY_HALFLIFE_DAYS   = 90   # half-life in calendar days
ML_RECENCY_MIN_WEIGHT      = 0.10  # floor: old bars never drop below this fraction of max weight

# ── Short GBM model (separate model for bear-mode shorts) ───
# Set ML_SHORT_ENABLED = True after training with: python ml_trainer_short.py
ML_SHORT_ENABLED = True           # enable the short GBM model in bear mode
ML_SHORT_ENTRY_THRESHOLD = 0.40   # minimum short GBM probability to enter
ML_SHORT_MIN_SCORE = 3            # minimum hand-crafted short score before consulting GBM
ML_SHORT_FORWARD_BARS = 5         # forward bars for short label generation
ML_SHORT_MIN_DROP_PCT = 0.03      # label = 1 if close ≤ entry × (1 - this)

# ── Inverse ETF GBM model (dedicated model for inverse ETF entries) ──
# Set ML_INVERSE_ENABLED = True after training with: python ml_trainer_inverse.py
ML_INVERSE_ENABLED = True             # enable the inverse ETF GBM model
ML_INVERSE_ENTRY_THRESHOLD = 0.40     # minimum inverse GBM probability to enter
ML_INVERSE_MIN_SCORE = 3              # minimum hand-crafted score before consulting GBM
ML_INVERSE_FORWARD_BARS = 5           # forward bars for inverse label generation
ML_INVERSE_MIN_GAIN_PCT = 0.03        # label = 1 if close ≥ entry × (1 + this)

# ─────────────────────────────────────────────
# NLP Sentiment (FinBERT)
# ─────────────────────────────────────────────
NLP_SENTIMENT_ENABLED = True
NLP_NEWS_LIMIT_PER_SYMBOL = 10    # how many recent headlines to fetch per ticker
NLP_MIN_SENTIMENT = -0.20         # reject trade if news sentiment is below this threshold (negative)
NLP_DEVICE = "auto"               # "auto", "cuda", or "cpu"
NLP_USE_FP16 = True               # use float16 on CUDA to reduce memory and improve throughput

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_LEVEL = "DEBUG"
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


# ─────────────────────────────────────────────────────────────
# Dashboard config overrides
# When the bot is launched as a subprocess by the dashboard server,
# any UI-saved config changes are injected as CFG_OVERRIDE_<KEY>=<value>
# environment variables so the bot process sees the correct values.
#
# When run standalone (python main.py), we also load config_overrides.json
# directly so that dashboard-saved settings are always respected.
# ─────────────────────────────────────────────────────────────
def _apply_file_overrides() -> None:
    """Load config_overrides.json if it exists and apply values to this module."""
    import ast
    overrides_path = os.path.join(os.path.dirname(__file__), "config_overrides.json")
    if not os.path.isfile(overrides_path):
        return
    try:
        with open(overrides_path) as f:
            overrides = json.load(f)
    except Exception:
        return

    for key, val in overrides.items():
        # Skip special keys (lists handled separately)
        if key.startswith("__") and key.endswith("__"):
            # Handle __WATCHLIST__ and __INVERSE_WATCHLIST__
            clean_key = key.strip("_")
            if clean_key in ("WATCHLIST", "INVERSE_WATCHLIST") and isinstance(val, list):
                globals()[clean_key] = val
            continue
        # Only override keys that already exist in this module
        if key not in globals():
            continue
        current = globals()[key]
        try:
            if isinstance(current, bool):
                coerced = val if isinstance(val, bool) else str(val).lower() in ("1", "true", "yes")
            elif isinstance(current, int):
                coerced = int(val)
            elif isinstance(current, float):
                coerced = float(val)
            else:
                coerced = val
            globals()[key] = coerced
        except (ValueError, TypeError):
            pass


def _apply_env_overrides() -> None:
    import ast
    _prefix = "CFG_OVERRIDE_"
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(_prefix):
            continue
        cfg_key = env_key[len(_prefix):]
        if not hasattr(__import__(__name__), cfg_key):
            continue
        current = globals().get(cfg_key)
        # Coerce string → correct Python type
        try:
            if isinstance(current, bool):
                coerced = env_val.lower() in ("1", "true", "yes")
            elif isinstance(current, int):
                coerced = int(env_val)
            elif isinstance(current, float):
                coerced = float(env_val)
            else:
                coerced = env_val
            globals()[cfg_key] = coerced
        except (ValueError, TypeError):
            pass


# Apply file overrides first, then env overrides (env takes priority)
import json
_apply_file_overrides()
_apply_env_overrides()
