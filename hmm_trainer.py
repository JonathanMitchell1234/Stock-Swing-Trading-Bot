"""
HMM Regime Trainer – builds a 3-state Gaussian Hidden Markov Model
for market regime detection (Bull / Bear / Chop).

Usage (CLI):
    python hmm_trainer.py                    # train on SPY, 3 years
    python hmm_trainer.py --months 48        # 4 years of data
    python hmm_trainer.py --states 2         # 2-state model (bull/bear only)
    python hmm_trainer.py --vol-window 10    # 10-day realised vol window

The trained model is saved to  models/hmm_regime.pkl
and a JSON metrics file to     models/hmm_regime_meta.json

IMPORTANT: Data is NEVER shuffled. HMMs are sequence models — the
transition matrix is only meaningful when rows are in chronological order.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from hmmlearn.hmm import GaussianHMM

import config
from logger import get_logger

log = get_logger("hmm_trainer")

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH  = MODEL_DIR / "hmm_regime.pkl"
META_PATH   = MODEL_DIR / "hmm_regime_meta.json"
SCALER_PATH = MODEL_DIR / "hmm_scaler.pkl"


# ═════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════

def load_spy_data(
    months: int = 36,
    end_date: Optional[dt.date] = None,
) -> pd.DataFrame:
    """
    Download SPY daily bars via Alpaca and return a clean OHLCV DataFrame.
    No indicator computation — just raw price data.
    """
    import alpaca_trade_api as tradeapi

    api = tradeapi.REST(
        key_id=config.ALPACA_API_KEY,
        secret_key=config.ALPACA_SECRET_KEY,
        base_url=config.BASE_URL,
        api_version="v2",
    )

    end = end_date or dt.date.today()
    start = end - dt.timedelta(days=months * 30 + 30)  # small extra buffer

    symbol = config.MARKET_REGIME_SYMBOL  # "SPY"

    bars = api.get_bars(
        symbol,
        config.BAR_TIMEFRAME,
        start=start.isoformat(),
        end=end.isoformat(),
        limit=10_000,
        feed=config.DATA_FEED,
    )
    df = bars.df.copy()
    df.index = pd.to_datetime(df.index)
    df = df[["open", "high", "low", "close", "volume"]]

    log.info(
        "Loaded %s: %d bars (%s → %s)",
        symbol, len(df), df.index[0].date(), df.index[-1].date(),
    )
    return df


# ═════════════════════════════════════════════════════════════
# Feature engineering
# ═════════════════════════════════════════════════════════════

def build_hmm_features(
    df: pd.DataFrame,
    vol_window: int = 5,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Build a 2D feature matrix of [log_return, realized_volatility] from
    daily OHLCV data.

    Returns:
        features : np.ndarray of shape (T, 2)
        dates    : DatetimeIndex aligned with features

    The first `vol_window` rows are consumed by the rolling window and dropped.
    Data is NEVER shuffled — chronological order is preserved.
    """
    close = df["close"].astype(float)

    # Daily log returns: ln(P_t / P_{t-1})
    log_returns = np.log(close / close.shift(1))

    # Realised volatility: rolling std of log returns (annualised not needed
    # for HMM — raw rolling std captures the variance regime directly)
    rolling_vol = log_returns.rolling(window=vol_window).std()

    # Stack into 2D array and drop NaN rows (from shift + rolling)
    features_df = pd.DataFrame({
        "log_return": log_returns,
        "volatility": rolling_vol,
    }, index=df.index)

    features_df = features_df.dropna()

    features = features_df[["log_return", "volatility"]].values
    dates = features_df.index

    log.info(
        "HMM features: %d samples, log_return range [%.4f, %.4f], "
        "vol range [%.6f, %.6f]",
        len(features),
        features[:, 0].min(), features[:, 0].max(),
        features[:, 1].min(), features[:, 1].max(),
    )

    return features, dates


# ═════════════════════════════════════════════════════════════
# Label states after training
# ═════════════════════════════════════════════════════════════

def label_states(model) -> dict[int, str]:
    """
    After training, inspect the learned means to assign human-readable
    labels to each hidden state.

    Sorting logic:
      - State with highest mean log_return → BULL
      - State with lowest mean log_return  → BEAR
      - Remaining state(s)                 → CHOP

    Returns a dict mapping state_id → label string.
    """
    means = model.means_  # shape (n_states, n_features)
    n_states = means.shape[0]

    # Sort states by mean log_return (column 0)
    sorted_by_return = np.argsort(means[:, 0])

    labels = {}
    if n_states == 2:
        labels[int(sorted_by_return[0])] = "BEAR"
        labels[int(sorted_by_return[1])] = "BULL"
    elif n_states >= 3:
        labels[int(sorted_by_return[0])] = "BEAR"
        labels[int(sorted_by_return[-1])] = "BULL"
        for idx in sorted_by_return[1:-1]:
            labels[int(idx)] = "CHOP"
    else:
        labels[0] = "UNKNOWN"

    return labels


# ═════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════

def train_hmm(
    features: np.ndarray,
    n_states: int = 3,
    n_iter: int = 200,
    random_state: int = 42,
) -> "tuple":
    """
    Standardize features and train a Gaussian HMM.

    Parameters:
        features     : (T, 2) raw array of [log_return, volatility]
        n_states     : number of hidden states (2 or 3 recommended)
        n_iter       : max EM iterations
        random_state : seed for reproducibility

    Returns (model, scaler) — the fitted GaussianHMM and its StandardScaler.
    The scaler MUST be saved and reused at inference time; features passed to
    model.predict() must be transformed with the same scaler.

    Why StandardScaler?
    Raw log_return ≈ 0.001 while raw volatility ≈ 0.015, so without scaling
    the EM algorithm is dominated by the feature with the larger absolute
    variance. Z-scoring forces equal weighting and produces well-separated,
    persistent states instead of a degenerate metronome transition matrix.
    """
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler

    # Fix 1: Z-score both features so direction and volatility are weighted equally
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    log.info(
        "Training %d-state GaussianHMM on %d samples "
        "(StandardScaler applied, max %d EM iterations)...",
        n_states, len(features), n_iter,
    )

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        tol=1e-4,
        verbose=False,
    )
    model.fit(X)

    log.info(
        "HMM training converged: %s (score=%.2f)",
        model.monitor_.converged, model.score(X),
    )

    return model, scaler


# ═════════════════════════════════════════════════════════════
# Save / Load
# ═════════════════════════════════════════════════════════════

def save_model(model, scaler, state_labels: dict, meta: dict) -> None:
    """Persist the trained HMM, its StandardScaler, and metadata to disk."""
    import joblib

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    log.info("Saved HMM model → %s", MODEL_PATH)

    # Scaler must be saved so inference uses identical feature normalisation
    joblib.dump(scaler, SCALER_PATH)
    log.info("Saved HMM scaler → %s", SCALER_PATH)

    meta["state_labels"] = {str(k): v for k, v in state_labels.items()}
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    log.info("Saved HMM metadata → %s", META_PATH)


# ═════════════════════════════════════════════════════════════
# Main training pipeline
# ═════════════════════════════════════════════════════════════

def run_training(
    months: int = 36,
    n_states: int = 3,
    vol_window: int = 5,
    n_iter: int = 200,
) -> None:
    """Full training pipeline: load → features → standardize → train → save."""

    log.info("=" * 60)
    log.info("HMM REGIME TRAINER")
    log.info("  Symbol   : %s", config.MARKET_REGIME_SYMBOL)
    log.info("  History  : %d months", months)
    log.info("  States   : %d", n_states)
    log.info("  Vol win  : %d days (shorter = sharper volatility signal)", vol_window)
    log.info("  Scaling  : StandardScaler (z-score both features)")
    log.info("=" * 60)

    # 1. Load data
    df = load_spy_data(months=months)
    if len(df) < vol_window + 50:
        log.error("Not enough data (%d bars) — need at least %d. Aborting.",
                  len(df), vol_window + 50)
        sys.exit(1)

    # 2. Build raw features (NEVER shuffled — sequential order preserved)
    features, dates = build_hmm_features(df, vol_window=vol_window)

    # 3. Train (StandardScaler is applied inside train_hmm)
    model, scaler = train_hmm(features, n_states=n_states, n_iter=n_iter)

    # 4. Label states (ordering is preserved under monotonic scaling)
    state_labels = label_states(model)

    # 5. Log learned parameters in original (unscaled) units via inverse_transform
    raw_means = scaler.inverse_transform(model.means_)
    log.info("─── Learned State Parameters (original units) ───")
    for sid in range(n_states):
        label = state_labels.get(sid, "?")
        mean_ret = raw_means[sid, 0]
        mean_vol = raw_means[sid, 1]
        log.info(
            "  State %d (%s): mean_return=%.5f  mean_vol=%.5f",
            sid, label, mean_ret, mean_vol,
        )

    # Sanity-check: warn if any two states are suspiciously similar (degenerate)
    for i in range(n_states):
        for j in range(i + 1, n_states):
            d_ret = abs(raw_means[i, 0] - raw_means[j, 0])
            d_vol = abs(raw_means[i, 1] - raw_means[j, 1])
            if d_ret < 0.0003 and d_vol < 0.001:
                log.warning(
                    "  ⚠ States %d (%s) and %d (%s) are nearly identical — "
                    "consider reducing n_states or adjusting vol_window.",
                    i, state_labels.get(i, "?"), j, state_labels.get(j, "?"),
                )

    log.info("─── Transition Matrix ───")
    for i in range(n_states):
        row = model.transmat_[i]
        parts = [f"  P({state_labels.get(i,'?')} → {state_labels.get(j,'?')})={row[j]:.4f}" for j in range(n_states)]
        log.info("  %s", "  ".join(parts))
        if row[i] < 0.50:
            log.warning(
                "  ⚠ State %d (%s) diagonal P=%.4f — regime is NOT persistent. "
                "This state will toggle every day. Consider re-tuning.",
                i, state_labels.get(i, "?"), row[i],
            )

    # 6. Decode full history using scaled features
    X_scaled = scaler.transform(features)
    hidden_states = model.predict(X_scaled)
    state_counts = {state_labels.get(s, "?"): int((hidden_states == s).sum()) for s in range(n_states)}

    log.info("─── State Distribution (training set) ───")
    for label, count in state_counts.items():
        log.info("  %s: %d days (%.1f%%)", label, count, 100 * count / len(hidden_states))

    # 7. Save
    meta = {
        "symbol": config.MARKET_REGIME_SYMBOL,
        "n_states": n_states,
        "vol_window": vol_window,
        "training_months": months,
        "training_samples": len(features),
        "training_date_range": f"{dates[0].date()} → {dates[-1].date()}",
        "trained_at": dt.datetime.now().isoformat(),
        "score": float(model.score(X_scaled)),
        "converged": bool(model.monitor_.converged),
        "state_distribution": state_counts,
        # Store unscaled means so the JSON is human-readable
        "state_means": {
            state_labels.get(s, str(s)): {
                "log_return": float(raw_means[s, 0]),
                "volatility": float(raw_means[s, 1]),
            }
            for s in range(n_states)
        },
        "transition_matrix": model.transmat_.tolist(),
    }
    save_model(model, scaler, state_labels, meta)

    log.info("=" * 60)
    log.info("HMM training complete!")
    log.info("=" * 60)


# ═════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HMM regime detector")
    parser.add_argument("--months", type=int, default=36,
                        help="Months of SPY history (default: 36)")
    parser.add_argument("--states", type=int, default=3,
                        help="Number of hidden states (default: 3)")
    parser.add_argument("--vol-window", type=int, default=5,
                        help="Rolling volatility window in days (default: 5)")
    parser.add_argument("--n-iter", type=int, default=200,
                        help="Max EM iterations (default: 200)")
    args = parser.parse_args()

    run_training(
        months=args.months,
        n_states=args.states,
        vol_window=args.vol_window,
        n_iter=args.n_iter,
    )
