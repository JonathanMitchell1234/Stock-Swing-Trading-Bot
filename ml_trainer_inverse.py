"""
ML Trainer for INVERSE ETF entries – builds a LightGBM gradient-boosting model
for inverse ETF entry prediction (buying inverse ETFs long in bear markets).

Usage (CLI):
    python ml_trainer_inverse.py                          # train on INVERSE_WATCHLIST, 2 years
    python ml_trainer_inverse.py --symbols SQQQ SPXS SH   # specific symbols
    python ml_trainer_inverse.py --months 36               # 3 years of data
    python ml_trainer_inverse.py --tune                    # run Optuna hyper-param search

The trained model is saved to  models/gbm_inverse.txt
and a JSON metrics file to    models/gbm_inverse_meta.json

This is structurally identical to ml_trainer.py but:
  - Trains exclusively on INVERSE_WATCHLIST symbols
  - Uses gain-based labels (price UP — we're buying these long)
  - Saves to separate model files (gbm_inverse.txt)

The model learns the unique inverse ETF pattern:
  Bullish technicals + Bearish macro = High probability of success.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from indicators import compute_all
from logger import get_logger
from ml_features import FEATURE_NAMES, NUM_FEATURES
from ml_features_inverse import (
    build_inverse_feature_matrix,
    generate_inverse_labels,
)

# Re-use the data loading and recency-weighting helpers from the long trainer
from ml_trainer import (
    load_historical_data,
    compute_recency_weights,
)

log = get_logger("ml_trainer_inverse")

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "gbm_inverse.txt"
META_PATH = MODEL_DIR / "gbm_inverse_meta.json"


# ═════════════════════════════════════════════════════════════
# Dataset construction (inverse-specific)
# ═════════════════════════════════════════════════════════════

def build_inverse_dataset(
    data: Dict[str, pd.DataFrame],
    forward_bars: int = 5,
    min_gain_pct: float = 0.03,
    progress_callback=None,
    spy_df: Optional[pd.DataFrame] = None,
    vixy_df: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Build (X, y, weights, symbols_per_row) for INVERSE ETF training.
    Labels are gain-based (same as long model — we buy inverse ETFs long).
    """
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_dates: list[pd.DatetimeIndex] = []
    all_syms: list[str] = []
    total = len(data)

    for i, (sym, df) in enumerate(data.items()):
        start_idx = max(60, config.EMA_LONG + 5) if hasattr(config, "EMA_LONG") else 210
        end_idx = len(df) - forward_bars - 1

        if end_idx <= start_idx:
            log.warning("%s: not enough bars for labeling — skipping", sym)
            continue

        X, valid_idx = build_inverse_feature_matrix(
            df, start_idx=start_idx, end_idx=end_idx,
            spy_df=spy_df, vixy_df=vixy_df,
        )
        if len(X) == 0:
            continue

        y = generate_inverse_labels(
            df, valid_idx,
            forward_bars=forward_bars,
            min_gain_pct=min_gain_pct,
        )

        all_X.append(X)
        all_y.append(y)
        all_dates.append(df.index[valid_idx])
        all_syms.extend([sym] * len(X))
        log.info("  %s: %d samples (%.1f%% positive/gain-profitable)",
                 sym, len(X), 100 * y.mean() if len(y) > 0 else 0)

        if progress_callback:
            progress_callback(i + 1, total, sym)

    if not all_X:
        return np.empty((0, NUM_FEATURES)), np.empty(0), np.empty(0), []

    X_out = np.vstack(all_X)
    y_out = np.concatenate(all_y)
    all_dates_combined = pd.DatetimeIndex(np.concatenate([d.values for d in all_dates]))

    if config.ML_RECENCY_WEIGHT_ENABLED:
        weights = compute_recency_weights(
            all_dates_combined,
            halflife_days=config.ML_RECENCY_HALFLIFE_DAYS,
            min_weight=config.ML_RECENCY_MIN_WEIGHT,
        )
        log.info(
            "Recency weights: halflife=%d days, min=%.2f, "
            "weight range [%.3f, %.3f] (mean=%.3f)",
            config.ML_RECENCY_HALFLIFE_DAYS,
            config.ML_RECENCY_MIN_WEIGHT,
            float(weights.min()),
            float(weights.max()),
            float(weights.mean()),
        )
    else:
        weights = np.ones(len(X_out), dtype=np.float32)

    return X_out, y_out, weights, all_syms


# ═════════════════════════════════════════════════════════════
# Training (re-uses the same LightGBM training from ml_trainer)
# ═════════════════════════════════════════════════════════════

def train_inverse_model(
    X: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
    params: Optional[dict] = None,
    n_splits: int = 5,
) -> Tuple:
    """
    Train a LightGBM binary classifier for INVERSE ETF entry prediction.
    Uses the same training logic as the long model.
    """
    from ml_trainer import train_model
    return train_model(X, y, weights=weights, params=params, n_splits=n_splits)


# ═════════════════════════════════════════════════════════════
# Hyperparameter tuning (re-uses from ml_trainer)
# ═════════════════════════════════════════════════════════════

def tune_inverse_hyperparams(
    X: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_trials: int = 50,
    n_splits: int = 3,
) -> dict:
    """Run Optuna to find the best LightGBM hyperparameters for INVERSE model."""
    from ml_trainer import tune_hyperparams
    return tune_hyperparams(X, y, weights=weights, n_trials=n_trials, n_splits=n_splits)


# ═════════════════════════════════════════════════════════════
# Save helpers
# ═════════════════════════════════════════════════════════════

def save_inverse_model(bst, meta: dict) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    bst.save_model(str(MODEL_PATH))
    META_PATH.write_text(json.dumps(meta, indent=2))
    log.info("Inverse model saved to %s", MODEL_PATH)
    log.info("Inverse metadata saved to %s", META_PATH)


def load_inverse_meta() -> Optional[dict]:
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return None


# ═════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════

def run_inverse_training(
    symbols: Optional[List[str]] = None,
    months: int = 24,
    tune: bool = False,
    forward_bars: int = 5,
    min_gain_pct: float = 0.03,
    progress_callback=None,
) -> dict:
    """
    Full INVERSE ETF training pipeline. Returns the metadata dict.
    Can be called programmatically (from dashboard) or from CLI.
    """
    # Default to INVERSE_WATCHLIST symbols only
    symbols = symbols or list(getattr(config, "INVERSE_WATCHLIST", []))

    if not symbols:
        raise RuntimeError(
            "No symbols to train on — config.INVERSE_WATCHLIST is empty. "
            "Add inverse ETF symbols (e.g. SQQQ, SPXS, SH) to INVERSE_WATCHLIST."
        )

    # Deduplicate preserving order
    seen = set()
    unique_symbols = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            unique_symbols.append(s)
    symbols = unique_symbols

    log.info("=" * 60)
    log.info("INVERSE ETF ML TRAINING START — %d symbols, %d months history",
             len(symbols), months)
    log.info("Symbols: %s", ", ".join(symbols))
    log.info("Label: ≥%.1f%% GAIN within %d bars (buying inverse ETFs long)",
             min_gain_pct * 100, forward_bars)
    if config.ML_RECENCY_WEIGHT_ENABLED:
        log.info(
            "Recency weighting: ENABLED  halflife=%d days  min_weight=%.2f",
            config.ML_RECENCY_HALFLIFE_DAYS,
            config.ML_RECENCY_MIN_WEIGHT,
        )
    else:
        log.info("Recency weighting: DISABLED (uniform weights)")
    log.info("=" * 60)

    t0 = time.time()

    # 1. Load data
    log.info("Step 1/3: Loading historical data…")
    data = load_historical_data(symbols, months=months)
    log.info("Loaded %d / %d symbols", len(data), len(symbols))

    if not data:
        raise RuntimeError("No data loaded — cannot train inverse model")

    # Load macro context data (SPY + VIXY) for regime features
    spy_df = None
    vixy_df = None
    macro_data = load_historical_data(
        [s for s in ["SPY", config.VIX_SYMBOL] if s not in data],
        months=months,
    )
    spy_df = data.get("SPY") if "SPY" in data else macro_data.get("SPY")
    _vix_sym = config.VIX_SYMBOL
    vixy_df = data.get(_vix_sym) if _vix_sym in data else macro_data.get(_vix_sym)

    if spy_df is not None:
        log.info("Macro context: SPY loaded (%d bars)", len(spy_df))
    if vixy_df is not None:
        log.info("Macro context: %s loaded (%d bars)", config.VIX_SYMBOL, len(vixy_df))

    # 2. Build dataset
    log.info("Step 2/3: Extracting INVERSE ETF features & labels…")
    X, y, weights, sym_labels = build_inverse_dataset(
        data,
        forward_bars=forward_bars,
        min_gain_pct=min_gain_pct,
        progress_callback=progress_callback,
        spy_df=spy_df,
        vixy_df=vixy_df,
    )
    log.info("Inverse dataset: %d samples, %d features, %.1f%% positive (gain occurred)",
             len(X), X.shape[1] if len(X) > 0 else 0,
             100 * y.mean() if len(y) > 0 else 0)

    if len(X) < 200:
        raise RuntimeError(
            f"Too few samples ({len(X)}). Inverse ETFs have limited history. "
            f"Try increasing --months or adding more symbols to INVERSE_WATCHLIST."
        )

    # 3. Train
    log.info("Step 3/3: Training INVERSE ETF LightGBM…")
    best_params = None
    if tune:
        log.info("Running Optuna hyperparameter search (50 trials)…")
        best_params = tune_inverse_hyperparams(X, y, weights=weights, n_trials=50)

    bst, meta = train_inverse_model(X, y, weights=weights, params=best_params)

    # Add training metadata
    meta["model_type"] = "inverse"
    meta["symbols"] = sorted(data.keys())
    meta["months"] = months
    meta["label_params"] = {
        "forward_bars": forward_bars,
        "min_gain_pct": min_gain_pct,
    }
    meta["recency_weighting"] = {
        "enabled": config.ML_RECENCY_WEIGHT_ENABLED,
        "halflife_days": config.ML_RECENCY_HALFLIFE_DAYS if config.ML_RECENCY_WEIGHT_ENABLED else None,
        "min_weight": config.ML_RECENCY_MIN_WEIGHT if config.ML_RECENCY_WEIGHT_ENABLED else None,
        "weight_min": round(float(weights.min()), 4),
        "weight_max": round(float(weights.max()), 4),
    }
    meta["cv_threshold"] = getattr(config, "ML_INVERSE_ENTRY_THRESHOLD",
                                    config.ML_ENTRY_THRESHOLD)
    meta["training_time_s"] = round(time.time() - t0, 1)

    save_inverse_model(bst, meta)

    # Print summary
    avg = meta["avg_metrics"]
    log.info("=" * 60)
    log.info("INVERSE ETF TRAINING COMPLETE in %.1fs", meta["training_time_s"])
    log.info("  Samples : %d (%d pos / %d neg)", meta["n_samples"],
             meta["n_positive"], meta["n_negative"])
    log.info("  CV Acc  : %.1f%%", avg["accuracy"] * 100)
    log.info("  CV Prec : %.1f%%", avg["precision"] * 100)
    log.info("  CV Rec  : %.1f%%", avg["recall"] * 100)
    log.info("  CV F1   : %.3f", avg["f1"])
    log.info("  CV AUC  : %.3f", avg["auc"])
    log.info("=" * 60)

    # Top 10 features
    top_feats = list(meta["feature_importance"].items())[:10]
    log.info("Top 10 features by importance (gain):")
    for fname, score in top_feats:
        log.info("  %4d  %s", score, fname)

    return meta


def main():
    parser = argparse.ArgumentParser(description="Train the INVERSE ETF GBM entry model")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Symbols to train on (default: INVERSE_WATCHLIST)")
    parser.add_argument("--months", type=int, default=24,
                        help="Months of historical data (default: 24)")
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter search first")
    parser.add_argument("--forward-bars", type=int, default=5,
                        help="Forward bars for label generation (default: 5)")
    parser.add_argument("--min-gain-pct", type=float, default=0.03,
                        help="Minimum gain %% for positive label (default: 0.03 = 3%%)")
    args = parser.parse_args()

    run_inverse_training(
        symbols=args.symbols,
        months=args.months,
        tune=args.tune,
        forward_bars=args.forward_bars,
        min_gain_pct=args.min_gain_pct,
    )


if __name__ == "__main__":
    main()
