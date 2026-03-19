"""
ML Trainer – builds a LightGBM gradient-boosting model for entry prediction.

Usage (CLI):
    python ml_trainer.py                          # train on all watchlist symbols, 2 years
    python ml_trainer.py --symbols AAPL MSFT NVDA # specific symbols
    python ml_trainer.py --months 36              # 3 years of data
    python ml_trainer.py --tune                   # run Optuna hyper-param search

The trained model is saved to  models/gbm_entry.txt
and a JSON metrics file to    models/gbm_entry_meta.json
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
from ml_features import (
    FEATURE_NAMES,
    NUM_FEATURES,
    build_feature_matrix,
    generate_labels,
)

log = get_logger("ml_trainer")

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "gbm_entry.txt"
META_PATH = MODEL_DIR / "gbm_entry_meta.json"


# ═════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════

def load_historical_data(
    symbols: List[str],
    months: int = 24,
    end_date: Optional[dt.date] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Download historical daily bars via Alpaca and compute indicators.
    Returns {symbol: indicator-enriched DataFrame}.
    """
    import alpaca_trade_api as tradeapi

    api = tradeapi.REST(
        key_id=config.ALPACA_API_KEY,
        secret_key=config.ALPACA_SECRET_KEY,
        base_url=config.BASE_URL,
        api_version="v2",
    )

    end = end_date or dt.date.today()
    warmup_days = 250  # extra bars for indicator warm-up
    start = end - dt.timedelta(days=months * 30 + warmup_days)

    data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            bars = api.get_bars(
                sym,
                config.BAR_TIMEFRAME,
                start=start.isoformat(),
                end=end.isoformat(),
                limit=10_000,
                feed=config.DATA_FEED,
            )
            df = bars.df.copy()
            df.index = pd.to_datetime(df.index)
            df = df[["open", "high", "low", "close", "volume"]]

            if len(df) < config.EMA_TREND + 60:
                log.warning("%s: only %d bars, need %d — skipping",
                            sym, len(df), config.EMA_TREND + 60)
                continue

            df = compute_all(df)
            data[sym] = df
            log.info("Loaded %s: %d bars (%s → %s)",
                     sym, len(df), df.index[0].date(), df.index[-1].date())
        except Exception as exc:
            log.warning("Failed to load %s: %s", sym, exc)

    return data


# ═════════════════════════════════════════════════════════════
# Recency weighting
# ═════════════════════════════════════════════════════════════

def compute_recency_weights(
    dates: pd.DatetimeIndex,
    halflife_days: int,
    min_weight: float,
    reference_date: Optional[dt.date] = None,
) -> np.ndarray:
    """
    Return a per-row sample-weight array that decays exponentially into the
    past so recent data receives more importance during training.

    Weight formula:
        w(t) = max(min_weight, 2 ^ (-(days_ago / halflife_days)))

    The array is normalised so its mean is 1.0.
    """
    # Reference calendar date (no timezone)
    ref = pd.Timestamp(reference_date or dt.date.today()).normalize().date()

    # Normalize input dates to calendar dates (datetime.date). This avoids
    # any tz-aware vs tz-naive arithmetic entirely because we only care about
    # full-calendar-day differences.
    try:
        date_array = dates.normalize().date
    except Exception:
        # Fallback: iterate and convert (safe but slightly slower)
        date_array = np.array([pd.Timestamp(d).normalize().date() for d in dates], dtype=object)

    # Compute days ago as integer array and clamp at 0
    days_ago = np.array([(ref - d).days for d in date_array], dtype=float)
    days_ago = np.clip(days_ago, 0.0, None)

    # Exponential decay + floor, then normalise to mean=1
    weights = np.power(2.0, -(days_ago / float(halflife_days)))
    weights = np.maximum(weights, float(min_weight))
    weights = weights / float(weights.mean())
    return weights.astype(np.float32)


# ═════════════════════════════════════════════════════════════
# Dataset construction
# ═════════════════════════════════════════════════════════════

def build_dataset(
    data: Dict[str, pd.DataFrame],
    forward_bars: int = 5,
    min_gain_pct: float = 0.03,
    progress_callback=None,
    spy_df: Optional[pd.DataFrame] = None,
    vixy_df: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Build (X, y, weights, symbols_per_row) from all loaded symbol DataFrames.
    weights is a float32 array of per-sample recency weights (mean=1.0) or
    an array of ones when recency weighting is disabled.
    """
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_dates: list[pd.DatetimeIndex] = []
    all_syms: list[str] = []
    total = len(data)

    for i, (sym, df) in enumerate(data.items()):
        # Features: start after enough warmup bars
        start_idx = max(60, config.EMA_LONG + 5) if hasattr(config, "EMA_LONG") else 210
        # Leave room at end for forward labels
        end_idx = len(df) - forward_bars - 1

        if end_idx <= start_idx:
            log.warning("%s: not enough bars for labeling — skipping", sym)
            continue

        X, valid_idx = build_feature_matrix(
            df, start_idx=start_idx, end_idx=end_idx,
            spy_df=spy_df, vixy_df=vixy_df,
        )
        if len(X) == 0:
            continue

        y = generate_labels(
            df, valid_idx,
            forward_bars=forward_bars,
            min_gain_pct=min_gain_pct,
        )

        all_X.append(X)
        all_y.append(y)
        all_dates.append(df.index[valid_idx])
        all_syms.extend([sym] * len(X))
        log.info("  %s: %d samples (%.1f%% positive)",
                 sym, len(X), 100 * y.mean() if len(y) > 0 else 0)

        if progress_callback:
            progress_callback(i + 1, total, sym)

    if not all_X:
        return np.empty((0, NUM_FEATURES)), np.empty(0), np.empty(0), []

    X_out = np.vstack(all_X)
    y_out = np.concatenate(all_y)
    all_dates_combined = pd.DatetimeIndex(np.concatenate([d.values for d in all_dates]))

    # Compute per-sample recency weights
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
# Training
# ═════════════════════════════════════════════════════════════

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
    params: Optional[dict] = None,
    n_splits: int = 5,
) -> Tuple["lgb.Booster", dict]:
    """
    Train a LightGBM binary classifier using TimeSeriesSplit CV.
    weights: optional per-sample float array (e.g. from compute_recency_weights).
             When None, all samples are weighted equally.
    Returns (booster, metrics_dict).
    """
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        log_loss,
    )

    if weights is None:
        weights = np.ones(len(y), dtype=np.float32)

    # ── Compute scale_pos_weight from actual class balance ───
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    spw = n_neg / n_pos if n_pos > 0 else 1.0

    default_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "max_depth": 7,
        "learning_rate": 0.05,
        "n_estimators": 800,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_samples": 50,
        "scale_pos_weight": spw,   # penalise missing the minority (winners)
        "verbose": -1,
        "random_state": 42,
        "feature_pre_filter": False,
    }
    if params:
        default_params.update(params)

    # Pull n_estimators out before passing to lgb.train — it is a sklearn alias
    # that LightGBM's native API does not recognise and will warn/error on.
    n_estimators = default_params.pop("n_estimators", 800)

    # ── Pre-build the full Dataset once ─────────────────────
    # Constructing lgb.Dataset from raw arrays inside each CV fold triggers
    # LightGBM's "use a Dataset" warning and repeats the expensive binning
    # step for every fold.  Building it once with free_raw_data=False keeps
    # the raw arrays alive so fold subsets can reference the same bin edges.
    ds_full_ref = lgb.Dataset(
        X, label=y, weight=weights,
        feature_name=FEATURE_NAMES,
        free_raw_data=False,
    ).construct()

    # ── Time-series CV ───────────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_metrics: list[dict] = []

    log.info("Training LightGBM with %d samples, %d features, %d CV folds",
             len(X), X.shape[1], n_splits)
    log.info("Class balance: %.1f%% positive (%d / %d), scale_pos_weight=%.2f",
             100 * y.mean(), y.sum(), len(y), spw)
    recency_on = config.ML_RECENCY_WEIGHT_ENABLED and not np.all(weights == 1.0)
    if recency_on:
        log.info(
            "Recency weighting ENABLED — weight range [%.3f, %.3f]",
            float(weights.min()), float(weights.max()),
        )

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        w_tr = weights[train_idx]

        # Build fold datasets from numpy arrays but reference the pre-built
        # full dataset so LightGBM reuses bin thresholds instead of re-binning.
        ds_train = lgb.Dataset(
            X_tr, label=y_tr, weight=w_tr,
            feature_name=FEATURE_NAMES,
            free_raw_data=False,
            reference=ds_full_ref,
        )
        ds_val_fold = lgb.Dataset(
            X_val, label=y_val,
            feature_name=FEATURE_NAMES,
            free_raw_data=False,
            reference=ds_train,
        )

        callbacks = [
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=0),  # silence
        ]

        bst = lgb.train(
            default_params,
            ds_train,
            num_boost_round=n_estimators,
            valid_sets=[ds_val_fold],
            callbacks=callbacks,
        )

        y_prob = bst.predict(X_val)
        # Use the live entry threshold so CV metrics reflect real-world
        # decision-making (default 0.50 is far too conservative for noisy
        # financial data — the bot already gates on hand-crafted score).
        cv_threshold = config.ML_ENTRY_THRESHOLD
        y_pred = (y_prob >= cv_threshold).astype(int)

        fold_metrics = {
            "fold": fold,
            "accuracy": round(accuracy_score(y_val, y_pred), 4),
            "precision": round(precision_score(y_val, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_val, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_val, y_pred, zero_division=0), 4),
            "auc": round(roc_auc_score(y_val, y_prob), 4) if len(np.unique(y_val)) > 1 else 0.0,
            "logloss": round(log_loss(y_val, y_prob), 4),
            "n_train": len(y_tr),
            "n_val": len(y_val),
            "best_iter": bst.best_iteration,
        }
        cv_metrics.append(fold_metrics)
        log.info("  Fold %d: acc=%.3f  prec=%.3f  rec=%.3f  f1=%.3f  auc=%.3f",
                 fold, fold_metrics["accuracy"], fold_metrics["precision"],
                 fold_metrics["recall"], fold_metrics["f1"], fold_metrics["auc"])

    # ── Final model: train on ALL data ───────────────────────
    # Reuse the already-constructed full dataset — no re-binning needed.
    log.info("Training final model on full dataset (%d samples)…", len(X))

    final_params = dict(default_params)  # n_estimators already popped above
    # Use the median best_iteration from CV as num_boost_round
    median_iters = int(np.median([m["best_iter"] for m in cv_metrics]))
    final_n_rounds = max(100, median_iters + 50)

    final_bst = lgb.train(
        final_params,
        ds_full_ref,
        num_boost_round=final_n_rounds,
        callbacks=[lgb.log_evaluation(period=0)],
    )

    # ── Feature importances ──────────────────────────────────
    importance = dict(zip(
        FEATURE_NAMES,
        [int(x) for x in final_bst.feature_importance(importance_type="gain")],
    ))
    sorted_imp = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    # Aggregate CV metrics
    avg_metrics = {
        "accuracy":  round(np.mean([m["accuracy"]  for m in cv_metrics]), 4),
        "precision": round(np.mean([m["precision"] for m in cv_metrics]), 4),
        "recall":    round(np.mean([m["recall"]    for m in cv_metrics]), 4),
        "f1":        round(np.mean([m["f1"]        for m in cv_metrics]), 4),
        "auc":       round(np.mean([m["auc"]       for m in cv_metrics]), 4),
        "logloss":   round(np.mean([m["logloss"]   for m in cv_metrics]), 4),
    }

    meta = {
        "trained_at": dt.datetime.now().isoformat(),
        "n_samples": int(len(X)),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
        "n_features": int(X.shape[1]),
        "n_cv_folds": n_splits,
        "cv_metrics": cv_metrics,
        "avg_metrics": avg_metrics,
        "feature_importance": sorted_imp,
        "final_n_rounds": final_n_rounds,
        "params": {k: v for k, v in final_params.items()
                   if k not in ("verbose", "random_state", "feature_pre_filter")},
    }

    return final_bst, meta


# ═════════════════════════════════════════════════════════════
# Hyperparameter tuning (optional)
# ═════════════════════════════════════════════════════════════

def tune_hyperparams(
    X: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_trials: int = 50,
    n_splits: int = 3,
) -> dict:
    """
    Run Optuna to find the best LightGBM hyperparameters.
    weights: optional per-sample recency weights passed to LightGBM.
    Returns the best params dict.
    """
    import optuna
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if weights is None:
        weights = np.ones(len(y), dtype=np.float32)

    # Compute scale_pos_weight for imbalance handling
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    spw = n_neg / n_pos if n_pos > 0 else 1.0

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "scale_pos_weight": spw,
            "verbose": -1,
            "feature_pre_filter": False,
        }

        tscv = TimeSeriesSplit(n_splits=n_splits)
        aucs = []

        for train_idx, val_idx in tscv.split(X):
            w_tr = weights[train_idx]
            ds_tr  = lgb.Dataset(X[train_idx], label=y[train_idx], weight=w_tr, feature_name=FEATURE_NAMES)
            ds_val = lgb.Dataset(X[val_idx],   label=y[val_idx],   reference=ds_tr)

            bst = lgb.train(
                params, ds_tr,
                num_boost_round=500,
                valid_sets=[ds_val],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
            )
            y_prob = bst.predict(X[val_idx])
            if len(np.unique(y[val_idx])) > 1:
                aucs.append(roc_auc_score(y[val_idx], y_prob))

        return float(np.mean(aucs)) if aucs else 0.5

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    log.info("Best AUC: %.4f", study.best_value)
    log.info("Best params: %s", study.best_params)
    return study.best_params


# ═════════════════════════════════════════════════════════════
# Save / load helpers
# ═════════════════════════════════════════════════════════════

def save_model(bst, meta: dict) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    bst.save_model(str(MODEL_PATH))
    META_PATH.write_text(json.dumps(meta, indent=2))
    log.info("Model saved to %s", MODEL_PATH)
    log.info("Metadata saved to %s", META_PATH)


def load_meta() -> Optional[dict]:
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return None


# ═════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════

def run_training(
    symbols: Optional[List[str]] = None,
    months: int = 24,
    tune: bool = False,
    forward_bars: int = 5,
    min_gain_pct: float = 0.03,
    progress_callback=None,
) -> dict:
    """
    Full training pipeline. Returns the metadata dict.
    Can be called programmatically (from dashboard) or from CLI.
    """
    symbols = symbols or (list(config.WATCHLIST) + list(config.INVERSE_WATCHLIST))
    # Deduplicate preserving order
    seen = set()
    unique_symbols = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            unique_symbols.append(s)
    symbols = unique_symbols

    log.info("=" * 60)
    log.info("ML TRAINING START — %d symbols, %d months history", len(symbols), months)
    log.info("Label: ≥%.1f%% gain within %d bars",
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
        raise RuntimeError("No data loaded — cannot train")

    # Load macro context data (SPY + VIXY) for regime features
    spy_df = None
    vixy_df = None
    macro_syms = {"SPY": None, config.VIX_SYMBOL: None}
    macro_data = load_historical_data(
        [s for s in macro_syms if s not in data],
        months=months,
    )
    # Avoid `df_a or df_b` — the `or` operator calls bool() on a
    # DataFrame which raises "truth value of a DataFrame is ambiguous".
    spy_df  = data.get("SPY") if "SPY" in data else macro_data.get("SPY")
    _vix_sym = config.VIX_SYMBOL
    vixy_df = data.get(_vix_sym) if _vix_sym in data else macro_data.get(_vix_sym)
    if spy_df is not None:
        log.info("Macro context: SPY loaded (%d bars)", len(spy_df))
    else:
        log.warning("SPY data unavailable — spy_sma200_dist feature will be 0")
    if vixy_df is not None:
        log.info("Macro context: %s loaded (%d bars)", config.VIX_SYMBOL, len(vixy_df))
    else:
        log.warning("%s data unavailable — vixy_relative feature will be 0",
                    config.VIX_SYMBOL)

    # 2. Build dataset
    log.info("Step 2/3: Extracting features & labels…")
    X, y, weights, sym_labels = build_dataset(
        data,
        forward_bars=forward_bars,
        min_gain_pct=min_gain_pct,
        progress_callback=progress_callback,
        spy_df=spy_df,
        vixy_df=vixy_df,
    )
    log.info("Dataset: %d samples, %d features, %.1f%% positive",
             len(X), X.shape[1] if len(X) > 0 else 0,
             100 * y.mean() if len(y) > 0 else 0)

    if len(X) < 500:
        raise RuntimeError(f"Too few samples ({len(X)}). Need at least 500 for a useful model.")

    # 3. Train
    log.info("Step 3/3: Training LightGBM…")
    best_params = None
    if tune:
        log.info("Running Optuna hyperparameter search (50 trials)…")
        best_params = tune_hyperparams(X, y, weights=weights, n_trials=50)

    bst, meta = train_model(X, y, weights=weights, params=best_params)

    # Add training metadata
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
    meta["cv_threshold"] = config.ML_ENTRY_THRESHOLD
    meta["training_time_s"] = round(time.time() - t0, 1)

    save_model(bst, meta)

    # Print summary
    avg = meta["avg_metrics"]
    log.info("=" * 60)
    log.info("TRAINING COMPLETE in %.1fs", meta["training_time_s"])
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
    parser = argparse.ArgumentParser(description="Train the GBM entry model")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Symbols to train on (default: full watchlist)")
    parser.add_argument("--months", type=int, default=24,
                        help="Months of historical data (default: 24)")
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter search first")
    parser.add_argument("--forward-bars", type=int, default=5,
                        help="Forward bars for label generation (default: 5)")
    parser.add_argument("--min-gain-pct", type=float, default=0.03,
                        help="Minimum gain %% for positive label (default: 0.03 = 3%%)")
    args = parser.parse_args()

    run_training(
        symbols=args.symbols,
        months=args.months,
        tune=args.tune,
        forward_bars=args.forward_bars,
        min_gain_pct=args.min_gain_pct,
    )


if __name__ == "__main__":
    main()
