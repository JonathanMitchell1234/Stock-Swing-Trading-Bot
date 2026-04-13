"""
ML Trainer – builds a LightGBM gradient-boosting model for entry prediction.

Usage (CLI):
    python ml_trainer.py                          # train on all watchlist symbols, 2 years
    python ml_trainer.py --symbols AAPL MSFT NVDA # specific symbols
    python ml_trainer.py --months 36              # 3 years of data
    python ml_trainer.py --tune                   # run Optuna hyper-param search
    python ml_trainer.py --target-precision 0.60  # tune threshold for higher win-rate

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
    ref = pd.Timestamp(reference_date or dt.date.today()).normalize().date()

    try:
        date_array = dates.normalize().date
    except Exception:
        date_array = np.array([pd.Timestamp(d).normalize().date() for d in dates], dtype=object)

    days_ago = np.array([(ref - d).days for d in date_array], dtype=float)
    days_ago = np.clip(days_ago, 0.0, None)

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
    atr_multiplier: float = 0.0,
    progress_callback=None,
    spy_df: Optional[pd.DataFrame] = None,
    vixy_df: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Build (X, y, weights, symbols_per_row) from all loaded symbol DataFrames.
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
            atr_multiplier=atr_multiplier,
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

    # Sort chronologically for proper TimeSeriesSplit
    sort_idx = np.argsort(all_dates_combined)
    X_out = X_out[sort_idx]
    y_out = y_out[sort_idx]
    all_dates_combined = all_dates_combined[sort_idx]
    all_syms = [all_syms[i] for i in sort_idx]

    # ── Clean features: replace inf/nan ──────────────────────
    X_out = np.nan_to_num(X_out, nan=0.0, posinf=0.0, neginf=0.0)

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
# Calibration — Platt scaling via isotonic / sigmoid
# ═════════════════════════════════════════════════════════════

def _calibrate_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_precision: float = 0.50,  # Updated default
    min_recall: float = 0.10,
) -> float:
    """
    Find the probability threshold that achieves *target_precision*
    while keeping recall ≥ min_recall.  Falls back to 0.50 if nothing
    satisfies both constraints.
    """
    from sklearn.metrics import precision_recall_curve

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    best_thresh = 0.50
    best_f1 = 0.0

    for p, r, t in zip(precisions, recalls, thresholds):
        if p >= target_precision and r >= min_recall:
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(t)

    # If no threshold meets the hard constraints, pick the one that
    # maximises F1 with at least some recall
    if best_f1 == 0:
        for p, r, t in zip(precisions, recalls, thresholds):
            if r >= min_recall:
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = float(t)

    return round(best_thresh, 4)


# ═════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
    params: Optional[dict] = None,
    n_splits: int = 5,
    forward_bars: int = 5,
    target_precision: float = 0.50,  # Passed in dynamically
) -> Tuple["lgb.Booster", dict]:
    """
    Train a LightGBM binary classifier using TimeSeriesSplit CV.
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

    # ── Class balance ─────────────────────────────────────────
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    global_spw = n_neg / n_pos if n_pos > 0 else 1.0

    # ── Default params — MUCH more conservative to reduce overfitting ──
    default_params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",             # CUDA support enabled
        "num_leaves": 31,
        "max_depth": 5,
        "learning_rate": 0.03,
        "n_estimators": 1200,
        "subsample": 0.7,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "colsample_bynode": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "min_child_samples": 80,
        "min_child_weight": 1e-3,
        "min_split_gain": 0.01,
        "max_bin": 255,
        "verbose": -1,
        "random_state": 42,
        "feature_pre_filter": False,
    }

    recency_active = config.ML_RECENCY_WEIGHT_ENABLED and not np.all(weights == 1.0)
    if recency_active:
        default_params["is_unbalance"] = True
    else:
        default_params["scale_pos_weight"] = global_spw

    if params:
        default_params.update(params)

    n_estimators = default_params.pop("n_estimators", 1200)

    lr = default_params.get("learning_rate", 0.03)
    es_patience = max(80, int(150 / (lr / 0.03)))
    n_estimators = max(n_estimators, int(1200 / (lr / 0.03)))

    construction_params = {"feature_pre_filter": default_params.pop("feature_pre_filter", False)}
    ds_full_ref = lgb.Dataset(
        X, label=y, weight=weights,
        feature_name=FEATURE_NAMES,
        free_raw_data=False,
        params=construction_params,
    ).construct()

    # ── Time-series CV ───────────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_metrics: list[dict] = []
    oof_y_true: list[np.ndarray] = []
    oof_y_prob: list[np.ndarray] = []

    log.info("Training LightGBM with %d samples, %d features, %d CV folds",
             len(X), X.shape[1], n_splits)
    log.info("Class balance: %.1f%% positive (%d / %d)",
             100 * y.mean(), y.sum(), len(y))
    log.info("Learning rate: %.4f → early stopping patience: %d, max rounds: %d",
             lr, es_patience, n_estimators)
    if recency_active:
        log.info(
            "Recency weighting ENABLED — weight range [%.3f, %.3f] — using is_unbalance=True",
            float(weights.min()), float(weights.max()),
        )
    else:
        log.info("Using scale_pos_weight=%.2f", default_params.get("scale_pos_weight", global_spw))

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # Purged CV: drop first forward_bars rows from validation
        if forward_bars > 0 and len(val_idx) > forward_bars:
            val_idx = val_idx[forward_bars:]

        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        w_tr = weights[train_idx]

        fold_params = dict(default_params)

        if not recency_active:
            fold_n_pos = int(y_tr.sum())
            fold_n_neg = int(len(y_tr) - fold_n_pos)
            fold_spw = fold_n_neg / fold_n_pos if fold_n_pos > 0 else 1.0
            fold_params["scale_pos_weight"] = fold_spw

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
            lgb.early_stopping(es_patience, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        bst = lgb.train(
            fold_params,
            ds_train,
            num_boost_round=n_estimators,
            valid_sets=[ds_val_fold],
            callbacks=callbacks,
        )

        y_prob = bst.predict(X_val)

        oof_y_true.append(y_val)
        oof_y_prob.append(y_prob)

        for thresh_name, thresh_val in [("0.50", 0.50), ("0.40", 0.40), ("0.35", 0.35)]:
            y_pred_t = (y_prob >= thresh_val).astype(int)
            n_pred_pos = int(y_pred_t.sum())
            if n_pred_pos > 0:
                p = precision_score(y_val, y_pred_t, zero_division=0)
                r = recall_score(y_val, y_pred_t, zero_division=0)
                log.debug("    Fold %d @ thresh=%s: prec=%.3f rec=%.3f pred_pos=%d",
                          fold, thresh_name, p, r, n_pred_pos)

        cv_threshold = 0.40
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
            "n_val_pos": int(y_val.sum()),
            "n_pred_pos": int(y_pred.sum()),
            "best_iter": bst.best_iteration,
            "cv_threshold": cv_threshold,
        }
        cv_metrics.append(fold_metrics)
        log.info("  Fold %d: acc=%.3f  prec=%.3f  rec=%.3f  f1=%.3f  auc=%.3f  "
                 "(val_pos=%d, pred_pos=%d, best_iter=%d)",
                 fold, fold_metrics["accuracy"], fold_metrics["precision"],
                 fold_metrics["recall"], fold_metrics["f1"], fold_metrics["auc"],
                 fold_metrics["n_val_pos"], fold_metrics["n_pred_pos"],
                 fold_metrics["best_iter"])

    # ── Calibrate threshold from OOF predictions ─────────────
    all_oof_y = np.concatenate(oof_y_true)
    all_oof_p = np.concatenate(oof_y_prob)
    
    # Use dynamically passed target_precision
    calibrated_threshold = _calibrate_threshold(
        all_oof_y, all_oof_p,
        target_precision=target_precision,
        min_recall=0.10,
    )
    log.info("Calibrated threshold from OOF: %.4f (target prec≥%.2f, rec≥0.10)",
             calibrated_threshold, target_precision)

    from sklearn.metrics import precision_score, recall_score, f1_score as f1_fn
    oof_pred = (all_oof_p >= calibrated_threshold).astype(int)
    oof_prec = precision_score(all_oof_y, oof_pred, zero_division=0)
    oof_rec = recall_score(all_oof_y, oof_pred, zero_division=0)
    oof_f1 = f1_fn(all_oof_y, oof_pred, zero_division=0)
    log.info("OOF @ calibrated threshold %.3f: prec=%.3f  rec=%.3f  f1=%.3f  pred_pos=%d/%d",
             calibrated_threshold, oof_prec, oof_rec, oof_f1,
             int(oof_pred.sum()), len(oof_pred))

    # ── Final model: train on ALL data ───────────────────────
    log.info("Training final model on full dataset (%d samples)…", len(X))

    final_params = dict(default_params)
    median_iters = int(np.median([m["best_iter"] for m in cv_metrics]))
    final_n_rounds = max(200, median_iters + 80)
    log.info("Final model: %d boost rounds (median CV best_iter=%d + 80)",
             final_n_rounds, median_iters)

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

    total_gain = sum(sorted_imp.values())
    if total_gain > 0:
        top_feat_name = list(sorted_imp.keys())[0]
        top_feat_pct = list(sorted_imp.values())[0] / total_gain * 100
        if top_feat_pct > 40:
            log.warning(
                "Feature '%s' dominates with %.1f%% of total gain — "
                "model may be over-relying on a single signal",
                top_feat_name, top_feat_pct,
            )

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
        "calibrated_threshold": calibrated_threshold,
        "oof_metrics_at_calibrated": {
            "precision": round(oof_prec, 4),
            "recall": round(oof_rec, 4),
            "f1": round(oof_f1, 4),
            "n_pred_pos": int(oof_pred.sum()),
            "n_total": len(oof_pred),
        },
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
    n_trials: int = 100,
    n_splits: int = 3,
    forward_bars: int = 5,
) -> dict:
    """
    Run Optuna to find the best LightGBM hyperparameters.
    Optimises for AUC (threshold-independent) — threshold calibration
    is handled separately after training.
    """
    import optuna
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if weights is None:
        weights = np.ones(len(y), dtype=np.float32)

    recency_active = config.ML_RECENCY_WEIGHT_ENABLED and not np.all(weights == 1.0)

    def objective(trial: optuna.Trial) -> float:
        max_depth = trial.suggest_int("max_depth", 4, 7)
        max_leaves = 2 ** max_depth
        num_leaves = trial.suggest_int("num_leaves", 8, min(max_leaves, 63))
        lr = trial.suggest_float("learning_rate", 0.02, 0.1, log=True)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": lr,
            "subsample": trial.suggest_float("subsample", 0.6, 0.85),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 120),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.02),
            "verbose": -1,
            "feature_pre_filter": False,
        }

        if recency_active:
            params["is_unbalance"] = True

        es_patience = max(60, int(100 / (lr / 0.03)))
        n_boost = max(800, int(1200 / (lr / 0.03)))

        tscv = TimeSeriesSplit(n_splits=n_splits)
        auc_scores = []

        for train_idx, val_idx in tscv.split(X):
            if forward_bars > 0 and len(val_idx) > forward_bars:
                val_idx = val_idx[forward_bars:]

            y_tr = y[train_idx]
            y_val = y[val_idx]
            w_tr = weights[train_idx]

            if len(np.unique(y_val)) < 2:
                continue

            if not recency_active:
                fold_n_pos = int(y_tr.sum())
                fold_n_neg = int(len(y_tr) - fold_n_pos)
                params["scale_pos_weight"] = fold_n_neg / fold_n_pos if fold_n_pos > 0 else 1.0

            ds_tr  = lgb.Dataset(X[train_idx], label=y_tr, weight=w_tr, feature_name=FEATURE_NAMES)
            ds_val = lgb.Dataset(X[val_idx],   label=y_val, reference=ds_tr)

            bst = lgb.train(
                params, ds_tr,
                num_boost_round=n_boost,
                valid_sets=[ds_val],
                callbacks=[lgb.early_stopping(es_patience, verbose=False), lgb.log_evaluation(0)],
            )
            y_prob = bst.predict(X[val_idx])
            auc_scores.append(roc_auc_score(y_val, y_prob))

        return float(np.mean(auc_scores)) if auc_scores else 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    log.info("Best AUC: %.4f", study.best_value)
    log.info("Best params: %s", study.best_params)

    return dict(study.best_params)


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
    atr_multiplier: float = 0.0,
    target_precision: float = 0.50,  # Exposed dynamically
    progress_callback=None,
) -> dict:
    """
    Full training pipeline. Returns the metadata dict.
    """
    symbols = symbols or list(config.WATCHLIST)
    seen = set()
    unique_symbols = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            unique_symbols.append(s)
    symbols = unique_symbols

    log.info("=" * 60)
    log.info("ML TRAINING START — %d symbols, %d months history", len(symbols), months)
    if atr_multiplier > 0:
        log.info("Label: ATR-based target (%.1f× ATR-14) within %d bars",
                 atr_multiplier, forward_bars)
    else:
        log.info("Label: ≥%.1f%% gain within %d bars",
                 min_gain_pct * 100, forward_bars)
    log.info("Target Precision: %.2f (for threshold calibration)", target_precision)
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

    # Load macro context data
    spy_df = None
    vixy_df = None
    macro_syms = {"SPY": None, config.VIX_SYMBOL: None}
    macro_data = load_historical_data(
        [s for s in macro_syms if s not in data],
        months=months,
    )
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
        atr_multiplier=atr_multiplier,
        progress_callback=progress_callback,
        spy_df=spy_df,
        vixy_df=vixy_df,
    )
    log.info("Dataset: %d samples, %d features, %.1f%% positive",
             len(X), X.shape[1] if len(X) > 0 else 0,
             100 * y.mean() if len(y) > 0 else 0)

    if len(X) < 500:
        raise RuntimeError(f"Too few samples ({len(X)}). Need at least 500 for a useful model.")

    # ── Data quality checks ──────────────────────────────────
    pos_rate = y.mean()
    if pos_rate < 0.15:
        log.warning(
            "Very low positive rate (%.1f%%). Consider reducing min_gain_pct "
            "(currently %.1f%%) or increasing forward_bars (currently %d). "
            "Target 25-40%% positive rate for balanced learning.",
            pos_rate * 100, min_gain_pct * 100, forward_bars,
        )
    elif pos_rate > 0.60:
        log.warning(
            "Very high positive rate (%.1f%%). The label is too easy — "
            "consider increasing min_gain_pct or reducing forward_bars.",
            pos_rate * 100,
        )

    # 3. Train
    log.info("Step 3/3: Training LightGBM…")
    best_params = None
    if tune:
        log.info("Running Optuna hyperparameter search (100 trials)…")
        best_params = tune_hyperparams(X, y, weights=weights, n_trials=100, forward_bars=forward_bars)

    # Pass the target_precision dynamically down to the trainer
    bst, meta = train_model(X, y, weights=weights, params=best_params, forward_bars=forward_bars, target_precision=target_precision)

    # Add training metadata
    meta["symbols"] = sorted(data.keys())
    meta["months"] = months
    meta["label_params"] = {
        "forward_bars": forward_bars,
        "min_gain_pct": min_gain_pct,
        "atr_multiplier": atr_multiplier,
        "target_precision": target_precision,
    }
    meta["recency_weighting"] = {
        "enabled": config.ML_RECENCY_WEIGHT_ENABLED,
        "halflife_days": config.ML_RECENCY_HALFLIFE_DAYS if config.ML_RECENCY_WEIGHT_ENABLED else None,
        "min_weight": config.ML_RECENCY_MIN_WEIGHT if config.ML_RECENCY_WEIGHT_ENABLED else None,
        "weight_min": round(float(weights.min()), 4),
        "weight_max": round(float(weights.max()), 4),
    }
    meta["training_time_s"] = round(time.time() - t0, 1)

    save_model(bst, meta)

    # Print summary
    avg = meta["avg_metrics"]
    cal_thresh = meta.get("calibrated_threshold", 0.40)
    oof_m = meta.get("oof_metrics_at_calibrated", {})
    log.info("=" * 60)
    log.info("TRAINING COMPLETE in %.1fs", meta["training_time_s"])
    log.info("  Samples : %d (%d pos / %d neg)", meta["n_samples"],
             meta["n_positive"], meta["n_negative"])
    log.info("  CV Acc  : %.1f%%", avg["accuracy"] * 100)
    log.info("  CV Prec : %.1f%%", avg["precision"] * 100)
    log.info("  CV Rec  : %.1f%%", avg["recall"] * 100)
    log.info("  CV F1   : %.3f", avg["f1"])
    log.info("  CV AUC  : %.3f", avg["auc"])
    log.info("  Calibrated Threshold: %.4f", cal_thresh)
    if oof_m:
        log.info("  OOF @ %.3f: prec=%.1f%%  rec=%.1f%%  f1=%.3f",
                 cal_thresh,
                 oof_m.get("precision", 0) * 100,
                 oof_m.get("recall", 0) * 100,
                 oof_m.get("f1", 0))
    log.info("=" * 60)

    # Top 10 features
    top_feats = list(meta["feature_importance"].items())[:10]
    total_gain = sum(meta["feature_importance"].values())
    log.info("Top 10 features by importance (gain):")
    for fname, score in top_feats:
        pct = score / total_gain * 100 if total_gain > 0 else 0
        log.info("  %6d (%4.1f%%)  %s", score, pct, fname)

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
    parser.add_argument("--atr-multiplier", type=float, default=0.0,
                        help="ATR multiplier for volatility-adjusted labels (default: 0 = use static %%)")
    parser.add_argument("--target-precision", type=float, default=0.50,
                        help="Target precision for threshold calibration (default: 0.50)")
    args = parser.parse_args()

    run_training(
        symbols=args.symbols,
        months=args.months,
        tune=args.tune,
        forward_bars=args.forward_bars,
        min_gain_pct=args.min_gain_pct,
        atr_multiplier=args.atr_multiplier,
        target_precision=args.target_precision,
    )


if __name__ == "__main__":
    main()