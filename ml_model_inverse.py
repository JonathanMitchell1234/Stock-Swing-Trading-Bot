"""
ML Model wrapper for INVERSE ETF entries – loads a trained LightGBM model and
provides predictions for bear-mode inverse ETF buying.

Separate from ml_model.py and ml_model_short.py so all three models (long,
short, inverse) are fully independent — different model files, different
metadata, different loading state.

Lazy-loaded: the model is only loaded from disk when first needed.
Thread-safe for dashboard use (FastAPI runs in threads).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config
from logger import get_logger
from ml_features import FEATURE_NAMES, NUM_FEATURES
from ml_features_inverse import extract_row_inverse

log = get_logger("ml_model_inverse")

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "gbm_inverse.txt"
META_PATH = MODEL_DIR / "gbm_inverse_meta.json"

# ─────────────────────────────────────────────────────────────
# Singleton model cache
# ─────────────────────────────────────────────────────────────
_lock = threading.Lock()
_booster = None          # type: Optional[lgb.Booster]
_meta: dict | None = None
_loaded = False


def _ensure_loaded() -> bool:
    """Load the model from disk if not yet loaded. Returns True if available."""
    global _booster, _meta, _loaded

    if _loaded:
        return _booster is not None

    with _lock:
        if _loaded:
            return _booster is not None

        if not MODEL_PATH.exists():
            log.info("No inverse GBM model found at %s — ML inverse scoring disabled", MODEL_PATH)
            _loaded = True
            return False

        try:
            import lightgbm as lgb
            _booster = lgb.Booster(model_file=str(MODEL_PATH))
            log.info("Loaded inverse GBM model from %s (%d trees)",
                     MODEL_PATH, _booster.num_trees())
        except Exception as exc:
            log.warning("Failed to load inverse GBM model: %s", exc)
            _booster = None

        if META_PATH.exists():
            try:
                _meta = json.loads(META_PATH.read_text())
            except Exception:
                _meta = None

        _loaded = True
        return _booster is not None


def reload_model() -> bool:
    """Force-reload the model from disk (e.g. after re-training)."""
    global _booster, _meta, _loaded
    with _lock:
        _loaded = False
        _booster = None
        _meta = None
    return _ensure_loaded()


def is_available() -> bool:
    """Check if a trained inverse model is available."""
    return _ensure_loaded()


def get_meta() -> dict | None:
    """Return the model metadata (training metrics, etc.)."""
    _ensure_loaded()
    return _meta


# ─────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────

def predict_inverse_proba(
    df: pd.DataFrame,
    idx: int = -1,
    weekly_bullish: bool = True,
    spy_df: pd.DataFrame | None = None,
    vixy_df: pd.DataFrame | None = None,
) -> float | None:
    """
    Predict the probability that a LONG buy on an inverse ETF at bar *idx*
    leads to a profitable trade (price gains by min_gain_pct within forward_bars).

    Returns a float in [0, 1], or None if the model isn't loaded or
    features can't be extracted.
    """
    if not _ensure_loaded():
        return None

    fv = extract_row_inverse(df, idx=idx, weekly_bullish=weekly_bullish,
                             spy_df=spy_df, vixy_df=vixy_df)
    if fv is None:
        return None

    X = np.array([fv], dtype=np.float32)
    try:
        prob = float(_booster.predict(X)[0])
        return prob
    except Exception as exc:
        log.warning("Inverse GBM prediction failed: %s", exc)
        return None


def predict_inverse_batch(
    feature_matrix: np.ndarray,
) -> np.ndarray | None:
    """
    Batch prediction for a feature matrix (N x NUM_FEATURES).
    Returns an array of probabilities, or None if model unavailable.
    """
    if not _ensure_loaded():
        return None

    try:
        return _booster.predict(feature_matrix)
    except Exception as exc:
        log.warning("Inverse GBM batch prediction failed: %s", exc)
        return None
