"""
ML Model wrapper for SHORT entries – loads a trained LightGBM model and
provides predictions for bear-mode short selling.

Separate from ml_model.py so the two models (long vs short) are fully
independent — different model files, different metadata, different
loading state.

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
from ml_features_short import extract_row_short

log = get_logger("ml_model_short")

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "gbm_short.txt"
META_PATH = MODEL_DIR / "gbm_short_meta.json"

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
            log.info("No short GBM model found at %s — ML short scoring disabled", MODEL_PATH)
            _loaded = True
            return False

        try:
            import lightgbm as lgb
            _booster = lgb.Booster(model_file=str(MODEL_PATH))
            log.info("Loaded short GBM model from %s (%d trees)",
                     MODEL_PATH, _booster.num_trees())
        except Exception as exc:
            log.warning("Failed to load short GBM model: %s", exc)
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
    """Check if a trained short model is available."""
    return _ensure_loaded()


def get_meta() -> dict | None:
    """Return the model metadata (training metrics, etc.)."""
    _ensure_loaded()
    return _meta


# ─────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────

def predict_short_proba(
    df: pd.DataFrame,
    idx: int = -1,
    weekly_bullish: bool = True,
    spy_df: pd.DataFrame | None = None,
    vixy_df: pd.DataFrame | None = None,
) -> float | None:
    """
    Predict the probability that a SHORT entry at bar *idx* leads to a
    profitable short trade (price drops by min_drop_pct within forward_bars).

    Returns a float in [0, 1], or None if the model isn't loaded or
    features can't be extracted.
    """
    if not _ensure_loaded():
        return None

    fv = extract_row_short(df, idx=idx, weekly_bullish=weekly_bullish,
                           spy_df=spy_df, vixy_df=vixy_df)
    if fv is None:
        return None

    X = np.array([fv], dtype=np.float32)
    try:
        prob = float(_booster.predict(X)[0])
        return prob
    except Exception as exc:
        log.warning("Short GBM prediction failed: %s", exc)
        return None


def predict_short_batch(
    feature_matrix: np.ndarray,
) -> np.ndarray | None:
    """
    Batch prediction for a feature matrix (N × NUM_FEATURES).
    Returns an array of probabilities, or None if model unavailable.
    """
    if not _ensure_loaded():
        return None

    try:
        return _booster.predict(feature_matrix)
    except Exception as exc:
        log.warning("Short GBM batch prediction failed: %s", exc)
        return None
