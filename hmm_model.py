"""
HMM Regime Model – loads a trained Hidden Markov Model and provides
real-time regime predictions for the executor.

Singleton pattern with thread-safe lazy loading (same as ml_model.py).

Usage:
    from hmm_model import predict_regime, is_available

    if is_available():
        result = predict_regime(spy_df)
        # result = {"state": "BEAR", "probabilities": {"BULL": 0.02, "BEAR": 0.96, "CHOP": 0.02}}
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

log = get_logger("hmm_model")

MODEL_DIR   = Path(__file__).parent / "models"
MODEL_PATH  = MODEL_DIR / "hmm_regime.pkl"
META_PATH   = MODEL_DIR / "hmm_regime_meta.json"
SCALER_PATH = MODEL_DIR / "hmm_scaler.pkl"

# ── Singleton state ──────────────────────────────────────────
_model = None
_scaler = None   # StandardScaler fitted during training — MUST match model
_meta: dict = {}
_lock = threading.Lock()
_loaded = False


def _load() -> bool:
    """Load the HMM model, scaler, and metadata from disk. Returns True on success."""
    global _model, _scaler, _meta, _loaded
    if not MODEL_PATH.exists():
        log.debug("HMM model not found at %s", MODEL_PATH)
        return False
    try:
        import joblib
        _model = joblib.load(MODEL_PATH)

        # Scaler is required — features must be transformed identically to training
        if SCALER_PATH.exists():
            _scaler = joblib.load(SCALER_PATH)
        else:
            log.warning(
                "HMM scaler not found at %s — predictions will use unscaled features. "
                "Re-train the model to fix this.",
                SCALER_PATH,
            )
            _scaler = None

        if META_PATH.exists():
            with open(META_PATH) as f:
                _meta = json.load(f)
        _loaded = True
        log.info("HMM regime model loaded (%d states, trained %s, scaler=%s)",
                 _meta.get("n_states", "?"), _meta.get("trained_at", "?"),
                 "yes" if _scaler is not None else "missing")
        return True
    except Exception as exc:
        log.warning("Failed to load HMM model: %s", exc)
        _model = None
        _scaler = None
        _meta = {}
        _loaded = False
        return False


def _ensure_loaded() -> bool:
    """Thread-safe lazy loading."""
    global _loaded
    if _loaded:
        return True
    with _lock:
        if _loaded:
            return True
        return _load()


def is_available() -> bool:
    """Check whether a trained HMM model exists and can be loaded."""
    return _ensure_loaded() and _model is not None


def reload_model() -> bool:
    """Force-reload the model and scaler from disk (e.g. after re-training)."""
    global _loaded, _scaler
    with _lock:
        _loaded = False
        _scaler = None
        return _load()


def get_state_labels() -> dict[int, str]:
    """Return the state_id → label mapping from metadata."""
    _ensure_loaded()
    raw = _meta.get("state_labels", {})
    return {int(k): v for k, v in raw.items()}


def get_vol_window() -> int:
    """Return the volatility window used during training."""
    _ensure_loaded()
    return int(_meta.get("vol_window", 10))


def predict_regime(
    spy_df: pd.DataFrame,
    lookback: int = 30,
) -> Optional[dict]:
    """
    Predict the current market regime from recent SPY daily bars.

    Parameters:
        spy_df   : DataFrame with at least `close` column (daily bars, chronological)
        lookback : number of recent bars to feed into the HMM prediction

    Returns a dict:
        {
            "state": "BULL" | "BEAR" | "CHOP",
            "state_id": int,
            "probabilities": {"BULL": float, "BEAR": float, "CHOP": float},
        }
    Or None if prediction fails.
    """
    if not _ensure_loaded() or _model is None:
        return None

    vol_window = get_vol_window()
    state_labels = get_state_labels()

    # Need enough bars for vol_window + lookback
    min_bars = vol_window + lookback + 5
    if spy_df is None or len(spy_df) < min_bars:
        log.warning(
            "HMM predict: need %d bars, got %d — skipping",
            min_bars, len(spy_df) if spy_df is not None else 0,
        )
        return None

    try:
        # Build features from recent bars (same pipeline as training)
        close = spy_df["close"].astype(float)
        log_returns = np.log(close / close.shift(1))
        rolling_vol = log_returns.rolling(window=vol_window).std()

        features_df = pd.DataFrame({
            "log_return": log_returns,
            "volatility": rolling_vol,
        }, index=spy_df.index)
        features_df = features_df.dropna()

        if len(features_df) < 2:
            log.warning("HMM predict: not enough valid feature rows after dropna")
            return None

        # Use the last `lookback` rows (or all available if fewer)
        tail = features_df.tail(lookback)
        X = tail[["log_return", "volatility"]].values

        # Apply the same StandardScaler that was fitted during training.
        # Without this, the model sees features on a completely different
        # scale than what it was trained on — producing garbage predictions.
        if _scaler is not None:
            X = _scaler.transform(X)

        # Predict probabilities for the most recent observation using
        # the full recent sequence (Viterbi considers the sequence)
        state_sequence = _model.predict(X)
        current_state = int(state_sequence[-1])

        # Get posterior probabilities for the last observation
        # predict_proba returns per-sample state probabilities
        proba_matrix = _model.predict_proba(X)
        current_proba = proba_matrix[-1]  # shape (n_states,)

        # Build result dict
        probabilities = {}
        for sid, prob in enumerate(current_proba):
            label = state_labels.get(sid, f"STATE_{sid}")
            probabilities[label] = round(float(prob), 4)

        current_label = state_labels.get(current_state, f"STATE_{current_state}")

        return {
            "state": current_label,
            "state_id": current_state,
            "probabilities": probabilities,
        }

    except Exception as exc:
        log.warning("HMM regime prediction failed: %s", exc)
        return None
