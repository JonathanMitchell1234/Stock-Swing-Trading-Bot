"""
ML Feature Extraction for SHORT entries – builds a fixed-width feature vector
from an indicator-enriched DataFrame, using the short-entry scoring system.

Mirrors ml_features.py but uses score_short_entry instead of score_entry,
and generates SHORT labels (price drops below entry within N bars).

The feature vector is IDENTICAL in layout to the long model so both models
share the same indicator infrastructure — only the hand-crafted score and
the training labels differ.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

# Re-use the same feature extraction machinery and names from the long model.
# The only difference is:
#   1. The "entry_score" feature uses the SHORT scoring function.
#   2. generate_short_labels looks for DROPS instead of gains.
from ml_features import (
    FEATURE_NAMES,
    NUM_FEATURES,
    _safe,
    _safe_div,
    extract_row as _extract_row_long,
)
from strategy import score_entry  # bull score (used by the long feature extractor)

# Import the short scoring function
from strategy import _score_short_entry_details


# ─────────────────────────────────────────────────────────────
# Per-row extraction (short variant)
# ─────────────────────────────────────────────────────────────

def extract_row_short(
    df: pd.DataFrame,
    idx: int = -1,
    weekly_bullish: bool = True,
    spy_df: pd.DataFrame | None = None,
    vixy_df: pd.DataFrame | None = None,
) -> list[float] | None:
    """
    Extract a feature vector for SHORT entry prediction.

    Uses the same feature layout as the long model, but replaces the
    "entry_score" feature with the short entry score so the model
    learns which short setups are high quality.
    """
    # Get the base feature vector from the long extractor
    fv = _extract_row_long(df, idx=idx, weekly_bullish=weekly_bullish,
                           spy_df=spy_df, vixy_df=vixy_df)
    if fv is None:
        return None

    # Replace the entry_score feature (index 38 — "entry_score" in FEATURE_NAMES)
    score_idx = FEATURE_NAMES.index("entry_score")

    abs_idx = idx if idx >= 0 else len(df) + idx
    sub_df = df.iloc[max(0, abs_idx - 80):abs_idx + 1]
    try:
        details = _score_short_entry_details(sub_df, weekly_bullish=weekly_bullish)
        short_score = details["score"]
    except Exception:
        short_score = 0

    fv[score_idx] = float(short_score)
    return fv


# ─────────────────────────────────────────────────────────────
# Bulk extraction (for training)
# ─────────────────────────────────────────────────────────────

def build_short_feature_matrix(
    df: pd.DataFrame,
    start_idx: int = 60,
    end_idx: int | None = None,
    spy_df: pd.DataFrame | None = None,
    vixy_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, list[int]]:
    """
    Extract short-entry features for every bar from *start_idx* to *end_idx*.
    Returns (X, valid_indices).
    """
    end_idx = end_idx or len(df)
    rows: list[list[float]] = []
    indices: list[int] = []

    for i in range(start_idx, end_idx):
        fv = extract_row_short(df, idx=i, spy_df=spy_df, vixy_df=vixy_df)
        if fv is not None:
            rows.append(fv)
            indices.append(i)

    if not rows:
        return np.empty((0, NUM_FEATURES)), []
    return np.array(rows, dtype=np.float32), indices


def generate_short_labels(
    df: pd.DataFrame,
    indices: list[int],
    forward_bars: int = 5,
    min_drop_pct: float = 0.03,
    **_kwargs,
) -> np.ndarray:
    """
    Generate binary labels for SHORT training.

    Target: "Does the closing price DROP ≥ *min_drop_pct* below the
    entry price within *forward_bars* bars?"

    Entry is assumed at the NEXT bar's open (short sell).
      - label = 1 if any close in [entry_bar .. entry_bar+forward_bars]
                  is ≤ entry_price × (1 - min_drop_pct)
      - label = 0 otherwise
    """
    labels = np.zeros(len(indices), dtype=np.int32)
    n = len(df)

    for i, bar_idx in enumerate(indices):
        entry_bar = bar_idx + 1  # enter on next bar's open
        if entry_bar >= n:
            continue

        entry_price = df.iloc[entry_bar]["open"]
        if pd.isna(entry_price) or entry_price <= 0:
            continue

        target = entry_price * (1.0 - min_drop_pct)

        # Walk forward — did any close drop to our profit target?
        end = min(entry_bar + forward_bars, n)
        for j in range(entry_bar, end):
            if df.iloc[j]["close"] <= target:
                labels[i] = 1
                break

    return labels
