"""
ML Feature Extraction for INVERSE ETF entries – builds a fixed-width feature
vector from an indicator-enriched DataFrame for inverse ETFs like SQQQ, SPXS.

Mirrors ml_features.py exactly (same feature layout, same score_entry for the
hand-crafted score, same GAIN-based labels) because we are BUYING inverse ETFs
long.  The only difference is the training data: this module is used alongside
ml_trainer_inverse.py which trains exclusively on INVERSE_WATCHLIST symbols.

The model learns the unique hybrid pattern:
  Bullish technicals (on the inverse ETF chart) + Bearish macro = High probability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

# Re-use the EXACT same feature extraction from the long model.
# Inverse ETFs are bought long, so score_entry() and gain-based labels
# are the correct choice — NOT the short scoring functions.
from ml_features import (
    FEATURE_NAMES,
    NUM_FEATURES,
    extract_row,
    build_feature_matrix,
    generate_labels,
)

# Re-export everything the trainer and model modules need
__all__ = [
    "FEATURE_NAMES",
    "NUM_FEATURES",
    "extract_row_inverse",
    "build_inverse_feature_matrix",
    "generate_inverse_labels",
]


# ─────────────────────────────────────────────────────────────
# Per-row extraction (inverse variant)
# ─────────────────────────────────────────────────────────────

def extract_row_inverse(
    df: pd.DataFrame,
    idx: int = -1,
    weekly_bullish: bool = True,
    spy_df: pd.DataFrame | None = None,
    vixy_df: pd.DataFrame | None = None,
) -> list[float] | None:
    """
    Extract a feature vector for INVERSE ETF entry prediction.

    Uses the identical feature layout and scoring as the long model
    because inverse ETFs are bought long.  The model differentiation
    comes entirely from the training data (inverse ETF symbols only).
    """
    return extract_row(df, idx=idx, weekly_bullish=weekly_bullish,
                       spy_df=spy_df, vixy_df=vixy_df)


# ─────────────────────────────────────────────────────────────
# Bulk extraction (for training)
# ─────────────────────────────────────────────────────────────

def build_inverse_feature_matrix(
    df: pd.DataFrame,
    start_idx: int = 60,
    end_idx: int | None = None,
    spy_df: pd.DataFrame | None = None,
    vixy_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, list[int]]:
    """
    Extract inverse-ETF features for every bar from *start_idx* to *end_idx*.
    Returns (X, valid_indices).
    """
    return build_feature_matrix(df, start_idx=start_idx, end_idx=end_idx,
                                spy_df=spy_df, vixy_df=vixy_df)


def generate_inverse_labels(
    df: pd.DataFrame,
    indices: list[int],
    forward_bars: int = 5,
    min_gain_pct: float = 0.03,
    **_kwargs,
) -> np.ndarray:
    """
    Generate binary labels for INVERSE ETF training.

    Target: "Does the closing price GAIN >= *min_gain_pct* above the
    entry price within *forward_bars* bars?"

    Identical to the long model's generate_labels() because we are
    buying inverse ETFs long — price going UP is the success condition.
    """
    return generate_labels(df, indices, forward_bars=forward_bars,
                           min_gain_pct=min_gain_pct)
