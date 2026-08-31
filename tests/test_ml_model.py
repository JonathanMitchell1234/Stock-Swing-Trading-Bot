import numpy as np
import pandas as pd
from indicators import compute_all
import ml_features
import ml_model


def test_ml_features_count_and_prediction():
    assert len(ml_features.FEATURE_NAMES) == 54
    assert ml_features.NUM_FEATURES == 54

    dates = pd.date_range('2026-01-01', periods=80, freq='D')
    df = pd.DataFrame({
        'open': np.linspace(100, 120, 80) + np.random.randn(80),
        'high': np.linspace(105, 125, 80) + np.random.randn(80),
        'low': np.linspace(95, 115, 80) + np.random.randn(80),
        'close': np.linspace(100, 120, 80) + np.random.randn(80),
        'volume': np.random.randint(1000, 5000, 80),
    }, index=dates)
    df = compute_all(df)

    row = ml_features.extract_row(df, idx=-1)
    assert row is not None
    assert len(row) == 54

    if ml_model.is_available():
        prob = ml_model.predict_entry_proba(df, idx=-1)
        assert prob is not None
        assert 0.0 <= prob <= 1.0
