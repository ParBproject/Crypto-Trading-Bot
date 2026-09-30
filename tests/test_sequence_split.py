"""Scaler fit stays on the training prefix, and the training seed is stable."""

import numpy as np
import pandas as pd

from src.predictor import SequenceBuilder, set_training_seed, training_seed


def _frame(n=100) -> pd.DataFrame:
    close = np.array([10.0] * 50 + [1_000.0] * 50)
    data = {
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": np.ones(n),
        "rsi": np.full(n, 50.0),
        "macd": np.zeros(n),
        "macd_signal": np.zeros(n),
        "macd_hist": np.zeros(n),
        "atr": np.ones(n),
        "ema_20": close,
        "ema_50": close,
        "volume_ratio": np.ones(n),
        "bb_upper": close,
        "bb_lower": close,
    }
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(data, index=index)


def test_validation_tail_does_not_shift_the_scaler():
    df = _frame()
    builder = SequenceBuilder(sequence_length=10, forecast_horizon=1)
    x_train, y_train, x_val, y_val = builder.split_train_val(df, val_fraction=0.5)

    close_idx = builder.feature_columns.index("close")
    assert builder.scaler.center_[close_idx] == 10.0
    full_median = float(np.median(df["close"].to_numpy()))
    assert full_median == 505.0
    assert builder.scaler.center_[close_idx] != full_median
    assert len(x_train) > 0
    assert len(x_val) > 0
    assert len(y_train) == len(x_train)
    assert len(y_val) == len(x_val)


def test_training_seed_helper_and_numpy_repeat():
    assert training_seed({}) == 42
    assert training_seed({"model": {"seed": 7}}) == 7
    set_training_seed(123)
    first = np.random.random(4)
    set_training_seed(123)
    second = np.random.random(4)
    assert np.array_equal(first, second)
