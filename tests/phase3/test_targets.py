from __future__ import annotations
import pytest
import pandas as pd
import numpy as np
from src.phase3_intra.targets import add_forward_returns, add_barrier_labels, add_regime_label, _BARRIER_CONFIG


def make_df(close_prices: list[float]) -> pd.DataFrame:
    n = len(close_prices)
    tz = "Asia/Kolkata"
    start = pd.Timestamp("2020-01-02 09:15:00", tz=tz)
    bar_close = [start + pd.Timedelta(minutes=5 * i) for i in range(n)]
    return pd.DataFrame({
        "bar_close": bar_close,
        "close": close_prices,
    })


def make_random_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    close = 12000.0 * np.cumprod(1 + np.random.normal(0.0002, 0.001, n))
    return make_df(close.tolist())


# ---------------------------------------------------------------------------
# T3.1 Forward returns
# ---------------------------------------------------------------------------

def test_forward_returns_shape():
    df = make_random_df(50)
    result = add_forward_returns(df)
    new_cols = [c for c in result.columns if c.startswith("ret_fwd_")]
    assert len(new_cols) == 5
    for col in ["ret_fwd_1", "ret_fwd_3", "ret_fwd_6", "ret_fwd_12", "ret_fwd_24"]:
        assert col in result.columns


def test_forward_returns_last_rows_nan():
    df = make_random_df(50)
    result = add_forward_returns(df)
    for n in [1, 3, 6, 12, 24]:
        col = f"ret_fwd_{n}"
        tail = result[col].iloc[-n:]
        assert tail.isna().all(), f"Last {n} rows of {col} should be NaN"


def test_forward_returns_value():
    close = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]
    df = make_df(close)
    result = add_forward_returns(df)
    expected = (101.0 / 100.0) - 1.0
    assert abs(result["ret_fwd_1"].iloc[0] - expected) < 1e-9


# ---------------------------------------------------------------------------
# T3.2 Barrier labels
# ---------------------------------------------------------------------------

def test_barrier_labels_valid_values():
    df = make_random_df(100)
    result = add_barrier_labels(df, "5m")
    non_nan = result["barrier_hit"].dropna()
    assert set(non_nan.unique()).issubset({-1, 0, 1})


def test_barrier_labels_last_rows_nan():
    df = make_random_df(100)
    result = add_barrier_labels(df, "5m")
    _, window = _BARRIER_CONFIG["5m"]
    tail = result["barrier_hit"].iloc[-window:]
    assert tail.isna().all(), f"Last {window} rows of barrier_hit should be NaN"


def test_barrier_upper_hit():
    # Bar 0: close=100; bar 2 rises by 1% → upper barrier (0.20%) is hit first for 5m
    n = 30
    close = [100.0] * n
    # Make bar 2 jump above upper barrier (0.20%)
    close[2] = 100.0 * 1.01  # +1% >> 0.20% threshold
    df = make_df(close)
    result = add_barrier_labels(df, "5m")
    assert result["barrier_hit"].iloc[0] == 1, "Expected +1 when upper barrier hit first"


def test_barrier_lower_hit():
    # Bar 0: close=100; bar 2 drops by 1% → lower barrier (-0.20%) is hit first for 5m
    n = 30
    close = [100.0] * n
    close[2] = 100.0 * 0.99  # -1% >> 0.20% threshold downward
    df = make_df(close)
    result = add_barrier_labels(df, "5m")
    assert result["barrier_hit"].iloc[0] == -1, "Expected -1 when lower barrier hit first"


# ---------------------------------------------------------------------------
# T3.3 Regime label
# ---------------------------------------------------------------------------

def test_regime_valid_values():
    df = make_random_df(100)
    result = add_regime_label(df)
    non_nan = result["regime"].dropna()
    assert set(non_nan.unique()).issubset({"trend_up", "trend_down", "chop"})


def test_regime_first_rows_nan():
    df = make_random_df(50)
    window = 20
    result = add_regime_label(df, window=window)
    head = result["regime"].iloc[: window - 1]
    assert head.isna().all(), f"First {window - 1} rows should be NaN"


def test_regime_trend_up():
    # Strongly ascending price series → "trend_up"
    close = [float(i) * 0.1 for i in range(50)]  # 0.0, 0.1, 0.2, ...
    df = make_df(close)
    result = add_regime_label(df, window=20)
    # All non-NaN rows should be trend_up for a perfectly ascending series
    non_nan = result["regime"].dropna()
    assert (non_nan == "trend_up").all(), f"Expected all trend_up, got: {non_nan.unique()}"


def test_regime_chop():
    # Flat / oscillating series → "chop" (R² will be very low)
    np.random.seed(0)
    close = [100.0 + np.sin(i * 0.5) * 0.001 for i in range(50)]  # tiny oscillation around 100
    df = make_df(close)
    result = add_regime_label(df, window=20)
    non_nan = result["regime"].dropna()
    assert (non_nan == "chop").all(), f"Expected all chop, got: {non_nan.unique()}"
