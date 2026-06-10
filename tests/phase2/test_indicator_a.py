import pytest
import pandas as pd
import numpy as np
from src.phase2_features.indicator_a import compute_indicator_a

def make_synthetic_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    tz = "Asia/Kolkata"
    start = pd.Timestamp("2020-01-02 09:15:00", tz=tz)
    timestamps = [start + pd.Timedelta(minutes=5 * i) for i in range(n)]
    close = 12000.0 * np.cumprod(1 + np.random.normal(0.0002, 0.001, n))
    return pd.DataFrame({
        "timestamp": timestamps,
        "bar_close": [t + pd.Timedelta(minutes=5) for t in timestamps],
        "open": close * 0.9995,
        "high": close * 1.001,
        "low": close * 0.999,
        "close": close,
        "session_date": [t.date() for t in timestamps],
        "bar_in_session": np.arange(1, n + 1),
    })

REQUIRED_A_COLS = [
    "A_macd", "A_signal", "A_hist", "A_hist_slope",
    "A_macd_sign", "A_hist_sign",
    "A_macd_zero_cross", "A_hist_zero_cross",
    "A_state",
]

def test_returns_all_required_columns():
    result = compute_indicator_a(make_synthetic_df())
    for col in REQUIRED_A_COLS:
        assert col in result.columns, f"Missing: {col}"

def test_input_columns_preserved():
    df = make_synthetic_df()
    result = compute_indicator_a(df)
    for col in df.columns:
        assert col in result.columns

def test_macd_no_nan_after_warmup():
    result = compute_indicator_a(make_synthetic_df(n=500))
    after = result.iloc[50:]
    for col in ["A_macd", "A_signal", "A_hist"]:
        assert after[col].isna().sum() == 0, f"{col} has NaN after warmup"

def test_hist_slope_equals_hist_diff():
    result = compute_indicator_a(make_synthetic_df())
    expected = result["A_hist"].diff()
    pd.testing.assert_series_equal(
        result["A_hist_slope"].iloc[1:].reset_index(drop=True),
        expected.iloc[1:].reset_index(drop=True),
        check_names=False,
        rtol=1e-6,
    )

def test_sign_values_in_minus1_0_1():
    result = compute_indicator_a(make_synthetic_df())
    for col in ["A_macd_sign", "A_hist_sign"]:
        assert set(result[col].dropna().unique()).issubset({-1, 0, 1})

def test_zero_cross_values_in_minus1_0_1():
    result = compute_indicator_a(make_synthetic_df())
    for col in ["A_macd_zero_cross", "A_hist_zero_cross"]:
        assert set(result[col].dropna().unique()).issubset({-1, 0, 1})

def test_hist_zero_cross_fires_on_sign_change():
    result = compute_indicator_a(make_synthetic_df(n=200))
    assert (result["A_hist_zero_cross"] == 1).sum() > 0
    assert (result["A_hist_zero_cross"] == -1).sum() > 0

def test_state_valid_values():
    valid = {"bull_accel", "bull_decel", "bear_decel", "bear_accel"}
    result = compute_indicator_a(make_synthetic_df())
    assert set(result["A_state"].dropna().unique()).issubset(valid)

def test_all_four_states_present():
    result = compute_indicator_a(make_synthetic_df(n=500))
    assert len(set(result["A_state"].dropna().unique())) == 4

def test_state_consistency_bull_accel():
    result = compute_indicator_a(make_synthetic_df(n=500)).dropna()
    mask = result["A_state"] == "bull_accel"
    assert (result.loc[mask, "A_hist"] > 0).all()
    assert (result.loc[mask, "A_hist_slope"] > 0).all()

def test_state_consistency_bear_accel():
    result = compute_indicator_a(make_synthetic_df(n=500)).dropna()
    mask = result["A_state"] == "bear_accel"
    assert (result.loc[mask, "A_hist"] < 0).all()
    assert (result.loc[mask, "A_hist_slope"] < 0).all()

def test_does_not_modify_input():
    df = make_synthetic_df()
    original_cols = set(df.columns)
    compute_indicator_a(df)
    assert set(df.columns) == original_cols
