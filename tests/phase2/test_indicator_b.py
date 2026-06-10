import pytest
import pandas as pd
import numpy as np
from src.phase2_features.indicator_b import compute_indicator_b

def make_synthetic_df(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """600 bars: covers EMA90 + ROC7 + zROC100 warmup."""
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

REQUIRED_B_COLS = [
    "B_gap", "B_gap_norm", "B_roc", "B_roc_invalid",
    "B_zroc", "B_gap_acc", "B_roc_slope", "B_zroc_extreme", "B_state",
]

def test_returns_all_required_columns():
    result = compute_indicator_b(make_synthetic_df())
    for col in REQUIRED_B_COLS:
        assert col in result.columns, f"Missing: {col}"

def test_input_columns_preserved():
    df = make_synthetic_df()
    result = compute_indicator_b(df)
    for col in df.columns:
        assert col in result.columns

def test_roc_no_inf():
    result = compute_indicator_b(make_synthetic_df())
    assert not result["B_roc"].isin([float("inf"), float("-inf")]).any()

def test_zroc_extreme_is_binary():
    result = compute_indicator_b(make_synthetic_df())
    assert set(result["B_zroc_extreme"].dropna().unique()).issubset({0, 1})

def test_roc_invalid_is_binary():
    result = compute_indicator_b(make_synthetic_df())
    assert set(result["B_roc_invalid"].dropna().unique()).issubset({0, 1})

def test_state_valid_values():
    valid = {"bull_accel", "bull_decel", "bear_decel", "bear_accel"}
    result = compute_indicator_b(make_synthetic_df())
    assert set(result["B_state"].dropna().unique()).issubset(valid)

def test_all_four_states_present():
    result = compute_indicator_b(make_synthetic_df(n=600))
    assert len(set(result["B_state"].dropna().unique())) == 4

def test_gap_norm_equals_gap_over_close():
    df = make_synthetic_df()
    result = compute_indicator_b(df)
    expected = result["B_gap"] / df["close"]
    pd.testing.assert_series_equal(
        result["B_gap_norm"].dropna(),
        expected.dropna(),
        check_names=False,
        rtol=1e-6,
    )

def test_gap_acc_equals_gap_minus_gap_lagged_7():
    result = compute_indicator_b(make_synthetic_df())
    expected = result["B_gap"] - result["B_gap"].shift(7)
    pd.testing.assert_series_equal(
        result["B_gap_acc"].dropna(),
        expected.dropna(),
        check_names=False,
        rtol=1e-6,
    )

def test_zroc_roughly_standardized():
    result = compute_indicator_b(make_synthetic_df(n=600))
    after = result.iloc[210:]["B_zroc"].dropna()
    assert abs(after.mean()) < 0.5
    assert 0.5 < after.std() < 2.0

def test_state_consistency_bull_accel():
    result = compute_indicator_b(make_synthetic_df(n=600)).dropna()
    mask = result["B_state"] == "bull_accel"
    assert (result.loc[mask, "B_roc"] > 0).all()
    assert (result.loc[mask, "B_roc_slope"] > 0).all()

def test_does_not_modify_input():
    df = make_synthetic_df()
    original_cols = set(df.columns)
    compute_indicator_b(df)
    assert set(df.columns) == original_cols
