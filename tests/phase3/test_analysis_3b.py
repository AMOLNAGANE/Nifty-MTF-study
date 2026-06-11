from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.phase3_intra.analysis_3b import (
    compute_cross_correlation,
    nested_regression_test,
    turn_confirmation_timing,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    df = pd.DataFrame()
    df["A_hist"] = np.random.normal(0, 1, n)
    df["B_roc"] = np.random.normal(0, 1, n)
    df["ret_fwd_1"] = np.random.normal(0.0002, 0.01, n)
    return df


# ---------------------------------------------------------------------------
# Test 1: cross-correlation length
# ---------------------------------------------------------------------------

def test_cross_correlation_length():
    df = make_synthetic_df()
    max_lag = 20
    result = compute_cross_correlation(df, "A_hist", "B_roc", max_lag=max_lag)
    assert isinstance(result, pd.Series), "Result must be a pd.Series"
    assert len(result) == 2 * max_lag + 1, (
        f"Expected {2 * max_lag + 1} entries, got {len(result)}"
    )
    assert result.index[0] == -max_lag, f"First index should be {-max_lag}, got {result.index[0]}"
    assert result.index[-1] == max_lag, f"Last index should be {max_lag}, got {result.index[-1]}"


# ---------------------------------------------------------------------------
# Test 2: lag=0 equals Pearson correlation
# ---------------------------------------------------------------------------

def test_cross_correlation_lag_zero_is_pearson():
    df = make_synthetic_df()
    result = compute_cross_correlation(df, "A_hist", "B_roc", max_lag=10)
    expected = df["A_hist"].corr(df["B_roc"])
    assert abs(result.loc[0] - expected) < 1e-10, (
        f"At lag=0, cross-correlation {result.loc[0]} != Pearson {expected}"
    )


# ---------------------------------------------------------------------------
# Test 3: known lead — col_a leads col_b by 3 bars
# ---------------------------------------------------------------------------

def test_cross_correlation_known_lead():
    """col_b = col_a.shift(3) means col_a leads col_b by 3.
    Peak cross-correlation should be near lag = -3."""
    np.random.seed(42)
    n = 200
    col_a = pd.Series(np.random.normal(0, 1, n))
    # col_b is col_a shifted forward 3 (col_a leads col_b by 3 bars)
    col_b = col_a.shift(3)
    df = pd.DataFrame({"col_a": col_a, "col_b": col_b})

    result = compute_cross_correlation(df, "col_a", "col_b", max_lag=10)
    peak_lag = result.idxmax()
    assert peak_lag == -3, (
        f"Expected peak cross-correlation at lag=-3 (A leads B), got {peak_lag}"
    )


# ---------------------------------------------------------------------------
# Test 4: nested regression returns all 7 keys
# ---------------------------------------------------------------------------

def test_nested_regression_returns_all_keys():
    df = make_synthetic_df(n=200)
    result = nested_regression_test(df, "ret_fwd_1", "A_hist", "B_roc")
    expected_keys = {"r2_restricted", "r2_unrestricted", "delta_r2",
                     "b_coef", "b_tstat", "b_pval", "n_obs"}
    assert set(result.keys()) == expected_keys, (
        f"Missing keys: {expected_keys - set(result.keys())}"
    )


# ---------------------------------------------------------------------------
# Test 5: delta_r2 > 0 when feature_b adds incremental info
# ---------------------------------------------------------------------------

def test_nested_regression_delta_r2_positive_when_b_adds_info():
    """feature_b is correlated with target but orthogonal to feature_a."""
    np.random.seed(42)
    n = 200
    feature_a = np.random.normal(0, 1, n)
    # Make feature_b orthogonal to feature_a but correlated with target
    feature_b = np.random.normal(0, 1, n)
    # Remove any correlation with feature_a via Gram-Schmidt
    feature_b = feature_b - (np.dot(feature_b, feature_a) / np.dot(feature_a, feature_a)) * feature_a
    # target depends on both feature_a and feature_b
    target = 0.3 * feature_a + 0.5 * feature_b + np.random.normal(0, 0.5, n)

    df = pd.DataFrame({"target": target, "feat_a": feature_a, "feat_b": feature_b})
    result = nested_regression_test(df, "target", "feat_a", "feat_b")
    assert result["delta_r2"] > 0, (
        f"Expected delta_r2 > 0, got {result['delta_r2']}"
    )


# ---------------------------------------------------------------------------
# Test 6: fewer than 30 obs returns NaN dict
# ---------------------------------------------------------------------------

def test_nested_regression_few_obs_returns_nan():
    np.random.seed(42)
    n = 25
    df = pd.DataFrame({
        "target": np.random.normal(0, 1, n),
        "feat_a": np.random.normal(0, 1, n),
        "feat_b": np.random.normal(0, 1, n),
    })
    result = nested_regression_test(df, "target", "feat_a", "feat_b")
    assert np.isnan(result["b_coef"]), (
        f"Expected NaN for b_coef with <30 obs, got {result['b_coef']}"
    )


# ---------------------------------------------------------------------------
# Test 7: turn_confirmation summary has all 5 keys
# ---------------------------------------------------------------------------

def test_turn_confirmation_all_keys_present():
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "A_hist_zero_cross": np.zeros(n, dtype=int),
        "B_roc": np.random.normal(0, 1, n),
    })
    _, summary = turn_confirmation_timing(df, "A_hist_zero_cross", "B_roc")
    expected_keys = {"n_flips", "n_confirmed", "n_not_confirmed",
                     "confirmation_rate", "median_bars_to_confirm"}
    assert set(summary.keys()) == expected_keys, (
        f"Missing keys: {expected_keys - set(summary.keys())}"
    )


# ---------------------------------------------------------------------------
# Test 8: detects confirmed event
# ---------------------------------------------------------------------------

def test_turn_confirmation_detects_confirmed():
    """A flip at row 5, B_roc crosses 0 at row 8 → confirmed=True, bars_to_confirm=3."""
    n = 40
    a_flip = np.zeros(n, dtype=int)
    a_flip[5] = 1  # flip at row 5

    b_roc = np.full(n, -0.5)   # negative by default
    b_roc[8] = 0.1             # crosses threshold at row 8 (3 bars after flip)

    df = pd.DataFrame({"A_flip": a_flip, "B_roc": b_roc})
    events_df, summary = turn_confirmation_timing(
        df, "A_flip", "B_roc", b_confirm_threshold=0.0, k_bars=20
    )

    assert summary["n_flips"] == 1
    assert summary["n_confirmed"] == 1
    assert len(events_df) == 1
    assert events_df.iloc[0]["confirmed"] == True
    assert events_df.iloc[0]["bars_to_confirm"] == 3


# ---------------------------------------------------------------------------
# Test 9: detects not-confirmed event
# ---------------------------------------------------------------------------

def test_turn_confirmation_not_confirmed():
    """A flip at row 5, B_roc stays negative for 20 bars → not_confirmed=1."""
    n = 50
    a_flip = np.zeros(n, dtype=int)
    a_flip[5] = 1

    b_roc = np.full(n, -0.5)  # always negative

    df = pd.DataFrame({"A_flip": a_flip, "B_roc": b_roc})
    events_df, summary = turn_confirmation_timing(
        df, "A_flip", "B_roc", b_confirm_threshold=0.0, k_bars=20
    )

    assert summary["n_not_confirmed"] == 1
    assert summary["n_confirmed"] == 0
    assert events_df.iloc[0]["confirmed"] == False
    assert np.isnan(events_df.iloc[0]["bars_to_confirm"])


# ---------------------------------------------------------------------------
# Test 10: no flips
# ---------------------------------------------------------------------------

def test_turn_confirmation_no_flips():
    """No A flips in df → n_flips=0, events_df is empty."""
    n = 30
    df = pd.DataFrame({
        "A_flip": np.zeros(n, dtype=int),
        "B_roc": np.random.normal(0, 1, n),
    })
    events_df, summary = turn_confirmation_timing(df, "A_flip", "B_roc")
    assert summary["n_flips"] == 0
    assert len(events_df) == 0
