from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.phase3_intra.analysis_3a import (
    compute_correlations,
    compute_decile_returns,
    compute_state_conditional_returns,
    apply_multiple_testing_correction,
    plot_decile_curves,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STATES = ["bull_accel", "bull_decel", "bear_decel", "bear_accel"]
_HORIZONS = ["ret_fwd_1", "ret_fwd_3", "ret_fwd_6", "ret_fwd_12", "ret_fwd_24"]
_FEATURES = ["A_hist", "A_macd", "B_roc"]


def make_synthetic_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    df = pd.DataFrame()
    # Feature columns
    df["A_hist"] = np.random.normal(0, 1, n)
    df["A_macd"] = np.random.normal(0, 1, n)
    df["B_roc"] = np.random.normal(0, 1, n)
    # Horizon (forward return) columns — small random returns
    for h in _HORIZONS:
        df[h] = np.random.normal(0.0002, 0.01, n)
    # State column — 4 states equally distributed
    states = np.tile(_STATES, n // 4 + 1)[:n]
    df["A_state"] = states
    return df


# ---------------------------------------------------------------------------
# T1: compute_correlations shape
# ---------------------------------------------------------------------------

def test_compute_correlations_shape():
    df = make_synthetic_df()
    pearson_df, spearman_df = compute_correlations(df, _FEATURES, _HORIZONS)
    assert pearson_df.shape == (len(_FEATURES), len(_HORIZONS)), (
        f"Pearson shape mismatch: {pearson_df.shape}"
    )
    assert spearman_df.shape == (len(_FEATURES), len(_HORIZONS)), (
        f"Spearman shape mismatch: {spearman_df.shape}"
    )
    assert list(pearson_df.index) == _FEATURES
    assert list(pearson_df.columns) == _HORIZONS


# ---------------------------------------------------------------------------
# T2: compute_correlations known value (perfect correlation)
# ---------------------------------------------------------------------------

def test_compute_correlations_known_value():
    np.random.seed(42)
    n = 200
    x = np.random.normal(0, 1, n)
    df = pd.DataFrame({
        "feature_x": x,
        "ret_fwd_1": x,  # perfect linear relationship
    })
    pearson_df, spearman_df = compute_correlations(df, ["feature_x"], ["ret_fwd_1"])
    r = pearson_df.loc["feature_x", "ret_fwd_1"]
    assert abs(r - 1.0) < 1e-9, f"Expected Pearson r ≈ 1.0, got {r}"
    rs = spearman_df.loc["feature_x", "ret_fwd_1"]
    assert abs(rs - 1.0) < 1e-9, f"Expected Spearman r ≈ 1.0, got {rs}"


# ---------------------------------------------------------------------------
# T3: compute_decile_returns shape
# ---------------------------------------------------------------------------

def test_compute_decile_returns_shape():
    df = make_synthetic_df()
    result = compute_decile_returns(df, "A_hist", _HORIZONS)
    assert len(result) == 10, f"Expected 10 decile rows, got {len(result)}"
    for h in _HORIZONS:
        assert h in result.columns, f"Missing horizon column: {h}"
    assert "count" in result.columns


# ---------------------------------------------------------------------------
# T4: compute_decile_returns monotonic (feature = return)
# ---------------------------------------------------------------------------

def test_compute_decile_returns_monotonic():
    np.random.seed(42)
    n = 200
    x = np.random.normal(0, 1, n)
    df = pd.DataFrame({
        "feature_x": x,
        "ret_fwd_1": x,
    })
    result = compute_decile_returns(df, "feature_x", ["ret_fwd_1"])
    means = result["ret_fwd_1"].values
    # Decile means should be non-decreasing (lowest decile = lowest return)
    assert all(means[i] <= means[i + 1] for i in range(len(means) - 1)), (
        f"Expected monotonically increasing means, got: {means}"
    )


# ---------------------------------------------------------------------------
# T5: compute_decile_returns empty for constant feature
# ---------------------------------------------------------------------------

def test_compute_decile_returns_empty_constant_feature():
    df = pd.DataFrame({
        "const_feat": [1.0] * 100,
        "ret_fwd_1": np.random.normal(0, 0.01, 100),
    })
    result = compute_decile_returns(df, "const_feat", ["ret_fwd_1"])
    assert result.empty, "Expected empty DataFrame for constant feature"


# ---------------------------------------------------------------------------
# T6: compute_state_conditional_returns shape
# ---------------------------------------------------------------------------

def test_state_conditional_returns_shape():
    df = make_synthetic_df()
    result = compute_state_conditional_returns(df, "A_state", _HORIZONS)
    n_states = df["A_state"].nunique()
    assert len(result) == n_states * len(_HORIZONS), (
        f"Expected {n_states * len(_HORIZONS)} rows, got {len(result)}"
    )


# ---------------------------------------------------------------------------
# T7: compute_state_conditional_returns columns
# ---------------------------------------------------------------------------

def test_state_conditional_returns_has_hac_columns():
    df = make_synthetic_df()
    result = compute_state_conditional_returns(df, "A_state", _HORIZONS)
    for col in ["mean", "count", "t_stat", "p_val_raw"]:
        assert col in result.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# T8: compute_state_conditional_returns low count → NaN
# ---------------------------------------------------------------------------

def test_state_conditional_returns_low_count_nan():
    np.random.seed(42)
    # Create a small group: "rare_state" with only 5 observations
    n = 200
    states = ["common"] * (n - 5) + ["rare_state"] * 5
    np.random.shuffle(states)
    df = pd.DataFrame({
        "A_state": states,
        "ret_fwd_1": np.random.normal(0, 0.01, n),
    })
    result = compute_state_conditional_returns(df, "A_state", ["ret_fwd_1"])
    # Find the rare_state row
    rare_row = result.xs("rare_state", level="state") if "state" in result.index.names else result.loc["rare_state"]
    t = rare_row["t_stat"].values[0] if hasattr(rare_row["t_stat"], "values") else rare_row["t_stat"]
    p = rare_row["p_val_raw"].values[0] if hasattr(rare_row["p_val_raw"], "values") else rare_row["p_val_raw"]
    assert np.isnan(t), f"Expected t_stat=NaN for low-count cell, got {t}"
    assert np.isnan(p), f"Expected p_val_raw=NaN for low-count cell, got {p}"


# ---------------------------------------------------------------------------
# T9: apply_multiple_testing_correction adds p_val_adj column
# ---------------------------------------------------------------------------

def test_apply_multiple_testing_correction_adds_adj_col():
    df = make_synthetic_df()
    raw = compute_state_conditional_returns(df, "A_state", _HORIZONS)
    result = apply_multiple_testing_correction(raw)
    assert "p_val_adj" in result.columns, "Missing p_val_adj column"
    assert "p_val_raw" in result.columns, "p_val_raw should still be present"


# ---------------------------------------------------------------------------
# T10: apply_multiple_testing_correction bonferroni
# ---------------------------------------------------------------------------

def test_apply_multiple_testing_correction_bonferroni():
    # Create a simple DataFrame with known p-values (no NaNs)
    p_vals = np.array([0.01, 0.05, 0.10, 0.20, 0.50])
    n = len(p_vals)
    df = pd.DataFrame({"p_val_raw": p_vals})
    result = apply_multiple_testing_correction(df, p_col="p_val_raw", method="bonferroni")
    expected = np.minimum(p_vals * n, 1.0)
    np.testing.assert_allclose(
        result["p_val_adj"].values,
        expected,
        atol=1e-9,
        err_msg="Bonferroni corrected p-values do not match expected",
    )


# ---------------------------------------------------------------------------
# T11: plot_decile_curves creates file
# ---------------------------------------------------------------------------

def test_plot_decile_curves_creates_file(tmp_path: Path):
    df = make_synthetic_df()
    decile_df = compute_decile_returns(df, "A_hist", _HORIZONS)
    plot_decile_curves(decile_df, feature_name="A_hist", tf="5m", out_dir=tmp_path)
    expected_file = tmp_path / "decile_5m_A_hist.png"
    assert expected_file.exists(), f"Expected plot file not found: {expected_file}"


# ---------------------------------------------------------------------------
# T12: plot_decile_curves empty skips silently
# ---------------------------------------------------------------------------

def test_plot_decile_curves_empty_skips(tmp_path: Path):
    empty_df = pd.DataFrame()
    # Should not raise, no file created
    plot_decile_curves(empty_df, feature_name="const_feat", tf="5m", out_dir=tmp_path)
    expected_file = tmp_path / "decile_5m_const_feat.png"
    assert not expected_file.exists(), "No file should be created for empty DataFrame"
