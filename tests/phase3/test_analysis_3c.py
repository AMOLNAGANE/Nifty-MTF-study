from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.phase3_intra.analysis_3c import (
    build_joint_state_matrix,
    test_joint_state_significance,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STATES = ["bull_accel", "bull_decel", "bear_decel", "bear_accel"]


def make_synthetic_df(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Create a DataFrame with all 4 states in both A and B columns."""
    np.random.seed(seed)
    # Assign states in a round-robin + random fashion so all 16 cells are populated
    a_states = np.tile(STATES, n // len(STATES) + 1)[:n]
    np.random.shuffle(a_states)
    b_states = np.tile(STATES, n // len(STATES) + 1)[:n]
    np.random.shuffle(b_states)
    df = pd.DataFrame({
        "state_a": a_states,
        "state_b": b_states,
        "ret_fwd": np.random.normal(0.0002, 0.01, n),
    })
    return df


# ---------------------------------------------------------------------------
# Test 1: result has exactly 16 rows (4×4 cells)
# ---------------------------------------------------------------------------

def test_build_joint_matrix_shape():
    df = make_synthetic_df(n=400)
    result = build_joint_state_matrix(df, "state_a", "state_b", "ret_fwd")
    assert len(result) == 16, f"Expected 16 rows, got {len(result)}"


# ---------------------------------------------------------------------------
# Test 2: result has exactly the right columns
# ---------------------------------------------------------------------------

def test_build_joint_matrix_columns():
    df = make_synthetic_df(n=400)
    result = build_joint_state_matrix(df, "state_a", "state_b", "ret_fwd")
    expected_cols = {"count", "mean_ret", "hit_rate", "t_stat", "p_val_raw"}
    assert set(result.columns) == expected_cols, (
        f"Expected columns {expected_cols}, got {set(result.columns)}"
    )


# ---------------------------------------------------------------------------
# Test 3: sum of count column equals len(df)
# ---------------------------------------------------------------------------

def test_build_joint_matrix_count_sums_to_n():
    df = make_synthetic_df(n=400)
    result = build_joint_state_matrix(df, "state_a", "state_b", "ret_fwd")
    assert result["count"].sum() == len(df), (
        f"Sum of counts {result['count'].sum()} != len(df) {len(df)}"
    )


# ---------------------------------------------------------------------------
# Test 4: all non-NaN hit_rate values are in [0, 1]
# ---------------------------------------------------------------------------

def test_build_joint_matrix_hit_rate_range():
    df = make_synthetic_df(n=400)
    result = build_joint_state_matrix(df, "state_a", "state_b", "ret_fwd")
    valid = result["hit_rate"].dropna()
    assert (valid >= 0).all() and (valid <= 1).all(), (
        f"hit_rate values out of [0,1]: {valid[~((valid >= 0) & (valid <= 1))]}"
    )


# ---------------------------------------------------------------------------
# Test 5: cell with only 3 rows has mean_ret = NaN (below threshold 5)
# ---------------------------------------------------------------------------

def test_build_joint_matrix_low_count_nan():
    """A cell with count < 5 must have mean_ret = NaN."""
    np.random.seed(42)
    # Build a df where bull_accel + bear_accel has only 3 rows
    base = make_synthetic_df(n=400)
    # Remove all bull_accel+bear_accel rows, then add back exactly 3
    mask_rare = (base["state_a"] == "bull_accel") & (base["state_b"] == "bear_accel")
    df_no_rare = base[~mask_rare].copy()
    rare_rows = pd.DataFrame({
        "state_a": ["bull_accel"] * 3,
        "state_b": ["bear_accel"] * 3,
        "ret_fwd": np.random.normal(0, 0.01, 3),
    })
    df = pd.concat([df_no_rare, rare_rows], ignore_index=True)

    result = build_joint_state_matrix(df, "state_a", "state_b", "ret_fwd")
    cell = result.loc[("bull_accel", "bear_accel")]
    assert cell["count"] == 3, f"Expected count=3, got {cell['count']}"
    assert pd.isna(cell["mean_ret"]), f"Expected mean_ret=NaN for count=3, got {cell['mean_ret']}"


# ---------------------------------------------------------------------------
# Test 6: cell with count < 20 has t_stat = NaN
# ---------------------------------------------------------------------------

def test_build_joint_matrix_tstat_nan_when_count_lt_20():
    """A cell with exactly 10 rows must have t_stat = NaN."""
    np.random.seed(42)
    base = make_synthetic_df(n=400)
    mask_rare = (base["state_a"] == "bull_accel") & (base["state_b"] == "bear_accel")
    df_no_rare = base[~mask_rare].copy()
    rare_rows = pd.DataFrame({
        "state_a": ["bull_accel"] * 10,
        "state_b": ["bear_accel"] * 10,
        "ret_fwd": np.random.normal(0, 0.01, 10),
    })
    df = pd.concat([df_no_rare, rare_rows], ignore_index=True)

    result = build_joint_state_matrix(df, "state_a", "state_b", "ret_fwd")
    cell = result.loc[("bull_accel", "bear_accel")]
    assert cell["count"] == 10, f"Expected count=10, got {cell['count']}"
    # mean_ret should NOT be NaN (count >= 5)
    assert not pd.isna(cell["mean_ret"]), f"Expected mean_ret not NaN for count=10"
    # t_stat MUST be NaN (count < 20)
    assert pd.isna(cell["t_stat"]), f"Expected t_stat=NaN for count=10, got {cell['t_stat']}"


# ---------------------------------------------------------------------------
# Test 7: result dict has all 5 keys
# ---------------------------------------------------------------------------

def test_significance_has_all_keys():
    df = make_synthetic_df(n=400)
    matrix = build_joint_state_matrix(df, "state_a", "state_b", "ret_fwd")
    result = test_joint_state_significance(matrix)
    expected_keys = {"chi2_stat", "chi2_pval", "n_significant_cells",
                     "diagonal_mean_ret", "offdiag_mean_ret"}
    assert set(result.keys()) == expected_keys, (
        f"Missing keys: {expected_keys - set(result.keys())}"
    )


# ---------------------------------------------------------------------------
# Test 8: chi2_stat >= 0
# ---------------------------------------------------------------------------

def test_significance_chi2_nonnegative():
    df = make_synthetic_df(n=400)
    matrix = build_joint_state_matrix(df, "state_a", "state_b", "ret_fwd")
    result = test_joint_state_significance(matrix)
    assert result["chi2_stat"] >= 0, f"chi2_stat must be >= 0, got {result['chi2_stat']}"


# ---------------------------------------------------------------------------
# Test 9: diagonal_mean_ret > offdiag_mean_ret when diagonal has high returns
# ---------------------------------------------------------------------------

def test_significance_diagonal_vs_offdiag():
    """When bull_accel+bull_accel rows have very high returns and all others are
    near zero, diagonal_mean_ret must be greater than offdiag_mean_ret."""
    np.random.seed(42)
    n_per_cell = 30  # enough for t-stat

    rows = []
    for sa in STATES:
        for sb in STATES:
            ret_mean = 0.5 if (sa == "bull_accel" and sb == "bull_accel") else 0.0
            rows.append(pd.DataFrame({
                "state_a": [sa] * n_per_cell,
                "state_b": [sb] * n_per_cell,
                "ret_fwd": np.random.normal(ret_mean, 0.001, n_per_cell),
            }))
    df = pd.concat(rows, ignore_index=True)

    matrix = build_joint_state_matrix(df, "state_a", "state_b", "ret_fwd")
    result = test_joint_state_significance(matrix)
    assert result["diagonal_mean_ret"] > result["offdiag_mean_ret"], (
        f"Expected diagonal_mean_ret ({result['diagonal_mean_ret']}) > "
        f"offdiag_mean_ret ({result['offdiag_mean_ret']})"
    )
