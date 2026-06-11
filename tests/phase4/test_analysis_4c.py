from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.phase4_inter.analysis_4c import (
    compute_confluence_score_bull,
    compute_confluence_score_bull_accel,
    score_bucketed_returns,
    plot_score_monotonicity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PAIR_SUFFIXES = ["", "_15m", "_1h", "_1d"]
_HORIZONS = ["ret_fwd_3", "ret_fwd_6", "ret_fwd_12"]


def _hist_roc_cols() -> list[str]:
    cols = []
    for suf in _PAIR_SUFFIXES:
        cols.append(f"A_hist{suf}")
        cols.append(f"B_roc{suf}")
    return cols


def _state_cols() -> list[str]:
    cols = []
    for suf in _PAIR_SUFFIXES:
        cols.append(f"A_state{suf}")
        cols.append(f"B_state{suf}")
    return cols


# ---------------------------------------------------------------------------
# T1: compute_confluence_score_bull -- known values
# ---------------------------------------------------------------------------

def test_compute_confluence_score_bull_known_values():
    cols = _hist_roc_cols()
    assert len(cols) == 8

    # Row 0: all positive -> score 8
    # Row 1: all negative -> score 0
    # Row 2: mixed -- exactly 4 positive, 4 negative -> score 4
    row0 = {c: 1.0 for c in cols}
    row1 = {c: -1.0 for c in cols}
    row2 = {c: (1.0 if i % 2 == 0 else -1.0) for i, c in enumerate(cols)}

    df = pd.DataFrame([row0, row1, row2])
    score = compute_confluence_score_bull(df)

    assert score.dtype == "Int64"
    assert score.iloc[0] == 8
    assert score.iloc[1] == 0
    assert score.iloc[2] == 4


# ---------------------------------------------------------------------------
# T2: NaN handling for _1d columns -> caps score at 6
# ---------------------------------------------------------------------------

def test_compute_confluence_score_bull_nan_1d():
    cols = _hist_roc_cols()
    row = {c: 1.0 for c in cols}
    row["A_hist_1d"] = np.nan
    row["B_roc_1d"] = np.nan

    df = pd.DataFrame([row])
    score = compute_confluence_score_bull(df)

    assert not pd.isna(score.iloc[0]), "Score should never itself be NaN"
    assert score.iloc[0] == 6


# ---------------------------------------------------------------------------
# T3: compute_confluence_score_bull_accel -- known values
# ---------------------------------------------------------------------------

def test_compute_confluence_score_bull_accel_known_values():
    cols = _state_cols()
    assert len(cols) == 8

    # Row 0: all "bull_accel" -> score 8
    # Row 1: all "bear_accel" -> score 0
    # Row 2: mixed -- 3 bull_accel, rest other states / NaN -> score 3
    row0 = {c: "bull_accel" for c in cols}
    row1 = {c: "bear_accel" for c in cols}

    row2 = {c: "bear_decel" for c in cols}
    # Make exactly 3 columns "bull_accel"
    for c in cols[:3]:
        row2[c] = "bull_accel"
    # And one column NaN (not bull_accel -> contributes 0)
    row2[cols[3]] = np.nan

    df = pd.DataFrame([row0, row1, row2])
    score = compute_confluence_score_bull_accel(df)

    assert score.dtype == "Int64"
    assert score.iloc[0] == 8
    assert score.iloc[1] == 0
    assert score.iloc[2] == 3


# ---------------------------------------------------------------------------
# T4: compute_confluence_score_bull_accel -- all NaN states -> score 0
# ---------------------------------------------------------------------------

def test_compute_confluence_score_bull_accel_all_nan():
    cols = _state_cols()
    row = {c: pd.NA for c in cols}
    df = pd.DataFrame([row])
    score = compute_confluence_score_bull_accel(df)

    assert not pd.isna(score.iloc[0])
    assert score.iloc[0] == 0


# ---------------------------------------------------------------------------
# T5: score_bucketed_returns -- group counts and threshold behavior
# ---------------------------------------------------------------------------

def make_score_df(seed: int = 42) -> pd.DataFrame:
    """Build a df with a 'score' column taking values {2, 5, 7} with varying
    row counts: 3 rows for score=2 (<5), 10 rows for score=5 (5-19),
    25 rows for score=7 (>=20)."""
    np.random.seed(seed)

    n_2, n_5, n_7 = 3, 10, 25
    scores = (
        [2] * n_2
        + [5] * n_5
        + [7] * n_7
    )
    n_total = len(scores)

    df = pd.DataFrame({
        "score": pd.array(scores, dtype="Int64"),
        "ret_fwd_12": np.random.normal(0.0005, 0.01, n_total),
    })
    return df


def test_score_bucketed_returns_only_present_scores():
    df = make_score_df()
    result = score_bucketed_returns(df, "score", ["ret_fwd_12"])

    scores_in_index = set(result.index.get_level_values("score").unique())
    assert scores_in_index == {2, 5, 7}, f"Unexpected scores in index: {scores_in_index}"


def test_score_bucketed_returns_low_count_nan():
    df = make_score_df()
    result = score_bucketed_returns(df, "score", ["ret_fwd_12"])

    row = result.loc[(2, "ret_fwd_12")]
    assert row["n"] == 3
    assert pd.isna(row["mean_ret"]), "mean_ret should be NaN for n < 5"
    assert pd.isna(row["hit_rate"]), "hit_rate should be NaN for n < 5"
    assert pd.isna(row["t_stat"])
    assert pd.isna(row["p_val_raw"])


def test_score_bucketed_returns_mid_count_mean_but_no_tstat():
    df = make_score_df()
    result = score_bucketed_returns(df, "score", ["ret_fwd_12"])

    row = result.loc[(5, "ret_fwd_12")]
    assert row["n"] == 10
    assert not pd.isna(row["mean_ret"]), "mean_ret should be populated for 5 <= n < 20"
    assert not pd.isna(row["hit_rate"])
    assert pd.isna(row["t_stat"]), "t_stat should be NaN for n < 20"
    assert pd.isna(row["p_val_raw"])


def test_score_bucketed_returns_high_count_all_populated():
    df = make_score_df()
    result = score_bucketed_returns(df, "score", ["ret_fwd_12"])

    row = result.loc[(7, "ret_fwd_12")]
    assert row["n"] == 25
    assert not pd.isna(row["mean_ret"])
    assert not pd.isna(row["hit_rate"])
    assert not pd.isna(row["t_stat"])
    assert not pd.isna(row["p_val_raw"])


def test_score_bucketed_returns_hit_rate_range():
    df = make_score_df()
    result = score_bucketed_returns(df, "score", ["ret_fwd_12"])
    valid = result["hit_rate"].dropna()
    assert (valid >= 0).all() and (valid <= 1).all(), (
        f"hit_rate values out of [0,1]: {valid[~((valid >= 0) & (valid <= 1))]}"
    )


# ---------------------------------------------------------------------------
# T6: score_bucketed_returns -- multiple horizons and ordering
# ---------------------------------------------------------------------------

def test_score_bucketed_returns_multiple_horizons_order():
    np.random.seed(1)
    n = 30
    df = pd.DataFrame({
        "score": pd.array([4] * n, dtype="Int64"),
        "ret_fwd_3": np.random.normal(0, 0.01, n),
        "ret_fwd_6": np.random.normal(0, 0.01, n),
        "ret_fwd_12": np.random.normal(0, 0.01, n),
    })
    horizon_cols = ["ret_fwd_3", "ret_fwd_6", "ret_fwd_12"]
    result = score_bucketed_returns(df, "score", horizon_cols)

    horizons_for_score4 = list(result.loc[4].index)
    assert horizons_for_score4 == horizon_cols

    expected_cols = {"n", "mean_ret", "hit_rate", "t_stat", "p_val_raw"}
    assert set(result.columns) == expected_cols


# ---------------------------------------------------------------------------
# T7: score_bucketed_returns -- empty input handling
# ---------------------------------------------------------------------------

def test_score_bucketed_returns_empty_df():
    df = pd.DataFrame(columns=["score", "ret_fwd_12"])
    result = score_bucketed_returns(df, "score", ["ret_fwd_12"])

    assert result.empty
    assert set(result.columns) == {"n", "mean_ret", "hit_rate", "t_stat", "p_val_raw"}
    assert list(result.index.names) == ["score", "horizon"]


def test_score_bucketed_returns_all_nan_score_col():
    n = 10
    df = pd.DataFrame({
        "score": pd.array([pd.NA] * n, dtype="Int64"),
        "ret_fwd_12": np.random.normal(0, 0.01, n),
    })
    result = score_bucketed_returns(df, "score", ["ret_fwd_12"])

    assert result.empty
    assert set(result.columns) == {"n", "mean_ret", "hit_rate", "t_stat", "p_val_raw"}
    assert list(result.index.names) == ["score", "horizon"]


# ---------------------------------------------------------------------------
# T8: plot_score_monotonicity -- creates file
# ---------------------------------------------------------------------------

def test_plot_score_monotonicity_creates_file(tmp_path: Path):
    df = make_score_df()
    score_table = score_bucketed_returns(df, "score", ["ret_fwd_12"])

    out_path = tmp_path / "subdir" / "confluence_bull.png"
    plot_score_monotonicity(score_table, ["ret_fwd_12"], "Confluence monotonicity (bull)", out_path)

    assert out_path.exists(), f"Expected plot file not found: {out_path}"


# ---------------------------------------------------------------------------
# T9: plot_score_monotonicity -- empty score_table -> no file, no error
# ---------------------------------------------------------------------------

def test_plot_score_monotonicity_empty_skips(tmp_path: Path):
    empty_index = pd.MultiIndex.from_arrays([[], []], names=["score", "horizon"])
    empty_table = pd.DataFrame(
        columns=["n", "mean_ret", "hit_rate", "t_stat", "p_val_raw"], index=empty_index
    )

    out_path = tmp_path / "empty.png"
    plot_score_monotonicity(empty_table, ["ret_fwd_12"], "Empty", out_path)

    assert not out_path.exists(), "No file should be created for empty score_table"
