from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.phase4_inter.analysis_4d import (
    compute_confluence_score_bear,
    score_bucketed_returns,
    compare_bull_bear_asymmetry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PAIR_SUFFIXES = ["", "_15m", "_1h", "_1d"]


def _hist_roc_cols() -> list[str]:
    cols = []
    for suf in _PAIR_SUFFIXES:
        cols.append(f"A_hist{suf}")
        cols.append(f"B_roc{suf}")
    return cols


# ---------------------------------------------------------------------------
# T1: compute_confluence_score_bear -- known values
# ---------------------------------------------------------------------------

def test_compute_confluence_score_bear_known_values():
    cols = _hist_roc_cols()
    assert len(cols) == 8

    # Row 0: all negative -> score 8
    # Row 1: all positive -> score 0
    # Row 2: mixed, exactly 4 negative + 4 positive -> score 4
    row0 = {c: -1.0 for c in cols}
    row1 = {c: 1.0 for c in cols}
    row2 = {c: (-1.0 if i % 2 == 0 else 1.0) for i, c in enumerate(cols)}

    df = pd.DataFrame([row0, row1, row2])
    score = compute_confluence_score_bear(df)

    assert score.dtype == "Int64"
    assert score.iloc[0] == 8
    assert score.iloc[1] == 0
    assert score.iloc[2] == 4


# ---------------------------------------------------------------------------
# T2: exact zeros don't count as bearish
# ---------------------------------------------------------------------------

def test_compute_confluence_score_bear_zeros_dont_count():
    cols = _hist_roc_cols()
    # All negative except 2 columns set to exactly 0.0
    row = {c: -1.0 for c in cols}
    row[cols[0]] = 0.0
    row[cols[1]] = 0.0

    df = pd.DataFrame([row])
    score = compute_confluence_score_bear(df)

    assert not pd.isna(score.iloc[0])
    assert score.iloc[0] == 6


# ---------------------------------------------------------------------------
# T3: NaN handling for _1d columns -> contributes 0 for those 2 pairs
# ---------------------------------------------------------------------------

def test_compute_confluence_score_bear_nan_1d():
    cols = _hist_roc_cols()
    # All other 6 columns negative, _1d columns are NaN
    row = {c: -1.0 for c in cols}
    row["A_hist_1d"] = np.nan
    row["B_roc_1d"] = np.nan

    df = pd.DataFrame([row])
    score = compute_confluence_score_bear(df)

    assert not pd.isna(score.iloc[0]), "Score should never itself be NaN"
    assert score.iloc[0] == 6


# ---------------------------------------------------------------------------
# T4: score_bucketed_returns -- group counts and threshold behavior
# ---------------------------------------------------------------------------

def make_score_df(seed: int = 7) -> pd.DataFrame:
    """Build a df with a 'score' column taking values {1, 4, 6} with varying
    row counts: 3 rows for score=1 (<5), 12 rows for score=4 (5-19),
    25 rows for score=6 (>=20)."""
    np.random.seed(seed)

    n_1, n_4, n_6 = 3, 12, 25
    scores = (
        [1] * n_1
        + [4] * n_4
        + [6] * n_6
    )
    n_total = len(scores)

    df = pd.DataFrame({
        "score": pd.array(scores, dtype="Int64"),
        "ret_fwd_12": np.random.normal(-0.0005, 0.01, n_total),
    })
    return df


def test_score_bucketed_returns_only_present_scores():
    df = make_score_df()
    result = score_bucketed_returns(df, "score", ["ret_fwd_12"])

    scores_in_index = set(result.index.get_level_values("score").unique())
    assert scores_in_index == {1, 4, 6}, f"Unexpected scores in index: {scores_in_index}"


def test_score_bucketed_returns_low_count_nan():
    df = make_score_df()
    result = score_bucketed_returns(df, "score", ["ret_fwd_12"])

    row = result.loc[(1, "ret_fwd_12")]
    assert row["n"] == 3
    assert pd.isna(row["mean_ret"]), "mean_ret should be NaN for n < 5"
    assert pd.isna(row["hit_rate"]), "hit_rate should be NaN for n < 5"
    assert pd.isna(row["t_stat"])
    assert pd.isna(row["p_val_raw"])


def test_score_bucketed_returns_mid_count_mean_but_no_tstat():
    df = make_score_df()
    result = score_bucketed_returns(df, "score", ["ret_fwd_12"])

    row = result.loc[(4, "ret_fwd_12")]
    assert row["n"] == 12
    assert not pd.isna(row["mean_ret"]), "mean_ret should be populated for 5 <= n < 20"
    assert not pd.isna(row["hit_rate"])
    assert pd.isna(row["t_stat"]), "t_stat should be NaN for n < 20"
    assert pd.isna(row["p_val_raw"])


def test_score_bucketed_returns_high_count_all_populated():
    df = make_score_df()
    result = score_bucketed_returns(df, "score", ["ret_fwd_12"])

    row = result.loc[(6, "ret_fwd_12")]
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
# T5: score_bucketed_returns -- empty input handling
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
# T6: score_bucketed_returns -- multiple horizons and ordering
# ---------------------------------------------------------------------------

def test_score_bucketed_returns_multiple_horizons_order():
    np.random.seed(2)
    n = 30
    df = pd.DataFrame({
        "score": pd.array([5] * n, dtype="Int64"),
        "ret_fwd_3": np.random.normal(0, 0.01, n),
        "ret_fwd_6": np.random.normal(0, 0.01, n),
        "ret_fwd_12": np.random.normal(0, 0.01, n),
    })
    horizon_cols = ["ret_fwd_3", "ret_fwd_6", "ret_fwd_12"]
    result = score_bucketed_returns(df, "score", horizon_cols)

    horizons_for_score5 = list(result.loc[5].index)
    assert horizons_for_score5 == horizon_cols

    expected_cols = {"n", "mean_ret", "hit_rate", "t_stat", "p_val_raw"}
    assert set(result.columns) == expected_cols


# ---------------------------------------------------------------------------
# T7: compare_bull_bear_asymmetry -- intersection only, abs_diff correctness
# ---------------------------------------------------------------------------

def _build_table(score_mean_n: dict[int, tuple[float, int]], horizon: str = "ret_fwd_12") -> pd.DataFrame:
    """Build a small MultiIndex (score, horizon) table with mean_ret/n columns
    (and dummy hit_rate/t_stat/p_val_raw columns to mimic score_bucketed_returns
    output shape)."""
    records = []
    index_tuples = []
    for score_val, (mean_ret, n) in score_mean_n.items():
        records.append({
            "n": n,
            "mean_ret": mean_ret,
            "hit_rate": float("nan"),
            "t_stat": float("nan"),
            "p_val_raw": float("nan"),
        })
        index_tuples.append((score_val, horizon))
    index = pd.MultiIndex.from_tuples(index_tuples, names=["score", "horizon"])
    return pd.DataFrame(records, index=index, columns=["n", "mean_ret", "hit_rate", "t_stat", "p_val_raw"])


def test_compare_bull_bear_asymmetry_intersection_and_abs_diff():
    # bull_table has scores {2, 3, 5}; bear_table has scores {3, 5, 7}
    bull_table = _build_table({
        2: (0.0002, 50),
        3: (0.0010, 60),
        5: (0.0030, 40),
    })
    bear_table = _build_table({
        3: (-0.0005, 55),
        5: (-0.0020, 35),
        7: (-0.0040, 30),
    })

    result = compare_bull_bear_asymmetry(bull_table, bear_table, ["ret_fwd_12"])

    scores_in_result = set(result.index.get_level_values("score").unique())
    assert scores_in_result == {3, 5}, f"Expected only intersection {{3,5}}, got {scores_in_result}"

    # Check score=3: bull mean_ret=0.001, bear mean_ret=-0.0005
    row3 = result.loc[(3, "ret_fwd_12")]
    assert row3["bull_mean_ret"] == pytest.approx(0.0010)
    assert row3["bear_mean_ret"] == pytest.approx(-0.0005)
    assert row3["bull_n"] == 60
    assert row3["bear_n"] == 55
    expected_abs_diff_3 = abs(0.0010) - abs(-0.0005)
    assert row3["abs_diff"] == pytest.approx(expected_abs_diff_3)
    assert row3["abs_diff"] == pytest.approx(0.0005)

    # Check score=5: bull mean_ret=0.003, bear mean_ret=-0.002
    row5 = result.loc[(5, "ret_fwd_12")]
    expected_abs_diff_5 = abs(0.0030) - abs(-0.0020)
    assert row5["abs_diff"] == pytest.approx(expected_abs_diff_5)

    expected_cols = ["bull_mean_ret", "bear_mean_ret", "bull_n", "bear_n", "abs_diff"]
    assert list(result.columns) == expected_cols


# ---------------------------------------------------------------------------
# T8: compare_bull_bear_asymmetry -- NaN mean_ret propagation
# ---------------------------------------------------------------------------

def test_compare_bull_bear_asymmetry_nan_propagation():
    bull_table = _build_table({
        2: (float("nan"), 2),   # below _MIN_COUNT_MEAN -> NaN mean_ret
        3: (0.0010, 60),
    })
    bear_table = _build_table({
        2: (-0.0008, 50),
        3: (-0.0005, 55),
    })

    result = compare_bull_bear_asymmetry(bull_table, bear_table, ["ret_fwd_12"])

    scores_in_result = set(result.index.get_level_values("score").unique())
    assert scores_in_result == {2, 3}

    # score=2: bull mean_ret is NaN -> abs_diff NaN
    row2 = result.loc[(2, "ret_fwd_12")]
    assert pd.isna(row2["bull_mean_ret"])
    assert pd.isna(row2["abs_diff"])

    # score=3: unaffected
    row3 = result.loc[(3, "ret_fwd_12")]
    assert not pd.isna(row3["abs_diff"])
    expected_abs_diff_3 = abs(0.0010) - abs(-0.0005)
    assert row3["abs_diff"] == pytest.approx(expected_abs_diff_3)


# ---------------------------------------------------------------------------
# T9: compare_bull_bear_asymmetry -- no overlap -> empty result
# ---------------------------------------------------------------------------

def test_compare_bull_bear_asymmetry_no_overlap():
    bull_table = _build_table({1: (0.001, 50), 2: (0.002, 60)})
    bear_table = _build_table({7: (-0.001, 50), 8: (-0.002, 60)})

    result = compare_bull_bear_asymmetry(bull_table, bear_table, ["ret_fwd_12"])

    assert result.empty
    assert list(result.columns) == ["bull_mean_ret", "bear_mean_ret", "bull_n", "bear_n", "abs_diff"]
    assert list(result.index.names) == ["score", "horizon"]


# ---------------------------------------------------------------------------
# T10: compare_bull_bear_asymmetry -- multi-horizon case
# ---------------------------------------------------------------------------

def test_compare_bull_bear_asymmetry_multi_horizon():
    horizon_cols = ["ret_fwd_3", "ret_fwd_12"]

    bull_records = []
    bull_index = []
    bear_records = []
    bear_index = []

    # scores {2, 4} overlapping, both horizons present in both tables
    bull_data = {
        (2, "ret_fwd_3"): (0.0005, 30),
        (2, "ret_fwd_12"): (0.0015, 30),
        (4, "ret_fwd_3"): (0.0008, 40),
        (4, "ret_fwd_12"): (0.0025, 40),
    }
    bear_data = {
        (2, "ret_fwd_3"): (-0.0004, 28),
        (2, "ret_fwd_12"): (-0.0010, 28),
        (4, "ret_fwd_3"): (-0.0009, 38),
        (4, "ret_fwd_12"): (-0.0030, 38),
    }

    for (score_val, horizon), (mean_ret, n) in bull_data.items():
        bull_records.append({
            "n": n, "mean_ret": mean_ret, "hit_rate": float("nan"),
            "t_stat": float("nan"), "p_val_raw": float("nan"),
        })
        bull_index.append((score_val, horizon))

    for (score_val, horizon), (mean_ret, n) in bear_data.items():
        bear_records.append({
            "n": n, "mean_ret": mean_ret, "hit_rate": float("nan"),
            "t_stat": float("nan"), "p_val_raw": float("nan"),
        })
        bear_index.append((score_val, horizon))

    bull_table = pd.DataFrame(
        bull_records,
        index=pd.MultiIndex.from_tuples(bull_index, names=["score", "horizon"]),
        columns=["n", "mean_ret", "hit_rate", "t_stat", "p_val_raw"],
    )
    bear_table = pd.DataFrame(
        bear_records,
        index=pd.MultiIndex.from_tuples(bear_index, names=["score", "horizon"]),
        columns=["n", "mean_ret", "hit_rate", "t_stat", "p_val_raw"],
    )

    result = compare_bull_bear_asymmetry(bull_table, bear_table, horizon_cols)

    scores_in_result = sorted(result.index.get_level_values("score").unique())
    assert scores_in_result == [2, 4]

    # For each score, both horizons should be present in horizon_cols order
    for score_val in [2, 4]:
        horizons_for_score = list(result.loc[score_val].index)
        assert horizons_for_score == horizon_cols, (
            f"Expected {horizon_cols} for score {score_val}, got {horizons_for_score}"
        )

    # Spot-check one value
    row = result.loc[(4, "ret_fwd_12")]
    assert row["bull_mean_ret"] == pytest.approx(0.0025)
    assert row["bear_mean_ret"] == pytest.approx(-0.0030)
    expected_abs_diff = abs(0.0025) - abs(-0.0030)
    assert row["abs_diff"] == pytest.approx(expected_abs_diff)
