from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.phase4_inter.analysis_4b import (
    cross_tf_followthrough,
    find_zero_cross_events,
    transition_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tf_df(zero_cross_values, freq="5min", col="A_hist_zero_cross"):
    """Build a minimal TF-native DataFrame with a bar_close column and a
    zero-cross column."""
    n = len(zero_cross_values)
    bar_close = pd.date_range("2024-01-01 09:15", periods=n, freq=freq, tz="Asia/Kolkata")
    df = pd.DataFrame({
        "bar_close": bar_close,
        col: np.array(zero_cross_values, dtype=int),
    })
    return df


# ---------------------------------------------------------------------------
# Test 1: find_zero_cross_events — known pattern
# ---------------------------------------------------------------------------

def test_find_zero_cross_events_known_pattern():
    values = [0, 0, 1, 0, -1, 0, 0, 1]
    df = make_tf_df(values)

    result = find_zero_cross_events(df, "A_hist_zero_cross")

    assert list(result.columns) == ["bar_close", "direction"]
    assert len(result) == 3

    expected_idx = [2, 4, 7]
    expected_dirs = [1, -1, 1]
    for i, (idx, direction) in enumerate(zip(expected_idx, expected_dirs)):
        assert result.iloc[i]["bar_close"] == df.iloc[idx]["bar_close"]
        assert result.iloc[i]["direction"] == direction

    # Ascending order, clean RangeIndex
    assert list(result.index) == [0, 1, 2]
    assert result["direction"].dtype == int


# ---------------------------------------------------------------------------
# Test 2: find_zero_cross_events — empty (all zeros)
# ---------------------------------------------------------------------------

def test_find_zero_cross_events_all_zeros_returns_empty():
    values = [0, 0, 0, 0, 0]
    df = make_tf_df(values)

    result = find_zero_cross_events(df, "A_hist_zero_cross")

    assert list(result.columns) == ["bar_close", "direction"]
    assert len(result) == 0
    # bar_close dtype should remain datetime-like
    assert pd.api.types.is_datetime64_any_dtype(result["bar_close"])
    assert result["direction"].dtype == int


# ---------------------------------------------------------------------------
# Test 3: cross_tf_followthrough — basic positive case (lag=2)
# ---------------------------------------------------------------------------

def test_followthrough_basic_positive_case():
    # lower_df: single bullish flip (+1) at index 2
    lower_values = [0, 0, 1, 0, 0]
    lower_df = make_tf_df(lower_values, freq="5min", col="A_hist_zero_cross")
    t_low = lower_df.iloc[2]["bar_close"]

    # higher_df: spaced bars; the 2nd bar strictly after T_low has +1
    higher_close = pd.date_range(t_low - pd.Timedelta(minutes=15), periods=6, freq="15min", tz="Asia/Kolkata")
    higher_zc = [0, 0, 0, 1, 0, 0]
    # Confirm: bars strictly > t_low are at positions where higher_close > t_low
    higher_df = pd.DataFrame({
        "bar_close": higher_close,
        "B_zero_cross": higher_zc,
    })

    # Sanity: figure out which higher bars are strictly future relative to t_low
    future_mask = higher_df["bar_close"] > t_low
    future_zc = higher_df.loc[future_mask, "B_zero_cross"].tolist()
    # The 2nd future bar should be the one with value 1
    assert future_zc[1] == 1

    result = cross_tf_followthrough(
        lower_df, higher_df,
        lower_zc_col="A_hist_zero_cross",
        higher_zc_col="B_zero_cross",
        k_bars=4,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["bar_close"] == t_low
    assert row["direction"] == 1
    assert row["followed_through"] == True
    assert row["lag_bars"] == 2


# ---------------------------------------------------------------------------
# Test 4: no-look-ahead boundary — same-instant higher bar must not count
# ---------------------------------------------------------------------------

def test_followthrough_no_lookahead_at_same_instant():
    # lower_df: bullish flip (+1) at index 2
    lower_values = [0, 0, 1, 0, 0]
    lower_df = make_tf_df(lower_values, freq="5min", col="A_hist_zero_cross")
    t_low = lower_df.iloc[2]["bar_close"]

    # higher_df: one bar at exactly T_low with matching direction (+1),
    # and all subsequent bars are 0 (no real follow-through).
    higher_close = pd.date_range(t_low, periods=4, freq="15min", tz="Asia/Kolkata")
    higher_zc = [1, 0, 0, 0]  # first bar == t_low has direction match, but must be excluded
    higher_df = pd.DataFrame({
        "bar_close": higher_close,
        "B_zero_cross": higher_zc,
    })

    result = cross_tf_followthrough(
        lower_df, higher_df,
        lower_zc_col="A_hist_zero_cross",
        higher_zc_col="B_zero_cross",
        k_bars=4,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["followed_through"] == False
    assert np.isnan(row["lag_bars"])


# ---------------------------------------------------------------------------
# Test 5: k_bars truncation — only 1 future bar exists, k_bars=4
# ---------------------------------------------------------------------------

def test_followthrough_k_bars_truncation_no_error():
    # lower_df: bullish flip (+1) at the LAST index
    lower_values = [0, 0, 0, 0, 1]
    lower_df = make_tf_df(lower_values, freq="5min", col="A_hist_zero_cross")
    t_low = lower_df.iloc[4]["bar_close"]

    # higher_df: only one bar strictly after t_low, with matching direction
    higher_close = pd.DatetimeIndex([
        t_low - pd.Timedelta(minutes=15),
        t_low + pd.Timedelta(minutes=10),  # only future bar
    ]).tz_localize(None).tz_localize("Asia/Kolkata")
    higher_zc = [0, 1]
    higher_df = pd.DataFrame({
        "bar_close": higher_close,
        "B_zero_cross": higher_zc,
    })

    result = cross_tf_followthrough(
        lower_df, higher_df,
        lower_zc_col="A_hist_zero_cross",
        higher_zc_col="B_zero_cross",
        k_bars=4,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["followed_through"] == True
    assert row["lag_bars"] == 1


# ---------------------------------------------------------------------------
# Test 6: negative case — no match in window
# ---------------------------------------------------------------------------

def test_followthrough_negative_case_no_match():
    lower_values = [0, 0, 1, 0, 0]
    lower_df = make_tf_df(lower_values, freq="5min", col="A_hist_zero_cross")
    t_low = lower_df.iloc[2]["bar_close"]

    higher_close = pd.date_range(t_low + pd.Timedelta(minutes=5), periods=4, freq="15min", tz="Asia/Kolkata")
    higher_zc = [0, 0, -1, 0]  # direction +1 never appears
    higher_df = pd.DataFrame({
        "bar_close": higher_close,
        "B_zero_cross": higher_zc,
    })

    result = cross_tf_followthrough(
        lower_df, higher_df,
        lower_zc_col="A_hist_zero_cross",
        higher_zc_col="B_zero_cross",
        k_bars=4,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["followed_through"] == False
    assert np.isnan(row["lag_bars"])


# ---------------------------------------------------------------------------
# Test 7: mixed directions — +1 and -1 flips evaluated independently
# ---------------------------------------------------------------------------

def test_followthrough_mixed_directions():
    # lower_df: +1 flip at index 1, -1 flip at index 4
    lower_values = [0, 1, 0, 0, -1, 0]
    lower_df = make_tf_df(lower_values, freq="5min", col="A_hist_zero_cross")
    t_low_pos = lower_df.iloc[1]["bar_close"]
    t_low_neg = lower_df.iloc[4]["bar_close"]

    # higher_df spans the whole period with regularly spaced 15min bars
    higher_close = pd.date_range("2024-01-01 09:15", periods=10, freq="15min", tz="Asia/Kolkata")
    higher_zc = np.zeros(10, dtype=int)
    higher_df = pd.DataFrame({
        "bar_close": higher_close,
        "B_zero_cross": higher_zc,
    })

    # Set up a +1 match in the window after t_low_pos
    future_pos_mask = higher_df["bar_close"] > t_low_pos
    future_pos_idx = higher_df.index[future_pos_mask].tolist()
    higher_df.loc[future_pos_idx[0], "B_zero_cross"] = 1  # lag=1 match for +1 flip

    # Set up a -1 match in the window after t_low_neg
    future_neg_mask = higher_df["bar_close"] > t_low_neg
    future_neg_idx = higher_df.index[future_neg_mask].tolist()
    assert len(future_neg_idx) >= 2
    higher_df.loc[future_neg_idx[1], "B_zero_cross"] = -1  # lag=2 match for -1 flip

    result = cross_tf_followthrough(
        lower_df, higher_df,
        lower_zc_col="A_hist_zero_cross",
        higher_zc_col="B_zero_cross",
        k_bars=5,
    )

    assert len(result) == 2

    pos_row = result[result["direction"] == 1].iloc[0]
    neg_row = result[result["direction"] == -1].iloc[0]

    assert pos_row["followed_through"] == True
    assert pos_row["lag_bars"] == 1

    assert neg_row["followed_through"] == True
    assert neg_row["lag_bars"] == 2


# ---------------------------------------------------------------------------
# Test 8: transition_summary — mixed followed/not-followed
# ---------------------------------------------------------------------------

def test_transition_summary_mixed():
    events_df = pd.DataFrame({
        "bar_close": pd.date_range("2024-01-01", periods=6, freq="D", tz="Asia/Kolkata"),
        "direction": [1, 1, 1, -1, -1, -1],
        "followed_through": [True, True, False, True, False, False],
        "lag_bars": [1.0, 3.0, np.nan, 2.0, np.nan, np.nan],
    })

    result = transition_summary(events_df)

    assert result.index.name == "direction"
    assert set(result.index) == {1, -1}

    pos_row = result.loc[1]
    assert pos_row["n_events"] == 3
    assert pytest.approx(pos_row["p_followthrough"], rel=1e-3) == 2 / 3
    assert pos_row["median_lag_bars"] == 2.0  # median of [1.0, 3.0]

    neg_row = result.loc[-1]
    assert neg_row["n_events"] == 3
    assert pytest.approx(neg_row["p_followthrough"], rel=1e-3) == 1 / 3
    assert neg_row["median_lag_bars"] == 2.0  # median of [2.0]


# ---------------------------------------------------------------------------
# Test 9: transition_summary — empty events_df
# ---------------------------------------------------------------------------

def test_transition_summary_empty():
    events_df = pd.DataFrame(columns=["bar_close", "direction", "followed_through", "lag_bars"])

    result = transition_summary(events_df)

    assert list(result.columns) == ["n_events", "p_followthrough", "median_lag_bars"]
    assert len(result) == 0
    assert result.index.name == "direction"


# ---------------------------------------------------------------------------
# Test 10: cross_tf_followthrough — empty lower_df (no flips)
# ---------------------------------------------------------------------------

def test_followthrough_no_flips_returns_empty():
    lower_values = [0, 0, 0, 0, 0]
    lower_df = make_tf_df(lower_values, freq="5min", col="A_hist_zero_cross")

    higher_close = pd.date_range("2024-01-01 09:15", periods=5, freq="15min", tz="Asia/Kolkata")
    higher_df = pd.DataFrame({
        "bar_close": higher_close,
        "B_zero_cross": np.zeros(5, dtype=int),
    })

    result = cross_tf_followthrough(
        lower_df, higher_df,
        lower_zc_col="A_hist_zero_cross",
        higher_zc_col="B_zero_cross",
        k_bars=4,
    )

    assert list(result.columns) == ["bar_close", "direction", "followed_through", "lag_bars"]
    assert len(result) == 0
