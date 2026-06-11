from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.phase4_inter.analysis_4c_compression import (
    detect_compression_expansion,
    event_study_h13,
)


# ---------------------------------------------------------------------------
# Test 1: detect_compression_expansion finds the expected event
# ---------------------------------------------------------------------------

def test_detect_compression_expansion_finds_event():
    """A compression bar followed shortly by a |zroc| > 1.5 bar -> one event."""
    np.random.seed(42)
    n = 30

    # Most bars: large |gap_norm| (not compression), small |zroc| (no expansion)
    gap_norm = np.full(n, 1.0)
    zroc = np.full(n, 0.1)

    # Make the lowest-decile threshold meaningful: with quantile=0.10 over 30
    # rows, roughly the 3 smallest |gap_norm| values define the threshold.
    # Set bars 10, 19, 20 to near-zero gap_norm (compression candidates),
    # but only test bar 10's expansion explicitly. Bar 10's lookahead window
    # (11..20) does not overlap bar 19/20's expansion search target (12), so
    # the event at position 12 can only be discovered via bar 10.
    compression_positions = [10, 19, 20]
    for p in compression_positions:
        gap_norm[p] = 0.0001

    # Expansion onset 2 bars after compression bar 10 -> position 12, positive zroc
    zroc[12] = 2.0  # |zroc| > 1.5, sign = +1

    bar_close = pd.date_range("2024-01-01", periods=n, freq="15min")

    df_tf = pd.DataFrame({
        "bar_close": bar_close,
        "B_gap_norm": gap_norm,
        "B_zroc": zroc,
    })

    result = detect_compression_expansion(
        df_tf,
        compression_quantile=0.10,
        max_lookahead_bars=10,
        expansion_threshold=1.5,
    )

    assert list(result.columns) == ["bar_close", "expansion_direction", "lag_bars"]
    assert len(result) >= 1, f"Expected at least one event, got:\n{result}"

    # The event triggered by compression bar 10 -> expansion at position 12.
    expected_bar_close = bar_close[12]
    matching = result[result["bar_close"] == expected_bar_close]
    assert len(matching) == 1, f"Expected exactly one event at {expected_bar_close}, got:\n{result}"

    row = matching.iloc[0]
    assert row["expansion_direction"] == 1, f"Expected direction +1, got {row['expansion_direction']}"
    # lag_bars = j - i = 12 - 10 = 2 (from compression bar 10)
    assert row["lag_bars"] == 2, f"Expected lag_bars=2, got {row['lag_bars']}"


# ---------------------------------------------------------------------------
# Test 2: dedup behavior — two adjacent compression bars map to same expansion
# ---------------------------------------------------------------------------

def test_detect_compression_expansion_dedup():
    """Compression at i and i+1 both finding expansion at j=i+2 -> one event."""
    np.random.seed(42)
    n = 30

    gap_norm = np.full(n, 1.0)
    zroc = np.full(n, 0.1)

    # Two adjacent compression bars at positions 10 and 11.
    gap_norm[10] = 0.0001
    gap_norm[11] = 0.0001
    # A few more low-gap bars elsewhere so the quantile threshold includes
    # positions 10 and 11 (with quantile=0.10 over n=30, ~3 lowest values
    # define the threshold).
    gap_norm[20] = 0.0001

    # Expansion onset at position 12 (= i+2 for i=10, = i+1 for i=11)
    zroc[12] = -2.0  # |zroc| > 1.5, sign = -1

    bar_close = pd.date_range("2024-01-01", periods=n, freq="15min")

    df_tf = pd.DataFrame({
        "bar_close": bar_close,
        "B_gap_norm": gap_norm,
        "B_zroc": zroc,
    })

    result = detect_compression_expansion(
        df_tf,
        compression_quantile=0.10,
        max_lookahead_bars=10,
        expansion_threshold=1.5,
    )

    expected_bar_close = bar_close[12]
    matching = result[result["bar_close"] == expected_bar_close]
    assert len(matching) == 1, (
        f"Expected exactly ONE deduped event at {expected_bar_close}, got:\n{result}"
    )

    row = matching.iloc[0]
    assert row["expansion_direction"] == -1
    # The FIRST (smallest lag) mapping wins: i=10 -> j=12 -> lag=2.
    assert row["lag_bars"] == 2, f"Expected lag_bars=2 (first/earliest mapping), got {row['lag_bars']}"


# ---------------------------------------------------------------------------
# Test 3: max_lookahead_bars boundary
# ---------------------------------------------------------------------------

def test_detect_compression_expansion_lookahead_boundary():
    """Expansion at exactly max_lookahead+1 -> not detected; at exactly max_lookahead -> detected."""
    np.random.seed(42)
    n = 40
    max_lookahead = 5

    gap_norm = np.full(n, 1.0)
    zroc = np.full(n, 0.1)

    # Compression bar A at position 5: expansion at i + max_lookahead + 1 = 11 -> OUT of window
    gap_norm[5] = 0.0001
    zroc[11] = 2.0  # would be lag=6, but max_lookahead=5 -> not detected for this compression bar

    # Compression bar B at position 20: expansion at i + max_lookahead = 25 -> IN window
    gap_norm[20] = 0.0001
    zroc[25] = 2.0  # lag=5 == max_lookahead -> detected

    # Add a few more low-gap bars so the quantile threshold is meaningful
    # and includes positions 5 and 20 (quantile=0.10 over n=40 -> ~4 lowest).
    gap_norm[30] = 0.0001
    gap_norm[31] = 0.0001

    bar_close = pd.date_range("2024-01-01", periods=n, freq="15min")

    df_tf = pd.DataFrame({
        "bar_close": bar_close,
        "B_gap_norm": gap_norm,
        "B_zroc": zroc,
    })

    result = detect_compression_expansion(
        df_tf,
        compression_quantile=0.10,
        max_lookahead_bars=max_lookahead,
        expansion_threshold=1.5,
    )

    # Expansion at position 11 (lag=6 from compression bar 5) should NOT be present.
    assert bar_close[11] not in result["bar_close"].values, (
        f"Expansion at lag={11-5}=6 > max_lookahead={max_lookahead} should NOT be detected:\n{result}"
    )

    # Expansion at position 25 (lag=5 from compression bar 20) SHOULD be present.
    matching = result[result["bar_close"] == bar_close[25]]
    assert len(matching) == 1, (
        f"Expansion at lag={25-20}=5 == max_lookahead={max_lookahead} should be detected:\n{result}"
    )
    assert matching.iloc[0]["lag_bars"] == max_lookahead


# ---------------------------------------------------------------------------
# Test 4: no events found -> empty DataFrame with correct columns/dtypes
# ---------------------------------------------------------------------------

def test_detect_compression_expansion_no_events():
    """No bar has |zroc| > threshold within range of any compression bar -> empty result."""
    np.random.seed(42)
    n = 30

    gap_norm = np.full(n, 1.0)
    # Make a few compression bars
    gap_norm[10] = 0.0001
    gap_norm[20] = 0.0001

    # zroc never exceeds the expansion threshold anywhere
    zroc = np.full(n, 0.1)

    bar_close = pd.date_range("2024-01-01", periods=n, freq="15min")

    df_tf = pd.DataFrame({
        "bar_close": bar_close,
        "B_gap_norm": gap_norm,
        "B_zroc": zroc,
    })

    result = detect_compression_expansion(
        df_tf,
        compression_quantile=0.10,
        max_lookahead_bars=10,
        expansion_threshold=1.5,
    )

    assert list(result.columns) == ["bar_close", "expansion_direction", "lag_bars"]
    assert len(result) == 0
    assert result["expansion_direction"].dtype == np.dtype("int64")
    assert result["lag_bars"].dtype == np.dtype("int64")


# ---------------------------------------------------------------------------
# Test 5: event_study_h13 basic correctness
# ---------------------------------------------------------------------------

def test_event_study_h13_basic():
    """Build a small master_df with bar_close_15m repeating in groups of 3."""
    np.random.seed(42)

    # 2 HTF groups, each spanning 3 5m rows = 6 rows total.
    bar_close_5m = pd.date_range("2024-01-01 09:15", periods=6, freq="5min")
    htf_close = pd.to_datetime([
        "2024-01-01 09:15", "2024-01-01 09:15", "2024-01-01 09:15",
        "2024-01-01 09:30", "2024-01-01 09:30", "2024-01-01 09:30",
    ])

    ret_fwd_3 = np.array([0.01, 0.02, 0.03, -0.02, -0.03, -0.04])
    ret_fwd_12 = np.array([0.05, 0.06, 0.07, -0.05, -0.06, -0.07])

    master_df = pd.DataFrame({
        "bar_close": bar_close_5m,
        "bar_close_15m": htf_close,
        "ret_fwd_3": ret_fwd_3,
        "ret_fwd_12": ret_fwd_12,
    })

    # Two events: one matching each HTF group.
    event_df = pd.DataFrame({
        "bar_close": [pd.Timestamp("2024-01-01 09:15"), pd.Timestamp("2024-01-01 09:30")],
        "expansion_direction": [1, -1],
        "lag_bars": [2, 3],
    })

    result = event_study_h13(master_df, event_df, "bar_close_15m", ["ret_fwd_3", "ret_fwd_12"])

    assert result["n_events"] == 2

    # First event matches the FIRST row of group 09:15 -> ret_fwd_3=0.01, ret_fwd_12=0.05, direction=+1
    # Second event matches the FIRST row of group 09:30 -> ret_fwd_3=-0.02, ret_fwd_12=-0.05, direction=-1
    expected_signed_ret_3 = np.mean([1 * 0.01, -1 * (-0.02)])  # = mean(0.01, 0.02) = 0.015
    expected_signed_ret_12 = np.mean([1 * 0.05, -1 * (-0.05)])  # = mean(0.05, 0.05) = 0.05

    assert result["signed_ret"]["ret_fwd_3"] == pytest.approx(expected_signed_ret_3)
    assert result["signed_ret"]["ret_fwd_12"] == pytest.approx(expected_signed_ret_12)

    expected_abs_ret_event_3 = np.mean([abs(0.01), abs(-0.02)])
    expected_abs_ret_event_12 = np.mean([abs(0.05), abs(-0.05)])

    assert result["abs_ret_event"]["ret_fwd_3"] == pytest.approx(expected_abs_ret_event_3)
    assert result["abs_ret_event"]["ret_fwd_12"] == pytest.approx(expected_abs_ret_event_12)

    expected_abs_ret_baseline_3 = np.mean(np.abs(ret_fwd_3))
    expected_abs_ret_baseline_12 = np.mean(np.abs(ret_fwd_12))

    assert result["abs_ret_baseline"]["ret_fwd_3"] == pytest.approx(expected_abs_ret_baseline_3)
    assert result["abs_ret_baseline"]["ret_fwd_12"] == pytest.approx(expected_abs_ret_baseline_12)


# ---------------------------------------------------------------------------
# Test 6: event_study_h13 picks the FIRST row of a matching HTF group
# ---------------------------------------------------------------------------

def test_event_study_h13_picks_first_row_of_group():
    """First vs last row of a group differ -> the FIRST row's value is used."""
    np.random.seed(42)

    bar_close_5m = pd.date_range("2024-01-01 09:15", periods=3, freq="5min")
    htf_close = pd.to_datetime(["2024-01-01 09:15"] * 3)

    # First row has ret_fwd_3 = 100.0 (distinctive); last row has -999.0 (decoy)
    ret_fwd_3 = np.array([100.0, 0.0, -999.0])

    master_df = pd.DataFrame({
        "bar_close": bar_close_5m,
        "bar_close_15m": htf_close,
        "ret_fwd_3": ret_fwd_3,
    })

    event_df = pd.DataFrame({
        "bar_close": [pd.Timestamp("2024-01-01 09:15")],
        "expansion_direction": [1],
        "lag_bars": [1],
    })

    result = event_study_h13(master_df, event_df, "bar_close_15m", ["ret_fwd_3"])

    assert result["n_events"] == 1
    # signed_ret = direction * matched_row value = 1 * 100.0 = 100.0 (NOT -999.0)
    assert result["signed_ret"]["ret_fwd_3"] == pytest.approx(100.0)
    assert result["abs_ret_event"]["ret_fwd_3"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Test 7: event_df with bar_close not present in master_df -> n_events == 0
# ---------------------------------------------------------------------------

def test_event_study_h13_no_match():
    """event_df bar_close values not in master_df[htf_bar_close_col] -> n_events=0."""
    np.random.seed(42)

    bar_close_5m = pd.date_range("2024-01-01 09:15", periods=3, freq="5min")
    htf_close = pd.to_datetime(["2024-01-01 09:15"] * 3)

    ret_fwd_3 = np.array([0.01, -0.02, 0.03])

    master_df = pd.DataFrame({
        "bar_close": bar_close_5m,
        "bar_close_15m": htf_close,
        "ret_fwd_3": ret_fwd_3,
    })

    # event bar_close does not match anything in master_df
    event_df = pd.DataFrame({
        "bar_close": [pd.Timestamp("2024-01-02 09:15")],
        "expansion_direction": [1],
        "lag_bars": [2],
    })

    result = event_study_h13(master_df, event_df, "bar_close_15m", ["ret_fwd_3"])

    assert result["n_events"] == 0
    assert pd.isna(result["signed_ret"]["ret_fwd_3"])
    assert pd.isna(result["abs_ret_event"]["ret_fwd_3"])

    # baseline still computed from master_df
    expected_baseline = np.mean(np.abs(ret_fwd_3))
    assert result["abs_ret_baseline"]["ret_fwd_3"] == pytest.approx(expected_baseline)


# ---------------------------------------------------------------------------
# Test 8: empty event_df -> same behavior as no-match case
# ---------------------------------------------------------------------------

def test_event_study_h13_empty_event_df():
    """Empty event_df -> n_events=0, signed/abs_ret_event NaN, baseline computed."""
    np.random.seed(42)

    bar_close_5m = pd.date_range("2024-01-01 09:15", periods=3, freq="5min")
    htf_close = pd.to_datetime(["2024-01-01 09:15"] * 3)
    ret_fwd_3 = np.array([0.01, -0.02, 0.03])

    master_df = pd.DataFrame({
        "bar_close": bar_close_5m,
        "bar_close_15m": htf_close,
        "ret_fwd_3": ret_fwd_3,
    })

    event_df = pd.DataFrame(columns=["bar_close", "expansion_direction", "lag_bars"])

    result = event_study_h13(master_df, event_df, "bar_close_15m", ["ret_fwd_3"])

    assert result["n_events"] == 0
    assert pd.isna(result["signed_ret"]["ret_fwd_3"])
    assert pd.isna(result["abs_ret_event"]["ret_fwd_3"])

    expected_baseline = np.mean(np.abs(ret_fwd_3))
    assert result["abs_ret_baseline"]["ret_fwd_3"] == pytest.approx(expected_baseline)


# ---------------------------------------------------------------------------
# Test 9: NaN handling in horizon columns
# ---------------------------------------------------------------------------

def test_event_study_h13_nan_handling():
    """NaN values in matched_row[h] / master_df[h] are ignored in nan-aware means."""
    np.random.seed(42)

    bar_close_5m = pd.date_range("2024-01-01 09:15", periods=6, freq="5min")
    htf_close = pd.to_datetime([
        "2024-01-01 09:15", "2024-01-01 09:15", "2024-01-01 09:15",
        "2024-01-01 09:30", "2024-01-01 09:30", "2024-01-01 09:30",
    ])

    # ret_fwd_3 has a NaN in the baseline (last row) and the matched event for
    # the second group has a NaN value too.
    ret_fwd_3 = np.array([0.01, 0.02, 0.03, np.nan, -0.05, -0.06])

    master_df = pd.DataFrame({
        "bar_close": bar_close_5m,
        "bar_close_15m": htf_close,
        "ret_fwd_3": ret_fwd_3,
    })

    event_df = pd.DataFrame({
        "bar_close": [pd.Timestamp("2024-01-01 09:15"), pd.Timestamp("2024-01-01 09:30")],
        "expansion_direction": [1, -1],
        "lag_bars": [1, 1],
    })

    result = event_study_h13(master_df, event_df, "bar_close_15m", ["ret_fwd_3"])

    assert result["n_events"] == 2

    # Matched rows: first row of each group -> values 0.01 (dir=+1) and NaN (dir=-1)
    # signed_ret = nanmean([1*0.01, -1*NaN]) = nanmean([0.01, NaN]) = 0.01
    assert result["signed_ret"]["ret_fwd_3"] == pytest.approx(0.01)
    # abs_ret_event = nanmean([|0.01|, NaN]) = 0.01
    assert result["abs_ret_event"]["ret_fwd_3"] == pytest.approx(0.01)

    # baseline = nanmean(|ret_fwd_3|) over all 6 rows, ignoring the one NaN
    expected_baseline = np.nanmean(np.abs(ret_fwd_3))
    assert result["abs_ret_baseline"]["ret_fwd_3"] == pytest.approx(expected_baseline)
