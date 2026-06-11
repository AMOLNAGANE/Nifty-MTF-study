from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.phase4_inter.analysis_4a import STATE_ORDER, stratify_by_htf_state

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HORIZONS = ["ret_fwd_1", "ret_fwd_3"]


def make_synthetic_df(n_htf_bars: int = 40, bars_per_htf: int = 3, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic 5m DataFrame with repeated HTF state/bar_close values.

    Each "HTF bar" spans `bars_per_htf` consecutive 5m rows, sharing the same
    `htf_state` and `htf_bar_close` value (mimicking the merge_asof repetition
    of higher-TF columns onto 5m rows). `setup_mask` is True for every row
    (so the "unconditional" group == all rows), and `A_state` cycles through
    STATE_ORDER so per-state masks can be tested too.
    """
    np.random.seed(seed)

    n = n_htf_bars * bars_per_htf

    htf_states = []
    htf_bar_close = []
    for i in range(n_htf_bars):
        # Cycle through STATE_ORDER, but make every 5th HTF bar NaN-state
        if i % 5 == 4:
            state = np.nan
        else:
            state = STATE_ORDER[i % len(STATE_ORDER)]
        ts = pd.Timestamp("2020-01-01") + pd.Timedelta(hours=i)
        for _ in range(bars_per_htf):
            htf_states.append(state)
            htf_bar_close.append(ts)

    df = pd.DataFrame({
        "htf_state": pd.Series(htf_states, dtype=object),
        "htf_bar_close": pd.to_datetime(htf_bar_close),
        "ret_fwd_1": np.random.normal(0.0005, 0.01, n),
        "ret_fwd_3": np.random.normal(0.0010, 0.015, n),
    })
    return df


# ---------------------------------------------------------------------------
# Test 1: output has exactly 5 groups x len(horizon_cols) rows
# ---------------------------------------------------------------------------

def test_output_shape():
    df = make_synthetic_df()
    setup_mask = pd.Series(True, index=df.index)
    result = stratify_by_htf_state(df, setup_mask, "htf_state", "htf_bar_close", HORIZONS)

    expected_groups = set(STATE_ORDER) | {"unconditional"}
    assert len(result) == 5 * len(HORIZONS), f"Expected {5 * len(HORIZONS)} rows, got {len(result)}"

    groups_in_result = set(result.index.get_level_values("group"))
    assert groups_in_result == expected_groups, (
        f"Expected groups {expected_groups}, got {groups_in_result}"
    )

    horizons_in_result = set(result.index.get_level_values("horizon"))
    assert horizons_in_result == set(HORIZONS)


# ---------------------------------------------------------------------------
# Test 2: n_5m for "unconditional" equals setup_mask.sum()
# ---------------------------------------------------------------------------

def test_unconditional_n5m_equals_setup_mask_sum():
    df = make_synthetic_df()
    # Use a partial setup mask (only every other row)
    setup_mask = pd.Series(False, index=df.index)
    setup_mask.iloc[::2] = True

    result = stratify_by_htf_state(df, setup_mask, "htf_state", "htf_bar_close", HORIZONS)

    for horizon in HORIZONS:
        n_5m = result.loc[("unconditional", horizon), "n_5m"]
        assert n_5m == setup_mask.sum(), (
            f"n_5m={n_5m} != setup_mask.sum()={setup_mask.sum()} for horizon={horizon}"
        )


# ---------------------------------------------------------------------------
# Test 3: sum of n_5m across 4 per-state groups + NaN-state rows ==
#         n_5m of "unconditional"
# ---------------------------------------------------------------------------

def test_n5m_arithmetic_with_nan_states():
    df = make_synthetic_df()
    setup_mask = pd.Series(True, index=df.index)

    result = stratify_by_htf_state(df, setup_mask, "htf_state", "htf_bar_close", HORIZONS)

    setup_df = df.loc[setup_mask]
    n_nan_state = int(setup_df["htf_state"].isna().sum())

    for horizon in HORIZONS:
        per_state_sum = sum(
            result.loc[(state, horizon), "n_5m"] for state in STATE_ORDER
        )
        unconditional_n5m = result.loc[("unconditional", horizon), "n_5m"]
        assert per_state_sum + n_nan_state == unconditional_n5m, (
            f"per_state_sum({per_state_sum}) + n_nan_state({n_nan_state}) != "
            f"unconditional_n5m({unconditional_n5m}) for horizon={horizon}"
        )


# ---------------------------------------------------------------------------
# Test 4: n_htf <= n_5m for every cell, and n_htf correctly counts unique
#         HTF bar_close values (30 5m rows -> 10 unique HTF bars)
# ---------------------------------------------------------------------------

def test_n_htf_unique_count_and_le_n5m():
    df = make_synthetic_df(n_htf_bars=10, bars_per_htf=3)  # 30 rows, 10 unique HTF bars
    setup_mask = pd.Series(True, index=df.index)

    result = stratify_by_htf_state(df, setup_mask, "htf_state", "htf_bar_close", HORIZONS)

    # n_htf <= n_5m everywhere
    assert (result["n_htf"] <= result["n_5m"]).all(), (
        f"Found n_htf > n_5m:\n{result[result['n_htf'] > result['n_5m']]}"
    )

    # unconditional: 30 5m rows -> 10 unique HTF bars
    for horizon in HORIZONS:
        cell = result.loc[("unconditional", horizon)]
        assert cell["n_5m"] == 30
        assert cell["n_htf"] == 10, f"Expected n_htf=10, got {cell['n_htf']}"

    # With n_htf_bars=10 and 5 NaN-state HTF bars dropped (every 5th: i=4, i=9 -> 2 NaN bars),
    # remaining 8 HTF bars cycle through STATE_ORDER (2 each), so each per-state
    # group should have n_5m=6 (2 HTF bars * 3 rows) and n_htf=2.
    for state in STATE_ORDER:
        for horizon in HORIZONS:
            cell = result.loc[(state, horizon)]
            assert cell["n_5m"] == 6, f"state={state}: expected n_5m=6, got {cell['n_5m']}"
            assert cell["n_htf"] == 2, f"state={state}: expected n_htf=2, got {cell['n_htf']}"


# ---------------------------------------------------------------------------
# Test 5: cell with n_5m < 5 has mean_ret and hit_rate == NaN
# ---------------------------------------------------------------------------

def test_low_n5m_nan_mean_and_hit_rate():
    np.random.seed(1)
    # 1 HTF bar with bull_accel state, only 3 5m rows -> n_5m=3 for that state
    df = make_synthetic_df(n_htf_bars=20, bars_per_htf=3)

    # Force only 3 rows of bull_accel total: take the full df, then restrict
    # setup_mask to just 3 of the bull_accel rows.
    bull_accel_idx = df.index[df["htf_state"] == "bull_accel"]
    assert len(bull_accel_idx) >= 3

    setup_mask = pd.Series(False, index=df.index)
    # Include all non-bull_accel rows plus exactly 3 bull_accel rows
    non_bull_idx = df.index[df["htf_state"] != "bull_accel"]
    setup_mask.loc[non_bull_idx] = True
    setup_mask.loc[bull_accel_idx[:3]] = True

    result = stratify_by_htf_state(df, setup_mask, "htf_state", "htf_bar_close", HORIZONS)

    for horizon in HORIZONS:
        cell = result.loc[("bull_accel", horizon)]
        assert cell["n_5m"] == 3
        assert pd.isna(cell["mean_ret"]), f"Expected mean_ret=NaN, got {cell['mean_ret']}"
        assert pd.isna(cell["hit_rate"]), f"Expected hit_rate=NaN, got {cell['hit_rate']}"


# ---------------------------------------------------------------------------
# Test 6: cell with 5 <= n_5m < 20 has mean_ret non-NaN but t_stat/p_val NaN
# ---------------------------------------------------------------------------

def test_mid_n5m_mean_ok_but_hac_nan():
    np.random.seed(2)
    df = make_synthetic_df(n_htf_bars=20, bars_per_htf=3)

    bull_accel_idx = df.index[df["htf_state"] == "bull_accel"]
    # bars_per_htf=3, n_htf_bars=20 -> bull_accel appears for HTF bar indices
    # 0, 5, 10, 15 (every 5th is NaN at i%5==4, bull_accel at i%4==0 within
    # non-NaN cycle -- just take however many rows are available, slice to 9)
    assert len(bull_accel_idx) >= 9

    setup_mask = pd.Series(False, index=df.index)
    non_bull_idx = df.index[df["htf_state"] != "bull_accel"]
    setup_mask.loc[non_bull_idx] = True
    setup_mask.loc[bull_accel_idx[:9]] = True  # 9 rows: >=5 and <20

    result = stratify_by_htf_state(df, setup_mask, "htf_state", "htf_bar_close", HORIZONS)

    for horizon in HORIZONS:
        cell = result.loc[("bull_accel", horizon)]
        assert cell["n_5m"] == 9
        assert not pd.isna(cell["mean_ret"]), f"Expected mean_ret not NaN, got NaN"
        assert pd.isna(cell["t_stat"]), f"Expected t_stat=NaN for n_5m=9, got {cell['t_stat']}"
        assert pd.isna(cell["p_val_raw"]), f"Expected p_val_raw=NaN for n_5m=9, got {cell['p_val_raw']}"


# ---------------------------------------------------------------------------
# Test 7: hit_rate values are in [0, 1] where non-NaN
# ---------------------------------------------------------------------------

def test_hit_rate_range():
    df = make_synthetic_df()
    setup_mask = pd.Series(True, index=df.index)

    result = stratify_by_htf_state(df, setup_mask, "htf_state", "htf_bar_close", HORIZONS)

    valid = result["hit_rate"].dropna()
    assert (valid >= 0).all() and (valid <= 1).all(), (
        f"hit_rate out of [0,1]: {valid[~((valid >= 0) & (valid <= 1))]}"
    )


# ---------------------------------------------------------------------------
# Test 8: NaN htf_state rows excluded from per-state groups but included in
#         "unconditional" -- explicit count arithmetic check
# ---------------------------------------------------------------------------

def test_nan_htf_state_excluded_from_per_state_included_in_unconditional():
    df = make_synthetic_df(n_htf_bars=20, bars_per_htf=3)
    setup_mask = pd.Series(True, index=df.index)

    result = stratify_by_htf_state(df, setup_mask, "htf_state", "htf_bar_close", HORIZONS)

    n_total = len(df)
    n_nan_state = int(df["htf_state"].isna().sum())
    assert n_nan_state > 0, "Test setup error: expected some NaN htf_state rows"

    for horizon in HORIZONS:
        unconditional_n5m = result.loc[("unconditional", horizon), "n_5m"]
        assert unconditional_n5m == n_total

        per_state_sum = sum(
            result.loc[(state, horizon), "n_5m"] for state in STATE_ORDER
        )
        # NaN rows must be excluded from per-state groups
        assert per_state_sum == n_total - n_nan_state
        # but included in unconditional
        assert per_state_sum + n_nan_state == unconditional_n5m


# ---------------------------------------------------------------------------
# Test 9: empty result handling (setup_mask all-False)
# ---------------------------------------------------------------------------

def test_empty_setup_mask():
    df = make_synthetic_df()
    setup_mask = pd.Series(False, index=df.index)

    result = stratify_by_htf_state(df, setup_mask, "htf_state", "htf_bar_close", HORIZONS)

    assert len(result) == 0
    assert list(result.index.names) == ["group", "horizon"]
    expected_cols = {"n_5m", "n_htf", "mean_ret", "hit_rate", "t_stat", "p_val_raw"}
    assert set(result.columns) == expected_cols
