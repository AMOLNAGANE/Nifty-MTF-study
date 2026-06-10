"""Gate 1: no-look-ahead validation.

Verifies that every HTF feature in master_5m comes from a bar that had
already closed by the time of the 5m bar. Failure here means Phase 3
results would be invalid — do NOT proceed until all tests pass.
"""
import pytest
import pandas as pd
from pathlib import Path

MASTER = Path("data/features/master_5m.parquet")
N_SAMPLE = 1000
SEED = 42


@pytest.fixture(scope="module")
def master():
    if not MASTER.exists():
        pytest.skip("master_5m.parquet not found — run Phase 2 first")
    return pd.read_parquet(MASTER)


def test_master_has_bar_close_tracking_columns(master):
    for col in ["bar_close", "bar_close_15m", "bar_close_1h", "bar_close_1d"]:
        assert col in master.columns, f"Missing: {col}"


def test_no_lookahead_15m_sample(master):
    sample = master.sample(n=N_SAMPLE, random_state=SEED)
    valid = sample["bar_close_15m"].notna()
    violations = (sample.loc[valid, "bar_close_15m"] > sample.loc[valid, "bar_close"]).sum()
    assert violations == 0, (
        f"GATE 1 FAIL: {violations}/{valid.sum()} rows have 15m look-ahead. "
        "Fix master_join.py before proceeding to Phase 3."
    )


def test_no_lookahead_1h_sample(master):
    sample = master.sample(n=N_SAMPLE, random_state=SEED)
    valid = sample["bar_close_1h"].notna()
    violations = (sample.loc[valid, "bar_close_1h"] > sample.loc[valid, "bar_close"]).sum()
    assert violations == 0, (
        f"GATE 1 FAIL: {violations}/{valid.sum()} rows have 1h look-ahead."
    )


def test_no_lookahead_1d_sample(master):
    sample = master.sample(n=N_SAMPLE, random_state=SEED)
    valid = sample["bar_close_1d"].notna()
    violations = (sample.loc[valid, "bar_close_1d"] > sample.loc[valid, "bar_close"]).sum()
    assert violations == 0, (
        f"GATE 1 FAIL: {violations}/{valid.sum()} rows have 1d look-ahead."
    )


def test_no_lookahead_full_population(master):
    for col, tf in [("bar_close_15m", "15m"), ("bar_close_1h", "1h"), ("bar_close_1d", "1d")]:
        valid = master[col].notna()
        v = (master.loc[valid, col] > master.loc[valid, "bar_close"]).sum()
        assert v == 0, f"GATE 1 FAIL: {v} {tf} violations (full population)"


def test_master_row_count_reasonable(master):
    assert 90_000 < len(master) < 100_000, f"Unexpected row count: {len(master)}"


def test_no_null_in_native_5m_state(master):
    null_pct = master["A_state"].isna().mean()
    assert null_pct < 0.005, f"A_state NaN rate {null_pct:.1%} > 0.5% threshold"


def test_htf_indicator_columns_present_for_all_tfs(master):
    for tf in ["15m", "1h", "1d"]:
        for col in [f"A_hist_{tf}", f"A_state_{tf}", f"B_roc_{tf}", f"B_state_{tf}"]:
            assert col in master.columns, f"Missing: {col}"
