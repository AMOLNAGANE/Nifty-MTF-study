import pytest
import pandas as pd
import numpy as np
from src.phase2_features.master_join import merge_ab_features, apply_warmup_drop, build_master_5m

TZ = "Asia/Kolkata"
WARMUP = 270


def _make_df(n, start_ts, indicator, close_val=100.0):
    start = pd.Timestamp(start_ts, tz=TZ)
    ts = [start + pd.Timedelta(minutes=5 * i) for i in range(n)]
    df = pd.DataFrame({
        "timestamp": ts,
        "bar_close": [t + pd.Timedelta(minutes=5) for t in ts],
        "open": close_val, "high": close_val + 1, "low": close_val - 1,
        "close": close_val,
        "session_date": [t.date() for t in ts],
        "bar_in_session": range(1, n + 1),
    })
    if indicator == "A":
        df["A_hist"] = np.random.randn(n)
        df["A_state"] = "bull_accel"
    else:
        df["B_roc"] = np.random.randn(n)
        df["B_state"] = "bear_decel"
    return df


def test_merge_ab_has_both_prefixes():
    df_a = _make_df(100, "2020-01-02 09:15:00", "A")
    df_b = _make_df(100, "2020-01-02 09:15:00", "B")
    result = merge_ab_features(df_a, df_b)
    assert "A_state" in result.columns
    assert "B_state" in result.columns


def test_merge_ab_row_count_unchanged():
    df_a = _make_df(100, "2020-01-02 09:15:00", "A")
    df_b = _make_df(100, "2020-01-02 09:15:00", "B")
    assert len(merge_ab_features(df_a, df_b)) == 100


def test_merge_ab_no_duplicate_ohlc_columns():
    df_a = _make_df(100, "2020-01-02 09:15:00", "A")
    df_b = _make_df(100, "2020-01-02 09:15:00", "B")
    result = merge_ab_features(df_a, df_b)
    for col in ["open", "high", "low", "close"]:
        assert result.columns.tolist().count(col) == 1


def test_apply_warmup_drop_removes_correct_rows():
    df = _make_df(500, "2020-01-02 09:15:00", "A")
    result = apply_warmup_drop(df, warmup=WARMUP)
    assert len(result) == 500 - WARMUP
    assert result["timestamp"].iloc[0] == df["timestamp"].iloc[WARMUP]


def test_apply_warmup_drop_resets_index():
    df = _make_df(500, "2020-01-02 09:15:00", "A")
    result = apply_warmup_drop(df, warmup=WARMUP)
    assert result.index.tolist() == list(range(len(result)))


def test_build_master_5m_no_lookahead():
    """bar_close_{tf} must be <= 5m bar_close in all non-NaN rows."""
    n5 = 300
    start = pd.Timestamp("2020-01-02 09:15:00", tz=TZ)
    ts5 = [start + pd.Timedelta(minutes=5 * i) for i in range(n5)]
    df5 = pd.DataFrame({
        "timestamp": ts5,
        "bar_close": [t + pd.Timedelta(minutes=5) for t in ts5],
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "session_date": [t.date() for t in ts5],
        "bar_in_session": range(1, n5 + 1),
        "A_hist": 1.0, "B_roc": 1.0, "A_state": "bull_accel", "B_state": "bull_accel",
    })
    ts15 = [start + pd.Timedelta(minutes=15 * i) for i in range(n5 // 3)]
    df15 = pd.DataFrame({
        "timestamp": ts15,
        "bar_close": [t + pd.Timedelta(minutes=15) for t in ts15],
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "session_date": [t.date() for t in ts15],
        "bar_in_session": range(1, len(ts15) + 1),
        "A_hist": 2.0, "B_roc": 2.0, "A_state": "bull_decel", "B_state": "bull_decel",
    })
    master = build_master_5m(df_5m=df5, df_15m=df15, df_1h=df15, df_1d=df15)
    # Only check rows where HTF bar_close is non-null (first few 5m bars may have no prior HTF bar)
    for col in ["bar_close_15m", "bar_close_1h", "bar_close_1d"]:
        valid = master[col].notna()
        assert valid.any(), f"No non-null rows for {col}"
        assert (master.loc[valid, col] <= master.loc[valid, "bar_close"]).all(), \
            f"Look-ahead violation in {col}"


def test_build_master_5m_htf_columns_suffixed():
    n5 = 100
    start = pd.Timestamp("2020-01-02 09:15:00", tz=TZ)
    ts = [start + pd.Timedelta(minutes=5 * i) for i in range(n5)]
    df5 = pd.DataFrame({
        "timestamp": ts, "bar_close": [t + pd.Timedelta(minutes=5) for t in ts],
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "session_date": [t.date() for t in ts], "bar_in_session": range(1, n5 + 1),
        "A_hist": 1.0, "B_roc": 1.0, "A_state": "bull_accel", "B_state": "bull_accel",
    })
    ts15 = [start + pd.Timedelta(minutes=15 * i) for i in range(n5 // 3)]
    df15 = pd.DataFrame({
        "timestamp": ts15, "bar_close": [t + pd.Timedelta(minutes=15) for t in ts15],
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "session_date": [t.date() for t in ts15], "bar_in_session": range(1, len(ts15) + 1),
        "A_hist": 2.0, "B_roc": 2.0, "A_state": "bull_decel", "B_state": "bull_decel",
    })
    master = build_master_5m(df5, df15, df15, df15)
    assert "A_hist_15m" in master.columns
    assert "B_roc_15m" in master.columns
    assert "A_state_15m" in master.columns
    assert "bar_close_15m" in master.columns


def test_build_master_5m_no_duplicate_bar_close_cols():
    """bar_close_{tf} must appear exactly once (not duplicated in column list)."""
    n5 = 60
    start = pd.Timestamp("2020-01-02 09:15:00", tz=TZ)
    ts = [start + pd.Timedelta(minutes=5 * i) for i in range(n5)]
    df5 = pd.DataFrame({
        "timestamp": ts, "bar_close": [t + pd.Timedelta(minutes=5) for t in ts],
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "session_date": [t.date() for t in ts], "bar_in_session": range(1, n5 + 1),
        "A_hist": 1.0, "B_roc": 1.0, "A_state": "bull_accel", "B_state": "bull_accel",
    })
    ts15 = [start + pd.Timedelta(minutes=15 * i) for i in range(n5 // 3)]
    df15 = pd.DataFrame({
        "timestamp": ts15, "bar_close": [t + pd.Timedelta(minutes=15) for t in ts15],
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "session_date": [t.date() for t in ts15], "bar_in_session": range(1, len(ts15) + 1),
        "A_hist": 2.0, "B_roc": 2.0, "A_state": "bull_decel", "B_state": "bull_decel",
    })
    master = build_master_5m(df5, df15, df15, df15)
    for col in ["bar_close_15m", "bar_close_1h", "bar_close_1d"]:
        assert master.columns.tolist().count(col) == 1, f"{col} duplicated in master columns"
