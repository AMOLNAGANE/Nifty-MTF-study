from __future__ import annotations
import pandas as pd


def _session_bin_start(ts: pd.Timestamp, bin_minutes: int) -> pd.Timestamp:
    session_open = ts.replace(hour=9, minute=15, second=0, microsecond=0)
    elapsed_min = int((ts - session_open).total_seconds() / 60)
    bin_idx = elapsed_min // bin_minutes
    return session_open + pd.Timedelta(minutes=bin_idx * bin_minutes)


def _agg_ohlc(grp: pd.DataFrame, bin_minutes: int) -> dict:
    first_ts = grp["timestamp"].iloc[0]
    return {
        "timestamp": first_ts,
        "bar_close": first_ts + pd.Timedelta(minutes=bin_minutes),
        "open": grp["open"].iloc[0],
        "high": grp["high"].max(),
        "low": grp["low"].min(),
        "close": grp["close"].iloc[-1],
        "session_date": grp["session_date"].iloc[0],
    }


def resample_to_15m(df_5m: pd.DataFrame) -> pd.DataFrame:
    df = df_5m.copy()
    df["_bin"] = df["timestamp"].apply(lambda ts: _session_bin_start(ts, 15))
    records = [
        _agg_ohlc(grp, 15)
        for (_, _), grp in df.groupby(["session_date", "_bin"], sort=True)
    ]
    result = pd.DataFrame(records).reset_index(drop=True)
    result["bar_in_session"] = result.groupby("session_date", sort=False).cumcount() + 1
    return result[["timestamp", "bar_close", "open", "high", "low", "close",
                   "session_date", "bar_in_session"]]


def resample_to_1h(df_5m: pd.DataFrame) -> pd.DataFrame:
    """Resample to session-anchored 1h bars. Drops the 15:15–15:30 stub (minutes >= 360)."""
    df = df_5m.copy()
    df["_mins"] = df["timestamp"].apply(
        lambda ts: int(
            (ts - ts.replace(hour=9, minute=15, second=0, microsecond=0)).total_seconds() / 60
        )
    )
    df = df[df["_mins"] < 360].copy()
    df["_bin"] = df["timestamp"].apply(lambda ts: _session_bin_start(ts, 60))
    records = [
        _agg_ohlc(grp, 60)
        for (_, _), grp in df.groupby(["session_date", "_bin"], sort=True)
    ]
    result = pd.DataFrame(records).reset_index(drop=True)
    result["bar_in_session"] = result.groupby("session_date", sort=False).cumcount() + 1
    return result[["timestamp", "bar_close", "open", "high", "low", "close",
                   "session_date", "bar_in_session"]]


def resample_to_1d(df_5m: pd.DataFrame) -> pd.DataFrame:
    records = []
    for date, grp in df_5m.groupby("session_date", sort=True):
        grp = grp.sort_values("timestamp")
        first_ts = grp["timestamp"].iloc[0]
        records.append({
            "timestamp": first_ts,
            "bar_close": first_ts.replace(hour=15, minute=30, second=0, microsecond=0),
            "open": grp["open"].iloc[0],
            "high": grp["high"].max(),
            "low": grp["low"].min(),
            "close": grp["close"].iloc[-1],
            "session_date": date,
            "bar_in_session": 1,
        })
    return pd.DataFrame(records).reset_index(drop=True)
