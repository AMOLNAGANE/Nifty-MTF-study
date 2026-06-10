from __future__ import annotations
import pandas as pd
from pathlib import Path


def load_5m_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
    df = df.sort_values("timestamp").reset_index(drop=True)

    n_dups = df.duplicated(subset=["timestamp"]).sum()
    if n_dups:
        print(f"[loader] Dropping {n_dups} duplicate timestamps")
        df = df.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)

    n_nonzero = (df["volume"] != 0.0).sum()
    if n_nonzero:
        print(f"[loader] WARNING: {n_nonzero} non-zero volume rows — "
              f"volume is synthetic for NIFTY index; dropping")
    else:
        print("[loader] Volume column all zero — dropping")
    df = df.drop(columns=["volume"])

    bad_high = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
    bad_low = (df["low"] > df[["open", "close"]].min(axis=1)).sum()
    if bad_high or bad_low:
        print(f"[loader] WARNING: {bad_high} bad highs, {bad_low} bad lows in OHLC sanity")

    df["bar_close"] = df["timestamp"] + pd.Timedelta(minutes=5)
    df["session_date"] = df["timestamp"].dt.date
    df["bar_in_session"] = df.groupby("session_date", sort=False).cumcount() + 1

    return df[["timestamp", "bar_close", "open", "high", "low", "close",
               "session_date", "bar_in_session"]]
