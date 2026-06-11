"""
Phase 3 Analysis 3a — Event Studies (H2–H7)
============================================
Implements hypothesis-driven event studies for the Multi-Timeframe Momentum
Confluence study. Each function accepts a DataFrame that already contains
indicator feature columns and forward-return columns, and returns structured
results (events DataFrame + optional summary dict).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

# Default forward-return horizons used throughout the module
_DEFAULT_HORIZONS: list[str] = [
    "ret_fwd_1",
    "ret_fwd_3",
    "ret_fwd_6",
    "ret_fwd_12",
    "ret_fwd_24",
]


# ---------------------------------------------------------------------------
# H2 — Peak timing: bull_accel → bull_decel transition
# ---------------------------------------------------------------------------


def event_study_h2_peak_timing(
    df: pd.DataFrame,
    horizon_cols: list[str],
) -> tuple[pd.DataFrame, dict]:
    """
    Detect bars where A_state transitions from "bull_accel" (row i) to
    "bull_decel" (row i+1).  The event bar is row i+1 (the momentum-peak bar).

    Parameters
    ----------
    df : pd.DataFrame
        Feature-enriched intraday DataFrame.
    horizon_cols : list[str]
        Forward-return column names to collect at each event bar.

    Returns
    -------
    event_df : pd.DataFrame
        One row per detected event; columns = horizon_cols.
    summary : dict
        {
            "n_events": int,
            "mean_fwd_ret": {horizon: float},
            "short_vs_long": float   # mean(ret_fwd_3)/mean(ret_fwd_12), or NaN
        }
    """
    states = df["A_state"].values
    n = len(states)
    event_rows: list[int] = []

    for i in range(n - 1):
        if states[i] == "bull_accel" and states[i + 1] == "bull_decel":
            event_rows.append(i + 1)  # event at the decel bar

    if not event_rows:
        empty_df = pd.DataFrame(columns=horizon_cols)
        mean_fwd: dict[str, float] = {h: float("nan") for h in horizon_cols}
        return empty_df, {"n_events": 0, "mean_fwd_ret": mean_fwd, "short_vs_long": float("nan")}

    event_df = df.iloc[event_rows][horizon_cols].reset_index(drop=True)

    mean_fwd = {h: float(event_df[h].mean()) for h in horizon_cols}

    r3 = mean_fwd.get("ret_fwd_3", 0.0)
    r12 = mean_fwd.get("ret_fwd_12", 0.0)
    if r12 != 0.0:
        short_vs_long = r3 / r12
    else:
        short_vs_long = float("nan")

    summary: dict = {
        "n_events": len(event_rows),
        "mean_fwd_ret": mean_fwd,
        "short_vs_long": short_vs_long,
    }
    return event_df, summary


# ---------------------------------------------------------------------------
# H3 — Aligned vs tired zero-cross
# ---------------------------------------------------------------------------


def event_study_h3_aligned_vs_tired(
    df: pd.DataFrame,
    horizon_cols: list[str],
) -> pd.DataFrame:
    """
    Split bullish MACD zero-crosses (A_macd_zero_cross == 1) into:
      - "aligned"  : A_hist > 0 AND A_hist_slope > 0
      - "tired"    : all others (A_hist <= 0 OR A_hist_slope <= 0)

    Parameters
    ----------
    df : pd.DataFrame
        Feature-enriched DataFrame.
    horizon_cols : list[str]
        Forward-return columns to collect.

    Returns
    -------
    pd.DataFrame
        Columns: ["group"] + horizon_cols.  One row per event.
    """
    cross_mask = df["A_macd_zero_cross"] == 1
    cross_df = df[cross_mask].copy()

    aligned_mask = (cross_df["A_hist"] > 0) & (cross_df["A_hist_slope"] > 0)

    records: list[dict] = []
    for idx in cross_df.index:
        group = "aligned" if aligned_mask.loc[idx] else "tired"
        row: dict = {"group": group}
        for h in horizon_cols:
            row[h] = cross_df.loc[idx, h]
        records.append(row)

    if not records:
        return pd.DataFrame(columns=["group"] + horizon_cols)

    return pd.DataFrame(records).reset_index(drop=True)


# ---------------------------------------------------------------------------
# H4 — First-pullback (hist re-cross stratified by MACD sign)
# ---------------------------------------------------------------------------


def event_study_h4_pullback(
    df: pd.DataFrame,
    horizon_cols: list[str],
) -> pd.DataFrame:
    """
    Split bullish histogram zero-crosses (A_hist_zero_cross == 1) into:
      - "with_trend"    : A_macd > 0 (pullback within established uptrend)
      - "counter_trend" : A_macd <= 0

    Parameters
    ----------
    df : pd.DataFrame
        Feature-enriched DataFrame.
    horizon_cols : list[str]
        Forward-return columns.

    Returns
    -------
    pd.DataFrame
        Columns: ["group"] + horizon_cols.  One row per event.
    """
    cross_mask = df["A_hist_zero_cross"] == 1
    cross_df = df[cross_mask].copy()

    records: list[dict] = []
    for idx in cross_df.index:
        group = "with_trend" if cross_df.loc[idx, "A_macd"] > 0 else "counter_trend"
        row: dict = {"group": group}
        for h in horizon_cols:
            row[h] = cross_df.loc[idx, h]
        records.append(row)

    if not records:
        return pd.DataFrame(columns=["group"] + horizon_cols)

    return pd.DataFrame(records).reset_index(drop=True)


# ---------------------------------------------------------------------------
# H6 — zROC term-structure: continuation vs exhaustion
# ---------------------------------------------------------------------------

_ZROC_BUCKETS = {
    "baseline": lambda z: np.isfinite(z) & (np.abs(z) <= 0.5),
    "moderate": lambda z: np.isfinite(z) & (z > 0.5) & (z <= 2.0),
    "extreme_bull": lambda z: np.isfinite(z) & (z > 2.0),
    "extreme_bear": lambda z: np.isfinite(z) & (z < -2.0),
}


def event_study_h6_zroc_term_structure(
    df: pd.DataFrame,
    horizon_cols: list[str],
) -> pd.DataFrame:
    """
    Categorise each bar by B_zroc bucket and compute mean forward returns.

    Buckets:
        "baseline"    : |B_zroc| <= 0.5
        "moderate"    : 0.5 < B_zroc <= 2.0
        "extreme_bull": B_zroc > 2.0
        "extreme_bear": B_zroc < -2.0

    Parameters
    ----------
    df : pd.DataFrame
        Feature-enriched DataFrame.
    horizon_cols : list[str]
        Forward-return columns.

    Returns
    -------
    pd.DataFrame
        Index = bucket name, columns = horizon_cols.  Only non-empty buckets included.
    """
    zroc = df["B_zroc"].values.astype(float)
    rows: dict[str, dict[str, float]] = {}

    for bucket, predicate in _ZROC_BUCKETS.items():
        mask = predicate(zroc)
        if not mask.any():
            continue
        bucket_returns: dict[str, float] = {}
        for h in horizon_cols:
            bucket_returns[h] = float(df.loc[mask, h].mean())
        rows[bucket] = bucket_returns

    if not rows:
        return pd.DataFrame(columns=horizon_cols)

    result = pd.DataFrame(rows).T
    result.index.name = None
    return result[horizon_cols]


# ---------------------------------------------------------------------------
# H7 — Early-turn: gap acceleration before gap zero-cross
# ---------------------------------------------------------------------------


def event_study_h7_early_turn(
    df: pd.DataFrame,
    m_bars: int = 3,
    horizon_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Detect "early turn" events: bars where B_gap < 0 but B_gap_acc > 0 for
    m_bars consecutive bars (gap is closing).  The event bar is the m_bars-th
    consecutive bar.

    For each event, check whether B_gap crosses 0 (becomes >= 0) within the
    next 20 bars.  Records a "confirmed" boolean.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-enriched DataFrame.
    m_bars : int
        Number of consecutive bars with B_gap_acc > 0 required.
    horizon_cols : list[str] or None
        Forward-return columns; defaults to _DEFAULT_HORIZONS.

    Returns
    -------
    event_df : pd.DataFrame
        Columns: ["confirmed"] + horizon_cols.  One row per event.
    summary : dict
        {"n_events": int, "false_start_rate": float}
    """
    if horizon_cols is None:
        horizon_cols = _DEFAULT_HORIZONS

    gap = df["B_gap"].values.astype(float)
    gap_acc = df["B_gap_acc"].values.astype(float)
    n = len(df)

    event_rows: list[int] = []
    consecutive = 0

    for i in range(n):
        if gap[i] < 0 and gap_acc[i] > 0:
            consecutive += 1
            if consecutive >= m_bars:
                event_rows.append(i)
        else:
            consecutive = 0

    if not event_rows:
        empty_df = pd.DataFrame(columns=["confirmed"] + horizon_cols)
        return empty_df, {"n_events": 0, "false_start_rate": float("nan")}

    records: list[dict] = []
    for row_idx in event_rows:
        # Check confirmation: B_gap crosses 0 in the next 20 bars
        look_ahead_end = min(row_idx + 21, n)   # rows row_idx+1 .. row_idx+20
        future_gap = gap[row_idx + 1 : look_ahead_end]
        confirmed = bool(len(future_gap) > 0 and (future_gap >= 0).any())

        record: dict = {"confirmed": confirmed}
        for h in horizon_cols:
            record[h] = float(df.iloc[row_idx][h])
        records.append(record)

    event_df = pd.DataFrame(records).reset_index(drop=True)

    n_events = len(event_df)
    false_starts = int((~event_df["confirmed"]).sum())
    false_start_rate = false_starts / n_events if n_events > 0 else float("nan")

    summary: dict = {
        "n_events": n_events,
        "false_start_rate": false_start_rate,
    }
    return event_df, summary
