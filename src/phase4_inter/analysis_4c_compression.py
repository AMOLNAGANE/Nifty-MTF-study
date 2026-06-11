from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Function 1: detect_compression_expansion
# ---------------------------------------------------------------------------

def detect_compression_expansion(
    df_tf: pd.DataFrame,
    gap_norm_col: str = "B_gap_norm",
    zroc_col: str = "B_zroc",
    bar_close_col: str = "bar_close",
    compression_quantile: float = 0.10,
    max_lookahead_bars: int = 10,
    expansion_threshold: float = 1.5,
) -> pd.DataFrame:
    """Detect compression -> expansion events on a TF-native DataFrame.

    Tests H13: "Compression -> expansion / squeeze analog". A bar is a
    "compression bar" when ``|gap_norm_col|`` is at or below a fixed
    in-sample quantile threshold (EMAs pinched together = trendless
    compression). For each compression bar at position ``i``, we scan
    forward up to ``max_lookahead_bars`` positions for the first bar ``j``
    where ``|zroc_col| > expansion_threshold`` ("expansion onset"). The
    sign of ``zroc_col`` at ``j`` is recorded as the predicted direction of
    the subsequent move.

    Parameters
    ----------
    df_tf : pd.DataFrame
        TF-native DataFrame (15m or 1h), sorted ascending by
        ``bar_close_col``, with no duplicate ``bar_close_col`` values and no
        NaNs in ``gap_norm_col`` / ``zroc_col`` (the caller is responsible
        for dropping warmup/NaN rows before calling this function).
    gap_norm_col : str
        Column holding the EMA-gap normalized by price.
    zroc_col : str
        Column holding the z-scored rate-of-change of the gap.
    bar_close_col : str
        Column holding the bar-close timestamp (returned for matched events).
    compression_quantile : float
        Quantile (e.g. 0.10 = lowest decile) of ``|gap_norm_col|`` used as
        the compression threshold.
    max_lookahead_bars : int
        Maximum number of bars after a compression bar to scan for an
        expansion onset (inclusive).
    expansion_threshold : float
        ``|zroc_col|`` must exceed this value to qualify as an "expansion
        onset".

    Notes
    -----
    **Threshold computation is NOT walk-forward.** ``thresh`` is computed as
    ``df_tf[gap_norm_col].abs().quantile(compression_quantile)`` over the
    WHOLE input DataFrame -- a single, fixed, in-sample threshold. This is a
    simplification appropriate for a descriptive event study (it answers
    "in this dataset, how do compression->expansion episodes behave?") but
    is NOT a true walk-forward / expanding-window threshold and must not be
    interpreted as a tradeable, point-in-time-valid signal without further
    work.

    Returns
    -------
    pd.DataFrame
        Columns ``["bar_close", "expansion_direction", "lag_bars"]``, one
        row per distinct expansion event (deduped on the expansion bar),
        sorted ascending by ``bar_close``. ``expansion_direction`` is
        ``int`` (+1 or -1) and ``lag_bars`` is ``int``. Empty DataFrame with
        these columns/dtypes if no events are found.
    """
    columns = ["bar_close", "expansion_direction", "lag_bars"]
    n = len(df_tf)

    if n == 0:
        return pd.DataFrame({
            "bar_close": pd.Series([], dtype=df_tf[bar_close_col].dtype if bar_close_col in df_tf.columns else "object"),
            "expansion_direction": pd.Series([], dtype=int),
            "lag_bars": pd.Series([], dtype=int),
        })[columns]

    gap_vals = df_tf[gap_norm_col].to_numpy(dtype=float)
    zroc_vals = df_tf[zroc_col].to_numpy(dtype=float)
    bar_close_vals = df_tf[bar_close_col].to_numpy()

    thresh = float(pd.Series(gap_vals).abs().quantile(compression_quantile))

    seen_j: set[int] = set()
    records: list[dict] = []

    for i in range(n):
        if abs(gap_vals[i]) > thresh:
            continue

        # Compression bar at position i; scan forward for expansion onset.
        end = min(i + max_lookahead_bars, n - 1)
        for j in range(i + 1, end + 1):
            if abs(zroc_vals[j]) > expansion_threshold:
                if j not in seen_j:
                    seen_j.add(j)
                    records.append({
                        "bar_close": bar_close_vals[j],
                        "expansion_direction": int(np.sign(zroc_vals[j])),
                        "lag_bars": j - i,
                    })
                break  # first qualifying j found for this compression bar

    if not records:
        return pd.DataFrame({
            "bar_close": pd.Series([], dtype=df_tf[bar_close_col].dtype),
            "expansion_direction": pd.Series([], dtype=int),
            "lag_bars": pd.Series([], dtype=int),
        })[columns]

    result = pd.DataFrame.from_records(records, columns=columns)
    result["expansion_direction"] = result["expansion_direction"].astype(int)
    result["lag_bars"] = result["lag_bars"].astype(int)
    result = result.sort_values("bar_close", ascending=True).reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Function 2: event_study_h13
# ---------------------------------------------------------------------------

def event_study_h13(
    master_df: pd.DataFrame,
    event_df: pd.DataFrame,
    htf_bar_close_col: str,
    horizon_cols: list[str],
) -> dict:
    """Event study for H13: does expansion direction predict subsequent moves?

    For each compression->expansion event in ``event_df`` (output of
    :func:`detect_compression_expansion`, with ``bar_close`` referring to the
    HIGHER-TF bar's ``bar_close``), find the first 5m row of ``master_df``
    whose ``htf_bar_close_col`` equals the event's ``bar_close`` -- this is
    the earliest 5m bar at which the higher-TF expansion event is "known"
    (no-look-ahead). Forward returns are measured from that 5m row.

    Parameters
    ----------
    master_df : pd.DataFrame
        5m master DataFrame, sorted ascending by its own (5m) ``bar_close``.
        Must contain ``htf_bar_close_col`` and all ``horizon_cols``.
    event_df : pd.DataFrame
        Output of :func:`detect_compression_expansion`. Columns
        ``["bar_close", "expansion_direction", "lag_bars"]``, where
        ``bar_close`` matches values appearing in
        ``master_df[htf_bar_close_col]``.
    htf_bar_close_col : str
        Name of the higher-TF ``bar_close`` column in ``master_df`` (e.g.
        ``"bar_close_15m"`` or ``"bar_close_1h"``).
    horizon_cols : list[str]
        5m forward-return columns to evaluate (e.g.
        ``["ret_fwd_3", "ret_fwd_12"]``).

    Returns
    -------
    dict
        ``{"n_events": int, "signed_ret": {h: float}, "abs_ret_event": {h: float},
        "abs_ret_baseline": {h: float}}``.

        - ``n_events``: number of MATCHED events (events whose
          ``bar_close`` is found in ``master_df[htf_bar_close_col]``).
        - ``signed_ret[h]``: mean of ``expansion_direction * matched_row[h]``
          across matched events (NaN-aware). Positive => expansion direction
          correctly predicted the subsequent move's sign, on average.
        - ``abs_ret_event[h]``: mean of ``|matched_row[h]|`` across matched
          events (NaN-aware).
        - ``abs_ret_baseline[h]``: mean of ``|master_df[h]|`` over ALL rows
          of ``master_df`` (NaN-aware), independent of events.

        If ``n_events == 0`` (``event_df`` empty or no matches), every
        ``signed_ret`` / ``abs_ret_event`` value is NaN, but
        ``abs_ret_baseline`` is still computed from ``master_df``.
    """
    # Baseline absolute return — independent of events.
    abs_ret_baseline: dict[str, float] = {}
    for h in horizon_cols:
        col = master_df[h].astype(float)
        abs_ret_baseline[h] = float(col.abs().mean())  # nan-aware: mean() skips NaN

    if event_df is None or event_df.empty:
        return {
            "n_events": 0,
            "signed_ret": {h: float("nan") for h in horizon_cols},
            "abs_ret_event": {h: float("nan") for h in horizon_cols},
            "abs_ret_baseline": abs_ret_baseline,
        }

    # For each event, find the first matching 5m row.
    matched_rows: list[pd.Series] = []
    matched_directions: list[int] = []

    htf_close_vals = master_df[htf_bar_close_col]

    for _, event in event_df.iterrows():
        target = event["bar_close"]
        mask = htf_close_vals == target
        if not bool(mask.any()):
            continue
        first_idx = mask.idxmax()  # first True position (idxmax on bool returns first max)
        matched_rows.append(master_df.loc[first_idx])
        matched_directions.append(int(event["expansion_direction"]))

    n_events = len(matched_rows)

    if n_events == 0:
        return {
            "n_events": 0,
            "signed_ret": {h: float("nan") for h in horizon_cols},
            "abs_ret_event": {h: float("nan") for h in horizon_cols},
            "abs_ret_baseline": abs_ret_baseline,
        }

    matched_df = pd.DataFrame(matched_rows).reset_index(drop=True)
    directions = np.array(matched_directions, dtype=float)

    signed_ret: dict[str, float] = {}
    abs_ret_event: dict[str, float] = {}

    for h in horizon_cols:
        ret_vals = matched_df[h].astype(float).to_numpy()
        signed_vals = directions * ret_vals
        signed_ret[h] = float(np.nanmean(signed_vals)) if not np.all(np.isnan(signed_vals)) else float("nan")
        abs_vals = np.abs(ret_vals)
        abs_ret_event[h] = float(np.nanmean(abs_vals)) if not np.all(np.isnan(abs_vals)) else float("nan")

    return {
        "n_events": n_events,
        "signed_ret": signed_ret,
        "abs_ret_event": abs_ret_event,
        "abs_ret_baseline": abs_ret_baseline,
    }
