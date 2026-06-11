from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Function 1: detect_peaks
# ---------------------------------------------------------------------------

def detect_peaks(
    series: pd.Series,
    min_spacing: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Find local highs and lows in a price/indicator series.

    Parameters
    ----------
    series : pd.Series
        The time series to analyse (integer-positional indexing assumed).
    min_spacing : int
        Minimum number of bars between consecutive peaks (maps to ``distance``
        in :func:`scipy.signal.find_peaks`).

    Returns
    -------
    high_indices : np.ndarray
        Integer positions of local highs (raw peak positions, NOT confirmation).
    low_indices : np.ndarray
        Integer positions of local lows (raw trough positions).

    Notes
    -----
    **Look-ahead safety**: a peak at bar T can only be *confirmed* at bar
    T + min_spacing // 2.  This function returns raw peak indices; callers are
    responsible for applying the confirmation offset before computing forward
    returns (see :func:`detect_divergence`).
    """
    values = series.to_numpy(dtype=float)

    high_indices, _ = find_peaks(values, distance=min_spacing)
    low_indices, _ = find_peaks(-values, distance=min_spacing)

    return high_indices, low_indices


# ---------------------------------------------------------------------------
# Function 2: detect_divergence
# ---------------------------------------------------------------------------

def detect_divergence(
    df: pd.DataFrame,
    price_col: str,
    indicator_col: str,
    min_spacing: int = 10,
) -> pd.DataFrame:
    """Detect classical price–indicator divergences.

    Bearish divergence
        Price makes a **higher high** but the indicator makes a **lower high**
        at the same swing point → signals potential downside reversal.

    Bullish divergence
        Price makes a **lower low** but the indicator makes a **higher low**
        at the same swing point → signals potential upside reversal.

    The returned ``confirm_idx`` is ``peak_idx + min_spacing // 2``.  Forward
    returns **must** be measured from ``confirm_idx``, not ``peak_idx``, to
    avoid look-ahead bias (see DESIGN_DOC.md §5.5).

    Parameters
    ----------
    df : pd.DataFrame
        Source data; must contain ``price_col`` and ``indicator_col``.
    price_col : str
        Column name for price.
    indicator_col : str
        Column name for the momentum/oscillator indicator.
    min_spacing : int
        Passed to :func:`detect_peaks`.

    Returns
    -------
    pd.DataFrame
        Columns: peak_idx, confirm_idx, div_type, price_direction,
        indicator_direction.  Events whose ``confirm_idx >= len(df)`` are
        excluded.
    """
    price_vals = df[price_col].to_numpy(dtype=float)
    indicator_vals = df[indicator_col].to_numpy(dtype=float)
    n = len(df)
    half_window = min_spacing // 2

    price_series = pd.Series(price_vals)
    indicator_series = pd.Series(indicator_vals)

    high_idx, low_idx = detect_peaks(price_series, min_spacing=min_spacing)

    records: list[dict] = []

    # ------------------------------------------------------------------ highs
    # For each consecutive pair of price highs, check for bearish divergence
    for i in range(len(high_idx) - 1):
        idx1, idx2 = int(high_idx[i]), int(high_idx[i + 1])
        confirm_idx = idx2 + half_window

        # Skip if confirmation bar falls outside the DataFrame
        if confirm_idx >= n:
            continue

        price_h1 = price_vals[idx1]
        price_h2 = price_vals[idx2]
        ind_h1 = indicator_vals[idx1]
        ind_h2 = indicator_vals[idx2]

        # Bearish divergence: price higher high + indicator lower high
        if price_h2 > price_h1 and ind_h2 < ind_h1:
            records.append({
                "peak_idx": idx2,
                "confirm_idx": confirm_idx,
                "div_type": "bearish",
                "price_direction": "higher_high",
                "indicator_direction": "lower_high",
            })

    # ------------------------------------------------------------------- lows
    # For each consecutive pair of price lows, check for bullish divergence
    for i in range(len(low_idx) - 1):
        idx1, idx2 = int(low_idx[i]), int(low_idx[i + 1])
        confirm_idx = idx2 + half_window

        # Skip if confirmation bar falls outside the DataFrame
        if confirm_idx >= n:
            continue

        price_l1 = price_vals[idx1]
        price_l2 = price_vals[idx2]
        ind_l1 = indicator_vals[idx1]
        ind_l2 = indicator_vals[idx2]

        # Bullish divergence: price lower low + indicator higher low
        if price_l2 < price_l1 and ind_l2 > ind_l1:
            records.append({
                "peak_idx": idx2,
                "confirm_idx": confirm_idx,
                "div_type": "bullish",
                "price_direction": "lower_low",
                "indicator_direction": "higher_low",
            })

    columns = ["peak_idx", "confirm_idx", "div_type", "price_direction", "indicator_direction"]
    if not records:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(records, columns=columns).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Function 3: compute_divergence_returns
# ---------------------------------------------------------------------------

def compute_divergence_returns(
    df: pd.DataFrame,
    divergence_df: pd.DataFrame,
    horizon_cols: list[str],
    ret_col_prefix: str = "ret_fwd_",
) -> pd.DataFrame:
    """Look up forward returns at the confirmation bar for each divergence event.

    The confirmation bar is ``confirm_idx`` (not ``peak_idx``).  Using
    ``peak_idx`` would be look-ahead bias (see DESIGN_DOC.md §5.5).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns named in ``horizon_cols`` (e.g. ``ret_fwd_1``).
    divergence_df : pd.DataFrame
        Output of :func:`detect_divergence`.  Must have columns ``confirm_idx``
        and ``div_type``.
    horizon_cols : list[str]
        Forward-return columns to look up (e.g. ``["ret_fwd_1", "ret_fwd_3"]``).
    ret_col_prefix : str
        Unused — retained for API compatibility.

    Returns
    -------
    pd.DataFrame
        Columns: ``["div_type"] + horizon_cols``.  One row per divergence
        event.  Values are NaN when ``confirm_idx`` is out of bounds or the
        forward return at that row is NaN.
    """
    n = len(df)
    rows: list[dict] = []

    for _, event in divergence_df.iterrows():
        confirm_idx = int(event["confirm_idx"])
        row: dict = {"div_type": event["div_type"]}

        for col in horizon_cols:
            if confirm_idx < 0 or confirm_idx >= n:
                row[col] = float("nan")
            else:
                val = df.iloc[confirm_idx][col]
                row[col] = float("nan") if pd.isna(val) else val

        rows.append(row)

    result_cols = ["div_type"] + list(horizon_cols)
    if not rows:
        return pd.DataFrame(columns=result_cols)

    return pd.DataFrame(rows, columns=result_cols).reset_index(drop=True)
