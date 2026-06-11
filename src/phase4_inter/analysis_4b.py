from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Function 1: Locate zero-cross events
# ---------------------------------------------------------------------------

def find_zero_cross_events(
    df_tf: pd.DataFrame,
    zero_cross_col: str,
    bar_close_col: str = "bar_close",
) -> pd.DataFrame:
    """Find every bar where df_tf[zero_cross_col] != 0.

    df_tf: TF-native DataFrame, sorted ascending by bar_close_col, containing
    zero_cross_col (values in {-1, 0, +1}).

    Returns a DataFrame with columns ["bar_close", "direction"] — one row for
    every bar where df_tf[zero_cross_col] != 0 (direction = that value, +1 or
    -1), in the same (ascending) order as df_tf. Index reset to a clean
    RangeIndex. If no such bars exist, return an empty DataFrame with these
    two columns (and correct dtypes: bar_close should remain datetime-like,
    direction int).
    """
    mask = df_tf[zero_cross_col] != 0
    events = df_tf.loc[mask, [bar_close_col, zero_cross_col]].reset_index(drop=True)
    events = events.rename(columns={bar_close_col: "bar_close", zero_cross_col: "direction"})
    events["direction"] = events["direction"].astype(int)

    if events.empty:
        # Preserve a datetime-like dtype for bar_close even when empty.
        events["bar_close"] = events["bar_close"].astype(df_tf[bar_close_col].dtype)

    return events[["bar_close", "direction"]]


# ---------------------------------------------------------------------------
# Function 2: Cross-TF follow-through (lead-lag / precedence)
# ---------------------------------------------------------------------------

def cross_tf_followthrough(
    lower_df: pd.DataFrame,
    higher_df: pd.DataFrame,
    lower_zc_col: str,
    higher_zc_col: str,
    k_bars: int,
    bar_close_col: str = "bar_close",
) -> pd.DataFrame:
    """For each lower-TF zero-cross flip, check higher-TF follow-through.

    lower_df, higher_df: TF-native DataFrames (sorted ascending by
    bar_close_col, one row per bar, no duplicate bar_close, no NaN in
    bar_close_col or the zero-cross columns).

    For each row in lower_df where lower_df[lower_zc_col] != 0 (a "lower-TF
    flip event" at time T_low = that row's bar_close, with direction = the
    sign of lower_zc_col at that row):

      1. Select rows of higher_df with bar_close STRICTLY GREATER than T_low.
      2. Take the first k_bars of those (ascending time order) — the
         "lookahead window".
      3. followed_through = True if ANY row in the window has
         higher_df[higher_zc_col] == direction.
      4. lag_bars = 1-based position of the FIRST matching row within the
         window. NaN if followed_through is False.

    Returns a DataFrame with one row per lower-TF flip event (ascending
    bar_close order), columns:
        ["bar_close", "direction", "followed_through", "lag_bars"]
    bar_close here is the LOWER-TF event's bar_close. dtypes: direction=int,
    followed_through=bool, lag_bars=float.

    If lower_df has no flip events, return an empty DataFrame with these 4
    columns.
    """
    out_cols = ["bar_close", "direction", "followed_through", "lag_bars"]

    flip_events = find_zero_cross_events(lower_df, lower_zc_col, bar_close_col=bar_close_col)

    if flip_events.empty:
        empty = pd.DataFrame(columns=out_cols)
        empty["bar_close"] = empty["bar_close"].astype(lower_df[bar_close_col].dtype)
        empty["direction"] = empty["direction"].astype(int)
        empty["followed_through"] = empty["followed_through"].astype(bool)
        empty["lag_bars"] = empty["lag_bars"].astype(float)
        return empty

    higher_close = higher_df[bar_close_col]
    higher_zc = higher_df[higher_zc_col].values

    records = []
    for _, ev in flip_events.iterrows():
        t_low = ev["bar_close"]
        direction = int(ev["direction"])

        # Strictly-future higher-TF bars (no-look-ahead).
        future_mask = (higher_close > t_low).values
        future_zc = higher_zc[future_mask]

        # Lookahead window: first k_bars of strictly-future bars.
        window = future_zc[:k_bars]

        followed_through = False
        lag_bars = float("nan")
        matches = np.flatnonzero(window == direction)
        if matches.size > 0:
            followed_through = True
            lag_bars = float(matches[0] + 1)  # 1-based position

        records.append({
            "bar_close": t_low,
            "direction": direction,
            "followed_through": followed_through,
            "lag_bars": lag_bars,
        })

    result = pd.DataFrame(records, columns=out_cols)
    result["direction"] = result["direction"].astype(int)
    result["followed_through"] = result["followed_through"].astype(bool)
    result["lag_bars"] = result["lag_bars"].astype(float)
    return result


# ---------------------------------------------------------------------------
# Function 3: Transition summary
# ---------------------------------------------------------------------------

def transition_summary(events_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize follow-through statistics by flip direction.

    events_df: output of cross_tf_followthrough (columns: bar_close,
    direction, followed_through, lag_bars).

    Group by `direction` (values +1 = bullish flip, -1 = bearish flip).
    Returns a DataFrame indexed by direction (only directions actually
    present) with columns:
        n_events        : int   — count of flip events of this direction
        p_followthrough : float — fraction with followed_through == True
        median_lag_bars : float — median of lag_bars among rows where
                          followed_through == True; NaN if none followed
                          through

    If events_df is empty, return an empty DataFrame with these 3 columns
    and an empty index (named "direction").
    """
    out_cols = ["n_events", "p_followthrough", "median_lag_bars"]

    if events_df.empty:
        result = pd.DataFrame(columns=out_cols)
        result.index.name = "direction"
        result["n_events"] = result["n_events"].astype(int)
        result["p_followthrough"] = result["p_followthrough"].astype(float)
        result["median_lag_bars"] = result["median_lag_bars"].astype(float)
        return result

    records = {}
    for direction, group in events_df.groupby("direction"):
        n_events = int(len(group))
        p_followthrough = float(group["followed_through"].mean())

        followed = group.loc[group["followed_through"], "lag_bars"]
        if len(followed) > 0:
            median_lag_bars = float(followed.median())
        else:
            median_lag_bars = float("nan")

        records[int(direction)] = {
            "n_events": n_events,
            "p_followthrough": p_followthrough,
            "median_lag_bars": median_lag_bars,
        }

    result = pd.DataFrame.from_dict(records, orient="index", columns=out_cols)
    result.index.name = "direction"
    result["n_events"] = result["n_events"].astype(int)
    result["p_followthrough"] = result["p_followthrough"].astype(float)
    result["median_lag_bars"] = result["median_lag_bars"].astype(float)
    return result
