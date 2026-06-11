from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_COUNT_MEAN = 5    # minimum rows to compute mean_ret / hit_rate
_MIN_COUNT_HAC = 20    # minimum rows to compute HAC t-stat

_PAIR_SUFFIXES = ["", "_15m", "_1h", "_1d"]


# ---------------------------------------------------------------------------
# Internal helper: HAC t-stat (mirrors analysis_3c._hac_stats)
# ---------------------------------------------------------------------------

def _hac_stats(returns: pd.Series, hac_lags: int) -> tuple[float, int, float, float]:
    """Return (mean, count, t_stat, p_val) with Newey-West HAC."""
    clean = returns.dropna()
    count = int(len(clean))
    mean = float(clean.mean()) if count > 0 else float("nan")

    if count < _MIN_COUNT_HAC:
        return mean, count, float("nan"), float("nan")

    y = clean.values.astype(float)
    X = np.ones((len(y), 1))
    model = OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": hac_lags, "use_correction": True},
    )
    t_stat = float(model.tvalues[0])
    p_val = float(model.pvalues[0])
    return mean, count, t_stat, p_val


# ---------------------------------------------------------------------------
# Function 1: compute_confluence_score_bear
# ---------------------------------------------------------------------------

def compute_confluence_score_bear(df: pd.DataFrame) -> pd.Series:
    """
    score_bear = sum over the 8 (indicator, TF) pairs of 1{indicator bearish}, where
    bearish(A, suffix) := df[f"A_hist{suffix}"] < 0
    bearish(B, suffix) := df[f"B_roc{suffix}"]  < 0
    for suffix in _PAIR_SUFFIXES = ["", "_15m", "_1h", "_1d"].

    A pair where the underlying column value is NaN or exactly 0 contributes 0
    (treated as "not bearish"). The score is always out of 8 (never NaN itself) --
    same convention as the sibling compute_confluence_score_bull (NaN -> 0
    contribution, no propagation).

    Returns a pandas nullable Int64 Series ∈ [0, 8], same index as df.
    """
    score = pd.Series(0, index=df.index, dtype="Int64")

    for suffix in _PAIR_SUFFIXES:
        a_col = f"A_hist{suffix}"
        b_col = f"B_roc{suffix}"

        a_bear = (df[a_col] < 0).fillna(False)
        b_bear = (df[b_col] < 0).fillna(False)

        score = score + a_bear.astype("Int64") + b_bear.astype("Int64")

    return score


# ---------------------------------------------------------------------------
# Function 2: score_bucketed_returns
# ---------------------------------------------------------------------------

def score_bucketed_returns(
    df: pd.DataFrame,
    score_col: str,
    horizon_cols: list[str],
    hac_lags: int = 12,
) -> pd.DataFrame:
    """
    Identical contract to the sibling function in analysis_4c.py (duplicated here
    so this module is self-contained -- DO NOT import from analysis_4c).

    df[score_col] is an integer score (values 0..8). Group by the distinct integer
    values ACTUALLY PRESENT in df[score_col] (absent scores are simply omitted from
    the output, not NaN-filled).

    For each (score, horizon), compute via _hac_stats:
        n          : int   -- count of non-NaN ret values
        mean_ret   : float -- NaN if n < _MIN_COUNT_MEAN
        hit_rate   : float -- fraction of non-NaN ret > 0; NaN if n < _MIN_COUNT_MEAN
        t_stat     : float -- NaN if n < _MIN_COUNT_HAC
        p_val_raw  : float -- NaN if n < _MIN_COUNT_HAC

    Returns a DataFrame with MultiIndex (score, horizon) [score ascending, horizons
    in the order of horizon_cols] and columns [n, mean_ret, hit_rate, t_stat, p_val_raw].
    Empty DataFrame with correct columns/named-empty-MultiIndex (["score","horizon"])
    if df is empty or score_col has no non-NaN values.
    """
    columns = ["n", "mean_ret", "hit_rate", "t_stat", "p_val_raw"]
    empty_index = pd.MultiIndex.from_arrays([[], []], names=["score", "horizon"])

    if df.empty or score_col not in df.columns:
        return pd.DataFrame(columns=columns, index=empty_index)

    scores_present = sorted(df[score_col].dropna().unique().tolist())
    if not scores_present:
        return pd.DataFrame(columns=columns, index=empty_index)

    records = []
    index_tuples = []

    for score_val in scores_present:
        mask = df[score_col] == score_val
        for horizon in horizon_cols:
            subset = df.loc[mask, horizon]
            mean_ret, count, t_stat, p_val_raw = _hac_stats(subset, hac_lags)

            if count < _MIN_COUNT_MEAN:
                mean_ret = float("nan")
                hit_rate = float("nan")
            else:
                clean = subset.dropna()
                hit_rate = (
                    float((clean > 0).sum()) / float(len(clean))
                    if len(clean) > 0
                    else float("nan")
                )

            records.append({
                "n": count,
                "mean_ret": mean_ret,
                "hit_rate": hit_rate,
                "t_stat": t_stat,
                "p_val_raw": p_val_raw,
            })
            index_tuples.append((score_val, horizon))

    index = pd.MultiIndex.from_tuples(index_tuples, names=["score", "horizon"])
    result = pd.DataFrame(records, index=index, columns=columns)
    return result


# ---------------------------------------------------------------------------
# Function 3: compare_bull_bear_asymmetry
# ---------------------------------------------------------------------------

def compare_bull_bear_asymmetry(
    bull_table: pd.DataFrame,
    bear_table: pd.DataFrame,
    horizon_cols: list[str],
) -> pd.DataFrame:
    """
    bull_table, bear_table: outputs of score_bucketed_returns (one for score_bull,
    one for score_bear) -- both MultiIndex (score, horizon) with a "mean_ret" column
    (and other columns, ignored here except as noted).

    For each score level S that appears in BOTH bull_table and bear_table (inner
    join on the "score" index level -- typically scores 0..8, but use whatever is
    actually present in both), and for each horizon in horizon_cols present in
    both tables, build one output row with:
        score          : int   -- the confluence-score level S (same scale 0..8 used
                                  for both bull and bear; "equivalent scores" per T4.14)
        horizon        : str
        bull_mean_ret  : float -- bull_table.loc[(S,horizon), "mean_ret"]
        bear_mean_ret  : float -- bear_table.loc[(S,horizon), "mean_ret"]
        bull_n         : int   -- bull_table.loc[(S,horizon), "n"]
        bear_n         : int   -- bear_table.loc[(S,horizon), "n"]
        abs_diff       : float -- abs(bull_mean_ret) - abs(bear_mean_ret)
                                  (positive => bullish edge at this score level is
                                  LARGER in magnitude than the bearish edge at the
                                  same score level; negative => bearish edge dominates)

    Returns a DataFrame with MultiIndex (score, horizon) [score ascending, then
    horizon in horizon_cols order] and columns
    [bull_mean_ret, bear_mean_ret, bull_n, bear_n, abs_diff].

    If there is no overlap between bull_table and bear_table scores/horizons,
    return an empty DataFrame with these columns and a properly-named empty
    MultiIndex (["score","horizon"]).
    NaN mean_ret values propagate naturally into abs_diff (NaN - x = NaN).
    """
    columns = ["bull_mean_ret", "bear_mean_ret", "bull_n", "bear_n", "abs_diff"]
    empty_index = pd.MultiIndex.from_arrays([[], []], names=["score", "horizon"])

    if bull_table.empty or bear_table.empty:
        return pd.DataFrame(columns=columns, index=empty_index)

    bull_scores = set(bull_table.index.get_level_values("score").unique())
    bear_scores = set(bear_table.index.get_level_values("score").unique())
    common_scores = sorted(bull_scores & bear_scores)

    bull_horizons = set(bull_table.index.get_level_values("horizon").unique())
    bear_horizons = set(bear_table.index.get_level_values("horizon").unique())
    common_horizons = [h for h in horizon_cols if h in bull_horizons and h in bear_horizons]

    if not common_scores or not common_horizons:
        return pd.DataFrame(columns=columns, index=empty_index)

    records = []
    index_tuples = []

    for score_val in common_scores:
        for horizon in common_horizons:
            bull_row = bull_table.loc[(score_val, horizon)]
            bear_row = bear_table.loc[(score_val, horizon)]

            bull_mean_ret = bull_row["mean_ret"]
            bear_mean_ret = bear_row["mean_ret"]
            bull_n = bull_row["n"]
            bear_n = bear_row["n"]
            abs_diff = abs(bull_mean_ret) - abs(bear_mean_ret)

            records.append({
                "bull_mean_ret": bull_mean_ret,
                "bear_mean_ret": bear_mean_ret,
                "bull_n": bull_n,
                "bear_n": bear_n,
                "abs_diff": abs_diff,
            })
            index_tuples.append((score_val, horizon))

    index = pd.MultiIndex.from_tuples(index_tuples, names=["score", "horizon"])
    result = pd.DataFrame(records, index=index, columns=columns)
    return result
