from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_COUNT_MEAN = 5    # minimum rows to compute mean_ret / hit_rate
_MIN_COUNT_HAC = 20    # minimum rows to compute HAC t-stat

_PAIR_SUFFIXES = ["", "_15m", "_1h", "_1d"]


# ---------------------------------------------------------------------------
# Internal helper: HAC t-stat (mirrors analysis_3a / analysis_3c._hac_stats)
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
# Function 1: compute_confluence_score_bull
# ---------------------------------------------------------------------------

def compute_confluence_score_bull(df: pd.DataFrame) -> pd.Series:
    """
    score_bull = sum over the 8 (indicator, TF) pairs of 1{indicator bullish}, where
    bullish(A, suffix) := df[f"A_hist{suffix}"] > 0
    bullish(B, suffix) := df[f"B_roc{suffix}"]  > 0
    for suffix in _PAIR_SUFFIXES.

    A pair where the underlying column value is NaN contributes 0 (treated as
    "not bullish", NOT excluded from the denominator -- the score is always out
    of 8).

    Returns a pandas nullable Int64 Series in [0, 8], same index as df.
    """
    score = pd.Series(0, index=df.index, dtype="Int64")

    for suffix in _PAIR_SUFFIXES:
        a_col = f"A_hist{suffix}"
        b_col = f"B_roc{suffix}"

        a_bull = (df[a_col] > 0).fillna(False)
        b_bull = (df[b_col] > 0).fillna(False)

        score = score + a_bull.astype("Int64") + b_bull.astype("Int64")

    return score


# ---------------------------------------------------------------------------
# Function 2: compute_confluence_score_bull_accel
# ---------------------------------------------------------------------------

def compute_confluence_score_bull_accel(df: pd.DataFrame) -> pd.Series:
    """
    Like compute_confluence_score_bull, but each pair counts only if its STATE
    column equals "bull_accel":
        bullish_accel(A, suffix) := df[f"A_state{suffix}"] == "bull_accel"
        bullish_accel(B, suffix) := df[f"B_state{suffix}"] == "bull_accel"
    NaN/pd.NA state values are simply not equal to "bull_accel" -> contribute 0.
    Returns a pandas nullable Int64 Series in [0, 8], same index as df.
    """
    score = pd.Series(0, index=df.index, dtype="Int64")

    for suffix in _PAIR_SUFFIXES:
        a_col = f"A_state{suffix}"
        b_col = f"B_state{suffix}"

        a_accel = (df[a_col] == "bull_accel").fillna(False)
        b_accel = (df[b_col] == "bull_accel").fillna(False)

        score = score + a_accel.astype("Int64") + b_accel.astype("Int64")

    return score


# ---------------------------------------------------------------------------
# Function 3: score_bucketed_returns
# ---------------------------------------------------------------------------

def score_bucketed_returns(
    df: pd.DataFrame,
    score_col: str,
    horizon_cols: list[str],
    hac_lags: int = 12,
) -> pd.DataFrame:
    """
    Group by the distinct integer values of df[score_col] actually present and,
    for each (score, horizon), compute n / mean_ret / hit_rate / t_stat / p_val_raw.

    Returns a DataFrame with a MultiIndex (score, horizon) [score sorted
    ascending, horizons in the order given by horizon_cols] and columns
    [n, mean_ret, hit_rate, t_stat, p_val_raw].
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
# Function 4: plot_score_monotonicity
# ---------------------------------------------------------------------------

def plot_score_monotonicity(
    score_table: pd.DataFrame,
    horizon_cols: list[str],
    title: str,
    out_path: Path,
) -> None:
    """
    Plot mean_ret vs score, one line per horizon. Save PNG to out_path.
    No-op (no file created, no error) if score_table is empty or has no rows
    for any of horizon_cols.
    """
    if score_table is None or score_table.empty:
        return

    horizons_present = score_table.index.get_level_values("horizon").unique()
    horizons_to_plot = [h for h in horizon_cols if h in horizons_present]
    if not horizons_to_plot:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    any_plotted = False

    for horizon in horizons_to_plot:
        sub = score_table.xs(horizon, level="horizon")
        sub = sub.sort_index()
        if sub.empty:
            continue
        ax.plot(sub.index.values, sub["mean_ret"].values, marker="o", label=horizon)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Confluence score")
    ax.set_ylabel("Mean forward return")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
