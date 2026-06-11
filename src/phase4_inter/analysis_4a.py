from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATE_ORDER = ["bull_accel", "bull_decel", "bear_decel", "bear_accel"]
_MIN_COUNT_MEAN = 5    # minimum rows to compute mean_ret / hit_rate
_MIN_COUNT_HAC = 20    # minimum rows to compute HAC t-stat


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
# Internal helper: compute one (group, horizon) cell
# ---------------------------------------------------------------------------

def _compute_cell(
    cell_df: pd.DataFrame,
    horizon_col: str,
    htf_bar_close_col: str,
    hac_lags: int,
    group: str,
    horizon: str,
) -> dict:
    n_5m = int(len(cell_df))
    n_htf = int(cell_df[htf_bar_close_col].nunique(dropna=True))

    returns = cell_df[horizon_col]

    if n_5m < _MIN_COUNT_MEAN:
        mean_ret = float("nan")
        hit_rate = float("nan")
        t_stat = float("nan")
        p_val_raw = float("nan")
    else:
        mean_ret, _, t_stat, p_val_raw = _hac_stats(returns, hac_lags)
        clean = returns.dropna()
        hit_rate = float((clean > 0).sum()) / float(len(clean)) if len(clean) > 0 else float("nan")

    return {
        "group": group,
        "horizon": horizon,
        "n_5m": n_5m,
        "n_htf": n_htf,
        "mean_ret": mean_ret,
        "hit_rate": hit_rate,
        "t_stat": t_stat,
        "p_val_raw": p_val_raw,
    }


# ---------------------------------------------------------------------------
# Function: stratify_by_htf_state
# ---------------------------------------------------------------------------

def stratify_by_htf_state(
    df: pd.DataFrame,
    setup_mask: pd.Series,
    htf_state_col: str,
    htf_bar_close_col: str,
    horizon_cols: list[str],
    hac_lags: int = 12,
) -> pd.DataFrame:
    """Stratify 5m setup rows by a higher-TF state and compute fwd-return stats.

    Among rows of `df` where `setup_mask` is True (the "5m setup" — e.g.
    A_state == 'bull_accel', or A_macd_zero_cross == 1), stratify by
    `htf_state_col` (one of STATE_ORDER, or NaN) and compute forward-return
    statistics per horizon.

    Returns a DataFrame with a MultiIndex (group, horizon), where `group`
    ranges over STATE_ORDER plus the literal string "unconditional" (5 groups
    total: 4 HTF states + unconditional). The "unconditional" group = ALL
    setup_mask rows regardless of htf_state_col value (including rows where
    htf_state_col is NaN). Rows belonging to the 4 STATE_ORDER groups are only
    those where setup_mask is True AND htf_state_col equals that state (NaN
    htf_state rows are excluded from the 4 per-state groups but DO count
    toward "unconditional").

    Columns:
        n_5m       : int  — number of 5m rows in this (group, horizon) cell
        n_htf      : int  — number of UNIQUE non-null values of
                     df[htf_bar_close_col] among the rows in this cell (the
                     "effective sample size" per plan.md L4 / TASK.md T4.4 —
                     use this, not n_5m, when reporting statistical power for
                     HTF-conditional results)
        mean_ret   : float — mean of df[horizon_col] over the cell; NaN if
                     n_5m < 5
        hit_rate   : float — fraction of non-NaN ret values > 0; NaN if
                     n_5m < 5
        t_stat     : float — Newey-West HAC t-stat of the mean; NaN if
                     n_5m < 20
        p_val_raw  : float — corresponding p-value; NaN if n_5m < 20

    If setup_mask selects zero rows, return an empty DataFrame with the above
    columns and a properly-typed empty MultiIndex (names ["group", "horizon"]).
    """
    columns = ["n_5m", "n_htf", "mean_ret", "hit_rate", "t_stat", "p_val_raw"]

    setup_mask = setup_mask.fillna(False).astype(bool)

    if not bool(setup_mask.any()):
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["group", "horizon"])
        return pd.DataFrame(columns=columns, index=empty_index)

    setup_df = df.loc[setup_mask]

    records = []

    # --- per-HTF-state groups ---------------------------------------------
    for state in STATE_ORDER:
        state_mask = setup_df[htf_state_col] == state
        cell_df = setup_df.loc[state_mask]

        for horizon_col in horizon_cols:
            records.append(
                _compute_cell(
                    cell_df, horizon_col, htf_bar_close_col, hac_lags,
                    group=state, horizon=horizon_col,
                )
            )

    # --- unconditional group -------------------------------------------------
    for horizon_col in horizon_cols:
        records.append(
            _compute_cell(
                setup_df, horizon_col, htf_bar_close_col, hac_lags,
                group="unconditional", horizon=horizon_col,
            )
        )

    result = pd.DataFrame.from_records(records).set_index(["group", "horizon"])
    return result[columns]
