from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.regression.linear_model import OLS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATE_ORDER = ["bull_accel", "bull_decel", "bear_decel", "bear_accel"]
_MIN_COUNT_MEAN = 5    # minimum rows to compute mean_ret / hit_rate
_MIN_COUNT_HAC = 20    # minimum rows to compute HAC t-stat


# ---------------------------------------------------------------------------
# Internal helper: HAC t-stat (mirrors analysis_3a._hac_stats)
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
# Function 1: build_joint_state_matrix
# ---------------------------------------------------------------------------

def build_joint_state_matrix(
    df: pd.DataFrame,
    state_a_col: str,
    state_b_col: str,
    horizon_col: str,
    hac_lags: int = 12,
) -> pd.DataFrame:
    """Build a 4×4 matrix of (state_a, state_b) cells with return statistics.

    Returns a MultiIndex DataFrame with index = (state_a, state_b) tuples and
    columns = [count, mean_ret, hit_rate, t_stat, p_val_raw].
    All 16 cells are always present; statistics are NaN when count is too small.
    """
    records = []

    for sa in STATE_ORDER:
        for sb in STATE_ORDER:
            mask = (df[state_a_col] == sa) & (df[state_b_col] == sb)
            subset = df.loc[mask, horizon_col]
            count = int(mask.sum())

            if count < _MIN_COUNT_MEAN:
                mean_ret = float("nan")
                hit_rate = float("nan")
                t_stat = float("nan")
                p_val_raw = float("nan")
            else:
                mean_ret, _, t_stat, p_val_raw = _hac_stats(subset, hac_lags)
                clean = subset.dropna()
                hit_rate = float((clean > 0).sum()) / float(len(clean)) if len(clean) > 0 else float("nan")

            records.append({
                "state_a": sa,
                "state_b": sb,
                "count": count,
                "mean_ret": mean_ret,
                "hit_rate": hit_rate,
                "t_stat": t_stat,
                "p_val_raw": p_val_raw,
            })

    result = (
        pd.DataFrame(records)
        .set_index(["state_a", "state_b"])
    )
    return result


# ---------------------------------------------------------------------------
# Function 2: test_joint_state_significance
# ---------------------------------------------------------------------------

def test_joint_state_significance(matrix_df: pd.DataFrame) -> dict:  # noqa: PT004
    """Compute chi-square independence test and return a 5-key summary dict.

    Keys:
        chi2_stat          : Chi-square statistic (independence of A vs B states)
        chi2_pval          : Corresponding p-value
        n_significant_cells: Number of cells with p_val_raw < 0.05
        diagonal_mean_ret  : Mean of mean_ret for diagonal cells (same state)
        offdiag_mean_ret   : Mean of mean_ret for off-diagonal cells
    """
    # ------------------------------------------------------------------ chi2
    # Build 4×4 contingency table from count column
    counts_pivot = (
        matrix_df["count"]
        .unstack(level="state_b")
        .reindex(index=STATE_ORDER, columns=STATE_ORDER)
        .fillna(0)
    )
    chi2_stat, chi2_pval, _, _ = chi2_contingency(counts_pivot.values)

    # ------------------------------------------ n_significant_cells
    p_vals = matrix_df["p_val_raw"]
    valid_p = p_vals.dropna()
    n_significant_cells = int((valid_p < 0.05).sum())

    # ------------------------------------------ diagonal vs off-diagonal
    diagonal_idx = [(s, s) for s in STATE_ORDER]
    offdiag_idx = [
        (sa, sb)
        for sa in STATE_ORDER
        for sb in STATE_ORDER
        if sa != sb
    ]

    diag_vals = matrix_df.loc[diagonal_idx, "mean_ret"].dropna()
    offdiag_vals = matrix_df.loc[offdiag_idx, "mean_ret"].dropna()

    diagonal_mean_ret = float(diag_vals.mean()) if len(diag_vals) > 0 else float("nan")
    offdiag_mean_ret = float(offdiag_vals.mean()) if len(offdiag_vals) > 0 else float("nan")

    return {
        "chi2_stat": float(chi2_stat),
        "chi2_pval": float(chi2_pval),
        "n_significant_cells": n_significant_cells,
        "diagonal_mean_ret": diagonal_mean_ret,
        "offdiag_mean_ret": offdiag_mean_ret,
    }


# Prevent pytest from collecting this function as a test case
test_joint_state_significance.__test__ = False  # type: ignore[attr-defined]
