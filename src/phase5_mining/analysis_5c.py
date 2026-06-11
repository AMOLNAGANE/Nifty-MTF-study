from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.regression.linear_model import OLS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_COUNT_MEAN = 5    # minimum rows to compute mean_ret / hit_rate
_MIN_COUNT_HAC = 20    # minimum rows to compute HAC t-stat


# ---------------------------------------------------------------------------
# Internal helper: HAC t-stat (mirrors analysis_4c / analysis_5a._hac_stats)
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
# T5.11: Split rule-matching rows into "worked" / "failed"
# ---------------------------------------------------------------------------

def split_worked_failed(
    df: pd.DataFrame,
    rule_mask: pd.Series,
    ret_col: str,
    expected_sign: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Among rows where rule_mask is True and df[ret_col] is not NaN, split into:
      - worked: sign(df[ret_col]) == expected_sign
      - failed: sign(df[ret_col]) != expected_sign

    baseline_hit_rate is the unconditional fraction of ALL rows in df (not just
    rule_mask) with sign(df[ret_col]) == expected_sign, among rows with
    non-NaN ret_col.

    Returns (worked_df, failed_df, stats), where stats has keys
    n_worked, n_failed, rule_hit_rate, baseline_hit_rate.
    """
    valid_ret = df[ret_col].notna()
    rule_rows = df.loc[rule_mask & valid_ret]

    sign = np.sign(rule_rows[ret_col])
    worked_mask = sign == expected_sign
    worked_df = rule_rows[worked_mask]
    failed_df = rule_rows[~worked_mask]

    n_worked = len(worked_df)
    n_failed = len(failed_df)
    n_rule = n_worked + n_failed

    if valid_ret.any():
        baseline_sign = np.sign(df.loc[valid_ret, ret_col])
        baseline_hit_rate = float((baseline_sign == expected_sign).mean())
    else:
        baseline_hit_rate = float("nan")

    stats = {
        "n_worked": n_worked,
        "n_failed": n_failed,
        "rule_hit_rate": n_worked / n_rule if n_rule > 0 else float("nan"),
        "baseline_hit_rate": baseline_hit_rate,
    }
    return worked_df, failed_df, stats


# ---------------------------------------------------------------------------
# T5.12: Per-feature mean difference between "worked" and "failed"
# ---------------------------------------------------------------------------

def feature_mean_diff(
    worked_df: pd.DataFrame,
    failed_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    For each feature in feature_cols, compute:
      mean_worked, mean_failed
      effect_size = Cohen's d = (mean_worked - mean_failed) / pooled_std
      t_stat, p_val: Welch's t-test (unequal variance) of worked vs failed

    effect_size/t_stat/p_val are NaN if either group has < 2 non-NaN
    observations, or pooled_std == 0.

    Returns columns [feature, mean_worked, mean_failed, effect_size, t_stat,
    p_val, n_worked, n_failed], ranked by |effect_size| descending (NaN sorts
    last).
    """
    n_worked = len(worked_df)
    n_failed = len(failed_df)

    records = []
    for feat in feature_cols:
        w = worked_df[feat].dropna()
        f = failed_df[feat].dropna()

        mean_w = float(w.mean()) if len(w) > 0 else float("nan")
        mean_f = float(f.mean()) if len(f) > 0 else float("nan")

        effect_size = float("nan")
        t_stat = float("nan")
        p_val = float("nan")

        if len(w) >= 2 and len(f) >= 2:
            var_w = float(w.var(ddof=1))
            var_f = float(f.var(ddof=1))
            pooled_var = ((len(w) - 1) * var_w + (len(f) - 1) * var_f) / (len(w) + len(f) - 2)
            pooled_std = np.sqrt(pooled_var)
            if pooled_std > 0:
                effect_size = (mean_w - mean_f) / pooled_std

            t_res = ttest_ind(w, f, equal_var=False)
            t_stat = float(t_res.statistic)
            p_val = float(t_res.pvalue)

        records.append({
            "feature": feat,
            "mean_worked": mean_w,
            "mean_failed": mean_f,
            "effect_size": effect_size,
            "t_stat": t_stat,
            "p_val": p_val,
            "n_worked": n_worked,
            "n_failed": n_failed,
        })

    result = pd.DataFrame(records)
    order = result["effect_size"].abs().fillna(-1).sort_values(ascending=False).index
    return result.loc[order].reset_index(drop=True)


# ---------------------------------------------------------------------------
# T5.13: Propose filters from large feature differences
# ---------------------------------------------------------------------------

def propose_filters(diff_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Select features with |effect_size| > threshold (NaN effect_size excluded)
    and propose a filter direction for each:
      midpoint = (mean_worked + mean_failed) / 2
      direction = ">" if mean_worked > mean_failed else "<"
    i.e. the filter keeps rows that look more like the "worked" distribution.

    Returns columns [feature, effect_size, mean_worked, mean_failed, midpoint,
    direction], ranked by |effect_size| descending.
    """
    valid = diff_df.dropna(subset=["effect_size"])
    selected = valid[valid["effect_size"].abs() > threshold].copy()

    selected["midpoint"] = (selected["mean_worked"] + selected["mean_failed"]) / 2.0
    selected["direction"] = np.where(selected["mean_worked"] > selected["mean_failed"], ">", "<")

    cols = ["feature", "effect_size", "mean_worked", "mean_failed", "midpoint", "direction"]
    selected = selected[cols]
    order = selected["effect_size"].abs().sort_values(ascending=False).index
    return selected.loc[order].reset_index(drop=True)


# ---------------------------------------------------------------------------
# T5.14: Before/after stats for a rule with an additional filter applied
# ---------------------------------------------------------------------------

def evaluate_filtered_rule(
    df: pd.DataFrame,
    rule_mask: pd.Series,
    filter_mask: pd.Series,
    ret_col: str,
    hac_lags: int = 12,
) -> pd.DataFrame:
    """
    Compute before/after stats for rule_mask vs (rule_mask & filter_mask):
    n, frequency (= n / rule_mask.sum()), mean_ret, hit_rate, t_stat,
    p_val_raw, via _hac_stats.

    Returns a 2-row DataFrame indexed ["before", "after"] with columns
    [n, frequency, mean_ret, hit_rate, t_stat, p_val_raw].
    """
    n_rule = int(rule_mask.sum())

    def _row(mask: pd.Series) -> dict:
        subset = df.loc[mask, ret_col]
        mean_ret, count, t_stat, p_val_raw = _hac_stats(subset, hac_lags)

        if count < _MIN_COUNT_MEAN:
            mean_ret = float("nan")
            hit_rate = float("nan")
        else:
            clean = subset.dropna()
            hit_rate = float((clean > 0).sum()) / float(len(clean)) if len(clean) > 0 else float("nan")

        return {
            "n": count,
            "frequency": count / n_rule if n_rule > 0 else float("nan"),
            "mean_ret": mean_ret,
            "hit_rate": hit_rate,
            "t_stat": t_stat,
            "p_val_raw": p_val_raw,
        }

    before = _row(rule_mask)
    after = _row(rule_mask & filter_mask)
    return pd.DataFrame([before, after], index=["before", "after"])
