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
_TF_LABELS = {"": "5m", "_15m": "15m", "_1h": "1h", "_1d": "1d"}

# The 8 (indicator, TF) bull flags, in canonical order:
# A_5m, B_5m, A_15m, B_15m, A_1h, B_1h, A_1d, B_1d.
# flag i (0-indexed) corresponds to bit i of rule_id (bit 0 = A_5m, bit 7 = B_1d).
FLAG_NAMES: list[str] = []
FLAG_COLUMNS: dict[str, str] = {}
for _suffix in _PAIR_SUFFIXES:
    _tf = _TF_LABELS[_suffix]
    for _ind, _col_prefix in (("A", "A_hist"), ("B", "B_roc")):
        _flag = f"{_ind}_{_tf}"
        FLAG_NAMES.append(_flag)
        FLAG_COLUMNS[_flag] = f"{_col_prefix}{_suffix}"


# ---------------------------------------------------------------------------
# Internal helper: HAC t-stat (mirrors analysis_3a / analysis_4c._hac_stats)
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
# T5.1: Enumerate 2^8 = 256 boolean rules
# ---------------------------------------------------------------------------

def enumerate_rules() -> pd.DataFrame:
    """
    Enumerate all 2^8 = 256 boolean combinations of the 8 (indicator, TF) bull
    flags in FLAG_NAMES.

    Returns a 256-row DataFrame with columns ["rule_id"] + FLAG_NAMES, where
    rule_id in [0, 255] and FLAG_NAMES[i] == bool((rule_id >> i) & 1).
    """
    records = []
    for rule_id in range(256):
        row: dict[str, object] = {"rule_id": rule_id}
        for i, flag in enumerate(FLAG_NAMES):
            row[flag] = bool((rule_id >> i) & 1)
        records.append(row)
    return pd.DataFrame(records, columns=["rule_id"] + FLAG_NAMES)


# ---------------------------------------------------------------------------
# T5.2: Evaluate each rule against the data
# ---------------------------------------------------------------------------

def evaluate_rules(
    df: pd.DataFrame,
    rules_df: pd.DataFrame,
    ret_col: str = "ret_fwd_12",
    hac_lags: int = 12,
) -> pd.DataFrame:
    """
    For each rule (row of rules_df), build a boolean mask over df that requires,
    for every flag in FLAG_NAMES, (df[FLAG_COLUMNS[flag]] > 0) == rule[flag]
    (NaN underlying values -> False, i.e. "not bullish" -- same convention as
    compute_confluence_score_bull in analysis_4c.py).

    For each rule's mask, compute via _hac_stats on df.loc[mask, ret_col]:
        n          : int   -- count of non-NaN ret values
        frequency  : float -- n / len(df)
        mean_ret   : float -- NaN if n < _MIN_COUNT_MEAN
        hit_rate   : float -- fraction of non-NaN ret > 0; NaN if n < _MIN_COUNT_MEAN
        t_stat     : float -- NaN if n < _MIN_COUNT_HAC
        p_val_raw  : float -- NaN if n < _MIN_COUNT_HAC
        p_val_bonf : float -- min(1, p_val_raw * len(rules_df)); NaN propagates

    Returns a DataFrame with columns
    ["rule_id"] + FLAG_NAMES + ["n", "frequency", "mean_ret", "hit_rate",
    "t_stat", "p_val_raw", "p_val_bonf"], one row per rule, in the same order
    as rules_df.
    """
    n_total = len(df)
    n_rules = len(rules_df)

    # Precompute the "is bullish" boolean series for each flag once.
    bull_series: dict[str, pd.Series] = {}
    for flag, col in FLAG_COLUMNS.items():
        bull_series[flag] = (df[col] > 0).fillna(False)

    records = []
    for _, rule in rules_df.iterrows():
        mask = pd.Series(True, index=df.index)
        for flag in FLAG_NAMES:
            mask &= bull_series[flag] == bool(rule[flag])

        subset = df.loc[mask, ret_col]
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

        record: dict[str, object] = {flag: bool(rule[flag]) for flag in FLAG_NAMES}
        record["rule_id"] = int(rule["rule_id"])
        record["n"] = count
        record["frequency"] = count / n_total if n_total > 0 else float("nan")
        record["mean_ret"] = mean_ret
        record["hit_rate"] = hit_rate
        record["t_stat"] = t_stat
        record["p_val_raw"] = p_val_raw
        records.append(record)

    result = pd.DataFrame(records)
    result["p_val_bonf"] = (result["p_val_raw"] * n_rules).clip(upper=1.0)

    cols = (
        ["rule_id"]
        + FLAG_NAMES
        + ["n", "frequency", "mean_ret", "hit_rate", "t_stat", "p_val_raw", "p_val_bonf"]
    )
    return result[cols]


# ---------------------------------------------------------------------------
# T5.3: Rank rules by composite score
# ---------------------------------------------------------------------------

def rank_rules(
    rules_df: pd.DataFrame, n_top: int = 20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute composite = frequency * |mean_ret| * (hit_rate - 0.5) for each rule
    (rows with NaN composite -- i.e. n < _MIN_COUNT_MEAN -- are dropped).

    Returns (top20, bottom20):
      - top20: the n_top rows with the largest composite (descending), reset index.
      - bottom20: the n_top rows with the smallest (most negative) composite
        (ascending), reset index.

    Both output DataFrames have all columns of rules_df plus "composite".
    """
    df = rules_df.copy()
    df["composite"] = df["frequency"] * df["mean_ret"].abs() * (df["hit_rate"] - 0.5)
    valid = df.dropna(subset=["composite"])

    top = valid.sort_values("composite", ascending=False).head(n_top).reset_index(drop=True)
    bottom = valid.sort_values("composite", ascending=True).head(n_top).reset_index(drop=True)
    return top, bottom


# ---------------------------------------------------------------------------
# T5.4: Remove duplicate / trivially-related rules
# ---------------------------------------------------------------------------

def _hamming_distance(row_a: pd.Series, row_b: pd.Series) -> int:
    return sum(int(bool(row_a[f])) != int(bool(row_b[f])) for f in FLAG_NAMES)


def dedupe_rules(ranked_df: pd.DataFrame, min_hamming: int = 2) -> pd.DataFrame:
    """
    Greedily walk ranked_df in order (it is assumed pre-sorted by importance,
    e.g. the output of rank_rules) and keep a rule only if its 8-flag vector
    differs by >= min_hamming bits from every rule already kept ("trivially
    related" rules -- those differing by a single flag -- are dropped).

    Returns a DataFrame with the same columns as ranked_df, <= len(ranked_df)
    rows, reset index.
    """
    survivors: list[pd.Series] = []
    for _, row in ranked_df.iterrows():
        if all(_hamming_distance(row, kept) >= min_hamming for kept in survivors):
            survivors.append(row)

    if not survivors:
        return ranked_df.iloc[0:0].reset_index(drop=True)

    return pd.DataFrame(survivors).reset_index(drop=True)
