from __future__ import annotations

import numpy as np
import pandas as pd

from src.phase5_mining.analysis_5a import (
    FLAG_NAMES,
    FLAG_COLUMNS,
    enumerate_rules,
    evaluate_rules,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_COUNT_MEAN = 5    # mirrors analysis_5a -- minimum n for mean_ret to be non-NaN
_FIRST_HOUR_BARS = 12  # 60 minutes / 5-minute bars (bar_in_session is 1-indexed)


# ---------------------------------------------------------------------------
# T6.1: Select the K carry-set rules (D9), unmodified, in D9 order
# ---------------------------------------------------------------------------

def select_carry_rules(rule_ids: list[int]) -> pd.DataFrame:
    """
    Return the rows of enumerate_rules() corresponding to rule_ids, in the
    given order, with columns ["rule_id"] + FLAG_NAMES. Flag definitions are
    taken verbatim from enumerate_rules() (5a, unmodified -- T6.1).
    """
    rules = enumerate_rules().set_index("rule_id")
    selected = rules.loc[rule_ids].reset_index()
    return selected[["rule_id"] + FLAG_NAMES]


# ---------------------------------------------------------------------------
# T6.4: In-sample / validation / test comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(
    in_sample_eval: pd.DataFrame,
    val_eval: pd.DataFrame,
    test_eval: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge three evaluate_rules() outputs (same rule_ids, possibly different
    order) on rule_id, in the row order of in_sample_eval. Produces one row
    per rule with rule_id, FLAG_NAMES, and {n,mean_ret,hit_rate,t_stat}
    suffixed _in / _val / _test, plus:

      expected_sign : int  -- sign(mean_ret_in) (+1 / -1 / 0)
      collapsed_val  : bool -- True if sign(mean_ret_val) != expected_sign
                               (NaN mean_ret counts as collapsed)
      collapsed_test : bool -- same, for the test split
    """
    keep = ["rule_id", "n", "mean_ret", "hit_rate", "t_stat"]

    merged = in_sample_eval[["rule_id"] + FLAG_NAMES + keep[1:]].copy()
    merged = merged.rename(columns={c: f"{c}_in" for c in keep[1:]})

    merged = merged.merge(
        val_eval[keep].rename(columns={c: f"{c}_val" for c in keep[1:]}),
        on="rule_id", how="left",
    )
    merged = merged.merge(
        test_eval[keep].rename(columns={c: f"{c}_test" for c in keep[1:]}),
        on="rule_id", how="left",
    )

    merged["expected_sign"] = np.sign(merged["mean_ret_in"]).astype(int)
    merged["collapsed_val"] = np.sign(merged["mean_ret_val"]) != merged["expected_sign"]
    merged["collapsed_test"] = np.sign(merged["mean_ret_test"]) != merged["expected_sign"]

    return merged


# ---------------------------------------------------------------------------
# T6.5: Random-feature baseline -- shuffle one defining flag's column
# ---------------------------------------------------------------------------

def random_feature_baseline(
    df: pd.DataFrame,
    rules_df: pd.DataFrame,
    ret_col: str = "ret_fwd_12",
    hac_lags: int = 12,
    shuffle_flag: str = "A_5m",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Permute the column FLAG_COLUMNS[shuffle_flag] across rows (breaking its
    relationship to ret_col and to the other 7 flags), then re-run
    evaluate_rules on the result. Each rule's mask now uses a random
    "is bullish" series for shuffle_flag while the other 7 flags are
    unchanged. Per DESIGN_DOC 8.4, the expectation is that mean_ret/hit_rate
    move toward the unconditional baseline ("edge -> 0").

    Returns the evaluate_rules() output (one row per rule in rules_df, same
    columns/order as evaluate_rules).
    """
    col = FLAG_COLUMNS[shuffle_flag]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(df))

    shuffled = df.copy()
    shuffled[col] = df[col].to_numpy()[perm]

    return evaluate_rules(shuffled, rules_df, ret_col=ret_col, hac_lags=hac_lags)


# ---------------------------------------------------------------------------
# T6.6: Year-by-year stability
# ---------------------------------------------------------------------------

def year_by_year_table(
    df: pd.DataFrame,
    rules_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    ret_col: str = "ret_fwd_12",
    hac_lags: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run evaluate_rules separately on each calendar year present in
    df["bar_close"]. Returns (year_df, summary_df):

      year_df: columns [rule_id, year, n, mean_ret, hit_rate, t_stat], one
        row per (rule, year).
      summary_df: columns [rule_id, expected_sign, n_years_consistent,
        n_years_total, single_year_driven], one row per rule (in the order
        of rules_df). n_years_consistent counts years where n >=
        _MIN_COUNT_MEAN and sign(mean_ret) == expected_sign (taken from
        comparison_df). single_year_driven = (n_years_consistent <= 1).
    """
    years = sorted(df["bar_close"].dt.year.unique())

    year_frames = []
    for year in years:
        year_df = df[df["bar_close"].dt.year == year]
        ev = evaluate_rules(year_df, rules_df, ret_col=ret_col, hac_lags=hac_lags)
        ev = ev[["rule_id", "n", "mean_ret", "hit_rate", "t_stat"]].copy()
        ev["year"] = year
        year_frames.append(ev)

    long_df = pd.concat(year_frames, ignore_index=True)
    long_df = long_df[["rule_id", "year", "n", "mean_ret", "hit_rate", "t_stat"]]

    expected = comparison_df.set_index("rule_id")["expected_sign"]

    summary_records = []
    for rule_id in rules_df["rule_id"]:
        rule_id = int(rule_id)
        group = long_df[long_df["rule_id"] == rule_id]
        exp_sign = int(expected.loc[rule_id])
        consistent = (
            (np.sign(group["mean_ret"]) == exp_sign) & (group["n"] >= _MIN_COUNT_MEAN)
        ).sum()
        summary_records.append({
            "rule_id": rule_id,
            "expected_sign": exp_sign,
            "n_years_consistent": int(consistent),
            "n_years_total": len(group),
            "single_year_driven": bool(consistent <= 1),
        })

    summary_df = pd.DataFrame(summary_records)
    return long_df, summary_df


# ---------------------------------------------------------------------------
# T6.7: Time-of-day baseline
# ---------------------------------------------------------------------------

def time_of_day_baseline(
    df: pd.DataFrame,
    rules_df: pd.DataFrame,
    ret_col: str = "ret_fwd_12",
) -> pd.DataFrame:
    """
    For each rule, compare its mean_ret to a "matched" time-of-day baseline:
    the unconditional per-bar_in_session mean(ret_col), averaged using the
    rule's own bar_in_session firing distribution as weights. If
    rule_mean_ret ~= tod_matched_mean_ret, the rule's edge is largely
    explained by *when in the session* it fires rather than by the
    indicator condition itself.

    Also reports pct_first_hour_fired / pct_first_hour_baseline (fraction of
    rows with bar_in_session <= _FIRST_HOUR_BARS).

    Returns columns [rule_id, n, rule_mean_ret, tod_matched_mean_ret,
    excess_over_tod, pct_first_hour_fired, pct_first_hour_baseline], one row
    per rule in rules_df.
    """
    valid = df[df[ret_col].notna()]
    baseline_by_bar = valid.groupby("bar_in_session")[ret_col].mean()
    pct_first_hour_baseline = float((valid["bar_in_session"] <= _FIRST_HOUR_BARS).mean())

    bull_series = {flag: (df[col] > 0).fillna(False) for flag, col in FLAG_COLUMNS.items()}

    records = []
    for _, rule in rules_df.iterrows():
        mask = pd.Series(True, index=df.index)
        for flag in FLAG_NAMES:
            mask &= bull_series[flag] == bool(rule[flag])

        fired = df.loc[mask & df[ret_col].notna()]
        n = len(fired)

        if n == 0:
            records.append({
                "rule_id": int(rule["rule_id"]),
                "n": 0,
                "rule_mean_ret": float("nan"),
                "tod_matched_mean_ret": float("nan"),
                "excess_over_tod": float("nan"),
                "pct_first_hour_fired": float("nan"),
                "pct_first_hour_baseline": pct_first_hour_baseline,
            })
            continue

        rule_mean_ret = float(fired[ret_col].mean())
        weights = fired["bar_in_session"].value_counts(normalize=True)
        tod_matched = float((weights * baseline_by_bar.reindex(weights.index)).sum())

        records.append({
            "rule_id": int(rule["rule_id"]),
            "n": n,
            "rule_mean_ret": rule_mean_ret,
            "tod_matched_mean_ret": tod_matched,
            "excess_over_tod": rule_mean_ret - tod_matched,
            "pct_first_hour_fired": float((fired["bar_in_session"] <= _FIRST_HOUR_BARS).mean()),
            "pct_first_hour_baseline": pct_first_hour_baseline,
        })

    return pd.DataFrame(records)
