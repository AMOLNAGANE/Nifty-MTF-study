from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.phase5_mining.analysis_5a import FLAG_NAMES, FLAG_COLUMNS, enumerate_rules
from src.phase6_validation.analysis_6 import (
    select_carry_rules,
    build_comparison_table,
    random_feature_baseline,
    year_by_year_table,
    time_of_day_baseline,
)


# ---------------------------------------------------------------------------
# T6.1: select_carry_rules
# ---------------------------------------------------------------------------

def test_select_carry_rules_order_and_flags():
    rule_ids = [151, 121, 0]
    selected = select_carry_rules(rule_ids)

    assert list(selected["rule_id"]) == rule_ids
    assert list(selected.columns) == ["rule_id"] + FLAG_NAMES

    full = enumerate_rules().set_index("rule_id")
    for rid in rule_ids:
        row = selected[selected["rule_id"] == rid].iloc[0]
        for f in FLAG_NAMES:
            assert bool(row[f]) == bool(full.loc[rid, f])


# ---------------------------------------------------------------------------
# T6.4: build_comparison_table
# ---------------------------------------------------------------------------

def _make_eval(rules: pd.DataFrame, n, mean_ret, hit_rate, t_stat) -> pd.DataFrame:
    df = rules.copy()
    df["n"] = n
    df["mean_ret"] = mean_ret
    df["hit_rate"] = hit_rate
    df["t_stat"] = t_stat
    return df


def test_build_comparison_table_collapse_flags():
    rule_ids = [255, 0]
    rules = select_carry_rules(rule_ids)

    in_sample = _make_eval(rules, [100, 100], [0.002, -0.0015], [0.60, 0.42], [3.0, -2.5])
    # both splits keep the same sign as in-sample -> not collapsed
    val = _make_eval(rules, [50, 50], [0.0010, -0.0010], [0.55, 0.45], [1.5, -1.2])
    # both splits flip sign relative to in-sample -> collapsed
    test = _make_eval(rules, [40, 40], [-0.0005, 0.0008], [0.48, 0.53], [-0.8, 1.0])

    table = build_comparison_table(in_sample, val, test)

    assert list(table["rule_id"]) == rule_ids
    assert list(table["expected_sign"]) == [1, -1]
    assert list(table["collapsed_val"]) == [False, False]
    assert list(table["collapsed_test"]) == [True, True]


def test_build_comparison_table_nan_treated_as_collapsed():
    rule_ids = [255]
    rules = select_carry_rules(rule_ids)

    in_sample = _make_eval(rules, [100], [0.002], [0.60], [3.0])
    val = _make_eval(rules, [0], [float("nan")], [float("nan")], [float("nan")])
    test = _make_eval(rules, [100], [0.001], [0.55], [2.0])

    table = build_comparison_table(in_sample, val, test)

    assert bool(table.iloc[0]["collapsed_val"]) is True
    assert bool(table.iloc[0]["collapsed_test"]) is False


# ---------------------------------------------------------------------------
# T6.5: random_feature_baseline
# ---------------------------------------------------------------------------

def test_random_feature_baseline_shrinks_edge_toward_zero():
    np.random.seed(0)
    n_pos, n_neg = 20, 20
    ret_pos = np.random.normal(0.02, 0.005, n_pos)
    ret_neg = np.random.normal(-0.02, 0.005, n_neg)

    data: dict[str, list[float]] = {}
    # all 7 other flags are unconstrained (always "bullish") so they don't
    # affect the rule-255 mask.
    for col in FLAG_COLUMNS.values():
        data[col] = [1.0] * (n_pos + n_neg)
    # A_5m's column is the only thing that distinguishes the two groups, and
    # it's perfectly correlated with ret_fwd_12.
    a5m_col = FLAG_COLUMNS["A_5m"]
    data[a5m_col] = [1.0] * n_pos + [-1.0] * n_neg
    data["ret_fwd_12"] = list(ret_pos) + list(ret_neg)

    df = pd.DataFrame(data)
    rules = select_carry_rules([255])

    from src.phase5_mining.analysis_5a import evaluate_rules
    pre = evaluate_rules(df, rules, ret_col="ret_fwd_12", hac_lags=12).iloc[0]
    post = random_feature_baseline(
        df, rules, ret_col="ret_fwd_12", hac_lags=12, shuffle_flag="A_5m", seed=42
    ).iloc[0]

    # shuffling preserves the count of positive values in the column
    assert post["n"] == pre["n"] == n_pos

    # pre-shuffle: rule 255 picks out the +0.02 group -> strong positive edge
    assert pre["mean_ret"] == pytest.approx(0.02, abs=0.01)
    assert pre["hit_rate"] > 0.9

    # post-shuffle: the "fired" rows are now a random mix of the two groups
    # -> edge collapses toward the unconditional mean (~0)
    assert abs(post["mean_ret"]) < abs(pre["mean_ret"])
    assert abs(post["mean_ret"]) < 0.01


# ---------------------------------------------------------------------------
# T6.6: year_by_year_table
# ---------------------------------------------------------------------------

def test_year_by_year_table_flags_single_year_driven():
    np.random.seed(1)
    rules = select_carry_rules([255, 0])

    years = [2021, 2022, 2023]
    # rule 255 (all flags True): positive edge in 2021/2022, negative in 2023
    group_p_rets = {2021: 0.02, 2022: 0.02, 2023: -0.02}
    # rule 0 (all flags False): negative edge only in 2021, positive otherwise
    group_n_rets = {2021: -0.02, 2022: 0.02, 2023: 0.02}

    rows = []
    for year in years:
        for _ in range(10):
            row = {c: 1.0 for c in FLAG_COLUMNS.values()}
            row["bar_close"] = pd.Timestamp(f"{year}-06-01")
            row["ret_fwd_12"] = group_p_rets[year] + np.random.normal(0, 0.001)
            rows.append(row)
        for _ in range(10):
            row = {c: -1.0 for c in FLAG_COLUMNS.values()}
            row["bar_close"] = pd.Timestamp(f"{year}-06-01")
            row["ret_fwd_12"] = group_n_rets[year] + np.random.normal(0, 0.001)
            rows.append(row)

    df = pd.DataFrame(rows)
    comparison_df = pd.DataFrame({"rule_id": [255, 0], "expected_sign": [1, -1]})

    long_df, summary_df = year_by_year_table(df, rules, comparison_df, ret_col="ret_fwd_12", hac_lags=12)

    assert set(long_df["year"]) == set(years)
    assert len(long_df) == 2 * len(years)

    summary = summary_df.set_index("rule_id")
    assert summary.loc[255, "n_years_consistent"] == 2
    assert summary.loc[255, "n_years_total"] == 3
    assert bool(summary.loc[255, "single_year_driven"]) is False

    assert summary.loc[0, "n_years_consistent"] == 1
    assert bool(summary.loc[0, "single_year_driven"]) is True


# ---------------------------------------------------------------------------
# T6.7: time_of_day_baseline
# ---------------------------------------------------------------------------

def test_time_of_day_baseline_separates_real_edge_from_tod():
    rows = []
    for b in range(1, 11):
        for _ in range(5):  # group F: fires rule 255 (all flags > 0)
            rows.append({
                **{c: 1.0 for c in FLAG_COLUMNS.values()},
                "bar_in_session": b,
                "ret_fwd_12": 0.0001 * b + 0.001,
            })
        for _ in range(5):  # group G: does not fire rule 255
            rows.append({
                **{c: -1.0 for c in FLAG_COLUMNS.values()},
                "bar_in_session": b,
                "ret_fwd_12": 0.0001 * b - 0.001,
            })

    df = pd.DataFrame(rows)
    rules = select_carry_rules([255])

    result = time_of_day_baseline(df, rules, ret_col="ret_fwd_12")
    row = result.iloc[0]

    assert row["n"] == 50
    # rule_mean_ret = mean(0.0001*b + 0.001) over b=1..10 = 0.0001*5.5 + 0.001
    assert row["rule_mean_ret"] == pytest.approx(0.0001 * 5.5 + 0.001)
    # baseline_by_bar[b] = mean over F/G at that b = 0.0001*b (the +/-0.001 cancels)
    # tod_matched = mean over b=1..10 (uniform weights) of 0.0001*b
    assert row["tod_matched_mean_ret"] == pytest.approx(0.0001 * 5.5)
    # the genuine edge beyond time-of-day is the +0.001 offset
    assert row["excess_over_tod"] == pytest.approx(0.001)


def test_time_of_day_baseline_zero_firings():
    df = pd.DataFrame({
        **{c: [-1.0] * 5 for c in FLAG_COLUMNS.values()},
        "bar_in_session": [1, 2, 3, 4, 5],
        "ret_fwd_12": [0.001, 0.002, -0.001, 0.0, 0.0015],
    })
    # rule 255 requires all 8 flags > 0; none of the rows match.
    rules = select_carry_rules([255])

    result = time_of_day_baseline(df, rules, ret_col="ret_fwd_12")
    row = result.iloc[0]

    assert row["n"] == 0
    assert pd.isna(row["rule_mean_ret"])
    assert pd.isna(row["excess_over_tod"])
    assert pd.isna(row["pct_first_hour_fired"])
    assert not pd.isna(row["pct_first_hour_baseline"])
