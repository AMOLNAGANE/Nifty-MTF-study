from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.phase5_mining.analysis_5c import (
    split_worked_failed,
    feature_mean_diff,
    propose_filters,
    evaluate_filtered_rule,
)


# ---------------------------------------------------------------------------
# T1: split_worked_failed -- basic counts and rates
# ---------------------------------------------------------------------------

def test_split_worked_failed_basic():
    # 20 rows total: rows 0-11 are "rule" rows (8 positive, 4 negative ret);
    # rows 12-19 are non-rule rows (4 positive, 4 negative ret).
    df = pd.DataFrame({
        "ret_fwd_12": [0.01] * 8 + [-0.01] * 4 + [0.01] * 4 + [-0.01] * 4,
    })
    rule_mask = pd.Series([True] * 12 + [False] * 8)

    worked, failed, stats = split_worked_failed(df, rule_mask, "ret_fwd_12", expected_sign=1)

    assert len(worked) == 8
    assert len(failed) == 4
    assert stats["n_worked"] == 8
    assert stats["n_failed"] == 4
    assert stats["rule_hit_rate"] == pytest.approx(8 / 12)
    # baseline: 12 positives out of 20 total rows
    assert stats["baseline_hit_rate"] == pytest.approx(12 / 20)


# ---------------------------------------------------------------------------
# T2: split_worked_failed -- NaN ret rows excluded from both rule and baseline
# ---------------------------------------------------------------------------

def test_split_worked_failed_excludes_nan_ret():
    df = pd.DataFrame({
        "ret_fwd_12": [0.01, -0.01, np.nan, 0.01, np.nan],
    })
    rule_mask = pd.Series([True] * 5)

    worked, failed, stats = split_worked_failed(df, rule_mask, "ret_fwd_12", expected_sign=1)

    assert stats["n_worked"] == 2
    assert stats["n_failed"] == 1
    # baseline over the 3 non-NaN rows: 2 positive
    assert stats["baseline_hit_rate"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# T3: feature_mean_diff -- hand-computable Cohen's d
# ---------------------------------------------------------------------------

def test_feature_mean_diff_known_values():
    worked = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0]})  # mean=3, var=2.5
    failed = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0, 4.0]})  # mean=2, var=2.5

    result = feature_mean_diff(worked, failed, ["f1"])
    row = result.iloc[0]

    pooled_std = np.sqrt(2.5)  # equal variances -> pooled_var = 2.5
    assert row["mean_worked"] == pytest.approx(3.0)
    assert row["mean_failed"] == pytest.approx(2.0)
    assert row["effect_size"] == pytest.approx((3.0 - 2.0) / pooled_std)
    assert not pd.isna(row["t_stat"])
    assert not pd.isna(row["p_val"])
    assert row["n_worked"] == 5
    assert row["n_failed"] == 5


# ---------------------------------------------------------------------------
# T4: feature_mean_diff -- ranking by |effect_size|
# ---------------------------------------------------------------------------

def test_feature_mean_diff_ranking():
    rng = np.random.default_rng(0)
    n = 30
    worked = pd.DataFrame({
        "f_big_diff": rng.normal(5, 1, n),
        "f_small_diff": rng.normal(0.1, 1, n),
        "f_no_diff": rng.normal(0, 1, n),
    })
    failed = pd.DataFrame({
        "f_big_diff": rng.normal(0, 1, n),
        "f_small_diff": rng.normal(0, 1, n),
        "f_no_diff": rng.normal(0, 1, n),
    })

    result = feature_mean_diff(worked, failed, ["f_big_diff", "f_small_diff", "f_no_diff"])

    assert result.iloc[0]["feature"] == "f_big_diff"
    abs_effects = result["effect_size"].abs()
    assert (abs_effects.diff().dropna() <= 1e-9).all()


# ---------------------------------------------------------------------------
# T5: feature_mean_diff -- insufficient observations -> NaN, sorts last
# ---------------------------------------------------------------------------

def test_feature_mean_diff_insufficient_obs():
    worked = pd.DataFrame({
        "f_good": [1.0, 2.0, 3.0, 4.0, 5.0],
        "f_one_obs": [1.0, np.nan, np.nan, np.nan, np.nan],
    })
    failed = pd.DataFrame({
        "f_good": [0.0, 1.0, 2.0, 3.0, 4.0],
        "f_one_obs": [2.0, 2.0, 2.0, 2.0, 2.0],
    })

    result = feature_mean_diff(worked, failed, ["f_good", "f_one_obs"])
    row = result[result["feature"] == "f_one_obs"].iloc[0]

    assert pd.isna(row["effect_size"])
    assert pd.isna(row["t_stat"])
    assert pd.isna(row["p_val"])
    assert result.iloc[0]["feature"] == "f_good"


# ---------------------------------------------------------------------------
# T6: propose_filters -- threshold and direction
# ---------------------------------------------------------------------------

def test_propose_filters_threshold_and_direction():
    diff_df = pd.DataFrame({
        "feature": ["f1", "f2", "f3", "f4"],
        "mean_worked": [5.0, 1.0, 3.0, 2.0],
        "mean_failed": [2.0, 4.0, 3.1, 2.0],
        "effect_size": [1.2, -0.8, 0.1, np.nan],
        "t_stat": [3.0, -2.0, 0.2, np.nan],
        "p_val": [0.01, 0.05, 0.8, np.nan],
        "n_worked": [10] * 4,
        "n_failed": [10] * 4,
    })

    result = propose_filters(diff_df, threshold=0.5)

    assert set(result["feature"]) == {"f1", "f2"}
    assert list(result["feature"]) == ["f1", "f2"]  # ranked by |effect_size| desc

    f1 = result[result["feature"] == "f1"].iloc[0]
    assert f1["midpoint"] == pytest.approx((5.0 + 2.0) / 2)
    assert f1["direction"] == ">"

    f2 = result[result["feature"] == "f2"].iloc[0]
    assert f2["midpoint"] == pytest.approx((1.0 + 4.0) / 2)
    assert f2["direction"] == "<"


# ---------------------------------------------------------------------------
# T7: evaluate_filtered_rule -- before/after with sufficient counts
# ---------------------------------------------------------------------------

def test_evaluate_filtered_rule_before_after():
    rng = np.random.default_rng(0)
    ret = np.concatenate([
        rng.normal(0.001, 0.005, 25),
        rng.normal(-0.01, 0.005, 5),
    ])
    df = pd.DataFrame({"ret_fwd_12": ret})

    rule_mask = pd.Series([True] * 30)
    filter_mask = pd.Series([True] * 25 + [False] * 5)

    result = evaluate_filtered_rule(df, rule_mask, filter_mask, "ret_fwd_12", hac_lags=12)

    assert list(result.index) == ["before", "after"]
    assert result.loc["before", "n"] == 30
    assert result.loc["after", "n"] == 25
    assert result.loc["before", "frequency"] == pytest.approx(1.0)
    assert result.loc["after", "frequency"] == pytest.approx(25 / 30)
    assert not pd.isna(result.loc["before", "t_stat"])
    assert not pd.isna(result.loc["after", "t_stat"])
    assert result.loc["after", "mean_ret"] > result.loc["before", "mean_ret"]


# ---------------------------------------------------------------------------
# T8: evaluate_filtered_rule -- low-count "after" -> NaN mean/hit/t-stat
# ---------------------------------------------------------------------------

def test_evaluate_filtered_rule_low_count_after():
    rng = np.random.default_rng(1)
    ret = rng.normal(0.001, 0.005, 30)
    df = pd.DataFrame({"ret_fwd_12": ret})

    rule_mask = pd.Series([True] * 30)
    filter_mask = pd.Series([True] * 3 + [False] * 27)

    result = evaluate_filtered_rule(df, rule_mask, filter_mask, "ret_fwd_12", hac_lags=12)

    assert result.loc["after", "n"] == 3
    assert pd.isna(result.loc["after", "mean_ret"])
    assert pd.isna(result.loc["after", "hit_rate"])
    assert pd.isna(result.loc["after", "t_stat"])
