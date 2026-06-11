from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.phase5_mining.analysis_5a import (
    FLAG_NAMES,
    FLAG_COLUMNS,
    enumerate_rules,
    evaluate_rules,
    rank_rules,
    dedupe_rules,
)


# ---------------------------------------------------------------------------
# T1: FLAG_NAMES / FLAG_COLUMNS layout
# ---------------------------------------------------------------------------

def test_flag_names_order_and_columns():
    assert FLAG_NAMES == [
        "A_5m", "B_5m",
        "A_15m", "B_15m",
        "A_1h", "B_1h",
        "A_1d", "B_1d",
    ]
    assert FLAG_COLUMNS["A_5m"] == "A_hist"
    assert FLAG_COLUMNS["B_5m"] == "B_roc"
    assert FLAG_COLUMNS["A_15m"] == "A_hist_15m"
    assert FLAG_COLUMNS["B_15m"] == "B_roc_15m"
    assert FLAG_COLUMNS["A_1h"] == "A_hist_1h"
    assert FLAG_COLUMNS["B_1h"] == "B_roc_1h"
    assert FLAG_COLUMNS["A_1d"] == "A_hist_1d"
    assert FLAG_COLUMNS["B_1d"] == "B_roc_1d"


# ---------------------------------------------------------------------------
# T2: enumerate_rules -- shape and bit mapping
# ---------------------------------------------------------------------------

def test_enumerate_rules_shape_and_range():
    rules = enumerate_rules()
    assert len(rules) == 256
    assert list(rules.columns) == ["rule_id"] + FLAG_NAMES
    assert set(rules["rule_id"]) == set(range(256))


def test_enumerate_rules_bit_mapping():
    rules = enumerate_rules().set_index("rule_id")

    # rule 0 -> all flags False
    row0 = rules.loc[0]
    assert not any(bool(row0[f]) for f in FLAG_NAMES)

    # rule 255 -> all flags True
    row255 = rules.loc[255]
    assert all(bool(row255[f]) for f in FLAG_NAMES)

    # rule 1 -> bit 0 (FLAG_NAMES[0] = A_5m) set, rest False
    row1 = rules.loc[1]
    assert bool(row1[FLAG_NAMES[0]]) is True
    for f in FLAG_NAMES[1:]:
        assert bool(row1[f]) is False

    # rule 2 -> bit 1 (FLAG_NAMES[1] = B_5m) set, rest False
    row2 = rules.loc[2]
    assert bool(row2[FLAG_NAMES[1]]) is True
    for f in [FLAG_NAMES[0]] + FLAG_NAMES[2:]:
        assert bool(row2[f]) is False


# ---------------------------------------------------------------------------
# Synthetic data for evaluate_rules
# ---------------------------------------------------------------------------

def _make_eval_df() -> pd.DataFrame:
    """30 rows: 25 'all bullish' (matches rule_id=255), 5 'all bearish'
    (matches rule_id=0, with one NaN that should still map to 'not bullish')."""
    n_pos, n_neg = 25, 5
    n = n_pos + n_neg

    data: dict[str, list[float]] = {}
    for col in FLAG_COLUMNS.values():
        data[col] = [1.0] * n_pos + [-1.0] * n_neg

    # NaN underlying value -> flag should still resolve to False (not bullish)
    data["A_hist_1d"][-1] = np.nan

    np.random.seed(0)
    ret = np.concatenate([
        np.random.normal(0.001, 0.005, n_pos),
        np.random.normal(-0.002, 0.005, n_neg),
    ])
    data["ret_fwd_12"] = list(ret)

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# T3: evaluate_rules -- output shape and columns
# ---------------------------------------------------------------------------

def test_evaluate_rules_shape_and_columns():
    df = _make_eval_df()
    rules = enumerate_rules()
    result = evaluate_rules(df, rules, ret_col="ret_fwd_12", hac_lags=12)

    assert len(result) == 256
    expected_cols = (
        ["rule_id"] + FLAG_NAMES
        + ["n", "frequency", "mean_ret", "hit_rate", "t_stat", "p_val_raw", "p_val_bonf"]
    )
    assert list(result.columns) == expected_cols


# ---------------------------------------------------------------------------
# T4: evaluate_rules -- "all bullish" rule (n=25, >= HAC threshold)
# ---------------------------------------------------------------------------

def test_evaluate_rules_all_true_rule():
    df = _make_eval_df()
    rules = enumerate_rules()
    result = evaluate_rules(df, rules, ret_col="ret_fwd_12", hac_lags=12)

    row = result[result["rule_id"] == 255].iloc[0]
    assert row["n"] == 25
    assert row["frequency"] == pytest.approx(25 / 30)
    assert not pd.isna(row["mean_ret"])
    assert not pd.isna(row["hit_rate"])
    assert not pd.isna(row["t_stat"])  # n >= _MIN_COUNT_HAC
    assert not pd.isna(row["p_val_raw"])
    assert not pd.isna(row["p_val_bonf"])
    assert 0.0 <= row["p_val_bonf"] <= 1.0


# ---------------------------------------------------------------------------
# T5: evaluate_rules -- "all bearish" rule (n=5, mean/hit populated, no t-stat)
# ---------------------------------------------------------------------------

def test_evaluate_rules_all_false_rule():
    df = _make_eval_df()
    rules = enumerate_rules()
    result = evaluate_rules(df, rules, ret_col="ret_fwd_12", hac_lags=12)

    row = result[result["rule_id"] == 0].iloc[0]
    assert row["n"] == 5
    assert row["frequency"] == pytest.approx(5 / 30)
    assert not pd.isna(row["mean_ret"])  # n >= _MIN_COUNT_MEAN
    assert not pd.isna(row["hit_rate"])
    assert pd.isna(row["t_stat"])  # n < _MIN_COUNT_HAC
    assert pd.isna(row["p_val_raw"])
    assert pd.isna(row["p_val_bonf"])  # NaN propagates through clip


# ---------------------------------------------------------------------------
# T6: evaluate_rules -- a rule matched by zero rows
# ---------------------------------------------------------------------------

def test_evaluate_rules_zero_count_rule():
    df = _make_eval_df()
    rules = enumerate_rules()
    result = evaluate_rules(df, rules, ret_col="ret_fwd_12", hac_lags=12)

    # rule 128 = only B_1d (FLAG_NAMES[7]) bullish, all others not -- no row matches
    row = result[result["rule_id"] == 128].iloc[0]
    assert row["n"] == 0
    assert row["frequency"] == 0.0
    assert pd.isna(row["mean_ret"])
    assert pd.isna(row["hit_rate"])
    assert pd.isna(row["t_stat"])
    assert pd.isna(row["p_val_raw"])
    assert pd.isna(row["p_val_bonf"])


# ---------------------------------------------------------------------------
# T7: evaluate_rules -- hit_rate always within [0, 1]
# ---------------------------------------------------------------------------

def test_evaluate_rules_hit_rate_in_range():
    df = _make_eval_df()
    rules = enumerate_rules()
    result = evaluate_rules(df, rules, ret_col="ret_fwd_12", hac_lags=12)

    valid = result["hit_rate"].dropna()
    assert (valid >= 0).all() and (valid <= 1).all()


# ---------------------------------------------------------------------------
# T8: rank_rules -- composite formula and top/bottom selection
# ---------------------------------------------------------------------------

def test_rank_rules_composite_and_selection():
    df = pd.DataFrame({
        "rule_id": [1, 2, 3, 4, 5],
        "frequency": [0.5, 0.5, 0.1, 0.1, 0.5],
        "mean_ret": [0.01, -0.01, 0.02, -0.02, np.nan],
        "hit_rate": [0.6, 0.4, 0.6, 0.4, 0.6],
    })

    top, bottom = rank_rules(df, n_top=2)

    # composite = frequency * |mean_ret| * (hit_rate - 0.5)
    # rule1: 0.5*0.01*0.1   =  0.0005
    # rule2: 0.5*0.01*-0.1  = -0.0005
    # rule3: 0.1*0.02*0.1   =  0.0002
    # rule4: 0.1*0.02*-0.1  = -0.0002
    # rule5: NaN (mean_ret) -> dropped
    assert "composite" in top.columns
    assert "composite" in bottom.columns

    assert list(top["rule_id"]) == [1, 3]
    assert list(bottom["rule_id"]) == [2, 4]

    assert 5 not in top["rule_id"].values
    assert 5 not in bottom["rule_id"].values

    assert top.iloc[0]["composite"] == pytest.approx(0.0005)
    assert bottom.iloc[0]["composite"] == pytest.approx(-0.0005)


def test_rank_rules_n_top_caps_length():
    n = 10
    df = pd.DataFrame({
        "rule_id": list(range(n)),
        "frequency": np.linspace(0.1, 1.0, n),
        "mean_ret": np.linspace(-0.01, 0.01, n),
        "hit_rate": np.linspace(0.4, 0.6, n),
    })
    top, bottom = rank_rules(df, n_top=3)
    assert len(top) == 3
    assert len(bottom) == 3


# ---------------------------------------------------------------------------
# T9: dedupe_rules -- Hamming-distance filtering
# ---------------------------------------------------------------------------

def test_dedupe_rules_drops_close_neighbors():
    base = {f: False for f in FLAG_NAMES}

    row_a = dict(base)
    row_a["rule_id"] = 0

    # differs from row_a in exactly 1 flag -> Hamming distance 1 < 2 -> dropped
    row_b = dict(base)
    row_b[FLAG_NAMES[0]] = True
    row_b["rule_id"] = 1

    # differs from row_a in exactly 2 flags -> Hamming distance 2 >= 2 -> kept
    row_c = dict(base)
    row_c[FLAG_NAMES[0]] = True
    row_c[FLAG_NAMES[1]] = True
    row_c["rule_id"] = 2

    ranked = pd.DataFrame([row_a, row_b, row_c])
    survivors = dedupe_rules(ranked, min_hamming=2)

    assert list(survivors["rule_id"]) == [0, 2]


def test_dedupe_rules_empty_input():
    empty = pd.DataFrame(columns=["rule_id"] + FLAG_NAMES)
    result = dedupe_rules(empty, min_hamming=2)

    assert result.empty
    assert list(result.columns) == list(empty.columns)


# ---------------------------------------------------------------------------
# T10: full pipeline smoke test
# ---------------------------------------------------------------------------

def test_full_pipeline_smoke():
    df = _make_eval_df()
    rules = enumerate_rules()
    evaluated = evaluate_rules(df, rules, ret_col="ret_fwd_12", hac_lags=12)
    top20, bottom20 = rank_rules(evaluated, n_top=20)
    survivors = dedupe_rules(top20, min_hamming=2)

    assert len(top20) <= 20
    assert len(bottom20) <= 20
    assert len(survivors) <= len(top20)
    for f in FLAG_NAMES:
        assert f in survivors.columns
