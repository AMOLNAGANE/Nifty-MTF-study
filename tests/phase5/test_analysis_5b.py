from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.phase5_mining.analysis_5b import (
    _NUMERIC_FEATURE_BASES,
    _PAIR_SUFFIXES,
    _STATE_BASES,
    feature_columns,
    prepare_features,
    class_balance,
    walk_forward_split,
    train_lgbm,
    evaluate_auc,
    permutation_importance_table,
    compute_shap_values,
    plot_shap_summary,
    plot_shap_dependence,
    top_splits,
)


# ---------------------------------------------------------------------------
# T1: feature_columns -- count and uniqueness
# ---------------------------------------------------------------------------

def test_feature_columns_count_and_unique():
    cols = feature_columns()
    assert len(cols) == 73
    assert len(set(cols)) == 73
    assert cols[0] == "bar_in_session"
    # 64 numeric + 8 ordinal state cols after bar_in_session
    assert len(cols) == 1 + 4 * len(_NUMERIC_FEATURE_BASES) + 4 * len(_STATE_BASES)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_master_like_df(n: int, seed: int = 0, start: str = "2022-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {}

    data["bar_close"] = pd.date_range(start, periods=n, freq="D", tz="Asia/Kolkata")
    data["bar_in_session"] = rng.integers(0, 75, n).astype(float)

    for suffix in _PAIR_SUFFIXES:
        for base in _NUMERIC_FEATURE_BASES:
            data[f"{base}{suffix}"] = rng.normal(0, 1, n)

    state_choices = np.array(
        ["bull_accel", "bull_decel", "bear_decel", "bear_accel", None], dtype=object
    )
    for suffix in _PAIR_SUFFIXES:
        for base in _STATE_BASES:
            data[f"{base}{suffix}"] = rng.choice(state_choices, n)

    # ret_fwd_12 strongly driven by A_hist (5m) -- gives the model a learnable signal
    noise = rng.normal(0, 0.0005, n)
    data["ret_fwd_12"] = 0.01 * np.asarray(data["A_hist"]) + noise

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# T2: prepare_features -- shape, columns, NaN-target dropping
# ---------------------------------------------------------------------------

def test_prepare_features_shape_and_columns():
    df = _make_master_like_df(n=10, seed=1)
    df.loc[df.index[:2], "ret_fwd_12"] = np.nan

    X, y = prepare_features(df)

    assert list(X.columns) == feature_columns()
    assert X.shape == (8, 73)
    assert set(y.unique()) <= {0, 1}
    assert X.index.equals(y.index)


# ---------------------------------------------------------------------------
# T3: prepare_features -- ordinal state encoding
# ---------------------------------------------------------------------------

def test_prepare_features_state_ordinal_encoding():
    df = _make_master_like_df(n=4, seed=2)

    df.loc[df.index[0], "A_state"] = "bull_accel"
    df.loc[df.index[0], "B_state"] = "bull_decel"
    df.loc[df.index[0], "A_state_15m"] = "bear_decel"
    df.loc[df.index[0], "B_state_15m"] = "bear_accel"
    df.loc[df.index[0], "A_state_1h"] = None
    df.loc[df.index[0], "B_state_1h"] = "unrecognized_value"

    X, _ = prepare_features(df)

    row0 = X.iloc[0]
    assert row0["A_state_ord"] == 2
    assert row0["B_state_ord"] == 1
    assert row0["A_state_15m_ord"] == -1
    assert row0["B_state_15m_ord"] == -2
    assert row0["A_state_1h_ord"] == 0
    assert row0["B_state_1h_ord"] == 0


# ---------------------------------------------------------------------------
# T4: class_balance
# ---------------------------------------------------------------------------

def test_class_balance():
    y = pd.Series([1, 1, 0, 0, 0])
    balance = class_balance(y)

    assert balance["n"] == 5
    assert balance["n_pos"] == 2
    assert balance["n_neg"] == 3
    assert balance["pct_pos"] == pytest.approx(0.4)
    assert balance["pct_neg"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# T5: walk_forward_split -- train <= 2023, val == 2024
# ---------------------------------------------------------------------------

def test_walk_forward_split_by_year():
    n = 5
    years = [2022] * n + [2023] * n + [2024] * n + [2025] * n
    df = pd.DataFrame({
        "bar_close": pd.to_datetime([f"{y}-06-01" for y in years]),
        "x": range(len(years)),
    })

    train, val = walk_forward_split(df)

    assert set(train["bar_close"].dt.year.unique()) == {2022, 2023}
    assert set(val["bar_close"].dt.year.unique()) == {2024}
    assert len(train) == 2 * n
    assert len(val) == n


# ---------------------------------------------------------------------------
# T6: train_lgbm + evaluate_auc -- model learns the planted signal
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_model_and_val():
    df = _make_master_like_df(n=400, seed=42)
    X, y = prepare_features(df)

    X_train, X_val = X.iloc[:300], X.iloc[300:]
    y_train, y_val = y.iloc[:300], y.iloc[300:]

    model = train_lgbm(X_train, y_train, seed=42)
    return model, X_val, y_val


def test_train_lgbm_and_evaluate_auc(trained_model_and_val):
    model, X_val, y_val = trained_model_and_val
    auc = evaluate_auc(model, X_val, y_val)

    assert 0.0 <= auc <= 1.0
    assert auc > 0.7, f"Expected model to learn the planted A_hist signal, got AUC={auc}"


# ---------------------------------------------------------------------------
# T7: permutation_importance_table
# ---------------------------------------------------------------------------

def test_permutation_importance_table(trained_model_and_val):
    model, X_val, y_val = trained_model_and_val
    table = permutation_importance_table(model, X_val, y_val, seed=42, n_repeats=5, n_top=15)

    assert list(table.columns) == ["feature", "importance_mean", "importance_std"]
    assert len(table) <= 15
    # sorted descending
    assert (table["importance_mean"].diff().dropna() <= 1e-12).all()
    # the planted signal feature should be the most important
    assert table.iloc[0]["feature"] == "A_hist"


# ---------------------------------------------------------------------------
# T8: top_splits
# ---------------------------------------------------------------------------

def test_top_splits(trained_model_and_val):
    model, _, _ = trained_model_and_val
    splits = top_splits(model, n=3)

    assert list(splits.columns) == ["split_feature", "threshold", "gain", "node_depth", "count"]
    assert len(splits) <= 3
    assert (splits["gain"].diff().dropna() <= 1e-9).all()
    # root split should be on the planted signal feature
    assert splits.iloc[0]["split_feature"] == "A_hist"


# ---------------------------------------------------------------------------
# T9: SHAP values + plots (lightweight)
# ---------------------------------------------------------------------------

def test_compute_shap_values_shape(trained_model_and_val):
    model, X_val, _ = trained_model_and_val
    shap_values = compute_shap_values(model, X_val)

    assert shap_values.shape == X_val.shape


def test_plot_shap_summary_creates_file(trained_model_and_val, tmp_path):
    model, X_val, _ = trained_model_and_val
    shap_values = compute_shap_values(model, X_val)

    out_path = tmp_path / "shap_summary.png"
    plot_shap_summary(shap_values, X_val, out_path, max_display=15)

    assert out_path.exists()


def test_plot_shap_dependence_creates_file(trained_model_and_val, tmp_path):
    model, X_val, _ = trained_model_and_val
    shap_values = compute_shap_values(model, X_val)

    out_path = tmp_path / "shap_dependence.png"
    plot_shap_dependence(shap_values, X_val, "A_hist", "A_hist_15m", out_path)

    assert out_path.exists()
