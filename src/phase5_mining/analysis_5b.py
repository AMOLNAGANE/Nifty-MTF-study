from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Constants -- feature set (T5.5)
# ---------------------------------------------------------------------------

_PAIR_SUFFIXES = ["", "_15m", "_1h", "_1d"]

# 16 numeric indicator columns per TF x 4 TFs = 64
_NUMERIC_FEATURE_BASES = [
    "A_macd", "A_signal", "A_hist", "A_hist_slope",
    "A_macd_sign", "A_hist_sign", "A_macd_zero_cross", "A_hist_zero_cross",
    "B_gap", "B_gap_norm", "B_roc", "B_roc_invalid",
    "B_zroc", "B_gap_acc", "B_roc_slope", "B_zroc_extreme",
]

# 2 state columns per TF x 4 TFs = 8 (ordinal-encoded)
_STATE_BASES = ["A_state", "B_state"]

# bull_accel > bull_decel > (NaN/unknown == 0) > bear_decel > bear_accel
_STATE_ORDINAL_MAP = {
    "bull_accel": 2,
    "bull_decel": 1,
    "bear_decel": -1,
    "bear_accel": -2,
}


def feature_columns() -> list[str]:
    """
    The 73 ML feature names (T5.5):
      - "bar_in_session" (1)
      - 16 numeric indicator cols x 4 TFs (64)
      - 2 ordinal-encoded state cols x 4 TFs (8)
    """
    cols = ["bar_in_session"]
    for suffix in _PAIR_SUFFIXES:
        for base in _NUMERIC_FEATURE_BASES:
            cols.append(f"{base}{suffix}")
    for suffix in _PAIR_SUFFIXES:
        for base in _STATE_BASES:
            cols.append(f"{base}{suffix}_ord")
    return cols


# ---------------------------------------------------------------------------
# T5.5: Build feature matrix X and binary target y
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame, ret_col: str = "ret_fwd_12"
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the ML feature matrix X (columns == feature_columns(), 73 cols) and the
    binary target y = (df[ret_col] > 0).astype(int).

    State columns (A_state{suffix}, B_state{suffix}) are ordinal-encoded via
    _STATE_ORDINAL_MAP; NaN/unrecognized values map to 0. All other features are
    passed through as float (LightGBM handles NaN natively, no imputation).

    Rows where ret_col is NaN are dropped (target undefined). Returns (X, y)
    sharing the same (filtered) index as df.
    """
    valid = df[df[ret_col].notna()]

    data: dict[str, pd.Series] = {"bar_in_session": valid["bar_in_session"].astype(float)}

    for suffix in _PAIR_SUFFIXES:
        for base in _NUMERIC_FEATURE_BASES:
            col = f"{base}{suffix}"
            data[col] = valid[col].astype(float)

    for suffix in _PAIR_SUFFIXES:
        for base in _STATE_BASES:
            col = f"{base}{suffix}"
            data[f"{col}_ord"] = valid[col].map(_STATE_ORDINAL_MAP).fillna(0).astype(float)

    X = pd.DataFrame(data, index=valid.index)[feature_columns()]
    y = (valid[ret_col] > 0).astype(int)
    return X, y


def class_balance(y: pd.Series) -> dict[str, float]:
    """Return {"n", "n_pos", "n_neg", "pct_pos", "pct_neg"} for a 0/1 target."""
    n = int(len(y))
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pct_pos": n_pos / n if n > 0 else float("nan"),
        "pct_neg": n_neg / n if n > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# T5.6: Walk-forward train/validation split
# ---------------------------------------------------------------------------

def walk_forward_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df by bar_close year:
      train = bar_close.dt.year <= 2023
      val   = bar_close.dt.year == 2024

    Returns (train_df, val_df), each a copy of the corresponding row subset.
    """
    train = df[df["bar_close"].dt.year <= 2023].copy()
    val = df[df["bar_close"].dt.year == 2024].copy()
    return train, val


# ---------------------------------------------------------------------------
# T5.7: Train LightGBM and evaluate AUC
# ---------------------------------------------------------------------------

def train_lgbm(X_train: pd.DataFrame, y_train: pd.Series, seed: int = 42):
    """Train an LGBMClassifier(max_depth=4, n_estimators=200, learning_rate=0.05)."""
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        max_depth=4,
        n_estimators=200,
        learning_rate=0.05,
        random_state=seed,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_auc(model, X: pd.DataFrame, y: pd.Series) -> float:
    """ROC-AUC of model's positive-class probability on (X, y)."""
    from sklearn.metrics import roc_auc_score

    proba = model.predict_proba(X)[:, 1]
    return float(roc_auc_score(y, proba))


# ---------------------------------------------------------------------------
# T5.8: Permutation importance
# ---------------------------------------------------------------------------

def permutation_importance_table(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int = 42,
    n_repeats: int = 10,
    n_top: int = 15,
) -> pd.DataFrame:
    """
    sklearn.inspection.permutation_importance(scoring="roc_auc"), returning the
    top n_top features by importance_mean (descending), with columns
    [feature, importance_mean, importance_std].
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model, X_val, y_val,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=seed,
    )
    table = pd.DataFrame({
        "feature": X_val.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })
    return table.sort_values("importance_mean", ascending=False).head(n_top).reset_index(drop=True)


# ---------------------------------------------------------------------------
# T5.9: SHAP values and plots
# ---------------------------------------------------------------------------

def compute_shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """SHAP values (positive class) for X via shap.TreeExplainer."""
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return shap_values


def plot_shap_summary(
    shap_values: np.ndarray, X: pd.DataFrame, out_path: Path, max_display: int = 15
) -> None:
    """Save a SHAP summary (beeswarm) plot of the top max_display features to out_path."""
    import shap

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close("all")


def plot_shap_dependence(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature: str,
    interaction_feature: str,
    out_path: Path,
) -> None:
    """Save a SHAP dependence plot of `feature` colored by `interaction_feature`."""
    import shap

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.dependence_plot(
        feature, shap_values, X,
        interaction_index=interaction_feature,
        ax=ax, show=False,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close("all")


# ---------------------------------------------------------------------------
# T5.10: Top splits of the first tree
# ---------------------------------------------------------------------------

def top_splits(model, n: int = 3) -> pd.DataFrame:
    """
    From model.booster_.trees_to_dataframe(), extract the root split and its
    immediate children ("depth-1") of tree 0, sorted by split_gain descending.

    Returns a DataFrame with columns
    [split_feature, threshold, gain, node_depth, count], at most n rows.
    """
    trees = model.booster_.trees_to_dataframe()
    tree0 = trees[trees["tree_index"] == 0]

    splits = tree0[tree0["split_feature"].notna()]
    if splits.empty:
        return pd.DataFrame(columns=["split_feature", "threshold", "gain", "node_depth", "count"])

    root_depth = int(splits["node_depth"].min())
    splits = splits[splits["node_depth"] <= root_depth + 1]

    result = splits[["split_feature", "threshold", "split_gain", "node_depth", "count"]].rename(
        columns={"split_gain": "gain"}
    )
    return result.sort_values("gain", ascending=False).head(n).reset_index(drop=True)
