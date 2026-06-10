from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before importing pyplot
import matplotlib.pyplot as plt
from pathlib import Path

from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Function 1: Pearson & Spearman correlations
# ---------------------------------------------------------------------------

def compute_correlations(
    df: pd.DataFrame,
    feature_cols: list[str],
    horizon_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pearson_df, spearman_df) — shape (n_features, n_horizons)."""
    pearson_data: dict[str, list[float]] = {h: [] for h in horizon_cols}
    spearman_data: dict[str, list[float]] = {h: [] for h in horizon_cols}

    for feature in feature_cols:
        for horizon in horizon_cols:
            pearson_r = df[feature].corr(df[horizon], method="pearson")
            spearman_r = df[feature].corr(df[horizon], method="spearman")
            pearson_data[horizon].append(pearson_r)
            spearman_data[horizon].append(spearman_r)

    pearson_df = pd.DataFrame(pearson_data, index=feature_cols)
    spearman_df = pd.DataFrame(spearman_data, index=feature_cols)
    return pearson_df, spearman_df


# ---------------------------------------------------------------------------
# Function 2: Decile-return table
# ---------------------------------------------------------------------------

def compute_decile_returns(
    df: pd.DataFrame,
    feature_col: str,
    horizon_cols: list[str],
) -> pd.DataFrame:
    """Bin feature into 10 deciles; return mean returns per decile + count."""
    try:
        decile_labels, bins = pd.qcut(
            df[feature_col], q=10, retbins=True, duplicates="drop"
        )
    except ValueError:
        return pd.DataFrame()

    # pd.qcut with duplicates='drop' may produce fewer than 10 bins for near-constant data
    n_unique_bins = len(bins) - 1
    if n_unique_bins < 2:
        return pd.DataFrame()

    tmp = df[horizon_cols].copy()
    tmp["_decile"] = decile_labels

    result = tmp.groupby("_decile", observed=True)[horizon_cols].mean()
    result["count"] = tmp.groupby("_decile", observed=True)[horizon_cols[0]].count()
    return result


# ---------------------------------------------------------------------------
# Function 3: State-conditional returns with Newey-West HAC standard errors
# ---------------------------------------------------------------------------

def _hac_stats(
    returns: pd.Series, hac_lags: int
) -> tuple[float, float, float, int]:
    """Compute mean, count, HAC t-stat, and raw p-value for a return series."""
    clean = returns.dropna()
    n = len(clean)
    mean = float(clean.mean()) if n > 0 else float("nan")
    if n < 20:
        return mean, n, float("nan"), float("nan")

    y = clean.values.astype(float)
    X = np.ones((len(y), 1))
    model = OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": hac_lags, "use_correction": True},
    )
    t_stat = float(model.tvalues[0])
    p_val = float(model.pvalues[0])
    return mean, n, t_stat, p_val


def compute_state_conditional_returns(
    df: pd.DataFrame,
    state_col: str,
    horizon_cols: list[str],
    hac_lags: int = 12,
) -> pd.DataFrame:
    """Return multi-index DataFrame (state, horizon) × (mean, count, t_stat, p_val_raw)."""
    states = sorted(df[state_col].dropna().unique())
    records = []

    for state in states:
        mask = df[state_col] == state
        for horizon in horizon_cols:
            mean, count, t_stat, p_val = _hac_stats(df.loc[mask, horizon], hac_lags)
            records.append(
                {
                    "state": state,
                    "horizon": horizon,
                    "mean": mean,
                    "count": count,
                    "t_stat": t_stat,
                    "p_val_raw": p_val,
                }
            )

    result = pd.DataFrame(records).set_index(["state", "horizon"])
    return result


# ---------------------------------------------------------------------------
# Function 4: Multiple-testing correction
# ---------------------------------------------------------------------------

def apply_multiple_testing_correction(
    result_df: pd.DataFrame,
    p_col: str = "p_val_raw",
    method: str = "fdr_bh",
) -> pd.DataFrame:
    """Add a p_val_adj column using Benjamini-Hochberg or Bonferroni correction."""
    out = result_df.copy()
    p_vals = out[p_col].values.astype(float)
    valid_mask = ~np.isnan(p_vals)
    adj = np.full(len(p_vals), float("nan"))

    if valid_mask.sum() > 0:
        _, adj_valid, _, _ = multipletests(p_vals[valid_mask], method=method)
        adj[valid_mask] = adj_valid

    out["p_val_adj"] = adj
    return out


# ---------------------------------------------------------------------------
# Function 5: Decile-curve plot
# ---------------------------------------------------------------------------

def plot_decile_curves(
    decile_df: pd.DataFrame,
    feature_name: str,
    tf: str,
    out_dir: Path,
) -> None:
    """Plot mean return vs decile rank (one line per horizon). Save PNG; skip if empty."""
    if decile_df is None or decile_df.empty:
        return

    horizon_cols = [c for c in decile_df.columns if c != "count"]
    if not horizon_cols:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(1, len(decile_df) + 1)
    for horizon in horizon_cols:
        ax.plot(list(x), decile_df[horizon].values, marker="o", label=horizon)

    ax.set_xlabel("Decile (1=lowest, 10=highest)")
    ax.set_ylabel("Mean forward return")
    ax.set_title(f"Decile returns: {feature_name} [{tf}]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = Path(out_dir) / f"decile_{tf}_{feature_name}.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
