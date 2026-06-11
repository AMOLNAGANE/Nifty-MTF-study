from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


# ---------------------------------------------------------------------------
# Function 1: Cross-correlation
# ---------------------------------------------------------------------------

def compute_cross_correlation(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    max_lag: int = 20,
) -> pd.Series:
    """Compute cross-correlation between df[col_a] and df[col_b] at lags.

    At lag k: correlation between col_a and col_b.shift(k).
    Positive k means col_b leads col_a (older B aligns with newer A).
    Negative k means col_a leads col_b.

    Returns a pd.Series indexed by integer lag from -max_lag to +max_lag.

    Note on expected results for A_hist vs B_roc:
        A leads B (peak cross-correlation at negative lag) because A uses
        shorter EMAs (12/26) vs B (40/90). If peak appears at positive lag,
        suspect a computation error.
    """
    lags = range(-max_lag, max_lag + 1)
    corrs = {}
    for k in lags:
        # At lag k: correlate col_a with col_b.shift(k).
        # Positive k = col_b leads col_a (col_b is shifted backward so older B aligns with newer A).
        # Negative k = col_a leads col_b.
        # corr(col_a[t], col_b[t-k]): peak at k s.t. t-k = t - lead, i.e. k = lead.
        # Equivalently: df[col_a].corr(df[col_b].shift(-k)) gives positive peak when B leads.
        # We use shift(k) so that negative k → col_a leads (consistent with spec convention).
        corrs[k] = df[col_a].corr(df[col_b].shift(k))
    return pd.Series(corrs, name="cross_correlation")


# ---------------------------------------------------------------------------
# Function 2: Nested regression test (H8 — incremental information)
# ---------------------------------------------------------------------------

def nested_regression_test(
    df: pd.DataFrame,
    target_col: str,
    feature_a: str,
    feature_b: str,
    hac_lags: int = 12,
) -> dict:
    """Test H8: does feature_b add incremental predictive power beyond feature_a?

    Fits two HAC-robust OLS regressions:
      Restricted:   target ~ feature_a
      Unrestricted: target ~ feature_a + feature_b

    Returns dict with keys:
        r2_restricted, r2_unrestricted, delta_r2,
        b_coef, b_tstat, b_pval, n_obs
    If fewer than 30 non-NaN observations remain, returns all-NaN dict.
    """
    _nan_result = {
        "r2_restricted": float("nan"),
        "r2_unrestricted": float("nan"),
        "delta_r2": float("nan"),
        "b_coef": float("nan"),
        "b_tstat": float("nan"),
        "b_pval": float("nan"),
        "n_obs": float("nan"),
    }

    # Drop rows with NaN in any relevant column
    subset = df[[target_col, feature_a, feature_b]].dropna()
    n_obs = len(subset)

    if n_obs < 30:
        return _nan_result

    y = subset[target_col].values
    X_a = add_constant(subset[[feature_a]].values)
    X_ab = add_constant(subset[[feature_a, feature_b]].values)

    hac_opts = {"cov_type": "HAC", "cov_kwds": {"maxlags": hac_lags, "use_correction": True}}

    res_restricted = OLS(y, X_a).fit(**hac_opts)
    res_unrestricted = OLS(y, X_ab).fit(**hac_opts)

    r2_restricted = float(res_restricted.rsquared)
    r2_unrestricted = float(res_unrestricted.rsquared)

    # feature_b is the 3rd parameter (index 2): intercept, feature_a, feature_b
    b_coef = float(res_unrestricted.params[2])
    b_tstat = float(res_unrestricted.tvalues[2])
    b_pval = float(res_unrestricted.pvalues[2])

    return {
        "r2_restricted": r2_restricted,
        "r2_unrestricted": r2_unrestricted,
        "delta_r2": r2_unrestricted - r2_restricted,
        "b_coef": b_coef,
        "b_tstat": b_tstat,
        "b_pval": b_pval,
        "n_obs": n_obs,
    }


# ---------------------------------------------------------------------------
# Function 3: Turn-confirmation timing (H10 — turn-confirmation timing)
# ---------------------------------------------------------------------------

def turn_confirmation_timing(
    df: pd.DataFrame,
    a_flip_col: str,
    b_confirm_col: str,
    b_confirm_threshold: float = 0.0,
    k_bars: int = 20,
) -> tuple[pd.DataFrame, dict]:
    """Test H10: when A flips bullish, how quickly does B confirm?

    Finds "A flip" events: bars where df[a_flip_col] == 1.
    For each flip at bar i, checks whether df[b_confirm_col] crosses above
    b_confirm_threshold within the next k_bars bars.

    Returns:
        events_df: DataFrame with columns [flip_idx, confirmed, bars_to_confirm]
            bars_to_confirm = NaN if not confirmed
        summary: dict with keys:
            n_flips, n_confirmed, n_not_confirmed,
            confirmation_rate, median_bars_to_confirm
    """
    flip_indices = df.index[df[a_flip_col] == 1].tolist()
    b_values = df[b_confirm_col].values
    idx_array = df.index.tolist()

    records = []
    for flip_idx in flip_indices:
        # Find positional location of flip_idx in the DataFrame
        pos = idx_array.index(flip_idx)

        # Window: bars i+1 to i+k_bars (inclusive), clipped to array length
        window_start = pos + 1
        window_end = min(pos + k_bars + 1, len(b_values))

        confirmed = False
        bars_to_confirm = float("nan")

        for offset, w_pos in enumerate(range(window_start, window_end), start=1):
            if b_values[w_pos] > b_confirm_threshold:
                confirmed = True
                bars_to_confirm = offset
                break

        records.append({
            "flip_idx": flip_idx,
            "confirmed": confirmed,
            "bars_to_confirm": bars_to_confirm,
        })

    events_df = pd.DataFrame(records, columns=["flip_idx", "confirmed", "bars_to_confirm"])

    n_flips = len(records)
    n_confirmed = int(events_df["confirmed"].sum()) if n_flips > 0 else 0
    n_not_confirmed = n_flips - n_confirmed

    if n_confirmed > 0:
        median_bars = float(events_df.loc[events_df["confirmed"], "bars_to_confirm"].median())
    else:
        median_bars = float("nan")

    confirmation_rate = (n_confirmed / n_flips) if n_flips > 0 else 0.0

    summary = {
        "n_flips": n_flips,
        "n_confirmed": n_confirmed,
        "n_not_confirmed": n_not_confirmed,
        "confirmation_rate": confirmation_rate,
        "median_bars_to_confirm": median_bars,
    }

    return events_df, summary
