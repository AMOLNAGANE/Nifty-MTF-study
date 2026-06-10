from __future__ import annotations
import pandas as pd
import numpy as np


def _compute_b_state(roc: pd.Series, roc_slope: pd.Series) -> pd.Series:
    state = pd.Series(pd.NA, index=roc.index, dtype=object)
    state[(roc > 0) & (roc_slope > 0)] = "bull_accel"
    state[(roc > 0) & (roc_slope <= 0)] = "bull_decel"
    state[(roc < 0) & (roc_slope >= 0)] = "bear_decel"
    state[(roc < 0) & (roc_slope < 0)] = "bear_accel"
    zero = roc == 0
    state[zero & (roc_slope > 0)] = "bear_decel"
    state[zero & (roc_slope < 0)] = "bull_decel"
    state[zero & (roc_slope == 0)] = pd.NA
    return state.ffill()


def compute_indicator_b(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA-Gap-ROC (40/90/7, zROC 100) features. ewm(adjust=False) matches Pine."""
    out = df.copy()
    close = out["close"]

    ema40 = close.ewm(span=40, adjust=False).mean()
    ema90 = close.ewm(span=90, adjust=False).mean()
    gap = ema40 - ema90
    gap_prev7 = gap.shift(7)

    # ROC: 0 when gap_prev == 0 exactly (Pine convention), NaN propagates from gap_prev NaN
    roc = pd.Series(np.nan, index=out.index)
    valid = gap_prev7.notna() & (gap_prev7 != 0)
    zero_prev = gap_prev7.notna() & (gap_prev7 == 0)
    roc[valid] = 100.0 * (gap[valid] - gap_prev7[valid]) / gap_prev7[valid]
    roc[zero_prev] = 0.0

    # B_roc_invalid: 1 when abs(gap_prev7) < 1st-percentile epsilon
    abs_prev = gap_prev7.abs().dropna()
    epsilon = float(abs_prev.quantile(0.01)) if len(abs_prev) > 0 else 0.0
    roc_invalid = (gap_prev7.abs() < epsilon).astype(int)

    roc_sma = roc.rolling(100, min_periods=100).mean()
    roc_std = roc.rolling(100, min_periods=100).std()
    zroc = (roc - roc_sma) / roc_std
    roc_slope = roc.diff(1)

    out["B_gap"] = gap
    out["B_gap_norm"] = gap / close
    out["B_roc"] = roc
    out["B_roc_invalid"] = roc_invalid
    out["B_zroc"] = zroc
    out["B_gap_acc"] = gap - gap.shift(7)
    out["B_roc_slope"] = roc_slope
    out["B_zroc_extreme"] = (zroc.abs() > 2).astype(int)
    out["B_state"] = _compute_b_state(roc, roc_slope)

    return out
