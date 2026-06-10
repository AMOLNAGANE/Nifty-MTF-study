from __future__ import annotations
import pandas as pd
import numpy as np


def _zero_cross_flag(series: pd.Series) -> pd.Series:
    """Pine-style strict zero cross: +1 when prev < 0 AND current > 0, -1 inverse."""
    prev = series.shift(1)
    flag = pd.Series(0, index=series.index, dtype=int)
    flag[(prev < 0) & (series > 0)] = 1
    flag[(prev > 0) & (series < 0)] = -1
    return flag


def _compute_state(hist: pd.Series, hist_slope: pd.Series) -> pd.Series:
    state = pd.Series(pd.NA, index=hist.index, dtype=object)
    state[(hist > 0) & (hist_slope > 0)] = "bull_accel"
    state[(hist > 0) & (hist_slope <= 0)] = "bull_decel"
    state[(hist < 0) & (hist_slope >= 0)] = "bear_decel"
    state[(hist < 0) & (hist_slope < 0)] = "bear_accel"
    zero = hist == 0
    state[zero & (hist_slope > 0)] = "bear_decel"
    state[zero & (hist_slope < 0)] = "bull_decel"
    state[zero & (hist_slope == 0)] = pd.NA
    return state.ffill()


def compute_indicator_a(df: pd.DataFrame) -> pd.DataFrame:
    """Add MACD_A (12/26/9) features. ewm(adjust=False) matches Pine's ta.ema."""
    out = df.copy()
    close = out["close"]

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    hist_slope = hist.diff(1)

    out["A_macd"] = macd
    out["A_signal"] = signal
    out["A_hist"] = hist
    out["A_hist_slope"] = hist_slope
    out["A_macd_sign"] = np.sign(macd).astype(int)
    out["A_hist_sign"] = np.sign(hist).astype(int)
    out["A_macd_zero_cross"] = _zero_cross_flag(macd)
    out["A_hist_zero_cross"] = _zero_cross_flag(hist)
    out["A_state"] = _compute_state(hist, hist_slope)

    return out
