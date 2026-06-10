from __future__ import annotations
import numpy as np
import pandas as pd

_FWD_HORIZONS = [1, 3, 6, 12, 24]

_BARRIER_CONFIG: dict[str, tuple[float, int]] = {
    "5m":  (0.0020, 24),
    "15m": (0.0035, 24),
    "1h":  (0.0060, 24),
    "1d":  (0.0120, 10),
}


def add_forward_returns(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    for n in _FWD_HORIZONS:
        shifted = close.shift(-n)
        out[f"ret_fwd_{n}"] = (shifted / close) - 1.0
    return out


def add_barrier_labels(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    threshold, window = _BARRIER_CONFIG[tf]
    out = df.copy()
    close = out["close"].to_numpy(dtype=float)
    n = len(close)
    labels = np.full(n, np.nan)

    for i in range(n - window):
        ref = close[i]
        upper = ref * (1.0 + threshold)
        lower = ref * (1.0 - threshold)
        result = 0
        for j in range(i + 1, i + window + 1):
            if close[j] >= upper:
                result = 1
                break
            if close[j] <= lower:
                result = -1
                break
        labels[i] = result

    out["barrier_hit"] = labels
    return out


def add_regime_label(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].to_numpy(dtype=float)
    n = len(close)
    regime = np.full(n, None, dtype=object)
    x = np.arange(window, dtype=float)

    for i in range(window - 1, n):
        y = close[i - window + 1 : i + 1]
        coeffs = np.polyfit(x, y, 1)
        a = coeffs[0]
        y_hat = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
        if r2 > 0.5 and a > 0:
            regime[i] = "trend_up"
        elif r2 > 0.5 and a < 0:
            regime[i] = "trend_down"
        else:
            regime[i] = "chop"

    out["regime"] = regime
    return out
