from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.phase3_intra.analysis_3d import (
    detect_peaks,
    detect_divergence,
    compute_divergence_returns,
)

# ---------------------------------------------------------------------------
# Test 1: detect_peaks finds local max in a sinusoidal series
# ---------------------------------------------------------------------------

def test_detect_peaks_finds_local_max():
    """A single-cycle sine wave has a clear maximum; a peak should be found near it."""
    np.random.seed(42)
    n = 50
    x = np.linspace(0, 2 * np.pi, n)
    series = pd.Series(np.sin(x))
    high_idx, _ = detect_peaks(series, min_spacing=5)
    # Maximum of sin(x) on [0, 2π] is at π/2 → index ~12 in a 50-point series
    expected_peak = int(n / 4)  # ~12
    assert len(high_idx) >= 1, "Expected at least one peak found"
    assert any(abs(h - expected_peak) <= 3 for h in high_idx), (
        f"No peak found near index {expected_peak}; found peaks at {high_idx}"
    )


# ---------------------------------------------------------------------------
# Test 2: detect_peaks finds local min in a sinusoidal series
# ---------------------------------------------------------------------------

def test_detect_peaks_finds_local_min():
    """A single-cycle sine wave has a clear trough; a low should be found near it."""
    np.random.seed(42)
    n = 50
    x = np.linspace(0, 2 * np.pi, n)
    series = pd.Series(np.sin(x))
    _, low_idx = detect_peaks(series, min_spacing=5)
    # Minimum of sin(x) on [0, 2π] is at 3π/2 → index ~37 in a 50-point series
    expected_trough = int(3 * n / 4)  # ~37
    assert len(low_idx) >= 1, "Expected at least one trough found"
    assert any(abs(l - expected_trough) <= 3 for l in low_idx), (
        f"No trough found near index {expected_trough}; found troughs at {low_idx}"
    )


# ---------------------------------------------------------------------------
# Test 3: min_spacing is respected — two peaks closer than min_spacing → only one
# ---------------------------------------------------------------------------

def test_detect_peaks_min_spacing_respected():
    """Two adjacent bumps 3 bars apart with min_spacing=10 → only one detected."""
    np.random.seed(42)
    # Build a series with two small peaks very close together (bars 10 and 13)
    series_vals = np.zeros(40)
    series_vals[10] = 1.0
    series_vals[13] = 0.9
    series = pd.Series(series_vals)
    high_idx, _ = detect_peaks(series, min_spacing=10)
    # With distance=10, only the taller peak should survive
    assert len(high_idx) == 1, (
        f"Expected 1 peak with min_spacing=10, got {len(high_idx)} at {high_idx}"
    )


# ---------------------------------------------------------------------------
# Test 4: detect_divergence detects bearish divergence
# ---------------------------------------------------------------------------

def test_detect_divergence_bearish():
    """Price makes higher high, indicator makes lower high → bearish divergence."""
    np.random.seed(42)
    n = 60
    # Build price: higher high at bar 40 vs bar 10
    price = np.ones(n) * 100.0
    price[10] = 110.0   # first high
    price[40] = 120.0   # second (higher) high → higher_high

    # Build indicator: lower high at bar 40 vs bar 10
    indicator = np.ones(n) * 50.0
    indicator[10] = 70.0  # first high
    indicator[40] = 60.0  # second (lower) high → lower_high (divergence)

    df = pd.DataFrame({"price": price, "rsi": indicator})
    div_df = detect_divergence(df, "price", "rsi", min_spacing=10)

    bearish = div_df[div_df["div_type"] == "bearish"]
    assert len(bearish) >= 1, (
        f"Expected at least one bearish divergence; got:\n{div_df}"
    )
    assert (bearish["price_direction"] == "higher_high").all()
    assert (bearish["indicator_direction"] == "lower_high").all()


# ---------------------------------------------------------------------------
# Test 5: detect_divergence detects bullish divergence
# ---------------------------------------------------------------------------

def test_detect_divergence_bullish():
    """Price makes lower low, indicator makes higher low → bullish divergence."""
    np.random.seed(42)
    n = 60
    # Build price: lower low at bar 40 vs bar 10
    price = np.ones(n) * 100.0
    price[10] = 90.0   # first low
    price[40] = 80.0   # second (lower) low → lower_low

    # Build indicator: higher low at bar 40 vs bar 10
    indicator = np.ones(n) * 50.0
    indicator[10] = 30.0  # first low
    indicator[40] = 40.0  # second (higher) low → higher_low (divergence)

    df = pd.DataFrame({"price": price, "rsi": indicator})
    div_df = detect_divergence(df, "price", "rsi", min_spacing=10)

    bullish = div_df[div_df["div_type"] == "bullish"]
    assert len(bullish) >= 1, (
        f"Expected at least one bullish divergence; got:\n{div_df}"
    )
    assert (bullish["price_direction"] == "lower_low").all()
    assert (bullish["indicator_direction"] == "higher_low").all()


# ---------------------------------------------------------------------------
# Test 6: no divergence when price and indicator agree (both higher highs)
# ---------------------------------------------------------------------------

def test_detect_divergence_no_divergence():
    """Price higher high + indicator higher high → no bearish divergence."""
    np.random.seed(42)
    n = 60
    price = np.ones(n) * 100.0
    price[10] = 110.0
    price[40] = 120.0   # higher high

    indicator = np.ones(n) * 50.0
    indicator[10] = 60.0
    indicator[40] = 70.0  # also higher high — no divergence

    df = pd.DataFrame({"price": price, "rsi": indicator})
    div_df = detect_divergence(df, "price", "rsi", min_spacing=10)

    bearish = div_df[div_df["div_type"] == "bearish"]
    assert len(bearish) == 0, (
        f"Expected no bearish divergence when indicator agrees; got:\n{bearish}"
    )


# ---------------------------------------------------------------------------
# Test 7: all confirm_idx values are < len(df)
# ---------------------------------------------------------------------------

def test_detect_divergence_confirm_idx_in_bounds():
    """No divergence event should have confirm_idx >= len(df)."""
    np.random.seed(42)
    n = 60
    price = np.ones(n) * 100.0
    price[10] = 110.0
    price[40] = 120.0

    indicator = np.ones(n) * 50.0
    indicator[10] = 70.0
    indicator[40] = 60.0

    df = pd.DataFrame({"price": price, "rsi": indicator})
    div_df = detect_divergence(df, "price", "rsi", min_spacing=10)

    if len(div_df) > 0:
        oob = div_df[div_df["confirm_idx"] >= len(df)]
        assert len(oob) == 0, (
            f"Some confirm_idx >= len(df)={len(df)}:\n{oob}"
        )


# ---------------------------------------------------------------------------
# Test 8: compute_divergence_returns shape
# ---------------------------------------------------------------------------

def test_compute_divergence_returns_shape():
    """One divergence event + 3 horizon cols → result has 1 row, 4 columns."""
    np.random.seed(42)
    n = 80
    horizon_cols = ["ret_fwd_1", "ret_fwd_3", "ret_fwd_6"]

    # Build df with forward return columns
    df = pd.DataFrame({
        "price": np.ones(n) * 100.0,
        "rsi": np.ones(n) * 50.0,
        "ret_fwd_1": np.random.normal(0, 0.01, n),
        "ret_fwd_3": np.random.normal(0, 0.01, n),
        "ret_fwd_6": np.random.normal(0, 0.01, n),
    })

    # Construct exactly one bearish divergence
    divergence_df = pd.DataFrame({
        "peak_idx": [20],
        "confirm_idx": [25],
        "div_type": ["bearish"],
        "price_direction": ["higher_high"],
        "indicator_direction": ["lower_high"],
    })

    result = compute_divergence_returns(df, divergence_df, horizon_cols)

    assert result.shape == (1, len(horizon_cols) + 1), (
        f"Expected shape (1, {len(horizon_cols) + 1}), got {result.shape}"
    )


# ---------------------------------------------------------------------------
# Test 9: compute_divergence_returns uses confirm_idx row, NOT peak_idx row
# ---------------------------------------------------------------------------

def test_compute_divergence_returns_uses_confirm_idx():
    """Returns must come from confirm_idx row, not peak_idx row."""
    np.random.seed(42)
    n = 80
    horizon_cols = ["ret_fwd_1"]

    ret_values = np.zeros(n)
    peak_idx = 20
    confirm_idx = 25
    ret_values[peak_idx] = 0.999   # decoy — should NOT be used
    ret_values[confirm_idx] = 0.123  # this is the correct value

    df = pd.DataFrame({
        "ret_fwd_1": ret_values,
    })

    divergence_df = pd.DataFrame({
        "peak_idx": [peak_idx],
        "confirm_idx": [confirm_idx],
        "div_type": ["bearish"],
        "price_direction": ["higher_high"],
        "indicator_direction": ["lower_high"],
    })

    result = compute_divergence_returns(df, divergence_df, horizon_cols)
    returned_value = result.iloc[0]["ret_fwd_1"]

    assert abs(returned_value - 0.123) < 1e-9, (
        f"Expected return from confirm_idx={confirm_idx} (0.123), "
        f"got {returned_value} (which may be from peak_idx={peak_idx})"
    )


# ---------------------------------------------------------------------------
# Test 10: compute_divergence_returns is NaN when forward return is NaN
# ---------------------------------------------------------------------------

def test_compute_divergence_returns_nan_when_oob():
    """If ret_fwd_12 at confirm_idx is NaN (end of series), result is NaN."""
    np.random.seed(42)
    n = 40
    horizon_cols = ["ret_fwd_1", "ret_fwd_12"]

    ret_fwd_1 = np.random.normal(0, 0.01, n)
    ret_fwd_12 = np.random.normal(0, 0.01, n)
    # Set last row's ret_fwd_12 to NaN (simulates insufficient future data)
    ret_fwd_12[-1] = np.nan

    df = pd.DataFrame({
        "ret_fwd_1": ret_fwd_1,
        "ret_fwd_12": ret_fwd_12,
    })

    # confirm_idx points to the last row (index n-1)
    confirm_idx = n - 1
    divergence_df = pd.DataFrame({
        "peak_idx": [confirm_idx - 5],
        "confirm_idx": [confirm_idx],
        "div_type": ["bearish"],
        "price_direction": ["higher_high"],
        "indicator_direction": ["lower_high"],
    })

    result = compute_divergence_returns(df, divergence_df, horizon_cols)

    assert pd.isna(result.iloc[0]["ret_fwd_12"]), (
        f"Expected NaN for ret_fwd_12 at last row, got {result.iloc[0]['ret_fwd_12']}"
    )
    # ret_fwd_1 should NOT be NaN (it's a normal value)
    assert not pd.isna(result.iloc[0]["ret_fwd_1"]), (
        f"Expected non-NaN for ret_fwd_1, got {result.iloc[0]['ret_fwd_1']}"
    )
