from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.phase3_intra.analysis_3a_events import (
    event_study_h2_peak_timing,
    event_study_h3_aligned_vs_tired,
    event_study_h4_pullback,
    event_study_h6_zroc_term_structure,
    event_study_h7_early_turn,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HORIZONS = ["ret_fwd_1", "ret_fwd_3", "ret_fwd_6", "ret_fwd_12", "ret_fwd_24"]


def _base_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Return a minimal base DataFrame with required columns filled with defaults."""
    np.random.seed(seed)
    df = pd.DataFrame(index=range(n))
    df["A_hist"] = np.random.normal(0, 1, n)
    df["A_hist_slope"] = np.random.normal(0, 0.5, n)
    df["A_state"] = "bull_decel"
    df["A_macd"] = np.random.normal(0, 1, n)
    df["A_signal"] = np.random.normal(0, 1, n)
    df["A_macd_sign"] = np.sign(df["A_macd"])
    df["A_hist_sign"] = np.sign(df["A_hist"])
    df["A_macd_zero_cross"] = 0
    df["A_hist_zero_cross"] = 0
    df["B_gap"] = np.random.normal(0, 1, n)
    df["B_gap_norm"] = df["B_gap"] / (df["B_gap"].abs().max() + 1e-9)
    df["B_roc"] = np.random.normal(0, 0.01, n)
    df["B_zroc"] = np.random.normal(0, 1, n)
    df["B_gap_acc"] = np.random.normal(0, 0.1, n)
    df["B_roc_slope"] = np.random.normal(0, 0.01, n)
    df["B_zroc_extreme"] = (df["B_zroc"].abs() > 2.0).astype(int)
    df["B_state"] = "neutral"
    for h in _HORIZONS:
        df[h] = np.random.normal(0.0002, 0.01, n)
    return df


# ---------------------------------------------------------------------------
# H2 — Peak timing
# ---------------------------------------------------------------------------


def test_h2_peak_timing_detects_transitions():
    """A known bull_accel→bull_decel transition at row 5 produces 1 event at row 5."""
    df = _base_df(n=100)
    # Insert the transition: row 4 = bull_accel, row 5 = bull_decel
    df["A_state"] = "bear_accel"           # everything else neutral
    df.loc[4, "A_state"] = "bull_accel"
    df.loc[5, "A_state"] = "bull_decel"

    event_df, summary = event_study_h2_peak_timing(df, _HORIZONS)

    assert len(event_df) == 1, f"Expected 1 event, got {len(event_df)}"
    assert summary["n_events"] == 1
    for h in _HORIZONS:
        assert h in event_df.columns


def test_h2_peak_timing_empty_when_no_transitions():
    """Constant A_state produces 0 events and n_events=0 in summary."""
    df = _base_df(n=100)
    df["A_state"] = "bull_accel"   # never transitions

    event_df, summary = event_study_h2_peak_timing(df, _HORIZONS)

    assert event_df.empty, "Expected empty DataFrame when no transitions"
    assert summary["n_events"] == 0


# ---------------------------------------------------------------------------
# H3 — Aligned vs tired zero-cross
# ---------------------------------------------------------------------------


def test_h3_aligned_groups_present():
    """DataFrame with both aligned and tired MACD zero-crosses produces both groups."""
    df = _base_df(n=100)
    df["A_macd_zero_cross"] = 0

    # Row 10: aligned cross (A_hist > 0, A_hist_slope > 0)
    df.loc[10, "A_macd_zero_cross"] = 1
    df.loc[10, "A_hist"] = 0.5
    df.loc[10, "A_hist_slope"] = 0.3

    # Row 20: tired cross (A_hist <= 0)
    df.loc[20, "A_macd_zero_cross"] = 1
    df.loc[20, "A_hist"] = -0.5
    df.loc[20, "A_hist_slope"] = 0.2

    result = event_study_h3_aligned_vs_tired(df, _HORIZONS)

    groups = result["group"].unique()
    assert "aligned" in groups, f"'aligned' group missing; groups={groups}"
    assert "tired" in groups, f"'tired' group missing; groups={groups}"
    for h in _HORIZONS:
        assert h in result.columns


def test_h3_aligned_group_only():
    """Only aligned crosses → only 'aligned' group in result."""
    df = _base_df(n=100)
    df["A_macd_zero_cross"] = 0

    # Two aligned crosses
    for row in [15, 30]:
        df.loc[row, "A_macd_zero_cross"] = 1
        df.loc[row, "A_hist"] = 1.0
        df.loc[row, "A_hist_slope"] = 1.0

    result = event_study_h3_aligned_vs_tired(df, _HORIZONS)

    groups = result["group"].unique()
    assert "aligned" in groups
    assert "tired" not in groups, f"Unexpected 'tired' group; groups={groups}"


# ---------------------------------------------------------------------------
# H4 — First-pullback
# ---------------------------------------------------------------------------


def test_h4_pullback_groups_present():
    """DataFrame with both with_trend and counter_trend hist zero-crosses produces both groups."""
    df = _base_df(n=100)
    df["A_hist_zero_cross"] = 0

    # Row 10: with_trend (A_macd > 0)
    df.loc[10, "A_hist_zero_cross"] = 1
    df.loc[10, "A_macd"] = 0.8

    # Row 25: counter_trend (A_macd <= 0)
    df.loc[25, "A_hist_zero_cross"] = 1
    df.loc[25, "A_macd"] = -0.5

    result = event_study_h4_pullback(df, _HORIZONS)

    groups = result["group"].unique()
    assert "with_trend" in groups, f"'with_trend' missing; groups={groups}"
    assert "counter_trend" in groups, f"'counter_trend' missing; groups={groups}"
    for h in _HORIZONS:
        assert h in result.columns


# ---------------------------------------------------------------------------
# H6 — zROC term-structure
# ---------------------------------------------------------------------------


def test_h6_zroc_term_structure_has_all_buckets():
    """DataFrame covering all four B_zroc buckets → all four present in result index."""
    df = _base_df(n=100)
    # Distribute B_zroc values across all four buckets
    df.loc[:24, "B_zroc"] = 0.1        # baseline (abs <= 0.5)
    df.loc[25:49, "B_zroc"] = 1.0      # moderate (0.5 < z <= 2.0)
    df.loc[50:74, "B_zroc"] = 3.0      # extreme_bull (> 2.0)
    df.loc[75:99, "B_zroc"] = -3.0     # extreme_bear (< -2.0)

    result = event_study_h6_zroc_term_structure(df, _HORIZONS)

    expected_buckets = {"baseline", "moderate", "extreme_bull", "extreme_bear"}
    actual_buckets = set(result.index)
    assert expected_buckets == actual_buckets, (
        f"Expected buckets {expected_buckets}, got {actual_buckets}"
    )
    for h in _HORIZONS:
        assert h in result.columns


def test_h6_zroc_baseline_bucket():
    """DataFrame with only neutral B_zroc values → only 'baseline' bucket in result."""
    df = _base_df(n=100)
    df["B_zroc"] = 0.3   # all neutral (abs <= 0.5)

    result = event_study_h6_zroc_term_structure(df, _HORIZONS)

    assert list(result.index) == ["baseline"], (
        f"Expected only 'baseline', got {list(result.index)}"
    )


# ---------------------------------------------------------------------------
# H7 — Early turn
# ---------------------------------------------------------------------------


def test_h7_early_turn_detects_events():
    """B_gap<0 and B_gap_acc>0 for 3 consecutive bars starting at row 5 → 1 event at row 7."""
    df = _base_df(n=100)
    df["B_gap"] = -1.0        # all bearish
    df["B_gap_acc"] = -0.1    # accelerating downward by default

    # Three consecutive bars of closing gap: rows 5, 6, 7
    df.loc[5, "B_gap_acc"] = 0.2
    df.loc[6, "B_gap_acc"] = 0.3
    df.loc[7, "B_gap_acc"] = 0.1

    event_df, summary = event_study_h7_early_turn(df, m_bars=3, horizon_cols=_HORIZONS)

    assert len(event_df) >= 1, f"Expected at least 1 event, got {len(event_df)}"
    assert summary["n_events"] >= 1
    assert "confirmed" in event_df.columns
    for h in _HORIZONS:
        assert h in event_df.columns


def test_h7_early_turn_false_start_rate():
    """Event where B_gap never crosses 0 in 20 bars → confirmed=False, false_start_rate=1.0."""
    df = _base_df(n=100)
    df["B_gap"] = -1.0        # stays negative always
    df["B_gap_acc"] = -0.1    # generally not closing

    # 3 consecutive acc bars starting at row 10
    for r in [10, 11, 12]:
        df.loc[r, "B_gap_acc"] = 0.5

    event_df, summary = event_study_h7_early_turn(df, m_bars=3, horizon_cols=_HORIZONS)

    assert len(event_df) >= 1, "Expected at least 1 event"
    # All events should be un-confirmed (B_gap stays < 0 throughout)
    assert not event_df["confirmed"].any(), (
        f"Expected all confirmed=False, got:\n{event_df['confirmed']}"
    )
    assert summary["false_start_rate"] == pytest.approx(1.0), (
        f"Expected false_start_rate=1.0, got {summary['false_start_rate']}"
    )


def test_h7_early_turn_confirmed():
    """Event where B_gap crosses 0 within 20 bars → confirmed=True, false_start_rate=0.0."""
    df = _base_df(n=100)
    df["B_gap"] = -1.0
    df["B_gap_acc"] = -0.1

    # 3 consecutive acc bars at rows 5, 6, 7 (event detected at row 7)
    for r in [5, 6, 7]:
        df.loc[r, "B_gap_acc"] = 0.5

    # B_gap crosses 0 at row 15 (within 20 bars of event at row 7: rows 8..27)
    df.loc[15, "B_gap"] = 0.5

    event_df, summary = event_study_h7_early_turn(df, m_bars=3, horizon_cols=_HORIZONS)

    assert len(event_df) >= 1, "Expected at least 1 event"
    # At least the event at row 7 should be confirmed
    assert event_df["confirmed"].any(), (
        f"Expected at least one confirmed=True:\n{event_df['confirmed']}"
    )
    assert summary["false_start_rate"] == pytest.approx(0.0), (
        f"Expected false_start_rate=0.0, got {summary['false_start_rate']}"
    )
