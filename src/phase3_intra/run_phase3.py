"""
Phase 3 Orchestration — run_phase3.py
======================================
Orchestrates all Phase 3 (intra-timeframe) analyses across all 4 timeframes
(5m, 15m, 1h, 1d) and writes CSV outputs, PNG plots, and Markdown reports.

Usage
-----
    python src/phase3_intra/run_phase3.py

or programmatically:
    from src.phase3_intra.run_phase3 import run_phase3
    run_phase3()
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd

# Ensure the project root is on sys.path so that `src.*` imports work whether
# the script is invoked with `python src/phase3_intra/run_phase3.py` or via
# `python -m src.phase3_intra.run_phase3`.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MASTER = Path("data/features/master_5m.parquet")

TFS = ["5m", "15m", "1h", "1d"]

HORIZON_COLS = ["ret_fwd_1", "ret_fwd_3", "ret_fwd_6", "ret_fwd_12", "ret_fwd_24"]

INDICATOR_A_NUMERIC = ["A_hist", "A_macd", "A_signal", "A_hist_slope"]
INDICATOR_B_NUMERIC = ["B_gap", "B_gap_norm", "B_roc", "B_zroc", "B_gap_acc", "B_roc_slope"]

# HAC lags per TF (approximately 1 session worth of bars, scaled down)
HAC_LAGS: dict[str, int] = {
    "5m": 12,
    "15m": 8,
    "1h": 6,
    "1d": 4,
}

# Output directories
REPORTS_DIR = Path("reports")
REPORTS_3A = REPORTS_DIR / "phase3a"
FIGS_3A = REPORTS_DIR / "figs" / "phase3a"


# ---------------------------------------------------------------------------
# Helper: extract TF-specific DataFrame from master
# ---------------------------------------------------------------------------

def extract_tf_df(master: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Extract a per-TF DataFrame from the merged master DataFrame.

    For tf="5m": returns the native 5m columns unchanged.
    For tf in ["15m","1h","1d"]: strips the _{tf} suffix from HTF columns,
    deduplicates by bar_close_{tf} (keep first = oldest 5m row per HTF bar),
    and returns a DataFrame with standard column names.
    """
    if tf == "5m":
        # Native 5m columns — no suffix stripping needed
        native_cols = [
            c for c in master.columns
            if not any(c.endswith(f"_{t}") for t in ["15m", "1h", "1d"])
        ]
        return master[native_cols].copy().reset_index(drop=True)

    suffix = f"_{tf}"
    bar_close_col = f"bar_close{suffix}"

    # All columns that carry the _{tf} suffix
    htf_cols = [c for c in master.columns if c.endswith(suffix)]

    if not htf_cols:
        raise ValueError(f"No columns found with suffix '{suffix}' in master DataFrame.")

    # Build mapping: original_name → stripped_name
    rename_map: dict[str, str] = {}
    for c in htf_cols:
        if c == bar_close_col:
            rename_map[c] = "bar_close"
        else:
            rename_map[c] = c[: -len(suffix)]

    # Keep also the 5m close (needed for regime labels and barrier computation)
    cols_to_extract = htf_cols + ["close"]
    # Avoid duplicate 'close' if somehow a HTF column also stripped to 'close'
    cols_to_extract = list(dict.fromkeys(cols_to_extract))

    df = master[cols_to_extract].copy()
    df.rename(columns=rename_map, inplace=True)

    # Deduplicate: drop rows that belong to the same HTF bar (keep first)
    df.drop_duplicates(subset=["bar_close"], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ---------------------------------------------------------------------------
# Markdown-formatting helpers
# ---------------------------------------------------------------------------

def _df_to_md(df: Optional[pd.DataFrame], max_rows: int = 20) -> str:
    """Convert a DataFrame to a compact Markdown table string."""
    if df is None or df.empty:
        return "_No data._\n"
    # Truncate if very long
    display = df.head(max_rows)
    return display.to_markdown(floatfmt=".6f") + "\n"


def _fmt(val: object, decimals: int = 6) -> str:
    """Format a float (or other) value as a string."""
    if isinstance(val, float):
        if val != val:  # NaN check
            return "NaN"
        return f"{val:.{decimals}f}"
    return str(val)


def _safe_float(val: object) -> float:
    """Return float or NaN for non-numeric."""
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# Report generation helpers
# ---------------------------------------------------------------------------

def _build_tf_report(
    tf: str,
    pearson_df: Optional[pd.DataFrame],
    spearman_df: Optional[pd.DataFrame],
    a_state_table: Optional[pd.DataFrame],
    b_state_table: Optional[pd.DataFrame],
    h2_summary: Optional[dict],
    h3_df: Optional[pd.DataFrame],
    h4_df: Optional[pd.DataFrame],
    h6_df: Optional[pd.DataFrame],
    h7_summary: Optional[dict],
    xcorr: Optional[pd.Series],
    nested_reg: Optional[dict],
    turn_summary: Optional[dict],
    joint_matrix: Optional[pd.DataFrame],
    joint_sig: Optional[dict],
    div_returns: Optional[pd.DataFrame],
    div_df: Optional[pd.DataFrame],
) -> str:
    lines: list[str] = []
    lines.append(f"# Phase 3 — Intra-Timeframe Analysis: {tf}\n")

    # ------------------------------------------------------------------ 3a
    lines.append("## 3a — Indicator vs. Price\n")

    lines.append("### Correlations\n")
    if pearson_df is not None and not pearson_df.empty:
        lines.append("**Pearson:**\n")
        lines.append(_df_to_md(pearson_df))
    else:
        lines.append("_Pearson correlations not available._\n")

    if spearman_df is not None and not spearman_df.empty:
        lines.append("**Spearman:**\n")
        lines.append(_df_to_md(spearman_df))
    else:
        lines.append("_Spearman correlations not available._\n")

    lines.append("### A_state Conditional Returns (H1)\n")
    lines.append(_df_to_md(a_state_table))

    lines.append("### B_state Conditional Returns (H5)\n")
    lines.append(_df_to_md(b_state_table))

    # H2
    lines.append("### H2 — Peak Timing (bull_accel → bull_decel)\n")
    if h2_summary:
        n_ev = h2_summary.get("n_events", 0)
        mean_fwd = h2_summary.get("mean_fwd_ret", {})
        svl = h2_summary.get("short_vs_long", float("nan"))
        fwd_str = ", ".join(f"{k}: {_fmt(v)}" for k, v in mean_fwd.items())
        lines.append(f"n_events: {n_ev}, mean_fwd_ret: {{{fwd_str}}}, short_vs_long: {_fmt(svl)}\n")
    else:
        lines.append("_Not computed._\n")

    # H3
    lines.append("### H3 — Aligned vs. Tired Zero-Cross\n")
    if h3_df is not None and not h3_df.empty:
        summary_h3 = h3_df.groupby("group")[HORIZON_COLS].mean()
        lines.append(_df_to_md(summary_h3))
    else:
        lines.append("_No zero-cross events found._\n")

    # H4
    lines.append("### H4 — First Pullback\n")
    if h4_df is not None and not h4_df.empty:
        summary_h4 = h4_df.groupby("group")[HORIZON_COLS].mean()
        lines.append(_df_to_md(summary_h4))
    else:
        lines.append("_No pullback events found._\n")

    # H6
    lines.append("### H6 — zROC Term Structure\n")
    lines.append(_df_to_md(h6_df))

    # H7
    lines.append("### H7 — Early Turn\n")
    if h7_summary:
        n_ev = h7_summary.get("n_events", 0)
        fsr = h7_summary.get("false_start_rate", float("nan"))
        lines.append(f"n_events: {n_ev}, false_start_rate: {_fmt(_safe_float(fsr) * 100, 2)}%\n")
    else:
        lines.append("_Not computed._\n")

    # ------------------------------------------------------------------ 3b
    lines.append("\n## 3b — Lead-Lag Analysis\n")

    lines.append("### Cross-Correlation (A_hist vs B_roc)\n")
    if xcorr is not None and not xcorr.empty:
        peak_lag = int(xcorr.abs().idxmax())
        peak_val = float(xcorr.iloc[xcorr.abs().argmax()])
        lines.append(
            f"Peak lag: {peak_lag} bars (negative = A leads B), "
            f"peak correlation: {_fmt(peak_val)}\n"
        )
    else:
        lines.append("_Not computed._\n")

    lines.append("### Nested Regression (H8): ret_fwd_12 ~ A_hist vs ~ A_hist + B_roc\n")
    if nested_reg:
        r2r = nested_reg.get("r2_restricted", float("nan"))
        r2u = nested_reg.get("r2_unrestricted", float("nan"))
        dr2 = nested_reg.get("delta_r2", float("nan"))
        bt = nested_reg.get("b_tstat", float("nan"))
        bp = nested_reg.get("b_pval", float("nan"))
        lines.append(
            f"R²_restricted: {_fmt(r2r)}, R²_unrestricted: {_fmt(r2u)}, "
            f"ΔR²: {_fmt(dr2)}, B coefficient t-stat: {_fmt(bt)} (p={_fmt(bp)})\n"
        )
    else:
        lines.append("_Not computed._\n")

    lines.append("### Turn Confirmation Timing (H10)\n")
    if turn_summary:
        n_fl = turn_summary.get("n_flips", 0)
        conf_rate = _safe_float(turn_summary.get("confirmation_rate", float("nan"))) * 100
        med_bars = turn_summary.get("median_bars_to_confirm", float("nan"))
        lines.append(
            f"n_flips: {n_fl}, confirmation_rate: {_fmt(conf_rate, 2)}%, "
            f"median_bars_to_confirm: {_fmt(med_bars, 1)}\n"
        )
    else:
        lines.append("_Not computed._\n")

    # ------------------------------------------------------------------ 3c
    lines.append("\n## 3c — A×B Agreement Matrix (H9)\n")
    if joint_matrix is not None and not joint_matrix.empty:
        # Pivot mean_ret to a 4×4 readable table
        try:
            pivot = joint_matrix["mean_ret"].unstack(level="state_b")
            lines.append(_df_to_md(pivot))
        except Exception:
            lines.append(_df_to_md(joint_matrix))
    else:
        lines.append("_Not computed._\n")

    if joint_sig:
        chi2 = joint_sig.get("chi2_stat", float("nan"))
        chi2p = joint_sig.get("chi2_pval", float("nan"))
        diag = joint_sig.get("diagonal_mean_ret", float("nan"))
        offdiag = joint_sig.get("offdiag_mean_ret", float("nan"))
        lines.append(
            f"chi2_stat: {_fmt(chi2)}, chi2_pval: {_fmt(chi2p)}, "
            f"diagonal_mean_ret: {_fmt(diag)}, offdiag_mean_ret: {_fmt(offdiag)}\n"
        )

    # ------------------------------------------------------------------ 3d
    lines.append("\n## 3d — Divergence Analysis (H14)\n")
    if div_df is not None and not div_df.empty:
        n_bearish = int((div_df["div_type"] == "bearish").sum())
        n_bullish = int((div_df["div_type"] == "bullish").sum())
        lines.append(f"n_bearish: {n_bearish}, n_bullish: {n_bullish}\n")
    else:
        lines.append("n_bearish: 0, n_bullish: 0\n")

    if div_returns is not None and not div_returns.empty:
        div_summary = div_returns.groupby("div_type")[HORIZON_COLS].mean()
        lines.append("Mean forward returns by divergence type:\n")
        lines.append(_df_to_md(div_summary))
    else:
        lines.append("_No divergence events found._\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def _build_summary_report(all_results: dict) -> str:
    """Build a cross-TF summary Markdown report from collected per-TF results."""
    lines: list[str] = []
    lines.append("# Phase 3 — Cross-Timeframe Summary\n")

    lines.append("## Key Metric: A_state 'bull_accel' Mean Forward Return at 12 Bars\n")
    best_tf = None
    best_val = float("-inf")
    for tf in TFS:
        res = all_results.get(tf, {})
        a_state_table = res.get("a_state_table")
        if a_state_table is not None and not a_state_table.empty:
            try:
                val = float(a_state_table.loc[("bull_accel", "ret_fwd_12"), "mean"])
                lines.append(f"- {tf}: mean ret_fwd_12 (bull_accel) = {_fmt(val)}")
                if val > best_val:
                    best_val = val
                    best_tf = tf
            except (KeyError, TypeError):
                lines.append(f"- {tf}: N/A")
        else:
            lines.append(f"- {tf}: N/A")

    if best_tf:
        lines.append(f"\nHighest bull_accel mean fwd ret at 12 bars: **{best_tf}** ({_fmt(best_val)})\n")

    lines.append("\n## Strongest Divergence Signal by TF\n")
    best_div_tf = None
    best_div_val = float("-inf")
    for tf in TFS:
        res = all_results.get(tf, {})
        div_returns = res.get("div_returns")
        if div_returns is not None and not div_returns.empty:
            try:
                bullish_mask = div_returns["div_type"] == "bullish"
                if bullish_mask.any():
                    val = float(div_returns.loc[bullish_mask, "ret_fwd_12"].mean())
                    lines.append(f"- {tf}: bullish div mean ret_fwd_12 = {_fmt(val)}")
                    if val > best_div_val:
                        best_div_val = val
                        best_div_tf = tf
                else:
                    lines.append(f"- {tf}: no bullish divergence events")
            except (KeyError, TypeError):
                lines.append(f"- {tf}: N/A")
        else:
            lines.append(f"- {tf}: N/A")

    if best_div_tf:
        lines.append(f"\nStrongest divergence signal: **{best_div_tf}** (bullish div ret_fwd_12 = {_fmt(best_div_val)})\n")

    lines.append("\n## Cross-TF Comparison: Key Metrics\n")
    rows = []
    for tf in TFS:
        res = all_results.get(tf, {})
        row: dict = {"TF": tf}

        # A_state bull_accel ret_fwd_12
        a_state_table = res.get("a_state_table")
        try:
            row["bull_accel_ret12"] = float(a_state_table.loc[("bull_accel", "ret_fwd_12"), "mean"])  # type: ignore[index]
        except (KeyError, TypeError, AttributeError):
            row["bull_accel_ret12"] = float("nan")

        # H2 n_events
        h2_summary = res.get("h2_summary")
        row["h2_n_events"] = h2_summary.get("n_events", 0) if h2_summary else 0

        # Cross-corr peak lag
        xcorr = res.get("xcorr")
        if xcorr is not None and not xcorr.empty:
            row["xcorr_peak_lag"] = int(xcorr.abs().idxmax())
        else:
            row["xcorr_peak_lag"] = float("nan")

        # Nested reg delta_r2
        nested_reg = res.get("nested_reg")
        row["nested_delta_r2"] = nested_reg.get("delta_r2", float("nan")) if nested_reg else float("nan")

        # Divergence bullish ret_fwd_12
        div_returns = res.get("div_returns")
        try:
            bullish_mask = div_returns["div_type"] == "bullish"  # type: ignore[index]
            row["div_bull_ret12"] = float(div_returns.loc[bullish_mask, "ret_fwd_12"].mean())  # type: ignore[index]
        except (KeyError, TypeError, AttributeError):
            row["div_bull_ret12"] = float("nan")

        rows.append(row)

    summary_table = pd.DataFrame(rows).set_index("TF")
    lines.append(_df_to_md(summary_table))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main orchestration function
# ---------------------------------------------------------------------------

def run_phase3() -> None:
    """Run all Phase 3 analyses across all 4 timeframes and write reports."""

    # ------------------------------------------------------------------ check
    if not MASTER.exists():
        print("[phase3] master_5m.parquet not found -- run Phase 2 first")
        sys.exit(0)

    # ------------------------------------------------------------------ setup
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_3A.mkdir(parents=True, exist_ok=True)
    FIGS_3A.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ load
    print("[phase3] Loading master_5m.parquet ...")
    master = pd.read_parquet(MASTER)
    print(f"[phase3] Loaded {len(master):,} rows x {len(master.columns)} columns")

    # Import analysis modules (once, outside per-TF loop)
    from src.phase3_intra.targets import (
        add_forward_returns,
        add_barrier_labels,
        add_regime_label,
    )
    from src.phase3_intra.analysis_3a import (
        compute_correlations,
        compute_decile_returns,
        compute_state_conditional_returns,
        apply_multiple_testing_correction,
        plot_decile_curves,
    )
    from src.phase3_intra.analysis_3a_events import (
        event_study_h2_peak_timing,
        event_study_h3_aligned_vs_tired,
        event_study_h4_pullback,
        event_study_h6_zroc_term_structure,
        event_study_h7_early_turn,
    )
    from src.phase3_intra.analysis_3b import (
        compute_cross_correlation,
        nested_regression_test,
        turn_confirmation_timing,
    )
    from src.phase3_intra.analysis_3c import (
        build_joint_state_matrix,
        test_joint_state_significance,
    )
    from src.phase3_intra.analysis_3d import detect_divergence, compute_divergence_returns

    # ------------------------------------------------------------------ per-TF
    all_results: dict = {}

    for tf in TFS:
        print(f"\n[phase3] -- TF: {tf} --------------------------")
        results: dict = {}

        try:
            # -------------------------------------------------------- targets
            print(f"[phase3] [{tf}] Extracting TF DataFrame ...")
            df_tf = extract_tf_df(master, tf)
            print(f"[phase3] [{tf}] Shape after extract: {df_tf.shape}")

            print(f"[phase3] [{tf}] Adding forward returns ...")
            df_tf = add_forward_returns(df_tf)

            print(f"[phase3] [{tf}] Adding barrier labels ...")
            df_tf = add_barrier_labels(df_tf, tf)

            print(f"[phase3] [{tf}] Adding regime labels ...")
            df_tf = add_regime_label(df_tf)

            hac_lags = HAC_LAGS[tf]

            # --------------------------------------------- 3a correlations
            print(f"[phase3] [{tf}] Computing correlations ...")
            numeric_features = [
                c for c in INDICATOR_A_NUMERIC + INDICATOR_B_NUMERIC
                if c in df_tf.columns
            ]
            present_horizons = [h for h in HORIZON_COLS if h in df_tf.columns]

            pearson_df, spearman_df = None, None
            if numeric_features and present_horizons:
                pearson_df, spearman_df = compute_correlations(
                    df_tf, numeric_features, present_horizons
                )
                pearson_df.to_csv(REPORTS_3A / f"correlations_pearson_{tf}.csv")
                spearman_df.to_csv(REPORTS_3A / f"correlations_spearman_{tf}.csv")
                print(f"[phase3] [{tf}] Saved pearson/spearman CSVs")

            # --------------------------------------------- 3a state tables
            print(f"[phase3] [{tf}] Computing A_state conditional returns ...")
            a_state_table = None
            if "A_state" in df_tf.columns:
                a_state_table = compute_state_conditional_returns(
                    df_tf, "A_state", present_horizons, hac_lags=hac_lags
                )
                a_state_table = apply_multiple_testing_correction(a_state_table)
                a_state_table.to_csv(REPORTS_3A / f"a_state_conditional_{tf}.csv")

            print(f"[phase3] [{tf}] Computing B_state conditional returns ...")
            b_state_table = None
            if "B_state" in df_tf.columns:
                b_state_table = compute_state_conditional_returns(
                    df_tf, "B_state", present_horizons, hac_lags=hac_lags
                )
                b_state_table = apply_multiple_testing_correction(b_state_table)
                b_state_table.to_csv(REPORTS_3A / f"b_state_conditional_{tf}.csv")

            # --------------------------------------------- 3a deciles
            print(f"[phase3] [{tf}] Computing decile returns ...")
            for feat in ["A_hist", "B_roc", "B_zroc"]:
                if feat in df_tf.columns and present_horizons:
                    try:
                        decile_df = compute_decile_returns(df_tf, feat, present_horizons)
                        if not decile_df.empty:
                            decile_df.to_csv(REPORTS_3A / f"decile_{tf}_{feat}.csv")
                            plot_decile_curves(decile_df, feat, tf, FIGS_3A)
                    except Exception as e:
                        print(f"[phase3] [{tf}] Warning: decile for {feat} failed: {e}")

            # --------------------------------------------- 3a events
            print(f"[phase3] [{tf}] Running event studies ...")

            # H2 — peak timing
            h2_event_df, h2_summary = None, None
            if "A_state" in df_tf.columns:
                try:
                    h2_event_df, h2_summary = event_study_h2_peak_timing(df_tf, present_horizons)
                    if h2_event_df is not None and not h2_event_df.empty:
                        h2_event_df.to_csv(REPORTS_3A / f"h2_peak_timing_{tf}.csv", index=False)
                except Exception as e:
                    print(f"[phase3] [{tf}] Warning: H2 failed: {e}")

            # H3 — aligned vs tired zero-cross
            h3_df = None
            if all(c in df_tf.columns for c in ["A_macd_zero_cross", "A_hist", "A_hist_slope"]):
                try:
                    h3_df = event_study_h3_aligned_vs_tired(df_tf, present_horizons)
                    if h3_df is not None and not h3_df.empty:
                        h3_df.to_csv(REPORTS_3A / f"h3_aligned_tired_{tf}.csv", index=False)
                except Exception as e:
                    print(f"[phase3] [{tf}] Warning: H3 failed: {e}")

            # H4 — first pullback
            h4_df = None
            if all(c in df_tf.columns for c in ["A_hist_zero_cross", "A_macd"]):
                try:
                    h4_df = event_study_h4_pullback(df_tf, present_horizons)
                    if h4_df is not None and not h4_df.empty:
                        h4_df.to_csv(REPORTS_3A / f"h4_pullback_{tf}.csv", index=False)
                except Exception as e:
                    print(f"[phase3] [{tf}] Warning: H4 failed: {e}")

            # H6 — zROC term structure
            h6_df = None
            if "B_zroc" in df_tf.columns:
                try:
                    h6_df = event_study_h6_zroc_term_structure(df_tf, present_horizons)
                    if h6_df is not None and not h6_df.empty:
                        h6_df.to_csv(REPORTS_3A / f"h6_zroc_term_{tf}.csv")
                except Exception as e:
                    print(f"[phase3] [{tf}] Warning: H6 failed: {e}")

            # H7 — early turn
            h7_event_df, h7_summary = None, None
            if all(c in df_tf.columns for c in ["B_gap", "B_gap_acc"]):
                try:
                    h7_event_df, h7_summary = event_study_h7_early_turn(
                        df_tf, horizon_cols=present_horizons
                    )
                    if h7_event_df is not None and not h7_event_df.empty:
                        h7_event_df.to_csv(REPORTS_3A / f"h7_early_turn_{tf}.csv", index=False)
                except Exception as e:
                    print(f"[phase3] [{tf}] Warning: H7 failed: {e}")

            # --------------------------------------------- 3b lead-lag
            print(f"[phase3] [{tf}] Running lead-lag analysis (3b) ...")

            xcorr: pd.Series | None = None
            if all(c in df_tf.columns for c in ["A_hist", "B_roc"]):
                try:
                    xcorr = compute_cross_correlation(df_tf, "A_hist", "B_roc", max_lag=20)
                    xcorr.to_csv(REPORTS_3A / f"xcorr_ahist_broc_{tf}.csv", header=True)
                except Exception as e:
                    print(f"[phase3] [{tf}] Warning: cross-correlation failed: {e}")

            nested_reg: dict | None = None
            if all(c in df_tf.columns for c in ["ret_fwd_12", "A_hist", "B_roc"]):
                try:
                    nested_reg = nested_regression_test(
                        df_tf, "ret_fwd_12", "A_hist", "B_roc", hac_lags=hac_lags
                    )
                    pd.Series(nested_reg).to_csv(
                        REPORTS_3A / f"nested_reg_{tf}.csv", header=False
                    )
                except Exception as e:
                    print(f"[phase3] [{tf}] Warning: nested regression failed: {e}")

            turn_events_df, turn_summary = None, None
            if all(c in df_tf.columns for c in ["A_macd_zero_cross", "B_roc"]):
                try:
                    turn_events_df, turn_summary = turn_confirmation_timing(
                        df_tf,
                        a_flip_col="A_macd_zero_cross",
                        b_confirm_col="B_roc",
                        b_confirm_threshold=0.0,
                        k_bars=20,
                    )
                    if turn_events_df is not None and not turn_events_df.empty:
                        turn_events_df.to_csv(
                            REPORTS_3A / f"turn_confirmation_{tf}.csv", index=False
                        )
                except Exception as e:
                    print(f"[phase3] [{tf}] Warning: turn confirmation failed: {e}")

            # --------------------------------------------- 3c agreement matrix
            print(f"[phase3] [{tf}] Building AxB joint state matrix (3c) ...")

            joint_matrix: pd.DataFrame | None = None
            joint_sig: dict | None = None
            if all(c in df_tf.columns for c in ["A_state", "B_state", "ret_fwd_12"]):
                try:
                    joint_matrix = build_joint_state_matrix(
                        df_tf, "A_state", "B_state", "ret_fwd_12", hac_lags=hac_lags
                    )
                    joint_matrix.to_csv(REPORTS_3A / f"joint_state_matrix_{tf}.csv")

                    joint_sig = test_joint_state_significance(joint_matrix)
                    pd.Series(joint_sig).to_csv(
                        REPORTS_3A / f"joint_state_sig_{tf}.csv", header=False
                    )
                except Exception as e:
                    print(f"[phase3] [{tf}] Warning: joint state matrix failed: {e}")

            # --------------------------------------------- 3d divergence
            print(f"[phase3] [{tf}] Running divergence analysis (3d) ...")

            div_df: pd.DataFrame | None = None
            div_returns: pd.DataFrame | None = None
            if all(c in df_tf.columns for c in ["close", "A_hist"]) and present_horizons:
                try:
                    div_df = detect_divergence(df_tf, "close", "A_hist", min_spacing=10)
                    if div_df is not None and not div_df.empty:
                        div_df.to_csv(REPORTS_3A / f"divergence_events_{tf}.csv", index=False)
                        div_returns = compute_divergence_returns(
                            df_tf, div_df, present_horizons
                        )
                        if div_returns is not None and not div_returns.empty:
                            div_returns.to_csv(
                                REPORTS_3A / f"divergence_returns_{tf}.csv", index=False
                            )
                except Exception as e:
                    print(f"[phase3] [{tf}] Warning: divergence analysis failed: {e}")

            # --------------------------------------------- store results
            results = {
                "df_tf": df_tf,
                "pearson_df": pearson_df,
                "spearman_df": spearman_df,
                "a_state_table": a_state_table,
                "b_state_table": b_state_table,
                "h2_summary": h2_summary,
                "h3_df": h3_df,
                "h4_df": h4_df,
                "h6_df": h6_df,
                "h7_summary": h7_summary,
                "xcorr": xcorr,
                "nested_reg": nested_reg,
                "turn_summary": turn_summary,
                "joint_matrix": joint_matrix,
                "joint_sig": joint_sig,
                "div_returns": div_returns,
                "div_df": div_df,
            }

            # -------------------------------------------- write TF markdown
            print(f"[phase3] [{tf}] Writing Markdown report ...")
            report_md = _build_tf_report(
                tf=tf,
                pearson_df=pearson_df,
                spearman_df=spearman_df,
                a_state_table=a_state_table,
                b_state_table=b_state_table,
                h2_summary=h2_summary,
                h3_df=h3_df,
                h4_df=h4_df,
                h6_df=h6_df,
                h7_summary=h7_summary,
                xcorr=xcorr,
                nested_reg=nested_reg,
                turn_summary=turn_summary,
                joint_matrix=joint_matrix,
                joint_sig=joint_sig,
                div_returns=div_returns,
                div_df=div_df,
            )
            report_path = REPORTS_DIR / f"phase3_intra_{tf}.md"
            report_path.write_text(report_md, encoding="utf-8")
            print(f"[phase3] [{tf}] Report written: {report_path}")

        except Exception as exc:
            print(f"[phase3] [{tf}] ERROR: {exc}")
            traceback.print_exc()

        all_results[tf] = results

    # ------------------------------------------------------------------ summary
    print("\n[phase3] Writing cross-TF summary report ...")
    summary_md = _build_summary_report(all_results)
    summary_path = REPORTS_DIR / "phase3_summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")
    print(f"[phase3] Summary written: {summary_path}")

    print("\n[phase3] Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_phase3()
