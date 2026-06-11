"""
Phase 4 Orchestration — run_phase4.py
======================================
Orchestrates all Phase 4 (inter-timeframe) analyses on master_5m.parquet and
writes CSV outputs, PNG plots, and the Markdown report
reports/phase4_inter_tf.md.

Usage
-----
    python src/phase4_inter/run_phase4.py

or programmatically:
    from src.phase4_inter.run_phase4 import run_phase4
    run_phase4()
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure the project root is on sys.path so that `src.*` imports work whether
# the script is invoked with `python src/phase4_inter/run_phase4.py` or via
# `python -m src.phase4_inter.run_phase4`.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.phase3_intra.run_phase3 import extract_tf_df
from src.phase3_intra.targets import add_forward_returns
from src.phase4_inter.analysis_4a import stratify_by_htf_state, STATE_ORDER
from src.phase4_inter.analysis_4b import (
    find_zero_cross_events,
    cross_tf_followthrough,
    transition_summary,
)
from src.phase4_inter.analysis_4c import (
    compute_confluence_score_bull,
    compute_confluence_score_bull_accel,
    score_bucketed_returns as score_bucketed_returns_bull,
    plot_score_monotonicity,
)
from src.phase4_inter.analysis_4c_compression import (
    detect_compression_expansion,
    event_study_h13,
)
from src.phase4_inter.analysis_4d import (
    compute_confluence_score_bear,
    score_bucketed_returns as score_bucketed_returns_bear,
    compare_bull_bear_asymmetry,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MASTER = Path("data/features/master_5m.parquet")

HORIZON_COLS = ["ret_fwd_3", "ret_fwd_6", "ret_fwd_12"]
HAC_LAGS = 12  # all returns measured on the 5m grid throughout Phase 4

REPORTS_DIR = Path("reports")
REPORTS_4 = REPORTS_DIR / "phase4"
FIGS_4 = REPORTS_DIR / "figs" / "phase4"

# 4A: 5m setups (top Phase 3 findings) -> (state column, target value)
SETUPS_4A: dict[str, tuple[str, str]] = {
    "A_bull_accel": ("A_state", "bull_accel"),
    "A_bear_accel": ("A_state", "bear_accel"),
    "B_bull_accel": ("B_state", "bull_accel"),
}

SETUP_DESCRIPTIONS: dict[str, str] = {
    "A_bull_accel": "5m `A_state == bull_accel` (H1 core setup)",
    "A_bear_accel": "5m `A_state == bear_accel` (most robust A-state finding in Phase 3)",
    "B_bull_accel": "5m `B_state == bull_accel` (H5 core setup)",
}

# 4A: HTF state columns to stratify each setup by -> (label, state col, bar_close col)
HTF_STATE_COLS_4A: list[tuple[str, str, str]] = [
    ("1h_A_state", "A_state_1h", "bar_close_1h"),
    ("1h_B_state", "B_state_1h", "bar_close_1h"),
    ("1d_A_state", "A_state_1d", "bar_close_1d"),
    ("1d_B_state", "B_state_1d", "bar_close_1d"),
]

# 4B: lower-TF -> higher-TF lead-lag pairs and lookahead window (in higher-TF bars)
LEAD_LAG_PAIRS: list[tuple[str, str]] = [("5m", "15m"), ("15m", "1h"), ("1h", "1d")]
LEAD_LAG_K = 4

INDICATOR_LABELS_4B: dict[str, str] = {
    "A_hist_zero_cross": "Indicator A (`A_hist` zero-cross)",
    "B_roc_zero_cross": "Indicator B (`B_roc` zero-cross)",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _zero_cross_flag(series: pd.Series) -> pd.Series:
    """Pine-style strict zero cross: +1 when prev < 0 AND current > 0, -1 inverse."""
    prev = series.shift(1)
    flag = pd.Series(0, index=series.index, dtype=int)
    flag[(prev < 0) & (series > 0)] = 1
    flag[(prev > 0) & (series < 0)] = -1
    return flag


def _df_to_md(df: pd.DataFrame | None, max_rows: int = 20) -> str:
    """Convert a DataFrame to a compact Markdown table string."""
    if df is None or df.empty:
        return "_No data._\n"
    display = df.head(max_rows)
    return display.to_markdown(floatfmt=".6f") + "\n"


def _fmt(val: object, decimals: int = 6) -> str:
    """Format a float (or other) value as a string."""
    if isinstance(val, float):
        if val != val:  # NaN check
            return "NaN"
        return f"{val:.{decimals}f}"
    return str(val)


def _pivot(table: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot a (score, horizon)-MultiIndex table to rows=score, cols=horizon."""
    if table is None or table.empty:
        return pd.DataFrame()
    pivot = table[value_col].unstack(level="horizon")
    cols = [h for h in HORIZON_COLS if h in pivot.columns]
    return pivot[cols]


# ---------------------------------------------------------------------------
# 4A — Higher-TF regime filter (H11)
# ---------------------------------------------------------------------------

def _run_4a(master: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    results: dict[str, dict[str, pd.DataFrame]] = {}

    for setup_name, (col, value) in SETUPS_4A.items():
        setup_mask = master[col] == value
        results[setup_name] = {}

        for htf_label, htf_state_col, htf_bar_close_col in HTF_STATE_COLS_4A:
            table = stratify_by_htf_state(
                master, setup_mask, htf_state_col, htf_bar_close_col,
                HORIZON_COLS, hac_lags=HAC_LAGS,
            )
            table.to_csv(REPORTS_4 / f"4a_{setup_name}_{htf_label}.csv")
            results[setup_name][htf_label] = table

    return results


# ---------------------------------------------------------------------------
# 4B — Cross-TF lead-lag (T4.5-T4.8)
# ---------------------------------------------------------------------------

def _run_4b(master: pd.DataFrame) -> dict:
    tf_dfs: dict[str, pd.DataFrame] = {}
    for tf in ["5m", "15m", "1h", "1d"]:
        df = extract_tf_df(master, tf).dropna(subset=["bar_close"]).reset_index(drop=True)
        df["B_roc_zero_cross"] = _zero_cross_flag(df["B_roc"])
        tf_dfs[tf] = df

    # T4.5 — explicit event list (timestamp, direction) for 15m A_hist zero-crosses
    events_15m_a = find_zero_cross_events(tf_dfs["15m"], "A_hist_zero_cross")
    events_15m_a.to_csv(REPORTS_4 / "4b_events_15m_A_hist_zero_cross.csv", index=False)

    summaries: dict[str, dict[str, pd.DataFrame]] = {ind: {} for ind in INDICATOR_LABELS_4B}

    for indicator in INDICATOR_LABELS_4B:
        for lower_tf, higher_tf in LEAD_LAG_PAIRS:
            lower_df = tf_dfs[lower_tf]
            higher_df = tf_dfs[higher_tf]

            events = cross_tf_followthrough(
                lower_df, higher_df, indicator, indicator, k_bars=LEAD_LAG_K,
            )
            events.to_csv(
                REPORTS_4 / f"4b_events_{indicator}_{lower_tf}_to_{higher_tf}.csv",
                index=False,
            )

            summary = transition_summary(events)
            summary.to_csv(REPORTS_4 / f"4b_summary_{indicator}_{lower_tf}_to_{higher_tf}.csv")
            summaries[indicator][f"{lower_tf}_to_{higher_tf}"] = summary

    return {"events_15m_a": events_15m_a, "summaries": summaries}


# ---------------------------------------------------------------------------
# 4C — Confluence scoring (H12)
# ---------------------------------------------------------------------------

def _run_4c(master: pd.DataFrame) -> dict:
    master["score_bull"] = compute_confluence_score_bull(master)
    master["score_bull_accel"] = compute_confluence_score_bull_accel(master)

    # T4.9 — distribution histograms
    for score_col, fname in [
        ("score_bull", "score_bull_hist.png"),
        ("score_bull_accel", "score_bull_accel_hist.png"),
    ]:
        counts = master[score_col].value_counts().sort_index()
        counts.to_csv(REPORTS_4 / f"4c_{score_col}_distribution.csv", header=["count"])

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(counts.index.astype(int), counts.values)
        ax.set_xlabel(score_col)
        ax.set_ylabel("count (5m bars)")
        ax.set_title(f"Distribution of {score_col}")
        ax.set_xticks(range(0, 9))
        fig.savefig(FIGS_4 / fname, dpi=100, bbox_inches="tight")
        plt.close(fig)

    bull_table = score_bucketed_returns_bull(master, "score_bull", HORIZON_COLS, hac_lags=HAC_LAGS)
    bull_table.to_csv(REPORTS_4 / "4c_score_bull_bucketed.csv")
    plot_score_monotonicity(
        bull_table, HORIZON_COLS, "score_bull vs mean forward return",
        FIGS_4 / "score_bull_monotonicity.png",
    )

    bull_accel_table = score_bucketed_returns_bull(
        master, "score_bull_accel", HORIZON_COLS, hac_lags=HAC_LAGS,
    )
    bull_accel_table.to_csv(REPORTS_4 / "4c_score_bull_accel_bucketed.csv")
    plot_score_monotonicity(
        bull_accel_table, HORIZON_COLS, "score_bull_accel vs mean forward return",
        FIGS_4 / "score_bull_accel_monotonicity.png",
    )

    return {"bull_table": bull_table, "bull_accel_table": bull_accel_table}


# ---------------------------------------------------------------------------
# 4C (cont.) — Compression -> expansion event study (H13, T4.12b)
# ---------------------------------------------------------------------------

def _run_4c_compression(master: pd.DataFrame) -> dict:
    results: dict[str, dict] = {}

    for tf, htf_bar_close_col in [("15m", "bar_close_15m"), ("1h", "bar_close_1h")]:
        tf_df = (
            extract_tf_df(master, tf)
            .dropna(subset=["bar_close", "B_gap_norm", "B_zroc"])
            .reset_index(drop=True)
        )
        event_df = detect_compression_expansion(tf_df)
        event_df.to_csv(REPORTS_4 / f"4c_compression_events_{tf}.csv", index=False)

        study = event_study_h13(master, event_df, htf_bar_close_col, HORIZON_COLS)

        study_df = pd.DataFrame({
            "signed_ret": study["signed_ret"],
            "abs_ret_event": study["abs_ret_event"],
            "abs_ret_baseline": study["abs_ret_baseline"],
        })
        study_df.index.name = "horizon"
        study_df.to_csv(REPORTS_4 / f"4c_compression_study_{tf}.csv")

        results[tf] = {"event_df": event_df, "study": study, "study_df": study_df}

    return results


# ---------------------------------------------------------------------------
# 4D — Asymmetry (T4.13-T4.14)
# ---------------------------------------------------------------------------

def _run_4d(master: pd.DataFrame, bull_table: pd.DataFrame) -> dict:
    master["score_bear"] = compute_confluence_score_bear(master)

    bear_table = score_bucketed_returns_bear(master, "score_bear", HORIZON_COLS, hac_lags=HAC_LAGS)
    bear_table.to_csv(REPORTS_4 / "4d_score_bear_bucketed.csv")

    asymmetry = compare_bull_bear_asymmetry(bull_table, bear_table, HORIZON_COLS)
    asymmetry.to_csv(REPORTS_4 / "4d_bull_bear_asymmetry.csv")

    return {"bear_table": bear_table, "asymmetry": asymmetry}


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _section_4a(results_4a: dict[str, dict[str, pd.DataFrame]]) -> list[str]:
    lines = ["## 4A — Higher-TF Regime Filter (H11)\n"]
    lines.append(
        "Tests **H11**: do 5m setups perform better when the higher-TF regime "
        "(1h/1d `A_state`/`B_state`) is aligned vs. counter-aligned? Each table "
        "below shows `ret_fwd_12` statistics for the 5m setup, stratified by the "
        "higher-TF state at the time of the setup. `n_htf` is the **effective "
        "sample size** (unique higher-TF bars) per the L4 caveat above -- always "
        "<= n_5m, often dramatically so for 1d. Full tables for all horizons "
        "{3,6,12} are saved under `reports/phase4/4a_*.csv`.\n"
    )

    for setup_name, htf_tables in results_4a.items():
        lines.append(f"### Setup: {SETUP_DESCRIPTIONS.get(setup_name, setup_name)}\n")
        for htf_label, table in htf_tables.items():
            lines.append(f"**Stratified by {htf_label}** (ret_fwd_12):\n")
            if table.empty:
                lines.append("_No data._\n")
                continue
            try:
                sub = table.xs("ret_fwd_12", level="horizon")
                order = [g for g in STATE_ORDER + ["unconditional"] if g in sub.index]
                sub = sub.reindex(order)
                lines.append(_df_to_md(sub))
            except KeyError:
                lines.append("_No data._\n")

    return lines


def _section_4b(results_4b: dict) -> list[str]:
    lines = ["## 4B — Cross-TF Lead-Lag (T4.5-T4.8)\n"]
    lines.append(
        f"For each lower-TF zero-cross flip, did the higher-TF indicator cross "
        f"the same direction within the next K={LEAD_LAG_K} higher-TF bars "
        f"(strictly future bars only -- no-look-ahead)? `direction = +1` is a "
        f"bullish flip (cross from negative to positive), `-1` is bearish. "
        f"`p_followthrough` is the fraction of flips followed through; "
        f"`median_lag_bars` is the median lag (in higher-TF bars) among "
        f"followed-through flips.\n"
    )

    lines.append("### T4.5 — Sample of 15m `A_hist` zero-cross events\n")
    events_15m_a = results_4b["events_15m_a"]
    lines.append(_df_to_md(events_15m_a, max_rows=10))
    lines.append(
        f"_(n={len(events_15m_a)} total events; full list saved to "
        f"`reports/phase4/4b_events_15m_A_hist_zero_cross.csv`)_\n"
    )

    for indicator, label in INDICATOR_LABELS_4B.items():
        lines.append(f"### {label}\n")
        for lower_tf, higher_tf in LEAD_LAG_PAIRS:
            key = f"{lower_tf}_to_{higher_tf}"
            summary = results_4b["summaries"][indicator][key]
            lines.append(f"**{lower_tf} -> {higher_tf}** (K={LEAD_LAG_K}):\n")
            lines.append(_df_to_md(summary))

    return lines


def _section_4c(results_4c: dict) -> list[str]:
    lines = ["## 4C — Confluence Scoring (H12)\n"]
    lines.append(
        "`score_bull` in [0,8] sums 1{indicator bullish} across the 8 (A/B x "
        "5m/15m/1h/1d) pairs (`A_hist > 0` or `B_roc > 0`; NaN -> not bullish, "
        "so the early ~21% of the dataset before 1d data is available is "
        "capped at 6). `score_bull_accel` is the stricter version requiring "
        "`state == bull_accel`. Distribution histograms saved to "
        "`reports/figs/phase4/score_bull*_hist.png`.\n"
    )

    sections = [
        ("score_bull", results_4c["bull_table"], "score_bull_monotonicity.png"),
        ("score_bull_accel", results_4c["bull_accel_table"], "score_bull_accel_monotonicity.png"),
    ]

    for score_name, table, png in sections:
        lines.append(f"### {score_name} — bucketed forward returns\n")
        if table.empty:
            lines.append("_No data._\n")
            continue

        lines.append("**Mean forward return:**\n")
        lines.append(_df_to_md(_pivot(table, "mean_ret")))
        lines.append("**Hit rate:**\n")
        lines.append(_df_to_md(_pivot(table, "hit_rate")))
        lines.append("**n (5m rows):**\n")
        lines.append(_df_to_md(_pivot(table, "n")))
        lines.append("**HAC t-stat:**\n")
        lines.append(_df_to_md(_pivot(table, "t_stat")))
        lines.append(f"![{score_name} monotonicity](figs/phase4/{png})\n")

        try:
            mr = table["mean_ret"].xs("ret_fwd_12", level="horizon").dropna()
            if len(mr) >= 3:
                corr = mr.index.to_series().astype(float).corr(mr, method="spearman")
                lines.append(
                    f"_Spearman corr(score, mean_ret @ ret_fwd_12) = {_fmt(corr, 3)}_\n"
                )
        except KeyError:
            pass

    return lines


def _section_4c_compression(results_comp: dict[str, dict]) -> list[str]:
    lines = ["## 4C (cont.) — Compression -> Expansion (H13, T4.12b)\n"]
    lines.append(
        "Tests **H13**: on the 15m/1h frame, find bars where `|B_gap_norm|` is "
        "in its lowest decile (EMAs pinched = compression), then scan forward "
        "up to 10 bars for the first `|B_zroc| > 1.5` (expansion onset). Does "
        "the *sign* of the zROC expansion predict the sign of the subsequent "
        "5m move (`signed_ret`), and is `|move|` larger than the unconditional "
        "baseline (`abs_ret_event` vs `abs_ret_baseline`)? Note: the "
        "compression threshold is a fixed in-sample quantile, not walk-forward "
        "(descriptive event study only).\n"
    )

    for tf, data in results_comp.items():
        n_total = len(data["event_df"])
        study = data["study"]
        lines.append(f"### {tf} frame\n")
        lines.append(
            f"Compression->expansion events detected: {n_total}; "
            f"matched to 5m master (n_events): {study['n_events']}\n"
        )
        lines.append(_df_to_md(data["study_df"]))

    return lines


def _section_4d(results_4d: dict) -> list[str]:
    lines = ["## 4D — Asymmetry (T4.13-T4.14)\n"]
    lines.append(
        "`score_bear` in [0,8] is the symmetric bearish confluence score "
        "(`A_hist < 0` or `B_roc < 0` across the 8 pairs; NaN/0 -> not "
        "bearish). `abs_diff = |bull_mean_ret| - |bear_mean_ret|` at "
        "equivalent score levels: positive => the bullish edge is larger in "
        "magnitude at that confluence level; negative => the bearish edge "
        "dominates.\n"
    )

    bear_table = results_4d["bear_table"]
    lines.append("### score_bear — bucketed forward returns (mean_ret)\n")
    lines.append(_df_to_md(_pivot(bear_table, "mean_ret")))

    asym = results_4d["asymmetry"]
    lines.append("### Bull vs. Bear asymmetry (abs_diff = |bull| - |bear|)\n")
    if asym.empty:
        lines.append("_No overlapping score levels._\n")
    else:
        lines.append(_df_to_md(_pivot(asym, "abs_diff")))
        diffs = asym["abs_diff"].dropna()
        n_pos = int((diffs > 0).sum())
        n_neg = int((diffs < 0).sum())
        lines.append(
            f"_{n_pos} of {n_pos + n_neg} (score, horizon) cells favor the "
            f"bullish edge (abs_diff > 0); {n_neg} favor the bearish edge._\n"
        )

    return lines


def _build_summary(
    results_4a: dict, results_4c: dict, results_comp: dict, results_4d: dict,
) -> str:
    lines: list[str] = []

    # --- 4A headline: A_bull_accel stratified by 1h B_state -----------------
    try:
        table = results_4a["A_bull_accel"]["1h_B_state"]
        sub = table.xs("ret_fwd_12", level="horizon")
        uncond = float(sub.loc["unconditional", "mean_ret"])
        bull_aligned = float(sub.loc[["bull_accel", "bull_decel"], "mean_ret"].mean())
        bear_aligned = float(sub.loc[["bear_accel", "bear_decel"], "mean_ret"].mean())
        if bull_aligned > uncond and bull_aligned > bear_aligned:
            verdict = "Aligned 1h B regime strengthens the 5m A_bull_accel edge."
        elif bear_aligned > bull_aligned:
            verdict = "5m A_bull_accel edge is actually stronger when 1h B regime is bearish (counter to H11)."
        else:
            verdict = "No clear strengthening from 1h B-regime alignment."
        lines.append(
            f"- **4A (H11)**: 5m `A_state==bull_accel` unconditional ret_fwd_12 "
            f"mean = {_fmt(uncond)}. When 1h `B_state` is bullish "
            f"(bull_accel/bull_decel), mean = {_fmt(bull_aligned)}; when 1h "
            f"`B_state` is bearish (bear_accel/bear_decel), mean = "
            f"{_fmt(bear_aligned)}. {verdict}"
        )
    except (KeyError, TypeError):
        lines.append("- **4A (H11)**: insufficient data for headline comparison.")

    # --- 4C: H12 monotonicity -----------------------------------------------
    try:
        bull_table = results_4c["bull_table"]
        mr = bull_table["mean_ret"].xs("ret_fwd_12", level="horizon").dropna()
        corr = float(mr.index.to_series().astype(float).corr(mr, method="spearman"))
        if corr > 0.7:
            verdict = "monotonic (H12 confirmed)"
        elif corr > 0.3:
            verdict = "roughly monotonic (H12 partially confirmed)"
        else:
            verdict = "flat / non-monotonic (H12 not confirmed)"
        lines.append(
            f"- **4C (H12)**: Spearman corr(score_bull, mean_ret@12) = "
            f"{_fmt(corr, 3)} -> {verdict}."
        )
    except (KeyError, TypeError, ValueError):
        lines.append("- **4C (H12)**: insufficient data.")

    # --- 4C compression: H13 -------------------------------------------------
    for tf, data in results_comp.items():
        study = data["study"]
        if study["n_events"] > 0:
            sr = study["signed_ret"].get("ret_fwd_12", float("nan"))
            ar_e = study["abs_ret_event"].get("ret_fwd_12", float("nan"))
            ar_b = study["abs_ret_baseline"].get("ret_fwd_12", float("nan"))
            dir_verdict = (
                "expansion direction predicts the subsequent move (H13 directional part confirmed)"
                if sr == sr and sr > 0
                else "no directional edge (coin flip or worse)"
            )
            size_verdict = (
                "larger than baseline (H13 magnitude part confirmed)"
                if ar_e == ar_e and ar_b == ar_b and ar_e > ar_b
                else "not larger than baseline"
            )
            lines.append(
                f"- **4C/H13 ({tf})**: n_events={study['n_events']}, "
                f"signed_ret@12={_fmt(sr)} ({dir_verdict}); "
                f"abs_ret_event@12={_fmt(ar_e)} vs abs_ret_baseline@12={_fmt(ar_b)} "
                f"({size_verdict})."
            )
        else:
            lines.append(f"- **4C/H13 ({tf})**: no matched events.")

    # --- 4D asymmetry ----------------------------------------------------------
    try:
        asym = results_4d["asymmetry"]
        if not asym.empty:
            diffs = asym["abs_diff"].dropna()
            mean_diff = float(diffs.mean())
            verdict = (
                "bullish edges dominate on average"
                if mean_diff > 0
                else "bearish edges dominate on average"
            )
            lines.append(
                f"- **4D**: mean(abs_diff) across score/horizon cells = "
                f"{_fmt(mean_diff)} -> {verdict}."
            )
        else:
            lines.append("- **4D**: insufficient overlap between bull/bear score tables.")
    except (KeyError, TypeError):
        lines.append("- **4D**: insufficient data.")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main orchestration function
# ---------------------------------------------------------------------------

def run_phase4() -> None:
    """Run all Phase 4 analyses and write reports/phase4_inter_tf.md."""

    if not MASTER.exists():
        print("[phase4] master_5m.parquet not found -- run Phase 2 first")
        sys.exit(0)

    REPORTS_4.mkdir(parents=True, exist_ok=True)
    FIGS_4.mkdir(parents=True, exist_ok=True)

    print("[phase4] Loading master_5m.parquet ...")
    master = pd.read_parquet(MASTER)
    master_orig_cols = list(master.columns)
    master = add_forward_returns(master)
    print(f"[phase4] Loaded {len(master):,} rows x {len(master.columns)} columns")

    print("[phase4] Running 4A (HTF regime filter, H11) ...")
    results_4a = _run_4a(master)

    print("[phase4] Running 4B (cross-TF lead-lag) ...")
    results_4b = _run_4b(master)

    print("[phase4] Running 4C (confluence scoring, H12) ...")
    results_4c = _run_4c(master)

    print("[phase4] Running 4C compression-expansion event study (H13) ...")
    results_comp = _run_4c_compression(master)

    print("[phase4] Running 4D (asymmetry) ...")
    results_4d = _run_4d(master, results_4c["bull_table"])

    print("[phase4] Writing report ...")
    lines: list[str] = ["# Phase 4 — Inter-Timeframe Pattern Analysis\n"]
    lines.append(
        "## L4 Caveat — Effective Sample Size\n\n"
        "Higher-TF (1h, 1d) feature values are repeated across every 5m bar "
        "within that higher-TF bar (e.g. one 1h bar's state appears in ~12 "
        "consecutive 5m rows; one 1d bar's state appears in ~75 5m rows). "
        "Throughout this report, `n_5m` is the raw 5m row count and `n_htf` "
        "is the count of **unique** higher-TF bars -- the effective sample "
        "size for any HTF-conditional statistic. A cell with `n_5m=1000` but "
        "`n_htf=15` should be interpreted with the caution warranted by 15 "
        "independent observations, not 1000.\n"
    )

    lines.extend(_section_4a(results_4a))
    lines.extend(_section_4b(results_4b))
    lines.extend(_section_4c(results_4c))
    lines.extend(_section_4c_compression(results_comp))
    lines.extend(_section_4d(results_4d))

    lines.append("## Summary — Does higher-TF alignment strengthen 5m edges?\n")
    lines.append(_build_summary(results_4a, results_4c, results_comp, results_4d))

    report_path = REPORTS_DIR / "phase4_inter_tf.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[phase4] Report written: {report_path}")

    # Persist confluence score columns (T4.9/T4.10/T4.13 -- "column added to master")
    score_cols = ["score_bull", "score_bull_accel", "score_bear"]
    to_save = master[master_orig_cols + score_cols]
    to_save.to_parquet(MASTER, index=False)
    print(f"[phase4] {MASTER}: added columns {score_cols} "
          f"({to_save.shape[1]} cols total)")

    print("[phase4] Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_phase4()
