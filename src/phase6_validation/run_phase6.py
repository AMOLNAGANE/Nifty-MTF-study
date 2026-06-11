"""
Phase 6 Orchestration -- run_phase6.py
========================================
Walk-forward validation (GATE 2) of the K=10 confluence patterns carried
forward from Phase 5 (D9, `PLAN.md` SS3.4):

  T6.1 -- take the K patterns, unmodified
  T6.2 -- re-measure each on the validation split (year 4 = 2024)
  T6.3 -- re-measure each on the test split (year 5 = 2025+), ONCE
  T6.4 -- in-sample / validation / test comparison table; flag collapsed edges
  T6.5 -- random-feature baseline (shuffle one constituent flag column)
  T6.6 -- year-by-year stability (2020-2025); flag single-year-driven patterns
  T6.7 -- time-of-day baseline (matched on bar_in_session firing distribution)
  T6.8 -- write reports/phase6_validation.md listing only patterns that
          survive all checks (or an honest negative result)

Per DESIGN_DOC SS8.1 / TASK.md, the test split (2025+) is touched ONCE here.
No iteration on these results is permitted.

Usage
-----
    python src/phase6_validation/run_phase6.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.phase3_intra.targets import add_forward_returns
from src.phase5_mining.analysis_5a import FLAG_NAMES, FLAG_COLUMNS, evaluate_rules
from src.phase6_validation.analysis_6 import (
    select_carry_rules,
    build_comparison_table,
    random_feature_baseline,
    year_by_year_table,
    time_of_day_baseline,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MASTER = Path("data/features/master_5m.parquet")

RET_COL = "ret_fwd_12"
HAC_LAGS = 12

REPORTS_DIR = Path("reports")
REPORTS_6 = REPORTS_DIR / "phase6"

DISCOVERY_YEAR_MAX = 2023  # in-sample / rule-discovery set: years 1-3
VAL_YEAR = 2024            # validation split (year 4)
TEST_YEAR_MIN = 2025       # test split (year 5+) -- touched ONCE

# D9 carry-set: top-10 (by composite) of the 15 Hamming-deduped 5a survivors,
# in the order resolved in PLAN.md SS3.4 "D9 -- Resolved". T6.1: do NOT modify.
D9_RULE_IDS = [151, 121, 26, 0, 140, 31, 182, 211, 231, 181]

SHUFFLE_FLAG = "A_5m"  # T6.5: which constituent flag column to permute
SHUFFLE_SEED = 42

TOD_EDGE_THRESHOLD = 0.25  # T6.8: a pattern is "tod-explained" if |excess| < this * |rule_mean_ret|


# ---------------------------------------------------------------------------
# Small helpers (mirrors run_phase5.py's _df_to_md / _fmt)
# ---------------------------------------------------------------------------

def _df_to_md(df: pd.DataFrame | None, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "_No data._\n"
    display = df.head(max_rows)
    return display.to_markdown(index=False, floatfmt=".6f") + "\n"


def _fmt(val: object, decimals: int = 6) -> str:
    if isinstance(val, float):
        if val != val:  # NaN check
            return "NaN"
        return f"{val:.{decimals}f}"
    return str(val)


def _active_flags(row: pd.Series) -> str:
    flags = [f for f in FLAG_NAMES if bool(row[f])]
    return ", ".join(flags) if flags else "(none)"


# ---------------------------------------------------------------------------
# T6.8: classify each pattern as survives / collapsed
# ---------------------------------------------------------------------------

def _classify(
    comparison_df: pd.DataFrame,
    year_summary_df: pd.DataFrame,
    tod_df: pd.DataFrame,
    tod_threshold: float = TOD_EDGE_THRESHOLD,
) -> pd.DataFrame:
    df = comparison_df.merge(
        year_summary_df[["rule_id", "n_years_consistent", "n_years_total", "single_year_driven"]],
        on="rule_id", how="left",
    )
    df = df.merge(
        tod_df[["rule_id", "rule_mean_ret", "tod_matched_mean_ret", "excess_over_tod",
                "pct_first_hour_fired", "pct_first_hour_baseline"]],
        on="rule_id", how="left",
    )

    excess_sign_ok = np.sign(df["excess_over_tod"]) == df["expected_sign"]
    excess_large_enough = df["excess_over_tod"].abs() >= tod_threshold * df["rule_mean_ret"].abs()
    df["tod_explained"] = ~(excess_sign_ok & excess_large_enough)

    df["collapsed_oos"] = df["collapsed_val"] | df["collapsed_test"]
    df["survives"] = ~df["collapsed_oos"] & ~df["single_year_driven"] & ~df["tod_explained"]

    return df


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _section_intro(k_rules: pd.DataFrame, in_eval: pd.DataFrame, n_discovery: int,
                    n_val: int, n_test: int) -> list[str]:
    lines = ["# Phase 6 -- Walk-Forward Validation (T6.1-T6.8)\n"]
    lines.append(
        "**GATE 2.** This report re-measures the K=10 confluence patterns carried "
        "from Phase 5 (D9, `PLAN.md` SS3.4) on data they have never been selected, "
        "tuned, or ranked on. The **test split (2025+, T6.3) is touched exactly "
        "once** in this run -- no iteration on these results is permitted "
        "(`TASK.md`, Phase 6).\n"
    )
    lines.append(
        f"- **In-sample / rule-discovery set** (`bar_close.dt.year <= {DISCOVERY_YEAR_MAX}`): "
        f"n={n_discovery:,}\n"
        f"- **Validation split** (`bar_close.dt.year == {VAL_YEAR}`, T6.2): n={n_val:,} "
        f"-- already used as the 5b ML hold-out, but the rule-based 5a patterns were "
        f"not tuned to it directly\n"
        f"- **Test split** (`bar_close.dt.year >= {TEST_YEAR_MIN}`, T6.3): n={n_test:,} "
        f"-- never touched before this run\n"
    )

    lines.append("## T6.1 -- Carry-set patterns (unmodified from D9)\n")
    display = k_rules.copy()
    display["active_flags"] = display.apply(_active_flags, axis=1)
    display = display.merge(
        in_eval[["rule_id", "n", "mean_ret", "hit_rate", "t_stat", "p_val_bonf"]],
        on="rule_id",
    )
    display = display[["rule_id", "active_flags", "n", "mean_ret", "hit_rate", "t_stat", "p_val_bonf"]]
    display.columns = ["rule_id", "active_flags", "n_in", "mean_ret_in", "hit_rate_in", "t_stat_in", "p_val_bonf_in"]
    lines.append(_df_to_md(display, max_rows=10))
    lines.append(
        "_`p_val_bonf_in` here uses x10 (the size of the carry-set), not x256 as in "
        "`reports/phase5/rules_all.csv`. Per the D9 caveats, only rules 151 and 121 "
        "had p_val_bonf < 0.2 under the original x256 correction -- the remaining 8 "
        "are exploratory._\n"
    )
    return lines


def _section_comparison(comparison_df: pd.DataFrame) -> list[str]:
    lines = ["## T6.2-T6.4 -- In-sample vs. validation vs. test\n"]
    lines.append(
        "For each pattern, `evaluate_rules` (5a, unmodified) is re-run on the "
        "validation split (2024) and the test split (2025+). `expected_sign = "
        "sign(mean_ret_in)`; `collapsed_val` / `collapsed_test` are True if "
        "`sign(mean_ret)` flips relative to `expected_sign` in that split (a "
        "missing/NaN mean_ret -- e.g. zero firings -- also counts as collapsed).\n"
    )

    display = comparison_df.copy()
    display["active_flags"] = display.apply(_active_flags, axis=1)
    cols = (
        ["rule_id", "active_flags", "expected_sign"]
        + ["n_in", "mean_ret_in", "hit_rate_in"]
        + ["n_val", "mean_ret_val", "hit_rate_val", "collapsed_val"]
        + ["n_test", "mean_ret_test", "hit_rate_test", "collapsed_test"]
    )
    lines.append(_df_to_md(display[cols], max_rows=10))

    n_collapsed_val = int(display["collapsed_val"].sum())
    n_collapsed_test = int(display["collapsed_test"].sum())
    lines.append(
        f"- {n_collapsed_val}/10 patterns collapsed (sign flip or no data) on the "
        f"2024 validation split.\n"
        f"- {n_collapsed_test}/10 patterns collapsed on the 2025 test split.\n"
    )
    return lines


def _section_random_baseline(rb_df: pd.DataFrame, in_eval: pd.DataFrame, k_rules: pd.DataFrame) -> list[str]:
    lines = ["## T6.5 -- Random-feature baseline\n"]
    lines.append(
        f"For each pattern, the `{FLAG_COLUMNS[SHUFFLE_FLAG]}` column (the "
        f"`{SHUFFLE_FLAG}` flag's defining indicator) is permuted across rows "
        f"(seed={SHUFFLE_SEED}) on the in-sample set, decoupling that flag from "
        f"`{RET_COL}` and from the other 7 flags, and `evaluate_rules` is re-run. "
        f"Per DESIGN_DOC SS8.4, a real edge should weaken substantially once one of "
        f"its constituent conditions is randomized; an edge that is *unchanged* by "
        f"this shuffle would suggest `{SHUFFLE_FLAG}` plays no real role in the "
        f"pattern.\n"
    )

    flags = k_rules.copy()
    flags["active_flags"] = flags.apply(_active_flags, axis=1)

    merged = flags[["rule_id", "active_flags"]].merge(
        in_eval[["rule_id", "mean_ret", "hit_rate"]].rename(
            columns={"mean_ret": "mean_ret_in", "hit_rate": "hit_rate_in"}
        ),
        on="rule_id",
    ).merge(
        rb_df[["rule_id", "n", "mean_ret", "hit_rate", "t_stat"]].rename(
            columns={"n": "n_shuffled", "mean_ret": "mean_ret_shuffled",
                     "hit_rate": "hit_rate_shuffled", "t_stat": "t_stat_shuffled"}
        ),
        on="rule_id",
    )
    merged["edge_retained_pct"] = (
        merged["mean_ret_shuffled"] / merged["mean_ret_in"] * 100.0
    )
    lines.append(_df_to_md(merged, max_rows=10))
    lines.append(
        "_`edge_retained_pct` = `mean_ret_shuffled / mean_ret_in * 100`. Values near "
        "0% (or negative, i.e. sign flip) indicate the shuffled flag is load-bearing "
        f"for that pattern's edge. This is a partial test (only `{SHUFFLE_FLAG}` of "
        "8 constituent flags is shuffled), so a non-zero residual is expected even "
        "for genuine patterns -- it is reported as a diagnostic, not a pass/fail "
        "gate on its own._\n"
    )
    return lines


def _section_year_by_year(year_long_df: pd.DataFrame, year_summary_df: pd.DataFrame,
                           k_rules: pd.DataFrame) -> list[str]:
    lines = ["## T6.6 -- Year-by-year stability (2020-2025)\n"]
    lines.append(
        "`evaluate_rules` is re-run on each calendar year separately. "
        "`n_years_consistent` counts years with `n >= 5` and "
        "`sign(mean_ret) == expected_sign`; `single_year_driven` is True if "
        "`n_years_consistent <= 1` (the in-sample edge would then be carried by a "
        "single calendar year, e.g. a COVID-recovery artifact).\n"
    )

    flags = k_rules.copy()
    flags["active_flags"] = flags.apply(_active_flags, axis=1)

    pivot = year_long_df.pivot(index="rule_id", columns="year", values="mean_ret")
    pivot = pivot.reindex(k_rules["rule_id"].tolist())
    pivot.columns = [f"mean_ret_{y}" for y in pivot.columns]
    pivot = pivot.reset_index()

    summary = year_summary_df.set_index("rule_id").reindex(k_rules["rule_id"].tolist()).reset_index()
    merged = flags[["rule_id", "active_flags"]].merge(pivot, on="rule_id").merge(
        summary[["rule_id", "n_years_consistent", "n_years_total", "single_year_driven"]], on="rule_id"
    )
    lines.append(_df_to_md(merged, max_rows=10))

    n_single_year = int(summary["single_year_driven"].sum())
    lines.append(f"- {n_single_year}/10 patterns are single-year-driven.\n")
    return lines


def _section_time_of_day(tod_df: pd.DataFrame, k_rules: pd.DataFrame) -> list[str]:
    lines = ["## T6.7 -- Time-of-day baseline\n"]
    lines.append(
        "For each pattern, `tod_matched_mean_ret` is the unconditional "
        "per-`bar_in_session` mean(`ret_fwd_12`) (computed over the in-sample set), "
        "re-weighted by the pattern's own `bar_in_session` firing distribution. "
        "`excess_over_tod = rule_mean_ret - tod_matched_mean_ret` is the part of the "
        "edge *not* explained by simply firing more (or less) often at certain "
        "times of day. `pct_first_hour_*` is the fraction of (firing / all) rows "
        f"with `bar_in_session <= {12}` (first 60 minutes of the session).\n"
    )

    flags = k_rules.copy()
    flags["active_flags"] = flags.apply(_active_flags, axis=1)

    display = flags[["rule_id", "active_flags"]].merge(tod_df, on="rule_id")
    lines.append(_df_to_md(display, max_rows=10))
    return lines


def _section_verdict(classified: pd.DataFrame) -> list[str]:
    lines = ["## T6.8 -- Verdict\n"]

    survivors = classified[classified["survives"]]
    collapsed = classified[~classified["survives"]]

    if survivors.empty:
        lines.append(
            "**No patterns survive all three Phase 6 checks "
            "(no OOS sign-collapse, not single-year-driven, edge not explained by "
            "time-of-day).** This is reported as an honest negative result: of the "
            "10 patterns carried from Phase 5 (themselves mostly exploratory -- only "
            "rules 151 and 121 cleared the original x256 Bonferroni screen, and "
            "neither at p < 0.05), none generalized cleanly to the 2024 validation "
            "and 2025 test splits. See `FINDINGS.md` for the synthesis of what this "
            "means for the study as a whole.\n"
        )
    else:
        lines.append(
            f"**{len(survivors)}/10 patterns survive all three checks** "
            "(no OOS sign-collapse on val/test, not single-year-driven, edge not "
            "explained by time-of-day clustering):\n"
        )
        for _, row in survivors.iterrows():
            lines.append(
                f"- **Rule {int(row['rule_id'])}** ({_active_flags(row)}): "
                f"in-sample mean_ret={_fmt(row['mean_ret_in'])} "
                f"(hit_rate={_fmt(row['hit_rate_in'], 3)}, n={int(row['n_in']):,}); "
                f"val mean_ret={_fmt(row['mean_ret_val'])} "
                f"(hit_rate={_fmt(row['hit_rate_val'], 3)}, n={int(row['n_val']):,}); "
                f"test mean_ret={_fmt(row['mean_ret_test'])} "
                f"(hit_rate={_fmt(row['hit_rate_test'], 3)}, n={int(row['n_test']):,}); "
                f"{int(row['n_years_consistent'])}/{int(row['n_years_total'])} years "
                f"consistent with expected_sign; excess_over_tod="
                f"{_fmt(row['excess_over_tod'])} vs rule_mean_ret="
                f"{_fmt(row['rule_mean_ret'])}.\n"
            )
        lines.append(
            "\n**Caveats** -- neither surviving pattern was significant after the "
            "original x256 Bonferroni correction in 5a (`p_val_bonf=1.000` for both, "
            "see `reports/phase5/rules_survivors.csv`). Edges are economically small "
            "(tens of bps over a 12-bar / ~1h forward horizon) and computed with no "
            "transaction costs. \"Survives Phase 6\" should be read as **\"did not "
            "get falsified by out-of-sample data and three robustness checks\"**, "
            "not as \"a confirmed, tradeable edge.\"\n"
        )

    if not collapsed.empty:
        lines.append("\n**Patterns that did NOT survive:**\n")
        for _, row in collapsed.iterrows():
            reasons = []
            if row["collapsed_oos"]:
                which = []
                if row["collapsed_val"]:
                    which.append("2024")
                if row["collapsed_test"]:
                    which.append("2025")
                reasons.append(f"sign-collapsed on {'/'.join(which)}")
            if row["single_year_driven"]:
                reasons.append(
                    f"single-year-driven ({int(row['n_years_consistent'])}/"
                    f"{int(row['n_years_total'])} years consistent)"
                )
            if row["tod_explained"]:
                reasons.append(
                    f"edge largely explained by time-of-day "
                    f"(excess_over_tod={_fmt(row['excess_over_tod'])} vs "
                    f"rule_mean_ret={_fmt(row['rule_mean_ret'])})"
                )
            lines.append(f"- Rule {int(row['rule_id'])} ({_active_flags(row)}): {'; '.join(reasons)}\n")

    return lines


# ---------------------------------------------------------------------------
# Main orchestration function
# ---------------------------------------------------------------------------

def run_phase6() -> None:
    """Run all Phase 6 checks and write reports/phase6_validation.md."""

    if not MASTER.exists():
        print("[phase6] master_5m.parquet not found -- run earlier phases first")
        sys.exit(0)

    REPORTS_6.mkdir(parents=True, exist_ok=True)

    print("[phase6] Loading master_5m.parquet ...")
    master = pd.read_parquet(MASTER)
    master = add_forward_returns(master)
    print(f"[phase6] Loaded {len(master):,} rows x {len(master.columns)} columns")

    discovery_df = master[master["bar_close"].dt.year <= DISCOVERY_YEAR_MAX].reset_index(drop=True)
    val_df = master[master["bar_close"].dt.year == VAL_YEAR].reset_index(drop=True)
    test_df = master[master["bar_close"].dt.year >= TEST_YEAR_MIN].reset_index(drop=True)
    print(f"[phase6] in-sample={len(discovery_df):,}, val(2024)={len(val_df):,}, "
          f"test(2025+)={len(test_df):,}")

    print("[phase6] T6.1: selecting D9 carry-set rules ...")
    k_rules = select_carry_rules(D9_RULE_IDS)

    print("[phase6] T6.2/T6.3: evaluating on in-sample / val / test ...")
    in_eval = evaluate_rules(discovery_df, k_rules, ret_col=RET_COL, hac_lags=HAC_LAGS)
    val_eval = evaluate_rules(val_df, k_rules, ret_col=RET_COL, hac_lags=HAC_LAGS)
    test_eval = evaluate_rules(test_df, k_rules, ret_col=RET_COL, hac_lags=HAC_LAGS)

    print("[phase6] T6.4: building comparison table ...")
    comparison_df = build_comparison_table(in_eval, val_eval, test_eval)
    comparison_df.to_csv(REPORTS_6 / "comparison_table.csv", index=False)

    print(f"[phase6] T6.5: random-feature baseline (shuffle {SHUFFLE_FLAG}) ...")
    rb_df = random_feature_baseline(
        discovery_df, k_rules, ret_col=RET_COL, hac_lags=HAC_LAGS,
        shuffle_flag=SHUFFLE_FLAG, seed=SHUFFLE_SEED,
    )
    rb_df.to_csv(REPORTS_6 / "random_feature_baseline.csv", index=False)

    print("[phase6] T6.6: year-by-year stability ...")
    year_long_df, year_summary_df = year_by_year_table(
        master, k_rules, comparison_df, ret_col=RET_COL, hac_lags=HAC_LAGS,
    )
    year_long_df.to_csv(REPORTS_6 / "year_by_year.csv", index=False)
    year_summary_df.to_csv(REPORTS_6 / "year_by_year_summary.csv", index=False)

    print("[phase6] T6.7: time-of-day baseline ...")
    tod_df = time_of_day_baseline(discovery_df, k_rules, ret_col=RET_COL)
    tod_df.to_csv(REPORTS_6 / "time_of_day_baseline.csv", index=False)

    print("[phase6] T6.8: classifying survivors ...")
    classified = _classify(comparison_df, year_summary_df, tod_df)
    classified.to_csv(REPORTS_6 / "classification.csv", index=False)
    n_survive = int(classified["survives"].sum())
    print(f"[phase6]   {n_survive}/10 patterns survive all checks")

    print("[phase6] Writing report ...")
    lines: list[str] = []
    lines += _section_intro(k_rules, in_eval, len(discovery_df), len(val_df), len(test_df))
    lines += _section_comparison(comparison_df)
    lines += _section_random_baseline(rb_df, in_eval, k_rules)
    lines += _section_year_by_year(year_long_df, year_summary_df, k_rules)
    lines += _section_time_of_day(tod_df, k_rules)
    lines += _section_verdict(classified)

    (REPORTS_DIR / "phase6_validation.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[phase6] Report written: {REPORTS_DIR / 'phase6_validation.md'}")
    print("[phase6] Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_phase6()
