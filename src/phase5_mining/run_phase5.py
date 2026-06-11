"""
Phase 5 Orchestration — run_phase5.py
======================================
Confluence pattern mining on master_5m.parquet:
  5A — boolean rule enumeration / ranking / dedup (T5.1-T5.4)
  5B — LightGBM feature importance, SHAP, top splits (T5.5-T5.10)
  5C — failure / inversion analysis on the top survivor rules (T5.11-T5.14)

Writes CSVs under reports/phase5/, figures under reports/figs/phase5/, and
three Markdown reports:
  reports/phase5_rules_top20.md
  reports/phase5_ml_importance.md
  reports/phase5_failure_analysis.md

**Train/val/test boundary**: 5a/5c run on the rule-discovery set
(`bar_close.dt.year <= 2023`, years 1-3). 5b's walk-forward split is
train=`year<=2023` / hold-out=`year==2024`. Year 2025 is never used in any
Phase 5 computation -- it is reserved for Phase 6 out-of-sample validation.

Usage
-----
    python src/phase5_mining/run_phase5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Ensure the project root is on sys.path so that `src.*` imports work whether
# the script is invoked with `python src/phase5_mining/run_phase5.py` or via
# `python -m src.phase5_mining.run_phase5`.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.phase3_intra.targets import add_forward_returns
from src.phase5_mining.analysis_5a import (
    FLAG_NAMES,
    FLAG_COLUMNS,
    enumerate_rules,
    evaluate_rules,
    rank_rules,
    dedupe_rules,
)
from src.phase5_mining.analysis_5b import (
    feature_columns,
    prepare_features,
    class_balance,
    walk_forward_split,
    train_lgbm,
    evaluate_auc,
    permutation_importance_table,
    compute_shap_values,
    plot_shap_summary,
    plot_shap_dependence,
    top_splits,
)
from src.phase5_mining.analysis_5c import (
    split_worked_failed,
    feature_mean_diff,
    propose_filters,
    evaluate_filtered_rule,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MASTER = Path("data/features/master_5m.parquet")

RET_COL = "ret_fwd_12"
HAC_LAGS = 12  # all returns measured on the 5m grid

REPORTS_DIR = Path("reports")
REPORTS_5 = REPORTS_DIR / "phase5"
FIGS_5 = REPORTS_DIR / "figs" / "phase5"

DISCOVERY_YEAR_MAX = 2023  # 5a/5c rule-discovery set: years 1-3
VAL_YEAR = 2024            # 5b hold-out year

N_TOP_RULES = 20
MIN_HAMMING = 2
N_FAILURE_RULES = 5
FILTER_THRESHOLD = 0.5

_RULE_OWN_COLS = set(FLAG_COLUMNS.values())  # excluded from 5c feature comparisons


# ---------------------------------------------------------------------------
# Small helpers (mirrors run_phase4.py's _df_to_md / _fmt)
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


# ---------------------------------------------------------------------------
# 5A — Boolean rule enumeration (T5.1-T5.4)
# ---------------------------------------------------------------------------

def _run_5a(discovery_df: pd.DataFrame) -> dict:
    rules_df = enumerate_rules()
    evaluated = evaluate_rules(discovery_df, rules_df, ret_col=RET_COL, hac_lags=HAC_LAGS)
    evaluated.to_csv(REPORTS_5 / "rules_all.csv", index=False)

    top20, bottom20 = rank_rules(evaluated, n_top=N_TOP_RULES)
    top20.to_csv(REPORTS_5 / "rules_top20.csv", index=False)
    bottom20.to_csv(REPORTS_5 / "rules_bottom20.csv", index=False)

    survivors = dedupe_rules(top20, min_hamming=MIN_HAMMING)
    survivors.to_csv(REPORTS_5 / "rules_survivors.csv", index=False)

    return {
        "evaluated": evaluated,
        "top20": top20,
        "bottom20": bottom20,
        "survivors": survivors,
    }


# ---------------------------------------------------------------------------
# 5B — LightGBM feature importance (T5.5-T5.10)
# ---------------------------------------------------------------------------

def _run_5b(master: pd.DataFrame) -> dict:
    train_df, val_df = walk_forward_split(master)

    X_train, y_train = prepare_features(train_df, ret_col=RET_COL)
    X_val, y_val = prepare_features(val_df, ret_col=RET_COL)

    balance_train = class_balance(y_train)
    balance_val = class_balance(y_val)

    model = train_lgbm(X_train, y_train, seed=42)
    auc = evaluate_auc(model, X_val, y_val)

    perm_table = permutation_importance_table(model, X_val, y_val, seed=42, n_repeats=10, n_top=15)
    perm_table.to_csv(REPORTS_5 / "permutation_importance.csv", index=False)

    shap_values = compute_shap_values(model, X_val)
    plot_shap_summary(shap_values, X_val, FIGS_5 / "shap_summary.png", max_display=15)

    top_features = list(perm_table["feature"].head(2))
    feat1 = top_features[0]
    feat2 = top_features[1] if len(top_features) > 1 else top_features[0]
    plot_shap_dependence(shap_values, X_val, feat1, feat2, FIGS_5 / "shap_dependence.png")

    feat1_idx = list(X_val.columns).index(feat1)
    interaction_corr = float(
        pd.Series(shap_values[:, feat1_idx]).corr(X_val[feat2].reset_index(drop=True))
    )

    splits_df = top_splits(model, n=3)
    splits_df.to_csv(REPORTS_5 / "top_splits.csv", index=False)

    return {
        "train_range": (train_df["bar_close"].min(), train_df["bar_close"].max()),
        "val_range": (val_df["bar_close"].min(), val_df["bar_close"].max()),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "balance_train": balance_train,
        "balance_val": balance_val,
        "auc": auc,
        "perm_table": perm_table,
        "splits_df": splits_df,
        "feat1": feat1,
        "feat2": feat2,
        "interaction_corr": interaction_corr,
    }


# ---------------------------------------------------------------------------
# 5C — Failure / inversion analysis (T5.11-T5.14)
# ---------------------------------------------------------------------------

def _build_rule_mask(df: pd.DataFrame, rule: pd.Series) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for flag in FLAG_NAMES:
        col = FLAG_COLUMNS[flag]
        bull = (df[col] > 0).fillna(False)
        mask &= bull == bool(rule[flag])
    return mask


def _run_5c(
    discovery_df: pd.DataFrame, survivors: pd.DataFrame, X_discovery: pd.DataFrame
) -> list[dict]:
    feature_cols = [c for c in feature_columns() if c not in _RULE_OWN_COLS]

    results: list[dict] = []
    for _, rule in survivors.head(N_FAILURE_RULES).iterrows():
        rule_id = int(rule["rule_id"])
        expected_sign = 1 if rule["mean_ret"] > 0 else -1

        rule_mask = _build_rule_mask(discovery_df, rule)
        worked_df, failed_df, split_stats = split_worked_failed(
            discovery_df, rule_mask, RET_COL, expected_sign,
        )

        worked_X = X_discovery.loc[X_discovery.index.intersection(worked_df.index)]
        failed_X = X_discovery.loc[X_discovery.index.intersection(failed_df.index)]

        diff_df = feature_mean_diff(worked_X, failed_X, feature_cols)
        diff_df.to_csv(REPORTS_5 / f"failure_rule{rule_id}_diffs.csv", index=False)

        filters_df = propose_filters(diff_df, threshold=FILTER_THRESHOLD)

        before_after_records = []
        for _, filt in filters_df.iterrows():
            feat = filt["feature"]
            midpoint = float(filt["midpoint"])
            direction = filt["direction"]

            if direction == ">":
                filter_mask_X = X_discovery[feat] > midpoint
            else:
                filter_mask_X = X_discovery[feat] < midpoint
            filter_mask = filter_mask_X.reindex(discovery_df.index, fill_value=False)

            ba = evaluate_filtered_rule(discovery_df, rule_mask, filter_mask, RET_COL, hac_lags=HAC_LAGS)
            before_after_records.append({
                "feature": feat,
                "direction": direction,
                "midpoint": midpoint,
                "effect_size": float(filt["effect_size"]),
                "n_before": ba.loc["before", "n"],
                "mean_ret_before": ba.loc["before", "mean_ret"],
                "hit_rate_before": ba.loc["before", "hit_rate"],
                "t_stat_before": ba.loc["before", "t_stat"],
                "n_after": ba.loc["after", "n"],
                "mean_ret_after": ba.loc["after", "mean_ret"],
                "hit_rate_after": ba.loc["after", "hit_rate"],
                "t_stat_after": ba.loc["after", "t_stat"],
            })

        before_after_df = pd.DataFrame(before_after_records)
        before_after_df.to_csv(REPORTS_5 / f"failure_rule{rule_id}_filters_before_after.csv", index=False)

        results.append({
            "rule_id": rule_id,
            "rule": rule,
            "expected_sign": expected_sign,
            "split_stats": split_stats,
            "diff_df": diff_df,
            "filters_df": filters_df,
            "before_after_df": before_after_df,
        })

    return results


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _section_5a(results_5a: dict, n_discovery: int) -> list[str]:
    lines = ["# Phase 5a — Confluence Rule Mining (T5.1-T5.4)\n"]
    lines.append(
        f"All 2^8 = 256 boolean combinations of the 8 (indicator, TF) bull "
        f"flags ({', '.join(FLAG_NAMES)}; flag = `A_hist{{suffix}} > 0` or "
        f"`B_roc{{suffix}} > 0`, NaN -> not bullish) are enumerated and "
        f"evaluated on the **rule-discovery set** "
        f"(`bar_close.dt.year <= {DISCOVERY_YEAR_MAX}`, n={n_discovery:,} rows "
        f"-- years 1-3). For each rule, `mean_ret` / `hit_rate` / `t_stat` are "
        f"computed on `{RET_COL}` (Newey-West HAC, lags={HAC_LAGS}); "
        f"`p_val_bonf` applies a Bonferroni correction (x256 rules). "
        f"`composite = frequency * |mean_ret| * (hit_rate - 0.5)` ranks rules "
        f"by a combination of prevalence, edge size, and edge-direction "
        f"consistency. Full 256-row table: `reports/phase5/rules_all.csv`.\n"
    )

    lines.append("## Top 20 rules (by composite)\n")
    lines.append(_df_to_md(results_5a["top20"], max_rows=20))

    lines.append("## Bottom 20 rules (by composite)\n")
    lines.append(_df_to_md(results_5a["bottom20"], max_rows=20))

    survivors = results_5a["survivors"]
    lines.append(
        f"## Survivors (Hamming-deduped, min_hamming={MIN_HAMMING})\n\n"
        f"Rules from the top-20 whose 8-flag vector differs by >= "
        f"{MIN_HAMMING} bits from every higher-ranked survivor (drops "
        f"trivially-related single-flag variants). n={len(survivors)}.\n"
    )
    lines.append(_df_to_md(survivors, max_rows=20))

    return lines


def _section_5b(results_5b: dict) -> list[str]:
    lines = ["# Phase 5b — LightGBM Feature Importance (T5.5-T5.10)\n"]

    bt = results_5b["balance_train"]
    bv = results_5b["balance_val"]
    lines.append(
        "## Walk-forward setup\n\n"
        f"- Train: bar_close in [{results_5b['train_range'][0]}, "
        f"{results_5b['train_range'][1]}], n={results_5b['n_train']:,}\n"
        f"- Validation (hold-out): bar_close in [{results_5b['val_range'][0]}, "
        f"{results_5b['val_range'][1]}], n={results_5b['n_val']:,}\n"
        f"- Target: y = (`{RET_COL}` > 0)\n"
        f"- Train class balance: {bt['n_pos']:,} / {bt['n']:,} positive "
        f"({bt['pct_pos'] * 100:.1f}%)\n"
        f"- Validation class balance: {bv['n_pos']:,} / {bv['n']:,} positive "
        f"({bv['pct_pos'] * 100:.1f}%)\n"
        f"- Model: `LGBMClassifier(max_depth=4, n_estimators=200, "
        f"learning_rate=0.05, random_state=42)`\n"
        f"- **Hold-out AUC ({VAL_YEAR}, in-sample-test only -- NOT the final "
        f"OOS check, see Phase 6)**: {results_5b['auc']:.4f}\n"
    )

    lines.append("## Permutation importance (top 15, scoring=roc_auc, n_repeats=10)\n")
    lines.append(_df_to_md(results_5b["perm_table"], max_rows=15))
    lines.append("_Full table: `reports/phase5/permutation_importance.csv`._\n")

    lines.append("## SHAP summary\n")
    lines.append("![SHAP summary](figs/phase5/shap_summary.png)\n")

    feat1, feat2 = results_5b["feat1"], results_5b["feat2"]
    corr = results_5b["interaction_corr"]
    lines.append(f"## SHAP dependence -- `{feat1}` colored by `{feat2}`\n")
    lines.append("![SHAP dependence](figs/phase5/shap_dependence.png)\n")

    if abs(corr) > 0.1:
        interp = (
            f"a notable interaction: the marginal effect of `{feat1}` on the "
            f"predicted probability shifts depending on the level of `{feat2}`."
        )
    else:
        interp = f"little interaction: `{feat1}`'s effect is largely independent of `{feat2}`."
    lines.append(f"_corr(SHAP({feat1}), {feat2}) = {_fmt(corr, 3)} -> {interp}_\n")

    lines.append("## Top splits of tree 0\n")
    lines.append(_df_to_md(results_5b["splits_df"], max_rows=3))

    return lines


def _section_5c(results_5c: list[dict]) -> list[str]:
    lines = ["# Phase 5c — Failure / Inversion Analysis (T5.11-T5.14)\n"]
    lines.append(
        f"For each of the top {len(results_5c)} survivor rules from 5a, rows "
        f"matching the rule on the rule-discovery set are split into "
        f"**worked** (`sign({RET_COL}) == expected_sign`, where expected_sign "
        f"is the sign of the rule's `mean_ret`) vs **failed** (otherwise). "
        f"The 65 ML features that are not one of the rule's own 8 defining "
        f"flag columns are compared between the two groups via Cohen's d "
        f"(`effect_size`) and Welch's t-test. Features with "
        f"|effect_size| > {FILTER_THRESHOLD} are proposed as filters; "
        f"before/after stats show the rule's `{RET_COL}` edge with vs "
        f"without each filter applied (filter direction chosen so the "
        f"filtered population looks more like 'worked').\n"
    )

    for res in results_5c:
        rule_id = res["rule_id"]
        rule = res["rule"]
        stats = res["split_stats"]
        flags_on = [f for f in FLAG_NAMES if bool(rule[f])]

        lines.append(f"## Rule {rule_id} (expected_sign={res['expected_sign']:+d})\n")
        lines.append(
            f"Active (bullish) flags: {', '.join(flags_on) if flags_on else '(none)'}. "
            f"5a stats: mean_ret={_fmt(rule['mean_ret'])}, "
            f"hit_rate={_fmt(rule['hit_rate'], 3)}, n={int(rule['n']):,}.\n\n"
            f"- n_worked={stats['n_worked']:,}, n_failed={stats['n_failed']:,}\n"
            f"- rule_hit_rate={_fmt(stats['rule_hit_rate'], 3)} vs. "
            f"baseline_hit_rate (unconditional, discovery set)="
            f"{_fmt(stats['baseline_hit_rate'], 3)}\n"
        )

        lines.append("**Top differentiating features (worked vs. failed):**\n")
        lines.append(_df_to_md(res["diff_df"], max_rows=10))

        lines.append("**Proposed filters (before -> after):**\n")
        if res["before_after_df"].empty:
            lines.append(f"_No features exceeded |effect_size| > {FILTER_THRESHOLD}._\n")
        else:
            lines.append(_df_to_md(res["before_after_df"], max_rows=10))

    return lines


# ---------------------------------------------------------------------------
# Main orchestration function
# ---------------------------------------------------------------------------

def run_phase5() -> None:
    """Run all Phase 5 analyses and write the three Phase 5 reports."""

    if not MASTER.exists():
        print("[phase5] master_5m.parquet not found -- run earlier phases first")
        sys.exit(0)

    REPORTS_5.mkdir(parents=True, exist_ok=True)
    FIGS_5.mkdir(parents=True, exist_ok=True)

    print("[phase5] Loading master_5m.parquet ...")
    master = pd.read_parquet(MASTER)
    master = add_forward_returns(master)
    print(f"[phase5] Loaded {len(master):,} rows x {len(master.columns)} columns")

    discovery_df = master[master["bar_close"].dt.year <= DISCOVERY_YEAR_MAX].reset_index(drop=True)
    print(f"[phase5] Rule-discovery set (year <= {DISCOVERY_YEAR_MAX}): {len(discovery_df):,} rows")

    print("[phase5] Running 5a (rule enumeration, ranking, dedup) ...")
    results_5a = _run_5a(discovery_df)
    print(f"[phase5]   survivors: {len(results_5a['survivors'])} rules")

    print(f"[phase5] Running 5b (LightGBM importance, train<={DISCOVERY_YEAR_MAX}, val=={VAL_YEAR}) ...")
    results_5b = _run_5b(master)
    print(f"[phase5]   hold-out AUC = {results_5b['auc']:.4f}")

    print("[phase5] Running 5c (failure/inversion analysis) ...")
    X_discovery, _ = prepare_features(discovery_df, ret_col=RET_COL)
    results_5c = _run_5c(discovery_df, results_5a["survivors"], X_discovery)

    print("[phase5] Writing reports ...")
    rules_md = _section_5a(results_5a, len(discovery_df))
    (REPORTS_DIR / "phase5_rules_top20.md").write_text("\n".join(rules_md), encoding="utf-8")

    importance_md = _section_5b(results_5b)
    (REPORTS_DIR / "phase5_ml_importance.md").write_text("\n".join(importance_md), encoding="utf-8")

    failure_md = _section_5c(results_5c)
    (REPORTS_DIR / "phase5_failure_analysis.md").write_text("\n".join(failure_md), encoding="utf-8")

    print(f"[phase5] Reports written: "
          f"{REPORTS_DIR / 'phase5_rules_top20.md'}, "
          f"{REPORTS_DIR / 'phase5_ml_importance.md'}, "
          f"{REPORTS_DIR / 'phase5_failure_analysis.md'}")
    print("[phase5] Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_phase5()
