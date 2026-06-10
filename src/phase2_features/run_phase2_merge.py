from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import pandas as pd
from src.phase2_features.master_join import (
    merge_ab_features, apply_warmup_drop, build_master_5m, WARMUP_BARS,
)

FEATURES = Path("data/features")
REPORTS = Path("reports")


def run() -> None:
    FEATURES.mkdir(parents=True, exist_ok=True)

    feature_dfs: dict[str, pd.DataFrame] = {}
    for tf in ["5m", "15m", "1h", "1d"]:
        print(f"[phase2_merge] Merging A+B for {tf}...")
        df_a = pd.read_parquet(FEATURES / f"features_A_{tf}.parquet")
        df_b = pd.read_parquet(FEATURES / f"features_B_{tf}.parquet")
        merged = merge_ab_features(df_a, df_b)
        dropped = apply_warmup_drop(merged, warmup=WARMUP_BARS)
        out = FEATURES / f"features_{tf}.parquet"
        dropped.to_parquet(out, index=False)
        feature_dfs[tf] = dropped
        print(f"[phase2_merge] {out.name}: {len(dropped):,} rows "
              f"({WARMUP_BARS} warmup dropped), {dropped.shape[1]} cols")

    print("[phase2_merge] Building master_5m.parquet...")
    master = build_master_5m(
        df_5m=feature_dfs["5m"],
        df_15m=feature_dfs["15m"],
        df_1h=feature_dfs["1h"],
        df_1d=feature_dfs["1d"],
    )
    out = FEATURES / "master_5m.parquet"
    master.to_parquet(out, index=False)
    print(f"[phase2_merge] master_5m.parquet: {len(master):,} rows, {master.shape[1]} cols")

    _write_validation_log(master, feature_dfs)
    print("[phase2_merge] Done.")


def _write_validation_log(master: pd.DataFrame, feature_dfs: dict) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    # Only sample rows where all HTF bar_close tracking cols are non-null
    htf_cols = ["bar_close_15m", "bar_close_1h", "bar_close_1d"]
    valid_mask = master[htf_cols].notna().all(axis=1)
    sample = master[valid_mask].sample(5, random_state=99).sort_values("bar_close")

    lines = [
        "# Phase 2 Validation Log",
        "",
        "## No-Look-Ahead Spot Check (5 randomly selected 5m bars)",
        "",
        "| 5m bar_close | bar_close_15m | bar_close_1h | bar_close_1d | OK? |",
        "|---|---|---|---|---|",
    ]
    for _, row in sample.iterrows():
        ok = (
            row["bar_close_15m"] <= row["bar_close"]
            and row["bar_close_1h"] <= row["bar_close"]
            and row["bar_close_1d"] <= row["bar_close"]
        )
        lines.append(
            f"| {row['bar_close']} | {row['bar_close_15m']} "
            f"| {row['bar_close_1h']} | {row['bar_close_1d']} "
            f"| {'✓' if ok else '✗ FAIL'} |"
        )
    lines += [
        "",
        "## Feature Parquet Summary",
        "",
        "| TF | Rows | Cols | A_state distribution |",
        "|---|---|---|---|",
    ]
    for tf, df in feature_dfs.items():
        dist = df["A_state"].value_counts(normalize=True).round(3).to_dict()
        lines.append(f"| {tf} | {len(df):,} | {df.shape[1]} | {dist} |")
    (REPORTS / "phase2_validation_log.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[phase2_merge] Wrote {REPORTS / 'phase2_validation_log.md'}")


if __name__ == "__main__":
    run()
