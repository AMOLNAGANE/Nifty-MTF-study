from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.phase1_data.loader import load_5m_csv
from src.phase1_data.resampler import resample_to_15m, resample_to_1h, resample_to_1d

RAW_CSV = Path("data/raw/nifty_5m.csv")
PROCESSED = Path("data/processed")
REPORTS = Path("reports")
FIGS = REPORTS / "figs" / "phase1"


def detect_gaps(df_5m: pd.DataFrame) -> list[dict]:
    gap_days = []
    for date, grp in df_5m.groupby("session_date"):
        deltas = grp.sort_values("timestamp")["timestamp"].diff().dropna()
        intra = deltas[deltas > pd.Timedelta(minutes=5)]
        if len(intra):
            gap_days.append({"date": str(date), "gaps": len(intra)})
    return gap_days


def write_quality_report(
    df_5m, df_15m, df_1h, df_1d, gap_days, bars_per_day
) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 1 Data Quality Report\n",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n",
        "## Dataset Overview",
        "| TF | Rows | Date range |",
        "|---|---|---|",
    ]
    for name, df in [("5m", df_5m), ("15m", df_15m), ("1h", df_1h), ("1d", df_1d)]:
        lines.append(
            f"| {name} | {len(df):,} | "
            f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()} |"
        )
    lines += [
        "",
        "## Data Quality",
        f"- Intra-session gaps (sessions with missing 5m bars): **{len(gap_days)}**",
    ]
    if gap_days:
        lines.append(
            "  - Sample affected dates: "
            + ", ".join(g["date"] for g in gap_days[:10])
        )
    lines += [
        "",
        "## Bars Per Day Distribution",
        f"- Mean: {bars_per_day.mean():.1f} | Min: {bars_per_day.min()} "
        f"| Max: {bars_per_day.max()}",
        "",
        "![Bars per day histogram](figs/phase1/bars_per_day_hist.png)",
        "",
        "## Spot-Check: First Trading Day at Each TF",
        "| TF | timestamp | bar_close | open | high | low | close |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, df in [("5m", df_5m), ("15m", df_15m), ("1h", df_1h), ("1d", df_1d)]:
        r = df.iloc[0]
        bar_close_val = r.get('bar_close', 'N/A')
        lines.append(
            f"| {name} | {r['timestamp']} | {bar_close_val} | "
            f"{r['open']:.2f} | {r['high']:.2f} | {r['low']:.2f} | {r['close']:.2f} |"
        )
    (REPORTS / "phase1_data_quality.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    if not RAW_CSV.exists():
        sys.exit(f"[phase1] ERROR: raw CSV not found at {RAW_CSV.resolve()}")
    print("[phase1] Loading 5m data...")
    df_5m = load_5m_csv(RAW_CSV)
    df_5m.to_parquet(PROCESSED / "nifty_5m.parquet", index=False)
    print(f"[phase1] nifty_5m.parquet: {len(df_5m):,} rows")

    print("[phase1] Resampling to 15m...")
    df_15m = resample_to_15m(df_5m)
    df_15m.to_parquet(PROCESSED / "nifty_15m.parquet", index=False)
    print(f"[phase1] nifty_15m.parquet: {len(df_15m):,} rows")

    print("[phase1] Resampling to 1h...")
    df_1h = resample_to_1h(df_5m)
    df_1h.to_parquet(PROCESSED / "nifty_1h.parquet", index=False)
    print(f"[phase1] nifty_1h.parquet: {len(df_1h):,} rows")

    print("[phase1] Resampling to 1d...")
    df_1d = resample_to_1d(df_5m)
    df_1d.to_parquet(PROCESSED / "nifty_1d.parquet", index=False)
    print(f"[phase1] nifty_1d.parquet: {len(df_1d):,} rows")

    gap_days = detect_gaps(df_5m)
    bars_per_day = df_5m.groupby("session_date").size()

    fig, ax = plt.subplots(figsize=(8, 4))
    bars_per_day.hist(bins=20, ax=ax)
    ax.set_xlabel("Bars per trading day")
    ax.set_ylabel("Count")
    ax.set_title("NIFTY 5m: Bars Per Trading Day")
    fig.savefig(FIGS / "bars_per_day_hist.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    write_quality_report(df_5m, df_15m, df_1h, df_1d, gap_days, bars_per_day)
    print(f"[phase1] Quality report written to {REPORTS / 'phase1_data_quality.md'}")
    print("[phase1] Done.")


if __name__ == "__main__":
    run()
