from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import pandas as pd
from src.phase2_features.indicator_a import compute_indicator_a

PROCESSED = Path("data/processed")
FEATURES = Path("data/features")


def run() -> None:
    FEATURES.mkdir(parents=True, exist_ok=True)
    for tf in ["5m", "15m", "1h", "1d"]:
        print(f"[indicator_a] Processing {tf}...")
        df = pd.read_parquet(PROCESSED / f"nifty_{tf}.parquet")
        result = compute_indicator_a(df)
        out = FEATURES / f"features_A_{tf}.parquet"
        result.to_parquet(out, index=False)
        print(f"[indicator_a] {out.name}: {len(result):,} rows, {result.shape[1]} cols")


if __name__ == "__main__":
    run()
