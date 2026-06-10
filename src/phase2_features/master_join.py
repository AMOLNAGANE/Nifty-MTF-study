from __future__ import annotations
import pandas as pd

WARMUP_BARS = 270


def merge_ab_features(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """Join Indicator B columns onto Indicator A DataFrame (same rows, inner join on index)."""
    b_only = [c for c in df_b.columns if c.startswith("B_")]
    return df_a.join(df_b[b_only])


def apply_warmup_drop(df: pd.DataFrame, warmup: int = WARMUP_BARS) -> pd.DataFrame:
    """Drop first `warmup` rows (EMA seed distortion), reset index."""
    return df.iloc[warmup:].reset_index(drop=True)


def build_master_5m(
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_1d: pd.DataFrame,
) -> pd.DataFrame:
    """Build master 5m DataFrame via merge_asof from higher-TF feature DataFrames.

    Each HTF gets its A_*/B_* columns renamed to A_*_{tf}/B_*_{tf}.
    A bar_close_{tf} column is retained for Gate 1 no-look-ahead testing.
    """
    master = df_5m.copy().sort_values("bar_close").reset_index(drop=True)

    for tf, df_htf in [("15m", df_15m), ("1h", df_1h), ("1d", df_1d)]:
        htf = df_htf.copy().sort_values("bar_close").reset_index(drop=True)

        # Preserve HTF bar_close as a named column before using it as the merge key
        htf[f"bar_close_{tf}"] = htf["bar_close"]

        # Rename all A_* and B_* indicator columns with _{tf} suffix
        rename = {
            c: f"{c}_{tf}"
            for c in htf.columns
            if c.startswith("A_") or c.startswith("B_")
        }
        htf = htf.rename(columns=rename)

        # Keep only the merge key and columns suffixed with _{tf}
        # bar_close_{tf} ends with _{tf} so the comprehension captures it without duplication
        keep = ["bar_close"] + [
            c for c in htf.columns if c.endswith(f"_{tf}")
        ]
        htf = htf[keep]

        master = pd.merge_asof(
            master,
            htf,
            on="bar_close",
            direction="backward",
            allow_exact_matches=True,
        )

    return master.reset_index(drop=True)
