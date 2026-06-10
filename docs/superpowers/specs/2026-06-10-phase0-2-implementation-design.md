# Implementation Design: Phase 0–2 + Gate 1
**Date:** 2026-06-10
**Project:** Multi-Timeframe Momentum Confluence Study — NIFTY 50
**Scope:** Project setup (Phase 0), Data Engineering (Phase 1), Feature Engineering (Phase 2), and the no-look-ahead Gate 1 unit test.
**Companion docs:** `DESIGN_DOC.md` (what/why), `PLAN.md` (execution strategy), `TASK.md` (atomic checklist)

---

## 1. Goal

Produce a validated, no-look-ahead feature dataset (`master_5m.parquet`) from raw 5m NIFTY 50 CSV data, with all indicator features for both MACD_A and MACD_B computed across four timeframes (5m, 15m, 1h, 1D), and a passing Gate 1 unit test that proves the no-look-ahead property.

---

## 2. Directory Structure

```
E:\vit-project\MultiTimeframe momentum confluence\
├── data/
│   ├── raw/                          # source: Data/nifty_intraday_5min.csv (symlink or copy)
│   ├── processed/                    # Phase 1 outputs
│   │   ├── nifty_5m.parquet
│   │   ├── nifty_15m.parquet
│   │   ├── nifty_1h.parquet
│   │   └── nifty_1d.parquet
│   └── features/                     # Phase 2 outputs
│       ├── features_A_5m.parquet     # intermediate (Agent 2)
│       ├── features_A_15m.parquet
│       ├── features_A_1h.parquet
│       ├── features_A_1d.parquet
│       ├── features_B_5m.parquet     # intermediate (Agent 3)
│       ├── features_B_15m.parquet
│       ├── features_B_1h.parquet
│       ├── features_B_1d.parquet
│       ├── features_5m.parquet       # merged (Agent 4)
│       ├── features_15m.parquet
│       ├── features_1h.parquet
│       ├── features_1d.parquet
│       └── master_5m.parquet         # final joined dataset (Agent 4)
├── src/
│   ├── phase1_data/
│   │   ├── __init__.py
│   │   ├── loader.py                 # load + validate + timestamp parsing
│   │   └── resampler.py              # session-anchored groupby resamplers
│   └── phase2_features/
│       ├── __init__.py
│       ├── indicator_a.py            # MACD 12/26/9 + all A_ derived columns
│       ├── indicator_b.py            # EMA-Gap-ROC 40/90/7 + all B_ derived columns
│       └── master_join.py            # merge_asof joins + warmup drops + master_5m
├── tests/
│   └── test_no_lookahead.py          # Gate 1: 1000-row random spot check
├── reports/
│   ├── figs/
│   │   └── phase1/
│   ├── phase1_data_quality.md
│   └── phase2_validation_log.md
└── requirements.txt
```

---

## 3. Agent Decomposition (Approach B — Parallel Indicators)

```
Agent 1 ──► Phase 0 setup + Phase 1 complete
               │
               ▼ (produces: nifty_{5m,15m,1h,1d}.parquet)
          ┌────┴────┐
       Agent 2   Agent 3          (run in parallel)
    Indicator A  Indicator B
         │            │
         ▼            ▼
    features_A_  features_B_
    {tf}.parquet  {tf}.parquet
          └────┬────┘
               ▼
           Agent 4
    Merge + master join + Gate 1
```

### Agent 1 responsibilities
- Create full directory structure
- Copy `Data/nifty_intraday_5min.csv` into `data/raw/`
- `src/phase1_data/loader.py`: load CSV, parse timezone to Asia/Kolkata, drop duplicates, sort, assert volume=0 then drop volume, compute `bar_close = timestamp + 5min`, compute `session_date` and `bar_in_session`, validate OHLC sanity
- `src/phase1_data/resampler.py`: session-anchored resampling using `groupby` on computed bin labels (NOT `df.resample()`), producing 15m / 1h / 1d OHLC parquets, each with `timestamp`, `bar_close`, `open`, `high`, `low`, `close`, `session_date`, `bar_in_session`
- `reports/phase1_data_quality.md`: date range, row counts per TF, gaps detected, bars-per-day histogram, spot-check table
- `requirements.txt`: pinned versions

### Agent 2 responsibilities (runs in parallel with Agent 3, after Agent 1)
- `src/phase2_features/indicator_a.py`: compute for each of the 4 TF parquets:
  - `A_macd`, `A_signal`, `A_hist`, `A_hist_slope`
  - `A_macd_sign`, `A_hist_sign`
  - `A_macd_zero_cross`, `A_hist_zero_cross` (signed ±1/0)
  - `A_state` (4-quadrant: bull_accel / bull_decel / bear_decel / bear_accel) with edge-case rule for hist==0
- Saves `data/features/features_A_{tf}.parquet` for all 4 TFs: contains all Phase 1 columns (`timestamp`, `bar_close`, `open`, `high`, `low`, `close`, `session_date`, `bar_in_session`) plus all `A_*` derived columns. Does NOT drop warmup yet — warmup drop is Agent 4's job.

### Agent 3 responsibilities (runs in parallel with Agent 2, after Agent 1)
- `src/phase2_features/indicator_b.py`: compute for each of the 4 TF parquets:
  - `B_gap = EMA(40) - EMA(90)`, `B_gap_norm = B_gap / close`
  - `B_roc = 100 * (gap - gap[7]) / gap[7]` with division-by-zero guard (set 0 when gap_prev==0)
  - `B_roc_invalid`: flag when `abs(gap_prev) < epsilon` (epsilon = 1st percentile of `abs(B_gap)` per TF)
  - `B_zroc = (B_roc - SMA(B_roc, 100)) / STDEV(B_roc, 100)`
  - `B_gap_acc = B_gap - B_gap[7]`
  - `B_roc_slope = B_roc - B_roc[1]`
  - `B_zroc_extreme = 1 if abs(B_zroc) > 2 else 0`
  - `B_state` (4-quadrant, same convention as A_state, using sign(roc) and sign(roc_slope))
- Saves `data/features/features_B_{tf}.parquet` for all 4 TFs: contains all Phase 1 columns plus all `B_*` derived columns. Does NOT drop warmup yet.

### Agent 4 responsibilities (sequential, after Agents 2+3)
- `src/phase2_features/master_join.py`:
  1. Load `features_A_{tf}` and `features_B_{tf}` for each TF; merge on `timestamp` + `bar_close` (inner join, same rows)
  2. Drop first 270 rows per TF (EMA warmup) → save `data/features/features_{tf}.parquet`
  3. Build `master_5m.parquet`: start from `features_5m`, then `merge_asof` ← 15m, ← 1h, ← 1d, each on `bar_close`, `direction='backward'`, `allow_exact_matches=True`; add TF suffix to all higher-TF columns (e.g. `A_hist_15m`)
  4. Verify no nulls in higher-TF columns (after warmup alignment)
- `tests/test_no_lookahead.py`: sample 1000 random rows from `master_5m.parquet`; for each, assert that the 15m/1h/1d feature timestamp is ≤ the 5m `bar_close`
- `reports/phase2_validation_log.md`: 5 manually selected spot-checks documenting correctness

---

## 4. Correctness Invariants

| # | Rule | Where enforced |
|---|---|---|
| I1 | `ewm(span=n, adjust=False)` to match Pine's EMA | `indicator_a.py`, `indicator_b.py` |
| I2 | All joins use `bar_close` (not `timestamp`) | `master_join.py` |
| I3 | `merge_asof(direction='backward', allow_exact_matches=True)` | `master_join.py` |
| I4 | First 270 bars per TF dropped AFTER computing all features | `master_join.py` step 2 |
| I5 | `B_roc_invalid` flag set for `abs(gap_prev) < epsilon` | `indicator_b.py` |
| I6 | Volume: assert all-zero then drop immediately | `loader.py` |
| I7 | Session-anchored resampling via `groupby` (not `resample()`) | `resampler.py` |
| I8 | 1h stub bar (15:15–15:30) dropped | `resampler.py` |

---

## 5. Output Contracts

### Phase 1 parquet schema (all 4 TFs)
Columns: `timestamp` (tz-aware, Asia/Kolkata), `bar_close` (tz-aware), `open`, `high`, `low`, `close`, `session_date`, `bar_in_session`
No volume column.

### Phase 2 feature parquet schema (per TF, after warmup drop)
All Phase 1 columns, plus all `A_*` and `B_*` derived columns from §2.1 and §2.2 of DESIGN_DOC.

### master_5m.parquet schema
All 5m feature columns (no suffix), plus all 15m/1h/1d feature columns with `_{tf}` suffix (e.g. `A_hist_15m`, `B_state_1h`). Row count = 5m feature rows after warmup drop and after earliest 1d warmup aligns.

---

## 6. Success Criteria

- All 4 TF parquets exist, pass OHLC sanity checks, and contain the expected column set
- Gate 1 test (`test_no_lookahead.py`) passes on 1000 random rows with exit code 0
- `phase1_data_quality.md` documents gaps, duplicates, bars-per-day
- `phase2_validation_log.md` documents 5 manual spot-checks
- `requirements.txt` exists with pinned library versions
