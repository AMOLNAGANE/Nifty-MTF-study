# task.md — Atomic Task Checklist

**Companion to:** `design.md` (the WHAT) and `plan.md` (the HOW).
**Use:** Tick boxes as you go. Each task has an acceptance criterion — don't tick until it's actually met.

Legend:
- `[ ]` open
- `[x]` complete
- 🔒 = **gate** — do not proceed past until satisfied
- ⚠ = correctness-critical
- 💡 = optional / nice-to-have

---

## Phase 0 — Setup (Day 0)

- [ ] **T0.1** Create repo structure per PLAN §5.3
  - Acceptance: `tree nifty_mtf_study/` shows all directories listed in PLAN §5.3
- [ ] **T0.2** Set up Python environment with the libraries listed in PLAN §5.1
  - Acceptance: `pip list` shows pandas, pyarrow, talib (or skip), scipy, statsmodels, arch, lightgbm, shap, matplotlib, seaborn, plotly, jupyter
- [ ] **T0.3** Place raw 5m NIFTY CSV in `data/raw/` and record its filename, row count, date range
  - Acceptance: row count + date range logged in `reports/phase1_data_quality.md` (header section)
- [ ] **T0.4** Record library versions for reproducibility
  - Acceptance: `pip freeze > requirements.txt` in repo root
- [ ] **T0.5** Resolve pre-decided decisions D1–D4 from PLAN §3.1; log resolutions in PLAN §3.1 itself
  - Acceptance: each row in §3.1 has a clear "Resolved: <choice>" annotation

---

## Phase 1 — Data Engineering (Days 1–2)

### 1A. Load and validate raw data ⚠

- [ ] **T1.1** Load raw 5m CSV into pandas DataFrame
  - Acceptance: `df.shape`, `df.dtypes`, `df['timestamp'].min()`, `df['timestamp'].max()` printed
- [ ] **T1.2** ⚠ Parse timestamps with explicit timezone (`+05:30`); verify with first/last row
  - Acceptance: `df['timestamp'].dt.tz` is `Asia/Kolkata` or equivalent fixed offset
- [ ] **T1.3** Drop duplicate timestamps (keep first); log count dropped
  - Acceptance: `df['timestamp'].is_unique == True` after drop
- [ ] **T1.4** Sort by timestamp ascending
  - Acceptance: `df['timestamp'].is_monotonic_increasing == True`
- [ ] **T1.4b** Assert volume column is all-zero, log confirmation, then drop the column
  - Acceptance: `(df['volume'] == 0).all()` asserted; column absent from all downstream parquets
- [ ] **T1.5** Compute `bar_close = timestamp + 5min` and add as column
  - Acceptance: column exists, type is timestamp with tz, no nulls
- [ ] **T1.6** ⚠ Compute `session_date = bar_close.date()` and `bar_in_session` (1-indexed within session)
  - Acceptance: session_date is non-null; bar_in_session ranges roughly 1..75 within a day
- [ ] **T1.7** Validate OHLC sanity: high ≥ max(open, close), low ≤ min(open, close)
  - Acceptance: count of violations logged; if >0, investigate before proceeding
- [ ] **T1.8** Compute bars-per-day distribution; identify holiday-truncated days
  - Acceptance: histogram saved as `reports/figs/bars_per_day_hist.png`
- [ ] **T1.9** Resolve D5 (holiday-truncated threshold) using T1.8 output
  - Acceptance: threshold value documented in PLAN §3.2 with rationale
- [ ] **T1.10** Detect intra-session gaps in 5m bars (missing bars between 09:15–15:30)
  - Acceptance: count of session-days with gaps logged; do NOT forward-fill

### 1B. Resample to higher TFs ⚠

- [ ] **T1.11** ⚠ Implement session-anchored resampler — do **NOT** use `df.resample('15T')` blindly
  - Acceptance: function `resample_to_tf(df_5m, tf)` exists and is unit-tested with a known small example
- [ ] **T1.12** Produce `nifty_15m.parquet` (3 consecutive 5m bars per group)
  - Acceptance: bars per day ≈ 25; spot-check first day matches manual computation
- [ ] **T1.13** Produce `nifty_1h.parquet` (session-anchored 09:15–10:15, etc.); drop or keep stub per D1
  - Acceptance: bars per day = 6 (if stub dropped) or 7 (if kept); spot-check matches
- [ ] **T1.14** Produce `nifty_1d.parquet`
  - Acceptance: bars = number of trading days; spot-check OHLC for one known day matches NSE bhavcopy
- [ ] **T1.15** Add `bar_close` column to each higher-TF parquet (= bar start + TF duration)
  - Acceptance: column present; for 1d, `bar_close` = end-of-session timestamp (e.g. 15:30:00)

### 1C. Phase 1 outputs

- [ ] **T1.16** Write `reports/phase1_data_quality.md` covering:
  - Date range, total rows per TF
  - Duplicates dropped, gaps detected
  - Holiday-truncated days (per D5 resolution)
  - Bars-per-day histogram
  - Spot-check of one day at each TF (manual verification)
  - Acceptance: file exists, sections present, plots embedded

---

## Phase 2 — Feature Engineering (Days 3–4)

### 2A. Indicator A (MACD 12/26/9) ⚠

- [ ] **T2.1** ⚠ Implement EMA matching Pine's `ta.ema` exactly using `ewm(adjust=False)`
  - Acceptance: side-by-side comparison with 1000 TradingView-exported values; agreement to 4 decimal places after warmup
- [ ] **T2.2** Compute `A_macd`, `A_signal`, `A_hist`, `A_hist_slope` for each TF
  - Acceptance: columns exist; no NaN beyond warmup period
- [ ] **T2.3** Compute sign features: `A_macd_sign`, `A_hist_sign`
  - Acceptance: values in {-1, 0, 1}; sum of signs distribution looks reasonable (~50/50 over time)
- [ ] **T2.4** Compute zero-cross flags: `A_macd_zero_cross`, `A_hist_zero_cross`
  - Acceptance: flag = 1 only when sign changed from previous bar; total crosses per year sensible (~50-200)
- [ ] **T2.5** Compute 4-quadrant `A_state` (bull_accel / bull_decel / bear_decel / bear_accel)
  - Acceptance: 4 distinct values; each populated; printed as percentage distribution

### 2B. Indicator B (EMA-Gap ROC 40/90/7) ⚠

- [ ] **T2.6** Compute `B_gap = EMA40 - EMA90` per TF
  - Acceptance: column exists; no NaN beyond warmup
- [ ] **T2.7** Compute `B_gap_norm = B_gap / close`
  - Acceptance: range typically [-0.05, +0.05] (5%) — sanity check
- [ ] **T2.8** Compute `B_roc` with **division-by-zero guard** (= 0 when gap_prev = 0)
  - Acceptance: no inf/nan in column
- [ ] **T2.9** ⚠ Compute `B_roc_invalid` flag (1 when abs(gap_prev) < epsilon, per L3 in PLAN)
  - Acceptance: epsilon resolved via D6; ~1-5% of bars flagged typically
- [ ] **T2.10** Compute `B_zroc = (B_roc - SMA100(B_roc)) / STDEV100(B_roc)`
  - Acceptance: mean ≈ 0, std ≈ 1 after warmup; <5% of values with abs > 2
- [ ] **T2.11** Compute `B_gap_acc`, `B_roc_slope`, `B_zroc_extreme` per spec
  - Acceptance: columns exist, sensible distributions
- [ ] **T2.12** Compute 4-quadrant `B_state` from (sign(roc), sign(roc_slope))
  - Acceptance: 4 values; distribution printed

### 2C. Write per-TF feature parquets

- [ ] **T2.13** Save `features_5m.parquet` with all features above
  - Acceptance: file exists; column count ≈ 20 indicator features + OHLC + bar_close
- [ ] **T2.14** Save `features_15m.parquet`
- [ ] **T2.15** Save `features_1h.parquet`
- [ ] **T2.16** Save `features_1d.parquet`
- [ ] **T2.17** ⚠ Drop EMA-warmup rows (first 270 bars per TF) before saving
  - Acceptance: row count per TF matches expectation (total bars − 270)

### 2D. Build master joined dataset ⚠

- [ ] **T2.18** ⚠ Use `pd.merge_asof(on='bar_close', direction='backward', allow_exact_matches=True)` to join 5m ← 15m ← 1h ← 1d
  - Acceptance: master_5m.parquet exists; row count = 5m feature rows; columns prefixed with TF (e.g. `A_hist_15m`)
- [ ] **T2.19** Verify no nulls in higher-TF columns (after sufficient warmup)
  - Acceptance: null count per column logged; explained if any present

### 2E. 🔒 GATE 1 — No-look-ahead unit test ⚠

- [ ] **T2.20** ⚠ Write `tests/test_no_lookahead.py` that:
  - Samples 1000 random rows from `master_5m.parquet`
  - For each row at 5m bar_close T:
    - Verifies the 15m feature came from a 15m bar with bar_close ≤ T
    - Verifies the 1h feature came from a 1h bar with bar_close ≤ T
    - Verifies the 1d feature came from a 1d bar with bar_close ≤ T
  - Acceptance: test runs; all 1000 rows pass; exit code 0
- [ ] **T2.21** ⚠ Manual spot-check: pick 5 random 5m timestamps, manually verify each higher-TF feature is from the correct preceding bar
  - Acceptance: results documented in `reports/phase2_validation_log.md`
- [ ] **T2.22** 🔒 **DO NOT PROCEED TO PHASE 3 UNTIL T2.20 AND T2.21 BOTH PASS**

---

## Phase 3 — Intra-Timeframe Pattern Analysis (Days 5–8)

### 3A. Target variables (build once)

- [ ] **T3.1** Compute `ret_fwd_N` for each TF, for N ∈ {1, 3, 6, 12, 24} bars
  - Acceptance: 5 new columns per TF; last N rows correctly NaN
- [ ] **T3.2** Compute first-touch barrier labels (±0.20% / ±0.35% / ±0.60% / ±1.20% per TF)
  - Acceptance: `barrier_hit` column with values in {-1, 0, +1}
- [ ] **T3.3** Compute rolling 20-bar regime label using linear-fit slope + R² (`trend_up` / `trend_down` / `chop`)
  - Acceptance: `regime` column populated; distribution roughly 30/30/40

### 3B. Sub-analysis 3a — Indicator vs. Price (per TF) ⚠

These tasks operationalize hypotheses **H1–H7** from design §2.4. Tag every output table with the hypothesis ID it tests.

- [ ] **T3.4** Compute Pearson + Spearman correlation of each indicator feature vs. each `ret_fwd_N`
  - Acceptance: correlation matrix saved as `reports/phase3a/correlations_{tf}.csv`
- [ ] **T3.5** Build conditional return tables by decile bins of each feature
  - Acceptance: per-feature CSV `reports/phase3a/decile_returns_{tf}_{feature}.csv`
- [ ] **T3.6** Build state-conditional table: group by `A_state` × forward return horizon — **tests H1**
  - Acceptance: 4×5 mean-return table per TF; t-stats reported with Newey-West SE; H1 verdict noted
- [ ] **T3.7** Same for `B_state` — **tests H5**
- [ ] **T3.7b** Event study: `bull_accel → bull_decel` transitions of A; lead-time to nearest local price high; fwd_ret_3 vs fwd_ret_12 decay — **tests H2**
  - Acceptance: lead-time distribution plot + short-vs-long horizon comparison per TF
- [ ] **T3.7c** Split A macd zero-crosses into aligned (hist rising) vs tired (hist falling); compare follow-through — **tests H3**
  - Acceptance: two-sample comparison table per TF with HAC SE
- [ ] **T3.7d** Hist re-cross above zero stratified by sign(macd) — **tests H4 (first-pullback)**
  - Acceptance: conditional table per TF
- [ ] **T3.7e** zROC bucket term-structure: zroc ∈ (0.5, 2] vs zroc > 2 vs baseline, at horizons {3, 6, 12, 24} — **tests H6a vs H6b**
  - Acceptance: term-structure table per TF; verdict: continuation, exhaustion, horizon-dependent, or neither
- [ ] **T3.7f** Early-turn event study: B_gap < 0 with sustained B_gap_acc > 0 for M bars → lag to gap zero-cross + price captured in window; false-start rate — **tests H7**
  - Acceptance: lag distribution, capture stats, false-start percentage per TF
- [ ] **T3.8** Apply Bonferroni / FDR correction for multiple-testing
  - Acceptance: "raw p-value" and "adjusted p-value" columns both present in output tables
- [ ] **T3.9** Plot decile-return curves for top 5 features per TF
  - Acceptance: 4 PNGs in `reports/figs/phase3a/`

### 3C. Sub-analysis 3b — Indicator-vs-indicator (per TF) — corrected logic per design §5.3

- [ ] **T3.10** Compute cross-correlation of `A_hist` vs `B_roc` at lags ∈ [-20, +20] bars per TF — **descriptive only**
  - Acceptance: 4 lag-correlation plots saved; peak lag shows A leading B (if B appears to lead A, treat as bug and investigate before proceeding)
- [ ] **T3.11** ⚠ Incremental-information test (H8): nested regression `ret_fwd_N ~ A_hist_norm` vs `~ A_hist_norm + B_roc` per TF
  - Acceptance: ΔR² and HAC t-stat on B's coefficient reported per TF; verdict on whether B is redundant given A
- [ ] **T3.11b** Turn-confirmation timing (H10): for each A bullish flip, measure bars until B confirms (roc > 0); split flips into confirmed-within-K vs never-confirmed; compare forward returns from flip
  - Acceptance: confirmation-lag distribution + two-group forward-return comparison per TF

### 3D. Sub-analysis 3c — Indicator A × Indicator B agreement (per TF)

- [ ] **T3.12** Build 4×4 `A_state × B_state` joint state matrix per TF; cell value = (count, mean_fwd_ret_12, hit_rate) — **tests H9 (trigger-in-regime vs against-regime) at single-TF level**
  - Acceptance: 4 matrices saved as CSV + heatmap PNG
- [ ] **T3.13** Statistical significance test: chi-square for independence + cell-wise t-tests with HAC SE
  - Acceptance: p-values reported per cell

### 3E. Sub-analysis 3d — Divergence detection (per TF)

- [ ] **T3.14** Use `scipy.signal.find_peaks` to identify price highs/lows with min spacing per TF
  - Acceptance: function returns (idx, value) pairs; visual sanity check on one month of 5m
- [ ] **T3.15** For consecutive same-type extrema, detect price vs. indicator divergence
  - Acceptance: event list with timestamp, type (bull/bear divergence), indicator (A/B)
- [ ] **T3.16** Forward-return distribution after each divergence event vs. baseline — **tests H14**
  - Acceptance: table comparing 4 (TFs) × 4 (bull/bear × A/B) = 16 distributions to baseline

### 3F. Phase 3 reports

- [ ] **T3.17** Write `reports/phase3_intra_5m.md`
  - Acceptance: includes 3a–3d sections; tables + plots embedded
- [ ] **T3.18** Write `reports/phase3_intra_15m.md`
- [ ] **T3.19** Write `reports/phase3_intra_1h.md`
- [ ] **T3.20** Write `reports/phase3_intra_1d.md`
- [ ] **T3.21** Write `reports/phase3_summary.md` cross-TF comparison
  - Acceptance: identifies which TF shows strongest within-TF signal
- [ ] **T3.22** Resolve D7, D8 from PLAN §3.3 based on Phase 3 results

---

## Phase 4 — Inter-Timeframe Pattern Analysis (Days 9–11)

Recall the critical caveat (L4): when stratifying 5m forward returns by higher-TF state, the **effective sample size** for the higher-TF feature is the unique count of higher-TF bars, not 5m rows.

### 4A. Sub-analysis 4a — Higher-TF as regime filter ⚠ — tests H11 (HTF regime + LTF trigger)

- [x] **T4.1** For each top setup from Phase 3 (5m-level), stratify by 1h `A_state` (4 levels)
  - Acceptance: per-setup table with 4 conditional rows; HAC SEs reported
- [x] **T4.2** Same for 1h `B_state`
- [x] **T4.3** Same for 1d `A_state` and 1d `B_state`
- [x] **T4.4** ⚠ Report effective sample size (unique 1h/1d bars) for each conditional, not 5m row count

### 4B. Sub-analysis 4b — Cross-TF lead-lag (precedence statistics)

- [x] **T4.5** Identify all 15m `A_hist` zero-crosses
  - Acceptance: event list with timestamp, direction
- [x] **T4.6** For each 15m cross, measure: did 1h `A_hist` cross same direction within next K bars (K=4 for 1h)?
  - Acceptance: transition probability matrix (15m flip direction → 1h follow-through prob)
- [x] **T4.7** Same exercise for 5m → 15m, 1h → 1d
- [x] **T4.8** Repeat for Indicator B zero-crosses

### 4C. Sub-analysis 4c — Confluence scoring — tests H12 (monotonicity)

- [x] **T4.9** Compute `score_bull` ∈ [0, 8] per 5m bar = sum of (indicator bull) across 8 (indicator, TF) pairs
  - Acceptance: column added to master; distribution histogram saved
- [x] **T4.10** Same for `score_bull_accel` (requires bull_accel state, not just sign)
- [x] **T4.11** Build score-bucketed forward return table at horizons {3, 6, 12} on 5m
  - Acceptance: 9-row table (scores 0..8) × 3 horizons; HAC SE
- [x] **T4.12** Plot monotonicity: score on x-axis, mean fwd return on y-axis
  - Acceptance: PNG saved; visual inspection — is it monotonic, flat, or U-shaped? **H12 verdict recorded**
- [x] **T4.12b** Compression→expansion event study (15m/1h): `|B_gap_norm|` lowest decile followed by `|zroc| > 1.5`; does expansion direction predict subsequent 5m–15m move, and is |move| > baseline? — **tests H13**
  - Acceptance: event list, signed and absolute forward-return comparison vs. baseline

### 4D. Sub-analysis 4d — Asymmetry

- [x] **T4.13** Build `score_bear` symmetric to `score_bull` (count of bearish flags)
- [x] **T4.14** Compare magnitudes of bull-edge vs bear-edge at equivalent scores
  - Acceptance: side-by-side table with comment on asymmetry

### 4E. Phase 4 report

- [x] **T4.15** Write `reports/phase4_inter_tf.md`
  - Acceptance: sections 4a–4d, plots embedded, caveat L4 explicitly stated
  - Key question to answer: **does higher-TF alignment strengthen 5m edges?**

---

## Phase 5 — Confluence Pattern Mining (Days 12–15)

### 5A. Sub-analysis 5a — Boolean rule enumeration

- [x] **T5.1** Enumerate 2^8 = 256 boolean combinations of (indicator, TF) bull flags
  - Acceptance: list/DataFrame of 256 rules
- [x] **T5.2** For each rule, compute: frequency, mean fwd_ret_12_5m, hit rate, HAC-adjusted t-stat
  - Acceptance: 256-row DataFrame saved as `reports/phase5/rules_all.csv`
- [x] **T5.3** Rank by composite score: `frequency × |mean_return| × (hit_rate - 0.5)`
  - Acceptance: top-20 extracted; bottom-20 also extracted (largest negative edges)
- [x] **T5.4** Manually inspect top-20 rules; remove duplicates / trivially-related rules
  - Acceptance: ≤20 distinct rules in survivor list

### 5B. Sub-analysis 5b — LightGBM feature importance

- [x] **T5.5** Prepare features (~80 cols across all 4 TFs) and target (sign of `ret_fwd_12_5m`)
  - Acceptance: train DataFrame ready, target balanced check (~50/50 sign distribution baseline)
- [x] **T5.6** ⚠ Walk-forward train: in-sample years 1–3, predict on year 4; NEVER random shuffle
  - Acceptance: walk-forward routine implemented; sample split timestamps logged
- [x] **T5.7** Train shallow LightGBM (max_depth=4, n_estimators=200, learning_rate=0.05)
  - Acceptance: model trained; AUC reported on year-4 hold-out (in-sample-test only here, NOT final test)
- [x] **T5.8** Compute permutation importance on year-4 data
  - Acceptance: top-15 features ranked
- [x] **T5.9** Compute SHAP values for top-5 features; inspect interaction effects
  - Acceptance: SHAP summary plot saved; one notable interaction documented
- [x] **T5.10** Inspect tree's top splits (depth 1–2 of root trees)
  - Acceptance: top-3 splits documented — these are the model's "first questions"

### 5C. Sub-analysis 5c — Failure / inversion analysis ⚠

This is where the real insight lives. Don't shortcut it.

- [x] **T5.11** For each top-5 rule from 5a, separate fired bars into "worked" vs "failed" based on fwd_ret_12 sign matching expected direction
  - Acceptance: per-rule (n_worked, n_failed, baseline_hit_rate)
- [x] **T5.12** For each rule, compute mean-difference of every other feature between worked vs. failed cases
  - Acceptance: ranked list of differentiating features (by effect size, with HAC SE)
- [x] **T5.13** Propose filter candidates: features that separate worked from failed by >0.5 std
  - Acceptance: per-rule filter candidate list
- [x] **T5.14** Re-measure rule edge **with each candidate filter applied**; check if hit rate / mean return improves materially
  - Acceptance: before/after comparison table for each rule

### 5D. Phase 5 reports

- [x] **T5.15** Write `reports/phase5_rules_top20.md`
  - Acceptance: top-20 rules, ranked, with frequency / edge / hit rate
- [x] **T5.16** Write `reports/phase5_ml_importance.md`
  - Acceptance: feature importance, SHAP plots, top splits
- [x] **T5.17** Write `reports/phase5_failure_analysis.md`
  - Acceptance: per-rule failure analysis + filter candidates
- [x] **T5.18** Resolve D9 (top-K survivors carried to Phase 6)
  - Acceptance: explicit list of K ≤ 10 patterns chosen for OOS test

---

## Phase 6 — Validation 🔒 (Day 16)

This is the final test. Touch the test set ONCE. No iteration.

- [ ] **T6.1** Take the K surviving patterns from T5.18; do NOT modify them
- [ ] **T6.2** Apply each pattern to the **validation split (year 4 = 2024)** — note: year 4 has already been used in Phase 5 ML hold-out, but the rule-based patterns from 5a were not tuned to year 4 directly. Acceptable; document this caveat.
- [ ] **T6.3** ⚠ Apply each pattern to the **test split (year 5 = 2025+)** — never touched before
  - Acceptance: edge metrics computed once and recorded
- [ ] **T6.4** Compare in-sample vs validation vs test edges; flag patterns whose edge collapsed
  - Acceptance: 3-column comparison table per pattern
- [ ] **T6.5** Random feature baseline: for each surviving pattern, shuffle one constituent feature column, re-run; expect edge → 0
  - Acceptance: shuffled-feature edge documented per pattern
- [ ] **T6.6** Year-by-year stability check: compute the pattern's edge per calendar year (2021–2025)
  - Acceptance: 5-year edge sequence per surviving pattern; flag any pattern carried by a single year
- [ ] **T6.7** Time-of-day baseline check: compare pattern firings to the time-of-day distribution they fire at; rule out "pattern = first hour drift"
  - Acceptance: time-of-day baseline computed per pattern
- [ ] **T6.8** Write `reports/phase6_validation.md` listing ONLY patterns that survive all three baselines
  - Acceptance: report exists; honest negative result reported if no survivors

---

## Phase 7 — Synthesis (Day 17)

- [ ] **T7.1** Write `FINDINGS.md` answering the 5 synthesis questions from DESIGN_DOC §9.2:
  - Q1: Within each TF, which indicator state most reliably precedes directional moves?
  - Q2: Does B lead A or A lead B at each TF?
  - Q3: When indicators agree on a single TF, is the edge real?
  - Q4: When indicators agree across TFs, is the edge stronger? By how much?
  - Q5: What's the best rule discovered? What's its failure mode? What filter improves it?
  - Acceptance: each question answered in 2-5 paragraphs with quantitative backing
- [ ] **T7.1b** Hypothesis verdict table: one row per H1–H14 with verdict (confirmed / rejected / inconclusive), effect size, and the report section containing the evidence
  - Acceptance: all 14 hypotheses have explicit verdicts; rejections are written up with the same care as confirmations
- [ ] **T7.2** Add explicit "negative findings" section — what did NOT work, and why this is itself useful
  - Acceptance: ≥3 negative findings documented
- [ ] **T7.3** Update PLAN §3 decision log with all resolved decisions
  - Acceptance: D1–D10 all marked resolved
- [ ] **T7.4** Add "Future work" section listing follow-ups for which the data now exists
  - Acceptance: 3-5 concrete follow-up project ideas

---

## Optional / Stretch (💡)

- [ ] **T8.1** 💡 Add INDIAVIX as a fifth context dimension; redo Phase 4 with VIX-regime stratification
- [ ] **T8.2** 💡 Add NIFTY futures volume as a separate feature (futures DO have meaningful volume); see if it adds signal to top patterns
- [ ] **T8.3** 💡 Plotly interactive dashboard for the master dataset — useful for ad-hoc exploration
- [ ] **T8.4** 💡 Document a "lessons learned" appendix in FINDINGS.md about MTF research methodology — useful for your future projects
- [ ] **T8.5** 💡 Open-source the framework (without the conclusions); could become a portfolio piece

---

## Completion checklist (for FINDINGS.md handoff)

- [ ] All 🔒 gates passed (T2.22, T6 complete)
- [ ] All ⚠ tasks done correctly (revisit if any uncertainty)
- [ ] All deliverables in `reports/` exist
- [ ] `requirements.txt` and random seeds recorded
- [ ] `FINDINGS.md` answers all 5 synthesis questions
- [ ] Negative findings documented
- [ ] PLAN decision log fully updated

---

## End of task.md
