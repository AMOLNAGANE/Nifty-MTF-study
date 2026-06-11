# plan.md — Multi-Timeframe Momentum Confluence Study

**Companion to:** `design.md` (defines WHAT and WHY)
**This document defines:** HOW we execute, sequencing, dependencies, decision points, risks
**Companion to:** `task.md` (atomic task checklist with acceptance criteria)

---

## 1. Logic Audit of design.md

Before execution, the design was audited for correctness. Findings:

### 1.1 ✅ Already correctly handled

| Issue | Where addressed | Verdict |
|---|---|---|
| Bar-OPEN vs bar-CLOSE timestamp convention | §3.1 | Correctly identifies NSE convention, mandates `bar_close` for all joins |
| No-look-ahead in cross-TF joins | §4.6 | `merge_asof` on `bar_close` with `allow_exact_matches=True` is the right operation |
| EMA warmup distortion | §4.4 | Drops first 270 bars per TF; correctly notes the 1d cost (~13 months) |
| Overlapping forward returns inflating significance | §8.3 | Newey-West + block-bootstrap; explicitly warns against naive t-stats |
| Multiple-testing inflation | §8.2 | Bonferroni for survivors, FDR for exploration |
| 1h bar stub (15:15–15:30) | §3.2 | Decision: drop. Documented |

### 1.2 ⚠ Additional clarifications added by this plan

| # | Item | Action |
|---|---|---|
| L1 | **[CONFIRMED by user — volume column is all zeros across the dataset.]** NIFTY 50 is an **index**, not a tradable instrument, so this is structural, not a data error. | **Rule:** Volume column is dropped at load time (assert all-zero, then drop). Volume is NOT used as a feature. Any volume-derived idea (e.g. volume-weighted ROC) is out of scope. If volume signal is desired later, use **NIFTY futures** data, not the index. |
| L2 | Some 1d bars may be holiday-truncated (Diwali Muhurat trading is ~1 hour). | Detect days with <60 5m bars; either drop or flag with `is_irregular_session=True`. Default: drop from 1d analysis. |
| L3 | Indicator B's `roc = 100 * (gap - gap_prev) / gap_prev` is mathematically unstable when `gap_prev` is small (gap crossing zero from either side). Setting it to 0 (as Pine does) **biases the distribution toward zero** near regime changes — exactly when we care most about ROC behavior. | Augment with `B_roc_invalid` flag (already in §2.3). In all stats, drop rows where `abs(gap_prev) < small_epsilon` rather than treat them as ROC=0. Choose `epsilon` empirically (e.g. 1st percentile of `abs(gap)`). |
| L4 | The `master_5m.parquet` will have one 1d row repeated across ~75 5m rows in the same trading day. Naive correlation analysis between 5m features and 1d features will treat these 75 copies as 75 independent observations. | When computing **statistics on higher-TF features**, the effective sample size is the number of unique higher-TF bars, not the number of 5m rows. Document this in every Phase 4 report. |
| L5 | **Logic error fixed in design §5.3 (lead-lag).** The original framing — "compute cross-correlation between A and B; whichever leads is the early signal" — was flawed. Both indicators are deterministic smoothed transforms of the *same* close series; A (EMA 12/26) has less filter lag than B (EMA 40/90), so A will *mechanically* lead B in cross-correlation. That lag is filter math, not information. | §5.3 rewritten: (1) measure the mechanical lag as descriptive context only; (2) the informational test is **incremental predictive power** — does B add to A in predicting future price (nested regression ΔR², joint state tables); (3) the legitimate use of the lag structure is **turn-confirmation timing** — A triggers, B's confirmation/refusal classifies the trigger (H8–H10 in the catalog). If your code ever shows B leading A, treat it as a bug signal. |
| L6 | "Find the pattern" was under-specified as content (the design described test machinery but not *which* patterns to test). | Design §2.4 now contains a **Pattern Hypothesis Catalog (H1–H14)** — 14 concrete, falsifiable hypotheses covering indicator-vs-price (H1–H7), indicator-vs-indicator (H8–H10), and cross-TF confluence (H11–H14). Every Phase 3–5 analysis now has explicit hypotheses to confirm or reject, and `FINDINGS.md` reports a verdict per hypothesis. |

### 1.3 No issues found

The brainstorming framework (§1), indicator specs (§2), feature derivations (§2.1, §2.2), pattern hierarchy decomposition, target families (§5.1), confluence scoring (§6.3), and walk-forward splits (§8.1) are all methodologically sound for a research-only study.

---

## 2. Execution Strategy

### 2.1 Top-level approach

Six phases, executed strictly in order. Each phase produces concrete artifacts (parquet, markdown, plots) that the next phase consumes. **Do not begin Phase N+1 until Phase N's outputs pass their validation gate.**

### 2.2 The critical path

```
Phase 1 (Data) ──► Phase 2 (Features) ──► no-look-ahead unit test (GATE 1)
                                              │
                                              ▼
                            Phase 3 (Intra-TF stats) ─┐
                                                      ├─► Phase 5 (ML + rules)
                            Phase 4 (Inter-TF stats) ─┘         │
                                                                ▼
                                                    Phase 6 (Validation GATE 2)
                                                                │
                                                                ▼
                                                        FINDINGS.md
```

**Gate 1 (after Phase 2):** No-look-ahead unit test must pass on 1,000 random 5m bars. If it fails, stop and fix — no exception.

**Gate 2 (after Phase 6):** A pattern is only reported in `FINDINGS.md` if it survives the validation split AND beats the random-feature baseline by a meaningful margin.

### 2.3 Milestones and timeline

Assuming ~3-4 focused hours per day:

| Milestone | Deliverable | Estimated effort | Cumulative |
|---|---|---|---|
| M1: Data loaded + resampled | 4× `nifty_{tf}.parquet` + `data_quality_report.md` | 1.5 days | 1.5 d |
| M2: Features computed | 4× `features_{tf}.parquet` + `master_5m.parquet` | 1.5 days | 3 d |
| **M3: GATE 1 passed** | No-look-ahead unit test passes | 0.5 day | 3.5 d |
| M4: Phase 3 complete | 4× `phase3_intra_{tf}.md` + `phase3_summary.md` | 4 days | 7.5 d |
| M5: Phase 4 complete | `phase4_inter_tf.md` | 3 days | 10.5 d |
| M6: Phase 5 complete | `phase5_rules_top20.md` + `phase5_ml_importance.md` + `phase5_failure_analysis.md` | 4 days | 14.5 d |
| **M7: GATE 2 passed** | `phase6_validation.md` with surviving rules | 1 day | 15.5 d |
| M8: Final synthesis | `FINDINGS.md` | 1 day | 16.5 d |

**Total: ~16-17 working days.** Add buffer; realistic completion is 3-4 calendar weeks at part-time pace.

---

## 3. Decision Points

Decisions you must make at each phase boundary. Pre-decide where possible; defer where the data should inform the choice.

### 3.1 Before Phase 1 — pre-decide

| # | Decision | Default | Why this default |
|---|---|---|---|
| D1 | Drop or keep 1h stub bar (15:15–15:30)? | Drop | Cleaner; cost is small |
| D2 | Include or exclude expiry-day bars in main analysis? | Include in main, also produce **separate** `expiry_only` analysis | Expiry behavior is genuinely different but excluding loses ~20% of data |
| D3 | Use `gap` raw or `gap_norm = gap/close`? | Compute **both**, prefer `gap_norm` for cross-time comparisons | Index level moved 1.8× over 5 years; raw gap is non-stationary |
| D4 | Compute 1d return as close-to-close or open-to-close? | **Open-to-close** for intraday analysis; close-to-close as secondary | Overnight gaps are dominated by global news, unrelated to your indicators |

#### D1-D4 — Resolved (2026-06-11)

- **D1 (1h stub bar): Resolved as planned.** `src/phase1_data/resampler.py:49` defines
  `_STUB_CUTOFF_MINS = 360` (09:15 + 360min = 15:15) and drops the 15:15-15:30 stub bar from the
  1h resample, matching the "Drop" default exactly.
- **D2 (expiry-day bars): Partially resolved — deviation.** Only the "include in main analysis"
  half of the default was implemented; expiry-day bars are present in all Phase 3-6 analyses with
  no special handling. The "also produce a separate `expiry_only` analysis" half was **never
  implemented** (no `expiry` column, flag, or report exists anywhere in `src/` or `reports/`).
  Given Phase 5/6 already found that the dominant sources of edge instability are year-to-year
  variance (R5/R10) and time-of-day effects (R3, T6.7), and that 8/10 carried rules failed OOS for
  those reasons, an expiry-day-specific analysis was deprioritized. This is documented as a
  deviation rather than silently dropped; FINDINGS.md §4 (Future Work) covers the regime-related
  follow-ups that would subsume it (an expiry-day flag could be added alongside the VIX
  stratifier in T8.1).
- **D3 (gap vs gap_norm): Resolved as planned — both computed.** `src/phase2_features/
  indicator_b.py` computes raw `B_gap` and `out["B_gap_norm"] = gap / close` (line 47).
  `B_gap_norm` (and its z-scored derivative `B_zroc`) is the column used throughout Phase 3-6's
  cross-time/cross-TF comparisons, per the "prefer gap_norm" guidance.
- **D4 (1d return convention): Resolved — deviation, documented honestly.** The default
  recommended open-to-close for 1d with close-to-close as a secondary check. In the implementation,
  `src/phase3_intra/targets.py` computes `ret_fwd_N = (close.shift(-N) / close) - 1` uniformly for
  **all four timeframes including 1d** — i.e. **close-to-close was used as the sole 1d convention**,
  not as a secondary. This means 1d forward returns include overnight/weekend gap risk that the
  original design wanted to exclude from the "indicator signal" measurement. No open-to-close
  variant was computed. Impact assessment: despite this, the 1d `A_state` results (H1, Q1) were
  directionally consistent and reached `p_val_adj < 0.05` at multiple horizons, so the gap-inclusive
  convention does not appear to have been a confound for the headline 1d findings — but a future
  re-run with `ret_fwd_N_intraday = (close.shift(-N)/open.shift(-N+1))-1`-style open-to-close
  returns would be needed to confirm this rigorously (see FINDINGS.md §4).

### 3.2 After Phase 1 — data-informed decisions

| # | Decision | Inputs needed |
|---|---|---|
| D5 | Holiday-truncated day threshold (drop if <X bars) | Histogram of bars-per-day across 5 years |
| D6 | The epsilon for `B_roc_invalid` flag (L3 above) | 1st-percentile of `abs(gap_prev)` per TF |

#### D5-D6 — Resolved (2026-06-11)

- **D5 (holiday-truncated day threshold): Resolved as "no filter applied" — deviation from L2's
  stated default.** `bars_per_day = df_5m.groupby("session_date").size()` is computed in
  `src/phase1_data/run_phase1.py:106` and reported descriptively in
  `reports/phase1_data_quality.md` (mean=74.5, min=6, max=75 bars/day), but grepping
  `src/phase1_data/resampler.py` and `run_phase1.py` for any drop/filter logic on this quantity
  found none — **no day was dropped, and no `is_irregular_session` flag was added.** All ~1,244
  trading days, including the handful of Diwali Muhurat-trading short sessions (~6 bars instead
  of ~75), are retained in every downstream feature and analysis. Post-hoc rationale: these
  truncated days represent roughly 5 days out of 1,244 (~0.4%) — an immaterial share of the
  thousands-to-tens-of-thousands-row samples used in Phase 3-6 — and a short session still
  produces valid OHLC bars with no NaN/look-ahead issue, only a different intraday volatility
  shape for that one day. The deviation is judged low-impact and is documented here rather than
  retroactively re-run.
- **D6 (`B_roc_invalid` epsilon): Resolved as planned.** `src/phase2_features/indicator_b.py:38`
  computes `epsilon = float(abs_prev.quantile(0.01)) if len(abs_prev) > 0 else 0.0` — the 1st
  percentile of `abs(gap_prev)`, computed independently per TF (since `indicator_b.py` runs once
  per TF's DataFrame), exactly matching the planned approach.

### 3.3 After Phase 3 — data-informed decisions

| # | Decision | Inputs needed |
|---|---|---|
| D7 | Forward-return horizons to carry into Phase 4 | Which N values showed largest/cleanest signal in Phase 3a |
| D8 | Whether to add VIX/INDIAVIX as a regime stratifier | Whether Phase 3 results show variance regime dependence |

#### D7-D8 — Resolved (2026-06-11)

- **D7 (forward-return horizon to carry into Phase 4): Resolved — `ret_fwd_12`.** Phase 3a
  computed all five horizons (`ret_fwd_1, 3, 6, 12, 24`, `src/phase3_intra/run_phase3.py:40`) per
  TF. `ret_fwd_12` was selected as the de facto standard horizon for Phase 4-6 (used in
  `phase4_inter_tf.md`'s stratified tables, `phase5_mining`'s `evaluate_rules(...,
  ret_col="ret_fwd_12")`, and all of `phase6_validation.md`'s comparison/year-by-year/ToD tables)
  because, per FINDINGS.md Q1, it produced the most `p_val_adj < 0.05` `A_state` cells across the
  most TFs (15m `bull_accel`/`bull_decel`, 1h `bull_accel`, 1d `bull_accel`/`bear_accel`), and at
  5m/15m it corresponds to roughly a 1-3 hour look-ahead — long enough to filter single-bar noise
  while remaining within "intraday momentum" rather than crossing into multi-day drift.
- **D8 (VIX/INDIAVIX regime stratifier): Resolved — not added.** Phase 3 did not surface variance-
  regime dependence strong enough to justify adding an INDIAVIX-based stratifier before Phase 4/5;
  the COVID-era variance risk (R5) was instead handled via the year-by-year breakdowns built
  directly into Phase 6 (T6.6). Those year-by-year tables *do* show some carried rules with
  outsized single-year contributions (e.g. rule 140's `mean_ret_2022 = 0.005975` vs.
  `0.0001-0.0004` in 2023-2025), which a volatility-regime stratifier could help explain. T8.1
  (VIX/INDIAVIX stratifier) remains an optional/stretch task and is carried forward as
  FINDINGS.md §4 Future Work item 3.

### 3.4 After Phase 5 — data-informed decisions

| # | Decision | Inputs needed |
|---|---|---|
| D9 | Top-K rules to carry into validation (Phase 6) | Phase 5a ranking |
| D10 | Whether to extend the study (e.g. add RSI, ADX) | Whether Phase 5 found compelling patterns or surfaced clear gaps |

#### D9 — Resolved (2026-06-11)

Phase 5a ranked all 256 confluence rules on the rule-discovery set (`bar_close.dt.year <= 2023`,
n=58,080) by `composite = frequency * |mean_ret| * (hit_rate - 0.5)`; 15 survivors remained after
Hamming-deduplication (min_hamming=2, see `reports/phase5/rules_survivors.csv`). The top 10 by
composite are the K=10 patterns carried into Phase 6 for OOS re-measurement on 2024 + 2025:

| rank | rule_id | active (bullish) flags | n | mean_ret | hit_rate | t_stat | p_val_bonf |
|---|---|---|---|---|---|---|---|
| 1 | 151 | A_5m, B_5m, A_15m, A_1h, B_1d | 324 | 0.001436 | 0.682 | 3.38 | 0.184 |
| 2 | 121 | A_5m, B_15m, A_1h, B_1h, A_1d | 211 | 0.001894 | 0.692 | 3.69 | 0.057 |
| 3 | 26 | B_5m, B_15m, A_1h | 76 | 0.003406 | 0.724 | 1.91 | 1.000 |
| 4 | 0 | (none -- all 8 flags bearish/NaN) | 722 | 0.000534 | 0.637 | 1.75 | 1.000 |
| 5 | 140 | A_15m, B_15m, B_1d | 133 | 0.002254 | 0.669 | 2.17 | 1.000 |
| 6 | 31 | A_5m, B_5m, A_15m, B_15m, A_1h | 687 | 0.000584 | 0.617 | 1.18 | 1.000 |
| 7 | 182 | B_5m, A_15m, A_1h, B_1h, B_1d | 160 | 0.001244 | 0.681 | 2.09 | 1.000 |
| 8 | 211 | A_5m, B_5m, A_1h, A_1d, B_1d | 51 | 0.001944 | 0.843 | 1.85 | 1.000 |
| 9 | 231 | A_5m, B_5m, A_15m, B_1h, A_1d, B_1d | 184 | 0.000930 | 0.696 | 2.18 | 1.000 |
| 10 | 181 | A_5m, A_15m, A_1h, B_1h, B_1d | 176 | 0.001469 | 0.619 | 2.06 | 1.000 |

Caveats carried forward from 5c (`reports/phase5_failure_analysis.md`):
- Only rules 151 and 121 survive Bonferroni correction (`p_val_bonf` < 0.2); the remaining 8 are
  exploratory and should be reported as such in Phase 6, not treated as pre-validated edges.
- **Rule 0** ("no confluence flags active") is partly an artifact of the dataset's first ~year, where
  the 1d-timeframe indicators are not yet computable (`B_roc_invalid_1d` / `B_zroc_extreme_1d` = 1 for
  60% of "worked" rows vs. 0% of "failed" rows in the 5c worked/failed split). Its 2024/2025 hit rate
  (where this warm-up period does not recur) may differ materially from the in-sample 0.637 -- report
  it separately in Phase 6 rather than pooling with the other 9 rules.
- For rules 151, 121, 26, 0 and 140, 5c also identified single-feature filters that materially
  improve in-sample `hit_rate` / `mean_ret` (e.g. rule 151 + `A_signal_1d < 73.77` raises hit_rate from
  0.682 to 0.837, n=324->135; full tables in `reports/phase5/failure_rule{id}_filters_before_after.csv`).
  These filtered variants are optional secondary checks in Phase 6 -- the unfiltered top-10 above are
  the primary D9 carry-set.

#### D10 — Resolved (2026-06-11)

**Decision: do not extend the study with new indicators (e.g. RSI, ADX) at this time.**

Phase 5/6 results show the bottleneck in this study is **OOS robustness, not signal availability**:
the existing 2-indicator (A=MACD, B=gap-ROC) x 4-TF x 256-rule confluence space already produced
more candidates (15 Hamming-deduped survivors, 10 carried into Phase 6) than could be validated --
only 2/10 (rules 140 and 31) survived all three Phase 6 checks (`reports/phase6_validation.md`
T6.8). Adding a third or fourth indicator family would multiply the rule space combinatorially
(256 -> several thousand for 3 indicators x 4 TFs) while Phase 6 already demonstrated that 80% of
rules from the *smaller* space failed to generalize past the discovery window.

Separately, FINDINGS.md negative finding #3 (H8) found that indicator B itself adds essentially no
incremental predictive power over indicator A (`ΔR² <= 0.0018` at every TF, B's own `B_state`
never reaching `p_val_adj < 0.05` at any TF/horizon) -- i.e. even the *current* 2-indicator
framework is past the point of diminishing returns for NIFTY 50 at these timeframes.

Recommended next steps instead (detailed in FINDINGS.md §4 Future Work) are: (1) build a
cost-aware backtest for rules 140/31 specifically, (2) re-validate rule 140's
`B_zroc_1d > 0.1698` filter OOS, (3) add the D8/T8.1 VIX regime stratifier to the *existing*
indicators to explain the year-to-year instability already observed, and (4) if a genuinely new
signal family is wanted, prioritize **volume-based confirmation on NIFTY futures data** (L1) over
additional price-derived oscillators, since oscillators derived from the same close series tend to
be highly collinear with A/B (as H8 demonstrated for B vs. A).

---

## 4. Risk Register

Risks ranked by impact × likelihood. Mitigation is built into design where possible.

| ID | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| R1 | No-look-ahead bug leaks future data into features | **Critical** — invalidates entire study | Medium | Phase 2 unit test (Gate 1); manual spot-check of 5 random rows |
| R2 | Overfitting to in-sample patterns | High | High | Strict train/val/test split; Phase 6 OOS validation; random-feature baseline |
| R3 | Found patterns are time-of-day artifacts | Medium | High | §8.4 explicitly tests against intraday-drift baseline |
| R4 | Indicator B numerical instability near `gap_prev ≈ 0` | Medium | Certain | L3 above — `B_roc_invalid` flag, drop from stats |
| R5 | 5-year window includes COVID (2020-21 anomaly) | Medium | Certain | Stratify Phase 3/4 by year; report year-by-year breakdown for top patterns |
| R6 | NIFTY weekly expiry day changes (was Thursday, now Tuesday) introduces structural break | Medium | Certain | Flag expiry day, run sensitivity check; consider segmenting by expiry-day-of-week |
| R7 | Compute time blows up in Phase 4 (8 features × 4 TFs × 256 boolean combos × bootstrap) | Low-Medium | Medium | Use vectorized ops + numba where needed; pre-compute aggregation tables |
| R8 | EMA(90) on 1d data loses first ~13 months of 1d-conditional analysis | Medium | Certain | Documented in §4.4; effective train window is years 2-3 for 1d-conditional, not 1-3 |
| R9 | Scope creep — adding indicators, sub-questions, or trading logic mid-study | Medium | Medium | §12 of design doc lists explicit non-goals; revisit before any extension |
| R10 | Patterns survive validation by luck (one lucky year) | Medium | Medium | Year-by-year sub-period stability check on every survivor |

---

## 5. Tooling & Environment

### 5.1 Python stack (recommended)

| Purpose | Library | Notes |
|---|---|---|
| Data IO | `pandas`, `pyarrow` | Parquet for type-safe persistence |
| EMA / TA | `pandas.ewm` or `talib` | `talib` is faster; `pandas.ewm(adjust=False)` matches Pine's EMA |
| Statistics | `scipy.stats`, `statsmodels` | `statsmodels` has Newey-West (`OLS().fit(cov_type='HAC', cov_kwds={'maxlags': N})`) |
| Block bootstrap | `arch.bootstrap.StationaryBootstrap` | For top-K rule confidence intervals |
| ML | `lightgbm`, `shap` | Shallow trees (max_depth=4); SHAP for interpretation |
| Plots | `matplotlib`, `seaborn`, `plotly` | Plotly for interactive heatmaps |
| Notebooks | `jupyter` | For exploratory work; final reports in `.md` |

### 5.2 Pine-equivalence note for EMAs

Pine's `ta.ema(x, n)` uses `alpha = 2 / (n+1)` with seeding from first available value, no Wilder smoothing. To match:

```python
df['ema_n'] = df['close'].ewm(span=n, adjust=False).mean()
```

`adjust=False` is critical. With `adjust=True` (the pandas default), early values are weighted differently and won't match Pine.

**Validate this once:** export 1000 bars from TradingView with the indicator plotted, recompute in Python, check that values agree to 4 decimal places after the warmup period.

### 5.3 Repository structure (suggested)

```
nifty_mtf_study/
├── design.md
├── plan.md
├── task.md
├── FINDINGS.md                  # produced at end
├── data/
│   ├── raw/                     # original 5m CSV
│   ├── processed/               # parquet outputs from Phase 1
│   └── features/                # parquet outputs from Phase 2
├── src/
│   ├── phase1_data/
│   ├── phase2_features/
│   ├── phase3_intra/
│   ├── phase4_inter/
│   ├── phase5_mining/
│   └── phase6_validation/
├── tests/
│   └── test_no_lookahead.py     # the Gate 1 test
├── reports/
│   ├── phase1_data_quality.md
│   ├── phase3_intra_{tf}.md     # 4 files
│   ├── phase4_inter_tf.md
│   ├── phase5_*.md              # 3 files
│   └── phase6_validation.md
└── notebooks/
    └── exploration.ipynb        # scratch work, not deliverable
```

---

## 6. Success Criteria

The study is "complete" when all of the following are true:

1. **Reproducibility:** Re-running all phases from raw data produces identical outputs (set random seeds; record library versions).
2. **No-look-ahead proven:** Gate 1 unit test passes; one manual spot-check documented in `phase2_validation_log.md`.
3. **Every phase deliverable produced** as listed in §9 of design doc.
4. **`FINDINGS.md` answers all 5 synthesis questions** (§9.2 of design doc) in plain language, with quantitative backing.
5. **At least one surviving pattern OR an honest negative result.** Negative results are useful too — the study's value is understanding, not finding edges. If nothing survives Phase 6, that itself is the finding.
6. **Year-by-year stability** reported for every surviving pattern.
7. **All open questions** (§10 of design + §3 here) have explicit resolutions logged.

---

## 7. What Comes After This Study

Out of scope for this project, but worth noting for sequencing:

1. **If a pattern survives:** a separate project to build a backtest with transaction costs, slippage, and risk sizing. New design doc required.
2. **If features look promising for options work:** integrate the surviving features into your existing NIFTY options ML pipeline (regime classifier → strategy selector → exit manager) as conditioning inputs.
3. **If multiple TFs show different dynamics:** consider extending to a 5-TF setup (add a weekly or 4h frame) — but only if the marginal information value is clear from this study.
4. **Indicator parameter optimization** (e.g. is 40/90/7 truly the best slow setup?) — explicitly deferred. Parameter mining without a hold-out set is one of the easiest ways to overfit. If pursued, requires a separate study with a strictly held-out test set never touched until final report.

---

## 8. How to Use These Three Documents Together

| Document | When to read | What it answers |
|---|---|---|
| `design.md` | Before starting; reference throughout | What is this study? Why these methods? |
| `plan.md` (this) | Before starting Phase 1; revisit at each phase boundary | How do I execute? What can go wrong? What decisions do I owe? |
| `task.md` | Daily | What's the next atomic thing to do? |

Update cadence:
- `design.md` — update only if scope changes (rare; should be deliberate)
- `plan.md` — update when a decision is resolved (move from §3 "Decision Points" to a decisions-log section)
- `task.md` — update continuously as you check off tasks

---

## End of plan.md
