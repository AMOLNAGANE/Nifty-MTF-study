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

### 3.2 After Phase 1 — data-informed decisions

| # | Decision | Inputs needed |
|---|---|---|
| D5 | Holiday-truncated day threshold (drop if <X bars) | Histogram of bars-per-day across 5 years |
| D6 | The epsilon for `B_roc_invalid` flag (L3 above) | 1st-percentile of `abs(gap_prev)` per TF |

### 3.3 After Phase 3 — data-informed decisions

| # | Decision | Inputs needed |
|---|---|---|
| D7 | Forward-return horizons to carry into Phase 4 | Which N values showed largest/cleanest signal in Phase 3a |
| D8 | Whether to add VIX/INDIAVIX as a regime stratifier | Whether Phase 3 results show variance regime dependence |

### 3.4 After Phase 5 — data-informed decisions

| # | Decision | Inputs needed |
|---|---|---|
| D9 | Top-K rules to carry into validation (Phase 6) | Phase 5a ranking |
| D10 | Whether to extend the study (e.g. add RSI, ADX) | Whether Phase 5 found compelling patterns or surfaced clear gaps |

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
