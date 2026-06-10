# Multi-Timeframe Momentum Confluence Study — NIFTY 50

**Author:** Amol
**Status:** Design v1.1 (research-only, no trading deployment; revised after logic audit)
**Dataset:** NIFTY 50, 5-minute OHLCV, ~5 years
**Indicators:** MACD (12/26/9, standard) + EMA-Gap-ROC (40/90/7, z-score 100)
**Timeframes:** 5m, 15m, 1h, 1D
**Goal:** Build empirical understanding of how these two indicators behave intra- and inter-timeframe, and which configurations precede directional momentum in 5m NIFTY.

---

## 0. Executive Summary

This study answers two questions:

1. **Intra-timeframe:** Within a single timeframe, how does each indicator relate to NIFTY price, and how do the two indicators relate to each other?
2. **Inter-timeframe:** When lower-timeframe indicators (5m, 15m, 1h) are in specific configurations, do they predict price/indicator behavior at higher timeframes — and conversely, does higher-timeframe state condition lower-timeframe edges?

The end deliverable is **understanding**, not a tradeable signal. We use a three-stage methodology: **statistical → visual → ML**, with the ML stage used purely to surface non-obvious feature interactions, not to deploy a model.

---

## 1. Brainstorming Framework Used

Two complementary techniques structure this design.

### 1.1 Pattern Hierarchy Decomposition (top-down)

"Find patterns" is too vague to act on. We decompose it into six explicit levels, each of which becomes a research sub-question:

| Level | Pattern type | Example | Phase |
|---|---|---|---|
| L0 | Raw state | `hist > 0` | Phase 2 |
| L1 | Single-indicator dynamics | `hist rising AND hist > 0` (bullish accelerating) | Phase 3a |
| L2 | Indicator vs. price | Divergence, lead-lag, confirmation | Phase 3b |
| L3 | Indicator vs. indicator (same TF) | A bullish + B bullish agreement | Phase 3c |
| L4 | Cross-timeframe alignment | 1h B bullish AND 5m A bullish-accelerating | Phase 4 |
| L5 | Confluence rules → forward returns | Boolean/score combinations of L1–L4 | Phase 5 |

Every analysis in the study maps to one of these levels. This prevents scope creep and ensures we produce concrete answers, not vague heatmaps.

### 1.2 Inversion (Munger)

For every "what causes momentum?" question, we explicitly also ask **"when does the same setup FAIL?"** Failure cases are usually more diagnostic than success cases. Phase 5c is dedicated to this.

---

## 2. Indicator Specifications (locked)

### 2.1 Indicator A — Standard MACD (`MACD_A`)

```
fast_ema     = EMA(close, 12)
slow_ema     = EMA(close, 26)
macd         = fast_ema - slow_ema
signal       = EMA(macd, 9)
hist         = macd - signal
```

**Derived features (computed per TF):**

| Feature | Formula | Meaning |
|---|---|---|
| `A_macd` | macd | Trend strength + direction |
| `A_signal` | signal | Smoothed trend |
| `A_hist` | hist | Momentum (acceleration of macd) |
| `A_hist_slope` | hist - hist[1] | Rate of change of momentum (jerk) |
| `A_macd_sign` | sign(macd) | Above/below zero |
| `A_hist_sign` | sign(hist) | Bullish/bearish momentum |
| `A_macd_zero_cross` | +1 if macd crossed 0 upward in last bar, -1 if downward, 0 otherwise | Trend regime change (signed) |
| `A_hist_zero_cross` | +1 / -1 / 0 (same convention as above) | Momentum regime change (signed) |
| `A_state` | one of {`bull_accel`, `bull_decel`, `bear_decel`, `bear_accel`} | 4-quadrant state |

The 4-state classification:
- `bull_accel`: hist > 0 and hist_slope > 0
- `bull_decel`: hist > 0 and hist_slope ≤ 0
- `bear_decel`: hist < 0 and hist_slope ≥ 0 (bearish but recovering)
- `bear_accel`: hist < 0 and hist_slope < 0

**Edge case (`hist == 0` exactly):** classify by hist_slope sign — if hist_slope > 0 treat as `bear_decel`, if hist_slope < 0 treat as `bull_decel`, if both are zero carry forward the previous bar's state. Rare with floating-point math; the rule prevents NaN states.

### 2.2 Indicator B — EMA-Gap ROC + zROC (`MACD_B`)

**Locked params:** Fast EMA = 40, Slow EMA = 90, ROC period = 7, z-score lookback = 100.

```
fast_ema = EMA(close, 40)
slow_ema = EMA(close, 90)
gap      = fast_ema - slow_ema
gap_prev = gap[7]
roc      = 100 * (gap - gap_prev) / gap_prev   if gap_prev != 0 else 0
zroc     = (roc - SMA(roc, 100)) / STDEV(roc, 100)
gap_acc  = gap - gap_prev   # used for histogram coloring in your Pine
```

**Derived features (computed per TF):**

| Feature | Formula | Meaning |
|---|---|---|
| `B_gap` | gap | Trend strength (raw EMA spread) |
| `B_gap_norm` | gap / close | Trend strength normalized to price |
| `B_roc` | roc | % change in gap over 7 bars |
| `B_zroc` | zroc | Standardized ROC — extreme readings flagged |
| `B_gap_acc` | gap - gap[7] | Same as Pine's `gapROC` (acceleration of gap) |
| `B_roc_slope` | roc - roc[1] | Is ROC itself accelerating? |
| `B_zroc_extreme` | 1 if abs(zroc) > 2 else 0 | Statistical extreme |
| `B_state` | 4-quadrant from (sign(roc), sign(roc_slope)), same edge-case rule as `A_state` | bull_accel / bull_decel / bear_decel / bear_accel |

### 2.3 Important note on ROC division-by-zero

Your Pine code returns `0` when `gap_prev == 0`. We will preserve this but **also add a flag** `B_roc_invalid` so we can exclude those rows from statistical tests rather than letting zeros pollute correlations.

### 2.4 Pattern Hypothesis Catalog — the brainstormed logic (H1–H14)

This catalog is the result of a structured brainstorm (Pattern Hierarchy from §1.1 crossed with classical momentum theory and Inversion from §1.2). Every hypothesis is stated as **definition → prediction → test → falsification**, so each one is directly codeable in Phases 3–5. The Phase 3/4/5 analyses are the *machinery*; this catalog is the *content* run through that machinery.

A note on roles, given the parameters: **A (12/26/9) is the trigger/momentum indicator; B (40/90, ROC 7) is the regime/trend indicator.** Almost every classical MTF pattern reduces to "trade A's triggers in the direction of B's regime." The catalog tests whether that folklore is empirically true for NIFTY.

#### Group 1 — Indicator A vs. price (intra-TF)

**H1 — Bullish momentum continuation (your core hypothesis, applied to A).**
- Definition: `A_state = bull_accel` (hist > 0 AND hist rising).
- Prediction: positive mean forward return over next 3–12 bars, above unconditional baseline.
- Test: Phase 3a state-conditional table, HAC t-stat, vs. baseline.
- Falsification: conditional mean ≤ baseline, or significance vanishes after time-of-day control.

**H2 — Histogram peak precedes price peak (momentum exhaustion timing).**
- Definition: transition `bull_accel → bull_decel` (hist > 0, slope flips negative) = local momentum peak.
- Prediction: price typically continues briefly then stalls; the hist peak leads the local price peak by a measurable median number of bars. Forward returns after this transition are positive but *decaying* with horizon (short horizon > long horizon).
- Test: for each hist peak, find the nearest subsequent local price high (find_peaks); report the lead-time distribution. Compare fwd_ret_3 vs fwd_ret_12 after the transition.
- Falsification: no consistent lead; or returns after the transition are flat/negative even at short horizons.

**H3 — Aligned vs. tired zero-cross.**
- Definition: `A_macd` crosses above zero (trend regime change). Split into (a) "aligned": hist > 0 and rising at the cross; (b) "tired": hist falling at the cross.
- Prediction: aligned crosses have materially better follow-through than tired crosses.
- Test: two-sample comparison of forward return distributions (a) vs (b).
- Falsification: no significant difference → the hist context adds nothing to the macd cross.

**H4 — First-pullback continuation.**
- Definition: macd > 0 (established uptrend) AND hist crosses below zero then back above zero (pullback completed) — the classic MACD pullback entry. Mirror for downtrends.
- Prediction: hist re-cross *conditioned on macd sign agreeing* outperforms unconditional hist re-crosses.
- Test: conditional table — hist zero-cross up, stratified by sign(macd).
- Falsification: macd sign doesn't stratify the re-cross edge.

#### Group 2 — Indicator B vs. price (intra-TF)

**H5 — Gap-ROC momentum (your stated core hypothesis, verbatim).**
- Definition: `B_roc > 0 AND B_roc_slope > 0` (gap expanding, expansion accelerating) = `B_state = bull_accel`.
- Prediction: NIFTY drifts higher over the following bars; edge strongest at this TF's "natural" horizon (~7–14 bars, matching the ROC period).
- Test: Phase 3a state-conditional table; decile analysis on B_roc.
- Falsification: flat or negative conditional returns; or monotonicity absent in deciles.

**H6 — zROC extremes: exhaustion vs. continuation (two competing sub-hypotheses, test both).**
- H6a (continuation): `zroc > +2` marks unusually strong trend impulse → further gains.
- H6b (exhaustion): `zroc > +2` marks overextension → mean reversion.
- Test: compare forward returns for zroc ∈ (0.5, 2] vs zroc > 2, both vs. baseline, at horizons {3, 6, 12, 24}. Often continuation at short horizon, exhaustion at long horizon — report the full term structure.
- Falsification: neither bucket distinguishable from baseline.

**H7 — Acceleration precedes the gap zero-cross (early regime turn).**
- Definition: `B_gap < 0` (bearish regime by level) but `B_gap_acc > 0` for M consecutive bars (gap closing). The EMA cross itself (gap zero-cross) is famously late.
- Prediction: the acceleration signal precedes the gap zero-cross by a median of K bars, and entering at the acceleration signal captures meaningfully more of the move than waiting for the cross.
- Test: event study — measure (i) lag from first sustained gap_acc>0 to gap zero-cross, (ii) price change in that window.
- Falsification: acceleration signals are mostly false starts (gap never crosses), making the "early" signal noise.

#### Group 3 — Indicator A vs. Indicator B (intra-TF)

**H8 — Mechanical lag, informational redundancy check.**
- Definition: per the corrected §5.3 — A mechanically leads B (shorter EMAs). The real question: does B carry incremental information about future price beyond A?
- Test: nested regression ΔR² and bivariate state tables (§5.3 items 2–3).
- Falsification (of B's usefulness): B's coefficient insignificant given A, AND B's column doesn't shift conditional means in the joint state table.

**H9 — Trigger-in-regime vs. trigger-against-regime.**
- Definition: A flips bullish. Split by B's regime: (a) B bullish (`roc > 0`), (b) B bearish.
- Prediction: (a) = trend continuation, larger and more persistent edge; (b) = counter-trend bounce, smaller and faster-decaying edge (or none).
- Test: forward-return term structure of A-flips stratified by B regime.
- Falsification: no stratification → B's regime context is worthless for A's triggers (would contradict most MTF folklore — a valuable negative result).

**H10 — Disagreement as regime-turn early warning.**
- Definition: A flips bullish while B is still bearish (state disagreement).
- Prediction: P(B turns bullish within K bars | A just flipped) is significantly higher than the unconditional P(B turns bullish in any K-bar window). I.e., A's flip is a leading indicator *of B's regime change* (mechanically expected) — and the subset where B then confirms has the best forward returns measured from A's flip (informational).
- Test: transition probabilities + the confirmation-split forward returns from §5.3 item 3.
- Falsification: confirmation subset no better than non-confirmation subset.

#### Group 4 — Cross-timeframe (the confluence logic)

**H11 — HTF regime + LTF trigger (the canonical MTF pattern).**
- Definition: 1h (or 1d) `B_state ∈ {bull_accel, bull_decel}` (higher-TF uptrend) AND 5m A flips bullish (lower-TF trigger).
- Prediction: 5m A-flips inside higher-TF B-uptrends have higher hit rate and mean return than (i) unconditional A-flips and (ii) A-flips against the higher-TF regime.
- Test: Phase 4a stratified matrices.
- Falsification: higher-TF state doesn't stratify the 5m trigger's outcomes.

**H12 — Confluence monotonicity.**
- Definition: `score_bull` ∈ [0, 8] as in §6.3.
- Prediction: mean forward 5m return is monotonically increasing in score; extreme scores (0 and 8) are rare but carry the largest absolute edges.
- Test: §6.3 bucket table + monotonicity plot.
- Falsification: flat or U-shaped curve → simple vote-counting is the wrong aggregation (and Phase 5b's tree will reveal what the right one is).

**H13 — Compression → expansion (squeeze analog).**
- Definition: `|B_gap_norm|` in its lowest decile (EMAs pinched = trendless compression) on the 15m or 1h frame, followed by `|zroc| > 1.5` (ROC expansion) — a volatility-expansion onset.
- Prediction: the *direction* of the zroc expansion predicts the direction of the subsequent 5m–15m move; compressed-then-expanding setups produce larger absolute moves than baseline.
- Test: event study on (compression, expansion-direction) events; compare |fwd_ret| and signed fwd_ret to baseline.
- Falsification: expansion direction is a coin flip for subsequent returns.

**H14 — Divergence reversal.**
- Definition: price makes a higher high but `A_hist` (or `B_roc`) makes a lower high across consecutive peaks (bearish divergence); mirror for bullish.
- Prediction: forward returns after divergence events skew opposite to the prior trend, more so at higher TFs (15m/1h) than at 5m (where divergence is mostly noise).
- Test: Phase 3d event study, per TF, per indicator.
- Falsification: post-divergence returns indistinguishable from baseline (a common honest finding in index data — worth knowing either way).

#### How the catalog maps to phases

| Hypotheses | Tested in |
|---|---|
| H1–H7 | Phase 3a (state/decile tables) + 3d (H2 peak timing, H14 setup) |
| H8–H10 | Phase 3b (corrected lead-lag) + 3c (joint state matrix) |
| H11–H13 | Phase 4a/4c + dedicated event studies |
| H14 | Phase 3d |
| Anything the catalog missed | Phase 5b (the tree's job is to surface interactions we didn't hypothesize) |

Every hypothesis verdict (confirmed / rejected / inconclusive, with effect size) gets a row in `FINDINGS.md`. **Rejections are findings.**

---

## 3. Phase 1 — Data Engineering

### 3.1 Bar timestamp convention (critical — read first)

Your sample data shows `2020-12-01 09:15:00+05:30` for the first 5m bar of the day. NSE convention is that this is the **bar START** time. So the bar at timestamp `09:15:00` covers the interval `[09:15:00, 09:20:00)` and closes at `09:20:00`.

This matters because **every no-look-ahead join in Phase 2 requires a bar-CLOSE timestamp, not a bar-START timestamp**. If you join higher-TF features using start timestamps directly, you will leak future information.

**Convention adopted throughout this study:**

- Raw input column `timestamp` = bar start (as given)
- Derived column `bar_close` = bar start + bar duration (5min, 15min, 1h, 1d-end-of-session)
- All no-look-ahead joins use `bar_close` exclusively
- All forward returns are computed against future `close` values, indexed by bar start (this is fine because forward returns by definition use future data; only feature joins need to use close timestamps)

Verification: at 5m bar start `10:20:00` (closes `10:25:00`), the most recent CLOSED 15m bar is `10:00–10:15` (closes `10:15:00`). The 15m bar `10:15–10:30` is still in progress and must NOT appear in the join.

### 3.2 Resampling rules

NIFTY trades 09:15 to 15:30 IST on weekdays. Resampling must respect session boundaries.

| Target TF | Rule |
|---|---|
| 5m | As-is (raw input) |
| 15m | Group every 3 consecutive 5m bars within the same session: 09:15–09:30, 09:30–09:45, … |
| 1h | **Session-anchored**, not clock-anchored: 09:15–10:15, 10:15–11:15, 11:15–12:15, 12:15–13:15, 13:15–14:15, 14:15–15:15. The final 09:15-to-15:30 bar (15 min remainder) is dropped OR kept as a stub — pick one and document. **Recommendation: drop the stub** (cleaner, but you lose the final 15 min of each day for 1h analysis). |
| 1D | Group all bars from 09:15 to 15:30 of same trading day |

OHLC aggregation per group:
- open = first
- high = max
- low = min
- close = last
- volume: **dropped at load time** — confirmed all-zero in the raw dataset (NIFTY 50 is an index, not a traded instrument; see plan.md L1). Do not aggregate, store, or compute anything from it.

**Critical:** Use pandas `groupby` on a session-anchored bin label, **not** `df.resample('1H')` — the latter will produce bars like 09:00–10:00 which span pre-market and partial sessions.

### 3.3 Data hygiene checks

Before any feature computation:

- Drop duplicate timestamps (keep first)
- Detect missing 5m bars within a session (NIFTY occasionally has gaps); log count, do **not** forward-fill (would corrupt EMAs)
- Volume: confirm the column is all-zero (one-line assert at load), then **drop it**. If any non-zero values appear, that's a data-vendor inconsistency worth logging — but still don't use it as a feature (index volume from vendors is synthetic/unreliable)
- Validate that high ≥ max(open, close) and low ≤ min(open, close)
- Print summary: bars per year, trading days per year, expected vs. actual

### 3.4 Output of Phase 1

Four parquet files (parquet preferred over CSV for type safety + speed):
- `nifty_5m.parquet`
- `nifty_15m.parquet`
- `nifty_1h.parquet`
- `nifty_1d.parquet`

Each with columns: `timestamp, bar_close, open, high, low, close, session_date, bar_in_session`. (No volume column — see §3.3.)

---

## 4. Phase 2 — Feature Engineering

### 4.1 Per-timeframe feature matrix

For each of the 4 TFs, compute all features from §2.1 and §2.2. Save as:
- `features_5m.parquet` (~75 columns including all derived states)
- `features_15m.parquet`
- `features_1h.parquet`
- `features_1d.parquet`

Each row carries the indicator state **as of bar close**. This is your primary unit of analysis.

### 4.4 EMA warmup — drop early bars per TF

EMAs need time to converge from their seed value. We drop the first N bars of each TF before any analysis:

| TF | Slowest EMA | Bars to drop | Wall-clock cost |
|---|---|---|---|
| 5m | 90 (B slow) | 270 | ~3.5 trading days |
| 15m | 90 | 270 | ~10 trading days |
| 1h | 90 | 270 | ~45 trading days (~2 months) |
| 1d | 90 | 270 | **~270 trading days (~13 months)** |

The 1d cost is large: with ~1250 daily bars over 5 years, we lose the first year of 1d analysis. Two implications:

- The walk-forward train split (Years 1–3) effectively becomes Years 2–3 for 1d-conditional analyses
- Phase 4 inter-TF analyses involving 1d features have ~4 years of usable joint data, not 5

This is unavoidable given the 90-period EMA. The alternative is to use the full series with the understanding that early 1d values are unreliable — we choose the cleaner approach (drop them).

### 4.5 Master joined dataset (for Phase 4 inter-TF analysis)

Build a single 5m-indexed DataFrame where each 5m row carries:
- All 5m features (current bar)
- All 15m features (from the most recently CLOSED 15m bar — strict no-look-ahead)
- All 1h features (from the most recently CLOSED 1h bar)
- All 1d features (from the most recently CLOSED 1d bar — i.e. yesterday's daily close until today's market closes)

**Naming convention:** `A_hist_5m`, `A_hist_15m`, `A_hist_1h`, `A_hist_1d`, etc.

Save as `master_5m.parquet`. This is the single dataset that feeds Phases 3, 4, and 5.

### 4.6 No-look-ahead enforcement

This is the single most important correctness check in the entire study. The rule:

> **At a 5m bar with `bar_close = T`, the value of any feature from a higher TF must come from the most recent higher-TF bar whose `bar_close ≤ T`.**

Concretely: at 5m `bar_close = 10:25:00`, the "current" 1h feature is from the bar with `bar_close = 10:15:00`. The 1h bar with `bar_close = 11:15:00` does not exist yet at this point.

We enforce this with `pd.merge_asof(left_on='bar_close', right_on='bar_close', direction='backward', allow_exact_matches=True)` joining 5m → higher-TF. `allow_exact_matches=True` is correct because a higher-TF bar that closes at exactly the same instant as the 5m bar IS observable (e.g. 5m bar `09:25:00–09:30:00` and 15m bar `09:15:00–09:30:00` both close at `09:30:00`; at that instant we know both).

A unit test: for 1000 random 5m timestamps, manually verify the 1h feature pulled is from the previously-closed 1h bar. If even one violates, the entire downstream analysis is invalid.

---

## 5. Phase 3 — Intra-Timeframe Pattern Analysis

For each TF independently, run the following. This is the **statistical** stage.

### 5.1 Target variables (built once, used everywhere)

Per your answer, we build all three:

| Target family | Definition | Use |
|---|---|---|
| `ret_fwd_N` | (close[t+N] / close[t]) - 1, for N ∈ {1, 3, 6, 12, 24} bars | Continuous regression / correlation |
| `barrier_hit` | Triple-barrier label: within a fixed window of N bars, did price hit +U% (upper) first, -L% (lower) first, or neither (time barrier)? Output: {+1, -1, 0} | Directional classification |
| `regime` | Rolling 20-bar regime label: `trend_up` / `trend_down` / `chop` based on linear-fit slope and R² | Conditional analysis |

**Triple-barrier parameters per TF** (upper/lower calibrated to ~1.5× ATR(14) on that TF; time window matches the longest forward-return horizon):

| TF | Upper | Lower | Time window | Notes |
|---|---|---|---|---|
| 5m | +0.20% | -0.20% | 24 bars (=2 hours) | Most labels will hit a barrier; "neither" is rare |
| 15m | +0.35% | -0.35% | 24 bars (=6 hours) | |
| 1h | +0.60% | -0.60% | 24 bars (=4 days) | |
| 1d | +1.20% | -1.20% | 10 bars (=2 weeks) | |

ATR-based numbers are starting points; refine after seeing the empirical ATR distribution in Phase 1 quality report. If a barrier is so tight that >95% of bars hit it within 2 bars, the label degenerates to "what direction in next 2 bars" — widen.

### 5.2 Sub-analysis 3a — Indicator vs. Price

For each indicator (A, B), each derived feature, each forward-return horizon:

1. **Pearson + Spearman correlation** of feature vs. `ret_fwd_N`. Spearman matters more (captures monotonic non-linear relationships).
2. **Conditional return tables**: Bin the feature into deciles, compute mean/median/std/hit-rate of `ret_fwd_N` per decile. Look for monotonic patterns.
3. **State-conditional analysis**: Group by `A_state` (the 4-quadrant label), compute forward return distribution per state. This directly tests your hypothesis: "if ROC of gap is increasing AND positive, does NIFTY move higher?"
4. **Statistical significance**: For each conditional cell, compute t-statistic vs. unconditional mean and report p-value with Bonferroni correction (we'll be running ~hundreds of tests).

**Output per TF:** `phase3a_indicator_vs_price_{tf}.csv` + a summary markdown report.

### 5.3 Sub-analysis 3b — Lead-Lag (within indicator family)

**⚠ Logic correction (important).** A naive reading is "compute cross-correlation between A and B; whichever leads is the early signal." That reasoning is flawed for these two indicators, because **both are deterministic smoothed transforms of the same close series**. Indicator A uses EMA(12/26) — less smoothing lag. Indicator B uses EMA(40/90) — more smoothing lag. Cross-correlation will therefore *mechanically* show A leading B by some bars. This is a property of the filter math, not new information. Concluding "A is an early signal for B" is true but useless — and concluding "B leads A" would be a red flag for a computation bug.

What we actually test in 3b:

1. **Mechanical lag measurement (descriptive, not predictive):** Cross-correlation between `A_hist` and `B_roc` at lags ∈ [-20, +20] bars per TF. Expected result: A leads B; the peak lag quantifies how much earlier the fast indicator turns. This is useful *context* (e.g. "on 5m, A turns ~6 bars before B confirms") but is not an edge by itself.
2. **The informational question — incremental predictive power:** Does B add anything about *future price* beyond what A already says (and vice versa)? Test with:
   - Bivariate conditional tables: forward return given (A_state, B_state) jointly vs. given A_state alone — does adding B's column shift the conditional means significantly? (This overlaps with 3c and is the real payoff.)
   - A simple nested regression: `ret_fwd_N ~ A_hist_norm` vs. `ret_fwd_N ~ A_hist_norm + B_roc` — report ΔR² and the t-stat (HAC) on B's coefficient. If B's coefficient is insignificant once A is included, B is redundant at this TF.
3. **Turn-confirmation timing:** When A flips bullish (hist zero-cross up), measure the distribution of bars until B confirms (roc crosses positive). Then split A-flips into "B confirmed within K bars" vs. "B never confirmed" and compare forward returns from the flip. **This is the legitimate use of the A→B lag structure**: the fast indicator triggers, the slow indicator's confirmation (or refusal) classifies the trigger.

### 5.4 Sub-analysis 3c — Indicator A vs. Indicator B (agreement)

Build a 4×4 agreement matrix: rows = `A_state`, columns = `B_state`. Each cell holds:
- Count of bars in this joint state
- Mean forward return at horizon N
- Hit rate (% of bars with positive forward return)

Diagonals (both bull_accel, both bear_accel) test the **confluence hypothesis**. Off-diagonals test **disagreement** (which often precedes regime change).

### 5.5 Sub-analysis 3d — Divergence analysis

Classical divergence: price makes higher high but indicator makes lower high (bearish divergence), or inverse (bullish divergence).

Algorithmic detection:
- Find local price highs/lows using `scipy.signal.find_peaks` with minimum spacing of N bars
- For consecutive same-type extrema, check if price direction agrees with indicator direction
- Flag divergence events

**Critical look-ahead issue:** `find_peaks` requires bilateral neighbors. A peak at bar T can only be **confirmed** at bar T + W (where W = half the spacing window). Forward returns must be measured from the **confirmation bar**, not the peak bar itself. Failing to do so leaks future information into the divergence signal.

Concretely: if you find a peak at bar 100 using ±5 neighbors, that peak is only knowable at bar 105 at the earliest. Forward returns must be `close[105+N] / close[105] - 1`, not `close[100+N] / close[100] - 1`. The former is honest; the latter is look-ahead masquerading as a strategy.

Quantifies what most traders only eyeball.

---

## 6. Phase 4 — Inter-Timeframe Pattern Analysis

All on the `master_5m.parquet` dataset, with the target being **5m forward returns** (your stated focus is "momentum in lower time frame price").

### 6.1 Sub-analysis 4a — Higher TF as regime filter

For each lower-TF setup that showed an edge in Phase 3, **stratify** by higher TF state and re-measure the edge.

Example matrix:

| 5m setup | 1h `B_state = bull_accel` | 1h `bull_decel` | 1h `bear_decel` | 1h `bear_accel` |
|---|---|---|---|---|
| `A_state_5m = bull_accel` | mean fwd ret + n + p-val | … | … | … |
| `A_state_5m = bull_decel` | … | … | … | … |
| etc. | | | | |

Hypothesis: edges from Phase 3 should **strengthen** in aligned higher-TF regimes and **weaken or invert** in counter-trend higher-TF regimes.

### 6.2 Sub-analysis 4b — Cross-TF lead-lag

When does a 15m `A_hist` zero-cross lead a 1h `A_hist` zero-cross?

- Identify all 15m hist zero-crosses
- For each, measure: did the 1h hist cross zero in the next K bars? What's the median lag?
- Build a transition probability matrix: P(1h flips bullish in next 4h | 15m just flipped bullish)

### 6.3 Sub-analysis 4c — Confluence scoring

Define a **bullish confluence score** ∈ [0, 8]:

```
score = (A_5m bull) + (B_5m bull) + (A_15m bull) + (B_15m bull)
      + (A_1h bull) + (B_1h bull) + (A_1d bull) + (B_1d bull)
```

where `bull` for A means `hist > 0`, for B means `roc > 0`.

Then a stricter **bullish-accelerating score**:

```
score_accel = sum of (state == bull_accel) across all 8 (indicator, TF) pairs
```

Measure forward 5m return at horizons {3, 6, 12} bars, bucketed by score:

| Score bucket | n | mean fwd ret 12 bars | hit rate | std |
|---|---|---|---|---|
| 0 | … | … | … | … |
| 1 | … | … | … | … |
| ... | | | | |
| 8 | … | … | … | … |

If the curve is monotonic, confluence is real. If it's flat or U-shaped, confluence as a simple sum is the wrong abstraction.

### 6.4 Sub-analysis 4d — Asymmetry test

Is the bullish-confluence edge symmetric with the bearish-confluence edge? Markets often show asymmetry (e.g. fast crashes, slow grinds up). Reporting both sides separately matters.

---

## 7. Phase 5 — Confluence Pattern Mining

This is the **ML** stage. Goal is **discovery**, not deployment.

### 7.1 Sub-analysis 5a — Rule enumeration

Enumerate all boolean combinations of the 8 (indicator, TF, sign) flags — 2^8 = 256 combinations. For each, compute:
- Frequency in dataset
- Mean forward 5m return at horizon 12
- Hit rate
- Sharpe-like ratio (mean / std)

Rank by `frequency × |mean_return| × hit_rate_excess_over_50%`. Inspect top 20 manually — these are your interpretable rules.

### 7.2 Sub-analysis 5b — Tree-based feature importance

Train a **shallow LightGBM** (max_depth=4, n_estimators=200) on:
- Features: all numeric features from all 4 TFs (~80 columns)
- Target: sign of `ret_fwd_12_5m`
- Evaluation: walk-forward, never random split (time-series leakage)

Inspect:
- Permutation importance (more reliable than gain importance)
- Top 5 splits at root level — these are the "first questions" the tree asks
- SHAP values for top features — reveals interaction effects

This is **purely interpretive**. You're not deploying this model; you're using it to find feature interactions that you didn't specify in 5a.

### 7.3 Sub-analysis 5c — Inversion / failure analysis

For the top 5 rules from 5a:
- Find all bars where the rule fired
- Split into "worked" (forward return matched expectation) and "failed"
- For failed cases, find the feature(s) most different from worked cases (mean-difference test, ranked by effect size)
- These differentiating features become **filter candidates** that could be added to the rule

This is where the real insight comes from. A rule that works 60% of the time is useless. A rule that works 60% baseline but works 80% when an additional filter is applied is **knowledge**.

---

## 8. Phase 6 — Walk-Forward Validation

Even though this is research, not trading, validation discipline matters or your "patterns" will be overfit noise.

### 8.1 Splits

5 years of 5m data ≈ 1250 trading days × ~75 bars/day ≈ ~94,000 bars.

- **Train (in-sample for pattern discovery):** Years 1–3 (2020-12 → 2023-12)
- **Validation (rule selection):** Year 4 (2024)
- **Test (final reporting only, look once):** Year 5 (2025+)

All Phase 3, 4, 5 analyses run on **train**. Top rules and top features are then re-measured on **validation** to filter out lucky patterns. Final summary of survivors is reported on **test** — and **only once**, with no iteration based on test results.

### 8.2 Multiple-testing correction

We will run hundreds of conditional return tests. Without correction, ~5% will appear "significant" by chance alone. Use **Bonferroni** (conservative) for the final survivor list. For exploration, FDR (Benjamini-Hochberg) is fine.

### 8.3 Overlapping forward returns — correct variance estimation

This is a subtle but serious issue. Computing `ret_fwd_12` at every 5m bar produces a series where consecutive observations share 11 out of 12 return components. They are **not independent**. A naive t-test treats them as independent and produces p-values that can be 3-5× too optimistic.

Three options, in order of statistical rigor:

1. **Block-bootstrap** (best): resample blocks of length ≥ N (the forward horizon) when computing confidence intervals. Implementation: `scipy` doesn't provide this directly; use `arch.bootstrap.StationaryBootstrap` or hand-roll.
2. **Newey-West HAC standard errors**: adjust variance for autocorrelation up to lag N-1. Available in `statsmodels`.
3. **Non-overlapping subsampling**: take every N-th bar so observations don't overlap. Loses 11/12 of the data but is statistically clean.

**Recommendation:** Use Newey-West for all conditional return tables (cheap, well-understood). Cross-check the top 5 survivors with block-bootstrap (more expensive but more robust). Never report naive t-stats from overlapping returns — they are not reliable.

### 8.4 Sanity benchmarks

Every reported edge must be compared to:
- Unconditional baseline (mean forward return regardless of indicator state)
- Random feature baseline (shuffle the feature column, re-run; should produce zero edge)
- Time-of-day baseline (NIFTY has known intraday patterns; an "edge" that's just "first hour drift" is not interesting)

---

## 9. Deliverables

Since you'll code it yourself, the design specifies **what to produce**, not the code itself.

### 9.1 Per-phase outputs

| Phase | Output |
|---|---|
| 1 | 4× parquet files (one per TF) + a `data_quality_report.md` |
| 2 | 4× feature parquet files + 1× `master_5m.parquet` + a no-look-ahead unit test passing |
| 3 | Per TF: a markdown report `phase3_intra_{tf}.md` with tables + plots; plus `phase3_summary.md` comparing TFs |
| 4 | `phase4_inter_tf.md` with the stratified matrices and confluence-score curves |
| 5 | `phase5_rules_top20.md`, `phase5_ml_importance.md`, `phase5_failure_analysis.md` |
| 6 | `phase6_validation.md` listing rules that survived OOS |

### 9.2 Final synthesis

A `FINDINGS.md` document — your equivalent of a research conclusion — that answers in plain language:

1. Within each TF, which indicator state most reliably precedes directional moves?
2. Does B lead A or A lead B at each TF?
3. When indicators agree on a single TF, is the edge real?
4. When indicators agree across TFs, is the edge stronger? By how much?
5. What's the best rule discovered? What's its failure mode? What filter improves it?

---

## 10. Open Questions / Risks

| # | Question | Why it matters | Default if you don't decide |
|---|---|---|---|
| 1 | 1h bar — drop the 15-min stub or include? | Affects ~6% of 1h bars | Drop |
| 2 | Use raw `gap` or `gap_norm = gap/close`? Index moved from ~13k to ~24k over 5 years | Raw `gap` is non-stationary; `gap_norm` is more comparable across the dataset | Use both, prefer `gap_norm` for cross-time comparisons |
| 3 | Forward return horizons — fixed bars or fixed clock-time? | 12 bars on 5m = 1h; on 1d = 12 days. Apples to oranges | Use fixed bars per TF, document explicitly |
| 4 | Include overnight gaps in 1d returns? | Overnight gaps are often the largest moves but unrelated to intraday indicators | Exclude (compute 1d return as close-of-day vs. open-of-day, not close-vs-close) |
| 5 | Volatility regime stratification | Indicator behavior likely differs in high-vol vs. low-vol periods (2020 COVID, 2024 election) | Add VIX regime as a fifth "TF" of context if VIX data is available |
| 6 | Friday/expiry-week effects | NIFTY weekly options expire Thursdays (recent change to Tuesdays). Last-day-of-expiry behavior is anomalous | Flag expiry days, run all analyses with and without them |

---

## 11. Execution Order (suggested)

The phases are dependencies, not parallel tracks. Suggested sequence:

```
Phase 1 (Data)              → 1-2 days
Phase 2 (Features)          → 1 day
Phase 3 (Intra-TF stats)    → 3-4 days (most analytical work)
Phase 4 (Inter-TF stats)    → 2-3 days
Phase 5 (Rules + ML)        → 3-4 days
Phase 6 (Validation)        → 1 day
Synthesis (FINDINGS.md)     → 1 day
```

Total: ~2 weeks of focused work. Don't skip Phase 6 — every "discovery" in Phases 3-5 is provisional until OOS-validated.

---

## 12. What This Study Will NOT Do

Explicit non-goals (to prevent scope creep):

- ❌ Will not produce a trading signal or backtest with PnL
- ❌ Will not model transaction costs or slippage
- ❌ Will not optimize indicator parameters (12/26/9 and 40/90/7 are locked per your spec)
- ❌ Will not include other indicators (RSI, ADX, etc.) — keep the study focused
- ❌ Will not include options Greeks / OI data (separate domain)
- ❌ Will not use deep learning — LightGBM is the ceiling, for interpretability

If a finding from this study suggests a tradeable pattern, that becomes a **separate** project with its own design doc, including walk-forward backtest, transaction cost model, and risk sizing.

---

## End of Design Document
