# FINDINGS — Multi-Timeframe Momentum Confluence Study (NIFTY 50, 2020-2025)

**Status:** Final synthesis (Phase 7). This document answers DESIGN_DOC §9.2's five synthesis
questions (T7.1), gives an explicit verdict for every hypothesis in the H1-H14 catalog (T7.1b),
documents the negative findings (T7.2), and lists concrete follow-ups (T7.4). The PLAN.md
decision log (D1-D10) is updated separately (T7.3).

## 0. Executive summary

Five years of 5m/15m/1h/1d NIFTY 50 data (Nov 2020 - Nov 2025, ~92.4k 5m bars) were used to test
whether a standard MACD (`A`, 12/26/9) and an EMA-gap-ROC indicator (`B`, 40/90/7) carry
forward-return information, individually, against each other, and in cross-timeframe confluence.

The headline result is a **mostly negative one, and that is itself the finding**: of the 256
enumerated 8-flag confluence rules, 15 survived in-sample dedup/ranking, 10 were carried into
walk-forward validation (D9), and **only 2/10 survived Phase 6's three OOS robustness checks**
(rules 140 and 31). Both surviving edges are economically tiny (4-6 bps over a ~1h horizon),
neither is significant after Bonferroni correction, and neither was traded with costs. Several
"classical" MTF hypotheses from the catalog (H3, H5, H7, H9, H14) were directionally **rejected**
or contradicted the textbook framing. The hypotheses that held up best were the simplest ones:
indicator A's `bull_accel` state continues to drift in its direction at 15m/1h/1d (H1), and a
higher-timeframe regime filter measurably strengthens a lower-timeframe trigger's edge (H11),
provided "agreement" is defined strictly (H12).

---

## 1. Synthesis questions (T7.1)

### Q1 — Within each TF, which indicator state most reliably precedes directional moves?

**Indicator A's histogram-momentum state (`A_state`) is consistently more informative than
indicator B's gap-ROC state (`B_state`) at every timeframe** — across all four phase3 reports,
not a single `B_state` cell survives the FDR-adjusted significance threshold of `p_val_adj < 0.05`,
while `A_state` does at every TF. *Which* `A_state` matters, and at *which* horizon, changes with
the timeframe in an intuitive way:

- **5m**: `bear_accel` is the standout state — significant at 4 of 5 horizons (`ret_fwd_1/3/6/24`,
  `p_val_adj` between 0.012 and 0.048), with `mean_ret_fwd_12 = +0.000138` (t=2.53). `bull_accel`
  does *not* survive FDR at any horizon (`ret_fwd_12` t=2.02, `p_val_adj=0.108`). So at 5m the
  reliable signal is a small mean-reverting bounce after a fresh bearish-accelerating histogram
  reading, not the "textbook" bullish-continuation state.
- **15m**: both `bull_accel` and `bull_decel` are significant at the longer horizons —
  `bull_accel` `ret_fwd_12=+0.000481` (t=2.89, p_adj=0.019) and `ret_fwd_24=+0.000922` (t=3.54,
  p_adj=0.004); `bull_decel` `ret_fwd_12=+0.000493` (t=3.27, p_adj=0.0073) and
  `ret_fwd_24=+0.000875` (t=3.53, p_adj=0.0041). At 15m, `A_state` clearly dominates `B_state`
  (best `B_state` cell, `bull_decel` `ret_fwd_24`, only reaches p_adj=0.066).
- **1h**: `bull_accel` is the single cleanest signal in the whole study at the intraday TFs —
  significant at 4 of 5 horizons (`ret_fwd_1,3,6,24`, p_adj from 0.0027 to 0.020), peaking at
  `ret_fwd_6=+0.001465` (t=3.81). Interestingly `ret_fwd_12` itself (t=1.95, p_adj=0.140) is the
  *one* horizon that misses significance — the edge is concentrated in the 1-6 bar window and
  partially fades by bar 12.
- **1d**: `bull_accel` drives short-horizon continuation (`ret_fwd_1=+0.001484`, t=3.13, p_adj=0.017;
  `ret_fwd_3=+0.003747`, t=3.67, p_adj=0.0049), while `bear_accel` drives a much larger
  *medium/long-horizon mean-reversion* signal — `ret_fwd_6=+0.005063` (t=2.63, p_adj=0.043) and
  `ret_fwd_24=+0.011610` (t=2.92, p_adj=0.024), the single largest mean forward return anywhere in
  this study (+1.16% over the next 24 trading days after a 1d `bear_accel` reading).

**Net**: `A_state` is the workhorse indicator at every TF; `B_state` never clears FDR anywhere.
The "shape" of the edge moves from a small bearish-bounce signal at 5m, to bullish continuation at
15m/1h, to a bimodal short-horizon-continuation / long-horizon-reversal pattern at 1d.

### Q2 — Does B lead A or A lead B at each TF?

**Neither, in any economically meaningful sense — B carries essentially no incremental
information about future price beyond A, at any TF.** The corrected lead-lag framing from L5/H8
(mechanical lag is descriptive only; the real test is incremental R²) gives:

| TF | peak xcorr lag (bars) | peak xcorr | ΔR² (A+B vs A) | B coef. t-stat (p) |
|---|---:|---:|---:|---:|
| 5m | 0 | 0.0135 | 0.000027 | 2.04 (p=0.042) |
| 15m | 6 | 0.0087 | 0.000052 | 1.23 (p=0.220) |
| 1h | 7 | 0.0334 | 0.000114 | 1.23 (p=0.219) |
| 1d | -19 | -0.0759 | 0.001786 | 1.10 (p=0.272) |

ΔR² ranges from 0.00003 (5m) to 0.0018 (1d) — even the largest value means B explains less than
two-tenths of one percent of additional variance in `ret_fwd_12`. The 5m case is "significant"
(p=0.042) only because of the enormous sample size (n=23,370); a 0.0135 correlation and ΔR²=0.00003
have no practical meaning. The 1d cross-correlation peak at lag -19 (the largest in magnitude) is
based on n=64 events and almost certainly noise. The mechanical-lag pattern predicted by L5 (A's
faster 12/26 EMAs should lead B's slower 40/90 EMAs) shows up faintly at 15m/1h (peak lag of 6-7
bars, small positive correlation) but not at 5m or 1d. **Bottom line: A is sufficient; B adds
nothing measurable to a price-direction forecast at any TF** (see H8 verdict).

### Q3 — When indicators agree on a single TF, is the edge real?

**The joint A×B state distribution is never independent of forward returns (chi-square
p < 1e-6 at all four TFs) — so "agreement vs. disagreement" is real structure, not noise — but
"agreement is better" is not a uniform rule.** At 5m and 15m, the diagonal (A and B in the same
state) averages a higher `ret_fwd_12` than the off-diagonal (5m: 0.000124 vs 0.000082; 15m:
0.000315 vs 0.000208), and the single best cell at 5m is full agreement —
`(A=bull_accel, B=bull_accel) = 0.000213`, nearly double the unconditional `A=bull_accel` edge of
0.000112.

But at 1h and 1d the pattern reverses: the diagonal averages *lower* than the off-diagonal (1h:
0.000926 vs 0.001384; 1d: 0.005595 vs 0.007570), and in both cases the single best cell in the
`A=bull_accel` row is a *disagreement* cell — 15m `(bull_accel, bull_decel) = 0.000685` vs. its own
diagonal `(bull_accel, bull_accel) = 0.000330`; 1h `(bull_accel, bear_decel) = 0.001562` vs.
diagonal `0.001273`. These look like "A still accelerating while B is fading/turning" — a
divergence-flavored setup, not the "both indicators confirm" setup the catalog (H9) predicted to
be best.

So: yes, agreement carries real, statistically distinguishable information (Q3's first half is
true), but the direction in which it matters flips with timeframe, and at higher TFs the *best*
single-TF combinations are disagreement cells with small effective sample sizes (n_htf in the
hundreds for 1h, dozens for 1d) — these specific cells should be read as suggestive, not as
established edges.

### Q4 — When indicators agree across TFs, is the edge stronger? By how much?

**Yes — cross-TF alignment with a higher-timeframe `A_state` roughly doubles to triples the size
of a 5m setup's edge, and typically turns a marginal t-stat into a clearly significant one
(H11, confirmed):**

| 5m setup | unconditional ret_fwd_12 (t) | aligned-HTF stratum | aligned ret_fwd_12 (t) | multiple |
|---|---|---|---|---|
| `A_state==bull_accel` | 0.000112 (t=2.02) | 1h `A_state==bull_accel` | 0.000349 (t=2.52, hit=0.587) | 3.1x |
| `A_state==bull_accel` | 0.000112 (t=2.02) | 1d `A_state==bull_accel` | 0.000280 (t=3.00, p=0.0027) | 2.5x |
| `A_state==bear_accel` | 0.000138 (t=2.53) | 1d `A_state==bull_accel` | 0.000287 (t=3.59, p=0.0003) | 2.1x |
| `B_state==bull_accel` | 0.000143 (t=2.03) | 1d `A_state==bull_accel` | 0.000353 (t=2.78, p=0.0054) | 2.5x |

The strongest multiplier (3.1x) and the strongest absolute significance (p=0.0003) both involve a
1d/1h `A_state==bull_accel` regime filter — i.e. "trade the 5m setup when the daily/hourly MACD
histogram is itself bullish-and-accelerating" is the single most effective stratifier found in
this study.

**However, this only holds for the *strict* definition of confluence (H12).** The loose
`score_bull` (sums "any bullish sign" across the 8 A/B x 5m/15m/1h/1d pairs, in [0,8]) is flat to
non-monotonic in `ret_fwd_12` (Spearman corr = 0.133); score=0 (`mean_ret=0.000541`,
hit_rate=0.638) actually *beats* score=8 (`mean_ret=0.000164`, hit_rate=0.587). The strict
`score_bull_accel` (requires the full `bull_accel` *state*, not just sign>0, at each of the 8
pairs) is strongly monotonic (Spearman corr = 0.717): `score=8` gives `mean_ret_fwd_12=0.000634`
and `hit_rate=0.786` (n=42, t=2.00) vs. `score=0` at `mean_ret=-0.000023`, `hit_rate=0.505`
(n=17,149). So **"agreement across TFs strengthens the edge" is true, by roughly 2-3x for
single-TF-filter setups and up to ~6x for the full 8-way `bull_accel` confluence — but only when
"agreement" means the stricter `bull_accel` momentum state, not merely "same sign."** This
distinction (state vs. sign) is the most actionable finding from Phase 4.

### Q5 — What's the best rule discovered? What's its failure mode? What filter improves it?

Of the 10 confluence rules carried into Phase 6 (D9), the two highest-ranked by in-sample
composite score and the only two that cleared the x256 Bonferroni bar in Phase 5a — **rule 151**
(`A_5m, B_5m, A_15m, A_1h, B_1d`, p_val_bonf=0.184) and **rule 121** (`A_5m, B_15m, A_1h, B_1h,
A_1d`, p_val_bonf=0.057) — are exactly the two that **failed out-of-sample**: rule 151
sign-collapsed on the 2025 test split (`mean_ret_test = -0.000057` vs. expected positive), and
rule 121 sign-collapsed on the 2024 validation split (`mean_ret_val = -0.000967` vs. expected
positive, n=84). This is the headline failure mode for Q5: **in-sample statistical significance
(even after correction) was not predictive of OOS survival** — 8 of the 10 carried rules
sign-collapsed on at least one OOS split.

The two rules that *did* survive all three Phase 6 checks (no sign collapse on val or test, not
single-year-driven, edge not explained by time-of-day) are:

- **Rule 140** (`A_15m, B_15m, B_1d` bullish): in-sample `mean_ret=0.002254`, `hit_rate=0.669`,
  n=133. Validation `mean_ret=0.000405` (n=61), test `mean_ret=0.000443` (n=65) — both same sign
  as in-sample, and 4/4 years with data (2022-2025) consistent. `excess_over_tod=0.002101` of
  `0.002254` (93% of the edge survives time-of-day matching). This is the largest in-sample
  edge of the two survivors, but the smallest sample (n=133-65 per split) and `p_val_bonf=1.000`
  even at the 10-rule level.
- **Rule 31** (`A_5m, B_5m, A_15m, B_15m, A_1h` bullish): in-sample `mean_ret=0.000584`,
  `hit_rate=0.617`, n=687. Validation `mean_ret=0.000343` (n=198), test `mean_ret=0.000152`
  (n=101) — smaller but still positive and same-signed in both splits, and the **only rule of the
  10 with 5/5 data-years consistent** (2021-2025 all positive). The random-feature baseline
  (T6.5, shuffling the `A_5m`/`A_hist` column) is the most informative result of the whole table:
  rule 31's edge **flips sign and collapses** (`edge_retained_pct=-6.9%`) — the strongest evidence
  in the study that a single constituent flag (`A_5m` bullish) is genuinely load-bearing for an
  edge, rather than the rule's significance being an artifact of the other 7 flags or of overall
  sample composition.

**Failure mode, in one sentence**: edges found by exhaustively searching 256 boolean confluence
combinations on 3 years of data are dominated by combinations that happen to fit those 3 years
(151, 121, and 6 others), and the small minority that generalize (140, 31) have *decaying*,
economically tiny effect sizes (4-6 bps per ~1h) once measured OOS.

**What filter improves it**: Phase 5c's failure/inversion analysis (run only on the top-5
in-sample rules: 151, 121, 26, 0, 140) found, for **rule 140**, that adding `B_zroc_1d > 0.1698`
(the daily gap-ROC z-score must be moderately positive — i.e. the daily timeframe is itself in a
positive-momentum-expansion regime) improves in-sample `mean_ret` from 0.002254 to **0.006074**
and `hit_rate` from 0.669 to **0.878**, at the cost of shrinking n from 133 to 41. This filter is
the single largest improvement found across all 5 rules examined in 5c, and it is applied to
exactly the one rule (140) that also independently survived Phase 6's unfiltered OOS checks — but
the filtered version was **not itself re-validated OOS** (D9 explicitly scoped these as
"optional secondary checks"). It is the natural next step (see Future Work §4).

---

## 2. Hypothesis verdict table (T7.1b)

| # | Hypothesis (short) | Verdict | Effect size | Report section |
|---|---|---|---|---|
| H1 | A `bull_accel` → forward continuation | **Confirmed** (15m, 1h, 1d short horizons; not 5m) | 15m `ret_fwd_12=+0.000481` (t=2.89, p_adj=0.019); 1h `ret_fwd_6=+0.001465` (t=3.81, p_adj=0.0027) | `phase3_intra_{15m,1h}.md` §3a |
| H2 | Hist peak → decaying ("exhaustion") term structure | **Rejected** | `short_vs_long` (ret_fwd_3/ret_fwd_12) = 0.10 (5m), 0.31 (15m), 0.55 (1h) — all <1, returns *build* through bar 12 instead of decaying | `phase3_intra_*.md` §3a "H2" |
| H3 | Aligned zero-cross > tired zero-cross | **Rejected** (reversed at all 4 TFs) | 15m: tired `ret_fwd_12=0.001038` vs aligned `0.000416` (2.5x reversed); 1h tired `0.003188` vs aligned `0.000683` | `phase3_intra_*.md` §3a "H3" |
| H4 | First-pullback (with-trend) > counter-trend re-cross | **Partially confirmed** (TF-dependent) | 5m with-trend `ret_fwd_12=+0.000179` vs counter `-0.000030` (confirmed); 1h reversed: with-trend `0.001955` vs counter `0.002752` | `phase3_intra_*.md` §3a "H4" |
| H5 | B `bull_accel` (gap-ROC) → drift higher | **Rejected / not robust** | Best case 1h `ret_fwd_12=+0.001759` (t=2.41, p_adj=0.064, n.s.); 15m essentially zero (t=-0.02) | `phase3_intra_*.md` §3a "B_state" |
| H6 | zROC extremes: continuation (H6a) vs exhaustion (H6b) | **H6a confirmed**, H6b mostly rejected | 1h `extreme_bull` `ret_fwd_12=0.003771` vs baseline `0.000572` (6.6x), no decay through 24 bars | `phase3_intra_*.md` §3a "H6" |
| H7 | Acceleration precedes gap zero-cross, not a false start | **Rejected (intraday)**; not rejected at 1d | False-start rate 70.8% (5m) / 64.0% (15m) / 61.7% (1h) vs **20.7% (1d, n=82)** | `phase3_intra_*.md` §3a "H7" |
| H8 | B carries incremental info beyond A | **Rejected** | ΔR² = 0.00003 (5m, p=0.04) to 0.0018 (1d, p=0.27) — all economically negligible | `phase3_intra_*.md` §3b |
| H9 | A-trigger-in-B-regime (confirm) > A-trigger-against-B-regime (bounce) | **Rejected as stated / inconclusive** | Joint independence rejected (chi2 774/745/258/147, all p<1e-6) but direction inconsistent: at 15m/1h the best `A=bull_accel` cell is a *disagreement* cell (15m `(bull_accel,bull_decel)=0.000685` vs diagonal `0.000330`) | `phase3_intra_*.md` §3c |
| H10 | A-flip → B-confirm is the early-warning / best-return subset | **Inconclusive** | Confirmation rate 65.3%-78.6% (all >50%), median lag 3-7 bars — descriptive only, no baseline or confirmed-vs-not return split available | `phase3_intra_*.md` §3b "Turn Confirmation" |
| H11 | HTF regime + LTF trigger = stronger edge (canonical MTF pattern) | **Confirmed** | 5m `bear_accel` \| 1d `A=bull_accel` = 0.000287 (t=3.59, p=0.0003) vs unconditional 0.000138 (2.1x); 5m `bull_accel` \| 1h `A=bull_accel` = 0.000349 vs 0.000112 (3.1x) | `phase4_inter_tf.md` §4A |
| H12 | Confluence score monotonic in forward return | **Partially confirmed** — depends entirely on score definition | `score_bull` (sign-only) Spearman=0.133 (flat, rejected); `score_bull_accel` (state-strict) Spearman=0.717, score=8 hit_rate=0.786 (n=42) vs score=0 hit_rate=0.505 (n=17,149) | `phase4_inter_tf.md` §4C |
| H13 | Compression → directional expansion, larger than baseline | **Partially confirmed** | Directional part confirmed at 15m (`signed_ret@12=+0.000401`) and 1h (`+0.000082`); magnitude part confirmed only at 1h (`0.002514 > 0.002269` baseline), not 15m (`0.002234 < 0.002269`) | `phase4_inter_tf.md` §4C(cont) |
| H14 | Divergence → reversal opposite the prior trend | **Rejected (5m, 15m); inconclusive (1h, 1d, small n)** | 5m bearish-div `ret_fwd_12=+0.000343` (predicted negative — wrong sign); 1h bullish-div `ret_fwd_12=+0.002229` matches predicted reversal direction but n=67 | `phase3_intra_*.md` §3d |

---

## 3. Negative findings (T7.2)

Negative results are listed with the same rigor as positive ones — per DESIGN_DOC §6, "at least
one surviving pattern OR an honest negative result" is success criteria, and this study has both.

1. **"Aligned" zero-crosses underperform "tired" ones, at every timeframe (H3).** The textbook
   intuition — that a MACD zero-cross backed by a still-rising histogram ("aligned") has better
   follow-through than one where the histogram is already falling ("tired") — is not just
   unsupported, it is *reversed* at all four TFs, sometimes by a large margin (15m:
   `tired=0.001038` vs `aligned=0.000416`; 1h: `tired=0.003188` vs `aligned=0.000683`). A
   deceleration *before* the zero-cross appears to be informative in the opposite direction from
   what the catalog hypothesized — possibly because a "tired" cross often follows a sharper prior
   move, leaving more room for continuation. Either way, "wait for the histogram to be rising at
   the cross" is not a useful filter on this data.

2. **"Early" gap-acceleration signals are mostly false starts intraday (H7).** 61-71% of the
   times `B_gap_acc` turns positive while `B_gap` is still negative — the textbook "early warning
   before the EMA cross" — the gap *never actually crosses zero* within the lookout window, at
   5m/15m/1h. Trying to front-run the gap zero-cross using acceleration alone would be wrong about
   2 times in 3 at intraday timeframes. (At 1d the false-start rate drops to 20.7%, but n=82 is
   small.)

3. **Indicator B (EMA-gap-ROC) adds essentially no incremental forecasting power over indicator A
   (MACD), at any timeframe (H8).** ΔR² from adding B to a regression of `ret_fwd_12` on A ranges
   from 0.00003 (5m) to 0.0018 (1d) — all practically zero, and only "significant" at 5m because
   of the huge sample size. B's own state (`B_state`) also never produces an FDR-significant
   conditional-return cell at any TF (Q1). Given that B was the user's "stated core hypothesis,
   verbatim" (H5) for this study, this is the most consequential negative finding: **the
   gap-ROC family, as specified (40/90 EMA gap, 7-period ROC), does not appear to carry usable
   directional information on NIFTY 50 at these timeframes**, on its own or as a complement to A.

4. **8 of the 10 confluence rules carried forward from Phase 5 sign-collapsed out-of-sample**,
   including both rules that survived Bonferroni correction in-sample (151, 121). This is the
   clearest demonstration in the study of why Gate 2 (walk-forward validation) is mandatory: an
   in-sample search over 256 rule combinations, even with Bonferroni correction and
   Hamming-deduplication, produced a "top 2" that both failed the very next thing they were
   tested against. Phase 5 alone, without Phase 6, would have reported rules 151/121 as the
   study's headline findings — and both would have been wrong.

5. **Divergence does not predict reversal at 5m/15m, and "bearish divergence" in particular
   precedes *continuation*, not reversal (H14).** At 5m, a textbook bearish divergence (price
   higher high, indicator lower high — normally read as "the rally is running out of steam") is
   followed by a *positive* `ret_fwd_12` of +0.000343 — the rally continues. The same holds at
   15m (+0.001024) and even 1d (+0.009383). Only "bullish divergence" at 1h (n=67) shows the
   hypothesized reversal direction (+0.002229). On this dataset, classic divergence-spotting is
   not a reliable reversal signal at the timeframes most discretionary traders would use it (5m,
   15m).

---

## 4. Future work (T7.4)

1. **Build a separate backtest project for rules 140 and 31** (DESIGN_DOC §7 "what comes after
   this study"), including the `B_zroc_1d > 0.1698` filter for rule 140 from Phase 5c, with
   transaction costs, slippage, and position sizing, and — critically — a **fresh OOS window**
   collected after this study's data ends (Nov 2025), since "survives Phase 6" here means
   "survived one ~15.8k-row, ~1-year test split," not "validated across multiple independent
   periods."

2. **Re-validate the rule-140 filter (`B_zroc_1d > 0.1698`) out-of-sample.** It was derived from
   the discovery set only (D9 scoped it as an "optional secondary check") and never run through
   Phase 6's val/test splits. A focused 1-rule re-run of `analysis_6` against the 2024/2025
   splits, with and without the filter, would directly test whether the 0.669→0.878 hit-rate
   improvement generalizes or is itself an in-sample artifact (cf. negative finding #4).

3. **Add INDIAVIX-based regime stratification (T8.1)**, but targeted specifically at the
   *year-to-year instability* problem rather than as a general extension. Several of the 10
   carried rules (e.g. 26, 121, 182) had their best year (often 2022) followed by a sign flip in
   2023/2024 — a volatility-regime split (COVID-recovery 2021-22 vs. calmer 2023-25) might explain
   why some edges were "real" only during a specific macro/vol regime, which would itself be a
   useful (negative-but-informative) finding.

4. **Use `bar_in_session` and `A_hist_slope_1h` more deliberately** — these were the #1 and #2
   features by permutation importance in Phase 5b, and `bar_in_session` alone explains a
   meaningful share of several rules' edges via the T6.7 time-of-day baseline (e.g. rule 0's
   `excess_over_tod` is only 80% of its raw edge). A dedicated session-phase study (e.g.
   first-hour vs. mid-session vs. last-hour conditional returns, independent of any indicator)
   would clarify how much of the "edge" anywhere in this study is actually an intraday seasonality
   effect in disguise.

5. **Re-run Phase 5/6-style mining on NIFTY futures data with real volume (T8.2)**, restricted to
   rules 140 and 31 plus their near-neighbors (Hamming distance 1) from `rules_all.csv`. Volume
   confirmation is a classical filter for momentum signals and was explicitly out of scope here
   (NIFTY 50 index has no volume); futures data would let this be tested specifically on the two
   patterns that already cleared every other bar in this study, rather than re-opening the full
   256-rule search.
