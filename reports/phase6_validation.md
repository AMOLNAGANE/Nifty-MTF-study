# Phase 6 -- Walk-Forward Validation (T6.1-T6.8)

**GATE 2.** This report re-measures the K=10 confluence patterns carried from Phase 5 (D9, `PLAN.md` SS3.4) on data they have never been selected, tuned, or ranked on. The **test split (2025+, T6.3) is touched exactly once** in this run -- no iteration on these results is permitted (`TASK.md`, Phase 6).

- **In-sample / rule-discovery set** (`bar_close.dt.year <= 2023`): n=58,080
- **Validation split** (`bar_close.dt.year == 2024`, T6.2): n=18,495 -- already used as the 5b ML hold-out, but the rule-based 5a patterns were not tuned to it directly
- **Test split** (`bar_close.dt.year >= 2025`, T6.3): n=15,837 -- never touched before this run

## T6.1 -- Carry-set patterns (unmodified from D9)

|   rule_id | active_flags                        |   n_in |   mean_ret_in |   hit_rate_in |   t_stat_in |   p_val_bonf_in |
|----------:|:------------------------------------|-------:|--------------:|--------------:|------------:|----------------:|
|       151 | A_5m, B_5m, A_15m, A_1h, B_1d       |    324 |      0.001436 |      0.682099 |    3.382506 |        0.007183 |
|       121 | A_5m, B_15m, A_1h, B_1h, A_1d       |    211 |      0.001894 |      0.691943 |    3.690514 |        0.002238 |
|        26 | B_5m, B_15m, A_1h                   |     76 |      0.003406 |      0.723684 |    1.911351 |        0.559595 |
|         0 | (none)                              |    722 |      0.000534 |      0.637119 |    1.752229 |        0.797344 |
|       140 | A_15m, B_15m, B_1d                  |    133 |      0.002254 |      0.669173 |    2.170376 |        0.299784 |
|        31 | A_5m, B_5m, A_15m, B_15m, A_1h      |    687 |      0.000584 |      0.617176 |    1.181778 |        1.000000 |
|       182 | B_5m, A_15m, A_1h, B_1h, B_1d       |    160 |      0.001244 |      0.681250 |    2.085209 |        0.370503 |
|       211 | A_5m, B_5m, A_1h, A_1d, B_1d        |     51 |      0.001944 |      0.843137 |    1.849419 |        0.643973 |
|       231 | A_5m, B_5m, A_15m, B_1h, A_1d, B_1d |    184 |      0.000930 |      0.695652 |    2.179388 |        0.293029 |
|       181 | A_5m, A_15m, A_1h, B_1h, B_1d       |    176 |      0.001469 |      0.619318 |    2.061184 |        0.392855 |

_`p_val_bonf_in` here uses x10 (the size of the carry-set), not x256 as in `reports/phase5/rules_all.csv`. Per the D9 caveats, only rules 151 and 121 had p_val_bonf < 0.2 under the original x256 correction -- the remaining 8 are exploratory._

## T6.2-T6.4 -- In-sample vs. validation vs. test

For each pattern, `evaluate_rules` (5a, unmodified) is re-run on the validation split (2024) and the test split (2025+). `expected_sign = sign(mean_ret_in)`; `collapsed_val` / `collapsed_test` are True if `sign(mean_ret)` flips relative to `expected_sign` in that split (a missing/NaN mean_ret -- e.g. zero firings -- also counts as collapsed).

|   rule_id | active_flags                        |   expected_sign |   n_in |   mean_ret_in |   hit_rate_in |   n_val |   mean_ret_val |   hit_rate_val | collapsed_val   |   n_test |   mean_ret_test |   hit_rate_test | collapsed_test   |
|----------:|:------------------------------------|----------------:|-------:|--------------:|--------------:|--------:|---------------:|---------------:|:----------------|---------:|----------------:|----------------:|:-----------------|
|       151 | A_5m, B_5m, A_15m, A_1h, B_1d       |               1 |    324 |      0.001436 |      0.682099 |      95 |       0.000907 |       0.663158 | False           |      142 |       -0.000057 |        0.450704 | True             |
|       121 | A_5m, B_15m, A_1h, B_1h, A_1d       |               1 |    211 |      0.001894 |      0.691943 |      84 |      -0.000967 |       0.321429 | True            |       39 |        0.000219 |        0.307692 | False            |
|        26 | B_5m, B_15m, A_1h                   |               1 |     76 |      0.003406 |      0.723684 |      28 |      -0.001520 |       0.500000 | True            |       19 |        0.000433 |        0.421053 | False            |
|         0 | (none)                              |               1 |    722 |      0.000534 |      0.637119 |      20 |       0.000984 |       0.700000 | False           |        1 |      nan        |      nan        | True             |
|       140 | A_15m, B_15m, B_1d                  |               1 |    133 |      0.002254 |      0.669173 |      61 |       0.000405 |       0.475410 | False           |       65 |        0.000443 |        0.600000 | False            |
|        31 | A_5m, B_5m, A_15m, B_15m, A_1h      |               1 |    687 |      0.000584 |      0.617176 |     198 |       0.000343 |       0.555556 | False           |      101 |        0.000152 |        0.534653 | False            |
|       182 | B_5m, A_15m, A_1h, B_1h, B_1d       |               1 |    160 |      0.001244 |      0.681250 |       1 |     nan        |     nan        | True            |       79 |        0.000169 |        0.518987 | False            |
|       211 | A_5m, B_5m, A_1h, A_1d, B_1d        |               1 |     51 |      0.001944 |      0.843137 |       0 |     nan        |     nan        | True            |       26 |       -0.001968 |        0.384615 | True             |
|       231 | A_5m, B_5m, A_15m, B_1h, A_1d, B_1d |               1 |    184 |      0.000930 |      0.695652 |     181 |      -0.000415 |       0.447514 | True            |        7 |       -0.000035 |        0.428571 | True             |
|       181 | A_5m, A_15m, A_1h, B_1h, B_1d       |               1 |    176 |      0.001469 |      0.619318 |       0 |     nan        |     nan        | True            |      121 |       -0.000128 |        0.586777 | True             |

- 6/10 patterns collapsed (sign flip or no data) on the 2024 validation split.
- 5/10 patterns collapsed on the 2025 test split.

## T6.5 -- Random-feature baseline

For each pattern, the `A_hist` column (the `A_5m` flag's defining indicator) is permuted across rows (seed=42) on the in-sample set, decoupling that flag from `ret_fwd_12` and from the other 7 flags, and `evaluate_rules` is re-run. Per DESIGN_DOC SS8.4, a real edge should weaken substantially once one of its constituent conditions is randomized; an edge that is *unchanged* by this shuffle would suggest `A_5m` plays no real role in the pattern.

|   rule_id | active_flags                        |   mean_ret_in |   hit_rate_in |   n_shuffled |   mean_ret_shuffled |   hit_rate_shuffled |   t_stat_shuffled |   edge_retained_pct |
|----------:|:------------------------------------|--------------:|--------------:|-------------:|--------------------:|--------------------:|------------------:|--------------------:|
|       151 | A_5m, B_5m, A_15m, A_1h, B_1d       |      0.001436 |      0.682099 |          349 |            0.000941 |            0.638968 |          2.126873 |           65.526162 |
|       121 | A_5m, B_15m, A_1h, B_1h, A_1d       |      0.001894 |      0.691943 |          412 |            0.000884 |            0.606796 |          2.775580 |           46.650172 |
|        26 | B_5m, B_15m, A_1h                   |      0.003406 |      0.723684 |          175 |            0.001655 |            0.662857 |          2.337047 |           48.589456 |
|         0 | (none)                              |      0.000534 |      0.637119 |          787 |            0.000347 |            0.580686 |          1.453978 |           64.998711 |
|       140 | A_15m, B_15m, B_1d                  |      0.002254 |      0.669173 |          294 |            0.000765 |            0.551020 |          2.089026 |           33.946997 |
|        31 | A_5m, B_5m, A_15m, B_15m, A_1h      |      0.000584 |      0.617176 |          688 |           -0.000041 |            0.572674 |         -0.133986 |           -6.944091 |
|       182 | B_5m, A_15m, A_1h, B_1h, B_1d       |      0.001244 |      0.681250 |           98 |            0.000819 |            0.663265 |          0.833049 |           65.838385 |
|       211 | A_5m, B_5m, A_1h, A_1d, B_1d        |      0.001944 |      0.843137 |           28 |            0.001474 |            0.821429 |          1.054306 |           75.791772 |
|       231 | A_5m, B_5m, A_15m, B_1h, A_1d, B_1d |      0.000930 |      0.695652 |          128 |            0.000840 |            0.625000 |          1.786303 |           90.272184 |
|       181 | A_5m, A_15m, A_1h, B_1h, B_1d       |      0.001469 |      0.619318 |          142 |            0.000655 |            0.542254 |          0.884068 |           44.576477 |

_`edge_retained_pct` = `mean_ret_shuffled / mean_ret_in * 100`. Values near 0% (or negative, i.e. sign flip) indicate the shuffled flag is load-bearing for that pattern's edge. This is a partial test (only `A_5m` of 8 constituent flags is shuffled), so a non-zero residual is expected even for genuine patterns -- it is reported as a diagnostic, not a pass/fail gate on its own._

## T6.6 -- Year-by-year stability (2020-2025)

`evaluate_rules` is re-run on each calendar year separately. `n_years_consistent` counts years with `n >= 5` and `sign(mean_ret) == expected_sign`; `single_year_driven` is True if `n_years_consistent <= 1` (the in-sample edge would then be carried by a single calendar year, e.g. a COVID-recovery artifact).

|   rule_id | active_flags                        |   mean_ret_2020 |   mean_ret_2021 |   mean_ret_2022 |   mean_ret_2023 |   mean_ret_2024 |   mean_ret_2025 |   n_years_consistent |   n_years_total | single_year_driven   |
|----------:|:------------------------------------|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|---------------------:|----------------:|:---------------------|
|       151 | A_5m, B_5m, A_15m, A_1h, B_1d       |      nan        |      nan        |        0.001412 |        0.001474 |        0.000907 |       -0.000057 |                    3 |               6 | False                |
|       121 | A_5m, B_15m, A_1h, B_1h, A_1d       |      nan        |      nan        |        0.002524 |        0.000510 |       -0.000967 |        0.000219 |                    3 |               6 | False                |
|        26 | B_5m, B_15m, A_1h                   |      nan        |      nan        |        0.011370 |        0.001286 |       -0.001520 |        0.000433 |                    3 |               6 | False                |
|         0 | (none)                              |        0.000721 |        0.000425 |       -0.002051 |       -0.001432 |        0.000984 |      nan        |                    3 |               6 | False                |
|       140 | A_15m, B_15m, B_1d                  |      nan        |      nan        |        0.005975 |        0.000152 |        0.000405 |        0.000443 |                    4 |               6 | False                |
|        31 | A_5m, B_5m, A_15m, B_15m, A_1h      |      nan        |        0.000969 |        0.000233 |        0.000029 |        0.000343 |        0.000152 |                    5 |               6 | False                |
|       182 | B_5m, A_15m, A_1h, B_1h, B_1d       |      nan        |      nan        |        0.002021 |       -0.000939 |      nan        |        0.000169 |                    2 |               6 | False                |
|       211 | A_5m, B_5m, A_1h, A_1d, B_1d        |      nan        |      nan        |        0.000060 |        0.003756 |      nan        |       -0.001968 |                    2 |               6 | False                |
|       231 | A_5m, B_5m, A_15m, B_1h, A_1d, B_1d |      nan        |      nan        |        0.001684 |        0.000468 |       -0.000415 |       -0.000035 |                    2 |               6 | False                |
|       181 | A_5m, A_15m, A_1h, B_1h, B_1d       |      nan        |      nan        |        0.001433 |        0.001560 |      nan        |       -0.000128 |                    2 |               6 | False                |

- 0/10 patterns are single-year-driven.

## T6.7 -- Time-of-day baseline

For each pattern, `tod_matched_mean_ret` is the unconditional per-`bar_in_session` mean(`ret_fwd_12`) (computed over the in-sample set), re-weighted by the pattern's own `bar_in_session` firing distribution. `excess_over_tod = rule_mean_ret - tod_matched_mean_ret` is the part of the edge *not* explained by simply firing more (or less) often at certain times of day. `pct_first_hour_*` is the fraction of (firing / all) rows with `bar_in_session <= 12` (first 60 minutes of the session).

|   rule_id | active_flags                        |   n |   rule_mean_ret |   tod_matched_mean_ret |   excess_over_tod |   pct_first_hour_fired |   pct_first_hour_baseline |
|----------:|:------------------------------------|----:|----------------:|-----------------------:|------------------:|-----------------------:|--------------------------:|
|       151 | A_5m, B_5m, A_15m, A_1h, B_1d       | 324 |        0.001436 |               0.000140 |          0.001296 |               0.280864 |                  0.160950 |
|       121 | A_5m, B_15m, A_1h, B_1h, A_1d       | 211 |        0.001894 |               0.000238 |          0.001656 |               0.085308 |                  0.160950 |
|        26 | B_5m, B_15m, A_1h                   |  76 |        0.003406 |               0.000229 |          0.003177 |               0.078947 |                  0.160950 |
|         0 | (none)                              | 722 |        0.000534 |               0.000105 |          0.000429 |               0.138504 |                  0.160950 |
|       140 | A_15m, B_15m, B_1d                  | 133 |        0.002254 |               0.000152 |          0.002101 |               0.075188 |                  0.160950 |
|        31 | A_5m, B_5m, A_15m, B_15m, A_1h      | 687 |        0.000584 |               0.000063 |          0.000522 |               0.296943 |                  0.160950 |
|       182 | B_5m, A_15m, A_1h, B_1h, B_1d       | 160 |        0.001244 |               0.000103 |          0.001141 |               0.081250 |                  0.160950 |
|       211 | A_5m, B_5m, A_1h, A_1d, B_1d        |  51 |        0.001944 |               0.000219 |          0.001725 |               0.156863 |                  0.160950 |
|       231 | A_5m, B_5m, A_15m, B_1h, A_1d, B_1d | 184 |        0.000930 |               0.000107 |          0.000824 |               0.364130 |                  0.160950 |
|       181 | A_5m, A_15m, A_1h, B_1h, B_1d       | 176 |        0.001469 |               0.000109 |          0.001360 |               0.301136 |                  0.160950 |

## T6.8 -- Verdict

**2/10 patterns survive all three checks** (no OOS sign-collapse on val/test, not single-year-driven, edge not explained by time-of-day clustering):

- **Rule 140** (A_15m, B_15m, B_1d): in-sample mean_ret=0.002254 (hit_rate=0.669, n=133); val mean_ret=0.000405 (hit_rate=0.475, n=61); test mean_ret=0.000443 (hit_rate=0.600, n=65); 4/6 years consistent with expected_sign; excess_over_tod=0.002101 vs rule_mean_ret=0.002254.

- **Rule 31** (A_5m, B_5m, A_15m, B_15m, A_1h): in-sample mean_ret=0.000584 (hit_rate=0.617, n=687); val mean_ret=0.000343 (hit_rate=0.556, n=198); test mean_ret=0.000152 (hit_rate=0.535, n=101); 5/6 years consistent with expected_sign; excess_over_tod=0.000522 vs rule_mean_ret=0.000584.


**Caveats** -- neither surviving pattern was significant after the original x256 Bonferroni correction in 5a (`p_val_bonf=1.000` for both, see `reports/phase5/rules_survivors.csv`). Edges are economically small (tens of bps over a 12-bar / ~1h forward horizon) and computed with no transaction costs. "Survives Phase 6" should be read as **"did not get falsified by out-of-sample data and three robustness checks"**, not as "a confirmed, tradeable edge."


**Patterns that did NOT survive:**

- Rule 151 (A_5m, B_5m, A_15m, A_1h, B_1d): sign-collapsed on 2025

- Rule 121 (A_5m, B_15m, A_1h, B_1h, A_1d): sign-collapsed on 2024

- Rule 26 (B_5m, B_15m, A_1h): sign-collapsed on 2024

- Rule 0 ((none)): sign-collapsed on 2025

- Rule 182 (B_5m, A_15m, A_1h, B_1h, B_1d): sign-collapsed on 2024

- Rule 211 (A_5m, B_5m, A_1h, A_1d, B_1d): sign-collapsed on 2024/2025

- Rule 231 (A_5m, B_5m, A_15m, B_1h, A_1d, B_1d): sign-collapsed on 2024/2025

- Rule 181 (A_5m, A_15m, A_1h, B_1h, B_1d): sign-collapsed on 2024/2025
