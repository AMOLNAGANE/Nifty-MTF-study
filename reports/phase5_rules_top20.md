# Phase 5a — Confluence Rule Mining (T5.1-T5.4)

All 2^8 = 256 boolean combinations of the 8 (indicator, TF) bull flags (A_5m, B_5m, A_15m, B_15m, A_1h, B_1h, A_1d, B_1d; flag = `A_hist{suffix} > 0` or `B_roc{suffix} > 0`, NaN -> not bullish) are enumerated and evaluated on the **rule-discovery set** (`bar_close.dt.year <= 2023`, n=58,080 rows -- years 1-3). For each rule, `mean_ret` / `hit_rate` / `t_stat` are computed on `ret_fwd_12` (Newey-West HAC, lags=12); `p_val_bonf` applies a Bonferroni correction (x256 rules). `composite = frequency * |mean_ret| * (hit_rate - 0.5)` ranks rules by a combination of prevalence, edge size, and edge-direction consistency. Full 256-row table: `reports/phase5/rules_all.csv`.

## Top 20 rules (by composite)

|   rule_id | A_5m   | B_5m   | A_15m   | B_15m   | A_1h   | B_1h   | A_1d   | B_1d   |   n |   frequency |   mean_ret |   hit_rate |     t_stat |   p_val_raw |   p_val_bonf |   composite |
|----------:|:-------|:-------|:--------|:--------|:-------|:-------|:-------|:-------|----:|------------:|-----------:|-----------:|-----------:|------------:|-------------:|------------:|
|       151 | True   | True   | True    | False   | True   | False  | False  | True   | 324 |    0.005579 |   0.001436 |   0.682099 |   3.382506 |    0.000718 |     0.183879 |    0.000001 |
|       121 | True   | False  | False   | True    | True   | True   | True   | False  | 211 |    0.003633 |   0.001894 |   0.691943 |   3.690514 |    0.000224 |     0.057293 |    0.000001 |
|        26 | False  | True   | False   | True    | True   | False  | False  | False  |  76 |    0.001309 |   0.003406 |   0.723684 |   1.911351 |    0.055960 |     1.000000 |    0.000001 |
|         0 | False  | False  | False   | False   | False  | False  | False  | False  | 722 |    0.012431 |   0.000534 |   0.637119 |   1.752229 |    0.079734 |     1.000000 |    0.000001 |
|       140 | False  | False  | True    | True    | False  | False  | False  | True   | 133 |    0.002290 |   0.002254 |   0.669173 |   2.170376 |    0.029978 |     1.000000 |    0.000001 |
|        31 | True   | True   | True    | True    | True   | False  | False  | False  | 687 |    0.011829 |   0.000584 |   0.617176 |   1.181778 |    0.237294 |     1.000000 |    0.000001 |
|       182 | False  | True   | True    | False   | True   | True   | False  | True   | 160 |    0.002755 |   0.001244 |   0.681250 |   2.085209 |    0.037050 |     1.000000 |    0.000001 |
|       211 | True   | True   | False   | False   | True   | False  | True   | True   |  51 |    0.000878 |   0.001944 |   0.843137 |   1.849419 |    0.064397 |     1.000000 |    0.000001 |
|       231 | True   | True   | True    | False   | False  | True   | True   | True   | 184 |    0.003168 |   0.000930 |   0.695652 |   2.179388 |    0.029303 |     1.000000 |    0.000001 |
|       150 | False  | True   | True    | False   | True   | False  | False  | True   | 368 |    0.006336 |   0.000771 |   0.611413 |   1.574218 |    0.115437 |     1.000000 |    0.000001 |
|       181 | True   | False  | True    | False   | True   | True   | False  | True   | 176 |    0.003030 |   0.001469 |   0.619318 |   2.061184 |    0.039286 |     1.000000 |    0.000001 |
|       215 | True   | True   | True    | False   | True   | False  | True   | True   |  75 |    0.001291 |   0.001991 |   0.693333 |   1.956133 |    0.050449 |     1.000000 |    0.000000 |
|       189 | True   | False  | True    | True    | True   | True   | False  | True   |  10 |    0.000172 |   0.005767 |   1.000000 | nan        |  nan        |   nan        |    0.000000 |
|        37 | True   | False  | True    | False   | False  | True   | False  | False  | 889 |    0.015306 |   0.000475 |   0.565804 |   1.438459 |    0.150304 |     1.000000 |    0.000000 |
|       208 | False  | False  | False   | False   | True   | False  | True   | True   | 129 |    0.002221 |   0.000859 |   0.736434 |   1.401964 |    0.160926 |     1.000000 |    0.000000 |
|        27 | True   | True   | False   | True    | True   | False  | False  | False  | 266 |    0.004580 |   0.000970 |   0.597744 |   1.810674 |    0.070191 |     1.000000 |    0.000000 |
|       173 | True   | False  | True    | True    | False  | True   | False  | True   | 101 |    0.001739 |   0.001678 |   0.643564 |   1.429175 |    0.152954 |     1.000000 |    0.000000 |
|       207 | True   | True   | True    | True    | False  | False  | True   | True   | 107 |    0.001842 |   0.001581 |   0.635514 |   2.075976 |    0.037896 |     1.000000 |    0.000000 |
|        84 | False  | False  | True    | False   | True   | False  | True   | False  |  33 |    0.000568 |   0.004989 |   0.636364 |   1.385303 |    0.165960 |     1.000000 |    0.000000 |
|       120 | False  | False  | False   | True    | True   | True   | True   | False  | 654 |    0.011260 |   0.000424 |   0.579511 |   1.481427 |    0.138493 |     1.000000 |    0.000000 |

## Bottom 20 rules (by composite)

|   rule_id | A_5m   | B_5m   | A_15m   | B_15m   | A_1h   | B_1h   | A_1d   | B_1d   |   n |   frequency |   mean_ret |   hit_rate |    t_stat |   p_val_raw |   p_val_bonf |   composite |
|----------:|:-------|:-------|:--------|:--------|:-------|:-------|:-------|:-------|----:|------------:|-----------:|-----------:|----------:|------------:|-------------:|------------:|
|       234 | False  | True   | False   | True    | False  | True   | True   | True   |  43 |    0.000740 |  -0.002920 |   0.162791 | -2.859912 |    0.004238 |     1.000000 |   -0.000001 |
|        77 | True   | False  | True    | True    | False  | False  | True   | False  | 306 |    0.005269 |  -0.000909 |   0.392157 | -2.099840 |    0.035743 |     1.000000 |   -0.000001 |
|       139 | True   | True   | False   | True    | False  | False  | False  | True   | 235 |    0.004046 |  -0.001128 |   0.387234 | -1.646718 |    0.099616 |     1.000000 |   -0.000001 |
|       137 | True   | False  | False   | True    | False  | False  | False  | True   |  98 |    0.001687 |  -0.001365 |   0.316327 | -2.370980 |    0.017741 |     1.000000 |   -0.000000 |
|        68 | False  | False  | True    | False   | False  | False  | True   | False  | 228 |    0.003926 |  -0.001005 |   0.416667 | -2.041663 |    0.041185 |     1.000000 |   -0.000000 |
|       155 | True   | True   | False   | True    | True   | False  | False  | True   |  40 |    0.000689 |  -0.001957 |   0.300000 | -1.759984 |    0.078410 |     1.000000 |   -0.000000 |
|       169 | True   | False  | False   | True    | False  | True   | False  | True   |  23 |    0.000396 |  -0.002330 |   0.217391 | -3.214661 |    0.001306 |     0.334333 |   -0.000000 |
|       197 | True   | False  | True    | False   | False  | False  | True   | True   | 464 |    0.007989 |  -0.000609 |   0.450431 | -1.870440 |    0.061423 |     1.000000 |   -0.000000 |
|        71 | True   | True   | True    | False   | False  | False  | True   | False  | 141 |    0.002428 |  -0.002119 |   0.453901 | -0.774700 |    0.438517 |     1.000000 |   -0.000000 |
|        75 | True   | True   | False   | True    | False  | False  | True   | False  | 266 |    0.004580 |  -0.000479 |   0.409774 | -1.108916 |    0.267467 |     1.000000 |   -0.000000 |
|        70 | False  | True   | True    | False   | False  | False  | True   | False  | 184 |    0.003168 |   0.002067 |   0.472826 |  1.096891 |    0.272689 |     1.000000 |   -0.000000 |
|        67 | True   | True   | False   | False   | False  | False  | True   | False  | 269 |    0.004632 |  -0.000608 |   0.438662 | -1.369881 |    0.170724 |     1.000000 |   -0.000000 |
|       138 | False  | True   | False   | True    | False  | False  | False  | True   | 311 |    0.005355 |  -0.000371 |   0.414791 | -1.104176 |    0.269517 |     1.000000 |   -0.000000 |
|       130 | False  | True   | False   | False   | False  | False  | False  | True   | 178 |    0.003065 |  -0.000462 |   0.382022 | -1.442612 |    0.149130 |     1.000000 |   -0.000000 |
|        34 | False  | True   | False   | False   | False  | True   | False  | False  | 367 |    0.006319 |   0.000427 |   0.438692 |  0.588768 |    0.556017 |     1.000000 |   -0.000000 |
|       243 | True   | True   | False   | False   | True   | True   | True   | True   |  98 |    0.001687 |  -0.000642 |   0.357143 | -2.152531 |    0.031356 |     1.000000 |   -0.000000 |
|        11 | True   | True   | False   | True    | False  | False  | False  | False  | 498 |    0.008574 |  -0.000344 |   0.449799 | -1.051988 |    0.292805 |     1.000000 |   -0.000000 |
|        10 | False  | True   | False   | True    | False  | False  | False  | False  | 676 |    0.011639 |  -0.000409 |   0.468935 | -1.254495 |    0.209662 |     1.000000 |   -0.000000 |
|       201 | True   | False  | False   | True    | False  | False  | True   | True   |  73 |    0.001257 |  -0.000582 |   0.301370 | -1.983421 |    0.047320 |     1.000000 |   -0.000000 |
|       171 | True   | True   | False   | True    | False  | True   | False  | True   |  34 |    0.000585 |  -0.001241 |   0.323529 | -0.833242 |    0.404708 |     1.000000 |   -0.000000 |

## Survivors (Hamming-deduped, min_hamming=2)

Rules from the top-20 whose 8-flag vector differs by >= 2 bits from every higher-ranked survivor (drops trivially-related single-flag variants). n=15.

|   rule_id | A_5m   | B_5m   | A_15m   | B_15m   | A_1h   | B_1h   | A_1d   | B_1d   |   n |   frequency |   mean_ret |   hit_rate |   t_stat |   p_val_raw |   p_val_bonf |   composite |
|----------:|:-------|:-------|:--------|:--------|:-------|:-------|:-------|:-------|----:|------------:|-----------:|-----------:|---------:|------------:|-------------:|------------:|
|       151 | True   | True   | True    | False   | True   | False  | False  | True   | 324 |    0.005579 |   0.001436 |   0.682099 | 3.382506 |    0.000718 |     0.183879 |    0.000001 |
|       121 | True   | False  | False   | True    | True   | True   | True   | False  | 211 |    0.003633 |   0.001894 |   0.691943 | 3.690514 |    0.000224 |     0.057293 |    0.000001 |
|        26 | False  | True   | False   | True    | True   | False  | False  | False  |  76 |    0.001309 |   0.003406 |   0.723684 | 1.911351 |    0.055960 |     1.000000 |    0.000001 |
|         0 | False  | False  | False   | False   | False  | False  | False  | False  | 722 |    0.012431 |   0.000534 |   0.637119 | 1.752229 |    0.079734 |     1.000000 |    0.000001 |
|       140 | False  | False  | True    | True    | False  | False  | False  | True   | 133 |    0.002290 |   0.002254 |   0.669173 | 2.170376 |    0.029978 |     1.000000 |    0.000001 |
|        31 | True   | True   | True    | True    | True   | False  | False  | False  | 687 |    0.011829 |   0.000584 |   0.617176 | 1.181778 |    0.237294 |     1.000000 |    0.000001 |
|       182 | False  | True   | True    | False   | True   | True   | False  | True   | 160 |    0.002755 |   0.001244 |   0.681250 | 2.085209 |    0.037050 |     1.000000 |    0.000001 |
|       211 | True   | True   | False   | False   | True   | False  | True   | True   |  51 |    0.000878 |   0.001944 |   0.843137 | 1.849419 |    0.064397 |     1.000000 |    0.000001 |
|       231 | True   | True   | True    | False   | False  | True   | True   | True   | 184 |    0.003168 |   0.000930 |   0.695652 | 2.179388 |    0.029303 |     1.000000 |    0.000001 |
|       181 | True   | False  | True    | False   | True   | True   | False  | True   | 176 |    0.003030 |   0.001469 |   0.619318 | 2.061184 |    0.039286 |     1.000000 |    0.000001 |
|        37 | True   | False  | True    | False   | False  | True   | False  | False  | 889 |    0.015306 |   0.000475 |   0.565804 | 1.438459 |    0.150304 |     1.000000 |    0.000000 |
|       208 | False  | False  | False   | False   | True   | False  | True   | True   | 129 |    0.002221 |   0.000859 |   0.736434 | 1.401964 |    0.160926 |     1.000000 |    0.000000 |
|       173 | True   | False  | True    | True    | False  | True   | False  | True   | 101 |    0.001739 |   0.001678 |   0.643564 | 1.429175 |    0.152954 |     1.000000 |    0.000000 |
|       207 | True   | True   | True    | True    | False  | False  | True   | True   | 107 |    0.001842 |   0.001581 |   0.635514 | 2.075976 |    0.037896 |     1.000000 |    0.000000 |
|        84 | False  | False  | True    | False   | True   | False  | True   | False  |  33 |    0.000568 |   0.004989 |   0.636364 | 1.385303 |    0.165960 |     1.000000 |    0.000000 |
