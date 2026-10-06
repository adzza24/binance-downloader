# BTC 2024 Matched W3 Lookalike Trajectory Comparison

**RESEARCH ONLY. No live strategy or automation changes. 2025/2026 were not read.**

- Winner event anchors: **108**
- Matched non-W3 controls: **540** (5 per event)
- Controls excluded within ±24h of any W3 anchor and matched within the same quarter.
- Median state-space distance: **0.328**; p90 **0.877**.
- Median absolute post-match SMD across matching variables: **0.131**; max **0.296**.

## Current-state matching balance

| feature         |   anchor_mean |   all_negative_mean |   matched_control_mean |   smd_before |   smd_after |   abs_smd_after |
|:----------------|--------------:|--------------------:|-----------------------:|-------------:|------------:|----------------:|
| rsi14           |       37.4113 |             51.5256 |                37.3424 |      -1.2501 |      0.0067 |          0.0067 |
| bb_z            |       -1.7436 |              0.0977 |                -1.7032 |      -1.5373 |     -0.0377 |          0.0377 |
| rsi6            |       27.3400 |             51.5288 |                27.9011 |      -1.5909 |     -0.0424 |          0.0424 |
| cci20           |     -144.0589 |              8.3755 |              -138.0982 |      -1.4032 |     -0.0593 |          0.0593 |
| stoch_k         |       20.5867 |             53.5736 |                19.3587 |      -1.4309 |      0.0670 |          0.0670 |
| macd_hist_pct   |       -0.0020 |              0.0000 |                -0.0017 |      -1.0503 |     -0.1184 |          0.1184 |
| close_vs_vwap24 |       -0.0159 |              0.0017 |                -0.0141 |      -1.2924 |     -0.1433 |          0.1433 |
| close_vs_ema24  |       -0.0163 |              0.0012 |                -0.0138 |      -1.1874 |     -0.1736 |          0.1736 |
| log_ret         |       -0.0079 |              0.0001 |                -0.0065 |      -1.0893 |     -0.1784 |          0.1784 |
| close_vs_ema12  |       -0.0134 |              0.0006 |                -0.0111 |      -1.3331 |     -0.2230 |          0.2230 |
| close_vs_ema6   |       -0.0095 |              0.0002 |                -0.0076 |      -1.3526 |     -0.2598 |          0.2598 |
| atr14_pct       |        0.0090 |              0.0074 |                 0.0079 |       0.3910 |      0.2956 |          0.2956 |

## What happened to matched failures

| failure_type          |   controls |   share |
|:----------------------|-----------:|--------:|
| STOP_FIRST            |        261 |  0.4833 |
| NO_TARGET_WITHIN_DATA |        221 |  0.4093 |
| TARGET_AFTER_24H      |         58 |  0.1074 |

## Strongest *trajectory* differences after matching current state

| feature              | base_indicator   |   n_events |   winner_mean |   control_mean |   mean_paired_diff |   median_paired_diff |   paired_effect |   abs_paired_effect | direction   |   sign_consistency |   p_value | shape_only   |   fdr_q |
|:---------------------|:-----------------|-----------:|--------------:|---------------:|-------------------:|---------------------:|----------------:|--------------------:|:------------|-------------------:|----------:|:-------------|--------:|
| ema_gap_6_24__d1     | ema_gap_6_24     |        108 |       -0.0024 |        -0.0018 |            -0.0005 |              -0.0002 |         -0.5038 |              0.5038 | WINNER_LOW  |             0.7500 |    0.0000 | True         |  0.0010 |
| close_vs_vwap24__d3  | close_vs_vwap24  |        108 |       -0.0110 |        -0.0076 |            -0.0034 |              -0.0026 |         -0.4961 |              0.4961 | WINNER_LOW  |             0.6759 |    0.0000 | True         |  0.0010 |
| macd_pct__d1         | macd_pct         |        108 |       -0.0012 |        -0.0009 |            -0.0003 |              -0.0001 |         -0.4861 |              0.4861 | WINNER_LOW  |             0.7500 |    0.0000 | True         |  0.0010 |
| close_vs_sma72__d3   | close_vs_sma72   |        108 |       -0.0133 |        -0.0099 |            -0.0034 |              -0.0025 |         -0.4743 |              0.4743 | WINNER_LOW  |             0.7315 |    0.0000 | True         |  0.0010 |
| macd_hist_pct__d1    | macd_hist_pct    |        108 |       -0.0007 |        -0.0005 |            -0.0002 |              -0.0001 |         -0.4695 |              0.4695 | WINNER_LOW  |             0.7130 |    0.0000 | True         |  0.0011 |
| close_vs_ema72__d3   | close_vs_ema72   |        108 |       -0.0128 |        -0.0095 |            -0.0033 |              -0.0022 |         -0.4585 |              0.4585 | WINNER_LOW  |             0.7222 |    0.0000 | True         |  0.0012 |
| close_vs_ema48__d3   | close_vs_ema48   |        108 |       -0.0123 |        -0.0090 |            -0.0033 |              -0.0021 |         -0.4566 |              0.4566 | WINNER_LOW  |             0.7315 |    0.0000 | True         |  0.0012 |
| close_vs_ema168__d3  | close_vs_ema168  |        108 |       -0.0135 |        -0.0103 |            -0.0032 |              -0.0021 |         -0.4511 |              0.4511 | WINNER_LOW  |             0.6944 |    0.0000 | True         |  0.0014 |
| close_vs_ema240__d3  | close_vs_ema240  |        108 |       -0.0136 |        -0.0105 |            -0.0032 |              -0.0022 |         -0.4494 |              0.4494 | WINNER_LOW  |             0.6944 |    0.0000 | True         |  0.0014 |
| close_vs_ema24__d3   | close_vs_ema24   |        108 |       -0.0110 |        -0.0079 |            -0.0031 |              -0.0020 |         -0.4435 |              0.4435 | WINNER_LOW  |             0.7037 |    0.0000 | True         |  0.0014 |
| close_vs_sma24__d3   | close_vs_sma24   |        108 |       -0.0120 |        -0.0087 |            -0.0033 |              -0.0023 |         -0.4403 |              0.4403 | WINNER_LOW  |             0.6852 |    0.0000 | True         |  0.0014 |
| atr24_pct__d3        | atr24_pct        |        108 |        0.0006 |         0.0003 |             0.0004 |               0.0002 |          0.4398 |              0.4398 | WINNER_HIGH |             0.6852 |    0.0000 | True         |  0.0014 |
| close_vs_sma240__d3  | close_vs_sma240  |        108 |       -0.0138 |        -0.0106 |            -0.0032 |              -0.0022 |         -0.4385 |              0.4385 | WINNER_LOW  |             0.6759 |    0.0000 | True         |  0.0014 |
| close_vs_sma168__d3  | close_vs_sma168  |        108 |       -0.0137 |        -0.0106 |            -0.0031 |              -0.0022 |         -0.4382 |              0.4382 | WINNER_LOW  |             0.6944 |    0.0000 | True         |  0.0014 |
| close_vs_vwap72__d3  | close_vs_vwap72  |        108 |       -0.0126 |        -0.0097 |            -0.0029 |              -0.0017 |         -0.4348 |              0.4348 | WINNER_LOW  |             0.7222 |    0.0000 | True         |  0.0016 |
| body_pct__w24__pos   | body_pct         |        108 |        0.2580 |         0.3213 |            -0.0633 |              -0.0784 |         -0.4284 |              0.4284 | WINNER_LOW  |             0.6481 |    0.0000 | True         |  0.0019 |
| log_ret__w24__pos    | log_ret          |        108 |        0.2595 |         0.3231 |            -0.0636 |              -0.0790 |         -0.4280 |              0.4280 | WINNER_LOW  |             0.6481 |    0.0000 | True         |  0.0019 |
| close_vs_vwap240__d3 | close_vs_vwap240 |        108 |       -0.0136 |        -0.0106 |            -0.0029 |              -0.0020 |         -0.4253 |              0.4253 | WINNER_LOW  |             0.6667 |    0.0000 | True         |  0.0020 |
| dist_high240__d3     | dist_high240     |        108 |       -0.0126 |        -0.0097 |            -0.0029 |              -0.0020 |         -0.4222 |              0.4222 | WINNER_LOW  |             0.7500 |    0.0000 | True         |  0.0022 |
| close_vs_ema12__d3   | close_vs_ema12   |        108 |       -0.0090 |        -0.0062 |            -0.0028 |              -0.0018 |         -0.4183 |              0.4183 | WINNER_LOW  |             0.6667 |    0.0000 | True         |  0.0025 |

## Strongest raw sequence-shape differences

|   window_h |   offset_h | channel        |   winner_mean_z |   control_mean_z |   paired_effect |   shape_paired_effect |   abs_shape_effect |
|-----------:|-----------:|:---------------|----------------:|-----------------:|----------------:|----------------------:|-------------------:|
|        240 |       -180 | stoch_k        |         -0.1774 |           0.0855 |         -0.4156 |               -0.4728 |             0.4728 |
|         24 |         -1 | macd_hist_pct  |         -0.8798 |          -0.8593 |         -0.0270 |                0.4695 |             0.4695 |
|        240 |       -220 | range_pct      |          0.3712 |           0.6075 |         -0.2440 |               -0.4566 |             0.4566 |
|         24 |         -3 | close_vs_ema24 |         -0.7153 |          -0.7778 |          0.0606 |                0.4230 |             0.4230 |
|        240 |       -220 | atr14_pct      |          0.1559 |           0.3870 |         -0.1959 |               -0.4102 |             0.4102 |
|         72 |        -60 | log_ret        |          0.0142 |          -0.0938 |          0.1205 |                0.4079 |             0.4079 |
|         72 |        -60 | body_pct       |          0.0177 |          -0.0906 |          0.1210 |                0.4069 |             0.4069 |
|        240 |       -220 | bb_width       |          0.2158 |           0.5656 |         -0.2879 |               -0.3659 |             0.3659 |
|        240 |       -150 | trades_rel72   |          0.7222 |           0.1779 |          0.3992 |                0.3622 |             0.3622 |
|         24 |         -3 | volume_rel72   |          0.7042 |           1.0600 |         -0.2326 |               -0.3526 |             0.3526 |
|         24 |        -13 | close_location |          0.1208 |          -0.0759 |          0.2249 |                0.2995 |             0.2995 |
|        240 |       -220 | taker_share    |          0.0564 |          -0.1114 |          0.3948 |                0.2795 |             0.2795 |
|         72 |        -57 | lower_wick     |          0.2587 |           0.1204 |          0.2234 |                0.2598 |             0.2598 |
|        240 |       -110 | rsi14          |         -0.1298 |           0.0604 |         -0.1797 |               -0.2273 |             0.2273 |
|        240 |       -110 | upper_wick     |          0.2248 |           0.1518 |          0.2081 |                0.1796 |             0.1796 |

## Can trajectory pick the winner from five state-lookalikes?

| model              |   rows |   events |   class_precision_baseline |    auc |   average_precision |   winner_top1_rate |   winner_top2_rate |   winner_top3_rate |   median_winner_rank |   random_top1 |   random_top2 |
|:-------------------|-------:|---------:|---------------------------:|-------:|--------------------:|-------------------:|-------------------:|-------------------:|---------------------:|--------------:|--------------:|
| TEMPORAL_CHANGE    |    648 |      108 |                     0.1667 | 0.6850 |              0.3647 |             0.4630 |             0.5833 |             0.6944 |               2.0000 |        0.1667 |        0.3333 |
| RAW_SEQUENCE_SHAPE |    648 |      108 |                     0.1667 | 0.6737 |              0.3381 |             0.3611 |             0.5556 |             0.7037 |               2.0000 |        0.1667 |        0.3333 |

Random reference: one winner among six candidates gives 16.7% top-1 and 33.3% top-2.

These CV diagnostics are discovery-only within 2024; they are not a trading backtest and are not evidence of 2025/2026 performance.