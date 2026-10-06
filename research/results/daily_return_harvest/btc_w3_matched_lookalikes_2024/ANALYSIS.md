# BTC 2024 Matched W3 Lookalike Trajectory Comparison

**RESEARCH ONLY. No live strategy or automation changes. 2025/2026 were not read.**

- Winner event anchors: **108**
- Matched non-W3 controls: **540** (5 per event)
- Controls excluded within ±24h of any W3 anchor and matched within the same quarter.
- Median state-space distance: **0.474**; p90 **1.190**.
- Median absolute post-match SMD across matching variables: **0.391**; max **0.571**.

## Current-state matching balance

| feature         |   anchor_mean |   all_negative_mean |   matched_control_mean |   smd_before |   smd_after |   abs_smd_after |
|:----------------|--------------:|--------------------:|-----------------------:|-------------:|------------:|----------------:|
| stoch_k         |       20.5867 |             53.5736 |                22.1789 |      -1.4309 |     -0.0881 |          0.0881 |
| rsi14           |       37.4113 |             51.5256 |                39.0834 |      -1.2501 |     -0.1673 |          0.1673 |
| cci20           |     -144.0589 |              8.3755 |              -122.9285 |      -1.4032 |     -0.2165 |          0.2165 |
| bb_z            |       -1.7436 |              0.0977 |                -1.4964 |      -1.5373 |     -0.2402 |          0.2402 |
| rsi6            |       27.3400 |             51.5288 |                30.4925 |      -1.5909 |     -0.2478 |          0.2478 |
| macd_hist_pct   |       -0.0020 |              0.0000 |                -0.0014 |      -1.0503 |     -0.3628 |          0.3628 |
| close_vs_vwap24 |       -0.0159 |              0.0017 |                -0.0109 |      -1.2924 |     -0.4186 |          0.4186 |
| close_vs_ema24  |       -0.0163 |              0.0012 |                -0.0103 |      -1.1874 |     -0.4396 |          0.4396 |
| atr14_pct       |        0.0090 |              0.0074 |                 0.0072 |       0.3910 |      0.4696 |          0.4696 |
| log_ret         |       -0.0079 |              0.0001 |                -0.0043 |      -1.0893 |     -0.4862 |          0.4862 |
| close_vs_ema12  |       -0.0134 |              0.0006 |                -0.0082 |      -1.3331 |     -0.5263 |          0.5263 |
| close_vs_ema6   |       -0.0095 |              0.0002 |                -0.0055 |      -1.3526 |     -0.5707 |          0.5707 |

## What happened to matched failures

| failure_type          |   controls |   share |
|:----------------------|-----------:|--------:|
| STOP_FIRST            |        258 |  0.4778 |
| NO_TARGET_WITHIN_DATA |        221 |  0.4093 |
| TARGET_AFTER_24H      |         61 |  0.1130 |

## Strongest *trajectory* differences after matching current state

| feature                 | base_indicator   |   n_events |   winner_mean |   control_mean |   mean_paired_diff |   median_paired_diff |   paired_effect |   abs_paired_effect | direction   |   sign_consistency |   p_value | shape_only   |   fdr_q |
|:------------------------|:-----------------|-----------:|--------------:|---------------:|-------------------:|---------------------:|----------------:|--------------------:|:------------|-------------------:|----------:|:-------------|--------:|
| ema_gap_6_24__d1        | ema_gap_6_24     |        108 |       -0.0024 |        -0.0013 |            -0.0010 |              -0.0007 |         -0.8085 |              0.8085 | WINNER_LOW  |             0.8611 |    0.0000 | True         |  0.0000 |
| macd_pct__d1            | macd_pct         |        108 |       -0.0012 |        -0.0007 |            -0.0005 |              -0.0004 |         -0.7827 |              0.7827 | WINNER_LOW  |             0.8241 |    0.0000 | True         |  0.0000 |
| close_vs_sma72__d1      | close_vs_sma72   |        108 |       -0.0074 |        -0.0041 |            -0.0033 |              -0.0024 |         -0.7726 |              0.7726 | WINNER_LOW  |             0.7593 |    0.0000 | True         |  0.0000 |
| close_vs_vwap240__d1    | close_vs_vwap240 |        108 |       -0.0075 |        -0.0044 |            -0.0032 |              -0.0022 |         -0.7685 |              0.7685 | WINNER_LOW  |             0.7778 |    0.0000 | True         |  0.0000 |
| close_vs_vwap72__d1     | close_vs_vwap72  |        108 |       -0.0070 |        -0.0040 |            -0.0030 |              -0.0024 |         -0.7634 |              0.7634 | WINNER_LOW  |             0.7500 |    0.0000 | True         |  0.0000 |
| close_vs_ema240__d1     | close_vs_ema240  |        108 |       -0.0076 |        -0.0043 |            -0.0033 |              -0.0023 |         -0.7600 |              0.7600 | WINNER_LOW  |             0.7593 |    0.0000 | True         |  0.0000 |
| dist_high240__d1        | dist_high240     |        108 |       -0.0071 |        -0.0040 |            -0.0031 |              -0.0019 |         -0.7595 |              0.7595 | WINNER_LOW  |             0.7778 |    0.0000 | True         |  0.0000 |
| close_vs_ema168__d1     | close_vs_ema168  |        108 |       -0.0075 |        -0.0042 |            -0.0033 |              -0.0022 |         -0.7591 |              0.7591 | WINNER_LOW  |             0.7778 |    0.0000 | True         |  0.0000 |
| close_vs_ema72__d1      | close_vs_ema72   |        108 |       -0.0072 |        -0.0040 |            -0.0032 |              -0.0023 |         -0.7568 |              0.7568 | WINNER_LOW  |             0.7685 |    0.0000 | True         |  0.0000 |
| close_vs_sma240__d1     | close_vs_sma240  |        108 |       -0.0077 |        -0.0044 |            -0.0033 |              -0.0023 |         -0.7562 |              0.7562 | WINNER_LOW  |             0.7407 |    0.0000 | True         |  0.0000 |
| dist_high72__d1         | dist_high72      |        108 |       -0.0072 |        -0.0040 |            -0.0032 |              -0.0018 |         -0.7519 |              0.7519 | WINNER_LOW  |             0.7685 |    0.0000 | True         |  0.0000 |
| close_vs_ema48__d1      | close_vs_ema48   |        108 |       -0.0069 |        -0.0038 |            -0.0032 |              -0.0023 |         -0.7504 |              0.7504 | WINNER_LOW  |             0.7685 |    0.0000 | True         |  0.0000 |
| close_vs_sma168__d1     | close_vs_sma168  |        108 |       -0.0076 |        -0.0043 |            -0.0033 |              -0.0024 |         -0.7500 |              0.7500 | WINNER_LOW  |             0.7685 |    0.0000 | True         |  0.0000 |
| close_vs_sma24__d1      | close_vs_sma24   |        108 |       -0.0068 |        -0.0037 |            -0.0031 |              -0.0022 |         -0.7402 |              0.7402 | WINNER_LOW  |             0.7870 |    0.0000 | True         |  0.0000 |
| abs_body_pct__w24__pos  | abs_body_pct     |        108 |        0.5517 |         0.3941 |             0.1576 |               0.1175 |          0.7329 |              0.7329 | WINNER_HIGH |             0.7685 |    0.0000 | True         |  0.0000 |
| close_vs_ema24__d1      | close_vs_ema24   |        108 |       -0.0063 |        -0.0034 |            -0.0029 |              -0.0019 |         -0.7238 |              0.7238 | WINNER_LOW  |             0.7685 |    0.0000 | True         |  0.0000 |
| macd_hist_pct__d1       | macd_hist_pct    |        108 |       -0.0007 |        -0.0003 |            -0.0003 |              -0.0002 |         -0.7204 |              0.7204 | WINNER_LOW  |             0.8333 |    0.0000 | True         |  0.0000 |
| body_pct__w24__pos      | body_pct         |        108 |        0.2580 |         0.3668 |            -0.1088 |              -0.1111 |         -0.6955 |              0.6955 | WINNER_LOW  |             0.7500 |    0.0000 | True         |  0.0000 |
| log_ret__w24__pos       | log_ret          |        108 |        0.2595 |         0.3688 |            -0.1092 |              -0.1116 |         -0.6947 |              0.6947 | WINNER_LOW  |             0.7500 |    0.0000 | True         |  0.0000 |
| close_vs_ema6__w24__pos | close_vs_ema6    |        108 |        0.1595 |         0.2570 |            -0.0974 |              -0.1069 |         -0.6786 |              0.6786 | WINNER_LOW  |             0.7870 |    0.0000 | True         |  0.0000 |

## Strongest raw sequence-shape differences

|   window_h |   offset_h | channel        |   winner_mean_z |   control_mean_z |   paired_effect |   shape_paired_effect |   abs_shape_effect |
|-----------:|-----------:|:---------------|----------------:|-----------------:|----------------:|----------------------:|-------------------:|
|         24 |         -1 | macd_hist_pct  |         -0.8798 |          -0.6825 |         -0.2344 |                0.7204 |             0.7204 |
|         24 |         -1 | close_vs_ema24 |         -1.2373 |          -0.8921 |         -0.3488 |                0.6608 |             0.6608 |
|         24 |        -13 | log_ret        |          0.1470 |          -0.1026 |          0.1463 |                0.6195 |             0.6195 |
|         24 |        -13 | body_pct       |          0.1509 |          -0.1001 |          0.1470 |                0.6176 |             0.6176 |
|        240 |       -200 | range_pct      |          0.3364 |           0.4521 |         -0.1182 |               -0.5554 |             0.5554 |
|         24 |         -3 | volume_rel72   |          0.7042 |           0.8751 |         -0.1051 |               -0.4941 |             0.4941 |
|         24 |         -1 | atr14_pct      |          0.5231 |          -0.0035 |          0.5868 |               -0.4445 |             0.4445 |
|         24 |         -7 | trades_rel72   |          0.3553 |           0.5773 |         -0.1645 |               -0.4363 |             0.4363 |
|         24 |         -1 | rsi14          |         -0.8360 |          -0.7753 |         -0.1158 |                0.4260 |             0.4260 |
|        240 |       -180 | stoch_k        |         -0.1774 |           0.0567 |         -0.3768 |               -0.3953 |             0.3953 |
|         24 |         -1 | bb_width       |          0.3181 |           0.0626 |          0.2852 |               -0.3745 |             0.3745 |
|         72 |        -15 | close_location |          0.0697 |          -0.0067 |          0.1899 |                0.3451 |             0.3451 |
|         24 |        -15 | lower_wick     |          0.4017 |           0.2940 |          0.1084 |                0.2929 |             0.2929 |
|         24 |         -9 | taker_share    |          0.1129 |          -0.0781 |          0.1738 |                0.2509 |             0.2509 |
|         24 |         -9 | upper_wick     |          0.3220 |           0.1393 |          0.1631 |                0.1914 |             0.1914 |

## Can trajectory pick the winner from five state-lookalikes?

| model              |   rows |   events |   class_precision_baseline |    auc |   average_precision |   winner_top1_rate |   winner_top2_rate |   winner_top3_rate |   median_winner_rank |   random_top1 |   random_top2 |
|:-------------------|-------:|---------:|---------------------------:|-------:|--------------------:|-------------------:|-------------------:|-------------------:|---------------------:|--------------:|--------------:|
| TEMPORAL_CHANGE    |    648 |      108 |                     0.1667 | 0.6156 |              0.2975 |             0.3333 |             0.5833 |             0.7130 |               2.0000 |        0.1667 |        0.3333 |
| RAW_SEQUENCE_SHAPE |    648 |      108 |                     0.1667 | 0.6931 |              0.3831 |             0.4167 |             0.6111 |             0.7130 |               2.0000 |        0.1667 |        0.3333 |

Random reference: one winner among six candidates gives 16.7% top-1 and 33.3% top-2.

These CV diagnostics are discovery-only within 2024; they are not a trading backtest and are not evidence of 2025/2026 performance.