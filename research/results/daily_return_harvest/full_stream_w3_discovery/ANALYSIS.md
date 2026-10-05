# Full-Stream W3 Signal Discovery

**RESEARCH ONLY. No live strategy or automation changes.**

## Training population

| target   |   rows |   positives |   negatives |   negative_to_positive |
|:---------|-------:|------------:|------------:|-----------------------:|
| ANCHOR   | 387273 |       14118 |      373155 |               26.4312  |
| EARLY3   | 411795 |       38640 |      373155 |                9.65722 |
| W3       | 412457 |       39302 |      373155 |                9.49455 |

## Full-stream results

| target   | split        |   threshold |   signals |   wins |   precision |   signals_per_day |   wins_per_day |
|:---------|:-------------|------------:|----------:|-------:|------------:|------------------:|---------------:|
| ANCHOR   | DEV          |      0.6410 |       118 |     47 |      0.3983 |            0.6484 |         0.2582 |
| ANCHOR   | VALIDATION   |      0.6410 |       855 |    294 |      0.3439 |            1.4016 |         0.4820 |
| ANCHOR   | CONFIRMATION |      0.6410 |       113 |     46 |      0.4071 |            0.4185 |         0.1704 |
| EARLY3   | DEV          |      0.6559 |       197 |     88 |      0.4467 |            1.0824 |         0.4835 |
| EARLY3   | VALIDATION   |      0.6559 |      1226 |    432 |      0.3524 |            2.0098 |         0.7082 |
| EARLY3   | CONFIRMATION |      0.6559 |       162 |     73 |      0.4506 |            0.6000 |         0.2704 |
| W3       | DEV          |      0.4707 |       981 |    353 |      0.3598 |            5.3901 |         1.9396 |
| W3       | VALIDATION   |      0.4707 |      4868 |   1529 |      0.3141 |            7.9803 |         2.5066 |
| W3       | CONFIRMATION |      0.4707 |       910 |    264 |      0.2901 |            3.3704 |         0.9778 |

## Raw W3 baseline

| split        |   eligible_hours |   raw_w3_rate |   symbols |
|:-------------|-----------------:|--------------:|----------:|
| CONFIRMATION |           344326 |        0.2388 |        86 |
| DEV          |           356297 |        0.2879 |        86 |
| VALIDATION   |          1143174 |        0.2785 |        86 |

## Threshold selection

```json
{
  "ANCHOR": {
    "threshold": 0.6410041791699217,
    "reason": "best precision with >=0.5/day"
  },
  "EARLY3": {
    "threshold": 0.6559379188104113,
    "reason": "best precision with >=0.5/day"
  },
  "W3": {
    "threshold": 0.4706804931616497,
    "reason": "best precision with >=0.5/day"
  }
}
```

## Top markers

### ANCHOR

| target   | feature                 |         gain |   gain_share |
|:---------|:------------------------|-------------:|-------------:|
| ANCHOR   | close_vs_ema6           | 284231.94113 |      0.28556 |
| ANCHOR   | close_vs_ema12          |  70368.45801 |      0.07070 |
| ANCHOR   | range_24h__chg_1h       |  46678.68254 |      0.04690 |
| ANCHOR   | green_body_pct          |  31690.01020 |      0.03184 |
| ANCHOR   | ret_2h                  |  28597.35811 |      0.02873 |
| ANCHOR   | ret_12h                 |  20641.06253 |      0.02074 |
| ANCHOR   | quote_volume_24h        |  16036.33158 |      0.01611 |
| ANCHOR   | atr168_pct__chg_1h      |  11800.20266 |      0.01186 |
| ANCHOR   | range_12h__chg_1h       |  11504.62273 |      0.01156 |
| ANCHOR   | range_6h__chg_1h        |   8579.84687 |      0.00862 |
| ANCHOR   | rebound_12h_low         |   6980.30773 |      0.00701 |
| ANCHOR   | atr6_pct__chg_1h        |   6865.71471 |      0.00690 |
| ANCHOR   | range_12h               |   6708.14995 |      0.00674 |
| ANCHOR   | rebound_24h_low         |   6471.06755 |      0.00650 |
| ANCHOR   | close_vs_vwap24         |   6377.53893 |      0.00641 |
| ANCHOR   | ret_3h                  |   5009.03779 |      0.00503 |
| ANCHOR   | btc_ret_168h__chg_168h  |   4926.80140 |      0.00495 |
| ANCHOR   | ema24_slope_6h          |   4751.84722 |      0.00477 |
| ANCHOR   | btc_atr24_pct__chg_168h |   4707.48640 |      0.00473 |
| ANCHOR   | quote_volume_7d         |   4676.17485 |      0.00470 |
| ANCHOR   | btc_atr24_pct           |   4666.10044 |      0.00469 |
| ANCHOR   | rebound_12h_low__chg_6h |   4405.77553 |      0.00443 |
| ANCHOR   | range_24h__chg_3h       |   4301.32311 |      0.00432 |
| ANCHOR   | btc_atr24_pct__chg_72h  |   4150.16955 |      0.00417 |
| ANCHOR   | btc_vs_ema168           |   4035.69592 |      0.00405 |

### EARLY3

| target   | feature                 |         gain |   gain_share |
|:---------|:------------------------|-------------:|-------------:|
| EARLY3   | ema6_slope_3h           | 363558.21288 |      0.20748 |
| EARLY3   | close_vs_ema12          | 120297.02350 |      0.06865 |
| EARLY3   | ret_12h                 |  81069.71244 |      0.04627 |
| EARLY3   | close_vs_ema24          |  56556.10232 |      0.03228 |
| EARLY3   | range_24h__chg_3h       |  40191.90440 |      0.02294 |
| EARLY3   | ema24_slope_6h          |  30626.37730 |      0.01748 |
| EARLY3   | btc_atr24_pct           |  26951.34898 |      0.01538 |
| EARLY3   | close_vs_vwap24         |  26851.00493 |      0.01532 |
| EARLY3   | quote_volume_24h        |  23655.08856 |      0.01350 |
| EARLY3   | btc_ret_168h__chg_168h  |  21208.57962 |      0.01210 |
| EARLY3   | btc_atr24_pct__chg_168h |  18385.45953 |      0.01049 |
| EARLY3   | btc_ret_168h            |  18134.59947 |      0.01035 |
| EARLY3   | btc_atr24_pct__chg_72h  |  16342.74021 |      0.00933 |
| EARLY3   | quote_volume_7d         |  16214.12959 |      0.00925 |
| EARLY3   | range_3h__chg_6h        |  16149.65396 |      0.00922 |
| EARLY3   | median_ret_24h__chg_24h |  15867.72379 |      0.00906 |
| EARLY3   | btc_vs_ema168           |  15686.06774 |      0.00895 |
| EARLY3   | ret_6h                  |  14908.38245 |      0.00851 |
| EARLY3   | rebound_12h_low__chg_6h |  13472.73562 |      0.00769 |
| EARLY3   | btc_atr24_pct__chg_12h  |  13312.19650 |      0.00760 |
| EARLY3   | rebound_12h_low__chg_3h |  12387.38073 |      0.00707 |
| EARLY3   | btc_atr24_pct__chg_24h  |  12263.04307 |      0.00700 |
| EARLY3   | range_6h__chg_6h        |  11326.89785 |      0.00646 |
| EARLY3   | btc_vs_ema168__chg_24h  |  10973.69883 |      0.00626 |
| EARLY3   | btc_vs_ema168__chg_72h  |  10814.41191 |      0.00617 |

### W3

| target   | feature                        |         gain |   gain_share |
|:---------|:-------------------------------|-------------:|-------------:|
| W3       | close_vs_ema6                  | 100344.63509 |      0.11461 |
| W3       | close_vs_ema12                 |  32122.95912 |      0.03669 |
| W3       | range_24h__chg_1h              |  23211.05064 |      0.02651 |
| W3       | btc_ret_168h__chg_168h         |  14855.95211 |      0.01697 |
| W3       | btc_atr24_pct__chg_72h         |  12648.00856 |      0.01445 |
| W3       | btc_atr24_pct                  |  12638.51188 |      0.01444 |
| W3       | btc_atr24_pct__chg_12h         |  11864.49851 |      0.01355 |
| W3       | btc_atr24_pct__chg_168h        |  11827.80512 |      0.01351 |
| W3       | btc_ret_72h__chg_168h          |  11720.13223 |      0.01339 |
| W3       | btc_ret_168h                   |  11703.91435 |      0.01337 |
| W3       | btc_vs_ema168                  |  11537.00278 |      0.01318 |
| W3       | btc_ret_24h                    |  10428.96735 |      0.01191 |
| W3       | btc_ret_72h                    |   8994.70227 |      0.01027 |
| W3       | btc_atr24_pct__chg_24h         |   8867.89618 |      0.01013 |
| W3       | median_ret_24h                 |   8590.80488 |      0.00981 |
| W3       | median_ret_24h__chg_24h        |   8577.53483 |      0.00980 |
| W3       | btc_atr24_pct__chg_1h          |   8403.16853 |      0.00960 |
| W3       | breadth_positive_24h__chg_168h |   8302.50878 |      0.00948 |
| W3       | btc_ret_72h__chg_72h           |   8196.78802 |      0.00936 |
| W3       | btc_ret_168h__chg_72h          |   7997.76620 |      0.00914 |
| W3       | median_ret_24h__chg_168h       |   7705.14023 |      0.00880 |
| W3       | btc_ret_24h__chg_24h           |   7628.02818 |      0.00871 |
| W3       | atr168_pct__chg_1h             |   7623.20885 |      0.00871 |
| W3       | btc_vs_ema168__chg_168h        |   7459.33270 |      0.00852 |
| W3       | ret_2h                         |   6974.32078 |      0.00797 |