# Full-Stream W3 Direct Classifier

**RESEARCH ONLY. No live strategy or automation changes.**

## Training population

| target        |    rows |   positives |   negatives |   negative_to_positive |
|:--------------|--------:|------------:|------------:|-----------------------:|
| W3_ALL        | 1086801 |      285636 |      801165 |                 2.8048 |
| EARLY3_ALLNEG |  839805 |       38640 |      801165 |                20.7341 |

## Results

| target        | split        |   threshold |   signals |   wins |   precision |   signals_per_day |   wins_per_day |
|:--------------|:-------------|------------:|----------:|-------:|------------:|------------------:|---------------:|
| W3_ALL        | DEV          |      0.4960 |      1812 |    793 |      0.4376 |            9.9560 |         4.3571 |
| W3_ALL        | VALIDATION   |      0.4960 |      6203 |   2345 |      0.3780 |           10.1689 |         3.8443 |
| W3_ALL        | CONFIRMATION |      0.4960 |      1509 |    373 |      0.2472 |            5.5889 |         1.3815 |
| EARLY3_ALLNEG | DEV          |      0.5508 |       128 |     69 |      0.5391 |            0.7033 |         0.3791 |
| EARLY3_ALLNEG | VALIDATION   |      0.5508 |      1323 |    439 |      0.3318 |            2.1689 |         0.7197 |
| EARLY3_ALLNEG | CONFIRMATION |      0.5508 |       218 |     74 |      0.3394 |            0.8074 |         0.2741 |

## Baseline

| split        |   eligible_hours |   raw_w3_rate |   symbols |
|:-------------|-----------------:|--------------:|----------:|
| CONFIRMATION |           344326 |        0.2388 |        86 |
| DEV          |           356297 |        0.2879 |        86 |
| VALIDATION   |          1143174 |        0.2785 |        86 |

## Choices

```json
{
  "W3_ALL": {
    "threshold": 0.49597275034920474,
    "reason": "best precision >=0.5/day"
  },
  "EARLY3_ALLNEG": {
    "threshold": 0.5508433001516617,
    "reason": "best precision >=0.5/day"
  }
}
```

## Top markers

### W3_ALL

| target   | feature                        |         gain |   gain_share |
|:---------|:-------------------------------|-------------:|-------------:|
| W3_ALL   | atr24_pct                      | 331257.71964 |      0.07678 |
| W3_ALL   | btc_ret_168h                   | 196566.65434 |      0.04556 |
| W3_ALL   | atr6_pct                       | 176577.39137 |      0.04093 |
| W3_ALL   | btc_atr24_pct                  | 166321.14085 |      0.03855 |
| W3_ALL   | btc_ret_168h__chg_168h         | 152355.08541 |      0.03531 |
| W3_ALL   | btc_atr24_pct__chg_72h         | 140408.15392 |      0.03254 |
| W3_ALL   | btc_atr24_pct__chg_168h        | 133385.28865 |      0.03091 |
| W3_ALL   | btc_vs_ema168                  | 123959.46631 |      0.02873 |
| W3_ALL   | btc_ret_72h                    | 110743.99333 |      0.02567 |
| W3_ALL   | atr168_pct                     | 107067.62301 |      0.02481 |
| W3_ALL   | btc_ret_72h__chg_168h          | 105619.05711 |      0.02448 |
| W3_ALL   | btc_vs_ema168__chg_72h         |  95830.25600 |      0.02221 |
| W3_ALL   | btc_ret_72h__chg_72h           |  87570.20485 |      0.02030 |
| W3_ALL   | btc_atr24_pct__chg_24h         |  86537.63657 |      0.02006 |
| W3_ALL   | btc_vs_ema168__chg_168h        |  79575.93552 |      0.01844 |
| W3_ALL   | btc_ret_168h__chg_72h          |  78294.52842 |      0.01815 |
| W3_ALL   | btc_atr24_pct__chg_12h         |  69255.38944 |      0.01605 |
| W3_ALL   | breadth_positive_24h__chg_72h  |  67058.75864 |      0.01554 |
| W3_ALL   | breadth_positive_24h__chg_168h |  64690.69567 |      0.01499 |
| W3_ALL   | median_ret_24h__chg_168h       |  64368.68981 |      0.01492 |
| W3_ALL   | btc_atr24_pct__chg_6h          |  59378.09208 |      0.01376 |
| W3_ALL   | btc_vs_ema168__chg_24h         |  58387.29922 |      0.01353 |
| W3_ALL   | btc_ret_24h__chg_168h          |  55863.19298 |      0.01295 |
| W3_ALL   | median_ret_24h__chg_24h        |  52193.96076 |      0.01210 |
| W3_ALL   | btc_ret_24h                    |  50934.72696 |      0.01181 |
| W3_ALL   | median_ret_24h                 |  50613.07424 |      0.01173 |
| W3_ALL   | breadth_positive_24h__chg_24h  |  49554.85793 |      0.01149 |
| W3_ALL   | btc_ret_24h__chg_72h           |  49213.51797 |      0.01141 |
| W3_ALL   | btc_ret_24h__chg_24h           |  49111.22704 |      0.01138 |
| W3_ALL   | btc_ret_168h__chg_24h          |  47675.84959 |      0.01105 |

### EARLY3_ALLNEG

| target        | feature                   |         gain |   gain_share |
|:--------------|:--------------------------|-------------:|-------------:|
| EARLY3_ALLNEG | ema6_slope_3h             | 544372.46146 |      0.22785 |
| EARLY3_ALLNEG | close_vs_ema12            | 130861.79308 |      0.05477 |
| EARLY3_ALLNEG | ret_12h                   |  78844.55444 |      0.03300 |
| EARLY3_ALLNEG | range_24h__chg_3h         |  74061.56130 |      0.03100 |
| EARLY3_ALLNEG | close_vs_vwap24           |  72828.72805 |      0.03048 |
| EARLY3_ALLNEG | close_vs_ema24            |  42531.51197 |      0.01780 |
| EARLY3_ALLNEG | btc_atr24_pct             |  39737.27963 |      0.01663 |
| EARLY3_ALLNEG | btc_ret_168h__chg_168h    |  34927.12414 |      0.01462 |
| EARLY3_ALLNEG | quote_volume_24h          |  34614.27779 |      0.01449 |
| EARLY3_ALLNEG | ema24_slope_6h            |  34249.55417 |      0.01434 |
| EARLY3_ALLNEG | quote_volume_7d           |  32450.52865 |      0.01358 |
| EARLY3_ALLNEG | btc_ret_168h              |  31679.80989 |      0.01326 |
| EARLY3_ALLNEG | btc_atr24_pct__chg_168h   |  31529.81622 |      0.01320 |
| EARLY3_ALLNEG | range_3h__chg_6h          |  28584.62966 |      0.01196 |
| EARLY3_ALLNEG | rebound_12h_low__chg_3h   |  26592.01893 |      0.01113 |
| EARLY3_ALLNEG | btc_vs_ema168             |  26229.14206 |      0.01098 |
| EARLY3_ALLNEG | btc_atr24_pct__chg_12h    |  26212.41458 |      0.01097 |
| EARLY3_ALLNEG | btc_atr24_pct__chg_72h    |  25686.78083 |      0.01075 |
| EARLY3_ALLNEG | median_ret_24h__chg_24h   |  21528.49584 |      0.00901 |
| EARLY3_ALLNEG | btc_atr24_pct__chg_24h    |  21411.16129 |      0.00896 |
| EARLY3_ALLNEG | ret_6h                    |  20809.58791 |      0.00871 |
| EARLY3_ALLNEG | volume_ratio_12h__chg_24h |  20798.36458 |      0.00871 |
| EARLY3_ALLNEG | volume_ratio_6h__chg_24h  |  18590.13573 |      0.00778 |
| EARLY3_ALLNEG | median_ret_1h__chg_3h     |  18496.51891 |      0.00774 |
| EARLY3_ALLNEG | rebound_12h_low__chg_6h   |  18151.31340 |      0.00760 |
| EARLY3_ALLNEG | btc_ret_72h               |  17997.70652 |      0.00753 |
| EARLY3_ALLNEG | btc_ret_168h__chg_72h     |  17957.55031 |      0.00752 |
| EARLY3_ALLNEG | rebound_6h_low__chg_3h    |  17798.14871 |      0.00745 |
| EARLY3_ALLNEG | atr168_pct                |  17638.90547 |      0.00738 |
| EARLY3_ALLNEG | btc_ret_72h__chg_72h      |  17619.09238 |      0.00737 |