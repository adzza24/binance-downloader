# BTC Dynamic Reversal State Engine: 2024-2025

**RESEARCH ONLY. Exploratory/in-sample: 2024 and 2025 were already used to discover the fingerprint. No live strategy or automation changes. 2026 was not read.**

## Engine

Every condition is recomputed independently each hour. If a condition turns false it stops contributing immediately. Signals occur only on a rising edge into TRIGGERED.

Primary variant: **BALANCED** with setup >= 4, transition >= 2, fast >= 2, total >= 9.

## Variant sensitivity

| variant   |   signals |   signals_per_day |   w3_wins |   w3_precision |   eventual_target_rate |   eventual_stop_rate |   median_holding_hours |
|:----------|----------:|------------------:|----------:|---------------:|-----------------------:|---------------------:|-----------------------:|
| LOOSE     |      1046 |            1.4369 |       177 |         0.1692 |                 0.3805 |               0.6185 |                19.0000 |
| BALANCED  |       946 |            1.2995 |       153 |         0.1617 |                 0.3837 |               0.6152 |                18.0000 |
| STRICT    |       812 |            1.1154 |       128 |         0.1576 |                 0.3645 |               0.6355 |                17.0000 |

## $1,000 continuous portfolio replay

- Starting capital: **$1,000.00**
- Final balance: **$316.51**
- Total return: **-68.35%**
- Max realised-equity drawdown: **-72.88%**
- Signals: **946**; taken: **387**; missed while already positioned: **559**
- All-signal W3 precision: **16.17%**
- Taken-signal W3 precision: **15.76%**
- Missed-signal W3 precision: **16.46%**
- Eventual target rate on taken trades: **34.63%**
- Eventual stop rate on taken trades: **65.37%**
- Median holding time: **14.0h**

## Calendar-year breakdown

|      year |   signals |   w3_wins |   w3_precision |   taken_trades |   missed_signals |   missed_w3_precision |   taken_mean_return |   taken_target_rate |   taken_stop_rate |   median_holding_hours |
|----------:|----------:|----------:|---------------:|---------------:|-----------------:|----------------------:|--------------------:|--------------------:|------------------:|-----------------------:|
| 2024.0000 |  474.0000 |   90.0000 |         0.1899 |       211.0000 |         263.0000 |                0.1977 |             -0.0022 |              0.3555 |            0.6445 |                13.0000 |
| 2025.0000 |  472.0000 |   63.0000 |         0.1335 |       176.0000 |         296.0000 |                0.1351 |             -0.0032 |              0.3352 |            0.6648 |                15.0000 |

Portfolio rule: the whole available balance is committed to each taken trade. A new signal while still positioned is not traded, but its standalone OCO/W3 outcome is retained in `signals_balanced.csv`. Same-candle target/stop ambiguity is conservatively treated as a stop.