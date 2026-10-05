# R01 Trend Flush Reversal — Round 001 Analysis

Research only. No live strategy changes.

## Primary result

- Accepted against the stated primary threshold: **NO**
- Locked-confirmation signals: **10** (0.26/week)
- +3% before -2.0% precision: **40.00%**
- Wilson 95% CI: **16.82% to 68.73%**
- Mean net return/signal after assumed fees/slippage: **-0.097%**
- Median MAE / MFE: **-1.99% / 0.74%**

The confirmation period was not used to choose thresholds. The reported best row is the strongest member of the shortlist frozen from Development/Validation.

## Best locked rule

`entry=confirm_60, wick_share>=0.35, wick/body>=2.0, range/ATR>=2.0, close_location>=0.55, volume_ratio>=1.5, trend_score>=3, support=none, BTC=none, stop=0.02`

The 90% target was not met under the minimum frequency/sample constraints, so Round 001 does **not** establish a 90%-accurate live signal.

## Dataset

- Event-entry rows analysed: 26,257
- Symbols represented: 40
- Development / Validation / Confirmation rows: 10,598 / 12,015 / 3,644

## Secondary exit diagnostic

|   target_pct |   stop_pct |   horizon_hours |   signals |   wins |   precision |   gross_expectancy |   net_expectancy |
|-------------:|-----------:|----------------:|----------:|-------:|------------:|-------------------:|-----------------:|
|       0.0300 |     0.0200 |         24.0000 |   10.0000 | 4.0000 |      0.4000 |             0.0020 |          -0.0010 |
|       0.0400 |     0.0200 |         48.0000 |   10.0000 | 4.0000 |      0.4000 |             0.0040 |           0.0010 |
|       0.0500 |     0.0200 |         48.0000 |   10.0000 | 4.0000 |      0.4000 |             0.0080 |           0.0050 |
|       0.0600 |     0.0200 |         48.0000 |   10.0000 | 4.0000 |      0.4000 |             0.0120 |           0.0090 |

## Files

See `shortlist.csv`, `confirmation_results.csv`, `year_summary.csv`, `regime_summary.csv`, `symbol_summary.csv`, `rule_grid.csv` and `events.csv.gz` for the audit trail.
