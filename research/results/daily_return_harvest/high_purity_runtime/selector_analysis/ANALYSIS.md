# High-Purity Same-Time Selector Analysis

Research-only. No live strategy changes.

Selector development used only the unused discovery tail from 2023-11-01. Validation and 2026 confirmation were not used to fit or tune the selector.

## Best result

- Candidate threshold delta: 0.0
- Validation precision: 0.9306; signals/day: 0.875; winners/day: 0.814
- 2026 confirmation precision: 0.9258; signals/day: 0.942; winners/day: 0.872
- Validation W5-2 rate: 0.5741; 2026 W5-2 rate: 0.5502
- Selector features: xrank_ret_6h | xrank_rs_6h | median_ret_24h | RANK__median_ret_6h__chg_168h | range_6h__chg_168h | trade_ratio_current | range_12h__chg_168h | RANK__range24_vs_72h__chg_24h | RANK__ret_1h__chg_72h | RANK__ret_1h__chg_3h | RANK__close_vs_ema6__chg_72h | RANK__dist_72h_high__chg_72h | RANK__dist_72h_high__chg_12h | RANK__rs_1h__chg_3h | RANK__taker_buy_current__chg_72h | RANK__rs_3h__chg_72h | RANK__close_vs_vwap72__chg_12h | RANK__close_vs_vwap24__chg_72h | RANK__ema24_slope_6h__chg_12h | RANK__close_vs_ema12__chg_72h | RANK__lower_wick_share__chg_3h | RANK__close_location__chg_3h | RANK__green_body_pct__chg_72h | RANK__green_body_pct__chg_3h | RANK__taker_buy_24h__chg_72h | volume_ratio_current | trade_ratio_current__chg_72h | volume_ratio_3h__chg_1h

See selector_results.csv for the full precision/frequency frontier and candidate-pool comparisons.
