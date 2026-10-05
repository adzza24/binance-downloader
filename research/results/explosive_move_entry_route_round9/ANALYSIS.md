# Explosive Move Entry Route Round 9

**RESEARCH ONLY. No live strategy or automation changes.**

## Design

- Same Round 8 full-population signals. Primary horizon is 30 days from the original signal for both routes.
- Route A: buy at signal close, OCO +10% / -5%.
- Route B: place buy limit 5% below signal; if filled, OCO +10% / -5% from fill.
- 0.30% round-trip costs on executed trades. Unfilled limits have zero return.
- Conservative 1h OHLC ordering: stop wins same-candle ties. A target on the limit-fill candle is not credited when the candle opened above the limit, because the high could precede the fill.
- Signal-time features are stored for DIRECT_PLUS10 vs DIP_THEN_PLUS10 analysis; no future features are used as predictors.

## Overall mature-signal result

- Mature signals: **463**.
- Straight OCO target rate: **35.6%**; average net per signal **0.05%**.
- -5% limit fill rate: **90.9%**.
- After a fill, limit-route OCO target rate: **28.5%**; average net per filled trade **-0.93%**.
- Limit-route average net per original signal including unfilled orders: **-0.84%**.
- Median wait for limit fill: **15.0h**.
- Path groups: DIRECT_PLUS10 **165**, DIP_THEN_PLUS10 **198**, NO_PLUS10_30D **100**.

## Commonality dataset

- Numeric signal-time comparison features saved: **43**.
- `direct_vs_dip_features.csv` ranks contemporaneous differences and checks whether direction is consistent across available years.
- `signal_strategy_paths.csv.gz` retains per-signal outcomes, fill timing, path class, candidate context, detector features and market signal breadth for later split-rule research.
- These comparisons are descriptive; 2024-2026 has already influenced earlier research and is not a pristine untouched holdout.
