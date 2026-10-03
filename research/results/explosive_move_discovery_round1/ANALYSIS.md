# Explosive Move Discovery Round 1

**Status:** RESEARCH ONLY. Separate from Early Breakout and live strategy.

## Event definition

A clean event starts at an hourly close from which price reaches +20% before ever touching -5%, within a maximum 30-day observation window. Overlapping starts are de-duplicated into one event. The 30-day cap prevents indefinite bull-market drift from becoming an event while allowing much slower moves than the previous 24-hour rule.

Universe is the current top-100 Binance USDT spot pairs by quote volume at research-run time, with current top-50 reported separately. This first pass therefore has current-universe survivorship bias; it is not a historically reconstructed top-100.

## Event population

| Universe | Events | Median to +20 | Median +5 to +20 | <=1d | <=3d | <=7d |
|---|---:|---:|---:|---:|---:|---:|
| TOP100 | 8683 | 87h | 63h | 17.9% | 44.6% | 70.2% |
| TOP50 | 4968 | 87h | 64h | 17.9% | 44.7% | 69.8% |

## Preliminary stable precursor features

Features below separate winners from same-symbol/year HARD FAILURES in the same direction across discovery, validation and holdout. A hard failure reaches +5% before -5% but fails to reach +20% before the -5% stop. AUC is univariate separation only, not a deployable signal.

| Lead | Feature | Discovery AUC | Validation AUC | Holdout AUC |
|---:|---|---:|---:|---:|
| 0h | ret_6h | 0.815 | 0.818 | 0.837 |
| 0h | range_24h | 0.746 | 0.776 | 0.751 |
| 0h | ret_24h | 0.744 | 0.763 | 0.784 |
| 0h | range_72h | 0.709 | 0.752 | 0.715 |
| 0h | atr24_pct | 0.704 | 0.748 | 0.708 |
| 0h | volume_ratio_6h | 0.676 | 0.677 | 0.706 |
| 0h | trade_ratio_6h | 0.675 | 0.678 | 0.701 |
| 0h | rs_24h | 0.667 | 0.720 | 0.737 |
| 0h | range_7d | 0.662 | 0.700 | 0.681 |
| 0h | dist_7d_high | 0.659 | 0.677 | 0.688 |
| 0h | atr24_vs_7d | 0.654 | 0.699 | 0.688 |
| 0h | volume_ratio_24h | 0.639 | 0.641 | 0.677 |
| 0h | range_14d | 0.628 | 0.660 | 0.624 |
| 0h | range24_vs_7d | 0.619 | 0.631 | 0.627 |
| 0h | worst_1h_ret_7d | 0.614 | 0.656 | 0.622 |
| 0h | atr7d_pct | 0.628 | 0.656 | 0.613 |
| 0h | ret_72h | 0.593 | 0.585 | 0.641 |
| 0h | taker_buy_6h | 0.590 | 0.601 | 0.585 |
| 0h | range_30d | 0.582 | 0.618 | 0.575 |
| 0h | dist_30d_high | 0.572 | 0.577 | 0.586 |
| 0h | worst_4h_ret_14d | 0.592 | 0.630 | 0.560 |
| 0h | atr30d_pct | 0.564 | 0.558 | 0.553 |
| 6h | range_24h | 0.661 | 0.685 | 0.662 |
| 6h | atr24_pct | 0.660 | 0.696 | 0.647 |
| 6h | range_72h | 0.655 | 0.687 | 0.634 |

## Outputs

- events.csv: de-duplicated clean +20%/-5% events and milestone timing.
- event_summary.csv: event counts and speed for top-50/top-100.
- precursor_samples.csv.gz: compressed winner/hard-failure/random-control snapshots from 30d before the move through the start.
- feature_contrasts.csv: winner vs control distributions by lead and time split.
- stable_features.csv: winner vs hard-failure cross-period feature stability.
- event_archetypes.csv and archetype_summary.csv: unsupervised event-shape clusters to test flat-base vs capitulation-style precursors.
- universe.csv: current top-100 universe used.
- manifest.json: exact methodology.

## Next step

Use the discovered event set to define a WATCH model days before the move and a separate ENTRY trigger constrained to fire before the event has advanced +5%. Do not optimise exits until the event detector shows useful out-of-sample precision.
