# Early Breakout Round 2 - Exit Architecture Research

**Status:** RESEARCH ONLY. Separate breakout-strategy track. No Early Breakout Watch or live-strategy edits.

**Cohort:** 3,939 de-duplicated Round 1 entry episodes regenerated from the frozen Round 1 implementation.

## Families

- TIME_ONLY: 1-7d and 30d forced caps with 3/5/7/10% or structural-capped stop.
- FIXED_TP: same caps with full +10% or +20% profit-taking.
- FULL_TRAIL: +5% -> breakeven; +30% -> +10% floor plus 30/40/50/60% accumulated-gain giveback; 7/30/90/180/365d caps.
- HALF_TP_RUNNER: take 50% at +10% or +20%, runner to breakeven, then the same +30% trail; 7/30/90/180/365d caps.

All results use independent 300 USDT trades with configured fees/slippage. Same-hour stop/target ambiguity is resolved conservatively in favour of the stop.

## Historical screening leaders

These are candidates for a narrower confirmation round, not promoted strategy rules. Prefer plateaus across neighbouring parameters over a single isolated maximum.

### FIXED_TP

| Variant | Trades | PF | Mean P&L | Win rate | Positive years | Worst year |
|---|---:|---:|---:|---:|---:|---:|
| `FIXEDTP_FIXED3_TP20_CAP1D` | 3928 | 1.214 | 1.73 | 16.6% | 6/8 | -241.63 |
| `FIXEDTP_FIXED3_TP20_CAP2D` | 3928 | 1.214 | 1.73 | 16.6% | 6/8 | -241.63 |
| `FIXEDTP_FIXED3_TP20_CAP3D` | 3928 | 1.214 | 1.73 | 16.6% | 6/8 | -241.63 |
| `FIXEDTP_FIXED3_TP20_CAP4D` | 3928 | 1.214 | 1.73 | 16.6% | 6/8 | -241.63 |
| `FIXEDTP_FIXED3_TP20_CAP5D` | 3928 | 1.214 | 1.73 | 16.6% | 6/8 | -241.63 |

### FULL_TRAIL

| Variant | Trades | PF | Mean P&L | Win rate | Positive years | Worst year |
|---|---:|---:|---:|---:|---:|---:|
| `FULL_TRAIL_FIXED5_GB60_CAP7D` | 3898 | 1.790 | 6.24 | 11.6% | 5/8 | -2597.86 |
| `FULL_TRAIL_FIXED5_GB60_CAP30D` | 3898 | 1.790 | 6.24 | 11.6% | 5/8 | -2597.86 |
| `FULL_TRAIL_FIXED5_GB60_CAP90D` | 3898 | 1.790 | 6.24 | 11.6% | 5/8 | -2597.86 |
| `FULL_TRAIL_FIXED5_GB60_CAP180D` | 3898 | 1.790 | 6.24 | 11.6% | 5/8 | -2597.86 |
| `FULL_TRAIL_FIXED5_GB60_CAP365D` | 3898 | 1.790 | 6.24 | 11.6% | 5/8 | -2597.86 |

### HALF_TP_RUNNER

| Variant | Trades | PF | Mean P&L | Win rate | Positive years | Worst year |
|---|---:|---:|---:|---:|---:|---:|
| `HALF_TP_RUNNER_FIXED10_PT20_GB60_CAP7D` | 3847 | 1.444 | 8.51 | 37.7% | 5/8 | -4905.92 |
| `HALF_TP_RUNNER_FIXED10_PT20_GB60_CAP30D` | 3847 | 1.444 | 8.51 | 37.7% | 5/8 | -4905.92 |
| `HALF_TP_RUNNER_FIXED10_PT20_GB60_CAP90D` | 3847 | 1.444 | 8.51 | 37.7% | 5/8 | -4905.92 |
| `HALF_TP_RUNNER_FIXED10_PT20_GB60_CAP180D` | 3847 | 1.444 | 8.51 | 37.7% | 5/8 | -4905.92 |
| `HALF_TP_RUNNER_FIXED10_PT20_GB60_CAP365D` | 3847 | 1.444 | 8.51 | 37.7% | 5/8 | -4905.92 |

### TIME_ONLY

| Variant | Trades | PF | Mean P&L | Win rate | Positive years | Worst year |
|---|---:|---:|---:|---:|---:|---:|
| `TIME_FIXED3_CAP1D` | 3815 | 0.000 | -9.74 | 0.0% | 0/8 | -7516.47 |
| `TIME_FIXED3_CAP2D` | 3815 | 0.000 | -9.74 | 0.0% | 0/8 | -7516.47 |
| `TIME_FIXED3_CAP3D` | 3815 | 0.000 | -9.74 | 0.0% | 0/8 | -7516.47 |
| `TIME_FIXED3_CAP4D` | 3815 | 0.000 | -9.74 | 0.0% | 0/8 | -7516.47 |
| `TIME_FIXED3_CAP5D` | 3815 | 0.000 | -9.74 | 0.0% | 0/8 | -7516.47 |

## Round 3H-like reference

`FULL_TRAIL_STRUCT_CAP10_GB60_CAP365D`: PF 1.710, mean P&L 6.67 USDT, win rate 13.7%, positive in 5/8 tested years.

## Output files

- `variant_summary.csv` - all aggregate variants.
- `variant_stage_summary.csv` - PRE_BREAKOUT vs BREAKOUT_ATTEMPT.
- `variant_year_summary.csv` - year robustness.
- `family_leaders.csv` - top ten by PF and mean P&L per family.
- `exit_reason_summary.csv` - stop/target/time-cap mix.
- `selected_trade_results.csv` - representative trade-level variants.
- `cohort.csv` - regenerated entry cohort.
- `manifest.json` - exact grid and methodology.

## Caveats

Independent overlapping trades are not a capital-constrained account simulation. The universe is the established research set plus named reference breakout coins, not a survivorship-free reconstruction of all Binance listings. Round 2 should narrow the search space for confirmation rather than directly become a live strategy.
