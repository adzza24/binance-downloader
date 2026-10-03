# Explosive Move Entry Discovery Round 2

**Status:** RESEARCH ONLY. No Early Breakout automation or Crypto Live Strategy Specification changes.

## Design

Round 2 keeps the Round 1 clean-event definition (+20% before -5%, maximum 30 days) but tests causal entry decisions separately for CAPITULATION/REVERSAL and BASE/ACCUMULATION. The Round 1 all-period cluster IDs are not reused: archetypes are refit using 2019-2022 discovery winners only, then frozen for 2023-2026 assignment.

The primary matched comparison is winner vs same-symbol/year HARD FAIL controls from Round 1, with hard controls rechecked to require +5% before a later -5% stop and no +20% first. Candidate decisions are observed at the first 0%, +1%, +2%, +3% and +5% progress checkpoints; decisions use only data known at that hourly close and entries occur at the next hourly open plus configured slippage.

Simple interpretable reversal/expansion rules are tested first. A shallow Random Forest is then fitted on discovery only; probability thresholds/checkpoints are selected on 2023-2024 validation. 2025-2026 is only reported after selection. A separate whole-history causal replay starts watches from discovery-derived archetype gates and measures real alert frequency and false alerts.

Frozen Round 1 ranking was filtered to the first 50 eligible pairs after removing stablecoins, known unavailable pairs, leveraged-token patterns and identified tokenised-equity symbols.

## Anchor cohort

| Archetype | Split | Winners | Hard failures |
|---|---|---:|---:|
| BASE | DISCOVERY_2019_2022 | 2032 | 6517 |
| BASE | HOLDOUT_2025_2026 | 1060 | 3180 |
| BASE | VALIDATION_2023_2024 | 1404 | 4098 |
| CAPITULATION | DISCOVERY_2019_2022 | 392 | 480 |
| CAPITULATION | HOLDOUT_2025_2026 | 87 | 105 |
| CAPITULATION | VALIDATION_2023_2024 | 34 | 14 |

## Discovery-only archetype fit

| Cluster | Name | Discovery winners | Median 30d return | Median 14d range | Median 7d ATR |
|---:|---|---:|---:|---:|---:|
| 0 | MOMENTUM | 226 | 201.9% | 173.7% | 3.77% |
| 1 | BASE | 2032 | -1.3% | 51.7% | 2.19% |
| 2 | CAPITULATION | 392 | -42.5% | 150.2% | 4.95% |

## Validation-selected simple triggers

| Archetype | Progress | Rule | Validation selected | +20/-5 precision | Winner recall | Binary expectancy* |
|---|---:|---|---:|---:|---:|---:|
| CAPITULATION | 0% | REBOUND12_1 | 39 | 66.7% | 82.4% | 11.4% |
| BASE | 1% | VOL_TRADE13 | 1683 | 16.3% | 33.3% | -1.2% |

## Validation-selected model triggers

| Archetype | Progress | Validation selected | +20/-5 precision | Winner recall | Binary expectancy* |
|---|---:|---:|---:|---:|---:|
| BASE | 0% | 166 | 96.4% | 11.5% | 18.8% |

## Untouched 2025-2026 matched-cohort results

| Method | Archetype | Progress | Selected | +20/-5 precision | Winner recall | Median entry advance | Binary expectancy* |
|---|---|---:|---:|---:|---:|---:|---:|
| SIMPLE | CAPITULATION | 0% | 135 | 37.0% | 60.9% | 0.0% | 4.0% |
| SIMPLE | BASE | 1% | 1401 | 17.5% | 42.4% | 1.4% | -0.9% |
| RF | BASE | 0% | 62 | 95.2% | 5.8% | 0.0% | 18.5% |

## Causal replay

This replay does not start from hindsight event timestamps. It starts a watch only when the frozen discovery-era archetype gate becomes observable, evaluates the selected trigger once at its specified progress checkpoint, enters next-hour open, and applies no exit optimisation. This is the main false-alert stress test.

| Method | Archetype | Split | Alerts/wk | False/wk | +20/-5 precision | Clean-event recall | Median captured lateness | Binary expectancy* |
|---|---|---|---:|---:|---:|---:|---:|---:|
| RF | BASE | HOLDOUT_2025_2026 | 0.13 | 0.10 | 18.2% | 0.2% | 0.0% | -0.7% |
| RF | BASE | VALIDATION_2023_2024 | 0.20 | 0.11 | 42.9% | 0.6% | 0.0% | 5.5% |
| SIMPLE | BASE | HOLDOUT_2025_2026 | 19.65 | 16.11 | 18.0% | 12.6% | 1.7% | -0.7% |
| SIMPLE | BASE | VALIDATION_2023_2024 | 16.20 | 12.65 | 21.9% | 7.2% | 1.4% | 0.2% |
| SIMPLE | CAPITULATION | HOLDOUT_2025_2026 | 0.40 | 0.30 | 25.7% | 10.3% | 0.1% | 1.2% |
| SIMPLE | CAPITULATION | VALIDATION_2023_2024 | 0.18 | 0.07 | 63.2% | 35.3% | 0.1% | 10.5% |

## Interpretation guardrails

*Binary expectancy assumes every success exits at +20% and every non-success at -5%, with configured fee/slippage applied. Under that deliberately simple framework, roughly 61.1% +20-before--5 precision is required to clear a +10% mean net-return hurdle. It is an entry-quality diagnostic, not an optimised exit test.

- Primary discrimination remains winner vs HARD FAIL, not winner vs random quiet periods.
- 2025-2026 was not used to choose rule, checkpoint, model threshold or watch-gate parameter.
- Whole-market replay includes ordinary non-event periods, so its false-alert rate is more realistic than matched-control precision.
- The frozen current-liquidity universe still has survivorship bias. Historical point-in-time liquidity reconstruction remains a later robustness task.
- No result is promoted to the live strategy by this workflow.
