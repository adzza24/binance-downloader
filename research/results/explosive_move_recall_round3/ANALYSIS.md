# Explosive Move Recall Round 3

**Status:** RESEARCH ONLY on isolated `round-3` branch. No live strategy or automation changes.

## Purpose

Reset the entry research around recall. The fixed target population is the original Round 1 catalogue of 8,683 de-duplicated +20%-before--5 events. Three precursor families are learned from discovery-era winner conditions, then separate family classifiers are trained to recognise those conditions. Thresholds are selected on 2023-2024 with an explicit preference for at least 80% family-level event recall before reducing control fire rate.

An event is counted as captured when a causal alert appears on the same symbol from 72 hours before the original event start through that event's +5% milestone. This evaluation window uses hindsight only for scoring recall; the alert itself is generated from contemporaneous features.

## Learned families

| Family | Events |
|---|---:|
| CAPITULATION | 7 |
| BASE | 2034 |
| MOMENTUM | 6642 |

## Validation-selected family thresholds

| Family | Threshold | Validation event recall proxy | Validation control fire |
|---|---:|---:|---:|
| BASE | 0.65 | 81.0% | 2.7% |
| CAPITULATION | 0.70 | 85.7% | 7.1% |
| MOMENTUM | 0.35 | 89.5% | 49.7% |

## End-to-end causal recall

| Split | Events | Captured | Event recall | Alerts/week | False alerts/week | Median timing vs event start |
|---|---:|---:|---:|---:|---:|---:|
| CONFIRMATION_2025_2026 | 2396 | 2363 | 98.6% | 284.6 | 206.5 | -51.0h |
| DISCOVERY_2019_2022 | 3705 | 3671 | 99.1% | 130.0 | 73.9 | -57.0h |
| VALIDATION_2023_2024 | 2582 | 2546 | 98.6% | 185.7 | 119.8 | -53.0h |

## Independent episode check

Events are also grouped into fixed 48-hour market episodes so simultaneous altcoin moves are not mistaken for independent evidence.

| Split | Episodes | Any captured | Market-wide episodes (3+ symbols) | Market-wide any captured | Median event recall inside episode |
|---|---:|---:|---:|---:|---:|
| CONFIRMATION_2025_2026 | 214 | 100.0% | 177 | 100.0% | 100.0% |
| DISCOVERY_2019_2022 | 413 | 99.5% | 305 | 100.0% | 100.0% |
| VALIDATION_2023_2024 | 241 | 100.0% | 187 | 100.0% | 100.0% |

## Guardrails

- This is recall-first discovery, not a deployable strategy.
- False-alert rate, timing and drawdown remain diagnostics; they are not allowed to silently shrink the target population to a tiny high-precision subset.
- The original event catalogue itself is defined as +20% before -5%; this round does not use -5% as an exit or entry-selection rule.
- 2025-2026 has been seen in prior research and is confirmation data, not a pristine holdout.
- Current-universe survivorship bias from Round 1 remains.
- No result is promoted to the live strategy automatically.

## Replay checkpointing

Causal replay is checkpointed per symbol under `checkpoints/`. A rerun reuses completed symbol files and retries only unfinished symbols. Checkpoints are persisted to the isolated `round-3` branch even if a later analysis step fails.

## Threshold-selection fallback

One or more signal families had no usable 2023-2024 validation winner rows. Those families used discovery-era data only for threshold selection rather than touching 2025-2026 confirmation data.

- CAPITULATION: DISCOVERY_2019_2022_FALLBACK, threshold 0.70.
