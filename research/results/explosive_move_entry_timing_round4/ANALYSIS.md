# Explosive Move Entry Timing Round 4

**Status:** RESEARCH ONLY. No live strategy or automation changes.

Round 4 keeps the Round 3 Random Forest qualification fixed (CAPITULATION and BASE) and tests only when to enter after the RF first qualifies a causal watch. No exit, trailing-stop or profit-taking rule is used for selection.

Timing rules are evaluated hourly after RF qualification. Delayed entries must be within 5% of the lowest price observed since qualification. Selection uses 2023-2024 validation and prioritises +20-before--5 path quality while requiring +20-any to stay within 10 percentage points of the immediate-entry baseline where possible. The -5% measure is used only as an ordering/path-quality metric, not as an exit assumption.

2025-2026 remains previously exposed confirmation data, not a pristine holdout.

## Selected timing rules

| Archetype | Rule | Validation alerts | Capture vs immediate | +20 any | +20 before -5 | Median MAE | Median delay |
|---|---|---:|---:|---:|---:|---:|---:|
| CAPITULATION | RF_STRENGTHEN_005 | 11 | 44.0% | 100.0% | 72.7% | -4.6% | 0.0h |
| BASE | POSTQUAL_REBOUND1 | 16 | 55.2% | 87.5% | 56.2% | -8.7% | 0.0h |

## Immediate vs selected timing

| Archetype | Split | Rule | Alerts | +20 any | +20 before -5 | +20 before -10 | Median MAE | Median h to +20 | Median delay |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| BASE | CONFIRMATION_2025_2026 | IMMEDIATE | 12 | 100.0% | 16.7% | 25.0% | -16.1% | 13.5 | 0.0h |
| BASE | CONFIRMATION_2025_2026 | POSTQUAL_REBOUND1 | 5 | 80.0% | 20.0% | 40.0% | -21.8% | 9.5 | 0.0h |
| BASE | VALIDATION_2023_2024 | IMMEDIATE | 29 | 93.1% | 48.3% | 72.4% | -6.9% | 18.0 | 0.0h |
| BASE | VALIDATION_2023_2024 | POSTQUAL_REBOUND1 | 16 | 87.5% | 56.2% | 75.0% | -8.7% | 17.5 | 0.0h |
| CAPITULATION | CONFIRMATION_2025_2026 | IMMEDIATE | 37 | 75.7% | 21.6% | 27.0% | -18.3% | 17.0 | 0.0h |
| CAPITULATION | CONFIRMATION_2025_2026 | RF_STRENGTHEN_005 | 15 | 60.0% | 20.0% | 40.0% | -21.8% | 20.0 | 0.0h |
| CAPITULATION | VALIDATION_2023_2024 | IMMEDIATE | 25 | 96.0% | 60.0% | 84.0% | -5.1% | 15.5 | 0.0h |
| CAPITULATION | VALIDATION_2023_2024 | RF_STRENGTHEN_005 | 11 | 100.0% | 72.7% | 100.0% | -4.6% | 13.0 | 0.0h |

## Guardrails

- Round 3 RF models and qualification thresholds are held fixed; this round changes timing only.
- +20-before--5 is a path-ordering metric, not a simulated stop-loss return.
- MFE/MAE and +20-any remain visible so a timing rule cannot look good merely by filtering out difficult paths.
- 2025-2026 has been seen in earlier research and is confirmation only.
- No result is promoted to the live strategy automatically.
