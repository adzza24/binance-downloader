# Explosive Move Candidate Evolution Round 7

**Status:** RESEARCH ONLY on isolated `round-3` branch. No live strategy or automation changes.

## Objective

Starting from the Round 5 90%-recall alerts collapsed into 96h same-symbol candidate episodes, test whether causal developments after candidate open separate wanted explosive candidates from noise more effectively than the first-alert snapshot.

- Evolution checkpoints: **3h, 6h, 12h, 24h, 48h**.
- Actionability gate: candidate has not traded more than **+5%** above its opening price by that checkpoint.
- Rule thresholds are generated from 2019-2022 discovery only.
- Intersections are selected using 2019-2024 development data. 2025-2026 is frozen confirmation only.
- The search maximises retained wanted candidate episodes for progressively lower candidate-level noise caps; weekly breadth is reported, never imposed as a cap on winners.

## Actionable population before evolution rules

| Checkpoint | Split | Wanted episodes | Noise episodes | Wanted share |
|---:|---|---:|---:|---:|
| 3h | DISCOVERY_2019_2022 | 1,657 | 1,405 | 54.1% |
| 3h | VALIDATION_2023_2024 | 1,563 | 1,808 | 46.4% |
| 3h | CONFIRMATION_2025_2026 | 1,553 | 2,298 | 40.3% |
| 6h | DISCOVERY_2019_2022 | 1,546 | 1,350 | 53.4% |
| 6h | VALIDATION_2023_2024 | 1,464 | 1,748 | 45.6% |
| 6h | CONFIRMATION_2025_2026 | 1,485 | 2,230 | 40.0% |
| 12h | DISCOVERY_2019_2022 | 1,359 | 1,258 | 51.9% |
| 12h | VALIDATION_2023_2024 | 1,313 | 1,632 | 44.6% |
| 12h | CONFIRMATION_2025_2026 | 1,348 | 2,065 | 39.5% |
| 24h | DISCOVERY_2019_2022 | 1,132 | 1,067 | 51.5% |
| 24h | VALIDATION_2023_2024 | 1,107 | 1,426 | 43.7% |
| 24h | CONFIRMATION_2025_2026 | 1,144 | 1,758 | 39.4% |
| 48h | DISCOVERY_2019_2022 | 893 | 837 | 51.6% |
| 48h | VALIDATION_2023_2024 | 885 | 1,196 | 42.5% |
| 48h | CONFIRMATION_2025_2026 | 909 | 1,462 | 38.3% |

## Strongest stable evolution discriminators

| Checkpoint | Feature | Direction | Discovery AUC | Validation AUC | Confirmation AUC |
|---:|---|---|---:|---:|---:|
| 48h | `cur_atr24_pct` | HIGHER | 0.683 | 0.733 | 0.668 |
| 48h | `cur_range_24h` | HIGHER | 0.671 | 0.715 | 0.660 |
| 48h | `cur_range_7d` | HIGHER | 0.686 | 0.720 | 0.659 |
| 48h | `cur_range_14d` | HIGHER | 0.666 | 0.712 | 0.657 |
| 48h | `cur_atr30d_pct` | HIGHER | 0.656 | 0.677 | 0.657 |
| 24h | `cur_atr24_pct` | HIGHER | 0.674 | 0.704 | 0.656 |
| 48h | `cur_atr7d_pct` | HIGHER | 0.668 | 0.706 | 0.655 |
| 48h | `path_range` | HIGHER | 0.676 | 0.703 | 0.650 |
| 24h | `cur_range_7d` | HIGHER | 0.670 | 0.693 | 0.649 |
| 48h | `cur_range_30d` | HIGHER | 0.649 | 0.684 | 0.664 |
| 48h | `cur_dist_7d_high` | LOWER | 0.660 | 0.703 | 0.648 |
| 48h | `cur_range_72h` | HIGHER | 0.670 | 0.712 | 0.647 |
| 24h | `cur_atr7d_pct` | HIGHER | 0.654 | 0.677 | 0.646 |
| 48h | `mae_since_open` | LOWER | 0.677 | 0.706 | 0.642 |
| 24h | `cur_atr30d_pct` | HIGHER | 0.641 | 0.653 | 0.648 |
| 12h | `cur_atr7d_pct` | HIGHER | 0.650 | 0.663 | 0.638 |
| 12h | `cur_range_7d` | HIGHER | 0.657 | 0.666 | 0.636 |
| 12h | `cur_atr30d_pct` | HIGHER | 0.635 | 0.639 | 0.644 |
| 12h | `cur_atr24_pct` | HIGHER | 0.657 | 0.672 | 0.635 |
| 24h | `cur_range_30d` | HIGHER | 0.635 | 0.666 | 0.650 |

## Best candidate-level frontiers by checkpoint

| Checkpoint | Dev noise cap | Dev wanted | Dev noise | Conf wanted | Conf noise | Conf events | Conf weeks | Precision |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3h | 1,000 | 1,878 | 993 | 651 | 655 | 947 | 83 | 49.8% |
| 3h | 500 | 1,263 | 500 | 371 | 310 | 562 | 77 | 54.5% |
| 3h | 200 | 714 | 199 | 190 | 128 | 323 | 68 | 59.7% |
| 3h | 100 | 467 | 100 | 133 | 72 | 236 | 61 | 64.9% |
| 3h | 50 | 292 | 49 | 68 | 31 | 127 | 47 | 68.7% |
| 3h | 20 | 177 | 20 | 43 | 19 | 81 | 30 | 69.4% |
| 3h | 10 | 142 | 10 | 20 | 13 | 43 | 20 | 60.6% |
| 3h | 5 | 109 | 5 | 16 | 11 | 37 | 18 | 59.3% |
| 3h | 2 | 85 | 2 | 14 | 10 | 31 | 16 | 58.3% |
| 3h | 1 | 76 | 1 | 10 | 9 | 23 | 11 | 52.6% |
| 3h | 0 | 61 | 0 | 8 | 7 | 19 | 9 | 53.3% |
| 6h | 1,000 | 1,783 | 992 | 665 | 628 | 942 | 83 | 51.4% |
| 6h | 500 | 1,212 | 499 | 384 | 346 | 577 | 80 | 52.6% |
| 6h | 200 | 671 | 192 | 180 | 104 | 305 | 70 | 63.4% |
| 6h | 100 | 420 | 98 | 114 | 50 | 209 | 58 | 69.5% |
| 6h | 50 | 280 | 48 | 60 | 28 | 103 | 34 | 68.2% |
| 6h | 20 | 147 | 20 | 38 | 20 | 62 | 22 | 65.5% |
| 6h | 10 | 118 | 10 | 47 | 17 | 76 | 24 | 73.4% |
| 6h | 5 | 80 | 5 | 9 | 3 | 14 | 11 | 75.0% |
| 6h | 2 | 65 | 2 | 19 | 7 | 30 | 10 | 73.1% |
| 6h | 1 | 59 | 1 | 8 | 4 | 18 | 11 | 66.7% |
| 6h | 0 | 57 | 0 | 27 | 10 | 43 | 17 | 73.0% |
| 12h | 1,000 | 1,625 | 996 | 653 | 677 | 912 | 85 | 49.1% |
| 12h | 500 | 1,086 | 500 | 364 | 341 | 547 | 79 | 51.6% |
| 12h | 200 | 661 | 198 | 206 | 138 | 341 | 71 | 59.9% |
| 12h | 100 | 441 | 100 | 140 | 86 | 232 | 53 | 61.9% |
| 12h | 50 | 292 | 49 | 94 | 39 | 167 | 41 | 70.7% |
| 12h | 20 | 187 | 19 | 68 | 17 | 109 | 32 | 80.0% |
| 12h | 10 | 143 | 10 | 60 | 19 | 93 | 22 | 75.9% |
| 12h | 5 | 106 | 5 | 47 | 11 | 78 | 20 | 81.0% |
| 12h | 2 | 93 | 2 | 53 | 9 | 84 | 21 | 85.5% |
| 12h | 1 | 73 | 1 | 47 | 7 | 78 | 20 | 87.0% |
| 12h | 0 | 55 | 0 | 39 | 5 | 67 | 18 | 88.6% |
| 24h | 1,000 | 1,572 | 988 | 682 | 673 | 939 | 85 | 50.3% |
| 24h | 500 | 1,054 | 500 | 446 | 339 | 650 | 76 | 56.8% |
| 24h | 200 | 613 | 196 | 231 | 99 | 362 | 65 | 70.0% |
| 24h | 100 | 412 | 100 | 150 | 63 | 251 | 62 | 70.4% |
| 24h | 50 | 288 | 45 | 121 | 31 | 213 | 52 | 79.6% |
| 24h | 20 | 168 | 20 | 75 | 12 | 131 | 34 | 86.2% |
| 24h | 10 | 129 | 10 | 59 | 9 | 106 | 29 | 86.8% |
| 24h | 5 | 94 | 5 | 42 | 5 | 66 | 17 | 89.4% |
| 24h | 2 | 80 | 2 | 31 | 4 | 57 | 17 | 88.6% |
| 24h | 1 | 75 | 1 | 30 | 4 | 49 | 17 | 88.2% |
| 24h | 0 | 69 | 0 | 26 | 2 | 44 | 15 | 92.9% |
| 48h | 1,000 | 1,412 | 999 | 659 | 736 | 898 | 84 | 47.2% |
| 48h | 500 | 1,006 | 498 | 435 | 356 | 620 | 78 | 55.0% |
| 48h | 200 | 650 | 200 | 250 | 147 | 397 | 63 | 63.0% |
| 48h | 100 | 444 | 97 | 172 | 64 | 281 | 52 | 72.9% |
| 48h | 50 | 328 | 48 | 131 | 43 | 214 | 44 | 75.3% |
| 48h | 20 | 222 | 20 | 77 | 14 | 127 | 23 | 84.6% |
| 48h | 10 | 168 | 10 | 84 | 12 | 138 | 29 | 87.5% |
| 48h | 5 | 154 | 5 | 63 | 13 | 100 | 17 | 82.9% |
| 48h | 2 | 114 | 2 | 41 | 3 | 66 | 14 | 93.2% |
| 48h | 1 | 107 | 1 | 41 | 2 | 66 | 14 | 95.3% |
| 48h | 0 | 88 | 0 | 41 | 3 | 68 | 14 | 93.2% |

## Descriptive confirmation highlight

Among searched solutions that still retain at least 50 confirmation wanted episodes, the highest observed confirmation precision is **87.5%** at the **+48h** checkpoint: **84 wanted vs 12 noise**, covering **138 unique wanted events across 29 weeks**.

This is descriptive only: confirmation data was not used to choose the rules.

## Guardrails

- Future checkpoint information is used only at the checkpoint when it would actually be known.
- Candidates that already traded >+5% from candidate-open before a checkpoint are excluded from that checkpoint as no longer early/actionable under this conservative proxy.
- Wanted/noise labels are still defined from the historical explosive-event catalogue, so this remains research rather than a deployable strategy.
- 2025-2026 has been exposed in prior research and is confirmation, not a pristine untouched holdout.
- No result is promoted to the live strategy automatically.
