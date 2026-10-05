# Explosive Move Zero-Noise Frontier Round 6

**Status:** RESEARCH ONLY on isolated `round-3` branch. No live strategy or automation changes.

## Objective

Starting from the Round 5 90%-recall filtered alerts, collapse repeated same-symbol alerts into 96h candidate episodes and progressively intersect contemporaneous first-alert conditions. Minimise candidate-level noise toward zero while retaining as many genuine wanted candidate episodes/events as possible. The ~381 active-week figure is a floor/coverage diagnostic, never a cap on winners.

- Dedupe window: **96h**.
- Candidate first-alert numeric features: **82**.
- Raw candidate conditions: **6083**.
- Prefiltered conditions entering search: **473**.
- Thresholds are generated from 2019-2022 discovery only.
- 2019-2024 is the development frontier; 2025-2026 is frozen confirmation and never participates in rule selection.
- All intersection features are from the first alert that opened the 96h candidate. No future episode information is used.

## Baseline 96h candidates

| Split | Wanted candidate episodes | Noise candidate episodes | Unique wanted events | Weeks covered |
|---|---:|---:|---:|---:|
| DISCOVERY_2019_2022 | 1,797 | 1,683 | 3,368 | 193 |
| VALIDATION_2023_2024 | 1,632 | 1,959 | 2,326 | 105 |
| CONFIRMATION_2025_2026 | 1,625 | 2,410 | 2,158 | 86 |

## Progressive noise frontier

| Dev noise cap | Rules | Dev wanted episodes | Dev noise episodes | Confirmation wanted episodes | Confirmation noise episodes | Confirmation unique events | Confirmation weeks |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,000 | 0 | 3,429 | 3,642 | 1,625 | 2,410 | 2,158 | 86 |
| 3,000 | 2 | 3,260 | 3,000 | 1,569 | 2,111 | 2,108 | 86 |
| 2,500 | 4 | 3,025 | 2,495 | 1,426 | 1,805 | 1,954 | 86 |
| 2,000 | 4 | 2,729 | 2,000 | 1,239 | 1,432 | 1,728 | 86 |
| 1,500 | 4 | 2,383 | 1,494 | 950 | 976 | 1,351 | 86 |
| 1,000 | 5 | 1,936 | 1,000 | 688 | 636 | 1,032 | 86 |
| 750 | 4 | 1,651 | 745 | 571 | 511 | 881 | 84 |
| 500 | 4 | 1,307 | 499 | 371 | 324 | 596 | 82 |
| 300 | 4 | 949 | 300 | 277 | 192 | 468 | 78 |
| 200 | 5 | 750 | 198 | 210 | 117 | 366 | 72 |
| 100 | 7 | 523 | 100 | 132 | 61 | 242 | 60 |
| 50 | 5 | 345 | 50 | 76 | 34 | 158 | 48 |
| 20 | 5 | 202 | 20 | 59 | 17 | 106 | 33 |
| 10 | 7 | 145 | 10 | 24 | 10 | 56 | 23 |
| 5 | 5 | 112 | 5 | 21 | 11 | 34 | 16 |
| 2 | 5 | 90 | 2 | 20 | 13 | 41 | 17 |
| 1 | 5 | 80 | 1 | 18 | 5 | 38 | 17 |
| 0 | 6 | 70 | 0 | 5 | 2 | 9 | 5 |

## Least-restrictive development zero-noise solution found

- Step 1: `ret_14d >= 0.64035871`.
- Step 2: `ret_90d >= 0.16004435`.
- Step 3: `range_72h >= 0.26297185`.
- Step 4: `MOMENTUM only: range_90d >= 1.6543747`.
- Step 5: `MOMENTUM only: atr30d_pct >= 0.012733484`.
- Step 6: `MOMENTUM only: volume_ratio_6h_delta_6h <= 0.3201291`.

This solution has zero candidate-level noise across 2019-2024 development data. Its frozen 2025-2026 confirmation outcome is shown in the frontier table and is the important generalisation check.

## Guardrails

- This is a candidate-filter study, not an entry/exit strategy.
- Zero noise in fitted/development history is not proof of future zero noise; confirmation performance is decisive.
- The original current-universe survivorship bias remains.
- 2025-2026 has been seen in earlier research and is confirmation, not a pristine untouched holdout.
- No result is promoted to the live strategy automatically.
