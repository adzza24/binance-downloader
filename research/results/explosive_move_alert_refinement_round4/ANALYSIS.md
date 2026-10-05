# Explosive Move Alert Refinement Round 4

**Status:** RESEARCH ONLY on isolated `round-3` branch. No live strategy or automation changes.

## Question

Starting from the very-high-recall Round 3 detector, which contemporaneous factors distinguish alerts that occur inside one of the original 8,683 explosive-event windows from unmatched/noise alerts?

- Alert population analysed: **69,407**.
- Original event population: **8,683**.
- Discovery candidate thresholds are derived from 2019-2022 only.
- The filter ladder is selected using 2023-2024 validation subject to explicit event-recall floors.
- 2025-2026 is reported as previously exposed confirmation data and is never used to choose rules.

## Most stable individual discriminators

| Feature | Direction | Discovery AUC | Validation AUC | Confirmation AUC |
|---|---|---:|---:|---:|
| range_7d | HIGHER | 0.655 | 0.642 | 0.622 |
| atr7d_pct | HIGHER | 0.647 | 0.622 | 0.619 |
| dist_7d_high | LOWER | 0.635 | 0.618 | 0.643 |
| range_14d | HIGHER | 0.641 | 0.626 | 0.613 |
| atr24_pct | HIGHER | 0.656 | 0.643 | 0.612 |
| worst_1h_ret_7d | LOWER | 0.619 | 0.610 | 0.614 |
| range_72h | HIGHER | 0.660 | 0.647 | 0.606 |
| range_30d | HIGHER | 0.615 | 0.605 | 0.605 |
| range_24h | HIGHER | 0.638 | 0.626 | 0.594 |
| score | HIGHER | 0.638 | 0.611 | 0.592 |
| worst_4h_ret_14d | LOWER | 0.615 | 0.585 | 0.595 |
| atr30d_pct | HIGHER | 0.608 | 0.584 | 0.610 |
| dist_30d_high_delta_24h | LOWER | 0.580 | 0.584 | 0.581 |
| dist_30d_high | LOWER | 0.580 | 0.574 | 0.612 |
| dist_30d_high_delta_12h | LOWER | 0.572 | 0.580 | 0.577 |

## Recall / noise Pareto ladder

| Target | Split | Rules | Event recall | Noise retained | Selected alerts |
|---:|---|---:|---:|---:|---:|
| 95% | CONFIRMATION_2025_2026 | 2 | 93.3% | 64.2% | 16,035 |
| 95% | DISCOVERY_2019_2022 | 2 | 93.6% | 54.1% | 14,854 |
| 95% | VALIDATION_2023_2024 | 2 | 95.0% | 59.7% | 12,183 |
| 90% | CONFIRMATION_2025_2026 | 2 | 87.5% | 44.6% | 11,534 |
| 90% | DISCOVERY_2019_2022 | 2 | 89.7% | 43.2% | 12,431 |
| 90% | VALIDATION_2023_2024 | 2 | 90.0% | 44.2% | 9,436 |
| 85% | CONFIRMATION_2025_2026 | 2 | 83.8% | 38.1% | 9,912 |
| 85% | DISCOVERY_2019_2022 | 2 | 87.2% | 37.4% | 10,636 |
| 85% | VALIDATION_2023_2024 | 2 | 85.4% | 37.6% | 8,087 |
| 80% | CONFIRMATION_2025_2026 | 2 | 78.6% | 36.5% | 9,836 |
| 80% | DISCOVERY_2019_2022 | 2 | 79.4% | 34.0% | 10,058 |
| 80% | VALIDATION_2023_2024 | 2 | 80.0% | 37.4% | 8,258 |
| 70% | CONFIRMATION_2025_2026 | 3 | 65.9% | 22.7% | 6,111 |
| 70% | DISCOVERY_2019_2022 | 3 | 74.7% | 24.5% | 7,575 |
| 70% | VALIDATION_2023_2024 | 3 | 70.0% | 23.6% | 5,317 |

## Selected interpretable rules

### 70% recall target

- Step 1: for all alerts, require `score_margin >= 0.0618992`.
- Step 2: for MOMENTUM alerts only, require `score_change_prev_alert >= 0.00116308`.
- Step 3: for MOMENTUM alerts only, require `ret_60d >= -0.656839`.

### 80% recall target

- Step 1: for all alerts, require `dist_30d_high_delta_24h <= -0.0212766`.
- Step 2: for MOMENTUM alerts only, require `dist_90d_high >= -0.728193`.

### 85% recall target

- Step 1: for all alerts, require `ret_24h_delta_12h <= -0.021788`.
- Step 2: for all alerts, require `ret_90d >= -0.725635`.

### 90% recall target

- Step 1: for MOMENTUM alerts only, require `ret_24h_delta_24h <= -0.0134951`.
- Step 2: for all alerts, require `ret_90d >= -0.725635`.

### 95% recall target

- Step 1: for MOMENTUM alerts only, require `ret_24h_delta_3h <= 0.00264371`.
- Step 2: for all alerts, require `ret_90d >= -0.725635`.

## Interpretation guardrails

- Positive alert rows are not independent observations. Event recall, not alert-level accuracy, is therefore the primary selection metric.
- Several alerts may map to one event; retaining any qualifying alert retains that event.
- AUC is descriptive only. Rules are selected using event-level recall and noise-alert retention.
- The ladder is a research filter study, not an entry or exit strategy.
- No result is promoted to the live strategy automatically.
