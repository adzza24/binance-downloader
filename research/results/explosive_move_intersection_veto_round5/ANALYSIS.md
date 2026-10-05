# Explosive Move Intersection / Veto Round 5

**Status:** RESEARCH ONLY on isolated `round-3` branch. No live strategy or automation changes.

## Question

Can a small intersection of contemporaneous conditions, including noise-veto rules, remove substantially more broad-detector noise while retaining at least ~90% of the original explosive events?

- Input alerts: **69,407** from Round 4.
- Original events: **8,683**.
- Numeric features considered: **82**.
- Raw one-sided candidate conditions generated from 2019-2022 only: **7,712**.
- Discovery-prefiltered conditions entering beam search: **450**.
- Candidate thresholds are generated from discovery only.
- Intersections are selected on 2023-2024 validation while requiring the same recall floor in discovery.
- 2025-2026 confirmation is frozen and never participates in rule selection.

## Best small intersections

| Target | Split | Rules | Event recall | Noise retained | Noise removed | Selected alerts |
|---:|---|---:|---:|---:|---:|---:|
| 90% | DISCOVERY_2019_2022 | 3 | 90.8% | 47.4% | 52.6% | 12,349 |
| 90% | VALIDATION_2023_2024 | 3 | 90.0% | 45.7% | 54.3% | 9,404 |
| 90% | CONFIRMATION_2025_2026 | 3 | 90.2% | 45.7% | 54.3% | 11,761 |
| 92% | DISCOVERY_2019_2022 | 3 | 93.8% | 50.8% | 49.2% | 13,459 |
| 92% | VALIDATION_2023_2024 | 3 | 92.1% | 48.7% | 51.3% | 10,048 |
| 92% | CONFIRMATION_2025_2026 | 3 | 91.7% | 48.8% | 51.2% | 12,551 |
| 95% | DISCOVERY_2019_2022 | 1 | 96.4% | 60.0% | 40.0% | 15,425 |
| 95% | VALIDATION_2023_2024 | 1 | 95.0% | 61.4% | 38.6% | 12,234 |
| 95% | CONFIRMATION_2025_2026 | 1 | 95.0% | 66.8% | 33.2% | 16,510 |

### 90% target rule set

- `ret_6h_delta_24h <= -0.012963373` (noise_q0.5).
- `ret_24h_delta_12h <= 0.029108777` (noise_q0.7).
- `ret_24h_delta_3h <= 0.033801522` (noise_q0.8).

### 92% target rule set

- `ret_6h_delta_24h <= -0.011562848` for MOMENTUM alerts only (noise_q0.5).
- `ret_24h_delta_6h <= -0.011583514` for MOMENTUM alerts only (noise_q0.5).
- `ret_6h_delta_24h <= 0.0021174755` (noise_q0.6).

### 95% target rule set

- `ret_24h_delta_3h <= 0.0015511966` (noise_q0.6).

## Primary 90% interpretation

Validation retains **90.0%** of events while retaining **45.7%** of unmatched alerts, removing **54.3%** of validation noise.
Frozen 2025-2026 confirmation retains **90.2%** of events while retaining **45.7%** of unmatched alerts, removing **54.3%** of confirmation noise.

## Guardrails

- Event recall, not alert-level accuracy, is the binding constraint because multiple alerts can map to one event.
- Family-specific conditions are only allowed for families with enough discovery observations; the tiny capitulation cluster is not independently tuned.
- This is a filter study, not an entry/exit strategy.
- Survivorship bias in the original Round 1 universe remains.
- 2025-2026 is previously exposed confirmation data, not a pristine holdout.
- No result is promoted to the live strategy automatically.