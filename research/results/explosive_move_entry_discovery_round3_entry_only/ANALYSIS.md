# Explosive Move Entry Discovery Round 3 - Entry Only

**Status:** RESEARCH ONLY. No Early Breakout automation or Crypto Live Strategy Specification changes.

## Design

This round isolates entry quality from exit design. CAPITULATION and BASE watches are generated causally from discovery-era context gates. During each watch, entry conditions are evaluated on every completed hourly candle and an entry is represented by the next hourly open plus configured slippage. Entries more than 5% above the watch anchor are excluded.

The Random Forest is trained only to predict whether price reaches +20% at any time within the following 30 days. No stop, trailing stop or profit-taking rule is part of the training label or trigger selection.

The primary entry metrics are +5/+10/+20/+30/+50 forward hit rates, 30-day MFE, 30-day MAE, target timing, and upside opportunity capped at +20%. +20-before--5 and +20-before--10 are reported only as secondary path-quality diagnostics.

The unchanged Round 2 REBOUND2_VOL11 rule is explicitly retained, alongside a separate watch-relative rebound variant.

2025-2026 is labelled CONFIRMATION, not untouched holdout, because prior rounds have already exposed results from that period.

Universe: 50 frozen eligible Binance USDT pairs. RF discovery training rows after deterministic per-symbol/archetype cap: 134430.

## Validation-selected configurations

| Method | Archetype | Trigger | Validation alerts | +20 any | Mean capped +20 opportunity | Median 30d MAE |
|---|---|---|---:|---:|---:|---:|
| SIMPLE | BASE | BREAK6_VOL13 | 3349 | 46.8% | 13.9% | -13.8% |
| RF | CAPITULATION | RF>=0.550 | 25 | 96.0% | 19.5% | -5.1% |
| RF | BASE | RF>=0.800 | 29 | 93.1% | 19.8% | -6.9% |

## End-to-end causal entry results

| Method | Archetype | Split | Alerts/wk | +10 any | +20 any | +30 any | Mean capped +20 opp. | Median MFE | Median MAE | +20 before -5 | +20 before -10 | Median h to +20 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RF | BASE | CONFIRMATION_2025_2026 | 0.14 | 100.0% | 100.0% | 58.3% | 20.0% | 35.3% | -16.1% | 16.7% | 25.0% | 13.5 |
| RF | BASE | VALIDATION_2023_2024 | 0.28 | 100.0% | 93.1% | 82.8% | 19.8% | 38.5% | -6.9% | 48.3% | 72.4% | 18.0 |
| RF | CAPITULATION | CONFIRMATION_2025_2026 | 0.43 | 89.2% | 75.7% | 51.4% | 17.7% | 30.3% | -18.3% | 21.6% | 27.0% | 17.0 |
| RF | CAPITULATION | VALIDATION_2023_2024 | 0.24 | 96.0% | 96.0% | 84.0% | 19.5% | 46.8% | -5.1% | 60.0% | 84.0% | 15.5 |
| SIMPLE | BASE | CONFIRMATION_2025_2026 | 40.91 | 59.7% | 35.1% | 22.0% | 12.4% | 13.0% | -18.2% | 16.2% | 26.0% | 248.5 |
| SIMPLE | BASE | VALIDATION_2023_2024 | 32.07 | 67.1% | 46.8% | 33.6% | 13.9% | 18.2% | -13.8% | 22.5% | 34.5% | 235.0 |

## Guardrails

- Mean capped +20 opportunity is not a realised return. It credits available upside up to +20% without pretending an exit captured it.
- MAE is shown beside MFE so a later rally after severe drawdown cannot masquerade as a clean entry.
- The -5% and -10% columns are diagnostics only; they do not select the trigger.
- No trailing stop or profit-taking logic is optimised here. Exit research remains separate.
- RF discovery rows are deterministically capped per symbol/archetype for tractability; validation and confirmation evaluate every eligible hourly candle inside each causal watch.
- The frozen current-liquidity universe retains survivorship bias.
- No result is promoted to the live strategy by this workflow.
