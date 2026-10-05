# Explosive Move Full-Population Replay Round 8

**Status:** RESEARCH ONLY on isolated `round-3`. No live strategy or automation changes.

## Test design

- Replay period: **2024-01-01 through 2026-10-05T15:00:00+00:00** using real Binance 1h candles.
- Signal generation is causal and does not consult the explosive-event catalogue or wanted/noise labels.
- Frozen pipeline: Round 3 family detector -> Round 5 90% precursor intersection -> 96h same-symbol candidate dedupe -> Round 7 checkpoint-union rules.
- Frozen union is the Round 7 winner-maximising development-noise-cap-5 combination: `3h=1 | 6h=0 | 12h=2 | 24h=1 | 48h=1`.
- Candidates that exceeded +5% from candidate open before a checkpoint are not eligible at that checkpoint.
- Outcomes are measured directly from subsequent candles after the signal, not from catalogue membership.
- Current-liquidity ranks 1-100 are reported separately from ranks 101-200, which were outside the original top-100 event universe.
- Simple target/stop diagnostics subtract **0.30%** round-trip costs and use conservative stop-first ordering if target and stop touch in the same 1h candle.

## Results

| Tier | Year | Signals | Weeks | Mature 30d | Median 30d MFE | Median 30d MAE | +10% hit | +20% hit | +20/-5 target rate | +20/-5 avg net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ORIGINAL_TOP100 | ALL | 236 | 65 | 232 | 22.2% | -23.6% | 78.0% | 56.0% | 16.8% | -1.0% |
| ORIGINAL_TOP100 | 2024 | 85 | 22 | 85 | 26.1% | -20.3% | 78.8% | 62.4% | 22.4% | 0.4% |
| ORIGINAL_TOP100 | 2025 | 115 | 28 | 115 | 20.3% | -25.2% | 78.3% | 50.4% | 11.3% | -2.5% |
| ORIGINAL_TOP100 | 2026 | 36 | 15 | 32 | 24.6% | -19.8% | 75.0% | 59.4% | 21.9% | 0.2% |
| EXTERNAL_101_200 | ALL | 233 | 88 | 231 | 22.0% | -24.9% | 78.8% | 55.8% | 23.4% | 0.5% |
| EXTERNAL_101_200 | 2024 | 81 | 25 | 81 | 27.4% | -20.5% | 80.2% | 66.7% | 34.6% | 3.3% |
| EXTERNAL_101_200 | 2025 | 107 | 37 | 107 | 18.2% | -28.0% | 77.6% | 46.7% | 15.0% | -1.6% |
| EXTERNAL_101_200 | 2026 | 45 | 26 | 43 | 26.0% | -27.5% | 79.1% | 58.1% | 23.3% | 0.5% |
| ALL_TOP200 | ALL | 469 | 104 | 463 | 22.2% | -24.2% | 78.4% | 55.9% | 20.1% | -0.3% |
| ALL_TOP200 | 2024 | 166 | 31 | 166 | 26.2% | -20.5% | 79.5% | 64.5% | 28.3% | 1.9% |
| ALL_TOP200 | 2025 | 222 | 41 | 222 | 19.6% | -26.0% | 77.9% | 48.6% | 13.1% | -2.0% |
| ALL_TOP200 | 2026 | 81 | 32 | 75 | 25.2% | -23.3% | 77.3% | 58.7% | 22.7% | 0.4% |

## Interpretation guardrails

- This is substantially less sample-conditioned than the previous event/noise studies, but it is not a pristine prospective test because the frozen rules were developed using portions of 2024-2026 history.
- The rank 101-200 tier is a useful symbol-level distribution-shift check, but the universe is still based on symbols/liquidity available at replay-run time, so delisted-coin survivorship bias remains.
- Recent signals without a full 30 days of forward candles are retained but excluded from 30-day target/stop return averages.
- No result is promoted to the live strategy automatically.
