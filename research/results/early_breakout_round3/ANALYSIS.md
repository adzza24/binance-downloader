# Early Breakout Round 3 - Signal Quality & Confirmation Research

**Status:** RESEARCH ONLY. No scheduled-task or live-strategy edits.

Frozen Round 1 cohort regenerated: 3,931 complete checkpoint-0 samples.

## Objective

Identify which signals are most likely to produce +20% within seven days before a -7% stop, and test whether delayed confirmation can raise mean P&L toward +30 USDT on a 300 USDT trade.

Temporal design: train 2019-2022, choose probability thresholds on 2023-2024 validation, report 2025-2026 as untouched holdout.

## Holdout model quality

| Checkpoint | Selected | Coverage | +20% precision | +10% hit | AUC |
|---:|---:|---:|---:|---:|---:|
| 4h | 82 | 6.6% | 20.7% | 41.5% | 0.642 |
| 0h | 34 | 2.7% | 17.6% | 38.2% | 0.597 |
| 24h | 44 | 3.5% | 13.6% | 38.6% | 0.563 |
| 8h | 116 | 9.3% | 12.1% | 34.5% | 0.592 |
| 12h | 45 | 3.6% | 11.1% | 35.6% | 0.580 |

## Best holdout strategy results

| Checkpoint | Exit | Trades | Mean P&L | Mean return | PF | Win rate |
|---:|---|---:|---:|---:|---:|---:|
| 24h | FIXED10_5 | 44 | 0.61 | 0.20% | 1.06 | 36.4% |
| 4h | FIXED20_7 | 82 | -0.02 | -0.01% | 1.00 | 26.8% |
| 4h | FULL_TRAIL_7 | 82 | -0.86 | -0.29% | 0.89 | 11.0% |
| 4h | FIXED10_5 | 82 | -0.93 | -0.31% | 0.91 | 32.9% |
| 8h | FIXED10_5 | 116 | -1.40 | -0.47% | 0.87 | 31.9% |
| 4h | LADDER10_20_RUNNER_7 | 82 | -1.69 | -0.56% | 0.86 | 45.1% |
| 12h | FIXED10_5 | 45 | -1.75 | -0.58% | 0.84 | 31.1% |
| 0h | FIXED20_7 | 34 | -2.69 | -0.90% | 0.84 | 23.5% |
| 4h | HALF20_RUNNER_7 | 82 | -3.41 | -1.14% | 0.79 | 26.8% |
| 24h | FULL_TRAIL_7 | 44 | -3.60 | -1.20% | 0.61 | 9.1% |

## 10% mean-return hurdle

No holdout checkpoint/exit combination reached +30 USDT mean P&L. The research does not support claiming a 10% average-trade strategy from this signal family yet.

## Notes

- Delayed checkpoints are genuine confirmation entries: no capital is committed before the checkpoint.
- The target is +20% within 7 days before a -7% adverse move.
- Exit tests include fixed +10%, fixed +20%, full runner, 50%@+20 plus runner, and 25%@+10 + 25%@+20 + 50% runner.
- Results are independent 300 USDT trades, not a capital-constrained portfolio simulation.
