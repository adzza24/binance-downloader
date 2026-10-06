# BASE +12h Strategy Research - Round 01 Analysis

**Research only. No live strategy, portfolio, log or automation changes.**

## Frozen strategy

- BASE candidate family.
- First accepted checkpoint exactly +12h.
- Enter at signal close.
- OCO +10% target / -2% stop.
- 0.30% round-trip cost.
- Stop-first on same 1h candle ties.
- 30-day maximum research holding period; positions still open at 30 days are marked to market and closed.

## Universe and coverage

- Cached USDT symbols inventoried: **200**.
- Symbols with at least one usable continuous segment: **173**.
- Processing failures: **0**.

## By year

| Year | Signals | Trades | Symbols | Weeks | Target rate | Avg net | Median net | Median hold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2021 | 1 | 1 | 1 | 1 | 0.00% | -2.30% | -2.30% | 2.0h |
| 2022 | 6 | 6 | 6 | 4 | 50.00% | 3.70% | 3.70% | 5.5h |
| 2024 | 16 | 16 | 16 | 6 | 43.75% | 2.95% | -2.30% | 6.5h |
| 2025 | 25 | 25 | 24 | 9 | 56.00% | 4.42% | 9.70% | 10.0h |
| 2026 | 12 | 12 | 12 | 7 | 41.67% | 2.70% | -2.30% | 11.0h |
| ALL | 60 | 60 | 49 | 27 | 48.33% | 3.50% | -2.30% | 9.5h |

## Overlap

- Trades overlapping an already-open trade: **46.7%**.
- Maximum observed simultaneous open trades: **11**.

## Annual $1,000 portfolio simulations

Each year resets to $1,000. Sleeve capital compounds within that year. Signals that arrive while every sleeve is occupied are skipped.

| Year | Sleeves | Ending equity | Annual return | Accepted | Capacity skipped | Max realised DD |
|---|---:|---:|---:|---:|---:|---:|
| 2021 | 1 | $977.00 | -2.30% | 1 | 0 | -2.30% |
| 2021 | 2 | $988.50 | -1.15% | 1 | 0 | -1.15% |
| 2021 | 3 | $992.33 | -0.77% | 1 | 0 | -0.77% |
| 2022 | 1 | $1,260.11 | 26.01% | 5 | 1 | -2.30% |
| 2022 | 2 | $1,118.56 | 11.86% | 6 | 0 | -2.30% |
| 2022 | 3 | $1,079.04 | 7.90% | 6 | 0 | -1.55% |
| 2024 | 1 | $1,175.15 | 17.51% | 8 | 8 | -4.55% |
| 2024 | 2 | $1,161.92 | 16.19% | 12 | 4 | -4.47% |
| 2024 | 3 | $1,166.52 | 16.65% | 15 | 1 | -5.94% |
| 2025 | 1 | $1,741.91 | 74.19% | 11 | 14 | -2.30% |
| 2025 | 2 | $1,458.82 | 45.88% | 14 | 11 | -4.47% |
| 2025 | 3 | $1,329.80 | 32.98% | 16 | 9 | -3.06% |
| 2026 | 1 | $1,046.60 | 4.66% | 8 | 4 | -8.89% |
| 2026 | 2 | $1,071.80 | 7.18% | 9 | 3 | -4.55% |
| 2026 | 3 | $1,080.20 | 8.02% | 10 | 2 | -3.05% |

## Interpretation guardrails

- This is a broad cached-universe robustness replay, not a pristine holdout.
- The cache defines the universe, so symbol/listing survivorship and incomplete historical coverage remain limitations.
- Portfolio results are capacity models, not proof of executable live performance.
- No result from this round is promoted to the live strategy automatically.
