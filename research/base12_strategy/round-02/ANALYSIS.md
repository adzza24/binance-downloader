# BASE +12h Strategy Research — Round 02 Analysis

**Research only. No live strategy, portfolio, log or automation changes.**

## Frozen strategy

- BASE candidate family.
- First accepted checkpoint exactly +12h.
- Enter at signal close.
- OCO +10% / -2%.
- 0.30% round-trip cost.
- No Round 02 parameter optimisation.

## Historical universe

- Historical USDT archive symbols included: **674**.
- Archive symbols explicitly excluded: **3,071**.
- Processing failures: **0**.
- Universe comes from Binance Vision historical monthly 1h archive listings, not current exchange ranking.

## Headline validation result

- Signals: **171**.
- Resolved trades: **171** across **144** symbols and **62** weeks.
- Target rate: **40.94%**.
- Average net return/trade: **2.61%**.
- Round 01 benchmark: 60 trades at **3.50%** average net/trade.

## By year

| Year | Signals | Trades | Symbols | Weeks | Target rate | Avg net | Median net | Median hold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 5 | 5 | 4 | 3 | 20.00% | 0.10% | -2.30% | 2.0h |
| 2021 | 4 | 4 | 4 | 3 | 0.00% | -2.30% | -2.30% | 12.0h |
| 2022 | 28 | 28 | 28 | 13 | 35.71% | 1.99% | -2.30% | 5.5h |
| 2023 | 5 | 5 | 5 | 3 | 0.00% | -2.30% | -2.30% | 2.0h |
| 2024 | 43 | 43 | 41 | 11 | 46.51% | 3.28% | -2.30% | 10.0h |
| 2025 | 62 | 62 | 60 | 18 | 48.39% | 3.51% | -2.30% | 10.0h |
| 2026 | 24 | 24 | 24 | 11 | 37.50% | 2.20% | -2.30% | 7.0h |
| ALL | 171 | 171 | 144 | 62 | 40.94% | 2.61% | -2.30% | 8.0h |

## Current survivors vs historical/delisted symbols

| Status | Trades | Symbols | Target rate | Avg net |
|---|---:|---:|---:|---:|
| CURRENT | 127 | 109 | 44.09% | 2.99% |
| DELISTED_OR_NOT_CURRENT | 44 | 35 | 31.82% | 1.52% |

## Early-history check

| Period | Signals | Trades | Symbols | Target rate | Avg net |
|---|---:|---:|---:|---:|---:|
| 2019_2022_DEVELOPMENT_ERA | 37 | 37 | 36 | 29.73% | 1.27% |
| 2023_PLUS | 134 | 134 | 117 | 44.03% | 2.98% |

## Overlap

- Trades overlapping an already-open trade: **52.6%**.
- Maximum observed simultaneous open trades: **20**.

## Annual-reset 1,000 USDT portfolio

| Year | Sleeves | Ending equity | Return | Accepted | Capacity skipped | Max realised DD |
|---|---:|---:|---:|---:|---:|---:|
| 2020 | 1 | 1,023.03 | 2.30% | 4 | 1 | -6.74% |
| 2020 | 2 | 1,000.02 | 0.00% | 5 | 0 | -4.62% |
| 2020 | 3 | 1,000.01 | 0.00% | 5 | 0 | -3.13% |
| 2021 | 1 | 932.57 | -6.74% | 3 | 1 | -6.74% |
| 2021 | 2 | 954.79 | -4.52% | 4 | 0 | -4.52% |
| 2021 | 3 | 969.86 | -3.01% | 4 | 0 | -3.01% |
| 2022 | 1 | 1,412.78 | 41.28% | 20 | 8 | -18.89% |
| 2022 | 2 | 1,293.97 | 29.40% | 28 | 0 | -11.72% |
| 2022 | 3 | 1,195.98 | 19.60% | 28 | 0 | -7.81% |
| 2023 | 1 | 932.57 | -6.74% | 3 | 2 | -6.74% |
| 2023 | 2 | 943.55 | -5.64% | 5 | 0 | -5.64% |
| 2023 | 3 | 962.37 | -3.76% | 5 | 0 | -3.76% |
| 2024 | 1 | 1,022.02 | 2.20% | 14 | 29 | -10.98% |
| 2024 | 2 | 1,126.58 | 12.66% | 20 | 23 | -6.44% |
| 2024 | 3 | 1,161.43 | 16.14% | 26 | 17 | -6.03% |
| 2025 | 1 | 1,228.69 | 22.87% | 26 | 36 | -16.99% |
| 2025 | 2 | 1,229.91 | 22.99% | 32 | 30 | -12.10% |
| 2025 | 3 | 1,302.67 | 30.27% | 36 | 26 | -6.02% |
| 2026 | 1 | 910.22 | -8.98% | 14 | 10 | -20.76% |
| 2026 | 2 | 991.00 | -0.90% | 16 | 8 | -10.74% |
| 2026 | 3 | 1,017.92 | 1.79% | 18 | 6 | -7.24% |

## Continuous 1,000 USDT compound simulation

| Sleeves | Ending equity | Total return | Accepted | Capacity skipped | Max realised DD |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,436.76 | 43.68% | 84 | 87 | -29.46% |
| 2 | 1,588.04 | 58.80% | 110 | 61 | -19.65% |
| 3 | 1,695.65 | 69.57% | 122 | 49 | -13.31% |

## Interpretation guardrails

- This test reduces current-symbol survivorship bias by discovering the universe from historical Binance Vision archives, including symbols no longer currently listed.
- It is still not a pristine prospective test because the signal was originally developed using overlapping historical periods.
- Liquidity is analysed descriptively; no new Round 02 liquidity cutoff was fitted.
- Portfolio capacity ordering is deterministic for same-time signals and should not be mistaken for an optimised selector.
- No result from this round is promoted to the live strategy automatically.
