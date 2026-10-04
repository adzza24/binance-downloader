# Daily Return Harvest - Round 1

**Status:** RESEARCH ONLY. No live strategy or automation changes.

Round 1 is a descriptive short-horizon edge study. It does not optimise exits or simulate a 2,000 USDT portfolio yet.

## What was tested

- Current top 120 eligible Binance USDT spot pairs by current 24h quote volume, with historical per-signal liquidity >= 1,000,000 USDT/day.
- Hourly completed-candle decisions; execution proxy is the next hourly open plus 0.05% entry slippage.
- Three fixed, predeclared signal families: MOMENTUM, PULLBACK_RECLAIM and FRESH_EXPANSION.
- Forward path at 1/3/6/12/24/48 hours, including MFE, MAE and net mark-to-market return after 0.10% fees/side and 0.05% slippage/side.
- Target-before-stop diagnostics use conservative hourly ordering: if target and stop are both touched in the same hourly candle, the stop is counted first.
- Splits: 2021-2023 discovery, 2024-2025 validation, 2026 holdout for this research round.

## Holdout headline

- Best family by median 24h net mark-to-market return: **FRESH_EXPANSION**.
- Holdout signals: 1,172; median 24h net return -0.95%; positive 24h rate 38.1%.
- Median 24h MFE 2.17%; median 24h MAE -2.84%; +5% reached within 48h on 32.7% of signals.
- Strongest inspected holdout target/stop ordering among the 2/3/5% targets and 1.5/2/3% stops: MOMENTUM, target 2.0%, stop 3.0%, resolved win rate 57.6%.
- Taking only the highest fixed-score candidate each signal hour produced 4,109 holdout observations across 243 days; median 24h net mark-to-market return -1.41%.

## Interpretation guardrails

- The 5%/day objective is not used to select or tune these rules. It remains a later portfolio-level acceptance test.
- This round is not a trade backtest: overlapping signals are allowed and capital constraints are ignored.
- Current-liquidity universe selection creates survivorship/current-selection bias in older history. Any promising edge should be re-tested with a point-in-time universe before promotion.
- One-hour bars cannot determine intrabar ordering. Round 2 should use 15m path data for shortlisted cohorts before exit optimisation.
- The 2026 holdout is consumed once these results are inspected. Any subsequent tuning must use a new walk-forward/held-out period or nested validation.

## Next gate

Advance a family only if validation and holdout both show positive net expectancy, useful target-before-stop asymmetry, adequate signal frequency, and no obvious dependence on a single BTC/breadth regime. Round 2 should then validate execution paths and exits; portfolio simulation comes after that.
