# BASE +12h Strategy Research - Round 01 Intention

**Status:** Research only. This round must not modify the Crypto Live Strategy Specification, live portfolio/log data, or any automation.

## Intended research scope

Evaluate the previously identified **BASE-family +12h checkpoint** signal as a standalone research strategy over the broadest real Binance 1h dataset already available to this repository, across every usable calendar year and every usable symbol in the restored hourly-data cache.

The frozen strategy under test is:

- precursor/candidate pipeline: the same causal Round 8 pipeline that produced the Round 11 subgroup;
- candidate family: `BASE`;
- signal checkpoint: `12h`;
- entry: signal-hour close;
- exit: first touch of **+10% target** or **-2% stop**;
- same-candle ambiguity: conservative stop-first ordering;
- round-trip trading-cost assumption: **0.30%**, matching the preceding research programme.

No rule, threshold, family definition, checkpoint condition, entry level, target, stop or cost assumption may be tuned from this Round 01 result.

## Intended outcome

Establish whether the BASE +12h result remains materially positive when moved away from the restricted/current-liquidity sample and replayed across a substantially broader symbol universe and longer history.

The round should answer:

1. How many BASE +12h signals occur by year and symbol?
2. What are the target rate, stop rate, average/median trade return and distribution of returns by year?
3. How consistent is performance across market regimes/years rather than only in aggregate?
4. How concentrated are returns by symbol, week and clustered market episode?
5. How frequently do trades overlap?
6. Is one portfolio sleeve sufficient, or does concurrency justify two or three sleeves?
7. Starting from **$1,000 at the beginning of each calendar year**, what ending equity, return, drawdown and trade utilisation result under 1-, 2- and 3-sleeve capacity models?
8. Does the broader replay support further research, or does the apparent edge collapse outside the narrower Round 11 population?

## Methodology

### Universe

- Restore the repository Binance 1h cache at `research/cache/binance/`.
- Enumerate **every symbol with cached `1h` monthly files** rather than selecting the current top 100/200 by liquidity.
- Exclude obvious stablecoins, leveraged-token patterns and symbols already excluded by the existing research constants where those exclusions can be applied consistently.
- Use only real Binance spot hourly candles available through the existing `research/binance_data.py` loader/cache.
- A symbol/year becomes eligible only when enough prior history exists to calculate the frozen detector/features causally.

### Time range

- Use every calendar year represented by usable cached hourly data.
- Allow lookback candles from the preceding year/months where required for causal feature calculation.
- Do not manufacture missing history.
- Report partial first/last years explicitly when data coverage is incomplete.

### Signal generation

Reuse the frozen causal machinery from the existing explosive-move research:

1. Round 3 frozen family detector.
2. Round 5 90% precursor intersection filter.
3. 96h same-symbol candidate dedupe.
4. Round 7 checkpoint-evolution rules.
5. Retain only signals whose `candidate_family == BASE` and whose first accepted checkpoint is `signal_checkpoint_h == 12`.

The signal must be generated without consulting future returns or the old explosive-event catalogue.

### Trade replay

For every qualifying signal:

- enter at the signal candle close;
- from the following hour onward, exit at first touch of +10% or -2%;
- if both are touched within the same 1h candle, count the stop first;
- subtract 0.30% round-trip cost from each executed trade;
- preserve exact entry/exit timestamps, holding hours, gross/net return, symbol and year;
- where the historical dataset ends before an exit, mark the trade unresolved rather than inventing an outcome.

### Overlap and portfolio simulation

Start each calendar year with **$1,000** and reset to $1,000 on 1 January so year-on-year strategy performance is directly comparable rather than dominated by cross-year compounding.

Run three capacity models in parallel:

- **1 sleeve:** one position at a time; the sleeve starts with the full $1,000 and compounds after each closed trade. Signals arriving while occupied are skipped for capacity.
- **2 sleeves:** two independent sleeves starting at $500 each; each sleeve compounds its own realised P&L. New signals are assigned to a free sleeve; if both are occupied the signal is skipped.
- **3 sleeves:** three independent sleeves starting at $333.33 each; same assignment/capacity rules.

Annual ending portfolio equity is the sum of sleeve equities. Profit/loss is carried forward between trades inside the same calendar year. No leverage or borrowing is allowed.

The purpose of testing all three is to measure the cost of missed overlapping opportunities versus dilution from splitting capital. Round 01 will not choose the sleeve count in advance.

### Metrics

At minimum report, overall and by year:

- symbols scanned / symbols usable;
- signals and distinct symbols/weeks;
- target/stop/unresolved counts;
- win rate;
- average and median net return per trade;
- return quartiles;
- median holding time;
- simultaneous-open-position distribution and maximum concurrency;
- fraction of signals overlapping another open trade;
- 1/2/3-sleeve executed/skipped trades;
- annual starting/ending equity and annual return;
- maximum portfolio drawdown;
- contribution by symbol and concentration of P&L;
- sensitivity of conclusions to years and symbol cohorts, without changing the strategy rules.

## Input knowledge

This round starts from the Round 11 finding that the following subgroup was materially positive in the prior current-top-200 replay:

`candidate family = BASE; signal checkpoint = +12h; straight entry; OCO +10% / -2%`

Prior observed results were approximately:

- 2024: 15 trades, +3.30% average net/trade;
- 2025: 23 trades, +3.96%;
- 2026: 11 trades, +3.15%;
- all: 49 trades, +3.58% average net/trade.

These prior results are **input knowledge only** and are not to be re-optimised in this round.

## Assumptions

- Hourly OHLC data is sufficient for conservative target/stop ordering; same-candle ties are stop-first.
- 0.30% round-trip cost is retained for comparability with preceding rounds.
- Entry at signal close is assumed executable at that price for research purposes.
- Each sleeve is fully deployed on its accepted trade; no leverage.
- Portfolio capital is reset to $1,000 at the start of each calendar year.
- When a sleeve is occupied, a new signal cannot use that sleeve until the prior trade exits.
- The historical Binance cache is the authoritative data universe for this round, not today's exchange ranking.

## Limitations

- This is **not a pristine holdout**. The detector/rules and BASE +12h subgroup were developed after inspecting portions of 2019-26 history.
- Cached-symbol availability can introduce survivorship/listing bias and may not include every asset that was tradable historically.
- Hourly candles do not reveal intra-candle path ordering; stop-first treatment is deliberately conservative.
- No order-book depth, spread, market impact or symbol-specific execution constraints are modelled beyond the 0.30% round-trip cost assumption.
- A broad historical replay can measure robustness but cannot prove future profitability.
- Signals can cluster during market-wide events, so trade count must not be interpreted as independent sample count.

## Expected outputs

All Round 01 outputs belong under `research/base12_strategy/round-01/`.

Expected files:

- `INTENTION.md` - this pre-test research contract;
- `signals.csv.gz` - qualifying BASE +12h signals;
- `trades.csv.gz` - exact trade paths/outcomes;
- `year_summary.csv` - signal/trade statistics by year;
- `overlap_summary.csv` - concurrency/overlap diagnostics;
- `portfolio_years.csv` - 1/2/3-sleeve annual portfolio results;
- `portfolio_trades.csv.gz` - accepted/skipped signal ledger for portfolio simulations;
- `symbol_summary.csv` - performance/contribution by symbol;
- `data_coverage.csv` - usable cached history by symbol;
- `failures.csv` - symbols/data segments that could not be processed;
- `ANALYSIS.md` - completed interpretation and conclusion;
- optional checkpoint files for resumability if the replay is computationally heavy.

## References / data / file locations

- Existing hourly loader/cache: `research/binance_data.py`, `research/cache/binance/`
- Full-population causal replay implementation: `research/explosive_move_real_world_replay_round8.py`
- Frozen Round 5 filter rules: `research/results/explosive_move_intersection_veto_round5/selected_rules.csv`
- Frozen Round 7 checkpoint rules: `research/results/explosive_move_candidate_evolution_round7/evolution_frontier_rules.csv`
- Round 11 research conclusion: `research/results/explosive_move_path_routing_round11/ANALYSIS.md`
- Prior broader real-world signals/results: `research/results/explosive_move_real_world_replay_round8/`

## Governance

This round is exploratory research only. Results must remain in research records and must **not** be written into the Crypto Live Strategy Specification unless a separate exact diff is proposed, reviewed and explicitly approved by the user under the project's standing governance rule.
