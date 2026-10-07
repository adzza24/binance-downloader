# BASE +12h Strategy Research — Round 02 Intention

**Status:** RESEARCH ONLY. This round must not modify the Crypto Live Strategy Specification, live portfolio, live logs, or any automation.

## Intended research scope

Validate the frozen **BASE +12h** signal on the broadest historical Binance spot USDT universe that can be reconstructed from Binance's public hourly archive, including symbols that are now delisted and therefore absent from the current exchange universe.

The purpose is specifically to test whether the profitability observed in Round 01 survives removal of current-symbol/current-liquidity survivorship bias.

## Frozen strategy under test

No strategy parameter may be optimised in this round.

- Candidate-generation pipeline: the same frozen explosive-move pipeline used by the prior real-world replay.
- Candidate family: **BASE**.
- Accepted signal: the candidate's **first accepted checkpoint is exactly +12 hours**.
- Entry: signal-hour close.
- Exit: OCO **+10% target / -2% stop**.
- Transaction-cost assumption: **0.30% round trip** on executed trades.
- Same-hour target/stop ambiguity: conservatively score the stop first.
- Maximum research holding period: 30 days; any unresolved position is closed/marked to market at the 30-day boundary.
- No parameter, threshold, OCO level, family definition, checkpoint rule, or feature may be changed after results are observed.

## Historical universe methodology

Primary universe:
- Every Binance **spot USDT** symbol for which Binance Vision exposes monthly 1-hour kline archives.
- Include delisted symbols.
- Availability is historical: a symbol can only generate signals during periods for which its archived hourly candles exist.
- Do **not** rank or select symbols using today's exchange status or today's liquidity.
- Exclude obvious stablecoin-base pairs, leveraged-token conventions (UP/DOWN/BULL/BEAR), and identifiable tokenised-equity/ETF instruments. The exclusion logic and excluded symbols must be persisted for audit.

Universe discovery should use Binance Vision archive listings rather than current `exchangeInfo`.

No primary minimum-liquidity threshold will be introduced in Round 02, because that would add a new fitted parameter. Instead, each signal should store contemporaneous quote-volume measures so results can be reported by historical liquidity strata without changing the primary result.

## Data and references

Repository: `adzza24/binance-downloader`, isolated branch `round-3`.

Relevant frozen research inputs:
- `research/explosive_move_discovery_round1.py`
- `research/explosive_move_real_world_replay_round8.py`
- `research/results/explosive_move_intersection_veto_round5/selected_rules.csv`
- `research/results/explosive_move_candidate_evolution_round7/evolution_frontier_rules.csv`
- `research/results/explosive_move_path_routing_round11/`
- `research/base12_strategy/round-01/`

Market data:
- Binance Vision spot monthly 1-hour kline archives.
- BTCUSDT 1-hour history is used where the frozen feature set requires BTC-relative calculations.

## Input knowledge

Round 01 broad-cache replay:
- 60 resolved trades.
- 49 symbols.
- 27 distinct signal weeks.
- 48.33% target rate.
- +3.50% average net return per trade after assumed costs.
- Signals were highly clustered: 46.7% overlapped an already-open trade.

Earlier discovery/path-routing work found approximately +3.58% average net per trade for the same BASE +12h / +10% / -2% rule.

These figures are comparison benchmarks only and must not be used to tune Round 02.

## Portfolio methodology

Trade-level validation is primary.

Portfolio simulations are secondary and should use:
- starting capital **$1,000**;
- annual-reset simulations: reset to $1,000 on 1 January of each tested year, compound realised sleeve capital within the year;
- 1, 2 and 3 equal-capital sleeve models;
- a signal is skipped only when every sleeve is occupied;
- profits/losses are carried forward within the same year;
- additionally report a true continuous $1,000 simulation from first eligible signal to last without annual resets for each sleeve count.

Persist accepted/skipped trades so capital-capacity effects are auditable.

## Methodology and comparisons

Report:
1. Historical-universe inventory: discovered symbols, included symbols, excluded symbols, first/last archived month, and data gaps.
2. Signal count, symbols, distinct weeks, target/stop/time-exit counts, target rate, mean/median net return and holding time.
3. Results by calendar year.
4. Results for delisted/not-current symbols separately where current status can be checked without using it to select the test population.
5. Contemporaneous-liquidity strata based on signal-time rolling quote volume.
6. Overlap/concurrency statistics.
7. Annual-reset $1,000 portfolio results for 1/2/3 sleeves.
8. Continuous $1,000 compound results for 1/2/3 sleeves.
9. Direct comparison with Round 01 benchmarks, without retuning.
10. Pre-2019/early-history results separately if sufficient data exist, because they are temporally outside the main 2019+ development period and are especially informative.

## Expected outputs

All Round 02 outputs belong under:
`research/base12_strategy/round-02/`

Expected files:
- `INTENTION.md` (this document; must pre-date the test implementation/results)
- `historical_universe.csv`
- `excluded_universe.csv`
- `signals.csv.gz`
- `year_summary.csv`
- `liquidity_summary.csv`
- `status_summary.csv`
- `overlap_summary.csv`
- `portfolio_annual.csv`
- `portfolio_continuous.csv`
- `failures.csv`
- `ANALYSIS.md`

Intermediate shard outputs may be stored as workflow artifacts and need not be committed if the combined auditable outputs above are persisted.

## Assumptions

- Binance Vision hourly archives are treated as the historical data source of record for this test.
- Archive presence is a practical proxy for historical Binance spot availability.
- Hourly bars cannot reveal within-candle path ordering; stop-first is deliberately conservative.
- Fee/slippage modelling remains the frozen 0.30% round-trip research assumption.
- Historical quote volume is used only descriptively/for stratification in this round, not to optimise a new entry filter.

## Limitations

- This is still not a pristine prospective test: the strategy was discovered using historical data that overlap much of the test period.
- Delisted-symbol inclusion reduces survivorship bias but cannot eliminate all exchange/listing-selection effects.
- Binance archive coverage may contain gaps or format anomalies.
- Identifying historical tokenised instruments from archive names is imperfect; exclusions must be disclosed.
- Pre-2019 observations, if present, may be few because Binance's USDT market was much smaller.
- The 0.30% cost model is approximate and does not reproduce order-book slippage.
- Cross-sectional historical liquidity rank is not reconstructed in the primary test; rolling signal-time quote volume is used instead.
- Any follow-up optimisation requires a new round with a new intention document. Round 02 itself is validation-only.

## Success criterion

The round does not have a single pass/fail threshold chosen after seeing results. Evidence is stronger if:
- average net return remains clearly positive on the expanded historical universe;
- profitability is not explained solely by current survivors;
- performance is reasonably distributed across years/symbols rather than one cluster;
- delisted-symbol results do not collapse;
- portfolio results remain positive under realistic overlap constraints.

Whatever the outcome, it must be reported without changing the frozen rule.
