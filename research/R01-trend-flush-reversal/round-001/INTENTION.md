# R01 — Trend Flush Reversal — Round 001

Status: RESEARCH ONLY. This round must not alter the live strategy specification, live portfolio, live logs, or trading automations.

## Purpose

Test whether a high-precision, causal signal can identify sharp downside flushes with strong rejection in coins that were already on a constructive medium-term trajectory, then capture the subsequent rebound.

The primary hypothesis is that a subset of large lower-wick 4h candles, when combined with trend, volume, support, BTC-regime and post-flush confirmation filters, can reach +3% before -2% with unusually high precision.

This is not a generic falling-knife strategy and must not treat a large drawdown alone as bullish.

## Primary acceptance target

The initial objective is a signal family that, on genuinely out-of-sample data:

- reaches +3.0% before -2.0% in at least 90% of signalled trades;
- produces enough opportunities to be useful, with at least one signal per week across the test universe as the minimum practical frequency;
- preferably approaches one signal per day without weakening precision merely to increase frequency;
- has enough out-of-sample observations to make the quoted precision meaningful. Tiny samples must be reported but must not qualify as accepted.

The 90% figure is a target to test, not a result to engineer. Parameter searching must not tune the confirmation period.

## Secondary objectives

1. Test whether -1.0% or -1.5% invalidation can replace -2.0% without materially reducing useful precision or expectancy.
2. If 90% precision at +3% is not robustly achievable, test whether lower hit-rate variants with larger winners (+4%, +5%, +6% and observed MFE) can deliver stronger expectancy while preserving approximately +3% average gross return per signal.
3. Characterise failure cases, MAE, MFE, time-to-target and performance by BTC regime, coin, year and setup morphology.
4. Determine whether confirmation after the wick materially improves precision relative to entering immediately after the flush candle.

## Universe

Use the existing 40-symbol research universe in `research/config.json` for Round 001. It is intentionally reduced to control GitHub Actions cost while retaining large caps, mid caps and more volatile Binance Spot assets.

Do not choose coins because their later performance is known. Existing account-exclusion lists are not relevant to the statistical hypothesis unless a symbol is unavailable as Binance Spot historical data.

BTCUSDT is also used as market-regime context.

## Historical period

Use hourly Binance Spot candles from 2023-01-01 through 2026-09-30, with earlier warm-up data loaded where needed for moving averages.

Temporal split:

- Development: 2023-01-01 to 2024-06-30
- Validation: 2024-07-01 to 2025-12-31
- Locked confirmation: 2026-01-01 to 2026-09-30

Selection and threshold tuning may use Development and Validation. Locked Confirmation must only be evaluated after candidate rules are selected.

Results must also be segmented by BTC regime rather than assuming one homogeneous market.

## Existing data references

Prefer existing cached Binance monthly 1h files restored by GitHub Actions under `research/cache/binance`.

The repository already contains reusable research infrastructure:

- `research/binance_data.py` — loads Binance monthly klines and caches downloaded ZIP files.
- `research/config.json` — existing 40-symbol 1h universe.
- `research/results/daily_return_harvest/round2_event_discovery/dataset_v1/` — previously persisted canonical research dataset and historical feature material. This may be used for reference or compatible features, but must not replace raw causal scanning where doing so would introduce winner/control sampling bias.

Round 001 should restore the existing Actions Binance cache before downloading missing months.

## Causal event construction

1. Aggregate 1h candles into closed 4h candles.
2. Identify flush candidates using only information available by the close of each 4h candle.
3. Candidate features should include, at minimum:
   - lower-wick share;
   - lower-wick/body ratio;
   - 4h range relative to ATR;
   - decline from prior close/open to candle low;
   - close location within the candle range;
   - current and recent volume ratios;
   - interaction with recent swing lows / support proxies;
   - distance from medium-term trend measures;
   - medium-term trend direction and slope;
   - degree of prior extension to avoid parabolic entries;
   - BTC trend, return and volatility context.
4. The default confirmed-entry variant must wait for post-flush evidence. A confirmation may occur on subsequent closed 1h candles, using a higher low and/or reclaim of the flush midpoint / local level. Entry must occur no earlier than the next tradable bar after confirmation.
5. An immediate-entry control should also be measured.

No feature may use future candles relative to the signal timestamp.

## Outcome definitions

Primary path test after entry:

- target: +3.0% gross price movement;
- stops: -1.0%, -1.5%, -2.0%;
- primary horizon: 24 hours;
- secondary horizon: 48 hours.

For a candle that touches both target and stop and where 1h OHLC cannot establish ordering, classify conservatively as stop-first/ambiguous failure unless lower-timeframe data is explicitly introduced in a later round.

Also record:

- maximum favourable excursion;
- maximum adverse excursion;
- time to target or stop;
- return at timeout;
- simulated net return after the repository's standard research assumption of 0.10% fee plus 0.05% slippage on each side.

## Candidate rule search

Round 001 is a structured threshold search, not an unconstrained model-fitting exercise.

Reasonable threshold families may be explored for:

- wick share and wick/body ratio;
- ATR-normalised flush magnitude;
- close-location recovery;
- relative volume;
- trend filters;
- recent-support proximity;
- BTC market filter;
- confirmation strength and delay.

Rules must be ranked primarily by out-of-sample Validation precision subject to minimum sample/frequency requirements, with Development used for discovery. A small shortlist must then be frozen before Locked Confirmation is evaluated.

Do not select a rule because it looks best on 2026 Confirmation results.

## Frequency and robustness reporting

For every serious candidate report:

- total signals and signals/week;
- wins, losses and ambiguous outcomes;
- +3% before stop precision;
- Wilson 95% confidence interval for precision;
- gross and net expectancy;
- median/mean MAE and MFE;
- median time to outcome;
- per-year results;
- BTC-regime results;
- per-symbol concentration;
- maximum losing streak.

A 90% point estimate with a very small sample must not be described as a 90%-accurate strategy.

## Incremental persistence / restartability

Round 001 must persist work as it progresses so a failed GitHub Action can resume instead of restarting from zero.

Directory layout:

```text
research/R01-trend-flush-reversal/round-001/
  INTENTION.md
  config.json
  run.py
  checkpoints/
    state.json
    symbols/
  results/
    ...
```

Process symbols in small batches. After each completed batch:

1. write/update per-symbol checkpoint output;
2. update `checkpoints/state.json`;
3. commit and push the checkpoint files to the research branch;
4. on restart, skip symbols whose compatible checkpoint is already complete.

Checkpoint/result commits must not retrigger the research workflow.

## Expected outputs

At minimum:

- `results/ANALYSIS.md`
- `results/manifest.json`
- `results/events.csv.gz`
- `results/rule_grid.csv`
- `results/shortlist.csv`
- `results/confirmation_results.csv`
- `results/year_summary.csv`
- `results/regime_summary.csv`
- `results/symbol_summary.csv`
- `results/failures.csv`
- `checkpoints/state.json`

If the primary target fails, `ANALYSIS.md` must state that plainly and identify the strongest robust alternative rather than forcing an accepted rule.

## Non-goals / governance

- Never place trades.
- Never edit `Crypto Live Strategy Specification` or any equivalent live-strategy authority.
- Never automatically promote research findings into live strategy rules.
- Never change the user's real portfolio or protective orders.
- Any future proposal to adopt an R01 rule into the live strategy requires a separate explicit diff and user approval before any live strategy document is changed.
