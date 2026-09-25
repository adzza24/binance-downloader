# Early Breakout Round 1 - Signal Cohort & Forward Path Study

**Status:** RESEARCH ONLY. Separate breakout-strategy track. No changes to the Early Breakout Market Watch task or Crypto Live Strategy Specification.

**Dataset:** 1h Binance archive data, 2019-01-01 through 2026-08-31. Existing 40-symbol research universe plus SYN, LSK, G, CELR and ONE where history exists. BTC is used only as the relative-strength benchmark and is excluded as an entry candidate. Known account/task unavailable symbols are excluded.

**Cohort:** 3,939 de-duplicated entry episodes: 3,536 PRE_BREAKOUT and 403 BREAKOUT_ATTEMPT. 3,183 signals have a complete 12-month forward path. Zero symbol-processing failures.

## Forward path

| Horizon | Median MFE | Median MAE | Median end return | +10% hit | +20% hit | +30% hit |
|---|---:|---:|---:|---:|---:|---:|
| 1d | +2.25% | -2.23% | -0.09% | 6.04% | 0.76% | 0.18% |
| 3d | +4.08% | -4.02% | -0.25% | 19.28% | 5.03% | 1.93% |
| 5d | +5.51% | -5.19% | -0.12% | 29.43% | 9.97% | 3.84% |
| 7d | +6.52% | -6.21% | -0.48% | 35.59% | 14.55% | 6.28% |
| 1m | +15.01% | -14.32% | -2.09% | 62.71% | 41.37% | 28.22% |
| 3m | +31.71% | -27.25% | -3.65% | 77.91% | 61.70% | 51.13% |
| 6m | +54.17% | -37.52% | -4.65% | 85.09% | 73.79% | 65.79% |
| 12m | +104.95% | -45.00% | +0.82% | 93.47% | 87.34% | 81.65% |

Long-horizon MFE must not be read as a hold recommendation. Median end return stays weak while adverse excursion and peak giveback become very large.

## Peak timing over complete 12m paths

- 2.83% of 12m peaks occur within 24h.
- 4.46% within 72h.
- 7.98% within 7d.
- 18.13% within 30d.
- 35.60% within 90d.
- 57.18% within 180d.

This shows substantial later trend potential, but only if an exit architecture can avoid the severe drawdowns/giveback seen in unconditional holding.

## Fast-winner adverse excursion

For signals that reached the target **within 7 days**:

| Target | Hits | Median adverse move before hit | 10th percentile adverse move |
|---|---:|---:|---:|
| +10% | 1,406 | -2.15% | -6.92% |
| +20% | 576 | -1.94% | -6.96% |
| +30% | 248 | -2.12% | -7.32% |

Approximate survival of 7-day +20% winners under fixed stops:
- -2% stop: 50.7%
- -3% stop: 63.2%
- -5% stop: 81.6%
- -7% stop: 90.1%
- -10% stop: 96.4%

This makes roughly 3%, 5%, 7% and 10% useful Round 2 stop-sensitivity candidates, with structural/hybrid stops also worth retaining.

## Stage observations

BREAKOUT_ATTEMPT signals are stronger over the intended short-term horizon:
- 7d +10% hit: 42.93% vs 34.75% for PRE_BREAKOUT.
- 7d +20% hit: 19.85% vs 13.95%.
- 1m +20% hit: 48.98% vs 40.51%.

Among PRE_BREAKOUT entries, 660 upgraded to BREAKOUT_ATTEMPT within 72h. Their 7d median MFE was ~10.03% versus ~5.58% for non-upgraded PRE_BREAKOUT entries, suggesting confirmation/upgrade state may be useful for later management research.

## Round 3H exit reference

Applied only as a benchmark to the 3,183 complete-12m signals:
- Combined independent-trade P&L: +36,303.10 USDT using 300 USDT per signal.
- Profit factor: 2.27.
- Win rate: 15.46%.
- Median trade: -0.75 USDT, essentially breakeven-stop friction.
- BREAKOUT_ATTEMPT PF: 2.66; PRE_BREAKOUT PF: 2.22.
- Annual benchmark P&L was positive in 2019, 2020, 2021, 2023 and 2024, negative in 2022 and 2025.
- Results are sums of overlapping independent 300 USDT trades, not a capital-constrained account return.

The benchmark is driven by relatively rare runners and should not be treated as the breakout strategy's final exit architecture.

## Round 2 implication

Do not optimise a giant grid. Use this Round 1 evidence to test:
1. short holding caps (1-7d, then ~30d);
2. stop sensitivity centred around 3%, 5%, 7%, 10% plus structural/hybrid invalidation;
3. fixed +10% and +20% profit-taking;
4. a partial-profit + runner architecture to preserve the long-tail upside;
5. separate PRE_BREAKOUT vs BREAKOUT_ATTEMPT treatment and possibly PRE -> ATTEMPT upgrade management.

The central finding is that the cohort has meaningful breakout/trend optionality, but unconditional long holding is poor because the path is highly volatile and gives back large gains. Round 2 should therefore optimise how to capture the early move while retaining a controlled runner rather than choosing either a pure short-term cap or pure long-term hold.

## Caveats

- This is not a survivorship-free reconstruction of every historical Binance listing. It uses the established research universe plus the Watch's named reference symbols.
- The Watch prompt is qualitative; Round 1 necessarily freezes a research-only mechanical operationalisation documented in the manifest.
- Repeated 72h-de-duplicated episodes can still share the same later market cycle, so long-horizon observations are correlated rather than fully independent.
- No live task or strategy document was edited.
