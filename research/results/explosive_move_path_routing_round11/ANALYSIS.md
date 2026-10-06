# Explosive Move Path Routing Round 11

**RESEARCH ONLY. No live strategy, portfolio, log or automation changes.**

## Question

Can information available when the Round 8/9 signal fires distinguish three subsequent 30-day path types well enough to route entries differently?

- `DIRECT_PLUS10`: +10% reached before -5%.
- `DIP_THEN_PLUS10`: -5% reached first, then +10% within 30 days.
- `NO_PLUS10_30D`: +10% not reached within 30 days.

The mature full-population replay contains 463 signals: 165 DIRECT, 198 DIP, 100 NO.

Predictors exclude all forward-path fields and use only signal-time/candidate-time information already stored by Round 9: detector features, candidate pre-signal MFE/MAE, candidate family/checkpoint and contemporaneous market-signal breadth. Current liquidity rank/replay-tier were excluded from the primary models because they are based on the current universe rather than historical state.

## Forward three-class classification

| Train | Test | Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---|---:|---:|---:|
| 2024 | 2025 | Logistic | 34.7% | 33.0% | 32.9% |
| 2024 | 2025 | Random forest | **38.3%** | **35.2%** | **35.2%** |
| 2024 | 2025 | Extra Trees | 31.5% | 32.5% | 31.4% |
| 2024-25 | 2026 | Logistic | **37.3%** | **37.5%** | **36.7%** |
| 2024-25 | 2026 | Random forest | 34.7% | 31.9% | 30.8% |
| 2024-25 | 2026 | Extra Trees | 33.3% | 34.3% | 33.5% |

This is weak. Balanced random classification is 33.3%. The majority-class accuracy is 44.1% in 2025 and 40.0% in 2026, so the multiclass models do not provide reliable path labels.

One-vs-rest discrimination is similarly weak in the 2025 forward test. No class/model combination materially separates the classes. In 2026 some relationships strengthen, including a Logistic NO-vs-rest AUC around 0.73, but that was not stable in 2025 and should not be treated as a robust rule.

## Forward routing performance

Hard routing maps predicted DIRECT to straight entry, predicted DIP to the best action learned from prior years, and predicted NO to no trade.

### 2025, models trained only on 2024

- Always straight +10/-5: **-0.232% per signal**.
- Always -5% limit then +10/-5: **-0.200% per signal**.
- Best model, random-forest hard routing: **+0.054% per signal**, 77 trades, **+0.155% per executed trade**.

The 2024 DIP class did not support a positive limit-route expectation, so the prior-data action map correctly learned to skip predicted DIP and NO signals in 2025.

### 2026, models trained on 2024-25

- Always straight +10/-5: **+0.700% per signal**.
- Always -5% limit then +10/-5: **+0.018% per signal**.
- Random-forest hard routing: **+0.619% per signal**, 54 trades, **+0.860% per executed trade**.
- Extra Trees hard routing produced +1.329% per signal in 2026, but it had been negative in the 2025 forward test and therefore is not an ex-ante selected result.

Thus the routing model helped avoid some 2025 losses, but the model that looked best in 2025 did **not** beat always-straight in 2026. The effect is too small and unstable to justify a general three-class router.

## OCO choice by true path class

This section uses the future-known path class only to understand the economic ceiling; it is not deployable.

- DIRECT: straight +10/-5 is consistently best and returns +9.7% net whenever the class label is known correctly.
- NO: no-trade is best overall.
- DIP is regime-dependent. In 2024 no-trade was best. In 2025 -5% limit/+10/-5 averaged +1.74% within the true DIP class. In 2026 -5% limit/+5/-5 was the best tested DIP action at +0.77%.
- Using 2024-25 to choose the DIP action selected -5% limit/+10/-3 for 2026; on the true 2026 DIP class it then averaged **-0.05%**. So even perfect DIP classification would not give us a stable fixed dip OCO yet.

Using all-period hindsight, the path-class oracle (straight +10/-5 for DIRECT, -5% limit/+10/-3 for DIP, skip NO) is about **+3.56% per original signal**. That is an oracle ceiling, not a causal strategy.

## Stable contemporaneous commonalities

### DIRECT

The clearest repeated commonality is **market-wide signal clustering / participation**:

- signals in previous 6h: median **4 DIRECT vs 2 other**, one-vs-rest separation AUC 0.615; direction consistent in 2024/25/26.
- previous-24h signal count: 6 vs 3, AUC 0.589.
- 24h volume ratio: 1.60 vs 1.44, AUC 0.581.
- 6h trade ratio: 1.55 vs 1.23, AUC 0.574.

A simple `signals_prev6h >= 4` subset has 182 signals and straight +10/-5 averages **+1.79% net** overall: +2.10% in 2024, +1.29% in 2025 and +3.27% in 2026. However those 182 signals occur in only 17 distinct weeks, so this is largely a clustered broad-market condition rather than a weekly selector.

### DIP

DIP is the hardest class. Stable single-feature AUCs are only about 0.52-0.54. No convincing causal separator was found. That is the main obstacle to choosing the -5% limit route reliably.

### NO +10

NO signals tend to have lower market-signal breadth, weaker ATR acceleration and less severe pre-signal deterioration. The strongest stable individual separator is again previous-6h breadth (median 1 vs 2, AUC 0.594), but it is not strong enough for reliable rejection on its own.

## Most interesting validated subgroup: BASE checkpoint 12

An interpretable rule search used 2024 as discovery and 2025 as validation. The strongest simple direct-entry condition that remained positive in both periods was:

> **candidate family = BASE; signal checkpoint = +12h; enter at signal; OCO +10% / -2%.**

Results:

| Period | Signals | Distinct weeks | Target rate | Avg net / trade |
|---|---:|---:|---:|---:|
| 2024 discovery | 15 | 6 | 46.7% | **+3.30%** |
| 2025 validation | 23 | 9 | 52.2% | **+3.96%** |
| 2026 confirmation | 11 | 6 | 45.5% | **+3.15%** |
| All | 49 | 21 | 49.0% | **+3.58%** |

This is much stronger and more stable than the general signal. On a $300 research trade, +3.58% is about **$10.73 average net per accepted trade**. It is still below the project's ~$30 / 10% economic hurdle, and 49 trades are not independent because crypto signals cluster, but it is the first path-routing subgroup in this programme with a materially positive result in discovery, validation and confirmation.

A broader `BASE` rule with +10/-2 retains 121 signals across 47 distinct weeks and is positive in every year, but expectancy falls to +1.37% overall (+1.57% / +1.51% / +0.81% by year).

## Conclusion

1. **A general three-way DIRECT/DIP/NO classifier does not work well enough with the current signal-time features.** Out-of-year classification is close to random and generic ML routing does not reliably beat always-straight.
2. **DIRECT has identifiable structure**, especially broad-market signal clustering and BASE-family behaviour.
3. **DIP vs NO remains unresolved.** The currently available features do not tell us reliably when a -5% resting limit is the right route.
4. **BASE +12h is a genuine lead worth isolating.** Straight entry with +10/-2 produced +3.30%, +3.96%, and +3.15% average net in 2024, 2025 and 2026 respectively.
5. The research should not be promoted to the live strategy. The subgroup needs further independent validation, especially because 2024-26 has been repeatedly inspected in this research programme and is not pristine holdout data.
