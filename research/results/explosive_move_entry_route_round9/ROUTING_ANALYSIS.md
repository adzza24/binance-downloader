# Round 9 Routing Analysis

**RESEARCH ONLY. No live strategy or automation changes.**

## Exact route overlap on 463 mature signals

- Both routes hit target: **35**.
- Straight OCO target, limit route not target: **130**.
- Straight OCO stop, limit route target: **85**.
- Both routes stop: **213**.
- Of the 165 DIRECT_PLUS10 signals, the straight route necessarily targets; the limit route targets only 35, stops 82, remains open 6 and is unfilled 42.
- Of the 198 DIP_THEN_PLUS10 signals, the straight route necessarily stops; the limit route targets 73 and averages **+0.23% net per original signal** in that group.
- Of the 100 NO_PLUS10_30D signals, the straight route stops; the limit route targets 12 and averages **-3.50% net per original signal**.

A perfect future-known routing rule (straight for DIRECT_PLUS10, limit otherwise) would average approximately **+2.80% net per signal**. This is only an oracle ceiling and is not causal or deployable.

## Contemporaneous separation

The strongest single contemporaneous separator between DIRECT_PLUS10 and DIP_THEN_PLUS10 is market signal breadth over the prior 6h:

- DIRECT_PLUS10 median `signals_prev6h`: **4**
- DIP_THEN_PLUS10 median: **2**
- Overall separation AUC: **0.601**
- Direction is consistent in 2024, 2025 and 2026.

Other same-direction features are also weak-to-moderate rather than decisive: 6h trade ratio, same-hour signal breadth, prior-24h signal breadth and 24h volume ratio. No individual feature exceeds about **0.60 AUC** for direct-vs-dip separation.

For the operationally more relevant label "limit route strictly better than straight", there are only **85/463** such cases. The best individual signal-time discriminators are again weak: prior-6h signal breadth (~0.596 AUC), 6h trade ratio (~0.595), and 6h volume ratio (~0.592).

## Simple routing sanity check

A shallow decision tree trained on 2024 to choose the better route selects the straight route for every 2025 signal; training on 2024-2025 likewise selects straight for every 2026 signal at depths 1-2. Deeper trees route a small minority to the limit approach but reduce 2026 average return from the always-straight **+0.70%** to about **+0.30%**. This is exploratory only, but it indicates that the currently stored signal-time features do not yet provide a robust causal way to identify the 85 limit-route rescues.

## Implication

The -5% limit route is useful for a real subset of signals, but applying it indiscriminately destroys too many straight winners. The economic opportunity therefore lies in identifying the rescue subset causally. The current contemporaneous features show some structure, especially broad-market signal clustering and short-term participation, but separation is not yet strong enough to justify a routing rule.
