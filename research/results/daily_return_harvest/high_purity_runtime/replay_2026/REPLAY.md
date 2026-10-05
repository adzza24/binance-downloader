# High-Purity A/B/C 2026 Full-Stream Portfolio Replay

Research-only. No live strategy or portfolio was changed.

## Design

- Score every eligible historical coin-hour in the frozen Round 2 universe with high-purity-abc-v1.
- New signal = rising edge from non-signal to signal for that symbol.
- Two sleeves starting at 1,000 units each; no GBP/USDT FX modelling.
- Enter at next-hour open with 0.10% fee + 0.05% slippage per side.
- OCO remains +3% net target / -2% net stop with no forced 24h close.
- If both sleeves are occupied and a new signal arrives, oldest position aged >=24h is closed at that hour's open and replaced; otherwise new signal is skipped.
- No predictive ranking. Same-hour overflow is resolved by deterministic symbol order.
- Same-candle target+stop ambiguity is conservatively a stop.
- Survivors at data end are marked to market.

## Result

```json
{
  "period": "2026-01-01 through available Round 2 data ending 2026-09-30",
  "starting_capital": 2000.0,
  "sleeves": 2,
  "starting_sleeve_size": 1000.0,
  "raw_new_signal_onsets": 1344,
  "raw_w3_rate": 0.28794642857142855,
  "entries": 856,
  "closed_trades": 856,
  "targets": 252,
  "stops": 589,
  "replacements_after_24h": 13,
  "end_marks": 2,
  "skipped_no_capacity_under_24h": 399,
  "skipped_same_symbol_already_held": 89,
  "final_balance": 270.1086893141416,
  "net_profit": -1729.8913106858583,
  "total_return_pct": -86.49456553429292,
  "max_drawdown_pct": -88.22590126051921,
  "mean_closed_trade_return_pct": -0.48501009386430205,
  "median_closed_trade_return_pct": -1.9999999999999907,
  "median_holding_hours": 4.0
}
```
