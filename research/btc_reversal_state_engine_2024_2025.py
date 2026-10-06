from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from binance_data import load_symbol
from btc_forensic_transition_2024 import build_indicators
from daily_return_harvest_round2_event_discovery import classify_paths, FEE_RATE, SLIPPAGE_RATE

OUT = Path("research/results/daily_return_harvest/btc_reversal_state_engine_2024_2025")
OUT.mkdir(parents=True, exist_ok=True)

SYMBOL = "BTCUSDT"
INTERVAL = "1h"
LOAD_START = "2023-11-01"
LOAD_END = "2025-12-31"
DECISION_START = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
DECISION_END = pd.Timestamp("2025-12-28 23:00:00", tz="UTC")
STARTING_CAPITAL = 1000.0

# Exploratory engine designed from the already-inspected 2024/2025 cross-year forensic findings.
# Conditions are evaluated afresh each hour. Nothing stays latched after its underlying condition turns false.
ENGINE_VARIANTS = {
    "LOOSE": {"setup_min": 3, "transition_min": 2, "fast_min": 1, "total_min": 7},
    "BALANCED": {"setup_min": 4, "transition_min": 2, "fast_min": 2, "total_min": 9},
    "STRICT": {"setup_min": 5, "transition_min": 3, "fast_min": 2, "total_min": 11},
}
PRIMARY_VARIANT = "BALANCED"


def bool01(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(bool).astype(np.int8)


def add_engine_conditions(x: pd.DataFrame) -> pd.DataFrame:
    z = x.copy()

    # Slow/current setup state. These can remain active for many hours, but only while currently true.
    setup = {
        "S_rsi6_oversold": z.rsi6 < 35,
        "S_bb_depressed": z.bb_z < -1.0,
        "S_stoch_oversold": z.stoch_k < 25,
        "S_cci_oversold": z.cci20 < -100,
        "S_below_vwap24": z.close_vs_vwap24 < -0.005,
        "S_below_ema24": z.close_vs_ema24 < -0.005,
        "S_below_ema72": z.close_vs_ema72 < -0.005,
    }

    # Medium-speed deterioration/transition measures. These explicitly encode how the setup is evolving.
    ema_gap_d1 = z.ema_gap_6_24 - z.ema_gap_6_24.shift(1)
    vwap24_d3 = z.close_vs_vwap24 - z.close_vs_vwap24.shift(3)
    ema24_d3 = z.close_vs_ema24 - z.close_vs_ema24.shift(3)
    macd_hist_d1 = z.macd_hist_pct - z.macd_hist_pct.shift(1)
    atr24_d3 = z.atr24_pct - z.atr24_pct.shift(3)
    transition = {
        "M_ema_gap_worsening": ema_gap_d1 < -0.0010,
        "M_vwap_dislocation_3h": vwap24_d3 < -0.0030,
        "M_ema24_dislocation_3h": ema24_d3 < -0.0030,
        "M_macd_worsening": (z.macd_hist_pct < 0) & (macd_hist_d1 < -0.00005),
        "M_fast_trend_down": z.ema6_slope_3h < -0.0015,
        "M_atr_accelerating": atr24_d3 > 0.00020,
    }

    # Immediate capitulation/participation trigger evidence.
    range_med24 = z.range_pct.shift(1).rolling(24, min_periods=12).median()
    fast = {
        "F_red_impulse": z.body_pct < -0.0035,
        "F_close_near_low": z.close_location < 0.30,
        "F_range_expansion": z.range_pct > (range_med24 * 1.20),
        "F_trade_participation": z.trades_rel72 > 1.0,
        "F_volume_participation": z.volume_rel72 > 1.0,
        "F_seller_dominance": z.taker_share < 0.49,
    }

    for name, cond in {**setup, **transition, **fast}.items():
        z[name] = bool01(cond)

    setup_cols = list(setup)
    transition_cols = list(transition)
    fast_cols = list(fast)
    z["setup_score"] = z[setup_cols].sum(axis=1).astype(np.int8)
    z["transition_score"] = z[transition_cols].sum(axis=1).astype(np.int8)
    z["fast_score"] = z[fast_cols].sum(axis=1).astype(np.int8)
    z["total_score"] = (z.setup_score + z.transition_score + z.fast_score).astype(np.int8)

    for variant, cfg in ENGINE_VARIANTS.items():
        armed = (z.setup_score >= cfg["setup_min"]) & (z.transition_score >= cfg["transition_min"])
        trig = armed & (z.fast_score >= cfg["fast_min"]) & (z.total_score >= cfg["total_min"])
        setup_state = z.setup_score >= max(2, cfg["setup_min"] - 1)
        state = np.select([trig, armed, setup_state], ["TRIGGERED", "ARMED", "SETUP"], default="NEUTRAL")
        z[f"state_{variant}"] = state
        z[f"trigger_{variant}"] = trig
        prev = trig.shift(1, fill_value=False)
        gap = z.time.diff().ne(pd.Timedelta(hours=1)).fillna(True)
        z[f"signal_{variant}"] = trig & ((~prev) | gap)

    return z


def oco_outcome(x: pd.DataFrame, decision_idx: int) -> dict:
    entry_idx = decision_idx + 1
    if entry_idx >= len(x):
        return {"entry_idx": np.nan, "entry_time": pd.NaT, "exit_idx": np.nan, "exit_time": pd.NaT,
                "oco_result": "NO_ENTRY", "oco_net_return": np.nan, "holding_hours": np.nan}
    raw = float(x.open.iloc[entry_idx])
    if not np.isfinite(raw) or raw <= 0:
        return {"entry_idx": entry_idx, "entry_time": x.time.iloc[entry_idx], "exit_idx": np.nan, "exit_time": pd.NaT,
                "oco_result": "NO_ENTRY", "oco_net_return": np.nan, "holding_hours": np.nan}

    entry_cash = raw * (1.0 + SLIPPAGE_RATE) * (1.0 + FEE_RATE)
    exit_factor = (1.0 - SLIPPAGE_RATE) * (1.0 - FEE_RATE)
    target_raw = entry_cash * 1.03 / exit_factor
    stop_raw = entry_cash * 0.98 / exit_factor

    for j in range(entry_idx, len(x)):
        hit_t = float(x.high.iloc[j]) >= target_raw
        hit_s = float(x.low.iloc[j]) <= stop_raw
        if hit_t and hit_s:
            return {"entry_idx": entry_idx, "entry_time": x.time.iloc[entry_idx], "exit_idx": j,
                    "exit_time": x.time.iloc[j], "oco_result": "STOP_SAME_CANDLE", "oco_net_return": -0.02,
                    "holding_hours": int(j - entry_idx)}
        if hit_s:
            return {"entry_idx": entry_idx, "entry_time": x.time.iloc[entry_idx], "exit_idx": j,
                    "exit_time": x.time.iloc[j], "oco_result": "STOP", "oco_net_return": -0.02,
                    "holding_hours": int(j - entry_idx)}
        if hit_t:
            return {"entry_idx": entry_idx, "entry_time": x.time.iloc[entry_idx], "exit_idx": j,
                    "exit_time": x.time.iloc[j], "oco_result": "TARGET", "oco_net_return": 0.03,
                    "holding_hours": int(j - entry_idx)}

    # End-of-data mark only. No forced time exit is imposed during the study.
    final_idx = len(x) - 1
    final_cash_per_unit = float(x.close.iloc[final_idx]) * exit_factor
    ret = final_cash_per_unit / entry_cash - 1.0
    return {"entry_idx": entry_idx, "entry_time": x.time.iloc[entry_idx], "exit_idx": final_idx,
            "exit_time": x.time.iloc[final_idx], "oco_result": "END_MARK", "oco_net_return": float(ret),
            "holding_hours": int(final_idx - entry_idx)}


def build_signals(x: pd.DataFrame, paths: pd.DataFrame, variant: str) -> pd.DataFrame:
    mask = (
        x[f"signal_{variant}"].fillna(False)
        & x.time.between(DECISION_START, DECISION_END)
        & paths.path_valid.eq(1)
    )
    rows = []
    condition_cols = [c for c in x.columns if c.startswith(("S_", "M_", "F_"))]
    for i in np.flatnonzero(mask.to_numpy()):
        o = oco_outcome(x, int(i))
        row = {
            "variant": variant,
            "decision_idx": int(i),
            "decision_time": x.time.iloc[i],
            "year": int(x.time.iloc[i].year),
            "state": x[f"state_{variant}"].iloc[i],
            "setup_score": int(x.setup_score.iloc[i]),
            "transition_score": int(x.transition_score.iloc[i]),
            "fast_score": int(x.fast_score.iloc[i]),
            "total_score": int(x.total_score.iloc[i]),
            "w3_2_24h": int(paths.w3_2_24h.iloc[i] == 1),
            "target3_hour_24h_label": paths.target3_hour.iloc[i],
            "stop2_hour_24h_label": paths.stop2_hour.iloc[i],
            **o,
        }
        for c in condition_cols:
            row[c] = int(x[c].iloc[i])
        rows.append(row)
    return pd.DataFrame(rows)


def replay_portfolio(signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if signals.empty:
        return pd.DataFrame(), signals.copy(), {"starting_capital": STARTING_CAPITAL, "final_balance": STARTING_CAPITAL}

    s = signals.sort_values("decision_idx").reset_index(drop=True).copy()
    cash = STARTING_CAPITAL
    active_exit_idx = -1
    trades = []
    equity_points = [{"time": DECISION_START, "balance": cash, "kind": "START"}]
    status = []

    for r in s.itertuples():
        # If the prior trade exited during or before the completed decision candle, cash is available now.
        available = active_exit_idx <= int(r.decision_idx)
        if not available:
            status.append("MISSED_ALREADY_POSITIONED")
            continue

        status.append("TAKEN")
        start_cash = cash
        ret = float(r.oco_net_return)
        cash = cash * (1.0 + ret)
        active_exit_idx = int(r.exit_idx)
        trades.append({
            "decision_time": r.decision_time,
            "entry_time": r.entry_time,
            "exit_time": r.exit_time,
            "entry_idx": int(r.entry_idx),
            "exit_idx": int(r.exit_idx),
            "year": int(r.year),
            "start_balance": start_cash,
            "net_return": ret,
            "end_balance": cash,
            "oco_result": r.oco_result,
            "holding_hours": r.holding_hours,
            "w3_2_24h": int(r.w3_2_24h),
        })
        equity_points.append({"time": r.exit_time, "balance": cash, "kind": r.oco_result})

    s["portfolio_status"] = status
    trades_df = pd.DataFrame(trades)
    eq = pd.DataFrame(equity_points).sort_values("time").reset_index(drop=True)
    eq["peak"] = eq.balance.cummax()
    eq["drawdown"] = eq.balance / eq.peak - 1.0

    taken = s[s.portfolio_status.eq("TAKEN")]
    missed = s[s.portfolio_status.ne("TAKEN")]
    summary = {
        "starting_capital": STARTING_CAPITAL,
        "final_balance": float(cash),
        "total_return_pct": float((cash / STARTING_CAPITAL - 1.0) * 100),
        "max_drawdown_pct": float(eq.drawdown.min() * 100),
        "total_signals": int(len(s)),
        "taken_trades": int(len(taken)),
        "missed_signals": int(len(missed)),
        "signal_w3_precision": float(s.w3_2_24h.mean()) if len(s) else np.nan,
        "taken_w3_precision": float(taken.w3_2_24h.mean()) if len(taken) else np.nan,
        "missed_w3_precision": float(missed.w3_2_24h.mean()) if len(missed) else np.nan,
        "taken_target_rate_eventual": float(taken.oco_result.eq("TARGET").mean()) if len(taken) else np.nan,
        "taken_stop_rate_eventual": float(taken.oco_result.str.startswith("STOP").mean()) if len(taken) else np.nan,
        "median_holding_hours": float(taken.holding_hours.median()) if len(taken) else np.nan,
    }
    return trades_df, s, summary


def year_breakdown(signals_with_status: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in [2024, 2025]:
        s = signals_with_status[signals_with_status.year.eq(year)]
        t = trades[trades.year.eq(year)] if len(trades) else pd.DataFrame()
        m = s[s.portfolio_status.eq("MISSED_ALREADY_POSITIONED")] if len(s) else s
        rows.append({
            "year": year,
            "signals": int(len(s)),
            "w3_wins": int(s.w3_2_24h.sum()) if len(s) else 0,
            "w3_precision": float(s.w3_2_24h.mean()) if len(s) else np.nan,
            "taken_trades": int(s.portfolio_status.eq("TAKEN").sum()) if len(s) else 0,
            "missed_signals": int(len(m)),
            "missed_w3_precision": float(m.w3_2_24h.mean()) if len(m) else np.nan,
            "taken_mean_return": float(t.net_return.mean()) if len(t) else np.nan,
            "taken_target_rate": float(t.oco_result.eq("TARGET").mean()) if len(t) else np.nan,
            "taken_stop_rate": float(t.oco_result.str.startswith("STOP").mean()) if len(t) else np.nan,
            "median_holding_hours": float(t.holding_hours.median()) if len(t) else np.nan,
        })
    return pd.DataFrame(rows)


def state_diagnostics(x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mask = x.time.between(DECISION_START, DECISION_END)
    for variant in ENGINE_VARIANTS:
        z = x.loc[mask, f"state_{variant}"]
        for state, n in z.value_counts().items():
            rows.append({"variant": variant, "state": state, "hours": int(n), "share": float(n / len(z))})
    return pd.DataFrame(rows)


def main():
    print("Loading BTC 2024-2025 hourly history", flush=True)
    raw = load_symbol(SYMBOL, INTERVAL, LOAD_START, LOAD_END)
    if len(raw) < 18000:
        raise RuntimeError(f"Insufficient BTC history: {len(raw)}")

    x = build_indicators(raw)
    x = add_engine_conditions(x)
    paths = classify_paths(x)

    # Ensure decisions have a valid next-hour entry and the standard 24h path label.
    eligible = x.time.between(DECISION_START, DECISION_END) & paths.path_valid.eq(1)
    print(f"Eligible decision hours: {int(eligible.sum())}", flush=True)

    state_diagnostics(x).to_csv(OUT / "state_hour_distribution.csv", index=False)

    variant_rows = []
    primary_signals = None
    for variant in ENGINE_VARIANTS:
        signals = build_signals(x, paths, variant)
        days = max(1.0, (DECISION_END - DECISION_START).total_seconds() / 86400.0)
        variant_rows.append({
            "variant": variant,
            "signals": int(len(signals)),
            "signals_per_day": float(len(signals) / days),
            "w3_wins": int(signals.w3_2_24h.sum()) if len(signals) else 0,
            "w3_precision": float(signals.w3_2_24h.mean()) if len(signals) else np.nan,
            "eventual_target_rate": float(signals.oco_result.eq("TARGET").mean()) if len(signals) else np.nan,
            "eventual_stop_rate": float(signals.oco_result.str.startswith("STOP").mean()) if len(signals) else np.nan,
            "median_holding_hours": float(signals.holding_hours.median()) if len(signals) else np.nan,
        })
        if variant == PRIMARY_VARIANT:
            primary_signals = signals

    variants = pd.DataFrame(variant_rows)
    variants.to_csv(OUT / "variant_sensitivity.csv", index=False)

    if primary_signals is None:
        raise RuntimeError("Primary variant not built")

    trades, signals_status, portfolio = replay_portfolio(primary_signals)
    signals_status.to_csv(OUT / "signals_balanced.csv", index=False)
    trades.to_csv(OUT / "portfolio_trades.csv", index=False)
    years = year_breakdown(signals_status, trades)
    years.to_csv(OUT / "year_breakdown.csv", index=False)

    # Active-condition prevalence at signals helps identify what actually drove triggers.
    condition_cols = [c for c in signals_status.columns if c.startswith(("S_", "M_", "F_"))]
    cond_rows = []
    for c in condition_cols:
        cond_rows.append({"condition": c, "active_share_all_signals": float(signals_status[c].mean()) if len(signals_status) else np.nan,
                          "active_share_winners": float(signals_status.loc[signals_status.w3_2_24h.eq(1), c].mean()) if signals_status.w3_2_24h.eq(1).any() else np.nan,
                          "active_share_losers": float(signals_status.loc[signals_status.w3_2_24h.eq(0), c].mean()) if signals_status.w3_2_24h.eq(0).any() else np.nan})
    pd.DataFrame(cond_rows).to_csv(OUT / "signal_condition_prevalence.csv", index=False)

    summary = {
        "study": "BTC dynamic reversal state engine 2024-2025",
        "status": "RESEARCH ONLY - exploratory/in-sample; no live strategy or automation changes",
        "primary_variant": PRIMARY_VARIANT,
        "engine_variants": ENGINE_VARIANTS,
        "conditions_recomputed_every_hour": True,
        "sticky_conditions": False,
        "entry": "next completed-hour open",
        "oco": "+3% / -2% net of modelled costs; remains open until first touch or end-of-data mark",
        "fee_rate_each_side": FEE_RATE,
        "slippage_rate_each_side": SLIPPAGE_RATE,
        "starting_capital": STARTING_CAPITAL,
        "portfolio": portfolio,
        "2026_touched": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    md = [
        "# BTC Dynamic Reversal State Engine: 2024-2025",
        "",
        "**RESEARCH ONLY. Exploratory/in-sample: 2024 and 2025 were already used to discover the fingerprint. No live strategy or automation changes. 2026 was not read.**",
        "",
        "## Engine",
        "",
        "Every condition is recomputed independently each hour. If a condition turns false it stops contributing immediately. Signals occur only on a rising edge into TRIGGERED.",
        "",
        f"Primary variant: **{PRIMARY_VARIANT}** with setup >= {ENGINE_VARIANTS[PRIMARY_VARIANT]['setup_min']}, transition >= {ENGINE_VARIANTS[PRIMARY_VARIANT]['transition_min']}, fast >= {ENGINE_VARIANTS[PRIMARY_VARIANT]['fast_min']}, total >= {ENGINE_VARIANTS[PRIMARY_VARIANT]['total_min']}.",
        "",
        "## Variant sensitivity",
        "",
        variants.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## $1,000 continuous portfolio replay",
        "",
        f"- Starting capital: **${STARTING_CAPITAL:,.2f}**",
        f"- Final balance: **${portfolio['final_balance']:,.2f}**",
        f"- Total return: **{portfolio['total_return_pct']:.2f}%**",
        f"- Max realised-equity drawdown: **{portfolio['max_drawdown_pct']:.2f}%**",
        f"- Signals: **{portfolio['total_signals']}**; taken: **{portfolio['taken_trades']}**; missed while already positioned: **{portfolio['missed_signals']}**",
        f"- All-signal W3 precision: **{portfolio['signal_w3_precision']:.2%}**",
        f"- Taken-signal W3 precision: **{portfolio['taken_w3_precision']:.2%}**",
        f"- Missed-signal W3 precision: **{portfolio['missed_w3_precision']:.2%}**" if np.isfinite(portfolio.get('missed_w3_precision', np.nan)) else "- Missed-signal W3 precision: n/a",
        f"- Eventual target rate on taken trades: **{portfolio['taken_target_rate_eventual']:.2%}**",
        f"- Eventual stop rate on taken trades: **{portfolio['taken_stop_rate_eventual']:.2%}**",
        f"- Median holding time: **{portfolio['median_holding_hours']:.1f}h**",
        "",
        "## Calendar-year breakdown",
        "",
        years.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Portfolio rule: the whole available balance is committed to each taken trade. A new signal while still positioned is not traded, but its standalone OCO/W3 outcome is retained in `signals_balanced.csv`. Same-candle target/stop ambiguity is conservatively treated as a stop.",
    ]
    (OUT / "ANALYSIS.md").write_text("\n".join(md))
    print("Completed state-engine study", flush=True)


if __name__ == "__main__":
    main()
