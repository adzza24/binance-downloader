from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from binance_data import load_symbol
from round2b_validation import add_live_features, controlled_activity
from round3a_exit_architecture import net_pnl, capped_stop

# All variants inherit the winning architecture up to +30%:
# structural stop capped at 10%, +5% -> BE, +30% -> +10% floor.
# There is NO maximum holding period. A trade runs until its strategy exit
# fires or the historical dataset ends. Dataset-end positions remain OPEN.
VARIANTS = {
    "OPEN_BASE_T30_L10": {},
    "MILESTONES_A_TP1000": {
        "floors": [(1.00, 0.50), (2.00, 1.00), (5.00, 3.00)],
        "take_profit": 10.00,
    },
    "MILESTONES_B_TP1000": {
        "floors": [(1.00, 0.30), (2.00, 0.75), (5.00, 2.50)],
        "take_profit": 10.00,
    },
    "TP500_FULL": {"take_profit": 5.00},
    "TP1000_FULL": {"take_profit": 10.00},
    "P200_TRAIL40": {"trail_trigger": 2.00, "trail_pct": 0.40},
    "P200_TRAIL50": {"trail_trigger": 2.00, "trail_pct": 0.50},
    "P200_TRAIL60": {"trail_trigger": 2.00, "trail_pct": 0.60},
    "P500_L300_THEN_TRAIL50_TP1000": {
        "floors": [(5.00, 3.00)],
        "trail_trigger": 5.00,
        "trail_pct": 0.50,
        "take_profit": 10.00,
    },
}

THRESHOLDS = [0.30, 0.50, 1.00, 2.00, 3.00, 5.00, 10.00]


def finish(sig, variant, exit_time, reason, pnl, initial_stop, stop, activated,
           peak, trough, exit_ret, bars_held, floor_level, trail_active, is_open):
    entry = float(sig["entry_price"])
    mfe = peak / entry - 1
    mae = trough / entry - 1
    return {
        "signal_id": sig["signal_id"], "symbol": sig["symbol"], "variant": variant,
        "entry_time": sig["entry_time"], "exit_time": exit_time, "exit_reason": reason,
        "pnl_usdt": pnl, "return_pct": pnl / 300.0,
        "initial_stop_pct": initial_stop / entry - 1,
        "final_stop_pct": stop / entry - 1,
        "activated": activated, "floor_level": floor_level,
        "trail_active": trail_active, "is_open": is_open,
        "mfe_pct": mfe, "mae_pct": mae,
        "exit_price_return_pct": exit_ret,
        "peak_giveback_pct": max(0.0, mfe - exit_ret),
        "bars_held": bars_held,
    }


def simulate(df, sig, variant, spec, cfg):
    start = int(sig["entry_index"])
    end = len(df)
    entry = float(sig["entry_price"])
    initial_stop = capped_stop(entry, float(sig["structural_stop"]), 0.10)
    stop = initial_stop
    activated = False
    floor30_done = False
    floor_level = 0.0
    trail_active = False
    peak = entry
    trough = entry

    for j in range(start, end):
        bar = df.iloc[j]
        low, high, close = float(bar.low), float(bar.high), float(bar.close)
        peak = max(peak, high)
        trough = min(trough, low)

        if not activated:
            if low <= initial_stop:
                pnl = net_pnl(entry, [(1.0, initial_stop)], cfg)
                return finish(sig, variant, bar.time, "STOP", pnl, initial_stop, initial_stop,
                              False, peak, trough, initial_stop / entry - 1, j-start+1,
                              0.0, False, False)
            if high >= entry * 1.05:
                activated = True
                stop = entry
                continue
        else:
            # Conservative hourly ordering: existing stop acts before a newly
            # reached milestone in the same bar; new rules apply afterwards.
            if low <= stop:
                pnl = net_pnl(entry, [(1.0, stop)], cfg)
                if trail_active:
                    reason = "PEAK_TRAIL_STOP"
                elif floor_level > 0:
                    reason = "PROFIT_FLOOR_STOP"
                else:
                    reason = "BREAKEVEN_STOP"
                return finish(sig, variant, bar.time, reason, pnl, initial_stop, stop,
                              True, peak, trough, stop / entry - 1, j-start+1,
                              floor_level, trail_active, False)

            tp = spec.get("take_profit")
            if tp is not None and high >= entry * (1 + tp):
                px = entry * (1 + tp)
                pnl = net_pnl(entry, [(1.0, px)], cfg)
                return finish(sig, variant, bar.time, "TAKE_PROFIT", pnl, initial_stop, stop,
                              True, peak, trough, tp, j-start+1,
                              floor_level, trail_active, False)

            if not floor30_done and high >= entry * 1.30:
                stop = max(stop, entry * 1.10)
                floor30_done = True
                floor_level = max(floor_level, 0.10)

            for trigger, lock in spec.get("floors", []):
                if high >= entry * (1 + trigger) and floor_level < lock:
                    stop = max(stop, entry * (1 + lock))
                    floor_level = lock

            trig = spec.get("trail_trigger")
            trail_pct = spec.get("trail_pct")
            if trig is not None and high >= entry * (1 + trig):
                trail_active = True
            if trail_active:
                candidate = peak * (1 - trail_pct)
                candidate = min(candidate, close * 0.999)
                stop = max(stop, entry * 1.10, candidate)

    # Dataset ended before a strategy exit. Mark to market but keep OPEN.
    last = df.iloc[-1]
    close = float(last.close)
    pnl = net_pnl(entry, [(1.0, close)], cfg)
    return finish(sig, variant, last.time, "DATA_END_OPEN", pnl, initial_stop, stop,
                  activated, peak, trough, close / entry - 1, end-start,
                  floor_level, trail_active, True)


def process_symbol(symbol, btc, cfg):
    df = btc.copy() if symbol == "BTCUSDT" else load_symbol(symbol, cfg["interval"], cfg["start"], cfg["end"])
    if len(df) < 800:
        return [], []
    x = add_live_features(df, btc)
    sigs = controlled_activity(symbol, x, cfg)
    rows = []
    for sig in sigs:
        for variant, spec in VARIANTS.items():
            rows.append(simulate(df, sig, variant, spec, cfg))
    return sigs, rows


def summarise(trades):
    rows, years = [], []
    q = trades.copy()
    q["year"] = pd.to_datetime(q.entry_time).dt.year
    for variant, g in q.groupby("variant"):
        closed = g[~g.is_open]
        opened = g[g.is_open]
        pc = closed.pnl_usdt.to_numpy(float)
        wc, lc = pc[pc > 0], pc[pc < 0]
        mtm_total = g.pnl_usdt.sum()
        rows.append({
            "variant": variant,
            "trades": len(g),
            "closed_trades": len(closed),
            "open_trades": len(opened),
            "open_rate": len(opened) / len(g),
            "realised_net_pnl_usdt": closed.pnl_usdt.sum(),
            "open_mtm_pnl_usdt": opened.pnl_usdt.sum(),
            "total_mtm_pnl_usdt": mtm_total,
            "realised_expectancy_usdt_per_closed": pc.mean() if len(pc) else np.nan,
            "mtm_expectancy_usdt_per_signal": mtm_total / len(g),
            "realised_median_pnl_usdt": np.median(pc) if len(pc) else np.nan,
            "realised_win_rate": (pc > 0).mean() if len(pc) else np.nan,
            "realised_profit_factor": wc.sum()/abs(lc.sum()) if len(lc) and lc.sum() else math.inf,
            "avg_hold_days_all": g.bars_held.mean()/24,
            "avg_open_hold_days": opened.bars_held.mean()/24 if len(opened) else 0.0,
            "tp_rate": (g.exit_reason == "TAKE_PROFIT").mean(),
            "trail_stop_rate": (g.exit_reason == "PEAK_TRAIL_STOP").mean(),
            "floor_stop_rate": (g.exit_reason == "PROFIT_FLOOR_STOP").mean(),
            "avg_mfe_pct": g.mfe_pct.mean(),
            "avg_peak_giveback_pct": g.peak_giveback_pct.mean(),
        })
        for y, gy in g.groupby("year"):
            gc = gy[~gy.is_open]
            go = gy[gy.is_open]
            years.append({
                "variant": variant, "year": int(y), "trades": len(gy),
                "closed_trades": len(gc), "open_trades": len(go),
                "realised_net_pnl_usdt": gc.pnl_usdt.sum(),
                "open_mtm_pnl_usdt": go.pnl_usdt.sum(),
                "total_mtm_pnl_usdt": gy.pnl_usdt.sum(),
                "open_rate": len(go)/len(gy),
            })
    return pd.DataFrame(rows), pd.DataFrame(years)


def main():
    cfg = json.loads(Path("research/config.json").read_text())
    out = Path("research/results/round3e_open"); out.mkdir(parents=True, exist_ok=True)
    btc = load_symbol("BTCUSDT", cfg["interval"], cfg["start"], cfg["end"])
    all_s, all_t = [], []
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut = {ex.submit(process_symbol, s, btc, cfg): s for s in cfg["symbols"]}
        for f in as_completed(fut):
            symbol = fut[f]
            try:
                sigs, rows = f.result(); all_s += sigs; all_t += rows
                print(symbol, len(sigs), len(rows), flush=True)
            except Exception as e:
                print("ERROR", symbol, repr(e), flush=True)

    signals = pd.DataFrame(all_s); trades = pd.DataFrame(all_t)
    signals.to_csv(out / "signals.csv", index=False)
    trades.to_csv(out / "trades.csv", index=False)
    if trades.empty:
        raise RuntimeError("Round 3E open-ended produced no trades")

    summary, yearly = summarise(trades)
    summary.to_csv(out / "summary.csv", index=False)
    yearly.to_csv(out / "year_summary.csv", index=False)
    trades[trades.is_open].to_csv(out / "open_positions.csv", index=False)

    base = trades[trades.variant == "OPEN_BASE_T30_L10"].copy()
    thresh_rows = []
    for t in THRESHOLDS:
        hits = base.mfe_pct >= t
        thresh_rows.append({"threshold_pct": int(t*100), "trades": len(base), "hits": int(hits.sum()), "hit_rate": float(hits.mean())})
    pd.DataFrame(thresh_rows).to_csv(out / "threshold_summary.csv", index=False)

    manifest = {
        "study": "Round 3E open-ended large-winner exit architecture",
        "frozen_entry": "CONTROLLED_ACTIVITY from Round 2B",
        "inherited_base": "$300; structural stop capped at 10%; +5% -> breakeven; +30% -> +10% floor",
        "holding_period": "No maximum holding period. Each trade runs until strategy exit or the final available historical candle.",
        "data_end_handling": "Positions surviving to the final candle remain OPEN. Their mark-to-market PnL is reported separately and is not counted as a strategy exit.",
        "variants": VARIANTS,
        "same_bar_rule": "existing stop acts before a newly reached floor/trail milestone in the same hourly candle; new rules apply after that bar",
        "round4_tabled": "Research causal hostile/non-tradable conditions, especially 2022 and 2026, using only timestamp-available inputs.",
        "position_usdt": cfg["position_usdt"], "fees": cfg["fee_rate"], "slippage": cfg["slippage_rate"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary.sort_values(["total_mtm_pnl_usdt", "realised_profit_factor"], ascending=False).to_string(index=False))

if __name__ == "__main__":
    main()
