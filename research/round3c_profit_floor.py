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

# All variants keep the Round 3B winner architecture fixed:
# structural stop capped at 10%, +5% moves whole position to breakeven,
# no continuous trail, 720h max holding period.
# Ratchets are (trigger_return, locked_return) pairs.
RATCHETS = {
    "BE_ONLY": [],
    "T15_L5": [(0.15, 0.05)],
    "T20_L5": [(0.20, 0.05)],
    "T20_L10": [(0.20, 0.10)],
    "T30_L10": [(0.30, 0.10)],
    "T30_L15": [(0.30, 0.15)],
    "T15_L5_T30_L15": [(0.15, 0.05), (0.30, 0.15)],
    "T20_L5_T40_L20": [(0.20, 0.05), (0.40, 0.20)],
}


def finish(sig: dict, variant: str, exit_time, reason: str, pnl: float,
           initial_stop: float, final_stop: float, activated: bool,
           ratchet_level: int, mfe: float, mae: float, exit_ret: float,
           ambiguous: int) -> dict:
    giveback = max(0.0, mfe - exit_ret)
    capture = exit_ret / mfe if mfe > 0 else np.nan
    return {
        "signal_id": sig["signal_id"],
        "symbol": sig["symbol"],
        "variant": variant,
        "entry_time": sig["entry_time"],
        "exit_time": exit_time,
        "exit_reason": reason,
        "pnl_usdt": pnl,
        "return_pct": pnl / 300.0,
        "initial_stop": initial_stop,
        "initial_stop_pct": initial_stop / sig["entry_price"] - 1,
        "final_stop": final_stop,
        "final_stop_pct": final_stop / sig["entry_price"] - 1,
        "activated": activated,
        "ratchet_level": ratchet_level,
        "ambiguous_bars": ambiguous,
        "mfe_pct": mfe,
        "mae_pct": mae,
        "exit_price_return_pct": exit_ret,
        "peak_giveback_pct": giveback,
        "mfe_capture_ratio": capture,
    }


def simulate(df: pd.DataFrame, sig: dict, variant: str, cfg: dict) -> dict:
    ratchets = RATCHETS[variant]
    start = int(sig["entry_index"])
    end = min(len(df), start + int(cfg["max_holding_hours"]) + 1)
    entry = float(sig["entry_price"])
    initial_stop = capped_stop(entry, float(sig["structural_stop"]), 0.10)
    activation = entry * 1.05
    stop = initial_stop
    activated = False
    ratchet_level = 0
    max_high = entry
    min_low = entry
    ambiguous = 0

    for j in range(start, end):
        bar = df.iloc[j]
        low, high = float(bar.low), float(bar.high)
        max_high = max(max_high, high)
        min_low = min(min_low, low)
        mfe = max_high / entry - 1
        mae = min_low / entry - 1

        if not activated:
            hit_s = low <= initial_stop
            hit_a = high >= activation
            if hit_s and hit_a:
                ambiguous += 1
            if hit_s:
                pnl = net_pnl(entry, [(1.0, initial_stop)], cfg)
                return finish(sig, variant, bar.time, "STOP", pnl,
                              initial_stop, initial_stop, False, 0, mfe, mae,
                              initial_stop / entry - 1, ambiguous)
            if hit_a:
                activated = True
                stop = entry
                # As in Round 3B, activation takes effect after this hourly bar.
                continue
        else:
            # Existing stop is checked before any new milestone reached in this bar.
            if low <= stop:
                pnl = net_pnl(entry, [(1.0, stop)], cfg)
                reason = "RATCHET_STOP" if ratchet_level else "BREAKEVEN_STOP"
                return finish(sig, variant, bar.time, reason, pnl,
                              initial_stop, stop, True, ratchet_level, mfe, mae,
                              stop / entry - 1, ambiguous)

            # Ratchet only after the completed bar demonstrates the milestone.
            while ratchet_level < len(ratchets):
                trigger, lock = ratchets[ratchet_level]
                if high < entry * (1 + trigger):
                    break
                stop = max(stop, entry * (1 + lock))
                ratchet_level += 1

    last = df.iloc[end - 1]
    close = float(last.close)
    max_high = max(max_high, float(last.high))
    min_low = min(min_low, float(last.low))
    mfe = max_high / entry - 1
    mae = min_low / entry - 1
    pnl = net_pnl(entry, [(1.0, close)], cfg)
    return finish(sig, variant, last.time, "TIME", pnl,
                  initial_stop, stop, activated, ratchet_level, mfe, mae,
                  close / entry - 1, ambiguous)


def process_symbol(symbol: str, btc: pd.DataFrame, cfg: dict):
    df = btc.copy() if symbol == "BTCUSDT" else load_symbol(
        symbol, cfg["interval"], cfg["start"], cfg["end"]
    )
    if len(df) < 800:
        return [], []
    x = add_live_features(df, btc)
    sigs = controlled_activity(symbol, x, cfg)
    rows = []
    for sig in sigs:
        for variant in RATCHETS:
            rows.append(simulate(df, sig, variant, cfg))
    return sigs, rows


def summarise(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, years = [], []
    q = trades.copy()
    q["year"] = pd.to_datetime(q.entry_time).dt.year
    for variant, g in q.groupby("variant"):
        p = g.pnl_usdt.to_numpy(float)
        w, l = p[p > 0], p[p < 0]
        activated = g[g.activated == True]
        rows.append({
            "variant": variant,
            "trades": len(g),
            "net_pnl_usdt": p.sum(),
            "expectancy_usdt": p.mean(),
            "median_pnl_usdt": np.median(p),
            "win_rate": (p > 0).mean(),
            "profit_factor": w.sum() / abs(l.sum()) if len(l) and l.sum() else math.inf,
            "avg_loss_usdt": l.mean() if len(l) else 0,
            "worst_loss_usdt": l.min() if len(l) else 0,
            "activation_rate": g.activated.mean(),
            "ratchet_hit_rate": (g.ratchet_level > 0).mean(),
            "top_ratchet_hit_rate": (g.ratchet_level >= len(RATCHETS[variant])).mean() if RATCHETS[variant] else 0.0,
            "avg_mfe_pct": g.mfe_pct.mean(),
            "avg_peak_giveback_pct": g.peak_giveback_pct.mean(),
            "median_peak_giveback_pct": g.peak_giveback_pct.median(),
            "avg_mfe_capture_activated": activated.mfe_capture_ratio.replace([np.inf, -np.inf], np.nan).mean() if len(activated) else np.nan,
            "timeout_rate": (g.exit_reason == "TIME").mean(),
            "ratchet_stop_rate": (g.exit_reason == "RATCHET_STOP").mean(),
            "ambiguous_bars": int(g.ambiguous_bars.sum()),
        })
        for y, gy in g.groupby("year"):
            pp = gy.pnl_usdt.to_numpy(float)
            ww, ll = pp[pp > 0], pp[pp < 0]
            years.append({
                "variant": variant,
                "year": int(y),
                "trades": len(gy),
                "net_pnl_usdt": pp.sum(),
                "expectancy_usdt": pp.mean(),
                "win_rate": (pp > 0).mean(),
                "profit_factor": ww.sum() / abs(ll.sum()) if len(ll) and ll.sum() else math.inf,
                "ratchet_hit_rate": (gy.ratchet_level > 0).mean(),
                "avg_peak_giveback_pct": gy.peak_giveback_pct.mean(),
                "timeout_rate": (gy.exit_reason == "TIME").mean(),
            })
    return pd.DataFrame(rows), pd.DataFrame(years)


def main():
    cfg = json.loads(Path("research/config.json").read_text())
    out = Path("research/results/round3c")
    out.mkdir(parents=True, exist_ok=True)
    btc = load_symbol("BTCUSDT", cfg["interval"], cfg["start"], cfg["end"])
    all_s, all_t = [], []
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut = {ex.submit(process_symbol, s, btc, cfg): s for s in cfg["symbols"]}
        for f in as_completed(fut):
            symbol = fut[f]
            try:
                sigs, rows = f.result()
                all_s.extend(sigs)
                all_t.extend(rows)
                print(symbol, len(sigs), len(rows), flush=True)
            except Exception as e:
                print("ERROR", symbol, repr(e), flush=True)

    signals = pd.DataFrame(all_s)
    trades = pd.DataFrame(all_t)
    signals.to_csv(out / "signals.csv", index=False)
    trades.to_csv(out / "trades.csv", index=False)
    if trades.empty:
        raise RuntimeError("Round 3C produced no trades")

    summary, yearly = summarise(trades)
    summary.to_csv(out / "summary.csv", index=False)
    yearly.to_csv(out / "year_summary.csv", index=False)

    manifest = {
        "study": "Round 3C milestone profit-floor ratchets",
        "frozen_entry": "CONTROLLED_ACTIVITY from Round 2B",
        "frozen_architecture": "$300 full runner; structural stop capped at 10%; +5% moves stop to breakeven; no continuous trailing",
        "frozen_horizon": "720h / 30 days from entry",
        "only_variable": "post-activation milestone trigger and locked-profit floor",
        "variants": RATCHETS,
        "same_bar_rule": "existing stop is evaluated before a newly reached milestone; new profit floor becomes active after that hourly bar",
        "round4_tabled": "Later research hostile/non-tradable regimes, especially 2022 and 2026, using only information available at each historical timestamp.",
        "position_usdt": cfg["position_usdt"],
        "fees": cfg["fee_rate"],
        "slippage": cfg["slippage_rate"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary.sort_values(["net_pnl_usdt", "profit_factor"], ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
