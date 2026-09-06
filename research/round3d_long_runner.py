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

H30 = 30 * 24
H180 = 180 * 24

VARIANTS = {
    # Audit only: exact Round 3C winner logic on all signals with 30d horizon.
    "AUDIT_30D_T30_L10": {"horizon": H30, "matched180": False, "floor30": True},
    # Matched cohort controls / contenders: only signals with a full 180d future window.
    "MATCHED_30D_T30_L10": {"horizon": H30, "matched180": True, "floor30": True},
    "MATCHED_180D_T30_L10": {"horizon": H180, "matched180": True, "floor30": True},
    "MATCHED_180D_T30_L10_P50_TRAIL10": {"horizon": H180, "matched180": True, "floor30": True, "trail_trigger": 0.50, "trail_pct": 0.10},
    "MATCHED_180D_T30_L10_P50_TRAIL20": {"horizon": H180, "matched180": True, "floor30": True, "trail_trigger": 0.50, "trail_pct": 0.20},
    "MATCHED_180D_T30_L10_P50_TRAIL30": {"horizon": H180, "matched180": True, "floor30": True, "trail_trigger": 0.50, "trail_pct": 0.30},
    "MATCHED_180D_T30_L10_P30_TRAIL20": {"horizon": H180, "matched180": True, "floor30": True, "trail_trigger": 0.30, "trail_pct": 0.20},
    "MATCHED_180D_T30_L10_P30_TRAIL30": {"horizon": H180, "matched180": True, "floor30": True, "trail_trigger": 0.30, "trail_pct": 0.30},
    "MATCHED_TP30_FULL": {"horizon": H180, "matched180": True, "take_profit": 0.30},
}

THRESHOLDS = [0.30, 0.50, 1.00, 2.00, 3.00, 4.00]
CHECKPOINTS = [H30, 60 * 24, 90 * 24, H180]


def finish(sig, variant, exit_time, reason, pnl, initial_stop, stop, activated,
           peak, trough, exit_ret, bars_held, trail_active=False):
    mfe = peak / sig["entry_price"] - 1
    mae = trough / sig["entry_price"] - 1
    return {
        "signal_id": sig["signal_id"], "symbol": sig["symbol"], "variant": variant,
        "entry_time": sig["entry_time"], "exit_time": exit_time, "exit_reason": reason,
        "pnl_usdt": pnl, "return_pct": pnl / 300.0,
        "initial_stop": initial_stop, "initial_stop_pct": initial_stop / sig["entry_price"] - 1,
        "final_stop": stop, "final_stop_pct": stop / sig["entry_price"] - 1,
        "activated": activated, "trail_active": trail_active,
        "mfe_pct": mfe, "mae_pct": mae, "exit_price_return_pct": exit_ret,
        "peak_giveback_pct": max(0.0, mfe - exit_ret), "bars_held": bars_held,
    }


def full_180_available(df, sig):
    return int(sig["entry_index"]) + H180 < len(df)


def path_diagnostics(df, sig):
    start = int(sig["entry_index"])
    entry = float(sig["entry_price"])
    if start + H180 >= len(df):
        return None
    path = df.iloc[start:start + H180 + 1].copy()
    highs = path.high.astype(float).to_numpy()
    lows = path.low.astype(float).to_numpy()
    closes = path.close.astype(float).to_numpy()
    peak_i = int(np.argmax(highs))
    peak = float(highs[peak_i])
    running_peak = np.maximum.accumulate(highs)
    dd = lows / running_peak - 1.0
    out = {
        "signal_id": sig["signal_id"], "symbol": sig["symbol"], "entry_time": sig["entry_time"],
        "mfe_180_pct": peak / entry - 1, "peak_hour": peak_i,
        "max_peak_drawdown_180_pct": float(dd.min()),
    }
    for h in CHECKPOINTS:
        sl = slice(0, h + 1)
        out[f"mfe_{h//24}d_pct"] = float(np.max(highs[sl]) / entry - 1)
        out[f"close_{h//24}d_pct"] = float(closes[h] / entry - 1)
    for t in THRESHOLDS:
        hits = np.flatnonzero(highs >= entry * (1 + t))
        out[f"hit_{int(t*100)}"] = bool(len(hits))
        out[f"first_hit_{int(t*100)}_hour"] = int(hits[0]) if len(hits) else np.nan
    return out


def simulate(df, sig, variant, spec, cfg):
    start = int(sig["entry_index"])
    horizon = int(spec["horizon"])
    end = min(len(df), start + horizon + 1)
    entry = float(sig["entry_price"])
    initial_stop = capped_stop(entry, float(sig["structural_stop"]), 0.10)
    stop = initial_stop
    activated = False
    floor30_done = False
    trail_active = False
    peak = entry
    trough = entry

    for j in range(start, end):
        bar = df.iloc[j]
        low, high = float(bar.low), float(bar.high)
        peak = max(peak, high)
        trough = min(trough, low)

        if not activated:
            hit_stop = low <= initial_stop
            hit_be = high >= entry * 1.05
            if hit_stop:
                pnl = net_pnl(entry, [(1.0, initial_stop)], cfg)
                return finish(sig, variant, bar.time, "STOP", pnl, initial_stop, initial_stop,
                              False, peak, trough, initial_stop / entry - 1, j - start + 1)
            if hit_be:
                activated = True
                stop = entry
                continue
        else:
            if low <= stop:
                pnl = net_pnl(entry, [(1.0, stop)], cfg)
                reason = "PEAK_TRAIL_STOP" if trail_active else ("PROFIT_FLOOR_STOP" if floor30_done else "BREAKEVEN_STOP")
                return finish(sig, variant, bar.time, reason, pnl, initial_stop, stop,
                              True, peak, trough, stop / entry - 1, j - start + 1, trail_active)

            tp = spec.get("take_profit")
            if tp is not None and high >= entry * (1 + tp):
                px = entry * (1 + tp)
                pnl = net_pnl(entry, [(1.0, px)], cfg)
                return finish(sig, variant, bar.time, "TAKE_PROFIT", pnl, initial_stop, stop,
                              True, peak, trough, tp, j - start + 1)

            if spec.get("floor30") and not floor30_done and high >= entry * 1.30:
                stop = max(stop, entry * 1.10)
                floor30_done = True

            trig = spec.get("trail_trigger")
            trail_pct = spec.get("trail_pct")
            if trig is not None and high >= entry * (1 + trig):
                trail_active = True
            if trail_active:
                # Peak trail is evaluated after the completed hourly bar and only ratchets upward.
                candidate = peak * (1 - trail_pct)
                candidate = min(candidate, float(bar.close) * 0.999)
                stop = max(stop, entry * 1.10, candidate)

    last = df.iloc[end - 1]
    close = float(last.close)
    pnl = net_pnl(entry, [(1.0, close)], cfg)
    return finish(sig, variant, last.time, "TIME", pnl, initial_stop, stop, activated,
                  peak, trough, close / entry - 1, end - start)


def process_symbol(symbol, btc, cfg):
    df = btc.copy() if symbol == "BTCUSDT" else load_symbol(symbol, cfg["interval"], cfg["start"], cfg["end"])
    if len(df) < 800:
        return [], [], []
    x = add_live_features(df, btc)
    sigs = controlled_activity(symbol, x, cfg)
    rows, diag = [], []
    for sig in sigs:
        d = path_diagnostics(df, sig)
        if d is not None:
            diag.append(d)
        for variant, spec in VARIANTS.items():
            if spec.get("matched180") and not full_180_available(df, sig):
                continue
            rows.append(simulate(df, sig, variant, spec, cfg))
    return sigs, rows, diag


def summarise(trades):
    rows, years = [], []
    q = trades.copy()
    q["year"] = pd.to_datetime(q.entry_time).dt.year
    for variant, g in q.groupby("variant"):
        p = g.pnl_usdt.to_numpy(float)
        w, l = p[p > 0], p[p < 0]
        rows.append({
            "variant": variant, "trades": len(g), "net_pnl_usdt": p.sum(),
            "expectancy_usdt": p.mean(), "median_pnl_usdt": np.median(p),
            "win_rate": (p > 0).mean(),
            "profit_factor": w.sum() / abs(l.sum()) if len(l) and l.sum() else math.inf,
            "avg_loss_usdt": l.mean() if len(l) else 0, "worst_loss_usdt": l.min() if len(l) else 0,
            "avg_mfe_pct": g.mfe_pct.mean(), "avg_peak_giveback_pct": g.peak_giveback_pct.mean(),
            "timeout_rate": (g.exit_reason == "TIME").mean(), "avg_hold_days": g.bars_held.mean() / 24,
            "trail_stop_rate": (g.exit_reason == "PEAK_TRAIL_STOP").mean(),
        })
        for y, gy in g.groupby("year"):
            pp = gy.pnl_usdt.to_numpy(float); ww, ll = pp[pp > 0], pp[pp < 0]
            years.append({
                "variant": variant, "year": int(y), "trades": len(gy),
                "net_pnl_usdt": pp.sum(), "expectancy_usdt": pp.mean(),
                "win_rate": (pp > 0).mean(),
                "profit_factor": ww.sum() / abs(ll.sum()) if len(ll) and ll.sum() else math.inf,
                "avg_hold_days": gy.bars_held.mean() / 24,
            })
    return pd.DataFrame(rows), pd.DataFrame(years)


def diagnostic_summary(diag):
    rows = []
    for d in [30, 60, 90, 180]:
        s = diag[f"mfe_{d}d_pct"]
        rows.append({"horizon_days": d, "signals": len(diag), "median_mfe": s.median(),
                     "p75_mfe": s.quantile(.75), "p90_mfe": s.quantile(.90),
                     "p95_mfe": s.quantile(.95), "max_mfe": s.max()})
    return pd.DataFrame(rows)


def main():
    cfg = json.loads(Path("research/config.json").read_text())
    out = Path("research/results/round3d"); out.mkdir(parents=True, exist_ok=True)
    btc = load_symbol("BTCUSDT", cfg["interval"], cfg["start"], cfg["end"])
    all_s, all_t, all_d = [], [], []
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut = {ex.submit(process_symbol, s, btc, cfg): s for s in cfg["symbols"]}
        for f in as_completed(fut):
            symbol = fut[f]
            try:
                sigs, rows, diag = f.result(); all_s += sigs; all_t += rows; all_d += diag
                print(symbol, len(sigs), len(rows), len(diag), flush=True)
            except Exception as e:
                print("ERROR", symbol, repr(e), flush=True)

    signals = pd.DataFrame(all_s); trades = pd.DataFrame(all_t); diag = pd.DataFrame(all_d)
    signals.to_csv(out / "signals.csv", index=False); trades.to_csv(out / "trades.csv", index=False)
    diag.to_csv(out / "path_diagnostics_180d.csv", index=False)
    if trades.empty: raise RuntimeError("Round 3D produced no trades")
    summary, yearly = summarise(trades)
    summary.to_csv(out / "summary.csv", index=False); yearly.to_csv(out / "year_summary.csv", index=False)
    if not diag.empty:
        diagnostic_summary(diag).to_csv(out / "mfe_horizon_summary.csv", index=False)
        thresh = []
        for t in THRESHOLDS:
            col = f"hit_{int(t*100)}"
            thresh.append({"threshold_pct": int(t*100), "signals": len(diag), "hits": int(diag[col].sum()), "hit_rate": float(diag[col].mean())})
        pd.DataFrame(thresh).to_csv(out / "threshold_hit_summary.csv", index=False)

    manifest = {
        "study": "Round 3D long-duration runner and loose peak-trail research",
        "frozen_entry": "CONTROLLED_ACTIVITY from Round 2B",
        "base_architecture": "$300; structural stop capped at 10%; +5% -> breakeven; +30% -> +10% fixed profit floor",
        "main_horizon": "180 days / 4320h",
        "matched_cohort": "Primary comparisons only use signals with a complete 180-day future window; later censored signals excluded from matched variants",
        "audit": "AUDIT_30D_T30_L10 uses all signals to reproduce Round 3C winner",
        "variants": VARIANTS,
        "peak_trail": "percentage below highest high since entry; activates only after threshold; never loosens below +10% floor",
        "diagnostics": "30/60/90/180d MFE, close return, 180d peak timing, maximum drawdown from running peak, threshold hit timing",
        "round4_tabled": "Research causal non-tradable/hostile conditions, especially 2022 and 2026, without retrospective labels in live decision logic.",
        "position_usdt": cfg["position_usdt"], "fees": cfg["fee_rate"], "slippage": cfg["slippage_rate"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary.sort_values(["net_pnl_usdt", "profit_factor"], ascending=False).to_string(index=False))

if __name__ == "__main__": main()
