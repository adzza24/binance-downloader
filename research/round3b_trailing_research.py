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

TRAILS = [
    "NO_TRAIL",
    "LOW_24H",
    "LOW_48H",
    "LOW_72H",
    "LOW_96H",
    "SWING_3X3",
    "ATR24_3X",
    "ATR24_4X",
    "HYBRID_SWING_ATR3",
]


def add_atr(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    prev = x.close.shift(1)
    tr = pd.concat([
        x.high - x.low,
        (x.high - prev).abs(),
        (x.low - prev).abs(),
    ], axis=1).max(axis=1)
    x["atr24"] = tr.rolling(24).mean()
    return x


def confirmed_swing_low(df: pd.DataFrame, j: int, start: int) -> float | None:
    # A 3-left / 3-right pivot is only considered once all 3 right bars have closed.
    p = j - 3
    if p < max(start + 3, 3):
        return None
    left = df.low.iloc[p-3:p]
    right = df.low.iloc[p+1:p+4]
    if len(left) < 3 or len(right) < 3:
        return None
    lp = float(df.low.iloc[p])
    if lp < float(left.min()) and lp <= float(right.min()):
        return lp * 0.997
    return None


def trail_candidate(df: pd.DataFrame, j: int, start: int, activation_i: int,
                    trail: str, swing_support: float | None) -> tuple[float | None, float | None]:
    if trail == "NO_TRAIL":
        return None, swing_support

    if trail.startswith("LOW_"):
        hours = int(trail.split("_")[1].replace("H", ""))
        left = max(start, j - hours)
        if j <= left:
            return None, swing_support
        return float(df.low.iloc[left:j].min()) * 0.997, swing_support

    new_swing = confirmed_swing_low(df, j, start)
    if new_swing is not None:
        swing_support = max(swing_support or -math.inf, new_swing)

    if trail == "SWING_3X3":
        return swing_support, swing_support

    atr = float(df.atr24.iloc[j]) if np.isfinite(df.atr24.iloc[j]) else np.nan
    if not np.isfinite(atr):
        return swing_support if trail == "HYBRID_SWING_ATR3" else None, swing_support
    hh = float(df.high.iloc[activation_i:j+1].max())

    if trail == "ATR24_3X":
        return hh - 3.0 * atr, swing_support
    if trail == "ATR24_4X":
        return hh - 4.0 * atr, swing_support
    if trail == "HYBRID_SWING_ATR3":
        atr_stop = hh - 3.0 * atr
        return max(atr_stop, swing_support) if swing_support is not None else atr_stop, swing_support
    raise ValueError(trail)


def finish(sig: dict, trail: str, exit_time, reason: str, pnl: float, stop: float,
           mfe: float, mae: float, exit_ret: float, activated: bool, ambiguous: int) -> dict:
    giveback = max(0.0, mfe - exit_ret)
    capture = exit_ret / mfe if mfe > 0 else np.nan
    return {
        "signal_id": sig["signal_id"],
        "symbol": sig["symbol"],
        "trail": trail,
        "entry_time": sig["entry_time"],
        "exit_time": exit_time,
        "exit_reason": reason,
        "pnl_usdt": pnl,
        "return_pct": pnl / 300.0,
        "initial_stop": stop,
        "initial_stop_pct": stop / sig["entry_price"] - 1,
        "activated": activated,
        "ambiguous_bars": ambiguous,
        "mfe_pct": mfe,
        "mae_pct": mae,
        "exit_price_return_pct": exit_ret,
        "peak_giveback_pct": giveback,
        "mfe_capture_ratio": capture,
    }


def simulate(df: pd.DataFrame, sig: dict, trail: str, cfg: dict) -> dict:
    start = int(sig["entry_index"])
    end = min(len(df), start + int(cfg["max_holding_hours"]) + 1)
    entry = float(sig["entry_price"])
    stop = capped_stop(entry, float(sig["structural_stop"]), 0.10)
    activation = entry * 1.05
    activated = False
    activation_i = -1
    runner_stop = stop
    swing_support = None
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
            hit_s = low <= stop
            hit_a = high >= activation
            if hit_s and hit_a:
                ambiguous += 1
            if hit_s:
                pnl = net_pnl(entry, [(1.0, stop)], cfg)
                return finish(sig, trail, bar.time, "STOP", pnl, stop, mfe, mae,
                              stop / entry - 1, False, ambiguous)
            if hit_a:
                activated = True
                activation_i = j
                runner_stop = entry
                continue
        else:
            if low <= runner_stop:
                pnl = net_pnl(entry, [(1.0, runner_stop)], cfg)
                return finish(sig, trail, bar.time, "RUNNER_STOP", pnl, stop, mfe, mae,
                              runner_stop / entry - 1, True, ambiguous)

            candidate, swing_support = trail_candidate(
                df, j, start, activation_i, trail, swing_support
            )
            if candidate is not None and np.isfinite(candidate):
                # Never loosen, never move below breakeven, and do not set a stop above
                # the just-completed close (which would imply an impossible fill).
                candidate = min(float(candidate), float(bar.close) * 0.999)
                runner_stop = max(runner_stop, entry, candidate)

    last = df.iloc[end - 1]
    max_high = max(max_high, float(last.high))
    min_low = min(min_low, float(last.low))
    mfe = max_high / entry - 1
    mae = min_low / entry - 1
    close = float(last.close)
    pnl = net_pnl(entry, [(1.0, close)], cfg)
    return finish(sig, trail, last.time, "TIME", pnl, stop, mfe, mae,
                  close / entry - 1, activated, ambiguous)


def process_symbol(symbol: str, btc: pd.DataFrame, cfg: dict):
    raw = btc.copy() if symbol == "BTCUSDT" else load_symbol(
        symbol, cfg["interval"], cfg["start"], cfg["end"]
    )
    if len(raw) < 800:
        return [], []
    x = add_live_features(raw, btc)
    sigs = controlled_activity(symbol, x, cfg)
    df = add_atr(raw)
    rows = []
    for sig in sigs:
        for trail in TRAILS:
            rows.append(simulate(df, sig, trail, cfg))
    return sigs, rows


def summarise(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, years = [], []
    q = trades.copy()
    q["year"] = pd.to_datetime(q.entry_time).dt.year
    for trail, g in q.groupby("trail"):
        p = g.pnl_usdt.to_numpy(float)
        w, l = p[p > 0], p[p < 0]
        activated = g[g.activated == True]
        rows.append({
            "trail": trail,
            "trades": len(g),
            "net_pnl_usdt": p.sum(),
            "expectancy_usdt": p.mean(),
            "median_pnl_usdt": np.median(p),
            "win_rate": (p > 0).mean(),
            "profit_factor": w.sum() / abs(l.sum()) if len(l) and l.sum() else math.inf,
            "avg_loss_usdt": l.mean() if len(l) else 0,
            "worst_loss_usdt": l.min() if len(l) else 0,
            "activation_rate": g.activated.mean(),
            "avg_mfe_pct": g.mfe_pct.mean(),
            "avg_peak_giveback_pct": g.peak_giveback_pct.mean(),
            "median_peak_giveback_pct": g.peak_giveback_pct.median(),
            "avg_mfe_capture_activated": activated.mfe_capture_ratio.replace([np.inf,-np.inf],np.nan).mean() if len(activated) else np.nan,
            "timeout_rate": (g.exit_reason == "TIME").mean(),
            "ambiguous_bars": int(g.ambiguous_bars.sum()),
        })
        for y, gy in g.groupby("year"):
            pp = gy.pnl_usdt.to_numpy(float)
            ww, ll = pp[pp > 0], pp[pp < 0]
            years.append({
                "trail": trail, "year": int(y), "trades": len(gy),
                "net_pnl_usdt": pp.sum(), "expectancy_usdt": pp.mean(),
                "win_rate": (pp > 0).mean(),
                "profit_factor": ww.sum() / abs(ll.sum()) if len(ll) and ll.sum() else math.inf,
                "avg_peak_giveback_pct": gy.peak_giveback_pct.mean(),
                "timeout_rate": (gy.exit_reason == "TIME").mean(),
            })
    return pd.DataFrame(rows), pd.DataFrame(years)


def main():
    cfg = json.loads(Path("research/config.json").read_text())
    out = Path("research/results/round3b")
    out.mkdir(parents=True, exist_ok=True)
    btc = load_symbol("BTCUSDT", cfg["interval"], cfg["start"], cfg["end"])
    all_s, all_t = [], []
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut = {ex.submit(process_symbol, s, btc, cfg): s for s in cfg["symbols"]}
        for f in as_completed(fut):
            s = fut[f]
            try:
                sig, tr = f.result()
                all_s.extend(sig)
                all_t.extend(tr)
                print(s, len(sig), len(tr), flush=True)
            except Exception as e:
                print("ERROR", s, repr(e), flush=True)

    signals = pd.DataFrame(all_s)
    trades = pd.DataFrame(all_t)
    signals.to_csv(out / "signals.csv", index=False)
    trades.to_csv(out / "trades.csv", index=False)
    if trades.empty:
        raise RuntimeError("Round 3B produced no trades")
    sm, yr = summarise(trades)
    sm.to_csv(out / "summary.csv", index=False)
    yr.to_csv(out / "year_summary.csv", index=False)
    manifest = {
        "study": "Round 3B runner trailing methodology",
        "frozen_entry": "CONTROLLED_ACTIVITY from Round 2B",
        "frozen_architecture": "$300 full runner, max 10% initial stop, +5% activation, move to breakeven",
        "only_variable": "post-activation trailing methodology",
        "trails": TRAILS,
        "swing_definition": "causal 3-left/3-right confirmed pivot low, 0.3% buffer",
        "atr_definition": "24h mean true range; chandelier from highest high since activation",
        "no_trail": "breakeven remains until hit or 720h timeout",
        "same_bar_rule": "before activation, stop wins if stop and +5% activation both occur in same hourly candle",
        "position_usdt": cfg["position_usdt"],
        "fees": cfg["fee_rate"],
        "slippage": cfg["slippage_rate"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(sm.sort_values(["net_pnl_usdt", "profit_factor"], ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
