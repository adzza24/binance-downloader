from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from binance_data import load_symbol
from round2b_validation import add_live_features, controlled_activity
from round3a_exit_architecture import capped_stop, net_pnl

VARIANTS = {
    "A_BE5_CURRENT": {"mode": "be", "activation": 0.05},
    "B_BE10": {"mode": "be", "activation": 0.10},
    "C_BE15": {"mode": "be", "activation": 0.15},
    "D_BINANCE_TRAIL10": {"mode": "binance_trail", "trail_delta": 0.10},
    "E_NO_BE_UNTIL30": {"mode": "none"},
}

GAIN_GIVEBACK = 0.60
TRAIL_START = 0.30
MIN_FLOOR_GAIN = 0.10
TP_GAIN = 10.0


def result(sig, name, t, reason, pnl, is_open, bars, ret, highest_gain, stop_gain):
    return {
        "signal_id": sig["signal_id"],
        "symbol": sig["symbol"],
        "entry_time": sig["entry_time"],
        "variant": name,
        "exit_time": t,
        "exit_reason": reason,
        "pnl_usdt": pnl,
        "is_open": is_open,
        "bars_held": bars,
        "exit_return_pct": ret,
        "highest_gain": highest_gain,
        "final_stop_gain": stop_gain,
    }


def simulate(df, sig, name, spec, cfg):
    start = int(sig["entry_index"])
    entry = float(sig["entry_price"])
    init_stop = capped_stop(entry, float(sig["structural_stop"]), 0.10)
    stop = init_stop
    peak = entry
    trail30_active = False

    for j in range(start, len(df)):
        b = df.iloc[j]
        low, high, close = map(float, (b.low, b.high, b.close))

        # Match prior Round 3 ordering: an already-live stop is checked before
        # any new same-bar activation/ratchet is applied.
        if low <= stop:
            reason = "INITIAL_STOP" if stop <= init_stop + 1e-12 else "PROTECTIVE_STOP"
            return result(
                sig, name, b.time, reason,
                net_pnl(entry, [(1, stop)], cfg), False, j - start + 1,
                stop / entry - 1, peak / entry - 1, stop / entry - 1,
            )

        peak = max(peak, high)
        peak_gain = peak / entry - 1

        if high >= entry * (1 + TP_GAIN):
            px = entry * (1 + TP_GAIN)
            return result(
                sig, name, b.time, "TAKE_PROFIT",
                net_pnl(entry, [(1, px)], cfg), False, j - start + 1,
                TP_GAIN, peak_gain, stop / entry - 1,
            )

        # Early protection variants. These only define how risk is managed
        # before the common +30% runner regime takes over.
        if not trail30_active:
            if spec["mode"] == "be":
                if peak_gain >= spec["activation"]:
                    stop = max(stop, entry)
            elif spec["mode"] == "binance_trail":
                # Binance-style trailing delta: stop is a fixed percentage
                # below the highest market price observed since entry/order.
                # A 10% delta therefore uses peak_price * 0.90, not 10% of gain.
                candidate = peak * (1 - spec["trail_delta"])
                candidate = min(candidate, close * 0.999)
                stop = max(stop, candidate)

        # Common frozen Round-3H runner architecture from +30% onward.
        if peak_gain >= TRAIL_START:
            trail30_active = True
            floor_from_gain_giveback = peak_gain * (1 - GAIN_GIVEBACK)
            candidate_gain = max(MIN_FLOOR_GAIN, floor_from_gain_giveback)
            candidate = entry * (1 + candidate_gain)
            candidate = min(candidate, close * 0.999)
            stop = max(stop, candidate)

    close = float(df.iloc[-1].close)
    return result(
        sig, name, df.iloc[-1].time, "DATA_END_OPEN",
        net_pnl(entry, [(1, close)], cfg), True, len(df) - start,
        close / entry - 1, peak / entry - 1, stop / entry - 1,
    )


def process_symbol(symbol, btc, cfg):
    df = btc.copy() if symbol == "BTCUSDT" else load_symbol(symbol, cfg["interval"], cfg["start"], cfg["end"])
    if len(df) < 800:
        return [], []
    x = add_live_features(df, btc)
    sigs = controlled_activity(symbol, x, cfg)
    rows = []
    for sig in sigs:
        for name, spec in VARIANTS.items():
            rows.append(simulate(df, sig, name, spec, cfg))
    return sigs, rows


def summarise(q):
    q = q.copy()
    rows = []
    for variant, g in q.groupby("variant"):
        closed = g[~g.is_open]
        opened = g[g.is_open]
        p = closed.pnl_usdt.astype(float)
        wins = p[p > 0]
        losses = p[p < 0]
        rows.append({
            "variant": variant,
            "trades": len(g),
            "closed": len(closed),
            "open": len(opened),
            "realised_pnl": closed.pnl_usdt.sum(),
            "open_mtm": opened.pnl_usdt.sum(),
            "combined_mtm": g.pnl_usdt.sum(),
            "profit_factor": wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() else math.inf,
            "win_rate": (p > 0).mean() if len(p) else None,
            "median_pnl": p.median() if len(p) else None,
            "avg_pnl": p.mean() if len(p) else None,
            "initial_stop_exits": int((closed.exit_reason == "INITIAL_STOP").sum()),
            "protective_stop_exits": int((closed.exit_reason == "PROTECTIVE_STOP").sum()),
            "tp_exits": int((closed.exit_reason == "TAKE_PROFIT").sum()),
        })
    return pd.DataFrame(rows)


def main():
    cfg = json.loads(Path("research/config.json").read_text())
    out = Path("research/results/round3i")
    out.mkdir(parents=True, exist_ok=True)

    btc = load_symbol("BTCUSDT", cfg["interval"], cfg["start"], cfg["end"])
    all_signals, all_trades = [], []
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut = {ex.submit(process_symbol, s, btc, cfg): s for s in cfg["symbols"]}
        for f in as_completed(fut):
            s = fut[f]
            try:
                sigs, rows = f.result()
                all_signals += sigs
                all_trades += rows
                print(s, len(sigs), flush=True)
            except Exception as e:
                print("ERROR", s, repr(e), flush=True)

    signals = pd.DataFrame(all_signals)
    trades = pd.DataFrame(all_trades)
    signals.to_csv(out / "signals.csv", index=False)
    trades.to_csv(out / "trades.csv", index=False)

    full = summarise(trades)
    full.to_csv(out / "summary.csv", index=False)
    for start_year in [2021, 2022]:
        sub = trades[pd.to_datetime(trades.entry_time).dt.year >= start_year]
        summarise(sub).to_csv(out / f"summary_{start_year}_onward.csv", index=False)

    trades["year"] = pd.to_datetime(trades.entry_time).dt.year
    year_rows = []
    for (variant, year), g in trades.groupby(["variant", "year"]):
        closed = g[~g.is_open]
        opened = g[g.is_open]
        year_rows.append({
            "variant": variant,
            "year": int(year),
            "trades": len(g),
            "realised_pnl": closed.pnl_usdt.sum(),
            "open_mtm": opened.pnl_usdt.sum(),
            "combined_mtm": g.pnl_usdt.sum(),
        })
    pd.DataFrame(year_rows).to_csv(out / "year_summary.csv", index=False)

    manifest = {
        "study": "Round 3I early-stop sensitivity",
        "purpose": "Test whether +5% to breakeven is too tight while freezing the controlled-activity entry stream and the post-+30% Round-3H runner architecture.",
        "variants": {
            "A_BE5_CURRENT": "structural stop capped at -10%; +5% peak -> breakeven",
            "B_BE10": "structural stop capped at -10%; +10% peak -> breakeven",
            "C_BE15": "structural stop capped at -10%; +15% peak -> breakeven",
            "D_BINANCE_TRAIL10": "structural stop initially; Binance-style 10% trailing delta from entry, stop = highest price since entry * 0.90, ratcheting only upward",
            "E_NO_BE_UNTIL30": "structural stop capped at -10%; no early breakeven move",
        },
        "common_after_30": "+30% peak -> minimum +10% floor and 60% accumulated-gain giveback trail; stop never loosens; full exit at +1000%",
        "binance_trail_note": "D uses a fixed 10% drop from total peak market price, matching Binance trailing-delta semantics; it is not a 10% accumulated-gain giveback rule.",
        "execution_ordering": "Matches prior Round 3 hourly methodology: pre-existing stop checked before newly activated/ratcheted same-bar protection.",
        "warning": "Exploratory same-history sensitivity test, not independent validation. No live strategy change is authorised by this run.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("FULL\n", full.sort_values("combined_mtm", ascending=False).to_string(index=False))
    sub22 = trades[pd.to_datetime(trades.entry_time).dt.year >= 2022]
    print("2022+\n", summarise(sub22).sort_values("combined_mtm", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
