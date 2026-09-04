from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from binance_data import load_symbol
from round2b_validation import add_live_features, controlled_activity

STOP_CAPS = [None, 0.02, 0.05, 0.10]


def net_pnl(entry: float, exits: list[tuple[float, float]], cfg: dict) -> float:
    notional = float(cfg["position_usdt"])
    fee = float(cfg["fee_rate"])
    qty = notional / entry
    pnl = -notional * fee
    for fraction, raw_price in exits:
        price = raw_price * (1 - cfg["slippage_rate"])
        proceeds = qty * fraction * price
        pnl += proceeds - notional * fraction - proceeds * fee
    return pnl


def capped_stop(entry: float, structural: float, cap: float | None) -> float:
    if cap is None:
        return structural
    return max(structural, entry * (1 - cap))


def variant_name(kind: str, cap: float | None, activation: str = "5PCT", lock: float = 0.0) -> str:
    cap_name = "STRUCT" if cap is None else f"CAP{int(cap*100)}"
    if kind == "A":
        return f"A_FULL_TP5_{cap_name}"
    if kind == "B":
        return f"B_HALF_TP5_{cap_name}"
    lock_name = "BE" if lock == 0 else f"LOCK{int(lock*100)}"
    return f"FULL_{lock_name}_{activation}_{cap_name}"


def activation_price(entry: float, stop: float, activation: str) -> float:
    if activation == "5PCT":
        return entry * 1.05
    risk = entry - stop
    if activation == "1R":
        return entry + risk
    if activation == "1_5R":
        return entry + 1.5 * risk
    raise ValueError(activation)


def finish(sig: dict, variant: str, exit_time, reason: str, pnl: float, ambiguous: int,
           stop: float, activation: float | None, mfe: float, mae: float, peak_giveback: float) -> dict:
    return {
        "signal_id": sig["signal_id"], "symbol": sig["symbol"], "variant": variant,
        "entry_time": sig["entry_time"], "exit_time": exit_time, "exit_reason": reason,
        "pnl_usdt": pnl, "return_pct": pnl / 300.0, "ambiguous_bars": ambiguous,
        "initial_stop": stop, "initial_stop_pct": stop / sig["entry_price"] - 1,
        "activation_price": activation, "mfe_pct": mfe, "mae_pct": mae,
        "peak_giveback_pct": peak_giveback,
    }


def simulate(df: pd.DataFrame, sig: dict, kind: str, cap: float | None, cfg: dict,
             activation: str = "5PCT", lock: float = 0.0) -> dict:
    start = int(sig["entry_index"])
    end = min(len(df), start + int(cfg["max_holding_hours"]) + 1)
    entry = float(sig["entry_price"])
    stop = capped_stop(entry, float(sig["structural_stop"]), cap)
    variant = variant_name(kind, cap, activation, lock)
    tp5 = entry * 1.05
    act = activation_price(entry, stop, activation) if kind == "FULL" else tp5
    activated = False
    half_taken = False
    runner_stop = stop
    ambiguous = 0
    max_high = entry
    min_low = entry

    for j in range(start, end):
        bar = df.iloc[j]
        low, high = float(bar.low), float(bar.high)
        max_high = max(max_high, high)
        min_low = min(min_low, low)
        mfe = max_high / entry - 1
        mae = min_low / entry - 1

        if kind == "A":
            hit_s, hit_t = low <= stop, high >= tp5
            if hit_s and hit_t: ambiguous += 1
            if hit_s:
                pnl = net_pnl(entry, [(1.0, stop)], cfg)
                return finish(sig, variant, bar.time, "STOP", pnl, ambiguous, stop, tp5, mfe, mae, max(0.0, mfe - (stop/entry-1)))
            if hit_t:
                pnl = net_pnl(entry, [(1.0, tp5)], cfg)
                return finish(sig, variant, bar.time, "TARGET", pnl, ambiguous, stop, tp5, mfe, mae, max(0.0, mfe - 0.05))

        elif kind == "B":
            if not half_taken:
                hit_s, hit_t = low <= stop, high >= tp5
                if hit_s and hit_t: ambiguous += 1
                if hit_s:
                    pnl = net_pnl(entry, [(1.0, stop)], cfg)
                    return finish(sig, variant, bar.time, "STOP", pnl, ambiguous, stop, tp5, mfe, mae, max(0.0, mfe - (stop/entry-1)))
                if hit_t:
                    half_taken = True
                    runner_stop = entry
                    continue
            else:
                if low <= runner_stop:
                    pnl = net_pnl(entry, [(0.5, tp5), (0.5, runner_stop)], cfg)
                    exit_ret = ((tp5/entry-1) + (runner_stop/entry-1)) / 2
                    return finish(sig, variant, bar.time, "RUNNER_STOP", pnl, ambiguous, stop, tp5, mfe, mae, max(0.0, mfe - exit_ret))
                left = max(start, j - 48)
                if j > left:
                    support = float(df.low.iloc[left:j].min()) * 0.997
                    runner_stop = max(entry, runner_stop, support)

        elif kind == "FULL":
            if not activated:
                hit_s, hit_a = low <= stop, high >= act
                if hit_s and hit_a: ambiguous += 1
                if hit_s:
                    pnl = net_pnl(entry, [(1.0, stop)], cfg)
                    return finish(sig, variant, bar.time, "STOP", pnl, ambiguous, stop, act, mfe, mae, max(0.0, mfe - (stop/entry-1)))
                if hit_a:
                    activated = True
                    runner_stop = max(entry * (1 + lock), entry)
                    continue
            else:
                if low <= runner_stop:
                    pnl = net_pnl(entry, [(1.0, runner_stop)], cfg)
                    exit_ret = runner_stop / entry - 1
                    return finish(sig, variant, bar.time, "RUNNER_STOP", pnl, ambiguous, stop, act, mfe, mae, max(0.0, mfe - exit_ret))
                left = max(start, j - 48)
                if j > left:
                    support = float(df.low.iloc[left:j].min()) * 0.997
                    runner_stop = max(runner_stop, entry * (1 + lock), support)

    last = df.iloc[end - 1]
    mfe = max_high / entry - 1
    mae = min_low / entry - 1
    if kind == "B" and half_taken:
        pnl = net_pnl(entry, [(0.5, tp5), (0.5, float(last.close))], cfg)
        exit_ret = ((tp5/entry-1) + (float(last.close)/entry-1)) / 2
        reason = "TIME_RUNNER"
    else:
        pnl = net_pnl(entry, [(1.0, float(last.close))], cfg)
        exit_ret = float(last.close)/entry - 1
        reason = "TIME"
    return finish(sig, variant, last.time, reason, pnl, ambiguous, stop, act, mfe, mae, max(0.0, mfe - exit_ret))


def variants():
    out=[]
    for cap in STOP_CAPS:
        out.append(("A",cap,"5PCT",0.0))
        out.append(("B",cap,"5PCT",0.0))
        for lock in [0.0,0.01,0.02]:
            out.append(("FULL",cap,"5PCT",lock))
        out.append(("FULL",cap,"1R",0.0))
        out.append(("FULL",cap,"1_5R",0.0))
    return out


def process_symbol(symbol: str, btc: pd.DataFrame, cfg: dict):
    df = btc.copy() if symbol == "BTCUSDT" else load_symbol(symbol,cfg["interval"],cfg["start"],cfg["end"])
    if len(df) < 800: return [], []
    x=add_live_features(df,btc)
    sigs=controlled_activity(symbol,x,cfg)
    rows=[]
    for sig in sigs:
        for kind, cap, activation, lock in variants():
            rows.append(simulate(df, sig, kind, cap, cfg, activation, lock))
    return sigs,rows


def summarise(trades: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    rows=[]; years=[]
    q=trades.copy(); q["year"]=pd.to_datetime(q.entry_time).dt.year
    for variant,g in q.groupby("variant"):
        p=g.pnl_usdt.to_numpy(float); w=p[p>0]; l=p[p<0]
        rows.append({
            "variant":variant,"trades":len(g),"net_pnl_usdt":p.sum(),"expectancy_usdt":p.mean(),
            "median_pnl_usdt":np.median(p),"win_rate":(p>0).mean(),
            "avg_loss_usdt":l.mean() if len(l) else 0,"worst_loss_usdt":l.min() if len(l) else 0,
            "profit_factor":w.sum()/abs(l.sum()) if len(l) and l.sum() else math.inf,
            "avg_mfe_pct":g.mfe_pct.mean(),"avg_peak_giveback_pct":g.peak_giveback_pct.mean(),
            "ambiguous_bars":int(g.ambiguous_bars.sum())
        })
        for y,gy in g.groupby("year"):
            pp=gy.pnl_usdt.to_numpy(float); ww=pp[pp>0]; ll=pp[pp<0]
            years.append({"variant":variant,"year":int(y),"trades":len(gy),"net_pnl_usdt":pp.sum(),
                          "expectancy_usdt":pp.mean(),"win_rate":(pp>0).mean(),
                          "profit_factor":ww.sum()/abs(ll.sum()) if len(ll) and ll.sum() else math.inf})
    return pd.DataFrame(rows),pd.DataFrame(years)


def main():
    cfg=json.loads(Path("research/config.json").read_text())
    out=Path("research/results/round3a"); out.mkdir(parents=True,exist_ok=True)
    btc=load_symbol("BTCUSDT",cfg["interval"],cfg["start"],cfg["end"])
    all_s=[]; all_t=[]
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut={ex.submit(process_symbol,s,btc,cfg):s for s in cfg["symbols"]}
        for f in as_completed(fut):
            s=fut[f]
            try:
                sig,tr=f.result(); all_s.extend(sig); all_t.extend(tr); print(s,len(sig),len(tr),flush=True)
            except Exception as e:
                print("ERROR",s,repr(e),flush=True)
    signals=pd.DataFrame(all_s); trades=pd.DataFrame(all_t)
    signals.to_csv(out/"signals.csv",index=False); trades.to_csv(out/"trades.csv",index=False)
    sm,yr=summarise(trades); sm.to_csv(out/"summary.csv",index=False); yr.to_csv(out/"year_summary.csv",index=False)
    manifest={
      "study":"Round 3A initial-risk and de-risk architecture",
      "entry_family":"CONTROLLED_ACTIVITY frozen from Round 2B",
      "trailing_rule":"48h completed-bar low * 0.997, unchanged",
      "stop_caps":["structural","2%","5%","10%"],
      "architectures":["A full exit at +5%","B half at +5% then runner","full runner activated at +5% with BE/+1%/+2% lock","full runner activated at 1R or 1.5R to breakeven"],
      "note":"No entry-rule changes. Same hourly conservative same-bar stop/activation ordering: stop wins when unknowable.",
      "position_usdt":cfg["position_usdt"],"fees":cfg["fee_rate"],"slippage":cfg["slippage_rate"]
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(sm.sort_values("net_pnl_usdt",ascending=False).to_string(index=False))

if __name__=="__main__": main()
