from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from binance_data import load_symbol
from backtest import simulate_method


def add_live_features(df: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    b = btc.set_index("time")
    x["ret_6h"] = x.close.pct_change(6)
    x["ret_12h"] = x.close.pct_change(12)
    x["ret_24h"] = x.close.pct_change(24)
    x["ret_72h"] = x.close.pct_change(72)
    btc6 = b.close.pct_change(6)
    btc24 = b.close.pct_change(24)
    btc72 = b.close.pct_change(72)
    x["btc_6h"] = x.time.map(btc6)
    x["btc_24h"] = x.time.map(btc24)
    x["btc_72h"] = x.time.map(btc72)
    x["rs_6h"] = x.ret_6h - x.btc_6h
    x["rs_24h"] = x.ret_24h - x.btc_24h
    x["rs_72h"] = x.ret_72h - x.btc_72h

    x["low24"] = x.low.shift(1).rolling(24).min()
    x["high24"] = x.high.shift(1).rolling(24).max()
    x["high72"] = x.high.shift(1).rolling(72).max()
    x["high720"] = x.high.shift(1).rolling(720).max()
    x["range_loc24"] = (x.close - x.low24) / (x.high24 - x.low24).replace(0, np.nan)
    x["dist_high72"] = x.close / x.high72 - 1
    x["range6"] = x.high.shift(1).rolling(6).max() / x.low.shift(1).rolling(6).min() - 1

    volbase = x.volume.shift(1).rolling(48).mean()
    trbase = x.trades.shift(1).rolling(48).mean()
    x["vol_ratio"] = x.volume / volbase
    x["trade_ratio"] = x.trades / trbase
    x["vol6_ratio"] = x.volume.shift(1).rolling(6).mean() / volbase
    taker = x.taker_buy_base / x.volume.replace(0, np.nan)
    x["taker"] = taker
    x["taker6"] = taker.shift(1).rolling(6).mean()
    x["taker6_delta"] = x.taker6 - x.taker6.shift(6)

    # Original pre-breakout fields, computed only from completed/current data.
    x["base_low120"] = x.low.shift(1).rolling(120).min()
    x["base_high120"] = x.high.shift(1).rolling(120).max()
    x["base_range120"] = x.base_high120 / x.base_low120 - 1
    x["resistance_distance"] = (x.base_high120 - x.close) / x.base_high120
    recent_vol = x.volume.shift(1).rolling(60).mean()
    prior_vol = x.volume.shift(61).rolling(60).mean()
    x["volume_contraction"] = recent_vol / prior_vol
    ref_close = x.close.shift(120)
    prior_low = x.low.shift(120).rolling(600).min()
    x["prior_impulse"] = ref_close / prior_low - 1
    return x


def make_signal(symbol: str, x: pd.DataFrame, i: int, family: str, structural: float, cfg: dict) -> dict | None:
    if i + 1 >= len(x):
        return None
    entry_i = i + 1
    entry = float(x.open.iloc[entry_i]) * (1 + cfg["slippage_rate"])
    if not np.isfinite(structural) or structural >= entry:
        return None
    overhead = float(x.high720.iloc[i]) if np.isfinite(x.high720.iloc[i]) else np.nan
    risk = entry - structural
    dynamic_rr = (overhead - entry) / risk if np.isfinite(overhead) and overhead > entry else np.nan
    return {
        "signal_id": f"{family}-{symbol}-{x.time.iloc[i].strftime('%Y%m%dT%H%M%SZ')}",
        "family": family,
        "symbol": symbol,
        "signal_time": x.time.iloc[i],
        "entry_time": x.time.iloc[entry_i],
        "entry_index": int(entry_i),
        "entry_price": entry,
        "structural_stop": structural,
        "dynamic_target": overhead if np.isfinite(dynamic_rr) and dynamic_rr >= 2 else np.nan,
        "dynamic_rr": dynamic_rr,
        "ret_6h": float(x.ret_6h.iloc[i]), "ret_24h": float(x.ret_24h.iloc[i]),
        "rs_6h": float(x.rs_6h.iloc[i]), "rs_24h": float(x.rs_24h.iloc[i]),
        "range_loc24": float(x.range_loc24.iloc[i]), "dist_high72": float(x.dist_high72.iloc[i]),
        "vol_ratio": float(x.vol_ratio.iloc[i]), "trade_ratio": float(x.trade_ratio.iloc[i]),
        "taker": float(x.taker.iloc[i]), "taker6": float(x.taker6.iloc[i]),
        "taker6_delta": float(x.taker6_delta.iloc[i]), "btc_6h": float(x.btc_6h.iloc[i]),
    }


def controlled_activity(symbol: str, x: pd.DataFrame, cfg: dict) -> list[dict]:
    # Round-2 activity hypothesis: controlled rather than extreme participation,
    # stronger taker buying, and no very mature prior impulse.
    cond = (
        (x.prior_impulse >= 0.12) & (x.prior_impulse <= 0.50)
        & (x.base_range120 <= 0.14)
        & x.resistance_distance.between(-0.01, 0.025)
        & (x.volume_contraction <= 0.80)
        & x.vol_ratio.between(1.50, 4.00)
        & x.trade_ratio.between(1.25, 4.00)
        & (x.taker >= 0.54)
        & (x.rs_72h >= 0)
        & (x.close > x.open)
    ).fillna(False)
    return collect(symbol, x, cfg, "CONTROLLED_ACTIVITY", cond, lambda i: float(x.base_low120.iloc[i]) * 0.997, 72)


def reversal_leadership(symbol: str, x: pd.DataFrame, cfg: dict) -> list[dict]:
    # Fixed from the qualitative Round-2A/2A-2 findings, not optimised here:
    # a meaningful washout followed by a modest first recovery, improving RS and
    # taker balance, no extreme activity, and not already extended.
    washout_6h_ago = x.ret_24h.shift(6)
    low_loc_6h_ago = x.range_loc24.shift(6)
    rs24_6h_ago = x.rs_24h.shift(6)
    cond = (
        (washout_6h_ago <= -0.03)
        & (low_loc_6h_ago <= 0.35)
        & (rs24_6h_ago <= 0.00)
        & x.ret_6h.between(0.0125, 0.06)
        & ((x.rs_6h - x.rs_6h.shift(6)) >= 0.0125)
        & (x.range_loc24 >= 0.40)
        & (x.taker6 >= 0.50)
        & (x.taker6_delta > 0)
        & (x.vol6_ratio <= 2.50)
        & (x.btc_6h >= -0.025)
        & (x.ret_24h <= 0.08)
    ).fillna(False)
    return collect(symbol, x, cfg, "REVERSAL_LEADERSHIP", cond, lambda i: float(x.low.iloc[max(0, i-30):i+1].min()) * 0.997, 72)


def hybrid_watch(symbol: str, x: pd.DataFrame, cfg: dict) -> list[dict]:
    # A reversal creates a watch state. Entry is allowed only 12-72h later when
    # price has rebuilt toward the 72h high with controlled activity and positive RS.
    washout = (
        (x.ret_24h <= -0.03) & (x.range_loc24 <= 0.35) & (x.rs_24h <= 0)
    ).fillna(False)
    confirm = (
        (x.ret_6h > 0) & (x.ret_24h > 0)
        & (x.rs_24h > 0)
        & (x.dist_high72 >= -0.04)
        & (x.range6 <= 0.06)
        & (x.taker6 >= 0.52)
        & x.vol_ratio.between(0.70, 3.00)
        & (x.ret_24h <= 0.10)
    ).fillna(False)
    idxs = []
    last = -10**9
    watch_until = -1
    earliest = -1
    wash = washout.to_numpy(); conf = confirm.to_numpy()
    for i in range(len(x)):
        if wash[i]:
            earliest = i + 12
            watch_until = i + 72
        if i >= earliest and i <= watch_until and conf[i] and i - last >= 72:
            idxs.append(i); last = i; watch_until = -1; earliest = -1
    cond = pd.Series(False, index=x.index)
    if idxs: cond.iloc[idxs] = True
    return collect(symbol, x, cfg, "HYBRID_WATCH", cond, lambda i: float(x.low.iloc[max(0, i-72):i+1].min()) * 0.997, 72)


def collect(symbol, x, cfg, family, cond, stop_fn, cooldown):
    out=[]; last=-10**9
    for i in np.flatnonzero(cond.to_numpy()):
        if i-last < cooldown: continue
        sig=make_signal(symbol,x,i,family,stop_fn(i),cfg)
        if sig is not None:
            out.append(sig); last=i
    return out


def process_symbol(symbol: str, btc: pd.DataFrame, cfg: dict):
    df = btc.copy() if symbol == "BTCUSDT" else load_symbol(symbol,cfg["interval"],cfg["start"],cfg["end"])
    if len(df) < 800: return [], []
    x=add_live_features(df,btc)
    sigs = controlled_activity(symbol,x,cfg) + reversal_leadership(symbol,x,cfg) + hybrid_watch(symbol,x,cfg)
    trades=[]
    for sig in sigs:
        for method in "ABCD":
            r=simulate_method(df,sig,method,cfg); r["family"]=sig["family"]; trades.append(r)
    return sigs,trades


def summaries(trades: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    rows=[]; years=[]
    q=trades[trades.status=="TRADE"].copy()
    q["year"]=pd.to_datetime(q.entry_time).dt.year
    for (fam,method),g in q.groupby(["family","method"]):
        pnl=g.pnl_usdt.to_numpy(float); wins=pnl[pnl>0]; losses=pnl[pnl<0]
        rows.append({"family":fam,"method":method,"trades":len(g),"net_pnl_usdt":pnl.sum(),"expectancy_usdt":pnl.mean(),"win_rate":(pnl>0).mean(),"avg_loss_usdt":losses.mean() if len(losses) else 0,"worst_loss_usdt":losses.min() if len(losses) else 0,"profit_factor":wins.sum()/abs(losses.sum()) if len(losses) and losses.sum() else math.inf})
        for y,gy in g.groupby("year"):
            p=gy.pnl_usdt.to_numpy(float); w=p[p>0]; l=p[p<0]
            years.append({"family":fam,"method":method,"year":int(y),"trades":len(gy),"net_pnl_usdt":p.sum(),"expectancy_usdt":p.mean(),"win_rate":(p>0).mean(),"profit_factor":w.sum()/abs(l.sum()) if len(l) and l.sum() else math.inf})
    return pd.DataFrame(rows),pd.DataFrame(years)


def main():
    cfg=json.loads(Path("research/config.json").read_text())
    out=Path("research/results/round2b"); out.mkdir(parents=True,exist_ok=True)
    btc=load_symbol("BTCUSDT",cfg["interval"],cfg["start"],cfg["end"])
    all_s=[]; all_t=[]
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut={ex.submit(process_symbol,s,btc,cfg):s for s in cfg["symbols"]}
        for f in as_completed(fut):
            s=fut[f]
            try:
                sig,tr=f.result(); all_s.extend(sig); all_t.extend(tr); print(s,len(sig),flush=True)
            except Exception as e:
                print("ERROR",s,repr(e),flush=True)
    signals=pd.DataFrame(all_s); trades=pd.DataFrame(all_t)
    signals.to_csv(out/"signals.csv",index=False); trades.to_csv(out/"trades.csv",index=False)
    sm,yr=summaries(trades); sm.to_csv(out/"summary.csv",index=False); yr.to_csv(out/"year_summary.csv",index=False)
    if len(signals):
        signals.assign(year=pd.to_datetime(signals.entry_time).dt.year).groupby(["family","year"]).size().rename("signals").reset_index().to_csv(out/"signal_counts.csv",index=False)
    manifest={
      "study":"Round 2B no-lookahead entry-family validation",
      "note":"All entry conditions use only data available at signal time. A/B/C/D exits are unchanged. Thresholds are hypothesis-driven from prior discovery and therefore this is not an untouched statistical holdout.",
      "families":["CONTROLLED_ACTIVITY","REVERSAL_LEADERSHIP","HYBRID_WATCH"],
      "position_usdt":cfg["position_usdt"],"fees":cfg["fee_rate"],"slippage":cfg["slippage_rate"]
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(sm.sort_values(["family","method"]).to_string(index=False))

if __name__=="__main__": main()
