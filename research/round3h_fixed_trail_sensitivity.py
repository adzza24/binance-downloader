from __future__ import annotations

import json, math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

from binance_data import load_symbol
from round2b_validation import add_live_features, controlled_activity
from round3f_gain_trail_discovery import simulate_gain_trail

TRAILS = {"FIXED_40":0.40,"FIXED_50":0.50,"FIXED_60":0.60,"FIXED_70":0.70}
SCHEDULE_STARTS = [0.30,0.50,1.00,2.00,3.00,5.00,7.50]


def process_symbol(symbol, btc, cfg):
    df = btc.copy() if symbol == "BTCUSDT" else load_symbol(symbol,cfg["interval"],cfg["start"],cfg["end"])
    if len(df) < 800: return [], []
    x = add_live_features(df,btc)
    sigs = controlled_activity(symbol,x,cfg)
    rows=[]
    for sig in sigs:
        for name, gb in TRAILS.items():
            schedule=[(m,gb) for m in SCHEDULE_STARTS]
            rows.append(simulate_gain_trail(df,sig,name,schedule,cfg,tp=10.0))
    return sigs, rows


def summarise(q):
    q=q.copy(); q["year"]=pd.to_datetime(q.entry_time).dt.year
    rows=[]
    for v,g in q.groupby("variant"):
        c=g[~g.is_open]; o=g[g.is_open]; p=c.pnl_usdt.astype(float)
        w=p[p>0]; l=p[p<0]
        rows.append({
            "variant":v,"trades":len(g),"closed":len(c),"open":len(o),
            "realised_pnl":c.pnl_usdt.sum(),"open_mtm":o.pnl_usdt.sum(),"combined_mtm":g.pnl_usdt.sum(),
            "profit_factor":w.sum()/abs(l.sum()) if len(l) and l.sum() else math.inf,
            "win_rate":(p>0).mean() if len(p) else None,
        })
    return pd.DataFrame(rows)


def main():
    cfg=json.loads(Path("research/config.json").read_text())
    out=Path("research/results/round3h"); out.mkdir(parents=True,exist_ok=True)
    btc=load_symbol("BTCUSDT",cfg["interval"],cfg["start"],cfg["end"])
    all_s=[]; all_t=[]
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut={ex.submit(process_symbol,s,btc,cfg):s for s in cfg["symbols"]}
        for f in as_completed(fut):
            s=fut[f]
            try:
                sig,tr=f.result(); all_s+=sig; all_t+=tr; print(s,len(sig),flush=True)
            except Exception as e: print("ERROR",s,repr(e),flush=True)
    signals=pd.DataFrame(all_s); trades=pd.DataFrame(all_t)
    signals.to_csv(out/"signals.csv",index=False); trades.to_csv(out/"trades.csv",index=False)
    full=summarise(trades); full.to_csv(out/"summary.csv",index=False)
    for start in [2021,2022]:
        sub=trades[pd.to_datetime(trades.entry_time).dt.year>=start]
        summarise(sub).to_csv(out/f"summary_{start}_onward.csv",index=False)
    years=[]; trades["year"]=pd.to_datetime(trades.entry_time).dt.year
    for (v,y),g in trades.groupby(["variant","year"]):
        c=g[~g.is_open]; o=g[g.is_open]
        years.append({"variant":v,"year":int(y),"trades":len(g),"realised_pnl":c.pnl_usdt.sum(),"open_mtm":o.pnl_usdt.sum(),"combined_mtm":g.pnl_usdt.sum()})
    pd.DataFrame(years).to_csv(out/"year_summary.csv",index=False)
    (out/"manifest.json").write_text(json.dumps({
        "study":"Round 3H fixed gain-trail sensitivity confirmation",
        "definition":"trail percentage = fraction of accumulated gain allowed to be given back",
        "variants":TRAILS,
        "common_rules":"structural stop capped 10%; +5% -> breakeven; +30% -> minimum +10% floor; fixed gain-giveback trail from +30 onward; full exit at +1000%",
        "warning":"Exploratory same-history sensitivity test, not independent validation."
    },indent=2))
    print("FULL\n",full.sort_values("combined_mtm",ascending=False).to_string(index=False))
    print("2022+\n",summarise(trades[pd.to_datetime(trades.entry_time).dt.year>=2022]).sort_values("combined_mtm",ascending=False).to_string(index=False))

if __name__=="__main__": main()
