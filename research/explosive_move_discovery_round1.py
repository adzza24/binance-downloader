from __future__ import annotations

import json, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from numba import njit
from sklearn.metrics import roc_auc_score

from binance_data import load_symbol
from early_breakout_round1 import TASK_EXCLUSIONS

OUT = Path("research/results/explosive_move_discovery_round1")
START = "2019-01-01"
END = "2026-08-31"
MAX_HOURS = 30 * 24
TARGET = 0.20
STOP = 0.05
EVENT_COOLDOWN_HOURS = 24
LEADS = [0, 6, 12, 24, 72, 168, 336, 720]
CONTROLS_PER_EVENT = 3
RNG = np.random.default_rng(42)

STABLE_BASES = {
    "USDC","FDUSD","TUSD","USDP","DAI","BUSD","EUR","EURI","AEUR","TRY","BRL",
    "GBP","AUD","UAH","RUB","BIDR","IDRT","NGN","ZAR","VAI","UST","USTC"
}
EXCL = set(TASK_EXCLUSIONS)

FEATURES = [
    "ret_6h","ret_24h","ret_72h","ret_7d","ret_14d","ret_30d","ret_60d","ret_90d",
    "range_24h","range_72h","range_7d","range_14d","range_30d","range_90d",
    "atr24_pct","atr7d_pct","atr30d_pct","atr24_vs_7d","range24_vs_7d",
    "dist_7d_high","dist_30d_high","dist_90d_high","rebound_7d_low","rebound_30d_low",
    "worst_1h_ret_7d","worst_4h_ret_14d","volume_ratio_6h","volume_ratio_24h",
    "trade_ratio_6h","taker_buy_6h","quote_volume_24h_log",
    "rs_24h","rs_7d","rs_30d"
]

def split_name(year):
    if year <= 2022: return "DISCOVERY_2019_2022"
    if year <= 2024: return "VALIDATION_2023_2024"
    return "HOLDOUT_2025_2026"

def get_json(url):
    req=Request(url,headers={"User-Agent":"explosive-move-discovery/1.0"})
    with urlopen(req,timeout=30) as r:
        return json.loads(r.read().decode())

def current_top100():
    info=get_json("https://api.binance.com/api/v3/exchangeInfo")
    tick=get_json("https://api.binance.com/api/v3/ticker/24hr")
    ok={}
    for s in info["symbols"]:
        sym=s.get("symbol",""); base=s.get("baseAsset",""); quote=s.get("quoteAsset","")
        if quote!="USDT" or s.get("status")!="TRADING" or not s.get("isSpotTradingAllowed",True): continue
        if base in STABLE_BASES or sym in EXCL: continue
        if re.search(r"(UP|DOWN|BULL|BEAR)USDT$",sym): continue
        ok[sym]=base
    vols={x["symbol"]:float(x.get("quoteVolume",0) or 0) for x in tick if x.get("symbol") in ok}
    ranked=sorted(vols.items(),key=lambda kv:kv[1],reverse=True)[:100]
    return pd.DataFrame([{"symbol":s,"current_quote_volume_24h":v,"current_liquidity_rank":i+1,"universe_group":"TOP50" if i<50 else "TOP100_51_100"} for i,(s,v) in enumerate(ranked)])

@njit
def clean_paths(close, high, low, max_hours, target_mult, stop_mult):
    n=len(close); target_idx=np.full(n,-1,np.int64); stop_idx=np.full(n,-1,np.int64)
    for i in range(n-1):
        e=close[i]
        if not np.isfinite(e) or e<=0: continue
        end=min(n,i+max_hours+1)
        for j in range(i+1,end):
            if low[j] <= e*stop_mult:
                stop_idx[i]=j
                break
            if high[j] >= e*target_mult:
                target_idx[i]=j
                break
    return target_idx, stop_idx

def first_cross(high, start, end, level):
    for j in range(start+1,end+1):
        if high[j] >= level: return j
    return -1

def build_features(df,btdf):
    x=df.copy().set_index("time"); b=btdf.set_index("time").close.reindex(x.index)
    c=x.close
    for h,name in [(6,"6h"),(24,"24h"),(72,"72h"),(168,"7d"),(336,"14d"),(720,"30d"),(1440,"60d"),(2160,"90d")]:
        x[f"ret_{name}"]=c.pct_change(h)
    for h,name in [(24,"24h"),(72,"72h"),(168,"7d"),(336,"14d"),(720,"30d"),(2160,"90d")]:
        hi=x.high.rolling(h).max(); lo=x.low.rolling(h).min()
        x[f"range_{name}"]=hi/lo-1
    tr=pd.concat([x.high-x.low,(x.high-c.shift()).abs(),(x.low-c.shift()).abs()],axis=1).max(axis=1)
    x["atr24_pct"]=tr.rolling(24).mean()/c
    x["atr7d_pct"]=tr.rolling(168).mean()/c
    x["atr30d_pct"]=tr.rolling(720).mean()/c
    x["atr24_vs_7d"]=x.atr24_pct/x.atr7d_pct
    x["range24_vs_7d"]=x.range_24h/x.range_7d
    x["dist_7d_high"]=c/x.high.rolling(168).max()-1
    x["dist_30d_high"]=c/x.high.rolling(720).max()-1
    x["dist_90d_high"]=c/x.high.rolling(2160).max()-1
    x["rebound_7d_low"]=c/x.low.rolling(168).min()-1
    x["rebound_30d_low"]=c/x.low.rolling(720).min()-1
    x["worst_1h_ret_7d"]=c.pct_change().rolling(168).min()
    x["worst_4h_ret_14d"]=c.pct_change(4).rolling(336).min()
    vbase=x.volume.shift(1).rolling(72).mean()
    tbase=x.trades.shift(1).rolling(72).mean()
    x["volume_ratio_6h"]=x.volume.rolling(6).mean()/vbase
    x["volume_ratio_24h"]=x.volume.rolling(24).mean()/vbase
    x["trade_ratio_6h"]=x.trades.rolling(6).mean()/tbase
    taker=x.taker_buy_base/x.volume.replace(0,np.nan)
    x["taker_buy_6h"]=taker.rolling(6).mean()
    x["quote_volume_24h_log"]=np.log1p(x.quote_volume.rolling(24).sum())
    x["rs_24h"]=x.ret_24h-b.pct_change(24)
    x["rs_7d"]=x.ret_7d-b.pct_change(168)
    x["rs_30d"]=x.ret_30d-b.pct_change(720)
    return x.reset_index()

def detect_events(symbol,rank,df):
    c=df.close.to_numpy(float); h=df.high.to_numpy(float); l=df.low.to_numpy(float)
    ti,si=clean_paths(c,h,l,MAX_HOURS,1+TARGET,1-STOP)
    valid=np.flatnonzero(ti>=0)
    rows=[]; blocked=-1; eid=0
    for i in valid:
        t=int(ti[i])
        if i<=blocked or i<2160: continue
        eid+=1; entry=c[i]
        milestones={}
        for pct in [.01,.02,.03,.05,.10,.20,.30,.50]:
            j=first_cross(h,i,min(len(h)-1,i+MAX_HOURS),entry*(1+pct))
            milestones[f"hours_to_{int(pct*100)}pct"]=(j-i) if j>=0 else np.nan
        end30=min(len(df),i+MAX_HOURS+1)
        mfe30=float(h[i+1:end30].max()/entry-1) if end30>i+1 else np.nan
        mae_to_target=float(l[i+1:t+1].min()/entry-1)
        rows.append({
            "event_id":f"{symbol}:{pd.Timestamp(df.time.iloc[i]).isoformat()}",
            "symbol":symbol,"current_liquidity_rank":int(rank),
            "universe_group":"TOP50" if rank<=50 else "TOP100_51_100",
            "start_index":int(i),"target_index":t,
            "start_time":pd.Timestamp(df.time.iloc[i]),"target_time":pd.Timestamp(df.time.iloc[t]),
            "year":int(pd.Timestamp(df.time.iloc[i]).year),"split":split_name(int(pd.Timestamp(df.time.iloc[i]).year)),
            "start_price":entry,"hours_to_20pct":t-i,"mae_to_target_pct":mae_to_target,
            "mfe_30d_pct":mfe30,**milestones
        })
        blocked=t+EVENT_COOLDOWN_HOURS
    return pd.DataFrame(rows),ti

def control_indices(df,ti,events):
    n=len(df); bad=np.zeros(n,dtype=np.bool_)
    bad[np.flatnonzero(ti>=0)]=True
    for _,e in events.iterrows():
        i=int(e.start_index); lo=max(2160,i-MAX_HOURS); hi=min(n,i+MAX_HOURS+1); bad[lo:hi]=True
    return np.flatnonzero((~bad) & (np.arange(n)>=2160) & (np.arange(n)<n-MAX_HOURS-1))

def sample_rows(symbol,events,ti,df,ff):
    rows=[]; controls=control_indices(df,ti,events)
    years=pd.to_datetime(df.time,utc=True).dt.year.to_numpy(); by_year={}
    for y in np.unique(years[controls]):
        by_year[int(y)]=controls[years[controls]==y]
    for _,e in events.iterrows():
        start_i=int(e.start_index); year=int(e.year); pool=by_year.get(year,np.array([],dtype=int))
        chosen=RNG.choice(pool,size=min(CONTROLS_PER_EVENT,len(pool)),replace=False) if len(pool) else []
        cases=[("WINNER",start_i,e.event_id)] + [("CONTROL",int(ci),e.event_id) for ci in chosen]
        for label,pseudo_i,eid in cases:
            for lead in LEADS:
                si=pseudo_i-lead
                if si<2160: continue
                r=ff.iloc[si]
                row={"event_id":eid,"label":label,"symbol":symbol,"lead_hours":lead,
                     "snapshot_time":pd.Timestamp(df.time.iloc[si]),"pseudo_start_time":pd.Timestamp(df.time.iloc[pseudo_i]),
                     "year":int(pd.Timestamp(df.time.iloc[pseudo_i]).year),"split":split_name(int(pd.Timestamp(df.time.iloc[pseudo_i]).year))}
                for f in FEATURES:
                    val=getattr(r,f); row[f]=float(val) if pd.notna(val) else np.nan
                rows.append(row)
    return rows

def process_symbol(meta,cfg,btc):
    symbol=meta.symbol; rank=int(meta.current_liquidity_rank)
    df=load_symbol(symbol,"1h",START,END)
    if len(df)<3000: return symbol,pd.DataFrame(),[],f"insufficient_history:{len(df)}"
    events,ti=detect_events(symbol,rank,df)
    if events.empty: return symbol,events,[],None
    ff=build_features(df,btc); samples=sample_rows(symbol,events,ti,df,ff)
    return symbol,events,samples,None

def auc_direction(z,feature):
    a=z[[feature,"label"]].dropna()
    if a.label.nunique()<2 or len(a)<20: return np.nan,np.nan
    y=(a.label=="WINNER").astype(int)
    try: auc=roc_auc_score(y,a[feature])
    except ValueError: return np.nan,np.nan
    direction=1 if auc>=.5 else -1
    return max(auc,1-auc),direction

def contrasts(samples):
    rows=[]
    for lead in LEADS:
        for split in ["DISCOVERY_2019_2022","VALIDATION_2023_2024","HOLDOUT_2025_2026"]:
            z=samples[(samples.lead_hours==lead)&(samples.split==split)]
            for f in FEATURES:
                w=pd.to_numeric(z.loc[z.label=="WINNER",f],errors="coerce").dropna()
                c=pd.to_numeric(z.loc[z.label=="CONTROL",f],errors="coerce").dropna()
                if len(w)<10 or len(c)<10: continue
                pooled=np.sqrt((w.var()+c.var())/2)
                auc,dirn=auc_direction(z,f)
                rows.append({"lead_hours":lead,"split":split,"feature":f,"winner_n":len(w),"control_n":len(c),
                             "winner_median":w.median(),"control_median":c.median(),
                             "standardised_diff":(w.mean()-c.mean())/pooled if pooled else np.nan,
                             "separation_auc":auc,"direction":dirn})
    return pd.DataFrame(rows)

def stable_features(ct):
    d=ct[ct.split=="DISCOVERY_2019_2022"][["lead_hours","feature","separation_auc","direction"]].rename(columns={"separation_auc":"discovery_auc","direction":"discovery_direction"})
    v=ct[ct.split=="VALIDATION_2023_2024"][["lead_hours","feature","separation_auc","direction"]].rename(columns={"separation_auc":"validation_auc","direction":"validation_direction"})
    h=ct[ct.split=="HOLDOUT_2025_2026"][["lead_hours","feature","separation_auc","direction"]].rename(columns={"separation_auc":"holdout_auc","direction":"holdout_direction"})
    x=d.merge(v,on=["lead_hours","feature"],how="left").merge(h,on=["lead_hours","feature"],how="left")
    x["same_direction_validation"]=x.discovery_direction==x.validation_direction
    x["same_direction_holdout"]=x.discovery_direction==x.holdout_direction
    x["min_auc"]=x[["discovery_auc","validation_auc","holdout_auc"]].min(axis=1)
    return x.sort_values(["lead_hours","min_auc"],ascending=[True,False])

def event_summary(events):
    rows=[]
    for group,g in [("TOP100",events),("TOP50",events[events.current_liquidity_rank<=50])]:
        splits=["ALL"]+sorted(g.split.unique().tolist())
        for split in splits:
            z=g if split=="ALL" else g[g.split==split]
            rows.append({"universe":group,"split":split,"events":len(z),
                         "median_hours_to_20":z.hours_to_20pct.median() if len(z) else np.nan,
                         "p25_hours_to_20":z.hours_to_20pct.quantile(.25) if len(z) else np.nan,
                         "p75_hours_to_20":z.hours_to_20pct.quantile(.75) if len(z) else np.nan,
                         "median_hours_to_5":z.hours_to_5pct.median() if len(z) else np.nan,
                         "median_5_to_20_hours":(z.hours_to_20pct-z.hours_to_5pct).median() if len(z) else np.nan,
                         "median_mae_to_target_pct":z.mae_to_target_pct.median() if len(z) else np.nan,
                         "median_mfe_30d_pct":z.mfe_30d_pct.median() if len(z) else np.nan,
                         "pct_target_within_1d":(z.hours_to_20pct<=24).mean() if len(z) else np.nan,
                         "pct_target_within_3d":(z.hours_to_20pct<=72).mean() if len(z) else np.nan,
                         "pct_target_within_7d":(z.hours_to_20pct<=168).mean() if len(z) else np.nan})
    return pd.DataFrame(rows)

def write_analysis(summary,stable):
    top=stable[(stable.same_direction_validation)&(stable.same_direction_holdout)&(stable.min_auc>=.55)]
    lines=["# Explosive Move Discovery Round 1","",
           "**Status:** RESEARCH ONLY. Separate from Early Breakout and live strategy.","",
           "## Event definition","",
           "A clean event starts at an hourly close from which price reaches +20% before ever touching -5%, within a maximum 30-day observation window. Overlapping starts are de-duplicated into one event. The 30-day cap prevents indefinite bull-market drift from becoming an event while allowing much slower moves than the previous 24-hour rule.","",
           "Universe is the current top-100 Binance USDT spot pairs by quote volume at research-run time, with current top-50 reported separately. This first pass therefore has current-universe survivorship bias; it is not a historically reconstructed top-100.","",
           "## Event population","",
           "| Universe | Events | Median to +20 | Median +5 to +20 | <=1d | <=3d | <=7d |","|---|---:|---:|---:|---:|---:|---:|"]
    for _,r in summary[summary.split=="ALL"].iterrows():
        lines.append(f"| {r.universe} | {int(r.events)} | {r.median_hours_to_20:.0f}h | {r.median_5_to_20_hours:.0f}h | {100*r.pct_target_within_1d:.1f}% | {100*r.pct_target_within_3d:.1f}% | {100*r.pct_target_within_7d:.1f}% |")
    lines += ["","## Preliminary stable precursor features","",
              "Features below separate winners from same-symbol/year controls in the same direction across discovery, validation and holdout. AUC is univariate separation only, not a deployable signal.","",
              "| Lead | Feature | Discovery AUC | Validation AUC | Holdout AUC |","|---:|---|---:|---:|---:|"]
    for _,r in top.head(25).iterrows():
        lines.append(f"| {int(r.lead_hours)}h | {r.feature} | {r.discovery_auc:.3f} | {r.validation_auc:.3f} | {r.holdout_auc:.3f} |")
    lines += ["","## Outputs","",
              "- events.csv: de-duplicated clean +20%/-5% events and milestone timing.",
              "- event_summary.csv: event counts and speed for top-50/top-100.",
              "- precursor_samples.csv: winner/control snapshots from 30d before the move through the start.",
              "- feature_contrasts.csv: winner vs control distributions by lead and time split.",
              "- stable_features.csv: cross-period feature stability.",
              "- universe.csv: current top-100 universe used.",
              "- manifest.json: exact methodology.",
              "",
              "## Next step","",
              "Use the discovered event set to define a WATCH model days before the move and a separate ENTRY trigger constrained to fire before the event has advanced +5%. Do not optimise exits until the event detector shows useful out-of-sample precision.",""]
    (OUT/"ANALYSIS.md").write_text("\n".join(lines))

def main():
    cfg=json.loads(Path("research/config.json").read_text())
    OUT.mkdir(parents=True,exist_ok=True)
    universe=current_top100(); universe.to_csv(OUT/"universe.csv",index=False)
    btc=load_symbol("BTCUSDT","1h",START,END)
    event_parts=[]; sample_rows_all=[]; failures=[]
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(process_symbol,r,cfg,btc):r.symbol for _,r in universe.iterrows()}
        for fut in as_completed(futs):
            s=futs[fut]
            try:
                _,ev,samp,err=fut.result()
                if len(ev): event_parts.append(ev)
                sample_rows_all.extend(samp)
                if err: failures.append({"symbol":s,"error":err})
                print(s,"events",len(ev),"samples",len(samp),flush=True)
            except Exception as e:
                failures.append({"symbol":s,"error":repr(e)}); print("ERROR",s,repr(e),flush=True)
    events=pd.concat(event_parts,ignore_index=True) if event_parts else pd.DataFrame()
    samples=pd.DataFrame(sample_rows_all)
    events.to_csv(OUT/"events.csv",index=False); samples.to_csv(OUT/"precursor_samples.csv",index=False); pd.DataFrame(failures).to_csv(OUT/"failures.csv",index=False)
    summ=event_summary(events); summ.to_csv(OUT/"event_summary.csv",index=False)
    ct=contrasts(samples); ct.to_csv(OUT/"feature_contrasts.csv",index=False)
    st=stable_features(ct); st.to_csv(OUT/"stable_features.csv",index=False)
    write_analysis(summ,st)
    manifest={
      "study":"Explosive Move Discovery Round 1",
      "status":"RESEARCH ONLY - no scheduled task or live strategy edits",
      "event_definition":{"target_pct":TARGET,"stop_pct":STOP,"max_observation_hours":MAX_HOURS,"dedupe_cooldown_hours_after_target":EVENT_COOLDOWN_HOURS},
      "universe":"Current top-100 Binance USDT spot pairs by current 24h quote volume; top-50 separately reported; stablecoins, leveraged tokens and known unavailable symbols excluded.",
      "survivorship_caveat":"Current-universe selection omits historically important delisted/now-illiquid assets and is not a point-in-time historical top-100.",
      "history":{"start":START,"end":END,"interval":"1h"},
      "precursor_lead_hours":LEADS,
      "controls":"Up to 3 same-symbol/same-year non-event controls per event, excluding timestamps within +/-30d of a clean event.",
      "features":FEATURES,
      "splits":{"discovery":"2019-2022","validation":"2023-2024","holdout":"2025-2026"},
      "next_stage":"WATCH prediction days early, then ENTRY trigger before +5% movement."
    }
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print("EVENTS",len(events),"SAMPLES",len(samples),"FAILURES",len(failures))
    print(summ.to_string(index=False))
    print(st[(st.same_direction_validation)&(st.same_direction_holdout)].groupby("lead_hours").head(5).to_string(index=False))

if __name__=="__main__":
    main()
