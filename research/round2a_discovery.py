from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from binance_data import load_symbol

FEATURES = [
    "ret_6h", "ret_24h", "ret_72h", "rs_24h", "rs_72h",
    "range_24h", "range_72h", "range_120h", "atr24_pct",
    "vol_ratio_6h", "vol_ratio_24h", "vol_ratio_72h",
    "trade_ratio_6h", "trade_ratio_24h", "trade_ratio_72h",
    "taker_6h", "taker_24h", "taker_trend_24h",
    "close_location_24h", "dist_high_72h", "dist_high_168h",
    "higher_lows_24h", "green_share_24h", "btc_ret_24h", "btc_ret_72h",
    "btc_atr24_pct"
]


def feature_frame(df: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().set_index("time"); b = btc.copy().set_index("time")
    for h in (6, 24, 72): x[f"ret_{h}h"] = x.close.pct_change(h)
    x["rs_24h"] = x.ret_24h - b.close.pct_change(24).reindex(x.index)
    x["rs_72h"] = x.ret_72h - b.close.pct_change(72).reindex(x.index)
    for h in (24, 72, 120):
        hi=x.high.shift(1).rolling(h).max(); lo=x.low.shift(1).rolling(h).min(); x[f"range_{h}h"]=hi/lo-1
    tr=pd.concat([(x.high-x.low),(x.high-x.close.shift()).abs(),(x.low-x.close.shift()).abs()],axis=1).max(axis=1)
    x["atr24_pct"]=tr.rolling(24).mean()/x.close
    base_vol=x.volume.shift(1).rolling(72).mean(); base_trades=x.trades.shift(1).rolling(72).mean()
    for h in (6,24,72):
        x[f"vol_ratio_{h}h"]=x.volume.shift(1).rolling(h).mean()/base_vol
        x[f"trade_ratio_{h}h"]=x.trades.shift(1).rolling(h).mean()/base_trades
    taker=x.taker_buy_base/x.volume.replace(0,np.nan)
    x["taker_6h"]=taker.shift(1).rolling(6).mean(); x["taker_24h"]=taker.shift(1).rolling(24).mean()
    x["taker_trend_24h"]=x.taker_6h-taker.shift(7).rolling(18).mean()
    hi24=x.high.shift(1).rolling(24).max(); lo24=x.low.shift(1).rolling(24).min()
    x["close_location_24h"]=(x.close-lo24)/(hi24-lo24)
    x["dist_high_72h"]=x.close/x.high.shift(1).rolling(72).max()-1
    x["dist_high_168h"]=x.close/x.high.shift(1).rolling(168).max()-1
    lows=x.low.shift(1); x["higher_lows_24h"]=(lows>lows.shift(6)).rolling(24).mean()
    x["green_share_24h"]=(x.close>x.open).shift(1).rolling(24).mean()
    x["btc_ret_24h"]=b.close.pct_change(24).reindex(x.index); x["btc_ret_72h"]=b.close.pct_change(72).reindex(x.index)
    btr=pd.concat([(b.high-b.low),(b.high-b.close.shift()).abs(),(b.low-b.close.shift()).abs()],axis=1).max(axis=1)
    x["btc_atr24_pct"]=(btr.rolling(24).mean()/b.close).reindex(x.index)
    return x.reset_index()


def discover_events(symbol: str, df: pd.DataFrame, ff: pd.DataFrame) -> pd.DataFrame:
    future_high=df.high.shift(-1).rolling(168,min_periods=24).max().shift(-167)
    future_low=df.low.shift(-1).rolling(72,min_periods=12).min().shift(-71)
    upside=future_high/df.close-1; adverse=future_low/df.close-1
    eligible=(ff.time>=pd.Timestamp("2019-01-08",tz="UTC")) & ff[FEATURES].notna().all(axis=1)
    winners=eligible & (upside>=0.20) & (adverse>=-0.08); controls=eligible & (upside<=0.08)
    keep_w=[]; last=-9999
    for i in np.flatnonzero(winners.to_numpy()):
        if i-last>=120: keep_w.append(i); last=i
    ctrl_idx=np.flatnonzero(controls.to_numpy()); used=set(); rows=[]
    for i in keep_w:
        rows.append({"symbol":symbol,"kind":"RALLY","time":ff.time.iloc[i],"future_upside_7d":upside.iloc[i],"future_adverse_3d":adverse.iloc[i],**{f:ff[f].iloc[i] for f in FEATURES}})
        if len(ctrl_idx):
            dt=np.abs((ff.time.iloc[ctrl_idx]-ff.time.iloc[i]).dt.total_seconds().to_numpy()); mask=dt<=45*86400
            cand=ctrl_idx[mask] if mask.any() else ctrl_idx[ff.time.iloc[ctrl_idx] < ff.time.iloc[i]]
            if len(cand):
                order=sorted(cand,key=lambda j:abs((ff.time.iloc[j]-ff.time.iloc[i]).total_seconds()))
                j=next((j for j in order if j not in used),order[0]); used.add(j)
                rows.append({"symbol":symbol,"kind":"CONTROL","time":ff.time.iloc[j],"future_upside_7d":upside.iloc[j],"future_adverse_3d":adverse.iloc[j],**{f:ff[f].iloc[j] for f in FEATURES}})
    return pd.DataFrame(rows)


def summary(events: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for f in FEATURES:
        a=events.loc[events.kind=="RALLY",f].dropna(); b=events.loc[events.kind=="CONTROL",f].dropna()
        if len(a)<5 or len(b)<5: continue
        pooled=np.sqrt((a.var()+b.var())/2)
        rows.append({"feature":f,"rally_median":a.median(),"control_median":b.median(),"median_diff":a.median()-b.median(),"standardised_diff":(a.mean()-b.mean())/pooled if pooled else np.nan,"rallies":len(a),"controls":len(b)})
    return pd.DataFrame(rows).sort_values("standardised_diff",key=lambda s:s.abs(),ascending=False)


def process_symbol(symbol: str, cfg: dict, btc: pd.DataFrame) -> pd.DataFrame:
    print("Discovering",symbol,flush=True)
    df=btc.copy() if symbol=="BTCUSDT" else load_symbol(symbol,cfg["interval"],cfg["start"],cfg["end"])
    if len(df)<500: return pd.DataFrame()
    return discover_events(symbol,df,feature_frame(df,btc))


def main():
    cfg=json.loads(Path("research/config.json").read_text()); out=Path("research/results/round2a"); out.mkdir(parents=True,exist_ok=True)
    btc=load_symbol("BTCUSDT",cfg["interval"],cfg["start"],cfg["end"]); all_events=[]
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures={ex.submit(process_symbol,s,cfg,btc):s for s in cfg["symbols"]}
        for fut in as_completed(futures):
            try:
                ev=fut.result()
                if not ev.empty: all_events.append(ev)
            except Exception as e:
                print("FAILED",futures[fut],repr(e),flush=True)
    events=pd.concat(all_events,ignore_index=True) if all_events else pd.DataFrame(); events.to_csv(out/"rally_control_events.csv",index=False)
    sm=summary(events); sm.to_csv(out/"feature_contrasts.csv",index=False)
    if not events.empty:
        events.assign(year=pd.to_datetime(events.time).dt.year).groupby(["year","kind"]).size().unstack(fill_value=0).to_csv(out/"event_counts_by_year.csv")
        events[events.kind=="RALLY"].sort_values("future_upside_7d",ascending=False).head(200).to_csv(out/"top_rallies.csv",index=False)
    Path(out/"manifest.json").write_text(json.dumps({"study":"Round 2A retrospective rally pattern discovery","label":"hindsight labels only; all features measured at event start","rally":"future high >= +20% within 168h and first-72h adverse >= -8%","control":"future high <= +8% within 168h, same-symbol time-matched","cooldown_hours":120,"features":FEATURES},indent=2))
    print(sm.head(30).to_string(index=False)); print("events",len(events),"rallies",int((events.kind=="RALLY").sum()) if len(events) else 0)

if __name__=="__main__": main()
