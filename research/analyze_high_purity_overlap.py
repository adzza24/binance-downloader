from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "research/cache/high_purity_model_source"
SPEC = ROOT / "research/high_purity_signal_features.json"
MODELS = ROOT / "research/results/daily_return_harvest/high_purity_runtime/models"
OUT = ROOT / "research/results/daily_return_harvest/high_purity_runtime/overlap_analysis"
TYPES = ["WINNER_EVENT", "HARD_NEGATIVE", "SAME_COIN_RANDOM", "SAME_TIME_CONTROL"]


def parts():
    out=[]
    for shard in range(4):
        out += sorted((SRC / f"drh2-canonical-{shard}").glob("samples_part_*.parquet"))
    if not out: raise RuntimeError("canonical shards missing")
    return out


def load_model(path):
    with path.open("rb") as f: return pickle.load(f)


def parse_time(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, utc=True)
    n=pd.to_numeric(s,errors="coerce")
    return pd.to_datetime(n,unit="ms",utc=True) if n.notna().mean()>.99 else pd.to_datetime(s,utc=True)


def exit_hours(row):
    t=row.target3_hour; s=row.stop2_hour
    t=float(t) if pd.notna(t) else np.nan; s=float(s) if pd.notna(s) else np.nan
    if bool(row.w3_2_24h) and np.isfinite(t) and t < 24:
        return min(24.0, t+1.0), "TARGET"
    if np.isfinite(s) and s < 24 and (not np.isfinite(t) or s <= t or t >= 24):
        return min(24.0, s+1.0), "STOP"
    return 24.0, "TIMEOUT"


def concurrency_stats(g):
    # conservative hourly availability: position stays occupied through its hit candle.
    ev=[]
    for r in g.itertuples():
        ev.append((r.entry_time,1)); ev.append((r.exit_time,-1))
    ev.sort(key=lambda x:(x[0],x[1]))  # exits before entries at identical time
    active=mx=0; hist=[]
    for _,d in ev:
        active+=d; mx=max(mx,active); hist.append(active)
    return mx


def slot_capture(g, slots):
    free=[pd.Timestamp.min.tz_localize("UTC") for _ in range(slots)]
    accepted=0; blocked=0
    # deterministic: same-time signals sorted family then score desc then symbol.
    famrank={"A":0,"B":1,"C":2}
    z=g.assign(_fr=g.family.map(famrank)).sort_values(["entry_time","_fr","score","symbol"],ascending=[True,True,False,True])
    for r in z.itertuples():
        available=[i for i,t in enumerate(free) if t <= r.entry_time]
        if not available:
            blocked+=1; continue
        i=available[0]; free[i]=r.exit_time; accepted+=1
    return accepted,blocked


def main():
    spec=json.loads(SPEC.read_text()); th=spec["thresholds"]
    a,af=load_model(MODELS/"event_hgb_model.pkl")
    b,bf=load_model(MODELS/"familyB_model.pkl")
    c,cf=load_model(MODELS/"familyC15_model.pkl")
    if bf != cf: raise RuntimeError("B/C feature mismatch")
    feat=list(dict.fromkeys(af+bf))
    cols=["sample_id","symbol","sample_type","split","decision_time","w3_2_24h","target3_hour","stop2_hour","end_net_24h"]+feat
    frames=[pd.read_parquet(p,columns=cols) for p in parts()]
    df=pd.concat(frames,ignore_index=True); df["decision_time"]=parse_time(df.decision_time)
    for x in feat: df[x]=pd.to_numeric(df[x],errors="coerce").astype(np.float32)
    base=df[df.sample_type.isin(TYPES)].copy()
    base["score_a"]=a.predict_proba(base[af])[:,1]
    base["score_b"]=b.predict_proba(base[bf])[:,1]
    base["score_c"]=c.predict_proba(base[bf])[:,1]
    aa=base.score_a.to_numpy()>=th["A"]
    bb=(~aa)&(base.score_b.to_numpy()>=th["B"])
    cc=(~aa)&(~bb)&(base.score_c.to_numpy()>=th["C"])
    sig=base[aa|bb|cc].copy()
    sig["family"]=np.where(aa[aa|bb|cc],"A",np.where(bb[aa|bb|cc],"B","C"))
    sig["score"]=np.where(sig.family.eq("A"),sig.score_a,np.where(sig.family.eq("B"),sig.score_b,sig.score_c))
    sig["entry_time"]=sig.decision_time+pd.Timedelta(hours=1)
    ex=sig.apply(exit_hours,axis=1,result_type="expand"); sig["occupied_hours"]=ex[0].astype(float);sig["exit_reason"]=ex[1]
    sig["exit_time"]=sig.entry_time+pd.to_timedelta(sig.occupied_hours,unit="h")
    sig["year"]=sig.entry_time.dt.year

    # exact duplicate check for live-equivalent coin/time signals in sampled rows.
    dup=sig.duplicated(["symbol","decision_time"],keep=False)
    dedup=sig.sort_values(["symbol","decision_time","sample_type"]).drop_duplicates(["symbol","decision_time"],keep="first").copy()

    rows=[]
    for year,g in dedup.groupby("year"):
        span_days=(g.entry_time.max().floor("D")-g.entry_time.min().floor("D")).days+1
        gaps=g.entry_time.sort_values().diff().dropna().dt.total_seconds()/3600
        same_hour=g.groupby("entry_time").size()
        mx=concurrency_stats(g)
        a1,b1=slot_capture(g,1); a2,b2=slot_capture(g,2); a3,b3=slot_capture(g,3)
        rows.append({
            "year":int(year),"signals":len(g),"span_days":span_days,"signals_per_day":len(g)/span_days,
            "w3_rate":float(pd.to_numeric(g.w3_2_24h).mean()),"median_occupied_hours":float(g.occupied_hours.median()),
            "p75_occupied_hours":float(g.occupied_hours.quantile(.75)),"p90_occupied_hours":float(g.occupied_hours.quantile(.90)),
            "same_hour_signal_pct":float((same_hour.reindex(g.entry_time).to_numpy()>1).mean()),
            "hours_with_2plus":int((same_hour>=2).sum()),"hours_with_3plus":int((same_hour>=3).sum()),"max_same_hour":int(same_hour.max()),
            "median_gap_hours":float(gaps.median()) if len(gaps) else np.nan,"pct_gap_lt_6h":float((gaps<6).mean()) if len(gaps) else np.nan,
            "pct_gap_lt_12h":float((gaps<12).mean()) if len(gaps) else np.nan,"pct_gap_lt_24h":float((gaps<24).mean()) if len(gaps) else np.nan,
            "max_concurrent_if_all_taken":mx,
            "one_slot_capture_pct":a1/len(g),"one_slot_blocked":b1,
            "two_slot_capture_pct":a2/len(g),"two_slot_blocked":b2,
            "three_slot_capture_pct":a3/len(g),"three_slot_blocked":b3,
            "targets":int((g.exit_reason=="TARGET").sum()),"stops":int((g.exit_reason=="STOP").sum()),"timeouts":int((g.exit_reason=="TIMEOUT").sum()),
        })
    summary=pd.DataFrame(rows)
    OUT.mkdir(parents=True,exist_ok=True)
    summary.to_csv(OUT/"overlap_by_year.csv",index=False)
    dedup[["symbol","decision_time","entry_time","exit_time","occupied_hours","exit_reason","family","score","sample_type","split","w3_2_24h","target3_hour","stop2_hour","end_net_24h"]].sort_values("entry_time").to_csv(OUT/"sampled_signal_timeline.csv",index=False)

    md=["# High-Purity A/B/C Sampled-Signal Overlap Analysis","","Research-only. This measures overlap among the constructed event/control signal rows; it is not yet the full-universe portfolio replay.","",f"- Raw signal rows: {len(sig)}",f"- Exact duplicate symbol/times: {int(dup.sum())} rows",f"- Unique sampled coin/time signals: {len(dedup)}","","## By year","","```text",summary.to_string(index=False),"```","","## Interpretation","","`occupied_hours` uses the saved W3 target/stop hit-hour. Capital is conservatively treated as unavailable until the end of the candle containing the first resolving hit. If neither resolves within 24h, the position occupies 24h.","","Slot capture is first-available with deterministic A>B>C / score ordering only to estimate capacity. It is not a proposed ranking rule."]
    (OUT/"OVERLAP_ANALYSIS.md").write_text("\n".join(md)+"\n")
    print(summary.to_string(index=False))

if __name__=="__main__": main()
