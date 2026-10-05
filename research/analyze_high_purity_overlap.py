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
    if not out:
        raise RuntimeError("canonical shards missing")
    return out


def load_model(path):
    with path.open("rb") as f:
        return pickle.load(f)


def parse_time(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, utc=True)
    n=pd.to_numeric(s,errors="coerce")
    return pd.to_datetime(n,unit="ms",utc=True) if n.notna().mean()>.99 else pd.to_datetime(s,utc=True)


def max_concurrency(g):
    ev=[]
    for r in g.itertuples():
        ev.append((r.entry_time,1))
        ev.append((r.exit_time,-1))
    ev.sort(key=lambda x:(x[0],x[1]))  # free capacity before same-time entries
    active=mx=0
    for _,d in ev:
        active+=d
        mx=max(mx,active)
    return mx


def slot_capture(g, slots):
    free=[pd.Timestamp.min.tz_localize("UTC") for _ in range(slots)]
    accepted=blocked=0
    famrank={"A":0,"B":1,"C":2}
    z=g.assign(_fr=g.family.map(famrank)).sort_values(
        ["entry_time","_fr","score","symbol"], ascending=[True,True,False,True]
    )
    for r in z.itertuples():
        available=[i for i,t in enumerate(free) if t <= r.entry_time]
        if not available:
            blocked+=1
            continue
        i=available[0]
        free[i]=r.exit_time
        accepted+=1
    return accepted,blocked


def main():
    spec=json.loads(SPEC.read_text())
    th=spec["thresholds"]
    a,af=load_model(MODELS/"event_hgb_model.pkl")
    b,bf=load_model(MODELS/"familyB_model.pkl")
    c,cf=load_model(MODELS/"familyC15_model.pkl")
    if bf != cf:
        raise RuntimeError("B/C feature mismatch")

    feat=list(dict.fromkeys(af+bf))
    cols=["sample_id","symbol","sample_type","split","decision_time","w3_2_24h","end_net_24h"]+feat
    df=pd.concat([pd.read_parquet(p,columns=cols) for p in parts()],ignore_index=True)
    df["decision_time"]=parse_time(df.decision_time)
    for x in feat:
        df[x]=pd.to_numeric(df[x],errors="coerce").astype(np.float32)

    base=df[df.sample_type.isin(TYPES)].copy()
    base["score_a"]=a.predict_proba(base[af])[:,1]
    base["score_b"]=b.predict_proba(base[bf])[:,1]
    base["score_c"]=c.predict_proba(base[bf])[:,1]
    aa=base.score_a.to_numpy()>=th["A"]
    bb=(~aa)&(base.score_b.to_numpy()>=th["B"])
    cc=(~aa)&(~bb)&(base.score_c.to_numpy()>=th["C"])
    mask=aa|bb|cc

    sig=base[mask].copy()
    sig["family"]=np.where(aa[mask],"A",np.where(bb[mask],"B","C"))
    sig["score"]=np.where(sig.family.eq("A"),sig.score_a,np.where(sig.family.eq("B"),sig.score_b,sig.score_c))
    sig["entry_time"]=sig.decision_time+pd.Timedelta(hours=1)
    sig["occupied_hours"]=24.0  # deliberate conservative upper bound
    sig["exit_time"]=sig.entry_time+pd.Timedelta(hours=24)
    sig["year"]=sig.entry_time.dt.year

    dup_rows=int(sig.duplicated(["symbol","decision_time"],keep=False).sum())
    dedup=sig.sort_values(["symbol","decision_time","sample_type"]).drop_duplicates(
        ["symbol","decision_time"],keep="first"
    ).copy()

    rows=[]
    for year,g in dedup.groupby("year"):
        span_days=(g.entry_time.max().floor("D")-g.entry_time.min().floor("D")).days+1
        ordered=g.entry_time.sort_values()
        gaps=ordered.diff().dropna().dt.total_seconds()/3600
        same=g.groupby("entry_time").size()
        a1,b1=slot_capture(g,1)
        a2,b2=slot_capture(g,2)
        a3,b3=slot_capture(g,3)
        rows.append({
            "year":int(year),
            "signals":len(g),
            "span_days":span_days,
            "signals_per_day":len(g)/span_days,
            "w3_rate":float(pd.to_numeric(g.w3_2_24h).mean()),
            "hours_with_2plus":int((same>=2).sum()),
            "hours_with_3plus":int((same>=3).sum()),
            "max_same_hour":int(same.max()),
            "signal_rows_in_multi_signal_hours_pct":float((same.reindex(g.entry_time).to_numpy()>1).mean()),
            "median_gap_hours":float(gaps.median()) if len(gaps) else np.nan,
            "pct_gap_lt_6h":float((gaps<6).mean()) if len(gaps) else np.nan,
            "pct_gap_lt_12h":float((gaps<12).mean()) if len(gaps) else np.nan,
            "pct_gap_lt_24h":float((gaps<24).mean()) if len(gaps) else np.nan,
            "max_concurrent_24h_hold":max_concurrency(g),
            "one_slot_capture_pct":a1/len(g),"one_slot_blocked":b1,
            "two_slot_capture_pct":a2/len(g),"two_slot_blocked":b2,
            "three_slot_capture_pct":a3/len(g),"three_slot_blocked":b3,
        })

    summary=pd.DataFrame(rows)
    OUT.mkdir(parents=True,exist_ok=True)
    summary.to_csv(OUT/"overlap_by_year.csv",index=False)
    dedup[["symbol","decision_time","entry_time","exit_time","family","score","sample_type","split","w3_2_24h","end_net_24h"]].sort_values("entry_time").to_csv(OUT/"sampled_signal_timeline.csv",index=False)

    md=[
        "# High-Purity A/B/C Sampled-Signal Overlap Analysis",
        "",
        "Research-only. This measures overlap among the constructed event/control signal rows; it is not yet the full-universe portfolio replay.",
        "",
        f"- Raw signal rows: {len(sig)}",
        f"- Duplicate symbol/time rows participating in an exact duplicate: {dup_rows}",
        f"- Unique sampled coin/time signals: {len(dedup)}",
        "- Occupancy assumption: every accepted position holds its slot for the full 24h. This is intentionally conservative because compact samples do not preserve target/stop hit hour.",
        "",
        "## By year",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "Slot capture uses deterministic A>B>C / score ordering only to estimate capacity. It is not a proposed ranking rule.",
    ]
    (OUT/"OVERLAP_ANALYSIS.md").write_text("\n".join(md)+"\n")
    print("raw_signals",len(sig),"duplicates",dup_rows,"unique",len(dedup))
    print(summary.to_string(index=False))

if __name__=="__main__":
    main()
