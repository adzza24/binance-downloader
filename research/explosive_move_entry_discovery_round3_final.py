from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

import explosive_move_entry_discovery_round2 as r2
from binance_data import load_symbol

OUT = Path("research/results/explosive_move_entry_discovery_round3_entry_only")
START = r2.START
END = r2.END
MAX_HOURS = 30 * 24
MAX_ENTRY_ADVANCE = 0.05
WATCH_MAX = {"CAPITULATION": 72, "BASE": 168}
TRAIN_MAX_PER_SYMBOL_ARCH = 4000
RF_THRESHOLDS = [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.925,0.95,0.97,0.98,0.99]
LEVELS = [5,10,20,30,50]
DYNAMIC_FEATURES = [
    "watch_age_hours","watch_return","watch_low_drawdown",
    "rebound_from_watch_low","watch_high_advance","distance_from_watch_high",
]
MODEL_FEATURES = [f for f in r2.MODEL_FEATURES if f != "hours_since_anchor"] + DYNAMIC_FEATURES

SIMPLE_RULES = {
    "CAPITULATION": [
        "REBOUND2_VOL11",           # unchanged Round 2 rule: 12h-low rebound >=2%, 3h vol >=1.10
        "WATCH_REBOUND2_VOL11",     # >=2% rebound from low made since causal watch began
        "WATCH_REBOUND1_VOL11",
        "REBOUND2_TRADE11",
        "EMA6_RECLAIM_VOL11",
        "BREAK3_VOL12",
    ],
    "BASE": [
        "EXPAND3_VOL12","BREAK3_VOL12","BREAK6_VOL13","VOL_TRADE13","EMA6_RECLAIM_VOL11",
    ],
}


def split_years(split: str) -> tuple[int,int]:
    if split == "DISCOVERY_2019_2022":
        return 2019,2022
    if split == "VALIDATION_2023_2024":
        return 2023,2024
    return 2025,2026


def precompute_forward_extrema(df: pd.DataFrame) -> tuple[np.ndarray,np.ndarray]:
    high = pd.Series(df.high.to_numpy(dtype=float))
    low = pd.Series(df.low.to_numpy(dtype=float))
    fh = high.iloc[::-1].rolling(MAX_HOURS,min_periods=1).max().iloc[::-1].to_numpy()
    fl = low.iloc[::-1].rolling(MAX_HOURS,min_periods=1).min().iloc[::-1].to_numpy()
    return fh,fl


def fast_path(entry_i: int, entry: float, fh: np.ndarray, fl: np.ndarray) -> dict:
    mfe = float(fh[entry_i] / entry - 1)
    mae = float(fl[entry_i] / entry - 1)
    out = {
        "mfe_30d":mfe,
        "mae_30d":mae,
        "capped_upside20":min(max(mfe,0.0),0.20),
    }
    for p in LEVELS:
        out[f"hit{p}_any_30d"] = int(mfe >= p/100)
    return out


def full_path(df: pd.DataFrame, entry_i: int, entry: float) -> dict:
    end=min(len(df),entry_i+MAX_HOURS)
    min_low=entry; max_high=entry
    hit_idx={p:None for p in LEVELS}
    adverse5=None; adverse10=None; mae_before20=np.nan
    for j in range(entry_i,end):
        lo=float(df.low.iloc[j]); hi=float(df.high.iloc[j])
        min_low=min(min_low,lo); max_high=max(max_high,hi)
        if adverse5 is None and lo<=entry*.95: adverse5=j
        if adverse10 is None and lo<=entry*.90: adverse10=j
        for p in LEVELS:
            if hit_idx[p] is None and hi>=entry*(1+p/100):
                hit_idx[p]=j
                if p==20: mae_before20=min_low/entry-1
    h20=hit_idx[20]
    out={
        "mfe_30d":max_high/entry-1,
        "mae_30d":min_low/entry-1,
        "capped_upside20":min(max(max_high/entry-1,0.0),.20),
        "mae_before_20":mae_before20,
        "hit20_before_adverse5":int(h20 is not None and (adverse5 is None or h20<adverse5)),
        "hit20_before_adverse10":int(h20 is not None and (adverse10 is None or h20<adverse10)),
    }
    for p in LEVELS:
        out[f"hit{p}_any_30d"]=int(hit_idx[p] is not None)
        out[f"hours_to_{p}"]=(hit_idx[p]-entry_i) if hit_idx[p] is not None else np.nan
    return out


def simple_pass(row: pd.Series, rule: str) -> bool:
    if rule=="REBOUND2_VOL11":
        return bool(row.rebound_12h_low>=.02 and row.volume_ratio_3h>=1.10)
    if rule=="WATCH_REBOUND2_VOL11":
        return bool(row.rebound_from_watch_low>=.02 and row.volume_ratio_3h>=1.10)
    if rule=="WATCH_REBOUND1_VOL11":
        return bool(row.rebound_from_watch_low>=.01 and row.volume_ratio_3h>=1.10)
    if rule=="REBOUND2_TRADE11":
        return bool(row.rebound_from_watch_low>=.02 and row.trade_ratio_3h>=1.10)
    if rule=="EMA6_RECLAIM_VOL11":
        return bool(row.close_vs_ema6>=0 and row.volume_ratio_3h>=1.10)
    if rule=="BREAK3_VOL12":
        return bool(row.close_vs_prev3h_high>=0 and row.volume_ratio_current>=1.20)
    if rule=="EXPAND3_VOL12":
        return bool(row.ret_3h>=.01 and row.volume_ratio_current>=1.20)
    if rule=="BREAK6_VOL13":
        return bool(row.close_vs_prev6h_high>=0 and row.volume_ratio_current>=1.30)
    if rule=="VOL_TRADE13":
        return bool(row.volume_ratio_current>=1.30 and row.trade_ratio_current>=1.30 and row.ret_1h>0)
    return False


def candidate_dict(symbol: str, arch: str, split: str, episode_id: str,
                   df: pd.DataFrame, ff: pd.DataFrame, watch_i: int, j: int,
                   watch_low: float, watch_high: float, cfg: dict,
                   fh: np.ndarray, fl: np.ndarray) -> dict | None:
    if j+1>=len(df): return None
    entry=float(df.open.iloc[j+1])*(1+float(cfg["slippage_rate"]))
    anchor=float(df.close.iloc[watch_i])
    advance=entry/anchor-1
    if advance>MAX_ENTRY_ADVANCE: return None
    src=ff.iloc[j]
    close=float(df.close.iloc[j])
    row={
        "episode_id":episode_id,"symbol":symbol,"archetype":arch,"split":split,
        "watch_start":pd.Timestamp(df.time.iloc[watch_i]),
        "decision_time":pd.Timestamp(df.time.iloc[j]),
        "entry_time":pd.Timestamp(df.time.iloc[j+1]),
        "entry_index":j+1,
        "watch_anchor_price":anchor,"entry_price":entry,
        "entry_progress_from_watch_pct":advance,
        "watch_age_hours":j-watch_i,
        "watch_return":close/anchor-1,
        "watch_low_drawdown":watch_low/anchor-1,
        "rebound_from_watch_low":close/watch_low-1,
        "watch_high_advance":watch_high/anchor-1,
        "distance_from_watch_high":close/watch_high-1,
    }
    for f in r2.MODEL_FEATURES:
        if f=="hours_since_anchor": continue
        v=src.get(f,np.nan)
        row[f]=float(v) if pd.notna(v) else np.nan
    row.update(fast_path(j+1,entry,fh,fl))
    return row


def iter_episode_rows(symbol: str, arch: str, split: str, df: pd.DataFrame, ff: pd.DataFrame,
                      params: pd.Series, cfg: dict, fh: np.ndarray, fl: np.ndarray):
    y0,y1=split_years(split)
    years=pd.to_datetime(df.time,utc=True).dt.year.to_numpy()
    valid=np.flatnonzero((years>=y0)&(years<=y1))
    if len(valid)<2: return
    i=max(int(valid[0]),2160); end=min(int(valid[-1]),len(df)-2)
    ep=0
    while i<=end:
        if not r2.watch_gate(ff.iloc[i],arch,params):
            i+=1; continue
        ep+=1
        watch_i=i; anchor=float(df.close.iloc[i])
        deadline=min(end,watch_i+WATCH_MAX[arch])
        watch_low=float(df.low.iloc[watch_i]); watch_high=float(df.high.iloc[watch_i])
        rows=[]
        actual_end=watch_i
        for j in range(watch_i,deadline+1):
            actual_end=j
            watch_low=min(watch_low,float(df.low.iloc[j]))
            watch_high=max(watch_high,float(df.high.iloc[j]))
            eid=f"{symbol}:{arch}:{split}:{pd.Timestamp(df.time.iloc[watch_i]).isoformat()}:{ep}"
            row=candidate_dict(symbol,arch,split,eid,df,ff,watch_i,j,watch_low,watch_high,cfg,fh,fl)
            if row is not None:
                rows.append(row)
            if float(df.close.iloc[j])/anchor-1 > MAX_ENTRY_ADVANCE:
                break
        if rows:
            yield rows
        i=actual_end+24


def sample_training_rows(universe: pd.DataFrame, data: dict, arch_model, gates: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out=[]; pmap={r.archetype:r for _,r in gates.iterrows()}
    for symbol in universe.symbol:
        if symbol not in data: continue
        df,ff0=data[symbol]; ff=r2.apply_arch_to_feature_frame(ff0,arch_model); fh,fl=precompute_forward_extrema(df)
        for arch in ["CAPITULATION","BASE"]:
            local=[]
            for rows in iter_episode_rows(symbol,arch,"DISCOVERY_2019_2022",df,ff,pmap[arch],cfg,fh,fl):
                local.extend(rows)
            if len(local)>TRAIN_MAX_PER_SYMBOL_ARCH:
                take=np.linspace(0,len(local)-1,TRAIN_MAX_PER_SYMBOL_ARCH,dtype=int)
                local=[local[k] for k in take]
            out.extend(local)
            print("TRAIN",symbol,arch,len(local),flush=True)
    return pd.DataFrame(out)


def fit_models(train: pd.DataFrame):
    models={}; imp_rows=[]
    for arch in ["CAPITULATION","BASE"]:
        z=train[train.archetype==arch].copy()
        if len(z)<200 or z.hit20_any_30d.nunique()<2: continue
        imp=SimpleImputer(strategy="median")
        X=imp.fit_transform(z[MODEL_FEATURES])
        clf=RandomForestClassifier(
            n_estimators=300,max_depth=6,min_samples_leaf=50,
            class_weight="balanced_subsample",random_state=44,n_jobs=-1
        )
        clf.fit(X,z.hit20_any_30d)
        models[arch]=(imp,clf)
        for f,v in zip(MODEL_FEATURES,clf.feature_importances_):
            imp_rows.append({"archetype":arch,"feature":f,"importance":float(v)})
    return models,pd.DataFrame(imp_rows)


def evaluate_all(universe: pd.DataFrame, data: dict, arch_model, gates: pd.DataFrame,
                 cfg: dict, models: dict) -> pd.DataFrame:
    records=[]; pmap={r.archetype:r for _,r in gates.iterrows()}
    for split in ["VALIDATION_2023_2024","CONFIRMATION_2025_2026"]:
        for symbol in universe.symbol:
            if symbol not in data: continue
            df,ff0=data[symbol]; ff=r2.apply_arch_to_feature_frame(ff0,arch_model); fh,fl=precompute_forward_extrema(df)
            for arch in ["CAPITULATION","BASE"]:
                model=models.get(arch)
                for rows in iter_episode_rows(symbol,arch,split,df,ff,pmap[arch],cfg,fh,fl):
                    ep=pd.DataFrame(rows)
                    for rule in SIMPLE_RULES[arch]:
                        mask=ep.apply(lambda x:simple_pass(x,rule),axis=1).to_numpy()
                        ids=np.flatnonzero(mask)
                        if len(ids):
                            d=dict(rows[int(ids[0])]); d.update({"method":"SIMPLE","trigger":rule})
                            records.append(d)
                    if model is not None:
                        imp,clf=model
                        probs=clf.predict_proba(imp.transform(ep[MODEL_FEATURES]))[:,1]
                        for th in RF_THRESHOLDS:
                            ids=np.flatnonzero(probs>=th)
                            if len(ids):
                                k=int(ids[0]); d=dict(rows[k])
                                d.update({"method":"RF","trigger":f"RF>={th:.3f}","model_probability":float(probs[k])})
                                records.append(d)
            print("EVAL",split,symbol,flush=True)
    return pd.DataFrame(records)


def metric_dict(g: pd.DataFrame) -> dict:
    if g.empty: return {"alerts":0}
    return {
        "alerts":len(g),
        "hit5_any_rate":float(g.hit5_any_30d.mean()),
        "hit10_any_rate":float(g.hit10_any_30d.mean()),
        "hit20_any_rate":float(g.hit20_any_30d.mean()),
        "hit30_any_rate":float(g.hit30_any_30d.mean()),
        "hit50_any_rate":float(g.hit50_any_30d.mean()),
        "mean_capped_upside20":float(g.capped_upside20.mean()),
        "median_mfe_30d":float(g.mfe_30d.median()),
        "median_mae_30d":float(g.mae_30d.median()),
        "median_entry_progress_pct":float(g.entry_progress_from_watch_pct.median()),
    }


def candidate_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (method,arch,trigger,split),g in records.groupby(["method","archetype","trigger","split"]):
        r={"method":method,"archetype":arch,"trigger":trigger,"split":split}; r.update(metric_dict(g)); rows.append(r)
    return pd.DataFrame(rows)


def choose(summary: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for method in ["SIMPLE","RF"]:
        for arch in ["CAPITULATION","BASE"]:
            v=summary[(summary.method==method)&(summary.archetype==arch)&
                      (summary.split=="VALIDATION_2023_2024")&(summary.alerts>=25)].copy()
            if v.empty: continue
            v=v.sort_values(["hit20_any_rate","mean_capped_upside20","median_mae_30d"],ascending=[False,False,False])
            r=v.iloc[0]
            rows.append({
                "method":method,"archetype":arch,"trigger":r.trigger,
                "validation_alerts":int(r.alerts),
                "validation_hit20_any_rate":float(r.hit20_any_rate),
                "validation_mean_capped_upside20":float(r.mean_capped_upside20),
                "validation_median_mae_30d":float(r.median_mae_30d),
            })
    return pd.DataFrame(rows)


def selected_records(records: pd.DataFrame, selected: pd.DataFrame, data: dict) -> pd.DataFrame:
    chunks=[]
    for _,s in selected.iterrows():
        z=records[(records.method==s.method)&(records.archetype==s.archetype)&(records.trigger==s.trigger)].copy()
        chunks.append(z)
    if not chunks: return pd.DataFrame()
    out=pd.concat(chunks,ignore_index=True)
    detailed=[]
    for _,r in out.iterrows():
        df,_=data[r.symbol]
        detailed.append(full_path(df,int(r.entry_index),float(r.entry_price)))
    for i,d in enumerate(detailed):
        for k,v in d.items(): out.at[i,k]=v
    return out


def final_summary(alerts: pd.DataFrame) -> pd.DataFrame:
    spans={
        "VALIDATION_2023_2024":(pd.Timestamp("2023-01-01",tz="UTC"),pd.Timestamp("2024-12-31 23:00",tz="UTC")),
        "CONFIRMATION_2025_2026":(pd.Timestamp("2025-01-01",tz="UTC"),pd.Timestamp("2026-08-31 23:00",tz="UTC")),
    }
    rows=[]
    for (method,arch,trigger,split),g in alerts.groupby(["method","archetype","trigger","split"]):
        start,end=spans[split]; weeks=(end-start).total_seconds()/(7*24*3600)
        r={"method":method,"archetype":arch,"trigger":trigger,"split":split,"alerts_per_week":len(g)/weeks}
        r.update(metric_dict(g))
        r.update({
            "hit20_before_adverse5_rate":float(g.hit20_before_adverse5.mean()),
            "hit20_before_adverse10_rate":float(g.hit20_before_adverse10.mean()),
            "median_mae_before_20":float(g.loc[g.hit20_any_30d==1,"mae_before_20"].median()) if (g.hit20_any_30d==1).any() else np.nan,
            "median_hours_to_20":float(g.loc[g.hit20_any_30d==1,"hours_to_20"].median()) if (g.hit20_any_30d==1).any() else np.nan,
            "max_coin_alert_share":float(g.symbol.value_counts(normalize=True).iloc[0]),
        })
        rows.append(r)
    return pd.DataFrame(rows)


def write_analysis(universe,train,selected,summary):
    lines=[
        "# Explosive Move Entry Discovery Round 3 - Entry Only",
        "",
        "**Status:** RESEARCH ONLY. No Early Breakout automation or Crypto Live Strategy Specification changes.",
        "",
        "## Design",
        "",
        "This round isolates entry quality from exit design. CAPITULATION and BASE watches are generated causally from discovery-era context gates. During each watch, entry conditions are evaluated on every completed hourly candle and an entry is represented by the next hourly open plus configured slippage. Entries more than 5% above the watch anchor are excluded.",
        "",
        "The Random Forest is trained only to predict whether price reaches +20% at any time within the following 30 days. No stop, trailing stop or profit-taking rule is part of the training label or trigger selection.",
        "",
        "The primary entry metrics are +5/+10/+20/+30/+50 forward hit rates, 30-day MFE, 30-day MAE, target timing, and upside opportunity capped at +20%. +20-before--5 and +20-before--10 are reported only as secondary path-quality diagnostics.",
        "",
        "The unchanged Round 2 REBOUND2_VOL11 rule is explicitly retained, alongside a separate watch-relative rebound variant.",
        "",
        "2025-2026 is labelled CONFIRMATION, not untouched holdout, because prior rounds have already exposed results from that period.",
        "",
        f"Universe: {len(universe)} frozen eligible Binance USDT pairs. RF discovery training rows after deterministic per-symbol/archetype cap: {len(train)}.",
        "",
        "## Validation-selected configurations",
        "",
        "| Method | Archetype | Trigger | Validation alerts | +20 any | Mean capped +20 opportunity | Median 30d MAE |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _,r in selected.iterrows():
        lines.append(f"| {r.method} | {r.archetype} | {r.trigger} | {int(r.validation_alerts)} | {100*r.validation_hit20_any_rate:.1f}% | {100*r.validation_mean_capped_upside20:.1f}% | {100*r.validation_median_mae_30d:.1f}% |")
    lines += [
        "",
        "## End-to-end causal entry results",
        "",
        "| Method | Archetype | Split | Alerts/wk | +10 any | +20 any | +30 any | Mean capped +20 opp. | Median MFE | Median MAE | +20 before -5 | +20 before -10 | Median h to +20 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _,r in summary.sort_values(["method","archetype","split"]).iterrows():
        lines.append(
            f"| {r.method} | {r.archetype} | {r.split} | {r.alerts_per_week:.2f} | "
            f"{100*r.hit10_any_rate:.1f}% | {100*r.hit20_any_rate:.1f}% | {100*r.hit30_any_rate:.1f}% | "
            f"{100*r.mean_capped_upside20:.1f}% | {100*r.median_mfe_30d:.1f}% | {100*r.median_mae_30d:.1f}% | "
            f"{100*r.hit20_before_adverse5_rate:.1f}% | {100*r.hit20_before_adverse10_rate:.1f}% | {r.median_hours_to_20:.1f} |"
        )
    lines += [
        "",
        "## Guardrails",
        "",
        "- Mean capped +20 opportunity is not a realised return. It credits available upside up to +20% without pretending an exit captured it.",
        "- MAE is shown beside MFE so a later rally after severe drawdown cannot masquerade as a clean entry.",
        "- The -5% and -10% columns are diagnostics only; they do not select the trigger.",
        "- No trailing stop or profit-taking logic is optimised here. Exit research remains separate.",
        "- RF discovery rows are deterministically capped per symbol/archetype for tractability; validation and confirmation evaluate every eligible hourly candle inside each causal watch.",
        "- The frozen current-liquidity universe retains survivorship bias.",
        "- No result is promoted to the live strategy by this workflow.",
        "",
    ]
    (OUT/"ANALYSIS.md").write_text("\n".join(lines))


def main():
    cfg=json.loads(Path("research/config.json").read_text())
    OUT.mkdir(parents=True,exist_ok=True)
    universe=r2.load_frozen_top50(); universe.to_csv(OUT/"universe.csv",index=False)
    btc=load_symbol("BTCUSDT","1h",START,END)

    data={}; failures=[]
    def load_one(symbol):
        df=load_symbol(symbol,"1h",START,END)
        if len(df)<3000: return symbol,None,None,f"insufficient_history:{len(df)}"
        return symbol,df,r2.build_features(df,btc),None
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(load_one,s):s for s in universe.symbol}
        for fut in as_completed(futs):
            s=futs[fut]
            try:
                symbol,df,ff,err=fut.result()
                if err: failures.append({"symbol":symbol,"kind":"load","error":err})
                else: data[symbol]=(df,ff)
                print("LOAD",symbol,0 if df is None else len(df),err or "OK",flush=True)
            except Exception as e:
                failures.append({"symbol":s,"kind":"load","error":repr(e)}); print("ERROR",s,repr(e),flush=True)

    anchors,recheck=r2.build_anchor_cohort(universe,data); failures.extend(recheck)
    arch_model,arch_summary=r2.fit_archetypes(anchors)
    anchors=r2.assign_archetypes(anchors,arch_model)
    gates=r2.discovery_watch_gate_params(anchors,arch_model)
    arch_summary.to_csv(OUT/"archetype_summary.csv",index=False)
    gates.to_csv(OUT/"watch_gate_params.csv",index=False)

    train=sample_training_rows(universe,data,arch_model,gates,cfg)
    train.to_csv(OUT/"rf_training_sample.csv.gz",index=False,compression="gzip")
    models,imps=fit_models(train)
    imps.to_csv(OUT/"model_feature_importance.csv",index=False)

    records=evaluate_all(universe,data,arch_model,gates,cfg,models)
    records.to_csv(OUT/"all_trigger_first_alerts.csv.gz",index=False,compression="gzip")
    cand_summary=candidate_summary(records); cand_summary.to_csv(OUT/"candidate_trigger_summary.csv",index=False)
    selected=choose(cand_summary); selected.to_csv(OUT/"selected_triggers.csv",index=False)

    alerts=selected_records(records,selected,data)
    alerts.to_csv(OUT/"selected_entry_alerts.csv.gz",index=False,compression="gzip")
    summary=final_summary(alerts); summary.to_csv(OUT/"causal_entry_summary.csv",index=False)

    if not alerts.empty:
        alerts.assign(year=pd.to_datetime(alerts.entry_time,utc=True).dt.year).groupby(
            ["method","archetype","trigger","split","year"]
        ).agg(
            alerts=("symbol","size"),hit20_any_rate=("hit20_any_30d","mean"),
            mean_capped_upside20=("capped_upside20","mean"),
            median_mfe_30d=("mfe_30d","median"),median_mae_30d=("mae_30d","median"),
            hit20_before_adverse5_rate=("hit20_before_adverse5","mean"),
            hit20_before_adverse10_rate=("hit20_before_adverse10","mean"),
        ).reset_index().to_csv(OUT/"year_summary.csv",index=False)

        alerts.groupby(["method","archetype","trigger","split","symbol"]).agg(
            alerts=("symbol","size"),hit20_any_rate=("hit20_any_30d","mean"),
            mean_capped_upside20=("capped_upside20","mean"),median_mae_30d=("mae_30d","median"),
        ).reset_index().to_csv(OUT/"coin_summary.csv",index=False)

    pd.DataFrame(failures).to_csv(OUT/"failures.csv",index=False)
    write_analysis(universe,train,selected,summary)

    manifest={
        "study":"Explosive Move Entry Discovery Round 3 - Entry Only",
        "status":"RESEARCH ONLY - no live strategy or automation edits",
        "history":{"start":START,"end":END,"interval":"1h"},
        "universe":"Frozen Round 2 eligible top-50 universe; survivorship bias remains.",
        "watch_logic":"Discovery-derived CAPITULATION/BASE context gates; evaluate every completed hour inside watch; next-hour open plus slippage; entry <=5% above watch anchor.",
        "primary_target":"+20% reached at any time within 30d, regardless of interim drawdown.",
        "primary_metrics":["hit5/10/20/30/50_any_30d","mfe_30d","mae_30d","capped_upside20","hours_to_targets","mae_before_20"],
        "secondary_diagnostics":["hit20_before_adverse5","hit20_before_adverse10"],
        "exit_policy":"None for training/selection. No fixed stop, trailing stop or profit-taking optimisation.",
        "rf":{"trees":300,"max_depth":6,"min_samples_leaf":50,"training":"2019-2022 causal watch-hours, deterministic max 4000 rows per symbol/archetype","thresholds":RF_THRESHOLDS,"selection":"2023-2024 validation only, >=25 alerts; rank +20-any then capped-upside then less-negative MAE"},
        "period_note":"2025-2026 is confirmation rather than untouched holdout because earlier research exposed the period.",
        "costs":{"fee_rate":cfg["fee_rate"],"slippage_rate":cfg["slippage_rate"]},
    }
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))

    print("TRAIN_ROWS",len(train),"TRIGGER_RECORDS",len(records),"SELECTED",flush=True)
    print(selected.to_string(index=False) if not selected.empty else "none",flush=True)
    print("SUMMARY",flush=True)
    print(summary.to_string(index=False) if not summary.empty else "none",flush=True)


if __name__=="__main__":
    main()
