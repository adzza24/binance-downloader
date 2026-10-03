from __future__ import annotations

import json, math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from binance_data import load_symbol
from early_breakout_round1 import REFERENCE_SYMBOLS, TASK_EXCLUSIONS, feature_frame, raw_candidates, dedupe_episodes

OUT = Path("research/results/early_breakout_round3")
POS = 300.0
CHECKPOINTS = [0, 4, 8, 12, 24]
TRAIL_GIVEBACK = 0.60
MIN_VALIDATION_ACCEPTED = 50
MIN_VALIDATION_COVERAGE = 0.05

STATIC_FEATURES = [
    "stage_breakout", "base_range_120h", "range_24h", "range_72h",
    "atr24_pct", "atr120_pct", "distance_to_resistance",
    "rs_24h", "rs_72h", "volume_ratio_6h", "trade_ratio_6h",
    "volume_ratio_current", "trade_ratio_current", "taker_buy_6h",
    "taker_buy_current", "ret_24h", "ret_72h", "prior_30d_return",
    "distance_from_30d_high", "log_quote_volume_24h",
]
DYNAMIC_FEATURES = [
    "cp_return", "cp_mfe", "cp_mae", "cp_close_vs_resistance",
    "cp_above_resistance_share", "cp_green_share", "cp_drawdown_from_high",
    "cp_volume_ratio_current", "cp_trade_ratio_current", "cp_taker_buy_current",
    "cp_volume_ratio_6h", "cp_trade_ratio_6h", "cp_taker_buy_6h",
    "cp_rs_24h", "cp_rs_72h",
]
EXIT_VARIANTS = ["FULL_TRAIL_7", "HALF20_RUNNER_7", "LADDER10_20_RUNNER_7", "FIXED10_5", "FIXED20_7"]

def split_name(year: int) -> str:
    if year <= 2022:
        return "TRAIN_2019_2022"
    if year <= 2024:
        return "VALIDATION_2023_2024"
    return "HOLDOUT_2025_2026"

def net_pnl(entry: float, exits: list[tuple[float, float]], cfg: dict) -> float:
    fee = float(cfg["fee_rate"]); slip = float(cfg["slippage_rate"]); qty = POS / entry
    out = -POS * fee
    for frac, raw in exits:
        px = raw * (1 - slip)
        proceeds = qty * frac * px
        out += proceeds - POS * frac - proceeds * fee
    return float(out)

def first_hit(arr: np.ndarray) -> int | None:
    z = np.flatnonzero(arr)
    return int(z[0]) if len(z) else None

def future_label(df: pd.DataFrame, start_i: int, entry: float, days: int = 7):
    start_t = pd.Timestamp(df.time.iloc[start_i])
    end_t = start_t + pd.Timedelta(days=days)
    w = df[(df.time >= start_t) & (df.time < end_t)]
    complete = bool(len(df) and pd.Timestamp(df.time.iloc[-1]) >= end_t - pd.Timedelta(hours=1))
    if w.empty or not complete:
        return False, False, np.nan, np.nan, False
    hi = w.high.to_numpy(float); lo = w.low.to_numpy(float)
    hit20 = first_hit(hi >= entry * 1.20)
    hit10 = first_hit(hi >= entry * 1.10)
    stop7 = first_hit(lo <= entry * 0.93)
    hit20_before_stop = hit20 is not None and (stop7 is None or hit20 < stop7)
    hit10_before_stop = hit10 is not None and (stop7 is None or hit10 < stop7)
    return hit20_before_stop, hit10_before_stop, float(hi.max()/entry-1), float(lo.min()/entry-1), True

def checkpoint_row(df, ff, sig, entry_i, cp, cfg):
    decision_i = entry_i + cp
    delayed_i = decision_i + 1 if cp > 0 else entry_i
    if delayed_i >= len(df) or decision_i >= len(df):
        return None
    decision_t = pd.Timestamp(df.time.iloc[decision_i])
    delayed_t = pd.Timestamp(df.time.iloc[delayed_i])
    entry = float(df.open.iloc[delayed_i]) * (1 + float(cfg["slippage_rate"]))
    res = float(sig.resistance_168h)
    hist = df.iloc[entry_i:decision_i+1]
    ffrow = ff.iloc[decision_i]
    label20, label10, future_mfe, future_mae, complete = future_label(df, delayed_i, entry, 7)
    if not complete:
        return None

    row = {
        "signal_id": sig.signal_id, "symbol": sig.symbol, "original_stage": sig.stage,
        "checkpoint_h": cp, "decision_time": decision_t, "entry_time": delayed_t,
        "entry_index": delayed_i, "entry_price": entry, "year": int(delayed_t.year),
        "split": split_name(int(delayed_t.year)),
        "target20_before_stop7_7d": int(label20), "target10_before_stop7_7d": int(label10),
        "future_mfe_7d": future_mfe, "future_mae_7d": future_mae,
        "stage_breakout": int(sig.stage == "BREAKOUT_ATTEMPT"),
        "base_range_120h": float(sig.base_range_120h), "range_24h": float(sig.range_24h),
        "range_72h": float(sig.range_72h), "atr24_pct": float(sig.atr24_pct),
        "atr120_pct": float(sig.atr120_pct), "distance_to_resistance": float(sig.distance_to_resistance),
        "rs_24h": float(sig.rs_24h), "rs_72h": float(sig.rs_72h),
        "volume_ratio_6h": float(sig.volume_ratio_6h), "trade_ratio_6h": float(sig.trade_ratio_6h),
        "volume_ratio_current": float(sig.volume_ratio_current), "trade_ratio_current": float(sig.trade_ratio_current),
        "taker_buy_6h": float(sig.taker_buy_6h), "taker_buy_current": float(sig.taker_buy_current),
        "ret_24h": float(sig.ret_24h), "ret_72h": float(sig.ret_72h),
        "prior_30d_return": float(sig.prior_30d_return) if pd.notna(sig.prior_30d_return) else np.nan,
        "distance_from_30d_high": float(sig.distance_from_30d_high) if pd.notna(sig.distance_from_30d_high) else np.nan,
        "log_quote_volume_24h": float(np.log1p(sig.quote_volume_24h)),
    }
    for f in DYNAMIC_FEATURES:
        row[f] = 0.0
    row["cp_close_vs_resistance"] = float(df.close.iloc[decision_i] / res - 1)
    row["cp_volume_ratio_current"] = float(ffrow.volume_ratio_current)
    row["cp_trade_ratio_current"] = float(ffrow.trade_ratio_current)
    row["cp_taker_buy_current"] = float(ffrow.taker_buy_current)
    row["cp_volume_ratio_6h"] = float(ffrow.volume_ratio_6h)
    row["cp_trade_ratio_6h"] = float(ffrow.trade_ratio_6h)
    row["cp_taker_buy_6h"] = float(ffrow.taker_buy_6h)
    row["cp_rs_24h"] = float(ffrow.rs_24h)
    row["cp_rs_72h"] = float(ffrow.rs_72h)
    if cp > 0:
        highs = hist.high.to_numpy(float); lows = hist.low.to_numpy(float); closes = hist.close.to_numpy(float)
        original_entry = float(sig.entry_price); peak = highs.max()
        row["cp_return"] = float(closes[-1] / original_entry - 1)
        row["cp_mfe"] = float(peak / original_entry - 1)
        row["cp_mae"] = float(lows.min() / original_entry - 1)
        row["cp_above_resistance_share"] = float((closes >= res).mean())
        row["cp_green_share"] = float((hist.close.to_numpy(float) > hist.open.to_numpy(float)).mean())
        row["cp_drawdown_from_high"] = float(closes[-1] / peak - 1)
    return row

def simulate_exit(df, start_i, entry, cfg, variant):
    start_t = pd.Timestamp(df.time.iloc[start_i]); end_t = start_t + pd.Timedelta(days=365)
    stop = entry * (0.95 if variant == "FIXED10_5" else 0.93)
    peak = entry; active = False; exits=[]; remaining=1.0; first_partial=False; second_partial=False
    for j in range(start_i, len(df)):
        b=df.iloc[j]; t=pd.Timestamp(b.time); lo,hi,cl=float(b.low),float(b.high),float(b.close)
        if t >= end_t:
            exits.append((remaining,float(b.open)))
            return {"pnl_usdt":net_pnl(entry,exits,cfg),"exit_reason":"365D_CAP","duration_hours":(t-start_t).total_seconds()/3600}
        if variant == "FIXED10_5":
            target=entry*1.10; hit_s=lo<=stop; hit_t=hi>=target
            if hit_s: exits=[(1.0,stop)]; reason="STOP_AMBIGUOUS" if hit_t else "STOP"
            elif hit_t: exits=[(1.0,target)]; reason="TARGET10"
            else: continue
            return {"pnl_usdt":net_pnl(entry,exits,cfg),"exit_reason":reason,"duration_hours":(t-start_t).total_seconds()/3600}
        if variant == "FIXED20_7":
            target=entry*1.20; hit_s=lo<=stop; hit_t=hi>=target
            if hit_s: exits=[(1.0,stop)]; reason="STOP_AMBIGUOUS" if hit_t else "STOP"
            elif hit_t: exits=[(1.0,target)]; reason="TARGET20"
            else: continue
            return {"pnl_usdt":net_pnl(entry,exits,cfg),"exit_reason":reason,"duration_hours":(t-start_t).total_seconds()/3600}
        if lo <= stop:
            exits.append((remaining,stop))
            return {"pnl_usdt":net_pnl(entry,exits,cfg),"exit_reason":"RUNNER_STOP" if active else "INITIAL_STOP","duration_hours":(t-start_t).total_seconds()/3600}
        if variant == "HALF20_RUNNER_7" and not first_partial and hi >= entry*1.20:
            exits.append((0.50,entry*1.20)); remaining=0.50; first_partial=True; active=True; stop=max(stop,entry)
        elif variant == "LADDER10_20_RUNNER_7":
            if not first_partial and hi >= entry*1.10:
                exits.append((0.25,entry*1.10)); remaining=0.75; first_partial=True; active=True; stop=max(stop,entry)
            if first_partial and not second_partial and hi >= entry*1.20:
                exits.append((0.25,entry*1.20)); remaining=0.50; second_partial=True
        if variant == "FULL_TRAIL_7" and not active and hi >= entry*1.05:
            active=True; stop=max(stop,entry)
        peak=max(peak,hi)
        if hi >= entry*11:
            exits.append((remaining,entry*11))
            return {"pnl_usdt":net_pnl(entry,exits,cfg),"exit_reason":"TAKE_PROFIT_1000","duration_hours":(t-start_t).total_seconds()/3600}
        if active and peak >= entry*1.30:
            stop=max(stop,entry*1.10)
            candidate=entry+(1-TRAIL_GIVEBACK)*(peak-entry)
            stop=max(stop,min(candidate,cl*.999))
    exits.append((remaining,float(df.close.iloc[-1])))
    return {"pnl_usdt":net_pnl(entry,exits,cfg),"exit_reason":"DATA_END","duration_hours":(pd.Timestamp(df.time.iloc[-1])-start_t).total_seconds()/3600}

def process_symbol(symbol, cfg, btc):
    df=load_symbol(symbol,cfg["interval"],cfg["start"],cfg["end"])
    if len(df)<900:
        return symbol,[],{},f"insufficient_history:{len(df)}"
    ff=feature_frame(df,btc); sigs=dedupe_episodes(symbol,ff,raw_candidates(ff),cfg)
    idx=pd.Series(df.index.to_numpy(),index=df.time).to_dict(); rows=[]; data={}
    for _,sig in sigs.iterrows():
        ei=idx.get(pd.Timestamp(sig.entry_time))
        if ei is None: continue
        for cp in CHECKPOINTS:
            row=checkpoint_row(df,ff,sig,ei,cp,cfg)
            if row is not None:
                rows.append(row); data[(row["signal_id"],cp)]=(df,int(row["entry_index"]),float(row["entry_price"]))
    return symbol,rows,data,None

def feature_contrasts(samples):
    rows=[]
    for cp in CHECKPOINTS:
        z=samples[(samples.checkpoint_h==cp)&(samples.split=="TRAIN_2019_2022")]
        feats=STATIC_FEATURES + ([] if cp==0 else DYNAMIC_FEATURES)
        for f in feats:
            w=pd.to_numeric(z.loc[z.target20_before_stop7_7d==1,f],errors="coerce").dropna()
            l=pd.to_numeric(z.loc[z.target20_before_stop7_7d==0,f],errors="coerce").dropna()
            if len(w)<10 or len(l)<10: continue
            pooled=np.sqrt((w.var()+l.var())/2)
            rows.append({"checkpoint_h":cp,"feature":f,"winner_median":w.median(),"other_median":l.median(),"standardised_diff":(w.mean()-l.mean())/pooled if pooled else np.nan,"winner_n":len(w),"other_n":len(l)})
    return pd.DataFrame(rows)

def select_models(samples):
    pred_parts=[]; model_rows=[]; imp_rows=[]
    for cp in CHECKPOINTS:
        z=samples[samples.checkpoint_h==cp].copy()
        feats=STATIC_FEATURES + ([] if cp==0 else DYNAMIC_FEATURES)
        tr=z[z.split=="TRAIN_2019_2022"]; va=z[z.split=="VALIDATION_2023_2024"]; ho=z[z.split=="HOLDOUT_2025_2026"]
        if min(len(tr),len(va),len(ho))<40: continue
        imputer=SimpleImputer(strategy="median")
        Xtr=imputer.fit_transform(tr[feats]); Xva=imputer.transform(va[feats]); Xho=imputer.transform(ho[feats])
        clf=RandomForestClassifier(n_estimators=400,max_depth=5,min_samples_leaf=25,class_weight="balanced",random_state=42,n_jobs=-1)
        clf.fit(Xtr,tr.target20_before_stop7_7d)
        pva=clf.predict_proba(Xva)[:,1]; pho=clf.predict_proba(Xho)[:,1]
        try: auc_va=roc_auc_score(va.target20_before_stop7_7d,pva)
        except ValueError: auc_va=np.nan
        try: auc_ho=roc_auc_score(ho.target20_before_stop7_7d,pho)
        except ValueError: auc_ho=np.nan
        candidates=[]
        for th in np.unique(np.quantile(pva,np.linspace(.50,.95,19))):
            a=pva>=th; n=int(a.sum())
            if n<MIN_VALIDATION_ACCEPTED or n/len(va)<MIN_VALIDATION_COVERAGE: continue
            candidates.append((float(va.target20_before_stop7_7d.to_numpy()[a].mean()),float(va.target10_before_stop7_7d.to_numpy()[a].mean()),n/len(va),float(th),n))
        if not candidates: continue
        candidates.sort(key=lambda x:(x[0],x[1],x[2]),reverse=True)
        precision,hit10,cov,th,n=candidates[0]
        for split,frame,p,auc in [("VALIDATION_2023_2024",va,pva,auc_va),("HOLDOUT_2025_2026",ho,pho,auc_ho)]:
            a=p>=th
            pred_parts.append(frame.assign(model_probability=p,model_selected=a,model_checkpoint_h=cp))
            model_rows.append({"checkpoint_h":cp,"split":split,"threshold":th,"auc":auc,"samples":len(frame),"selected":int(a.sum()),"coverage":float(a.mean()),"precision20_before_stop7":float(frame.target20_before_stop7_7d.to_numpy()[a].mean()) if a.sum() else np.nan,"hit10_before_stop7":float(frame.target10_before_stop7_7d.to_numpy()[a].mean()) if a.sum() else np.nan})
        for f,v in zip(feats,clf.feature_importances_):
            imp_rows.append({"checkpoint_h":cp,"feature":f,"importance":float(v)})
    return pd.concat(pred_parts,ignore_index=True),pd.DataFrame(model_rows),pd.DataFrame(imp_rows)

def evaluate_selected(preds, path_data, cfg):
    trade_rows=[]
    for _,r in preds[preds.model_selected==True].iterrows():
        key=(r.signal_id,int(r.checkpoint_h))
        if key not in path_data: continue
        df,ei,entry=path_data[key]
        for v in EXIT_VARIANTS:
            out=simulate_exit(df,ei,entry,cfg,v)
            trade_rows.append({"signal_id":r.signal_id,"symbol":r.symbol,"original_stage":r.original_stage,"checkpoint_h":int(r.checkpoint_h),"split":r.split,"year":int(r.year),"variant":v,**out})
    trades=pd.DataFrame(trade_rows); rows=[]
    if trades.empty: return trades,pd.DataFrame()
    for keys,g in trades.groupby(["checkpoint_h","split","variant"]):
        x=g.pnl_usdt.to_numpy(float); pos=x[x>0].sum(); neg=x[x<0].sum()
        rows.append({"checkpoint_h":keys[0],"split":keys[1],"variant":keys[2],"trades":len(g),"mean_pnl_usdt":float(x.mean()),"mean_return_pct":float(x.mean()/POS),"median_pnl_usdt":float(np.median(x)),"profit_factor":float(pos/abs(neg)) if neg<0 else math.inf,"win_rate":float((x>0).mean()),"median_duration_hours":float(g.duration_hours.median()),"positive_10pct_mean":bool(x.mean()>=30)})
    return trades,pd.DataFrame(rows)

def heuristic_summary(samples):
    rows=[]
    for cp in [4,8,12,24]:
        z=samples[samples.checkpoint_h==cp].copy()
        for split in ["VALIDATION_2023_2024","HOLDOUT_2025_2026"]:
            zz=z[z.split==split]
            masks={
                "ABOVE_RES": zz.cp_close_vs_resistance>=0,
                "ABOVE_RES_RET_POS": (zz.cp_close_vs_resistance>=0)&(zz.cp_return>0),
                "ABOVE_RES_RET2": (zz.cp_close_vs_resistance>=0)&(zz.cp_return>=.02),
                "HOLD50_RET_POS": (zz.cp_above_resistance_share>=.5)&(zz.cp_return>0),
                "HOLD50_VOL12_TAKER53": (zz.cp_above_resistance_share>=.5)&(zz.cp_volume_ratio_6h>=1.2)&(zz.cp_taker_buy_6h>=.53),
            }
            for name,m in masks.items():
                sel=zz[m]
                rows.append({"checkpoint_h":cp,"split":split,"rule":name,"samples":len(zz),"selected":len(sel),"coverage":len(sel)/len(zz) if len(zz) else np.nan,"precision20_before_stop7":sel.target20_before_stop7_7d.mean() if len(sel) else np.nan,"hit10_before_stop7":sel.target10_before_stop7_7d.mean() if len(sel) else np.nan,"mean_future_mfe_7d":sel.future_mfe_7d.mean() if len(sel) else np.nan})
    return pd.DataFrame(rows)

def write_analysis(modelq, strat, cohort_n):
    hold=strat[strat.split=="HOLDOUT_2025_2026"].sort_values(["mean_pnl_usdt","profit_factor"],ascending=False)
    best=hold.head(10); targets=hold[hold.mean_pnl_usdt>=30]
    mq=modelq[modelq.split=="HOLDOUT_2025_2026"].sort_values("precision20_before_stop7",ascending=False)
    lines=["# Early Breakout Round 3 - Signal Quality & Confirmation Research","",
        "**Status:** RESEARCH ONLY. No scheduled-task or live-strategy edits.","",
        f"Frozen Round 1 cohort regenerated: {cohort_n:,} complete checkpoint-0 samples.","",
        "## Objective","",
        "Identify which signals are most likely to produce +20% within seven days before a -7% stop, and test whether delayed confirmation can raise mean P&L toward +30 USDT on a 300 USDT trade.","",
        "Temporal design: train 2019-2022, choose probability thresholds on 2023-2024 validation, report 2025-2026 as untouched holdout.","",
        "## Holdout model quality","",
        "| Checkpoint | Selected | Coverage | +20% precision | +10% hit | AUC |","|---:|---:|---:|---:|---:|---:|"]
    for _,r in mq.iterrows():
        lines.append(f"| {int(r.checkpoint_h)}h | {int(r.selected)} | {100*r.coverage:.1f}% | {100*r.precision20_before_stop7:.1f}% | {100*r.hit10_before_stop7:.1f}% | {r.auc:.3f} |")
    lines += ["","## Best holdout strategy results","",
        "| Checkpoint | Exit | Trades | Mean P&L | Mean return | PF | Win rate |","|---:|---|---:|---:|---:|---:|---:|"]
    for _,r in best.iterrows():
        lines.append(f"| {int(r.checkpoint_h)}h | {r.variant} | {int(r.trades)} | {r.mean_pnl_usdt:.2f} | {100*r.mean_return_pct:.2f}% | {r.profit_factor:.2f} | {100*r.win_rate:.1f}% |")
    lines += ["","## 10% mean-return hurdle",""]
    if len(targets):
        lines.append(f"{len(targets)} holdout checkpoint/exit combinations met or exceeded +30 USDT mean P&L. They require year/coin concentration checks before any strategy promotion.")
    else:
        lines.append("No holdout checkpoint/exit combination reached +30 USDT mean P&L. The research does not support claiming a 10% average-trade strategy from this signal family yet.")
    lines += ["","## Notes","",
        "- Delayed checkpoints are genuine confirmation entries: no capital is committed before the checkpoint.",
        "- The target is +20% within 7 days before a -7% adverse move.",
        "- Exit tests include fixed +10%, fixed +20%, full runner, 50%@+20 plus runner, and 25%@+10 + 25%@+20 + 50% runner.",
        "- Results are independent 300 USDT trades, not a capital-constrained portfolio simulation.",""]
    (OUT/"ANALYSIS.md").write_text("\n".join(lines))

def main():
    cfg=json.loads(Path("research/config.json").read_text()); OUT.mkdir(parents=True,exist_ok=True)
    universe=list(dict.fromkeys(cfg["symbols"]+REFERENCE_SYMBOLS))
    universe=[s for s in universe if s not in TASK_EXCLUSIONS and s!="BTCUSDT"]
    btc=load_symbol("BTCUSDT",cfg["interval"],cfg["start"],cfg["end"])
    allrows=[]; path_data={}; failures=[]
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs={ex.submit(process_symbol,s,cfg,btc):s for s in universe}
        for fut in as_completed(futs):
            s=futs[fut]
            try:
                _,rows,data,err=fut.result(); allrows.extend(rows); path_data.update(data)
                if err: failures.append({"symbol":s,"error":err})
                print(s,"samples",len(rows),flush=True)
            except Exception as e:
                failures.append({"symbol":s,"error":repr(e)}); print("ERROR",s,repr(e),flush=True)
    samples=pd.DataFrame(allrows); cohort_n=int((samples.checkpoint_h==0).sum())
    samples.to_csv(OUT/"samples.csv",index=False); pd.DataFrame(failures).to_csv(OUT/"failures.csv",index=False)
    feature_contrasts(samples).to_csv(OUT/"feature_contrasts.csv",index=False)
    preds,modelq,imp=select_models(samples)
    modelq.to_csv(OUT/"model_quality.csv",index=False); imp.to_csv(OUT/"feature_importance.csv",index=False)
    heuristic_summary(samples).to_csv(OUT/"heuristic_confirmation.csv",index=False)
    trades,strat=evaluate_selected(preds,path_data,cfg)
    trades.to_csv(OUT/"selected_trades.csv",index=False); strat.to_csv(OUT/"strategy_summary.csv",index=False)
    if not trades.empty:
        trades.groupby(["checkpoint_h","variant","year"]).agg(trades=("pnl_usdt","size"),mean_pnl_usdt=("pnl_usdt","mean"),combined_pnl_usdt=("pnl_usdt","sum"),win_rate=("pnl_usdt",lambda x:(x>0).mean())).reset_index().to_csv(OUT/"strategy_year_summary.csv",index=False)
        trades.groupby(["checkpoint_h","variant","symbol"]).agg(trades=("pnl_usdt","size"),mean_pnl_usdt=("pnl_usdt","mean"),combined_pnl_usdt=("pnl_usdt","sum")).reset_index().to_csv(OUT/"strategy_coin_summary.csv",index=False)
    write_analysis(modelq,strat,cohort_n)
    manifest={
        "study":"Early Breakout Round 3 - signal quality and confirmation",
        "status":"RESEARCH ONLY - no live task/strategy edits",
        "entry_source":"Frozen Round 1 mechanical Early Breakout operationalisation",
        "target":"+20% within 7 days after candidate entry before touching -7%",
        "checkpoints_hours":CHECKPOINTS,
        "temporal_split":{"train":"2019-2022","validation":"2023-2024","holdout":"2025-2026 through dataset end"},
        "model":{"type":"RandomForestClassifier","n_estimators":400,"max_depth":5,"min_samples_leaf":25,"class_weight":"balanced","threshold_selection":"Highest validation +20% precision subject to >=50 selected and >=5% coverage; holdout untouched."},
        "exit_variants":EXIT_VARIANTS,
        "runner":"7% initial stop; full runner moves to BE at +5%; runner variants use +10% minimum floor once peak >=+30%, then 60% accumulated-gain giveback; max 365d.",
        "position_usdt":POS,
        "costs":{"fee_rate":cfg["fee_rate"],"slippage_rate":cfg["slippage_rate"]},
        "dataset":{"start":cfg["start"],"end":cfg["end"],"interval":cfg["interval"],"universe":universe},
        "success_hurdle":"Holdout mean P&L >=30 USDT on a 300 USDT trade (10%) after fees/slippage."
    }
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print("SAMPLES",len(samples),"COHORT",cohort_n,"FAILURES",len(failures))
    print(modelq.to_string(index=False))
    print(strat[strat.split=="HOLDOUT_2025_2026"].sort_values("mean_pnl_usdt",ascending=False).head(20).to_string(index=False))

if __name__=="__main__":
    main()
