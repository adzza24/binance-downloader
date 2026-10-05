from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from high_purity_signal_watch import BASE_FEATURES, CONTEXT_FEATURES, DELTA_HOURS, add_cross_section_context
from replay_high_purity_2026 import trajectory_matrix

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "research/cache/full_stream_w3_source"
OUT = ROOT / "research/results/daily_return_harvest/full_stream_w3_discovery"
START = pd.Timestamp("2021-11-01", tz="UTC")
TRAIN_END = pd.Timestamp("2023-11-01", tz="UTC")
DEV_END = pd.Timestamp("2024-05-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
CONF_END = pd.Timestamp("2026-09-28", tz="UTC")
PRE_CONTEXT = START - pd.Timedelta(hours=169)
MIN_QV = 1_000_000.0
RNG = np.random.default_rng(20261005)
NEG_RANDOM_PER_SYMBOL_MONTH = 180
NEG_HARD_PER_SYMBOL_MONTH = 60
TARGETS = ("ANCHOR", "EARLY3", "W3")
FEATURES = list(dict.fromkeys(list(BASE_FEATURES) + list(CONTEXT_FEATURES) + [f"{f}__chg_{h}h" for f in list(BASE_FEATURES) + list(CONTEXT_FEATURES) for h in DELTA_HOURS]))


def shard_dirs():
    ds = [SRC / f"drh2-shard-{i}" for i in range(4)]
    missing = [str(d) for d in ds if not (d / "panel.parquet").exists()]
    if missing:
        raise RuntimeError(f"Missing full Round 2 shards: {missing}")
    return ds


def read_parts(name: str):
    ps = []
    for d in shard_dirs():
        p = d / name
        if p.exists():
            z = pd.read_parquet(p)
            if len(z): ps.append(z)
    return pd.concat(ps, ignore_index=True) if ps else pd.DataFrame()


def load_context():
    ps = []
    for d in shard_dirs():
        z = pd.read_parquet(d / "panel.parquet")
        z["time"] = pd.to_datetime(z.time, utc=True)
        ps.append(z[(z.time >= PRE_CONTEXT) & (z.time < CONF_END)])
    p = pd.concat(ps, ignore_index=True).drop_duplicates(["symbol", "time"]).sort_values(["time", "symbol"])
    return add_cross_section_context(p)


def symbol_files():
    out = []
    for d in shard_dirs():
        out.extend((p.name.removesuffix(".pkl.gz"), p) for p in (d / "features").glob("*.pkl.gz"))
    return sorted(out)


def ctx_for(context, symbol):
    z = context.loc[context.symbol.eq(symbol), ["time"] + list(CONTEXT_FEATURES)].copy()
    return z.drop_duplicates("time").set_index("time").sort_index()


def positive_maps(events, candidates):
    events = events.copy(); candidates = candidates.copy()
    events["anchor_time"] = pd.to_datetime(events.anchor_time, utc=True)
    candidates["decision_time"] = pd.to_datetime(candidates.decision_time, utc=True)
    a, e, w = {}, {}, {}
    for s, g in events.groupby("symbol"):
        g = g[(g.anchor_time >= START) & (g.anchor_time < TRAIN_END)]
        a[s] = set(g.anchor_time)
    for s, g in candidates.groupby("symbol"):
        g = g[(g.decision_time >= START) & (g.decision_time < TRAIN_END)].copy()
        e[s] = set(g.loc[g.hours_from_event_start <= 3, "decision_time"])
        chosen = []
        for _, q in g.groupby("event_id"):
            q = q.sort_values("decision_time")
            ids = sorted(set([0, len(q)//2, len(q)-1]))
            chosen.extend(pd.Timestamp(q.iloc[i].decision_time) for i in ids)
        w[s] = set(chosen)
    return a, e, w


def negative_times(ff):
    z = ff[(ff.time >= START) & (ff.time < TRAIN_END) & ff.eligible.fillna(False).astype(bool)].copy()
    if "account_excluded" in z: z = z[~z.account_excluded.fillna(False).astype(bool)]
    z = z[pd.to_numeric(z.w3_2_24h, errors="coerce").eq(0)]
    if z.empty: return set()
    z["month"] = z.time.dt.tz_localize(None).dt.to_period("M")
    ids = []
    for _, g in z.groupby("month", sort=False):
        hard = g[pd.to_numeric(g.mfe_24h, errors="coerce") >= 0.03]
        normal = g.drop(index=hard.index)
        if len(hard): ids += RNG.choice(hard.index.to_numpy(), min(NEG_HARD_PER_SYMBOL_MONTH, len(hard)), replace=False).tolist()
        if len(normal): ids += RNG.choice(normal.index.to_numpy(), min(NEG_RANDOM_PER_SYMBOL_MONTH, len(normal)), replace=False).tolist()
    return set(pd.to_datetime(ff.loc[sorted(set(ids)), "time"], utc=True))


def build_train(context, events, candidates):
    amap, emap, wmap = positive_maps(events, candidates)
    xs, rows = [], []
    for n, (symbol, path) in enumerate(symbol_files(), 1):
        ff = pd.read_pickle(path, compression="gzip").sort_values("time").drop_duplicates("time").reset_index(drop=True)
        ff["time"] = pd.to_datetime(ff.time, utc=True)
        neg = negative_times(ff); pa = amap.get(symbol, set()); pe = emap.get(symbol, set()); pw = wmap.get(symbol, set())
        times = sorted(neg | pa | pe | pw)
        if not times: continue
        x = trajectory_matrix(ff, ctx_for(context, symbol), FEATURES).reindex(pd.DatetimeIndex(times))
        ok = x.notna().sum(axis=1) >= int(len(FEATURES) * 0.80)
        x = x.loc[ok]
        if x.empty: continue
        xs.append(x.to_numpy(np.float32))
        for t in x.index:
            rows.append({"symbol": symbol, "time": t, "neg": t in neg, "ANCHOR": int(t in pa), "EARLY3": int(t in pe), "W3": int(t in pw)})
        print("TRAIN_SAMPLE", n, symbol, len(x), flush=True)
    X = np.concatenate(xs, axis=0)
    m = pd.DataFrame(rows)
    if len(X) != len(m): raise RuntimeError("training metadata mismatch")
    return X, m


def fit_models(X, meta):
    models, pop = {}, []
    for i, target in enumerate(TARGETS):
        use = meta.neg.to_numpy(bool) | meta[target].eq(1).to_numpy(bool)
        y = meta.loc[use, target].to_numpy(np.int8); Xt = X[use]
        model = LGBMClassifier(objective="binary", n_estimators=500, learning_rate=.035, num_leaves=31, min_child_samples=80,
                               subsample=.85, colsample_bytree=.70, reg_alpha=.5, reg_lambda=6, max_bin=127,
                               random_state=20261005+i, n_jobs=-1, verbosity=-1)
        model.fit(Xt, y, feature_name=FEATURES)
        models[target] = model
        pos = int(y.sum()); neg = int(len(y)-pos)
        pop.append({"target": target, "rows": len(y), "positives": pos, "negatives": neg, "negative_to_positive": neg/pos})
        print("FIT", target, len(y), pos, neg, flush=True)
    return models, pd.DataFrame(pop)


def score_stream(context, models):
    out = []
    for n, (symbol, path) in enumerate(symbol_files(), 1):
        ff = pd.read_pickle(path, compression="gzip").sort_values("time").drop_duplicates("time").reset_index(drop=True)
        ff["time"] = pd.to_datetime(ff.time, utc=True)
        x = trajectory_matrix(ff, ctx_for(context, symbol), FEATURES)
        meta = ff.set_index("time").reindex(x.index)
        elig = meta.eligible.fillna(False).astype(bool) & (x.index >= TRAIN_END) & (x.index < CONF_END)
        elig &= pd.to_numeric(meta.quote_volume_24h, errors="coerce") >= MIN_QV
        if "account_excluded" in meta: elig &= ~meta.account_excluded.fillna(False).astype(bool)
        x = x.loc[elig]
        if x.empty: continue
        row = pd.DataFrame({"symbol": symbol, "time": x.index, "w3": pd.to_numeric(meta.loc[x.index, "w3_2_24h"], errors="coerce").fillna(0).astype(np.int8).to_numpy()})
        arr = x.to_numpy(np.float32)
        for target, model in models.items(): row[f"score_{target}"] = model.predict_proba(arr)[:,1].astype(np.float32)
        out.append(row); print("SCORE", n, symbol, len(row), flush=True)
    z = pd.concat(out, ignore_index=True).sort_values(["symbol", "time"]).reset_index(drop=True)
    z["split"] = np.select([z.time < DEV_END, z.time < VAL_END], ["DEV", "VALIDATION"], default="CONFIRMATION")
    return z


def onsets(df, col, th):
    z = df[["symbol","time","w3",col]].sort_values(["symbol","time"]).copy(); z["on"] = z[col] >= th
    prev = z.groupby("symbol").on.shift(1, fill_value=False); pt = z.groupby("symbol").time.shift(1)
    gap = (z.time-pt).dt.total_seconds().div(3600)
    return z[z.on & (~prev | gap.ne(1))]


def metrics(g, a, b):
    days = (b-a).total_seconds()/86400; n = len(g); wins = int(g.w3.sum()) if n else 0
    return {"signals": n, "wins": wins, "precision": wins/n if n else np.nan, "signals_per_day": n/days, "wins_per_day": wins/days}


def choose(dev, col):
    s = dev[col].dropna().to_numpy(float)
    qs = np.unique(np.r_[np.linspace(.97,.995,60), np.linspace(.995,.999,80), np.linspace(.999,.99995,100)])
    rows = []
    for th in np.unique(np.quantile(s, qs)):
        rows.append({"threshold": float(th), **metrics(onsets(dev,col,float(th)), TRAIN_END, DEV_END)})
    f = pd.DataFrame(rows)
    v = f[(f.precision >= .90) & (f.signals_per_day >= .5)]
    if len(v): pick = v.sort_values(["wins_per_day","precision"], ascending=False).iloc[0]; reason = ">=90% precision and >=0.5/day; max winners/day"
    else:
        v = f[f.signals_per_day >= .5]
        if len(v): pick = v.sort_values(["precision","wins_per_day"], ascending=False).iloc[0]; reason = "best precision with >=0.5/day"
        else: pick = f.sort_values(["precision","signals_per_day"], ascending=False).iloc[0]; reason = "maximum precision"
    return float(pick.threshold), reason, f


def evaluate(scores):
    periods = {"DEV": (TRAIN_END, DEV_END), "VALIDATION": (DEV_END, VAL_END), "CONFIRMATION": (VAL_END, CONF_END)}
    results, fronts, selected, choices = [], [], [], {}
    dev = scores[scores.split.eq("DEV")]
    for target in TARGETS:
        col = f"score_{target}"; th, reason, fr = choose(dev, col); fr["target"] = target; fronts.append(fr); choices[target] = {"threshold": th, "reason": reason}
        for split,(a,b) in periods.items():
            g = onsets(scores[scores.split.eq(split)], col, th); results.append({"target":target,"split":split,"threshold":th,**metrics(g,a,b)})
            if split != "DEV": g = g.assign(target_model=target, split=split); selected.append(g)
    return pd.DataFrame(results), pd.concat(fronts,ignore_index=True), pd.concat(selected,ignore_index=True), choices


def importance(models):
    rows=[]
    for target,m in models.items():
        imp=m.booster_.feature_importance(importance_type="gain"); total=imp.sum() or 1
        rows += [{"target":target,"feature":f,"gain":float(v),"gain_share":float(v/total)} for f,v in zip(FEATURES,imp)]
    return pd.DataFrame(rows).sort_values(["target","gain"],ascending=[True,False])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    context=load_context(); events=read_parts("events.parquet"); candidates=read_parts("event_candidates.parquet")
    X,meta=build_train(context,events,candidates); models,pop=fit_models(X,meta); del X
    scores=score_stream(context,models); results,fronts,on,choices=evaluate(scores); imp=importance(models)
    pop.to_csv(OUT/"training_population.csv",index=False); results.to_csv(OUT/"full_stream_results.csv",index=False); fronts.to_csv(OUT/"threshold_frontiers.csv",index=False)
    imp.to_csv(OUT/"feature_importance.csv",index=False); on.to_csv(OUT/"selected_signal_onsets.csv.gz",index=False,compression="gzip"); (OUT/"selection.json").write_text(json.dumps(choices,indent=2))
    baseline=scores.groupby("split").agg(eligible_hours=("w3","size"),raw_w3_rate=("w3","mean"),symbols=("symbol","nunique")).reset_index()
    lines=["# Full-Stream W3 Signal Discovery","","**RESEARCH ONLY. No live strategy or automation changes.**","","## Training population","",pop.to_markdown(index=False),"","## Full-stream results","",results.to_markdown(index=False,floatfmt=".4f"),"","## Raw W3 baseline","",baseline.to_markdown(index=False,floatfmt=".4f"),"","## Threshold selection","","```json",json.dumps(choices,indent=2),"```","","## Top markers"]
    for target in TARGETS: lines += ["",f"### {target}","",imp[imp.target.eq(target)].head(25).to_markdown(index=False,floatfmt=".5f")]
    (OUT/"ANALYSIS.md").write_text("\n".join(lines)); print(results.to_string(index=False),flush=True)

if __name__ == "__main__": main()
