from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from high_purity_signal_watch import CONTEXT_FEATURES, add_cross_section_context
from replay_high_purity_2026 import trajectory_matrix

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "research/cache/full_stream_w3_direct_source"
PREV = ROOT / "research/results/daily_return_harvest/full_stream_w3_discovery"
OUT = ROOT / "research/results/daily_return_harvest/full_stream_w3_direct"
START = pd.Timestamp("2021-11-01", tz="UTC")
TRAIN_END = pd.Timestamp("2023-11-01", tz="UTC")
DEV_END = pd.Timestamp("2024-05-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
CONF_END = pd.Timestamp("2026-09-28", tz="UTC")
PRE_CONTEXT = START - pd.Timedelta(hours=169)
MIN_QV = 1_000_000.0
TOP_FEATURES = 140
TARGETS = ("W3_ALL", "EARLY3_ALLNEG")


def shard_dirs():
    ds=[SRC/f"drh2-shard-{i}" for i in range(4)]
    missing=[str(d) for d in ds if not (d/"panel.parquet").exists()]
    if missing: raise RuntimeError(f"Missing shards {missing}")
    return ds


def read_parts(name):
    a=[]
    for d in shard_dirs():
        p=d/name
        if p.exists():
            z=pd.read_parquet(p)
            if len(z): a.append(z)
    return pd.concat(a,ignore_index=True) if a else pd.DataFrame()


def symbol_files():
    out=[]
    for d in shard_dirs(): out += [(p.name.removesuffix('.pkl.gz'),p) for p in (d/'features').glob('*.pkl.gz')]
    return sorted(out)


def load_context():
    a=[]
    for d in shard_dirs():
        z=pd.read_parquet(d/'panel.parquet'); z['time']=pd.to_datetime(z.time,utc=True)
        a.append(z[(z.time>=PRE_CONTEXT)&(z.time<CONF_END)])
    p=pd.concat(a,ignore_index=True).drop_duplicates(['symbol','time']).sort_values(['time','symbol'])
    return add_cross_section_context(p)


def ctx_for(ctx,symbol):
    z=ctx.loc[ctx.symbol.eq(symbol),['time']+list(CONTEXT_FEATURES)].copy()
    return z.drop_duplicates('time').set_index('time').sort_index()


def select_features():
    imp=pd.read_csv(PREV/'feature_importance.csv')
    g=imp.groupby('feature',as_index=False).gain_share.sum().sort_values('gain_share',ascending=False)
    feats=g.head(TOP_FEATURES).feature.tolist()
    pd.DataFrame({'feature':feats,'combined_gain_share':[float(g.set_index('feature').loc[f,'gain_share']) for f in feats]}).to_csv(OUT/'selected_features.csv',index=False)
    return feats


def early_map():
    c=read_parts('event_candidates.parquet'); c['decision_time']=pd.to_datetime(c.decision_time,utc=True)
    c=c[(c.decision_time>=START)&(c.decision_time<TRAIN_END)&(c.hours_from_event_start<=3)]
    return {s:set(g.decision_time) for s,g in c.groupby('symbol')}


def eligible(ff,idx,start,end):
    m=ff.set_index('time').reindex(idx)
    e=m.eligible.fillna(False).astype(bool)&(idx>=start)&(idx<end)
    e &= pd.to_numeric(m.quote_volume_24h,errors='coerce')>=MIN_QV
    if 'account_excluded' in m: e &= ~m.account_excluded.fillna(False).astype(bool)
    return e,m


def build_train(ctx,features,early):
    xs=[]; yw=[]; ye=[]; stats=[]
    for n,(symbol,path) in enumerate(symbol_files(),1):
        ff=pd.read_pickle(path,compression='gzip').sort_values('time').drop_duplicates('time').reset_index(drop=True); ff['time']=pd.to_datetime(ff.time,utc=True)
        x=trajectory_matrix(ff,ctx_for(ctx,symbol),features); e,m=eligible(ff,x.index,START,TRAIN_END); x=x.loc[e]
        if x.empty: continue
        w=pd.to_numeric(m.loc[x.index,'w3_2_24h'],errors='coerce').fillna(0).astype(np.int8).to_numpy()
        et=np.array([int(t in early.get(symbol,set())) for t in x.index],dtype=np.int8)
        xs.append(x.to_numpy(np.float32)); yw.append(w); ye.append(et)
        stats.append({'symbol':symbol,'rows':len(x),'w3':int(w.sum()),'early3':int(et.sum())})
        print('TRAIN',n,symbol,len(x),int(w.sum()),int(et.sum()),flush=True)
    return np.concatenate(xs),np.concatenate(yw),np.concatenate(ye),pd.DataFrame(stats)


def fit(X,y,seed):
    m=LGBMClassifier(objective='binary',n_estimators=650,learning_rate=.03,num_leaves=31,min_child_samples=150,
                     max_bin=63,subsample=.85,colsample_bytree=.75,reg_alpha=1.0,reg_lambda=8.0,
                     random_state=seed,n_jobs=-1,verbosity=-1)
    m.fit(X,y)
    return m


def score_all(ctx,features,models):
    a=[]
    for n,(symbol,path) in enumerate(symbol_files(),1):
        ff=pd.read_pickle(path,compression='gzip').sort_values('time').drop_duplicates('time').reset_index(drop=True); ff['time']=pd.to_datetime(ff.time,utc=True)
        x=trajectory_matrix(ff,ctx_for(ctx,symbol),features); e,m=eligible(ff,x.index,TRAIN_END,CONF_END); x=x.loc[e]
        if x.empty: continue
        arr=x.to_numpy(np.float32); r=pd.DataFrame({'symbol':symbol,'time':x.index,'w3':pd.to_numeric(m.loc[x.index,'w3_2_24h'],errors='coerce').fillna(0).astype(np.int8).to_numpy()})
        for k,mod in models.items(): r['score_'+k]=mod.predict_proba(arr)[:,1].astype(np.float32)
        a.append(r); print('SCORE',n,symbol,len(r),flush=True)
    z=pd.concat(a,ignore_index=True).sort_values(['symbol','time']).reset_index(drop=True)
    z['split']=np.select([z.time<DEV_END,z.time<VAL_END],['DEV','VALIDATION'],default='CONFIRMATION')
    return z


def onsets(df,col,th):
    z=df[['symbol','time','w3',col]].sort_values(['symbol','time']).copy(); z['on']=z[col]>=th
    prev=z.groupby('symbol').on.shift(1,fill_value=False); pt=z.groupby('symbol').time.shift(1); gap=(z.time-pt).dt.total_seconds().div(3600)
    return z[z.on & (~prev|gap.ne(1))]


def met(g,a,b):
    d=(b-a).total_seconds()/86400; n=len(g); w=int(g.w3.sum()) if n else 0
    return {'signals':n,'wins':w,'precision':w/n if n else np.nan,'signals_per_day':n/d,'wins_per_day':w/d}


def frontier(dev,col):
    s=dev[col].dropna().to_numpy(float); qs=np.unique(np.r_[np.linspace(.90,.99,100),np.linspace(.99,.999,120),np.linspace(.999,.99999,120)])
    return pd.DataFrame([{'threshold':float(t),**met(onsets(dev,col,float(t)),TRAIN_END,DEV_END)} for t in np.unique(np.quantile(s,qs))])


def choose(f):
    v=f[(f.precision>=.90)&(f.signals_per_day>=.5)]
    if len(v): return v.sort_values(['wins_per_day','precision'],ascending=False).iloc[0],'>=90% precision >=0.5/day'
    v=f[f.signals_per_day>=.5]
    if len(v): return v.sort_values(['precision','wins_per_day'],ascending=False).iloc[0],'best precision >=0.5/day'
    return f.sort_values(['precision','signals_per_day'],ascending=False).iloc[0],'maximum precision'


def main():
    OUT.mkdir(parents=True,exist_ok=True); features=select_features(); ctx=load_context(); early=early_map(); X,yw,ye,trainstats=build_train(ctx,features,early)
    # Direct W3 sees every training hour. Early-event model sees every true non-W3 hour plus early-event positives; later W3 opportunities are ignored rather than mislabeled.
    models={'W3_ALL':fit(X,yw,101)}
    mask=(yw==0)|(ye==1); models['EARLY3_ALLNEG']=fit(X[mask],ye[mask],202)
    pop=pd.DataFrame([{'target':'W3_ALL','rows':len(yw),'positives':int(yw.sum()),'negatives':int((yw==0).sum())},{'target':'EARLY3_ALLNEG','rows':int(mask.sum()),'positives':int(ye[mask].sum()),'negatives':int((yw[mask]==0).sum())}]); pop['negative_to_positive']=pop.negatives/pop.positives
    del X
    scores=score_all(ctx,features,models); dev=scores[scores.split.eq('DEV')]; periods={'DEV':(TRAIN_END,DEV_END),'VALIDATION':(DEV_END,VAL_END),'CONFIRMATION':(VAL_END,CONF_END)}
    rows=[]; fronts=[]; selected=[]; choices={}
    for target in TARGETS:
        col='score_'+target; f=frontier(dev,col); f['target']=target; fronts.append(f); pick,reason=choose(f); th=float(pick.threshold); choices[target]={'threshold':th,'reason':reason}
        for sp,(a,b) in periods.items():
            g=onsets(scores[scores.split.eq(sp)],col,th); rows.append({'target':target,'split':sp,'threshold':th,**met(g,a,b)})
            if sp!='DEV': selected.append(g.assign(target_model=target,split=sp))
    results=pd.DataFrame(rows); baseline=scores.groupby('split').agg(eligible_hours=('w3','size'),raw_w3_rate=('w3','mean'),symbols=('symbol','nunique')).reset_index()
    im=[]
    for target,m in models.items():
        gain=m.booster_.feature_importance(importance_type='gain'); total=gain.sum() or 1
        im += [{'target':target,'feature':f,'gain':float(v),'gain_share':float(v/total)} for f,v in zip(features,gain)]
    imp=pd.DataFrame(im).sort_values(['target','gain'],ascending=[True,False])
    pop.to_csv(OUT/'training_population.csv',index=False); trainstats.to_csv(OUT/'training_by_symbol.csv',index=False); results.to_csv(OUT/'full_stream_results.csv',index=False); pd.concat(fronts).to_csv(OUT/'threshold_frontiers.csv',index=False); imp.to_csv(OUT/'feature_importance.csv',index=False); pd.concat(selected).to_csv(OUT/'selected_signal_onsets.csv.gz',index=False,compression='gzip'); (OUT/'selection.json').write_text(json.dumps(choices,indent=2))
    lines=['# Full-Stream W3 Direct Classifier','', '**RESEARCH ONLY. No live strategy or automation changes.**','', '## Training population','',pop.to_markdown(index=False,floatfmt='.4f'),'','## Results','',results.to_markdown(index=False,floatfmt='.4f'),'','## Baseline','',baseline.to_markdown(index=False,floatfmt='.4f'),'','## Choices','', '```json',json.dumps(choices,indent=2),'```','', '## Top markers']
    for t in TARGETS: lines += ['',f'### {t}','',imp[imp.target.eq(t)].head(30).to_markdown(index=False,floatfmt='.5f')]
    (OUT/'ANALYSIS.md').write_text('\n'.join(lines)); print(results.to_string(index=False),flush=True)

if __name__=='__main__': main()
