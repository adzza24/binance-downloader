from __future__ import annotations

import json, math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from binance_data import load_symbol
from round2b_validation import add_live_features, controlled_activity
from round3f_gain_trail_discovery import simulate_gain_trail

BASE_SCHEDULE=[(m,0.60) for m in [0.30,0.50,1.00,2.00,3.00,5.00,7.50]]
START_CAPITAL=1750.0


def daily_close(df,symbol):
    x=df[['time','close']].copy(); x['time']=pd.to_datetime(x.time,utc=True)
    x=x.set_index('time').sort_index()['close'].astype(float).resample('1D').last().dropna()
    x.name=symbol
    return x


def symbol_work(symbol,btc,cfg):
    df=btc.copy() if symbol=='BTCUSDT' else load_symbol(symbol,cfg['interval'],cfg['start'],cfg['end'])
    if len(df)<800:return [],[],None
    x=add_live_features(df,btc); sigs=controlled_activity(symbol,x,cfg)
    trades=[simulate_gain_trail(df,s,'BASE_FIXED60',BASE_SCHEDULE,cfg,tp=10.0) for s in sigs]
    return sigs,trades,daily_close(df,symbol)


def build_regime_features(closes):
    p=pd.concat(closes,axis=1).sort_index()
    btc=p['BTCUSDT']
    r1=p.pct_change()
    ret20=p/p.shift(20)-1; ret7=p/p.shift(7)-1
    ma50=p.rolling(50,min_periods=35).mean()
    breadth_ma50=(p>ma50).mean(axis=1)
    pct_pos20=(ret20>0).mean(axis=1)
    pct_pos7=(ret7>0).mean(axis=1)
    btc20=btc/btc.shift(20)-1
    pct_out_btc20=ret20.sub(btc20,axis=0).gt(0).mean(axis=1)
    med_ret20=ret20.median(axis=1)
    dispersion20=ret20.std(axis=1)
    corr=[]
    for c in p.columns:
        if c=='BTCUSDT':continue
        corr.append(r1[c].rolling(20,min_periods=15).corr(r1['BTCUSDT']).rename(c))
    mean_corr20=pd.concat(corr,axis=1).mean(axis=1)
    btc_ma200=btc.rolling(200,min_periods=180).mean()
    f=pd.DataFrame(index=p.index)
    f['breadth_ma50']=breadth_ma50
    f['breadth_mom10']=breadth_ma50-breadth_ma50.shift(10)
    f['pct_pos20']=pct_pos20; f['pct_pos7']=pct_pos7; f['pct_out_btc20']=pct_out_btc20
    f['median_ret20']=med_ret20; f['dispersion20']=dispersion20; f['mean_corr20']=mean_corr20
    f['btc_ret20']=btc/btc.shift(20)-1; f['btc_ret30']=btc/btc.shift(30)-1; f['btc_ret90']=btc/btc.shift(90)-1
    f['btc_dist200']=btc/btc_ma200-1
    f['btc_ma200_slope20']=btc_ma200/btc_ma200.shift(20)-1
    vol20=r1['BTCUSDT'].rolling(20,min_periods=15).std()*np.sqrt(365)
    vol60=r1['BTCUSDT'].rolling(60,min_periods=45).std()*np.sqrt(365)
    f['btc_vol20']=vol20; f['btc_vol_ratio']=vol20/vol60
    # Shift one full daily bar so an intraday signal only uses completed prior-day information.
    return f.shift(1)


def risk_score(r):
    tests=[
        r.breadth_ma50<0.40,
        r.breadth_mom10<-0.10,
        (r.btc_dist200<0) and (r.btc_ma200_slope20<0),
        r.btc_ret30<-0.10,
        (r.btc_vol_ratio>1.35) and (r.btc_ret20<0),
        r.pct_pos20<0.35,
        (r.mean_corr20>0.65) and (r.median_ret20<0),
    ]
    return int(sum(bool(x) for x in tests if pd.notna(x)))


def exposure(name,r):
    score=int(r['risk_score'])
    if name=='BASE_FULL': return 1.0
    if name=='PAUSE_BREADTH30': return 0.0 if r.breadth_ma50<0.30 else 1.0
    if name=='PAUSE_BEAR_STRUCTURE': return 0.0 if (r.btc_dist200<0 and r.btc_ma200_slope20<0) else 1.0
    if name=='PAUSE_STRESS3': return 0.0 if score>=3 else 1.0
    if name=='HALF_STRESS2_PAUSE3': return 0.0 if score>=3 else (0.5 if score==2 else 1.0)
    if name=='DEFENSIVE_STRESS2': return 0.0 if score>=3 else (0.25 if score==2 else 1.0)
    raise KeyError(name)

VARIANTS=['BASE_FULL','PAUSE_BREADTH30','PAUSE_BEAR_STRUCTURE','PAUSE_STRESS3','HALF_STRESS2_PAUSE3','DEFENSIVE_STRESS2']


def attach_regime(trades,features):
    q=trades.copy(); q['entry_time']=pd.to_datetime(q.entry_time,utc=True); q['exit_time']=pd.to_datetime(q.exit_time,utc=True)
    dates=q.entry_time.dt.floor('D')
    rows=[]
    for i,d in enumerate(dates):
        if d not in features.index: rows.append(None); continue
        r=features.loc[d].copy(); r['risk_score']=risk_score(r); rows.append(r)
    rf=pd.DataFrame(rows,index=q.index)
    for c in rf.columns:q[c]=rf[c]
    return q.dropna(subset=['breadth_ma50','btc_ret30'])


def max_losing_streak(vals):
    best=cur=0
    for x in vals:
        if x<0:cur+=1;best=max(best,cur)
        else:cur=0
    return best


def metrics(q,variant):
    g=q.copy(); g['exposure']=g.apply(lambda r:exposure(variant,r),axis=1); g['scaled_pnl']=g.pnl_usdt.astype(float)*g.exposure
    closed=g[~g.is_open].copy(); vals=closed.scaled_pnl.to_numpy(float)
    pos=vals[vals>0];neg=vals[vals<0]
    ordered=closed.sort_values('exit_time'); curve=START_CAPITAL+ordered.scaled_pnl.cumsum(); peak=curve.cummax(); dd=curve-peak; dd_pct=dd/peak.replace(0,np.nan)
    roll20=ordered.scaled_pnl.rolling(20,min_periods=5).sum()
    return {
        'variant':variant,'signals':len(g),'full_exposure':int((g.exposure==1).sum()),'reduced_exposure':int(((g.exposure>0)&(g.exposure<1)).sum()),'skipped':int((g.exposure==0).sum()),
        'realised_pnl':float(closed.scaled_pnl.sum()),'open_mtm':float(g[g.is_open].scaled_pnl.sum()),'combined_pnl':float(g.scaled_pnl.sum()),
        'profit_factor':float(pos.sum()/abs(neg.sum())) if len(neg) and neg.sum()!=0 else math.inf,
        'win_rate_active':float((vals[vals!=0]>0).mean()) if (vals!=0).any() else np.nan,
        'avg_loss':float(neg.mean()) if len(neg) else 0.0,'worst_trade_loss':float(neg.min()) if len(neg) else 0.0,
        'max_losing_streak':max_losing_streak(vals[vals!=0]),'max_pseudo_drawdown_usdt':float(dd.min()) if len(dd) else 0.0,
        'max_pseudo_drawdown_pct':float(dd_pct.min()) if len(dd_pct) else 0.0,'min_pseudo_equity':float(curve.min()) if len(curve) else START_CAPITAL,
        'worst_rolling20_pnl':float(roll20.min()) if roll20.notna().any() else 0.0,
        'breach_50pct_capital':bool((curve<START_CAPITAL*0.5).any()) if len(curve) else False,
        'breach_25pct_capital':bool((curve<START_CAPITAL*0.25).any()) if len(curve) else False,
        'pseudo_ruin':bool((curve<=0).any()) if len(curve) else False,
    },g


def summarise_period(q,label,year_min=None,year_max=None):
    z=q.copy(); y=z.entry_time.dt.year
    if year_min is not None:z=z[y>=year_min]
    if year_max is not None:z=z[z.entry_time.dt.year<=year_max]
    rows=[]; variants={}
    for v in VARIANTS:
        m,g=metrics(z,v);m['period']=label;rows.append(m);variants[v]=g
    return pd.DataFrame(rows),variants


def yearly(q):
    rows=[]
    for yr,g in q.groupby(q.entry_time.dt.year):
        for v in VARIANTS:
            m,x=metrics(g,v); rows.append({'year':int(yr),'variant':v,'signals':len(x),'skipped':int((x.exposure==0).sum()),'reduced':int(((x.exposure>0)&(x.exposure<1)).sum()),'combined_pnl':x.scaled_pnl.sum(),'realised_pnl':x[~x.is_open].scaled_pnl.sum(),'max_pseudo_drawdown_usdt':m['max_pseudo_drawdown_usdt'],'worst_trade_loss':m['worst_trade_loss']})
    return pd.DataFrame(rows)


def score_diagnostics(q):
    x=q.copy();x['risk_score']=x.risk_score.astype(int)
    return x.groupby('risk_score').agg(signals=('signal_id','size'),mean_pnl=('pnl_usdt','mean'),sum_pnl=('pnl_usdt','sum'),win_rate=('pnl_usdt',lambda s:(s>0).mean()),breadth=('breadth_ma50','mean'),btc30=('btc_ret30','mean')).reset_index()


def main():
    cfg=json.loads(Path('research/config.json').read_text());out=Path('research/results/round4');out.mkdir(parents=True,exist_ok=True)
    btc=load_symbol('BTCUSDT',cfg['interval'],cfg['start'],cfg['end'])
    all_s=[];all_t=[];closes=[]
    with ThreadPoolExecutor(max_workers=6) as ex:
        fut={ex.submit(symbol_work,s,btc,cfg):s for s in cfg['symbols']}
        for f in as_completed(fut):
            s=fut[f]
            try:
                sig,tr,dc=f.result();all_s+=sig;all_t+=tr
                if dc is not None:closes.append(dc)
                print(s,len(sig),flush=True)
            except Exception as e:print('ERROR',s,repr(e),flush=True)
    signals=pd.DataFrame(all_s); trades=pd.DataFrame(all_t)
    features=build_regime_features(closes); q=attach_regime(trades,features)
    full,_=summarise_period(q,'FULL')
    disc,_=summarise_period(q,'DISCOVERY_2019_2024',2019,2024)
    eval_,_=summarise_period(q,'FORWARD_2025_2026',2025,2026)
    summaries=pd.concat([full,disc,eval_],ignore_index=True)
    yr=yearly(q); diag=score_diagnostics(q)
    signals.to_csv(out/'signals.csv',index=False); q.to_csv(out/'trades_with_regime.csv',index=False); features.to_csv(out/'daily_regime_features.csv')
    summaries.to_csv(out/'summary.csv',index=False);yr.to_csv(out/'year_summary.csv',index=False);diag.to_csv(out/'risk_score_diagnostics.csv',index=False)
    manifest={
      'study':'Round 4 causal regime gating - entry exposure only',
      'base_strategy':'Controlled Activity entries + frozen Round 3H FIXED_60 accumulated-profit giveback exit',
      'causality':'All regime features are daily and shifted one full day; each signal uses only completed data available before its entry day.',
      'variants':VARIANTS,
      'risk_score_components':['breadth_ma50<40%','10d breadth momentum<-10pp','BTC below 200d MA and 200d MA falling','BTC 30d return<-10%','BTC vol20/vol60>1.35 with negative 20d return','<35% universe positive over 20d','mean 20d alt/BTC correlation>0.65 with negative median 20d cross-sectional return'],
      'periods':'2019-2024 descriptive discovery; 2025-2026 chronological forward comparison for the regime gates. Underlying entry/exit strategy itself was previously optimised on full history, so this is not a pristine independent holdout.',
      'loss_metrics':'Includes worst trade loss, average loss, losing streak, worst rolling-20 trades, and a 1750-USDT cumulative-realised pseudo-equity drawdown/survival proxy. Because independent 300-USDT trades can overlap, pseudo-equity is not a deployable capital-constrained simulation.',
      'scope':'This first Round 4 pass gates/reduces NEW ENTRIES only; it does not force-regime-exit already-open positions. If entry gating is useful, forced de-risking can be tested separately without changing the frozen trade exit architecture.'
    }
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('\nSUMMARY\n',summaries.to_string(index=False));print('\nRISK SCORE\n',diag.to_string(index=False));print('\nYEARS\n',yr.to_string(index=False))

if __name__=='__main__':main()
