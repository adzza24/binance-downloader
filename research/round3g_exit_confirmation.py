from __future__ import annotations

import json, math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from binance_data import load_symbol
from round2b_validation import add_live_features, controlled_activity
from round3a_exit_architecture import net_pnl, capped_stop

# Confirm only three pre-specified contenders. Trail percentage means fraction of
# accumulated GAIN allowed to be given back, not percentage below market price.
GAIN_VARIANTS = {
    'DERIVED_70_55_50_TP1000': [(0.30,0.70),(2.00,0.55),(7.50,0.50)],
    'CONST_60_TP1000': [(0.30,0.60)],
}
MILESTONES_A = [(1.00,0.50),(2.00,1.00),(5.00,3.00)]
TP = 10.00


def result(sig,name,t,reason,pnl,is_open,bars,ret):
    return {'signal_id':sig['signal_id'],'symbol':sig['symbol'],'entry_time':sig['entry_time'],
            'variant':name,'exit_time':t,'exit_reason':reason,'pnl_usdt':pnl,
            'is_open':is_open,'bars_held':bars,'exit_return_pct':ret}


def simulate_gain(df,sig,name,schedule,cfg):
    start=int(sig['entry_index']); entry=float(sig['entry_price'])
    init_stop=capped_stop(entry,float(sig['structural_stop']),0.10)
    stop=init_stop; activated=False; peak=entry; stage_gb=None
    for j in range(start,len(df)):
        b=df.iloc[j]; low,high,close=map(float,(b.low,b.high,b.close)); peak=max(peak,high)
        if not activated:
            if low<=init_stop:
                return result(sig,name,b.time,'STOP',net_pnl(entry,[(1,init_stop)],cfg),False,j-start+1,init_stop/entry-1)
            if high>=entry*1.05:
                activated=True; stop=entry; continue
        else:
            if low<=stop:
                return result(sig,name,b.time,'GAIN_TRAIL_STOP',net_pnl(entry,[(1,stop)],cfg),False,j-start+1,stop/entry-1)
            if high>=entry*(1+TP):
                px=entry*(1+TP)
                return result(sig,name,b.time,'TAKE_PROFIT',net_pnl(entry,[(1,px)],cfg),False,j-start+1,TP)
            peak_gain=peak/entry-1
            for threshold,gb in schedule:
                if peak_gain>=threshold: stage_gb=gb
            if peak_gain>=0.30: stop=max(stop,entry*1.10)
            if stage_gb is not None:
                floor_gain=peak_gain*(1-stage_gb)
                candidate=min(entry*(1+floor_gain),close*0.999)
                stop=max(stop,candidate)
    close=float(df.iloc[-1].close)
    return result(sig,name,df.iloc[-1].time,'DATA_END_OPEN',net_pnl(entry,[(1,close)],cfg),True,len(df)-start,close/entry-1)


def simulate_milestones(df,sig,cfg):
    name='MILESTONES_A_TP1000'
    start=int(sig['entry_index']); entry=float(sig['entry_price'])
    init_stop=capped_stop(entry,float(sig['structural_stop']),0.10)
    stop=init_stop; activated=False; floor30=False
    for j in range(start,len(df)):
        b=df.iloc[j]; low,high,close=map(float,(b.low,b.high,b.close))
        if not activated:
            if low<=init_stop:
                return result(sig,name,b.time,'STOP',net_pnl(entry,[(1,init_stop)],cfg),False,j-start+1,init_stop/entry-1)
            if high>=entry*1.05:
                activated=True; stop=entry; continue
        else:
            if low<=stop:
                return result(sig,name,b.time,'PROFIT_FLOOR_STOP',net_pnl(entry,[(1,stop)],cfg),False,j-start+1,stop/entry-1)
            if high>=entry*(1+TP):
                px=entry*(1+TP)
                return result(sig,name,b.time,'TAKE_PROFIT',net_pnl(entry,[(1,px)],cfg),False,j-start+1,TP)
            if (not floor30) and high>=entry*1.30:
                stop=max(stop,entry*1.10); floor30=True
            for trigger,lock in MILESTONES_A:
                if high>=entry*(1+trigger): stop=max(stop,entry*(1+lock))
    close=float(df.iloc[-1].close)
    return result(sig,name,df.iloc[-1].time,'DATA_END_OPEN',net_pnl(entry,[(1,close)],cfg),True,len(df)-start,close/entry-1)


def process_symbol(symbol,btc,cfg):
    df=btc.copy() if symbol=='BTCUSDT' else load_symbol(symbol,cfg['interval'],cfg['start'],cfg['end'])
    if len(df)<800:return [],[]
    x=add_live_features(df,btc); sigs=controlled_activity(symbol,x,cfg); out=[]
    for sig in sigs:
        for name,sch in GAIN_VARIANTS.items(): out.append(simulate_gain(df,sig,name,sch,cfg))
        out.append(simulate_milestones(df,sig,cfg))
    return sigs,out


def summarise(q):
    q=q.copy(); q['year']=pd.to_datetime(q.entry_time).dt.year
    rows=[]; years=[]
    for v,g in q.groupby('variant'):
        c=g[~g.is_open]; o=g[g.is_open]; p=c.pnl_usdt.to_numpy(float); w=p[p>0]; l=p[p<0]
        rows.append({'variant':v,'trades':len(g),'closed':len(c),'open':len(o),
                     'realised_pnl':c.pnl_usdt.sum(),'open_mtm':o.pnl_usdt.sum(),
                     'combined_mtm':g.pnl_usdt.sum(),
                     'profit_factor':w.sum()/abs(l.sum()) if len(l) and l.sum() else math.inf,
                     'avg_hold_days':g.bars_held.mean()/24})
        for y,gy in g.groupby('year'):
            gc=gy[~gy.is_open]; go=gy[gy.is_open]
            years.append({'variant':v,'year':int(y),'trades':len(gy),
                          'realised_pnl':gc.pnl_usdt.sum(),'open_mtm':go.pnl_usdt.sum(),
                          'combined_mtm':gy.pnl_usdt.sum()})
    return pd.DataFrame(rows),pd.DataFrame(years)


def period_summary(trades,start_year):
    q=trades[pd.to_datetime(trades.entry_time).dt.year>=start_year].copy()
    sm,_=summarise(q); sm.insert(1,'period',f'{start_year}+')
    return sm


def main():
    cfg=json.loads(Path('research/config.json').read_text()); out=Path('research/results/round3g'); out.mkdir(parents=True,exist_ok=True)
    btc=load_symbol('BTCUSDT',cfg['interval'],cfg['start'],cfg['end'])
    all_s=[]; all_t=[]
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut={ex.submit(process_symbol,s,btc,cfg):s for s in cfg['symbols']}
        for f in as_completed(fut):
            s=fut[f]
            try:
                sig,tr=f.result(); all_s+=sig; all_t+=tr; print(s,len(sig),len(tr),flush=True)
            except Exception as e: print('ERROR',s,repr(e),flush=True)
    signals=pd.DataFrame(all_s); trades=pd.DataFrame(all_t)
    sm,yr=summarise(trades); p2022=period_summary(trades,2022); p2021=period_summary(trades,2021)
    signals.to_csv(out/'signals.csv',index=False); trades.to_csv(out/'trades.csv',index=False)
    sm.to_csv(out/'summary.csv',index=False); yr.to_csv(out/'year_summary.csv',index=False)
    p2022.to_csv(out/'summary_2022_onward.csv',index=False); p2021.to_csv(out/'summary_2021_onward.csv',index=False)
    manifest={'study':'Round 3G pre-specified exit confirmation','variants':{
        'DERIVED_70_55_50_TP1000':'+30-<200 allow 70% gain giveback; +200-<750 allow 55%; +750-<1000 allow 50%; +1000 full TP',
        'CONST_60_TP1000':'from +30 onward allow 60% accumulated-gain giveback, with +30->+10 minimum floor; +1000 full TP',
        'MILESTONES_A_TP1000':'+30->+10; +100->+50; +200->+100; +500->+300; +1000 full TP'},
        'base':'structural stop capped 10%; +5->breakeven; no max holding period; data-end positions remain open MTM',
        'note':'These variants were fixed before this confirmation run, but all were informed by prior analysis of the same historical dataset; this is confirmatory in-sample comparison, not independent out-of-sample validation.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('\nFULL HISTORY\n',sm.sort_values('combined_mtm',ascending=False).to_string(index=False))
    print('\n2022 ONWARD\n',p2022.sort_values('combined_mtm',ascending=False).to_string(index=False))
    print('\n2021 ONWARD\n',p2021.sort_values('combined_mtm',ascending=False).to_string(index=False))

if __name__=='__main__': main()
