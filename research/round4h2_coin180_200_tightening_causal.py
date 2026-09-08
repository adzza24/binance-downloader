from __future__ import annotations

import json, math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from binance_data import load_symbol
from round2b_validation import add_live_features, controlled_activity
from round3a_exit_architecture import capped_stop, net_pnl
from round4g_coin_regime_actions import daily_state, at_state, capital_sim

BASE_GIVEBACK=0.60
MAS=[180,200]
TIGHTEN=[0.50,0.40,0.30,0.20,0.10]
VARIANTS=[]
for w in MAS:
    VARIANTS += [f'COIN{w}_GATE']+[f'COIN{w}_TIGHTEN{int(g*100)}' for g in TIGHTEN]+[f'COIN{w}_STAGE_AWARE']

def parse(v):
    w=180 if v.startswith('COIN180_') else 200
    if v.endswith('_GATE'): return w,'GATE',None
    if v.endswith('_STAGE_AWARE'): return w,'STAGE_AWARE',None
    return w,'TIGHTEN',int(v.split('TIGHTEN')[-1])/100

def candidate(entry,peak,cap_price,giveback):
    pg=peak/entry-1
    if pg<0.30:return None
    return min(entry*(1+pg*(1-giveback)),cap_price*0.999)

def rec(sig,v,t,reason,pnl,bars,event=False,event_time=None,event_stage=None,event_pg=None,is_open=False):
    return {'signal_id':sig['signal_id'],'symbol':sig['symbol'],'entry_time':sig['entry_time'],'variant':v,'exit_time':t,
            'exit_reason':reason,'pnl_usdt':float(pnl),'bars_held':int(bars),'event':bool(event),'event_time':event_time,
            'event_stage':event_stage,'event_peak_gain':event_pg,'is_open':bool(is_open)}

def simulate(df,sig,v,states,cfg):
    w,action,gb=parse(v); state=states[w]; start=int(sig['entry_index']); entry=float(sig['entry_price'])
    init_stop=capped_stop(entry,float(sig['structural_stop']),0.10)
    if at_state(sig['entry_time'],state): return rec(sig,v,sig['entry_time'],'ENTRY_GATE',0,0)
    stop=init_stop; activated=False; peak=entry; last=at_state(df.iloc[start].time,state)
    event_seen=False; evt_time=None; evt_stage=None; evt_pg=None
    for j in range(start,len(df)):
        b=df.iloc[j]; low,high,close,op=map(float,(b.low,b.high,b.close,b.open))
        cur=at_state(b.time,state); state_start=cur and not last; last=cur
        # Event decision at bar open uses only the peak and stage known before this bar.
        pg_pre=peak/entry-1
        stage='INITIAL_PROTECTION' if not activated else ('TRAIL_30PLUS' if pg_pre>=0.30 else 'BREAKEVEN_PROTECTION')
        if action!='GATE' and state_start and j>start and not event_seen:
            event_seen=True; evt_time=b.time; evt_stage=stage; evt_pg=pg_pre
            if action=='TIGHTEN' and activated:
                c=candidate(entry,peak,op,gb)
                if c is not None: stop=max(stop,c)
            elif action=='STAGE_AWARE':
                if stage=='INITIAL_PROTECTION':
                    return rec(sig,v,b.time,f'COIN{w}_STAGE_INITIAL_EXIT',net_pnl(entry,[(1,op)],cfg),j-start+1,True,evt_time,evt_stage,evt_pg)
                if stage=='BREAKEVEN_PROTECTION': stop=max(stop,op*0.95)
                else:
                    c=candidate(entry,peak,op,0.50)
                    if c is not None: stop=max(stop,c)
        # The event stop is active from this bar open onward; now process this bar.
        peak=max(peak,high); pg=peak/entry-1
        if not activated:
            if low<=init_stop: return rec(sig,v,b.time,'STOP',net_pnl(entry,[(1,init_stop)],cfg),j-start+1,event_seen,evt_time,evt_stage,evt_pg)
            if high>=entry*1.05:
                activated=True; stop=max(stop,entry); continue
        else:
            if low<=stop: return rec(sig,v,b.time,'GAIN_TRAIL_STOP',net_pnl(entry,[(1,stop)],cfg),j-start+1,event_seen,evt_time,evt_stage,evt_pg)
            if high>=entry*11:
                px=entry*11; return rec(sig,v,b.time,'TAKE_PROFIT',net_pnl(entry,[(1,px)],cfg),j-start+1,event_seen,evt_time,evt_stage,evt_pg)
            if pg>=0.30: stop=max(stop,entry*1.10)
            c=candidate(entry,peak,close,BASE_GIVEBACK)
            if c is not None: stop=max(stop,c)
    close=float(df.iloc[-1].close)
    return rec(sig,v,df.iloc[-1].time,'DATA_END_OPEN',net_pnl(entry,[(1,close)],cfg),len(df)-start,event_seen,evt_time,evt_stage,evt_pg,True)

def process(symbol,btc,cfg):
    df=btc.copy() if symbol=='BTCUSDT' else load_symbol(symbol,cfg['interval'],cfg['start'],cfg['end'])
    if len(df)<800:return [],[]
    states={w:daily_state(df,w) for w in MAS}; sigs=controlled_activity(symbol,add_live_features(df,btc),cfg); rows=[]
    for sig in sigs:
        for v in VARIANTS: rows.append(simulate(df,sig,v,states,cfg))
    return sigs,rows

def pf(vals):
    a=np.asarray(vals,float); p=a[a>0]; n=a[a<0]
    return float(p.sum()/abs(n.sum())) if len(n) and n.sum() else math.inf

def summarise(trades):
    q=trades.copy(); q.entry_time=pd.to_datetime(q.entry_time,utc=True); q.exit_time=pd.to_datetime(q.exit_time,utc=True)
    rows=[]; years=[]
    for v,g in q.groupby('variant'):
        vals=g.pnl_usdt.to_numpy(float); active=g[g.exit_reason!='ENTRY_GATE']; cap,dd,skip=capital_sim(g)
        rows.append({'variant':v,'signals':len(g),'active_trades':len(active),'entry_gate_skips':int((g.exit_reason=='ENTRY_GATE').sum()),
                     'open_position_events':int(g.event.sum()),'combined_pnl':float(vals.sum()),'profit_factor':pf(vals[vals!=0]),
                     'win_rate_active':float((active.pnl_usdt>0).mean()) if len(active) else np.nan,'capital_profit_proxy':float(cap),
                     'capital_max_realised_dd':float(dd),'skipped_due_cash':int(skip)})
        for y,gy in g.groupby(g.entry_time.dt.year): years.append({'variant':v,'year':int(y),'pnl_usdt':float(gy.pnl_usdt.sum())})
    return pd.DataFrame(rows),pd.DataFrame(years)

def main():
    cfg=json.loads(Path('research/config.json').read_text()); out=Path('research/results/round4h2'); out.mkdir(parents=True,exist_ok=True)
    btc=load_symbol('BTCUSDT',cfg['interval'],cfg['start'],cfg['end']); alls=[]; allt=[]
    with ThreadPoolExecutor(max_workers=6) as ex:
        fut={ex.submit(process,s,btc,cfg):s for s in cfg['symbols']}
        for f in as_completed(fut):
            s=fut[f]
            try:
                sig,rows=f.result(); alls+=sig; allt+=rows; print(s,len(sig),flush=True)
            except Exception as e: print('ERROR',s,repr(e),flush=True)
    trades=pd.DataFrame(allt); pd.DataFrame(alls).to_csv(out/'signals.csv',index=False); trades.to_csv(out/'trades.csv',index=False)
    sm,yr=summarise(trades); sm.to_csv(out/'summary.csv',index=False); yr.to_csv(out/'year_summary.csv',index=False)
    (out/'manifest.json').write_text(json.dumps({'study':'Round 4H2 causal coin180/200 tightening','status':'RESEARCH ONLY - no live changes',
      'signal':'Coin below own MA AND own MA falling vs20 completed days earlier, shifted one day.',
      'event_execution':'Action set at first hourly bar OPEN of causal signal day using only prior-bar peak/stage; same event bar high/close are not used to choose the action.',
      'givebacks':TIGHTEN,'stage_aware':'INITIAL exit; BREAKEVEN 5% below event open; TRAIL_30PLUS 50% giveback.',
      'base_exit':'Frozen Round3H 60% giveback.','capital_proxy':'Realised/cost-basis, not MTM.'},indent=2))
    print(sm.sort_values('combined_pnl',ascending=False).to_string(index=False))
if __name__=='__main__': main()
