from __future__ import annotations

import json, math, heapq
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from binance_data import load_symbol
from round2b_validation import add_live_features, controlled_activity
from round3a_exit_architecture import capped_stop, net_pnl

START_CAPITAL=1750.0
POSITION_USDT=300.0
BASE_GIVEBACK=0.60
VARIANTS=['BTC200_GATE','COIN150_GATE','COIN180_GATE','COIN200_GATE','COIN200_FULL_EXIT','COIN200_TIGHTEN50','COIN200_TIGHTEN40','COIN200_STAGE_AWARE']


def daily_state(df, window):
    x=df[['time','close']].copy(); x['time']=pd.to_datetime(x.time,utc=True)
    d=x.set_index('time').close.astype(float).resample('1D').last().dropna()
    ma=d.rolling(window,min_periods=max(30,int(window*0.9))).mean()
    raw=(d<ma)&((ma/ma.shift(20)-1)<0)
    # Causal prior-day completed state.
    return raw.shift(1).fillna(False).astype(bool)


def at_state(t,s):
    return bool(s.get(pd.Timestamp(t).tz_convert('UTC').floor('D'),False))


def trail_stop(entry,peak,close,giveback):
    pg=peak/entry-1
    if pg<0.30:return None
    floor=pg*(1-giveback)
    return min(entry*(1+floor),close*0.999)


def result(sig,variant,exit_time,reason,pnl,bars,event=False,event_time=None,event_stage=None,is_open=False):
    return {'signal_id':sig['signal_id'],'symbol':sig['symbol'],'entry_time':sig['entry_time'],'variant':variant,
            'exit_time':exit_time,'exit_reason':reason,'pnl_usdt':float(pnl),'bars_held':int(bars),
            'coin200_event':bool(event),'coin200_event_time':event_time,'coin200_stage':event_stage,'is_open':bool(is_open)}


def simulate(df,sig,variant,states,cfg):
    start=int(sig['entry_index']); entry=float(sig['entry_price']); init_stop=capped_stop(entry,float(sig['structural_stop']),0.10)
    gate={'BTC200_GATE':'btc200','COIN150_GATE':'coin150','COIN180_GATE':'coin180','COIN200_GATE':'coin200',
          'COIN200_FULL_EXIT':'coin200','COIN200_TIGHTEN50':'coin200','COIN200_TIGHTEN40':'coin200','COIN200_STAGE_AWARE':'coin200'}[variant]
    entry_day=pd.Timestamp(sig['entry_time']).tz_convert('UTC').floor('D')
    if bool(states[gate].get(entry_day,False)):
        return result(sig,variant,sig['entry_time'],'ENTRY_GATE',0.0,0)

    stop=init_stop; activated=False; peak=entry; last_coin200=at_state(df.iloc[start].time,states['coin200'])
    event_seen=False
    for j in range(start,len(df)):
        b=df.iloc[j]; low,high,close,op=map(float,(b.low,b.high,b.close,b.open))
        current_coin200=at_state(b.time,states['coin200']); coin200_start=current_coin200 and not last_coin200; last_coin200=current_coin200
        peak=max(peak,high); pg=peak/entry-1
        stage='INITIAL_PROTECTION' if not activated else ('TRAIL_30PLUS' if pg>=0.30 else 'BREAKEVEN_PROTECTION')
        if variant.startswith('COIN200_') and variant!='COIN200_GATE' and coin200_start and j>start and not event_seen:
            event_seen=True
            if variant=='COIN200_FULL_EXIT':
                return result(sig,variant,b.time,'COIN200_FULL_EXIT',net_pnl(entry,[(1.0,op)],cfg),j-start+1,True,b.time,stage)
            if variant=='COIN200_TIGHTEN50' and activated:
                cand=trail_stop(entry,peak,close,0.50)
                if cand is not None: stop=max(stop,cand)
            elif variant=='COIN200_TIGHTEN40' and activated:
                cand=trail_stop(entry,peak,close,0.40)
                if cand is not None: stop=max(stop,cand)
            elif variant=='COIN200_STAGE_AWARE':
                if stage=='INITIAL_PROTECTION':
                    return result(sig,variant,b.time,'COIN200_STAGE_INITIAL_EXIT',net_pnl(entry,[(1.0,op)],cfg),j-start+1,True,b.time,stage)
                elif stage=='BREAKEVEN_PROTECTION':
                    stop=max(stop,op*0.95)
                elif stage=='TRAIL_30PLUS':
                    cand=trail_stop(entry,peak,close,0.50)
                    if cand is not None: stop=max(stop,cand)

        if not activated:
            if low<=init_stop:
                return result(sig,variant,b.time,'STOP',net_pnl(entry,[(1.0,init_stop)],cfg),j-start+1,event_seen)
            if high>=entry*1.05:
                activated=True; stop=max(stop,entry); continue
        else:
            if low<=stop:
                return result(sig,variant,b.time,'GAIN_TRAIL_STOP',net_pnl(entry,[(1.0,stop)],cfg),j-start+1,event_seen)
            if high>=entry*11.0:
                px=entry*11.0; return result(sig,variant,b.time,'TAKE_PROFIT',net_pnl(entry,[(1.0,px)],cfg),j-start+1,event_seen)
            if pg>=0.30: stop=max(stop,entry*1.10)
            cand=trail_stop(entry,peak,close,BASE_GIVEBACK)
            if cand is not None: stop=max(stop,cand)

    close=float(df.iloc[-1].close)
    return result(sig,variant,df.iloc[-1].time,'DATA_END_OPEN',net_pnl(entry,[(1.0,close)],cfg),len(df)-start,event_seen,is_open=True)


def process_symbol(symbol,btc,btc200,cfg):
    df=btc.copy() if symbol=='BTCUSDT' else load_symbol(symbol,cfg['interval'],cfg['start'],cfg['end'])
    if len(df)<800:return [],[]
    states={'btc200':btc200,'coin150':daily_state(df,150),'coin180':daily_state(df,180),'coin200':daily_state(df,200)}
    x=add_live_features(df,btc); sigs=controlled_activity(symbol,x,cfg); rows=[]
    for sig in sigs:
        for v in VARIANTS: rows.append(simulate(df,sig,v,states,cfg))
    return sigs,rows


def pf(vals):
    vals=np.asarray(vals,float); pos=vals[vals>0]; neg=vals[vals<0]
    return float(pos.sum()/abs(neg.sum())) if len(neg) and neg.sum() else math.inf


def capital_sim(g):
    x=g.sort_values(['entry_time','symbol','signal_id']).copy(); cash=START_CAPITAL; heap=[]; open_principal=0.0; peak=START_CAPITAL; max_dd=0.0; skipped=0
    for _,r in x.iterrows():
        t=r.entry_time
        while heap and heap[0][0]<=t:
            et,principal,pnl=heapq.heappop(heap); cash+=principal+pnl; open_principal-=principal
            eq=cash+open_principal; peak=max(peak,eq); max_dd=min(max_dd,eq-peak)
        if r.exit_reason=='ENTRY_GATE':continue
        if cash+1e-9<POSITION_USDT: skipped+=1; continue
        cash-=POSITION_USDT; open_principal+=POSITION_USDT; heapq.heappush(heap,(r.exit_time,POSITION_USDT,float(r.pnl_usdt)))
    while heap:
        et,principal,pnl=heapq.heappop(heap); cash+=principal+pnl; open_principal-=principal
        eq=cash+open_principal; peak=max(peak,eq); max_dd=min(max_dd,eq-peak)
    return cash-START_CAPITAL,max_dd,skipped


def summarise(trades):
    q=trades.copy(); q['entry_time']=pd.to_datetime(q.entry_time,utc=True); q['exit_time']=pd.to_datetime(q.exit_time,utc=True)
    rows=[]; years=[]
    for v,g in q.groupby('variant'):
        vals=g.pnl_usdt.to_numpy(float); active=g[g.exit_reason!='ENTRY_GATE']; cap,dd,skip=capital_sim(g)
        rows.append({'variant':v,'signals':len(g),'active_trades':len(active),'entry_gate_skips':int((g.exit_reason=='ENTRY_GATE').sum()),
                     'coin200_events':int(g.coin200_event.sum()),'combined_pnl':float(vals.sum()),'profit_factor':pf(vals[vals!=0]),
                     'win_rate_active':float((active.pnl_usdt>0).mean()) if len(active) else np.nan,'capital_profit_proxy':float(cap),
                     'capital_max_realised_dd':float(dd),'skipped_due_cash':int(skip)})
        for y,gy in g.groupby(g.entry_time.dt.year): years.append({'variant':v,'year':int(y),'pnl_usdt':float(gy.pnl_usdt.sum())})
    return pd.DataFrame(rows),pd.DataFrame(years)


def main():
    cfg=json.loads(Path('research/config.json').read_text()); out=Path('research/results/round4g'); out.mkdir(parents=True,exist_ok=True)
    btc=load_symbol('BTCUSDT',cfg['interval'],cfg['start'],cfg['end']); btc200=daily_state(btc,200)
    all_s=[]; all_t=[]
    with ThreadPoolExecutor(max_workers=6) as ex:
        fut={ex.submit(process_symbol,s,btc,btc200,cfg):s for s in cfg['symbols']}
        for f in as_completed(fut):
            s=fut[f]
            try:
                sig,rows=f.result(); all_s+=sig; all_t+=rows; print(s,len(sig),flush=True)
            except Exception as e: print('ERROR',s,repr(e),flush=True)
    trades=pd.DataFrame(all_t); pd.DataFrame(all_s).to_csv(out/'signals.csv',index=False); trades.to_csv(out/'trades.csv',index=False)
    sm,yr=summarise(trades); sm.to_csv(out/'summary.csv',index=False); yr.to_csv(out/'year_summary.csv',index=False)
    manifest={'study':'Round 4G coin-specific regime gate and open-position action comparison','status':'RESEARCH ONLY - no live strategy changes',
              'signal':'Each MA state = coin/BTC close below MA AND MA falling vs 20 completed days earlier, shifted one full day causally.',
              'gate_comparison':['BTC200_GATE','COIN150_GATE','COIN180_GATE','COIN200_GATE'],
              'coin200_actions':['COIN200_GATE','COIN200_FULL_EXIT','COIN200_TIGHTEN50','COIN200_TIGHTEN40','COIN200_STAGE_AWARE'],
              'base_exit':'Frozen Round 3H 60% accumulated-profit giveback, +5 breakeven, +30 +10 floor, +1000% take profit.',
              'tighten50':'At first confirmed coin200 transition while open, 30%+ trail temporarily ratchets to 50% giveback; ratchet never lowers.',
              'tighten40':'Same with 40% giveback.','stage_aware':'Initial stage exits; breakeven stage tightens to 5% below event open; 30%+ stage tightens to 50% giveback.',
              'capital_proxy':'1750 start, 300 per trade, realised/cost-basis drawdown proxy, not exact mark-to-market.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('\nSUMMARY\n',sm.sort_values('combined_pnl',ascending=False).to_string(index=False)); print('\nYEARS\n',yr.to_string(index=False))

if __name__=='__main__': main()
