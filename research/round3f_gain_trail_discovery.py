from __future__ import annotations

import json, math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from binance_data import load_symbol
from round2b_validation import add_live_features, controlled_activity
from round3a_exit_architecture import net_pnl, capped_stop

MILESTONES = [0.30, 0.50, 1.00, 2.00, 3.00, 5.00, 7.50, 10.00]
QUANTILES = [0.50, 0.75, 0.90, 0.95]


def base_path(df, sig):
    """Path actually available under inherited rules: -10% cap, +5->BE, +30->+10."""
    start = int(sig['entry_index']); entry = float(sig['entry_price'])
    init_stop = capped_stop(entry, float(sig['structural_stop']), 0.10)
    stop = init_stop; activated = False; floor30 = False
    rows = []
    for j in range(start, len(df)):
        b = df.iloc[j]; low, high, close = map(float, (b.low, b.high, b.close))
        rows.append((j, b.time, low, high, close))
        if not activated:
            if low <= init_stop: break
            if high >= entry * 1.05:
                activated = True; stop = entry; continue
        else:
            if low <= stop: break
            if (not floor30) and high >= entry * 1.30:
                stop = max(stop, entry * 1.10); floor30 = True
    return entry, rows


def req_giveback(entry, rows, start_pos, end_pos):
    """Max fraction of accumulated gain given back before end_pos, using lows vs running peak gain."""
    if end_pos <= start_pos + 1: return 0.0
    peak_gain = 0.0; worst = 0.0
    for k in range(start_pos, end_pos + 1):
        _, _, low, high, _ = rows[k]
        peak_gain = max(peak_gain, high / entry - 1)
        if peak_gain <= 0: continue
        low_gain = low / entry - 1
        giveback = (peak_gain - low_gain) / peak_gain
        worst = max(worst, giveback)
    return float(worst)


def analyse_path(sig, entry, rows):
    if not rows: return [], None
    highs = np.array([r[3] for r in rows], float)
    lows = np.array([r[2] for r in rows], float)
    gains = highs / entry - 1
    ultimate_i = int(np.argmax(gains)); ultimate = float(gains[ultimate_i])
    first = {}
    for m in MILESTONES:
        h = np.flatnonzero(gains >= m)
        first[m] = int(h[0]) if len(h) else None
    segs = []
    for a, b in zip(MILESTONES[:-1], MILESTONES[1:]):
        ia, ib = first[a], first[b]
        if ia is None or ib is None: continue
        # New trail/floor becomes active after the threshold bar, matching prior conservative ordering.
        req = req_giveback(entry, rows, ia + 1, ib)
        segs.append({
            'signal_id': sig['signal_id'], 'symbol': sig['symbol'], 'entry_time': sig['entry_time'],
            'from_gain': a, 'to_gain': b, 'required_gain_giveback': req,
            'hours_to_next': rows[ib][0] - rows[ia][0], 'ultimate_mfe': ultimate,
        })
    # Required giveback from each reached milestone to the ultimate peak.
    for a in MILESTONES[:-1]:
        ia = first[a]
        if ia is None or ultimate_i <= ia: continue
        req = req_giveback(entry, rows, ia + 1, ultimate_i)
        segs.append({
            'signal_id': sig['signal_id'], 'symbol': sig['symbol'], 'entry_time': sig['entry_time'],
            'from_gain': a, 'to_gain': -1.0, 'required_gain_giveback': req,
            'hours_to_next': rows[ultimate_i][0] - rows[ia][0], 'ultimate_mfe': ultimate,
        })
    # Giveback after ultimate peak to the end of the inherited base path.
    peak_gain = max(ultimate, 1e-12)
    post_low_gain = float(lows[ultimate_i:].min() / entry - 1)
    post_peak_giveback = (peak_gain - post_low_gain) / peak_gain
    summary = {
        'signal_id': sig['signal_id'], 'symbol': sig['symbol'], 'entry_time': sig['entry_time'],
        'ultimate_mfe': ultimate, 'ultimate_hour': rows[ultimate_i][0] - rows[0][0],
        'post_peak_gain_giveback': float(post_peak_giveback), 'base_path_hours': len(rows),
    }
    return segs, summary


def qsummary(seg):
    out=[]
    direct = seg[seg.to_gain > 0].copy()
    for (a,b), g in direct.groupby(['from_gain','to_gain']):
        row={'from_gain':a,'to_gain':b,'advancers':len(g)}
        for q in QUANTILES: row[f'p{int(q*100)}_required_giveback']=float(g.required_gain_giveback.quantile(q))
        out.append(row)
    return pd.DataFrame(out)


def rounded_schedule(qs, qcol):
    sched=[]
    for _,r in qs.sort_values('from_gain').iterrows():
        gb = float(r[qcol])
        gb = min(0.95, math.ceil(gb/0.05)*0.05)
        # At +30, inherited +10 floor is equivalent to max 66.667% gain giveback.
        if abs(float(r.from_gain)-0.30)<1e-9: gb=max(gb, 2/3)
        sched.append((float(r.from_gain), gb))
    return sched


def simulate_gain_trail(df, sig, name, schedule, cfg, tp=10.0):
    start=int(sig['entry_index']); entry=float(sig['entry_price'])
    init_stop=capped_stop(entry,float(sig['structural_stop']),0.10)
    stop=init_stop; activated=False; peak=entry; stage_gb=None
    for j in range(start,len(df)):
        b=df.iloc[j]; low,high,close=map(float,(b.low,b.high,b.close)); peak=max(peak,high)
        if not activated:
            if low<=init_stop:
                return result(sig,name,b.time,'STOP',net_pnl(entry,[(1,init_stop)],cfg),False,j-start+1,0)
            if high>=entry*1.05:
                activated=True; stop=entry; continue
        else:
            if low<=stop:
                return result(sig,name,b.time,'GAIN_TRAIL_STOP',net_pnl(entry,[(1,stop)],cfg),False,j-start+1,stop/entry-1)
            if high>=entry*(1+tp):
                px=entry*(1+tp)
                return result(sig,name,b.time,'TAKE_PROFIT',net_pnl(entry,[(1,px)],cfg),False,j-start+1,tp)
            peak_gain=peak/entry-1
            for threshold,gb in schedule:
                if peak_gain>=threshold: stage_gb=gb
            if peak_gain>=0.30:
                stop=max(stop,entry*1.10)
            if stage_gb is not None:
                floor_gain=peak_gain*(1-stage_gb)
                candidate=entry*(1+floor_gain)
                candidate=min(candidate, close*0.999)
                stop=max(stop,candidate)
    close=float(df.iloc[-1].close)
    return result(sig,name,df.iloc[-1].time,'DATA_END_OPEN',net_pnl(entry,[(1,close)],cfg),True,len(df)-start,close/entry-1)


def result(sig,name,t,reason,pnl,is_open,bars,ret):
    return {'signal_id':sig['signal_id'],'symbol':sig['symbol'],'entry_time':sig['entry_time'],'variant':name,
            'exit_time':t,'exit_reason':reason,'pnl_usdt':pnl,'is_open':is_open,'bars_held':bars,'exit_return_pct':ret}


def process_symbol(symbol,btc,cfg):
    df=btc.copy() if symbol=='BTCUSDT' else load_symbol(symbol,cfg['interval'],cfg['start'],cfg['end'])
    if len(df)<800:return [],[],[],df
    x=add_live_features(df,btc); sigs=controlled_activity(symbol,x,cfg)
    segs=[]; paths=[]
    for sig in sigs:
        entry,rows=base_path(df,sig); s,p=analyse_path(sig,entry,rows); segs+=s
        if p: paths.append(p)
    return sigs,segs,paths,df


def summarise_trades(t):
    rows=[]; y=[]; q=t.copy();q['year']=pd.to_datetime(q.entry_time).dt.year
    for v,g in q.groupby('variant'):
        c=g[~g.is_open];o=g[g.is_open];p=c.pnl_usdt.to_numpy(float);w=p[p>0];l=p[p<0]
        rows.append({'variant':v,'trades':len(g),'closed':len(c),'open':len(o),'realised_pnl':c.pnl_usdt.sum(),
                     'open_mtm':o.pnl_usdt.sum(),'combined_mtm':g.pnl_usdt.sum(),
                     'profit_factor':w.sum()/abs(l.sum()) if len(l) and l.sum() else math.inf})
        for yr,gy in g.groupby('year'):
            gc=gy[~gy.is_open];go=gy[gy.is_open]
            y.append({'variant':v,'year':int(yr),'realised_pnl':gc.pnl_usdt.sum(),'open_mtm':go.pnl_usdt.sum(),'combined_mtm':gy.pnl_usdt.sum(),'trades':len(gy)})
    return pd.DataFrame(rows),pd.DataFrame(y)


def main():
    cfg=json.loads(Path('research/config.json').read_text());out=Path('research/results/round3f');out.mkdir(parents=True,exist_ok=True)
    btc=load_symbol('BTCUSDT',cfg['interval'],cfg['start'],cfg['end'])
    all_s=[];all_seg=[];all_p=[]; dfs={}
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut={ex.submit(process_symbol,s,btc,cfg):s for s in cfg['symbols']}
        for f in as_completed(fut):
            s=fut[f]
            try:
                sig,segs,paths,df=f.result();all_s+=sig;all_seg+=segs;all_p+=paths;dfs[s]=df;print(s,len(sig),len(segs),flush=True)
            except Exception as e: print('ERROR',s,repr(e),flush=True)
    signals=pd.DataFrame(all_s);seg=pd.DataFrame(all_seg);paths=pd.DataFrame(all_p)
    qs=qsummary(seg); schedules={}
    for q in [0.75,0.90,0.95]: schedules[f'PATH_P{int(q*100)}']=rounded_schedule(qs,f'p{int(q*100)}_required_giveback')
    schedules['CONST_50']=[(m,0.50) for m in MILESTONES[:-1]]
    schedules['CONST_60']=[(m,0.60) for m in MILESTONES[:-1]]
    schedules['CONST_66_7']=[(m,2/3) for m in MILESTONES[:-1]]
    trades=[]
    for sig in all_s:
        df=dfs.get(sig['symbol']);
        if df is None: continue
        for name,sch in schedules.items(): trades.append(simulate_gain_trail(df,sig,name,sch,cfg))
    tr=pd.DataFrame(trades);sm,yr=summarise_trades(tr)
    signals.to_csv(out/'signals.csv',index=False);seg.to_csv(out/'segment_requirements.csv',index=False);paths.to_csv(out/'path_summary.csv',index=False)
    qs.to_csv(out/'required_giveback_quantiles.csv',index=False);tr.to_csv(out/'trades.csv',index=False);sm.to_csv(out/'summary.csv',index=False);yr.to_csv(out/'year_summary.csv',index=False)
    (out/'schedules.json').write_text(json.dumps(schedules,indent=2))
    manifest={'study':'Round 3F retrospective gain-trail path discovery','definition':'trail percentage = fraction of accumulated gain allowed to be given back, not percentage below total price','purpose':'derive generalised gain-giveback schedules from actual advancing winner paths, then descriptively simulate them','warning':'Schedules are discovered and evaluated on the same historical sample; results are exploratory/in-sample, not independent validation.','tp_cap':'+1000% hard cap in descriptive simulations','base':'structural stop capped 10%; +5->BE; +30->+10 inherited before/alongside trail','milestones':MILESTONES}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('\nREQUIRED GIVEBACK\n',qs.to_string(index=False));print('\nSCHEDULES\n',json.dumps(schedules,indent=2));print('\nRESULTS\n',sm.sort_values('combined_mtm',ascending=False).to_string(index=False))

if __name__=='__main__': main()
