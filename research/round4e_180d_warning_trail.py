from __future__ import annotations

import json, math, heapq
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from binance_data import load_symbol
from round2b_validation import add_live_features, controlled_activity
from round3a_exit_architecture import capped_stop, net_pnl

START_CAPITAL = 1750.0
POSITION_USDT = 300.0
BASE_GIVEBACK = 0.60
THRESHOLDS = [0.30, 0.50, 1.00, 2.00, 3.00, 5.00, 7.50]
WARNING_GIVEBACKS = [0.20, 0.30, 0.40, 0.50]
VARIANTS = ['CONTROL_STAGE_AWARE'] + [f'WARN{int(g*100)}_{mode}' for g in WARNING_GIVEBACKS for mode in ['RESET','RATCHET']]


def daily_states(btc: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    x = btc[['time','close']].copy()
    x['time'] = pd.to_datetime(x.time, utc=True)
    d = x.set_index('time').close.astype(float).resample('1D').last().dropna()
    ma180 = d.rolling(180, min_periods=150).mean()
    ma200 = d.rolling(200, min_periods=180).mean()
    warn_raw = (d < ma180) & ((ma180 / ma180.shift(20) - 1) < 0)
    # Round 4C warning diagnostic used 3-day confirmation for the 180D AND state.
    warn_confirmed = warn_raw.rolling(3).sum().eq(3)
    bear_raw = (d < ma200) & ((ma200 / ma200.shift(20) - 1) < 0)
    # Causal: intraday bars on day D see completed state through D-1 only.
    return warn_confirmed.shift(1).fillna(False).astype(bool), bear_raw.shift(1).fillna(False).astype(bool)


def state_at(t, s: pd.Series) -> bool:
    day = pd.Timestamp(t).tz_convert('UTC').floor('D')
    return bool(s.get(day, False))


def result(sig, variant, exit_time, reason, pnl, bars, normal_pnl=None, warn_seen=False,
           warn_starts=0, warn_clears=0, bear_seen=False, warning_stop_hits=0, is_open=False):
    normal_pnl = pnl if normal_pnl is None else normal_pnl
    return {
        'signal_id':sig['signal_id'],'symbol':sig['symbol'],'entry_time':sig['entry_time'],
        'variant':variant,'exit_time':exit_time,'exit_reason':reason,'pnl_usdt':float(pnl),
        'normal_pnl_usdt':float(normal_pnl),'counterfactual_delta_usdt':float(pnl-normal_pnl),
        'bars_held':int(bars),'warn_seen':bool(warn_seen),'warn_starts':int(warn_starts),
        'warn_clears':int(warn_clears),'bear_seen':bool(bear_seen),'warning_stop_hits':int(warning_stop_hits),
        'is_open':bool(is_open),
    }


def parse_variant(v):
    if v == 'CONTROL_STAGE_AWARE': return None, 'CONTROL'
    gb = int(v.split('_')[0].replace('WARN','')) / 100.0
    return gb, v.split('_')[1]


def trail_stop(entry, peak, close, giveback):
    peak_gain = peak / entry - 1
    if peak_gain < 0.30: return None
    floor_gain = peak_gain * (1 - giveback)
    return min(entry * (1 + floor_gain), close * 0.999)


def simulate(df, sig, variant, warning_daily, bear_daily, cfg, normal_pnl=None):
    start = int(sig['entry_index']); entry=float(sig['entry_price'])
    init_stop = capped_stop(entry, float(sig['structural_stop']), 0.10)
    entry_day = pd.Timestamp(sig['entry_time']).tz_convert('UTC').floor('D')
    if bool(bear_daily.get(entry_day, False)):
        return result(sig, variant, sig['entry_time'], 'ENTRY_GATE', 0.0, 0, normal_pnl=0.0)

    warn_gb, mode = parse_variant(variant)
    stop=init_stop; activated=False; peak=entry
    last_warn=state_at(df.iloc[start].time, warning_daily)
    last_bear=state_at(df.iloc[start].time, bear_daily)
    warning_active = last_warn and not last_bear
    warn_seen=warning_active; warn_starts=int(warning_active); warn_clears=0; bear_seen=False; warning_stop_hits=0

    for j in range(start, len(df)):
        b=df.iloc[j]; low,high,close,op=map(float,(b.low,b.high,b.close,b.open))
        current_warn=state_at(b.time, warning_daily)
        current_bear=state_at(b.time, bear_daily)
        warn_start=current_warn and not last_warn
        warn_clear=(not current_warn) and last_warn
        bear_start=current_bear and not last_bear
        last_warn=current_warn; last_bear=current_bear

        peak=max(peak,high)
        peak_gain=peak/entry-1
        stage='INITIAL_PROTECTION' if not activated else ('TRAIL_30PLUS' if peak_gain>=0.30 else 'BREAKEVEN_PROTECTION')

        if warn_start and not current_bear:
            warning_active=True; warn_seen=True; warn_starts+=1
        if warn_clear and warning_active and not current_bear:
            warning_active=False; warn_clears+=1
            if variant != 'CONTROL_STAGE_AWARE' and mode == 'RESET' and activated:
                # Literal reset requested: return stop to the current standard 60% trail, subject to inherited floors.
                base = entry
                if peak_gain >= 0.30: base=max(base,entry*1.10)
                cand=trail_stop(entry,peak,close,BASE_GIVEBACK)
                if cand is not None: base=max(base,cand)
                stop=base

        if bear_start and j > start:
            bear_seen=True; warning_active=False
            # Exact Round 4D winning STAGE_AWARE response.
            if stage == 'INITIAL_PROTECTION':
                pnl=net_pnl(entry,[(1.0,op)],cfg)
                return result(sig,variant,b.time,'BEAR_STAGE_INITIAL_LIQUIDATION',pnl,j-start+1,normal_pnl,
                              warn_seen,warn_starts,warn_clears,bear_seen,warning_stop_hits)
            if stage == 'BREAKEVEN_PROTECTION':
                stop=max(stop,op*0.95)
            # TRAIL_30PLUS remains on its current trail.

        if not activated:
            if low <= init_stop:
                pnl=net_pnl(entry,[(1.0,init_stop)],cfg)
                return result(sig,variant,b.time,'STOP',pnl,j-start+1,normal_pnl,warn_seen,warn_starts,warn_clears,bear_seen,warning_stop_hits)
            if high >= entry*1.05:
                activated=True; stop=max(stop,entry); continue
        else:
            if low <= stop:
                if warning_active and variant != 'CONTROL_STAGE_AWARE': warning_stop_hits += 1
                pnl=net_pnl(entry,[(1.0,stop)],cfg)
                return result(sig,variant,b.time,'GAIN_TRAIL_STOP',pnl,j-start+1,normal_pnl,warn_seen,warn_starts,warn_clears,bear_seen,warning_stop_hits)
            if high >= entry*11.0:
                px=entry*11.0; pnl=net_pnl(entry,[(1.0,px)],cfg)
                return result(sig,variant,b.time,'TAKE_PROFIT',pnl,j-start+1,normal_pnl,warn_seen,warn_starts,warn_clears,bear_seen,warning_stop_hits)

            gb = BASE_GIVEBACK
            if warning_active and variant != 'CONTROL_STAGE_AWARE': gb = warn_gb
            if peak_gain >= 0.30: stop=max(stop,entry*1.10)
            cand=trail_stop(entry,peak,close,gb)
            if cand is not None:
                if mode == 'RESET' and variant != 'CONTROL_STAGE_AWARE':
                    # While warning is active it can tighten; once inactive reset logic above may lower it.
                    stop=max(stop,cand)
                else:
                    stop=max(stop,cand)

    close=float(df.iloc[-1].close); pnl=net_pnl(entry,[(1.0,close)],cfg)
    return result(sig,variant,df.iloc[-1].time,'DATA_END_OPEN',pnl,len(df)-start,normal_pnl,
                  warn_seen,warn_starts,warn_clears,bear_seen,warning_stop_hits,True)


def process_symbol(symbol, btc, warning_daily, bear_daily, cfg):
    df=btc.copy() if symbol=='BTCUSDT' else load_symbol(symbol,cfg['interval'],cfg['start'],cfg['end'])
    if len(df)<800:return [],[]
    x=add_live_features(df,btc); sigs=controlled_activity(symbol,x,cfg); rows=[]
    for sig in sigs:
        base=simulate(df,sig,'CONTROL_STAGE_AWARE',warning_daily,bear_daily,cfg,None)
        base_pnl=float(base['pnl_usdt']); base['normal_pnl_usdt']=base_pnl; base['counterfactual_delta_usdt']=0.0
        rows.append(base)
        for v in VARIANTS[1:]: rows.append(simulate(df,sig,v,warning_daily,bear_daily,cfg,base_pnl))
    return sigs,rows


def pf(vals):
    vals=np.asarray(vals,float); pos=vals[vals>0]; neg=vals[vals<0]
    return float(pos.sum()/abs(neg.sum())) if len(neg) and neg.sum() else math.inf


def capital_sim(g):
    x=g.sort_values(['entry_time','symbol','signal_id']).copy(); cash=START_CAPITAL; heap=[]; open_principal=0.0; peak=START_CAPITAL; max_dd=0.0; skipped=0
    for _,r in x.iterrows():
        t=r.entry_time
        while heap and heap[0][0] <= t:
            et,principal,pnl=heapq.heappop(heap); cash+=principal+pnl; open_principal-=principal
            eq=cash+open_principal; peak=max(peak,eq); max_dd=min(max_dd,eq-peak)
        if r.exit_reason=='ENTRY_GATE':continue
        if cash+1e-9<POSITION_USDT: skipped+=1; continue
        cash-=POSITION_USDT; open_principal+=POSITION_USDT
        heapq.heappush(heap,(r.exit_time,POSITION_USDT,float(r.pnl_usdt)))
    while heap:
        et,principal,pnl=heapq.heappop(heap); cash+=principal+pnl; open_principal-=principal
        eq=cash+open_principal; peak=max(peak,eq); max_dd=min(max_dd,eq-peak)
    return cash-START_CAPITAL,max_dd,skipped


def summarise(trades):
    q=trades.copy();q['entry_time']=pd.to_datetime(q.entry_time,utc=True);q['exit_time']=pd.to_datetime(q.exit_time,utc=True)
    rows=[];years=[]
    for v,g in q.groupby('variant'):
        vals=g.pnl_usdt.to_numpy(float);active=g[g.exit_reason!='ENTRY_GATE'];cap,dd,skip=capital_sim(g)
        rows.append({'variant':v,'signals':len(g),'active_trades':len(active),'entry_gate_skips':int((g.exit_reason=='ENTRY_GATE').sum()),
                     'warning_exposed_trades':int(g.warn_seen.sum()),'warning_stop_hits':int(g.warning_stop_hits.sum()),
                     'bear_exposed_trades':int(g.bear_seen.sum()),'combined_pnl':float(vals.sum()),'profit_factor':pf(vals[vals!=0]),
                     'counterfactual_delta':float(g.counterfactual_delta_usdt.sum()),'capital_profit_proxy':float(cap),
                     'capital_max_realised_dd':float(dd),'skipped_due_cash':int(skip)})
        for y,gy in g.groupby(g.entry_time.dt.year):
            years.append({'variant':v,'year':int(y),'pnl_usdt':float(gy.pnl_usdt.sum()),
                          'counterfactual_delta':float(gy.counterfactual_delta_usdt.sum()),
                          'warning_exposed_trades':int(gy.warn_seen.sum()),'warning_stop_hits':int(gy.warning_stop_hits.sum())})
    return pd.DataFrame(rows),pd.DataFrame(years)


def main():
    cfg=json.loads(Path('research/config.json').read_text());out=Path('research/results/round4e');out.mkdir(parents=True,exist_ok=True)
    btc=load_symbol('BTCUSDT',cfg['interval'],cfg['start'],cfg['end']); warning,bear=daily_states(btc)
    all_s=[];all_t=[]
    with ThreadPoolExecutor(max_workers=6) as ex:
        fut={ex.submit(process_symbol,s,btc,warning,bear,cfg):s for s in cfg['symbols']}
        for f in as_completed(fut):
            s=fut[f]
            try:
                sig,rows=f.result();all_s+=sig;all_t+=rows;print(s,len(sig),flush=True)
            except Exception as e:print('ERROR',s,repr(e),flush=True)
    signals=pd.DataFrame(all_s);trades=pd.DataFrame(all_t);sm,yr=summarise(trades)
    signals.to_csv(out/'signals.csv',index=False);trades.to_csv(out/'trades.csv',index=False)
    sm.to_csv(out/'summary.csv',index=False);yr.to_csv(out/'year_summary.csv',index=False)
    pd.DataFrame({'warning180':warning,'bear200':bear}).to_csv(out/'regime_states.csv')
    manifest={
      'study':'Round 4E 180D early-warning temporary gain-trail tightening',
      'status':'RESEARCH ONLY - does not alter live strategy',
      'warning180':'BTC below 180D MA AND 180D MA falling vs 20d prior, 3 consecutive completed daily states, shifted one full day before use.',
      'bear200':'Frozen Round 4D definition: BTC below 200D MA AND 200D MA falling vs 20 completed days prior, shifted one full day before use.',
      'control':'MA200 entry gate + frozen 60% accumulated-profit giveback + Round 4D winning STAGE_AWARE response on MA200 confirmation.',
      'warning_variants':'Temporary giveback 20/30/40/50% while 180D warning active.',
      'reset_modes':{'RESET':'when warning clears before MA200, lower stop back to current standard 60% trail/floors','RATCHET':'when warning clears, resume 60% calculations but never lower the already-ratcheted stop'},
      'confirmed_bear_action':'Exact Round 4D STAGE_AWARE: INITIAL full liquidation; BREAKEVEN tighten to 5% below confirmation open; TRAIL_30PLUS unchanged.',
      'entry_gate':'Only MA200 bear state blocks new entries; 180D warning changes existing-position protection only.',
      'capital_proxy':'1750 USDT starting cash, 300 USDT per trade; realised/cost-basis drawdown proxy, not mark-to-market.'
    }
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('\nSUMMARY\n',sm.sort_values('combined_pnl',ascending=False).to_string(index=False));print('\nYEARS\n',yr.to_string(index=False))

if __name__=='__main__':main()
