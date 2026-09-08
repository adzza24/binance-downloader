from __future__ import annotations

import json, math, heapq
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from binance_data import load_symbol
from round4_regime_gating import BASE_SCHEDULE, build_regime_features, attach_regime, symbol_work

START_CAPITAL=1750.0
BASE_SIZE=300.0
VARIANTS=['BASE','BELOW200','FALLING200','BEAR_AND_FULL_PAUSE','BEAR_AND_50PCT','BEAR_AND_25PCT','BEAR_OR_FULL_PAUSE']


def exp_for(v,r):
    below=bool(r.btc_dist200<0); falling=bool(r.btc_ma200_slope20<0)
    if v=='BASE':return 1.0
    if v=='BELOW200':return 0.0 if below else 1.0
    if v=='FALLING200':return 0.0 if falling else 1.0
    if v=='BEAR_AND_FULL_PAUSE':return 0.0 if (below and falling) else 1.0
    if v=='BEAR_AND_50PCT':return 0.5 if (below and falling) else 1.0
    if v=='BEAR_AND_25PCT':return 0.25 if (below and falling) else 1.0
    if v=='BEAR_OR_FULL_PAUSE':return 0.0 if (below or falling) else 1.0
    raise KeyError(v)


def trade_metrics(q,v):
    g=q.copy();g['exposure']=g.apply(lambda r:exp_for(v,r),axis=1);g['scaled_pnl']=g.pnl_usdt.astype(float)*g.exposure
    c=g[~g.is_open];x=c.scaled_pnl.to_numpy(float);pos=x[x>0];neg=x[x<0]
    o=c.sort_values('exit_time');curve=START_CAPITAL+o.scaled_pnl.cumsum();peak=curve.cummax();dd=curve-peak
    return {
      'variant':v,'signals':len(g),'active':int((g.exposure>0).sum()),'skipped':int((g.exposure==0).sum()),'reduced':int(((g.exposure>0)&(g.exposure<1)).sum()),
      'realised_pnl':float(c.scaled_pnl.sum()),'open_mtm':float(g[g.is_open].scaled_pnl.sum()),'combined_pnl':float(g.scaled_pnl.sum()),
      'profit_factor':float(pos.sum()/abs(neg.sum())) if len(neg) and neg.sum()!=0 else math.inf,
      'worst_trade_loss':float(neg.min()) if len(neg) else 0,'avg_loss':float(neg.mean()) if len(neg) else 0,
      'max_pseudo_dd':float(dd.min()) if len(dd) else 0,'min_pseudo_equity':float(curve.min()) if len(curve) else START_CAPITAL,
    },g


def capital_sim(g):
    # Event-driven capital-constrained proxy: fixed 300 USDT full-size trades, scaled by regime exposure.
    # Open positions are carried at cost basis between events; realised P&L changes total equity at exit.
    x=g.sort_values(['entry_time','symbol','signal_id']).copy();cash=START_CAPITAL;realised=0.0;open_heap=[];open_principal=0.0
    peak=START_CAPITAL;min_eq=START_CAPITAL;max_dd=0.0;max_conc=0;skipped_cash=0;executed=0
    def settle_until(t):
        nonlocal cash,realised,open_principal,peak,min_eq,max_dd,max_conc
        while open_heap and open_heap[0][0] <= t:
            et,principal,pnl=heapq.heappop(open_heap);cash += principal+pnl;open_principal -= principal;realised += pnl
            eq=cash+open_principal;peak=max(peak,eq);min_eq=min(min_eq,eq);max_dd=min(max_dd,eq-peak)
    for _,r in x.iterrows():
        settle_until(r.entry_time)
        ex=float(r.exposure);principal=BASE_SIZE*ex
        if principal<=0:continue
        if cash+1e-9 < principal:
            skipped_cash+=1;continue
        cash-=principal;open_principal+=principal;executed+=1;max_conc=max(max_conc,len(open_heap)+1)
        pnl=float(r.pnl_usdt)*ex
        heapq.heappush(open_heap,(r.exit_time,principal,pnl))
        eq=cash+open_principal;peak=max(peak,eq);min_eq=min(min_eq,eq);max_dd=min(max_dd,eq-peak)
    settle_until(pd.Timestamp.max.tz_localize('UTC'))
    return {'executed_capital_trades':executed,'skipped_due_cash':skipped_cash,'final_capital_proxy':cash,'capital_profit_proxy':cash-START_CAPITAL,'capital_max_realised_dd':max_dd,'capital_min_equity_proxy':min_eq,'max_concurrent_positions':max_conc,'capital_ruin':bool(min_eq<=0)}


def period_summary(q,label,y0=None,y1=None):
    z=q.copy()
    if y0 is not None:z=z[z.entry_time.dt.year>=y0]
    if y1 is not None:z=z[z.entry_time.dt.year<=y1]
    rows=[]
    for v in VARIANTS:
        m,g=trade_metrics(z,v);m.update(capital_sim(g));m['period']=label;rows.append(m)
    return pd.DataFrame(rows)


def episodes(features,q):
    f=features.dropna(subset=['btc_dist200','btc_ma200_slope20']).copy();f['bear']=(f.btc_dist200<0)&(f.btc_ma200_slope20<0)
    grp=(f.bear!=f.bear.shift()).cumsum();rows=[]
    for _,g in f[f.bear].groupby(grp[f.bear]):
        start,end=g.index.min(),g.index.max();sub=q[(q.entry_time.dt.floor('D')>=start)&(q.entry_time.dt.floor('D')<=end)]
        rows.append({'start':start,'end':end,'days':len(g),'signals':len(sub),'base_pnl':sub.pnl_usdt.sum(),'btc_dist200_start':g.iloc[0].btc_dist200,'btc_ret30_start':g.iloc[0].btc_ret30})
    return pd.DataFrame(rows)


def main():
    cfg=json.loads(Path('research/config.json').read_text());out=Path('research/results/round4b');out.mkdir(parents=True,exist_ok=True)
    btc=load_symbol('BTCUSDT',cfg['interval'],cfg['start'],cfg['end']);all_s=[];all_t=[];closes=[]
    with ThreadPoolExecutor(max_workers=6) as ex:
        fut={ex.submit(symbol_work,s,btc,cfg):s for s in cfg['symbols']}
        for f in as_completed(fut):
            s=fut[f]
            try:
                sig,tr,dc=f.result();all_s+=sig;all_t+=tr
                if dc is not None:closes.append(dc)
                print(s,len(sig),flush=True)
            except Exception as e:print('ERROR',s,repr(e),flush=True)
    features=build_regime_features(closes);q=attach_regime(pd.DataFrame(all_t),features)
    full=period_summary(q,'FULL');disc=period_summary(q,'DISCOVERY_2019_2024',2019,2024);fwd=period_summary(q,'FORWARD_2025_2026',2025,2026)
    summary=pd.concat([full,disc,fwd],ignore_index=True);summary.to_csv(out/'summary.csv',index=False)
    eps=episodes(features,q);eps.to_csv(out/'bear_episodes.csv',index=False)
    years=[]
    for yr,g in q.groupby(q.entry_time.dt.year):
        for v in VARIANTS:
            m,x=trade_metrics(g,v);cm=capital_sim(x);years.append({'year':int(yr),'variant':v,'signals':len(x),'skipped':m['skipped'],'reduced':m['reduced'],'combined_pnl':m['combined_pnl'],'profit_factor':m['profit_factor'],'worst_trade_loss':m['worst_trade_loss'],'capital_profit_proxy':cm['capital_profit_proxy'],'capital_max_realised_dd':cm['capital_max_realised_dd'],'skipped_due_cash':cm['skipped_due_cash']})
    pd.DataFrame(years).to_csv(out/'year_summary.csv',index=False)
    q.to_csv(out/'trades_with_regime.csv',index=False);features.to_csv(out/'daily_regime_features.csv')
    (out/'manifest.json').write_text(json.dumps({
      'study':'Round 4B simple bear-regime confirmation + 1750 USDT capital survival proxy',
      'frozen_base':'Controlled Activity + Round3H FIXED60 exit unchanged',
      'primary_rule':'Pause new entries only when BTC is below its causal 200-day moving average AND that 200-day moving average is falling versus 20 days earlier.',
      'comparators':'Below-200 alone, falling-200 alone, OR rule, plus 25%/50% exposure rather than full pause under the AND state.',
      'causality':'Daily features shifted one full day before use.',
      'capital_proxy':'Starts 1750 USDT, max 300 USDT per full-size signal, refuses entries when free cash is unavailable, settles principal+strategy P&L at exit. Open positions are carried at cost basis between events, so realised drawdown is a survival proxy rather than exact mark-to-market drawdown.',
      'validation_warning':'Regime rules are compared chronologically on 2025-2026 after descriptive 2019-2024 review, but the underlying entry/exit strategy itself was developed on the full historical dataset; not pristine independent validation.'
    },indent=2))
    print(summary.to_string(index=False));print('\nEPISODES\n',eps.to_string(index=False))

if __name__=='__main__':main()
