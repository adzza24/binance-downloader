from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import explosive_move_real_world_replay_round8 as r8

SRC = Path('research/results/explosive_move_entry_route_round9')
OUT = Path('research/results/explosive_move_oco_grid_round10')
HORIZON_H = 720
COST = 0.003
TARGETS = [0.05, 0.07, 0.10]
STOPS = [0.02, 0.03, 0.05]
LIMIT_OFFSET = 0.05


def first_touch(open_, high, low, close, start, end, entry, target, stop, fill_candle=False):
    for k in range(start, end + 1):
        hs = low[k] <= entry * (1 - stop)
        ht = high[k] >= entry * (1 + target)
        if hs and ht:
            return 'STOP', k
        if hs:
            return 'STOP', k
        if ht:
            return 'TARGET', k
    return 'OPEN', None


def straight(open_, high, low, close, i, end_i, entry, target, stop):
    res, ex = first_touch(open_, high, low, close, i + 1, end_i, entry, target, stop)
    if res == 'TARGET': gross = target
    elif res == 'STOP': gross = -stop
    else: gross = close[end_i] / entry - 1
    return res, ex, float(gross - COST)


def limit_route(open_, high, low, close, i, end_i, signal_entry, target, stop):
    limit = signal_entry * (1 - LIMIT_OFFSET)
    fill_i = None
    for k in range(i + 1, end_i + 1):
        if low[k] <= limit:
            fill_i = k
            break
    if fill_i is None:
        return 'UNFILLED', None, 0.0, False, np.nan

    target_px = limit * (1 + target)
    stop_px = limit * (1 - stop)
    amb = bool(high[fill_i] >= target_px and open_[fill_i] > limit)

    # Conservative fill-candle sequencing.
    if low[fill_i] <= stop_px:
        res, ex = 'STOP', fill_i
    elif open_[fill_i] <= limit and high[fill_i] >= target_px:
        res, ex = 'TARGET', fill_i
    else:
        res, ex = first_touch(open_, high, low, close, fill_i + 1, end_i, limit, target, stop)

    if res == 'TARGET': gross = target
    elif res == 'STOP': gross = -stop
    else: gross = close[end_i] / limit - 1
    return res, ex, float(gross - COST), amb, fill_i - i


def process_symbol(symbol, sg, end):
    df = r8.load_extended(symbol, end)
    if df.empty: return [], [{'symbol': symbol, 'error': 'no_data'}]
    df['time'] = pd.to_datetime(df['time'], utc=True)
    time_to_i = {pd.Timestamp(t): i for i, t in enumerate(df['time'])}
    open_ = df.open.to_numpy(float); high = df.high.to_numpy(float)
    low = df.low.to_numpy(float); close = df.close.to_numpy(float)
    rows=[]; failures=[]
    for s in sg.itertuples(index=False):
        ts = pd.Timestamp(s.signal_time); i = time_to_i.get(ts)
        if i is None:
            failures.append({'symbol':symbol,'signal_time':str(ts),'error':'signal_time_not_found'}); continue
        if len(df) - i - 1 < HORIZON_H: continue
        end_i=i+HORIZON_H; entry=close[i]
        base = {'symbol':symbol,'signal_time':ts,'year':ts.year,'replay_tier':s.replay_tier,'candidate_family':s.candidate_family,'signal_checkpoint_h':s.signal_checkpoint_h}
        for t in TARGETS:
            for st in STOPS:
                sr,se,snet=straight(open_,high,low,close,i,end_i,entry,t,st)
                lr,le,lnet,amb,fh=limit_route(open_,high,low,close,i,end_i,entry,t,st)
                rows.append({**base,'target':t,'stop':st,
                             'straight_result':sr,'straight_net':snet,
                             'limit_result':lr,'limit_net_per_signal':lnet,
                             'limit_filled':lr!='UNFILLED','limit_fill_hours':fh,
                             'fill_target_ambiguous':amb})
    return rows, failures


def summarise(z):
    rows=[]
    for route in ['straight','limit']:
        for t in TARGETS:
            for st in STOPS:
                q=z[(z.target==t)&(z.stop==st)]
                for year in ['ALL',2024,2025,2026]:
                    g=q if year=='ALL' else q[q.year==year]
                    if g.empty: continue
                    if route=='straight':
                        trades=len(g); target=(g.straight_result=='TARGET').mean(); stop=(g.straight_result=='STOP').mean(); avg=g.straight_net.mean(); fill=1.0; filled=trades
                    else:
                        filledg=g[g.limit_filled]
                        trades=len(g); filled=len(filledg); fill=g.limit_filled.mean(); target=(filledg.limit_result=='TARGET').mean() if filled else np.nan; stop=(filledg.limit_result=='STOP').mean() if filled else np.nan; avg=g.limit_net_per_signal.mean()
                    rows.append({'route':route,'target':t,'stop':st,'year':year,'signals':len(g),'filled_trades':filled,'fill_rate':fill,'target_rate_per_executed':target,'stop_rate_per_executed':stop,'avg_net_per_original_signal':avg})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sig=pd.read_csv(SRC/'signal_strategy_paths.csv.gz',compression='gzip')
    sig['signal_time']=pd.to_datetime(sig.signal_time,utc=True)
    sig=sig[sig.mature_30d_recalc.astype(bool)].copy()
    end=sig.signal_time.max()+pd.Timedelta(hours=HORIZON_H+2)
    now=pd.Timestamp.now(tz='UTC').floor('h')-pd.Timedelta(hours=1); end=min(end,now)
    rows=[]; failures=[]
    for symbol,g in sig.groupby('symbol',sort=False):
        try:
            rr,ff=process_symbol(symbol,g,end); rows.extend(rr); failures.extend(ff); print(symbol,len(rr),flush=True)
        except Exception as e:
            failures.append({'symbol':symbol,'error':repr(e)}); print('ERROR',symbol,repr(e),flush=True)
    z=pd.DataFrame(rows); z.to_csv(OUT/'oco_grid_paths.csv.gz',index=False,compression='gzip')
    pd.DataFrame(failures).to_csv(OUT/'failures.csv',index=False)
    s=summarise(z); s.to_csv(OUT/'oco_grid_summary.csv',index=False)

    all_=s[s.year.astype(str)=='ALL'].copy()
    # Robust rank: high mean, then worst yearly mean.
    yr=s[s.year.astype(str)!='ALL'].copy()
    yr['year']=yr.year.astype(str)
    piv=yr.pivot_table(index=['route','target','stop'],columns='year',values='avg_net_per_original_signal')
    piv['worst_year']=piv.min(axis=1); piv['mean_year']=piv.mean(axis=1)
    ranked=all_.merge(piv.reset_index(),on=['route','target','stop'],how='left').sort_values(['avg_net_per_original_signal','worst_year'],ascending=False)
    ranked.to_csv(OUT/'oco_grid_ranked.csv',index=False)

    lines=['# Explosive Move OCO Grid Round 10','', '**RESEARCH ONLY. No live strategy or automation changes.**','',
           '- Same 463 mature Round 9 full-population signals.','- 30-day horizon from original signal for both routes.','- Straight route enters at signal close.','- Limit route buys 5% below signal if touched, then places OCO.','- 0.30% round-trip costs on executed trades; unfilled limits = 0 return.','- Conservative 1h ordering: stop wins same-candle target/stop ties; fill-candle target is only credited when open <= limit.','',
           '## Overall grid','', '| Route | Target | Stop | Target rate* | Fill rate | Avg net / original signal | 2024 | 2025 | 2026 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in ranked.itertuples(index=False):
        def p(x): return f'{100*x:.2f}%'
        lines.append(f'| {r.route} | +{100*r.target:.0f}% | -{100*r.stop:.0f}% | {p(r.target_rate_per_executed)} | {p(r.fill_rate)} | **{p(r.avg_net_per_original_signal)}** | {p(getattr(r,"2024"))} | {p(getattr(r,"2025"))} | {p(getattr(r,"2026"))} |')
    lines += ['', '*Target rate is per executed trade; straight-entry executes every signal.', '', 'This is exploratory parameter research on an already-used historical period, not a pristine holdout.']
    (OUT/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    print((OUT/'ANALYSIS.md').read_text(),flush=True)

if __name__=='__main__': main()
