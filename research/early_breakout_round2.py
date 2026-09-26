from __future__ import annotations

import json, math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

from binance_data import load_symbol
from early_breakout_round1 import REFERENCE_SYMBOLS, TASK_EXCLUSIONS, feature_frame, raw_candidates, dedupe_episodes

OUT=Path('research/results/early_breakout_round2')
POS=300.0
CAPS=[1,2,3,4,5,6,7,30]
RUN_CAPS=[7,30,90,180,365]
STOPS=['FIXED3','FIXED5','FIXED7','FIXED10','STRUCT_CAP10']
RUN_STOPS=['FIXED5','FIXED7','FIXED10','STRUCT_CAP10']
TPS=[.10,.20]
GBS=[.30,.40,.50,.60]
PTS=[.10,.20]
SELECTED={
 'TIME_FIXED5_CAP7D','TIME_FIXED7_CAP7D','TIME_STRUCT_CAP10_CAP7D','TIME_FIXED7_CAP30D',
 'FIXEDTP_FIXED5_TP10_CAP7D','FIXEDTP_FIXED7_TP10_CAP7D','FIXEDTP_FIXED5_TP20_CAP7D','FIXEDTP_FIXED7_TP20_CAP7D',
 'FIXEDTP_FIXED7_TP10_CAP30D','FIXEDTP_FIXED7_TP20_CAP30D',
 'FULL_TRAIL_FIXED7_GB30_CAP30D','FULL_TRAIL_FIXED7_GB40_CAP30D','FULL_TRAIL_FIXED7_GB60_CAP30D',
 'FULL_TRAIL_STRUCT_CAP10_GB60_CAP365D',
 'HALF_TP_RUNNER_FIXED7_PT10_GB40_CAP30D','HALF_TP_RUNNER_FIXED7_PT20_GB40_CAP30D',
 'HALF_TP_RUNNER_FIXED7_PT10_GB60_CAP365D','HALF_TP_RUNNER_FIXED7_PT20_GB60_CAP365D'}

def pf(x):
 x=np.asarray(x,float); p=x[x>0].sum(); n=x[x<0].sum(); return float(p/abs(n)) if n<0 else math.inf

def pnl(entry, exits, cfg):
 fee=float(cfg['fee_rate']); slip=float(cfg['slippage_rate']); qty=POS/entry; out=-POS*fee
 for frac,raw in exits:
  px=raw*(1-slip); proceeds=qty*frac*px; out += proceeds-POS*frac-proceeds*fee
 return float(out)

def stop_px(sig,mode):
 e=float(sig.entry_price); raw=min(float(sig.structural_support)*.995,e*.999)
 if mode.startswith('FIXED'): return e*(1-int(mode[5:])/100)
 if mode=='STRUCT_CAP10': return max(raw,e*.90)
 raise ValueError(mode)

def cap_i(times,entry_i,days):
 target=times[entry_i]+int(days*24*3600*1e9); j=int(np.searchsorted(times,target,'left')); return j if j<len(times) else None

def first(mask):
 x=np.flatnonzero(mask); return int(x[0]) if len(x) else None

def rec(sig,variant,family,df,j,reason,exits,cfg):
 et=pd.Timestamp(sig.entry_time); xt=pd.Timestamp(df.time.iloc[j])
 return (variant,family,sig.signal_id,sig.symbol,sig.stage,int(et.year),float((xt-et).total_seconds()/3600),reason,pnl(float(sig.entry_price),exits,cfg))

def time_out(df,ei,sig,mode,days,cfg):
 times=df.time.astype('int64').to_numpy(); ci=cap_i(times,ei,days); end=ci if ci is not None else len(df)
 stop=stop_px(sig,mode); hit=first(df.low.iloc[ei:end].to_numpy(float)<=stop); v=f'TIME_{mode}_CAP{days}D'
 if hit is not None: return rec(sig,v,'TIME_ONLY',df,ei+hit,'STOP',[(1.,stop)],cfg)
 if ci is not None: return rec(sig,v,'TIME_ONLY',df,ci,'TIME_CAP',[(1.,float(df.open.iloc[ci]))],cfg)

def tp_out(df,ei,sig,mode,tp,days,cfg):
 times=df.time.astype('int64').to_numpy(); ci=cap_i(times,ei,days); end=ci if ci is not None else len(df)
 e=float(sig.entry_price); stop=stop_px(sig,mode); target=e*(1+tp)
 lo=df.low.iloc[ei:end].to_numpy(float); hi=df.high.iloc[ei:end].to_numpy(float); si=first(lo<=stop); ti=first(hi>=target)
 v=f'FIXEDTP_{mode}_TP{int(tp*100)}_CAP{days}D'
 if si is not None and (ti is None or si<=ti): return rec(sig,v,'FIXED_TP',df,ei+si,'STOP_AMBIGUOUS' if ti==si else 'STOP',[(1.,stop)],cfg)
 if ti is not None: return rec(sig,v,'FIXED_TP',df,ei+ti,'TARGET',[(1.,target)],cfg)
 if ci is not None: return rec(sig,v,'FIXED_TP',df,ci,'TIME_CAP',[(1.,float(df.open.iloc[ci]))],cfg)

def runner_path(df,ei,sig,mode,gb,partial=None):
 e=float(sig.entry_price); stop=stop_px(sig,mode); peak=e; active=False; part_i=None; exits=[]; frac=1.; end=min(len(df),ei+365*24+1)
 for j in range(ei,end):
  b=df.iloc[j]; lo,hi,cl=float(b.low),float(b.high),float(b.close)
  if lo<=stop: return j,('RUNNER_STOP' if active else 'INITIAL_STOP'),exits+[(frac,stop)],part_i
  if partial is not None and part_i is None and hi>=e*(1+partial):
   exits.append((.5,e*(1+partial))); frac=.5; part_i=j; active=True; stop=max(stop,e); peak=max(peak,hi); continue
  if partial is None and not active and hi>=e*1.05:
   active=True; stop=max(stop,e); peak=max(peak,hi); continue
  peak=max(peak,hi)
  if hi>=e*11: return j,'TAKE_PROFIT_1000',exits+[(frac,e*11)],part_i
  if active and peak>=e*1.30:
   stop=max(stop,e*1.10); cand=e+(1-gb)*(peak-e); stop=max(stop,min(cand,cl*.999))
 return None,None,exits,part_i

def runner_outs(df,ei,sig,mode,gb,cfg,partial=None):
 times=df.time.astype('int64').to_numpy(); nj,reason,nexits,pi=runner_path(df,ei,sig,mode,gb,partial); fam='HALF_TP_RUNNER' if partial else 'FULL_TRAIL'; out=[]
 for days in RUN_CAPS:
  ci=cap_i(times,ei,days); part=f'PT{int(partial*100)}_' if partial else ''; v=f'{fam}_{mode}_{part}GB{int(gb*100)}_CAP{days}D'
  if nj is not None and (ci is None or nj<ci): out.append(rec(sig,v,fam,df,nj,reason,nexits,cfg)); continue
  if ci is None: continue
  if partial and pi is not None and pi<ci: exits=[(.5,float(sig.entry_price)*(1+partial)),(.5,float(df.open.iloc[ci]))]
  else: exits=[(1.,float(df.open.iloc[ci]))]
  out.append(rec(sig,v,fam,df,ci,'TIME_CAP',exits,cfg))
 return out

def variants_meta():
 rows=[]
 for s in STOPS:
  for d in CAPS:
   rows.append((f'TIME_{s}_CAP{d}D','TIME_ONLY',s,d,np.nan,np.nan,np.nan))
   for tp in TPS: rows.append((f'FIXEDTP_{s}_TP{int(tp*100)}_CAP{d}D','FIXED_TP',s,d,tp,np.nan,np.nan))
 for s in RUN_STOPS:
  for gb in GBS:
   for d in RUN_CAPS:
    rows.append((f'FULL_TRAIL_{s}_GB{int(gb*100)}_CAP{d}D','FULL_TRAIL',s,d,np.nan,gb,np.nan))
    for pt in PTS: rows.append((f'HALF_TP_RUNNER_{s}_PT{int(pt*100)}_GB{int(gb*100)}_CAP{d}D','HALF_TP_RUNNER',s,d,np.nan,gb,pt))
 return pd.DataFrame(rows,columns=['variant','family','stop_mode','cap_days','take_profit_pct','trail_giveback_pct','partial_take_profit_pct'])

def process(symbol,cfg,btc):
 df=load_symbol(symbol,cfg['interval'],cfg['start'],cfg['end'])
 if len(df)<900:return symbol,pd.DataFrame(),[],'insufficient_history'
 ff=feature_frame(df,btc); sigs=dedupe_episodes(symbol,ff,raw_candidates(ff),cfg)
 if sigs.empty:return symbol,sigs,[],None
 idx=pd.Series(df.index.to_numpy(),index=df.time).to_dict(); rows=[]
 for _,sig in sigs.iterrows():
  ei=idx.get(pd.Timestamp(sig.entry_time));
  if ei is None:continue
  for s in STOPS:
   for d in CAPS:
    o=time_out(df,ei,sig,s,d,cfg); rows += [o] if o else []
    for tp in TPS:
     o=tp_out(df,ei,sig,s,tp,d,cfg); rows += [o] if o else []
  for s in RUN_STOPS:
   for gb in GBS:
    rows += runner_outs(df,ei,sig,s,gb,cfg)
    for pt in PTS: rows += runner_outs(df,ei,sig,s,gb,cfg,pt)
 return symbol,sigs,rows,None

def summarise(trades,meta):
 def core(g):
  x=g.pnl_usdt.to_numpy(float)
  return pd.Series({'trades':len(g),'combined_pnl_usdt':x.sum(),'mean_pnl_usdt':x.mean(),'median_pnl_usdt':np.median(x),'profit_factor':pf(x),'win_rate':(x>0).mean(),'median_duration_hours':g.duration_hours.median(),'p10_pnl_usdt':np.quantile(x,.1),'p90_pnl_usdt':np.quantile(x,.9)})
 overall=trades.groupby('variant',sort=False).apply(core,include_groups=False).reset_index().merge(meta,on='variant',how='left')
 stage=trades.groupby(['variant','stage'],sort=False).apply(core,include_groups=False).reset_index().merge(meta,on='variant',how='left')
 year=trades.groupby(['variant','year'],sort=False).agg(trades=('pnl_usdt','size'),combined_pnl_usdt=('pnl_usdt','sum'),mean_pnl_usdt=('pnl_usdt','mean'),win_rate=('pnl_usdt',lambda x:(x>0).mean())).reset_index().merge(meta,on='variant',how='left')
 yagg=year.groupby('variant').agg(positive_years=('combined_pnl_usdt',lambda x:int((x>0).sum())),years_tested=('year','nunique'),worst_year_pnl_usdt=('combined_pnl_usdt','min'),best_year_pnl_usdt=('combined_pnl_usdt','max')).reset_index()
 overall=overall.merge(yagg,on='variant',how='left')
 reasons=trades.groupby(['variant','exit_reason']).size().rename('count').reset_index(); reasons['share']=reasons['count']/reasons.groupby('variant')['count'].transform('sum'); reasons=reasons.merge(meta,on='variant',how='left')
 return overall,stage,year,reasons

def analysis(overall,cohort):
 lines=['# Early Breakout Round 2 - Exit Architecture Research','', '**Status:** RESEARCH ONLY. Separate breakout-strategy track. No Early Breakout Watch or live-strategy edits.','',f'**Cohort:** {len(cohort):,} de-duplicated Round 1 entry episodes regenerated from the frozen Round 1 implementation.','', '## Families','', '- TIME_ONLY: 1-7d and 30d forced caps with 3/5/7/10% or structural-capped stop.','- FIXED_TP: same caps with full +10% or +20% profit-taking.','- FULL_TRAIL: +5% -> breakeven; +30% -> +10% floor plus 30/40/50/60% accumulated-gain giveback; 7/30/90/180/365d caps.','- HALF_TP_RUNNER: take 50% at +10% or +20%, runner to breakeven, then the same +30% trail; 7/30/90/180/365d caps.','', 'All results use independent 300 USDT trades with configured fees/slippage. Same-hour stop/target ambiguity is resolved conservatively in favour of the stop.','', '## Historical screening leaders','', 'These are candidates for a narrower confirmation round, not promoted strategy rules. Prefer plateaus across neighbouring parameters over a single isolated maximum.','']
 for fam,g in overall.groupby('family'):
  valid=g[g.trades>=max(500,int(g.trades.max()*.7))]; valid=valid if len(valid) else g
  top=valid.sort_values(['profit_factor','mean_pnl_usdt'],ascending=False).head(5)
  lines += [f'### {fam}','', '| Variant | Trades | PF | Mean P&L | Win rate | Positive years | Worst year |','|---|---:|---:|---:|---:|---:|---:|']
  for _,r in top.iterrows(): lines.append(f'| `{r.variant}` | {int(r.trades)} | {r.profit_factor:.3f} | {r.mean_pnl_usdt:.2f} | {100*r.win_rate:.1f}% | {int(r.positive_years)}/{int(r.years_tested)} | {r.worst_year_pnl_usdt:.2f} |')
  lines.append('')
 b='FULL_TRAIL_STRUCT_CAP10_GB60_CAP365D'
 if b in set(overall.variant):
  r=overall[overall.variant==b].iloc[0]; lines += ['## Round 3H-like reference','',f'`{b}`: PF {r.profit_factor:.3f}, mean P&L {r.mean_pnl_usdt:.2f} USDT, win rate {100*r.win_rate:.1f}%, positive in {int(r.positive_years)}/{int(r.years_tested)} tested years.','']
 lines += ['## Output files','', '- `variant_summary.csv` - all aggregate variants.','- `variant_stage_summary.csv` - PRE_BREAKOUT vs BREAKOUT_ATTEMPT.','- `variant_year_summary.csv` - year robustness.','- `family_leaders.csv` - top ten by PF and mean P&L per family.','- `exit_reason_summary.csv` - stop/target/time-cap mix.','- `selected_trade_results.csv` - representative trade-level variants.','- `cohort.csv` - regenerated entry cohort.','- `manifest.json` - exact grid and methodology.','', '## Caveats','', 'Independent overlapping trades are not a capital-constrained account simulation. The universe is the established research set plus named reference breakout coins, not a survivorship-free reconstruction of all Binance listings. Round 2 should narrow the search space for confirmation rather than directly become a live strategy.','']
 (OUT/'ANALYSIS.md').write_text('\n'.join(lines))

def main():
 cfg=json.loads(Path('research/config.json').read_text()); OUT.mkdir(parents=True,exist_ok=True); meta=variants_meta(); universe=list(dict.fromkeys(cfg['symbols']+REFERENCE_SYMBOLS)); universe=[s for s in universe if s not in TASK_EXCLUSIONS and s!='BTCUSDT']; btc=load_symbol('BTCUSDT',cfg['interval'],cfg['start'],cfg['end'])
 cohorts=[]; allrows=[]; fails=[]
 with ThreadPoolExecutor(max_workers=6) as ex:
  fut={ex.submit(process,s,cfg,btc):s for s in universe}
  for f in as_completed(fut):
   s=fut[f]
   try:
    _,sg,rows,err=f.result(); print(s,'signals',len(sg),'outcomes',len(rows),flush=True); cohorts += [sg] if len(sg) else []; allrows += rows; fails += [{'symbol':s,'error':err}] if err else []
   except Exception as e: fails.append({'symbol':s,'error':repr(e)}); print('ERROR',s,repr(e),flush=True)
 cohort=pd.concat(cohorts,ignore_index=True).sort_values(['entry_time','symbol']); cohort.to_csv(OUT/'cohort.csv',index=False); pd.DataFrame(fails).to_csv(OUT/'failures.csv',index=False)
 trades=pd.DataFrame(allrows,columns=['variant','family','signal_id','symbol','stage','year','duration_hours','exit_reason','pnl_usdt']);
 if trades.empty: raise RuntimeError('No Round 2 outcomes')
 overall,stage,year,reasons=summarise(trades,meta); overall=overall.sort_values(['family','profit_factor','mean_pnl_usdt'],ascending=[True,False,False]);
 overall.to_csv(OUT/'variant_summary.csv',index=False); stage.to_csv(OUT/'variant_stage_summary.csv',index=False); year.to_csv(OUT/'variant_year_summary.csv',index=False); reasons.to_csv(OUT/'exit_reason_summary.csv',index=False); trades[trades.variant.isin(SELECTED)].to_csv(OUT/'selected_trade_results.csv',index=False)
 leaders=[]
 for fam,g in overall.groupby('family'):
  valid=g[g.trades>=max(500,int(g.trades.max()*.7))]; valid=valid if len(valid) else g
  a=valid.sort_values(['profit_factor','mean_pnl_usdt'],ascending=False).head(10).copy(); a['leader_metric']='profit_factor'; a['leader_rank']=range(1,len(a)+1)
  b=valid.sort_values(['mean_pnl_usdt','profit_factor'],ascending=False).head(10).copy(); b['leader_metric']='mean_pnl'; b['leader_rank']=range(1,len(b)+1); leaders += [a,b]
 pd.concat(leaders,ignore_index=True).to_csv(OUT/'family_leaders.csv',index=False); analysis(overall,cohort)
 manifest={'study':'Early Breakout Round 2 - exit architecture','status':'RESEARCH ONLY - no live task/strategy edits','entry':'Frozen Round 1 research entry operationalisation','cohort_signals':int(len(cohort)),'dataset':{'start':cfg['start'],'end':cfg['end'],'interval':cfg['interval'],'universe':universe},'costs':{'position_usdt':POS,'fee_rate':cfg['fee_rate'],'slippage_rate':cfg['slippage_rate']},'time_caps_days':CAPS,'runner_caps_days':RUN_CAPS,'stops':STOPS,'runner_stops':RUN_STOPS,'fixed_take_profits':TPS,'trail_givebacks':GBS,'partial_targets':PTS,'ambiguity':'Initial stop wins same-hour ambiguity; BE/trail changes apply after the activation candle.','runner':'Full: +5% BE, +30% +10% floor and selected gain giveback. Half-runner: sell 50% at +10/+20, runner BE, same +30% trail. +1000% closes remaining.','comparison':'Independent 300 USDT trades. Natural exits are retained even when a later time cap is right-censored; unresolved trades without the required cap are excluded.'}; (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
 print('COHORT',len(cohort),'FAILURES',len(fails),'VARIANTS',len(overall)); print(overall.groupby('family').head(5)[['family','variant','trades','mean_pnl_usdt','profit_factor','win_rate','positive_years','years_tested']].to_string(index=False))
if __name__=='__main__': main()
