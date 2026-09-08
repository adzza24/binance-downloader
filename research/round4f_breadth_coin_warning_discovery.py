from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from binance_data import load_symbol
from round2b_validation import add_live_features, controlled_activity
from round4e_180d_warning_trail import daily_states, simulate

MA_WINDOWS = [50, 100, 150, 180, 200]
SLOPE_DAYS = 20
CONFIRM_DAYS = 3


def daily_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    x = df[['time','open','high','low','close']].copy()
    x['time'] = pd.to_datetime(x.time, utc=True)
    return x.set_index('time').resample('1D').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    d = daily_ohlc(df)
    out = pd.DataFrame(index=d.index)
    out['close'] = d['close'].astype(float)
    out['ret20'] = out.close.pct_change(20)
    out['ret60'] = out.close.pct_change(60)
    for w in MA_WINDOWS:
        ma = out.close.rolling(w, min_periods=max(30, int(w*0.8))).mean()
        out[f'ma{w}'] = ma
        out[f'below{w}'] = out.close < ma
        out[f'falling{w}'] = (ma / ma.shift(SLOPE_DAYS) - 1) < 0
        raw = out[f'below{w}'] & out[f'falling{w}']
        out[f'and{w}_c3'] = raw.rolling(CONFIRM_DAYS).sum().eq(CONFIRM_DAYS)
    return out.shift(1)


def first_true_after(s: pd.Series, t0, t1):
    q = s[(s.index >= t0) & (s.index <= t1)]
    idx = q[q.fillna(False).astype(bool)].index
    return idx[0] if len(idx) else pd.NaT


def main():
    cfg = json.loads(Path('research/config.json').read_text())
    out = Path('research/results/round4f'); out.mkdir(parents=True, exist_ok=True)
    btc = load_symbol('BTCUSDT', cfg['interval'], cfg['start'], cfg['end'])
    btc_feat = feature_frame(btc)
    warn180, bear200 = daily_states(btc)

    feats = {}
    raw = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(load_symbol, s, cfg['interval'], cfg['start'], cfg['end']): s for s in cfg['symbols']}
        for f in as_completed(futs):
            s = futs[f]
            try:
                df = btc.copy() if s == 'BTCUSDT' else f.result()
                if len(df) < 800: continue
                raw[s] = df
                feats[s] = feature_frame(df)
                print('loaded', s, len(df), flush=True)
            except Exception as e:
                print('ERROR load', s, repr(e), flush=True)

    # Common daily index; breadth denominator is assets with a valid MA that day.
    all_days = btc_feat.index
    breadth_rows = []
    alts = [s for s in feats if s != 'BTCUSDT']
    for day in all_days:
        r = {'day': day}
        for w in MA_WINDOWS:
            vals_below=[]; vals_fall=[]; vals_and=[]
            for s in alts:
                ff=feats[s]
                if day not in ff.index: continue
                if pd.notna(ff.at[day,f'ma{w}']):
                    vals_below.append(bool(ff.at[day,f'below{w}']))
                    vals_fall.append(bool(ff.at[day,f'falling{w}']))
                    vals_and.append(bool(ff.at[day,f'and{w}_c3']))
            r[f'n{w}']=len(vals_below)
            r[f'pct_below{w}']=100*np.mean(vals_below) if vals_below else np.nan
            r[f'pct_falling{w}']=100*np.mean(vals_fall) if vals_fall else np.nan
            r[f'pct_and{w}_c3']=100*np.mean(vals_and) if vals_and else np.nan
        # Relative weakness vs BTC over 20d / 60d.
        rw20=[]; rw60=[]
        if day in btc_feat.index:
            b20=btc_feat.at[day,'ret20']; b60=btc_feat.at[day,'ret60']
            for s in alts:
                ff=feats[s]
                if day not in ff.index: continue
                a20=ff.at[day,'ret20']; a60=ff.at[day,'ret60']
                if pd.notna(a20) and pd.notna(b20): rw20.append(a20 < b20)
                if pd.notna(a60) and pd.notna(b60): rw60.append(a60 < b60)
        r['pct_underperform_btc20']=100*np.mean(rw20) if rw20 else np.nan
        r['pct_underperform_btc60']=100*np.mean(rw60) if rw60 else np.nan
        breadth_rows.append(r)
    breadth = pd.DataFrame(breadth_rows).set_index('day')
    breadth.to_csv(out/'breadth_daily.csv')

    # BTC 200 bear starts and breadth lead diagnostics.
    bear_starts = bear200 & ~bear200.shift(1).fillna(False)
    events=[]
    thresholds=[30,40,50,60,70]
    breadth_cols=[]
    for w in [50,100,150,180,200]: breadth_cols += [f'pct_below{w}',f'pct_falling{w}',f'pct_and{w}_c3']
    breadth_cols += ['pct_underperform_btc20','pct_underperform_btc60']
    for t in bear_starts[bear_starts].index:
        e={'bear200_start':t}
        look0=t-pd.Timedelta(days=120)
        for col in breadth_cols:
            for th in thresholds:
                q=breadth.loc[(breadth.index>=look0)&(breadth.index<t),col].dropna()
                hits=q[q>=th]
                key=f'{col}_th{th}_lead_days'
                e[key]=float((t-hits.index[0]).days) if len(hits) else np.nan
        events.append(e)
    pd.DataFrame(events).to_csv(out/'bear_transition_breadth_leads.csv',index=False)

    # Coin-specific state around every Round 4E warning-exposed CONTROL trade.
    trade_rows=[]
    for s,df in raw.items():
        x=add_live_features(df,btc); sigs=controlled_activity(s,x,cfg)
        ff=feats[s]
        for sig in sigs:
            base=simulate(df,sig,'CONTROL_STAGE_AWARE',warn180,bear200,cfg,None)
            if base.get('exit_reason')=='ENTRY_GATE' or not base.get('warn_seen'): continue
            et=pd.Timestamp(sig['entry_time']); xt=pd.Timestamp(base['exit_time'])
            live=df[(pd.to_datetime(df.time,utc=True)>=et)&(pd.to_datetime(df.time,utc=True)<=xt)]
            if live.empty: continue
            peak_i=int(np.argmax(live.high.astype(float).to_numpy()))
            peak_t=pd.Timestamp(live.iloc[peak_i].time).floor('D')
            entry_day=et.floor('D')
            exit_day=xt.floor('D')
            wr=warn180[(warn180.index>=entry_day)&(warn180.index<=exit_day)]
            widx=wr[wr].index
            first_btc_warn=widx[0] if len(widx) else pd.NaT
            r={'signal_id':sig['signal_id'],'symbol':s,'entry_time':et,'exit_time':xt,'peak_time':peak_t,
               'first_btc180_warning':first_btc_warn,'normal_pnl_usdt':float(base['pnl_usdt'])}
            for w in [100,150,180,200]:
                st=first_true_after(ff[f'and{w}_c3'].fillna(False),entry_day,exit_day)
                r[f'coin_and{w}_first']=st
                r[f'coin_and{w}_minus_peak_days']=float((st-peak_t).days) if pd.notna(st) else np.nan
            trade_rows.append(r)
    trades=pd.DataFrame(trade_rows)
    trades.to_csv(out/'warning_trade_coin_states.csv',index=False)

    # Aggregate coin-warning usefulness.
    agg=[]
    for w in [100,150,180,200]:
        c=f'coin_and{w}_minus_peak_days'
        if c not in trades: continue
        q=trades[c].dropna()
        agg.append({'coin_ma':w,'trades_with_signal':int(len(q)),
                    'signal_before_or_at_peak':int((q<=0).sum()),'signal_after_peak':int((q>0).sum()),
                    'median_days_vs_peak':float(q.median()) if len(q) else np.nan,
                    'late_gt7d':int((q>7).sum()),'early_gt30d':int((q<-30).sum())})
    pd.DataFrame(agg).to_csv(out/'coin_warning_summary.csv',index=False)

    manifest={
      'study':'Round 4F breadth + coin-specific deterioration discovery',
      'status':'RESEARCH ONLY - no live strategy changes',
      'purpose':'Discover whether alt breadth or coin-specific long-term trend deterioration gives useful causal warning before frozen BTC MA200 bear confirmation.',
      'breadth_universe':'Configured Binance spot research symbols excluding BTC; denominator varies by asset history and MA availability.',
      'features':'Daily causal shifted states for MA50/100/150/180/200: below, falling over 20d, AND with 3-day confirmation; plus alt underperformance vs BTC over 20d/60d.',
      'bear_reference':'Frozen BTC MA200 AND 20d falling condition from Round 4D/4E.',
      'coin_trade_sample':'Round 4E CONTROL_STAGE_AWARE trades exposed to BTC 180D warning.',
      'warning':'Discovery only. Thresholds are diagnostics, not proposed live rules.'
    }
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))

    print('\nBEAR EVENTS', len(events))
    print('\nCOIN SUMMARY\n', pd.DataFrame(agg).to_string(index=False))
    print('\nBREADTH SNAPSHOT\n', breadth.tail().to_string())

if __name__=='__main__': main()
