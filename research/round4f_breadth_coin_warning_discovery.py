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
THRESHOLDS = [30, 40, 50, 60, 70]


def daily_ohlc(df):
    x = df[['time','open','high','low','close']].copy()
    x['time'] = pd.to_datetime(x.time, utc=True)
    return x.set_index('time').resample('1D').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()


def feature_frame(df):
    d = daily_ohlc(df)
    out = pd.DataFrame(index=d.index)
    out['close'] = d.close.astype(float)
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


def first_true(s, t0, t1):
    q = s[(s.index >= t0) & (s.index <= t1)].fillna(False).astype(bool)
    idx = q[q].index
    return idx[0] if len(idx) else pd.NaT


def latest_upcross_lead(series, threshold, event_time, lookback_days=120):
    q = series[(series.index >= event_time-pd.Timedelta(days=lookback_days)) & (series.index < event_time)].dropna()
    if q.empty: return np.nan, False
    cond = q >= threshold
    prev = cond.shift(1).fillna(False).astype(bool)
    crosses = q.index[cond & ~prev]
    if len(crosses):
        return float((event_time-crosses[-1]).days), False
    if bool(cond.iloc[-1]):
        return float(lookback_days), True
    return np.nan, False


def main():
    cfg = json.loads(Path('research/config.json').read_text())
    out = Path('research/results/round4f'); out.mkdir(parents=True, exist_ok=True)
    btc = load_symbol('BTCUSDT', cfg['interval'], cfg['start'], cfg['end'])
    btc_feat = feature_frame(btc)
    warn180, bear200 = daily_states(btc)

    feats, raw = {}, {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(load_symbol, s, cfg['interval'], cfg['start'], cfg['end']): s for s in cfg['symbols']}
        for f in as_completed(futs):
            s = futs[f]
            try:
                df = btc.copy() if s == 'BTCUSDT' else f.result()
                if len(df) < 800: continue
                raw[s] = df; feats[s] = feature_frame(df)
                print('loaded', s, len(df), flush=True)
            except Exception as e:
                print('ERROR load', s, repr(e), flush=True)

    alts = [s for s in feats if s != 'BTCUSDT']
    rows = []
    for day in btc_feat.index:
        r = {'day': day}
        for w in MA_WINDOWS:
            vb, vf, va = [], [], []
            for s in alts:
                ff = feats[s]
                if day not in ff.index or pd.isna(ff.at[day,f'ma{w}']): continue
                vb.append(bool(ff.at[day,f'below{w}']))
                vf.append(bool(ff.at[day,f'falling{w}']))
                va.append(bool(ff.at[day,f'and{w}_c3']))
            r[f'n{w}'] = len(vb)
            r[f'pct_below{w}'] = 100*np.mean(vb) if vb else np.nan
            r[f'pct_falling{w}'] = 100*np.mean(vf) if vf else np.nan
            r[f'pct_and{w}_c3'] = 100*np.mean(va) if va else np.nan
        rw20, rw60 = [], []
        b20, b60 = btc_feat.at[day,'ret20'], btc_feat.at[day,'ret60']
        for s in alts:
            ff = feats[s]
            if day not in ff.index: continue
            a20, a60 = ff.at[day,'ret20'], ff.at[day,'ret60']
            if pd.notna(a20) and pd.notna(b20): rw20.append(a20 < b20)
            if pd.notna(a60) and pd.notna(b60): rw60.append(a60 < b60)
        r['pct_underperform_btc20'] = 100*np.mean(rw20) if rw20 else np.nan
        r['pct_underperform_btc60'] = 100*np.mean(rw60) if rw60 else np.nan
        rows.append(r)
    breadth = pd.DataFrame(rows).set_index('day')
    breadth.to_csv(out/'breadth_daily.csv')

    # Correct causal starts: cast shifted series back to bool before bitwise inversion.
    b = bear200.fillna(False).astype(bool)
    prev_b = b.shift(1).fillna(False).astype(bool)
    bear_starts = b & ~prev_b
    breadth_cols = []
    for w in MA_WINDOWS:
        breadth_cols += [f'pct_below{w}', f'pct_falling{w}', f'pct_and{w}_c3']
    breadth_cols += ['pct_underperform_btc20','pct_underperform_btc60']

    events = []
    for t in bear_starts[bear_starts].index:
        e = {'bear200_start': t}
        for col in breadth_cols:
            for th in THRESHOLDS:
                lead, censored = latest_upcross_lead(breadth[col], th, t)
                e[f'{col}_th{th}_lead_days'] = lead
                e[f'{col}_th{th}_censored120'] = censored
        events.append(e)
    event_df = pd.DataFrame(events)
    event_df.to_csv(out/'bear_transition_breadth_leads.csv', index=False)

    # Aggregate threshold discovery: prefer high event coverage and moderate, repeatable lead rather than maximum lead.
    summ = []
    n_events = len(event_df)
    for col in breadth_cols:
        for th in THRESHOLDS:
            lc = f'{col}_th{th}_lead_days'; cc = f'{col}_th{th}_censored120'
            q = event_df[lc].dropna() if n_events else pd.Series(dtype=float)
            unc = event_df.loc[event_df[lc].notna() & ~event_df[cc].astype(bool), lc] if n_events else pd.Series(dtype=float)
            summ.append({'breadth_metric':col,'threshold_pct':th,'bear_events':n_events,
                         'events_warned':int(len(q)),'coverage':float(len(q)/n_events) if n_events else np.nan,
                         'median_lead_all_days':float(q.median()) if len(q) else np.nan,
                         'median_lead_uncensored_days':float(unc.median()) if len(unc) else np.nan,
                         'uncensored_events':int(len(unc)),'censored_120d_events':int(event_df[cc].sum()) if n_events else 0})
    pd.DataFrame(summ).to_csv(out/'breadth_threshold_summary.csv', index=False)

    # Coin-specific trend state for the exact Round 4E warning-exposed control trades.
    trade_rows = []
    for s, df in raw.items():
        x = add_live_features(df, btc); sigs = controlled_activity(s, x, cfg); ff = feats[s]
        for sig in sigs:
            base = simulate(df, sig, 'CONTROL_STAGE_AWARE', warn180, bear200, cfg, None)
            if base.get('exit_reason') == 'ENTRY_GATE' or not base.get('warn_seen'): continue
            et, xt = pd.Timestamp(sig['entry_time']), pd.Timestamp(base['exit_time'])
            live = df[(pd.to_datetime(df.time,utc=True)>=et) & (pd.to_datetime(df.time,utc=True)<=xt)]
            if live.empty: continue
            peak_t = pd.Timestamp(live.iloc[int(np.argmax(live.high.astype(float).to_numpy()))].time).floor('D')
            entry_day, exit_day = et.floor('D'), xt.floor('D')
            wr = warn180[(warn180.index>=entry_day)&(warn180.index<=exit_day)]
            widx = wr[wr].index
            r = {'signal_id':sig['signal_id'],'symbol':s,'entry_time':et,'exit_time':xt,'peak_time':peak_t,
                 'first_btc180_warning':widx[0] if len(widx) else pd.NaT,'normal_pnl_usdt':float(base['pnl_usdt'])}
            for w in [100,150,180,200]:
                st = first_true(ff[f'and{w}_c3'], entry_day, exit_day)
                r[f'coin_and{w}_first'] = st
                r[f'coin_and{w}_minus_peak_days'] = float((st-peak_t).days) if pd.notna(st) else np.nan
            trade_rows.append(r)
    trades = pd.DataFrame(trade_rows)
    trades.to_csv(out/'warning_trade_coin_states.csv', index=False)

    agg = []
    for w in [100,150,180,200]:
        c = f'coin_and{w}_minus_peak_days'; q = trades[c].dropna()
        agg.append({'coin_ma':w,'trades_with_signal':int(len(q)),
                    'signal_before_or_at_peak':int((q<=0).sum()),'signal_after_peak':int((q>0).sum()),
                    'median_days_vs_peak':float(q.median()) if len(q) else np.nan,
                    'within_7d_before_peak':int(((q>=-7)&(q<=0)).sum()),
                    'late_gt7d':int((q>7).sum()),'early_gt30d':int((q<-30).sum())})
    pd.DataFrame(agg).to_csv(out/'coin_warning_summary.csv', index=False)

    manifest = {
      'study':'Round 4F breadth + coin-specific deterioration discovery',
      'status':'RESEARCH ONLY - no live strategy changes',
      'purpose':'Discover whether alt breadth or coin-specific trend deterioration gives useful causal warning before frozen BTC MA200 bear confirmation.',
      'breadth_universe':'Configured Binance spot research symbols excluding BTC; denominator varies with asset history and MA availability.',
      'features':'Causal prior-day MA50/100/150/180/200 below, falling-20d and 3-day-confirmed AND states; alt underperformance vs BTC over 20d/60d.',
      'bear_reference':'Frozen BTC MA200 AND 20d falling condition.',
      'breadth_lead':'Most recent upward threshold crossing within 120 days before each distinct BTC bear-regime start; persistent conditions already active at window start are marked censored.',
      'coin_sample':'Exact Round 4E CONTROL_STAGE_AWARE trades exposed to BTC 180D warning.',
      'warning':'Discovery only. No thresholds or coin MA rules are promoted to live strategy.'
    }
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('\nBEAR EVENTS', len(event_df))
    print('\nCOIN SUMMARY\n', pd.DataFrame(agg).to_string(index=False))
    ss = pd.DataFrame(summ).sort_values(['coverage','median_lead_uncensored_days'], ascending=[False,True])
    print('\nBREADTH TOP COVERAGE\n', ss.head(30).to_string(index=False))

if __name__ == '__main__':
    main()
