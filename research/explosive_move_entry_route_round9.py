from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import explosive_move_discovery_round1 as disc
import explosive_move_real_world_replay_round8 as r8

SRC = Path('research/results/explosive_move_real_world_replay_round8')
OUT = Path('research/results/explosive_move_entry_route_round9')
HORIZON_H = 720
ROUND_TRIP_COST = 0.003


def first_touch_oco(high, low, start, end, entry, target=0.10, stop=0.05):
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


def limit_then_oco(open_, high, low, close, sig_i, end_i, signal_entry):
    limit = signal_entry * 0.95
    fill_i = None
    for k in range(sig_i + 1, end_i + 1):
        if low[k] <= limit:
            fill_i = k
            break
    if fill_i is None:
        return {
            'limit_filled': False, 'limit_fill_hours': np.nan, 'limit_fill_price': limit,
            'limit_result': 'UNFILLED', 'limit_exit_hours_from_signal': np.nan,
            'limit_gross_return': 0.0, 'limit_net_return_per_signal': 0.0,
            'limit_net_return_per_filled_trade': np.nan, 'fill_candle_target_ambiguous': False,
        }

    target_px = limit * 1.10
    stop_px = limit * 0.95
    amb = bool(high[fill_i] >= target_px and open_[fill_i] > limit)

    # Conservative fill-candle sequencing: a stop can occur after the downward limit cross;
    # a target on the same candle is not credited when the candle opened above the limit,
    # because its high may have occurred before the fill.
    if low[fill_i] <= stop_px:
        result, exit_i = 'STOP', fill_i
    elif open_[fill_i] <= limit and high[fill_i] >= target_px:
        result, exit_i = 'TARGET', fill_i
    else:
        result, exit_i = first_touch_oco(high, low, fill_i + 1, end_i, limit, 0.10, 0.05)

    if result == 'TARGET':
        gross = 0.10
    elif result == 'STOP':
        gross = -0.05
    else:
        gross = close[end_i] / limit - 1
    net = gross - ROUND_TRIP_COST
    return {
        'limit_filled': True,
        'limit_fill_hours': fill_i - sig_i,
        'limit_fill_price': limit,
        'limit_result': result,
        'limit_exit_hours_from_signal': (exit_i - sig_i) if exit_i is not None else end_i - sig_i,
        'limit_gross_return': float(gross),
        'limit_net_return_per_signal': float(net),
        'limit_net_return_per_filled_trade': float(net),
        'fill_candle_target_ambiguous': amb,
    }


def exact_signal_path(high, low, sig_i, end_i, entry):
    up_i = down_i = None
    for k in range(sig_i + 1, end_i + 1):
        if down_i is None and low[k] <= entry * 0.95:
            down_i = k
        if up_i is None and high[k] >= entry * 1.10:
            up_i = k
        if up_i is not None and down_i is not None:
            break
    if up_i is not None and (down_i is None or up_i < down_i):
        group = 'DIRECT_PLUS10'
    elif up_i is not None and down_i is not None and down_i <= up_i:
        group = 'DIP_THEN_PLUS10'
    else:
        group = 'NO_PLUS10_30D'
    return {
        'path_group': group,
        'exact_hours_to_signal_plus10': np.nan if up_i is None else up_i - sig_i,
        'exact_hours_to_signal_minus5': np.nan if down_i is None else down_i - sig_i,
    }


def signal_features(ff, sig_i):
    row = ff.iloc[sig_i]
    out = {}
    for f in disc.FEATURES:
        v = row.get(f, np.nan)
        out[f'sig_{f}'] = float(v) if pd.notna(v) else np.nan
    return out


def process_symbol(symbol, sg, btc, end):
    df = r8.load_extended(symbol, end)
    if df.empty:
        return [], [{'symbol': symbol, 'error': 'no_data'}]
    df['time'] = pd.to_datetime(df['time'], utc=True)
    time_to_i = {pd.Timestamp(t): i for i, t in enumerate(df['time'])}
    ff = disc.build_features(df, btc)
    ff['time'] = pd.to_datetime(ff['time'], utc=True)
    # build_features preserves candle alignment in this research stack.
    open_ = df['open'].to_numpy(float); high = df['high'].to_numpy(float)
    low = df['low'].to_numpy(float); close = df['close'].to_numpy(float)
    rows = []; failures = []
    for s in sg.itertuples(index=False):
        ts = pd.Timestamp(s.signal_time)
        i = time_to_i.get(ts)
        if i is None:
            failures.append({'symbol': symbol, 'signal_time': str(ts), 'error': 'signal_time_not_found'})
            continue
        avail = len(df) - i - 1
        mature = avail >= HORIZON_H
        end_i = min(len(df) - 1, i + HORIZON_H)
        entry = close[i]

        straight_result, straight_exit = first_touch_oco(high, low, i + 1, end_i, entry, 0.10, 0.05)
        if straight_result == 'TARGET': gross = 0.10
        elif straight_result == 'STOP': gross = -0.05
        else: gross = close[end_i] / entry - 1
        straight_net = gross - ROUND_TRIP_COST if mature else np.nan

        lr = limit_then_oco(open_, high, low, close, i, end_i, entry)
        if not mature:
            lr['limit_net_return_per_signal'] = np.nan
            lr['limit_net_return_per_filled_trade'] = np.nan

        path = exact_signal_path(high, low, i, end_i, entry)
        r = {c: getattr(s, c) for c in sg.columns}
        r.update({
            'mature_30d_recalc': bool(mature),
            'straight_result_recalc': straight_result,
            'straight_exit_hours_from_signal': np.nan if straight_exit is None else straight_exit - i,
            'straight_gross_return': float(gross),
            'straight_net_return': float(straight_net) if np.isfinite(straight_net) else np.nan,
        })
        r.update(lr); r.update(path); r.update(signal_features(ff, i))
        rows.append(r)
    return rows, failures


def add_cross_sectional_features(z):
    z = z.sort_values('signal_time').copy()
    times = pd.to_datetime(z['signal_time'], utc=True)
    idx = pd.Series(1, index=times).sort_index()
    c6 = idx.rolling('6h').sum(); c24 = idx.rolling('24h').sum()
    # Duplicate timestamps need map through grouped counts instead of direct reindex.
    h = z.groupby('signal_time').size().sort_index()
    h6 = h.rolling('6h').sum(); h24 = h.rolling('24h').sum()
    z['signals_same_hour'] = z['signal_time'].map(h).astype(float)
    z['signals_prev6h'] = z['signal_time'].map(h6).astype(float)
    z['signals_prev24h'] = z['signal_time'].map(h24).astype(float)
    return z


def strategy_summary(z):
    rows = []
    z = z.copy()
    z['year'] = pd.to_datetime(z['signal_time'], utc=True).dt.year
    for tier in ['ORIGINAL_TOP100','EXTERNAL_101_200','ALL_TOP200']:
        base = z if tier == 'ALL_TOP200' else z[z['replay_tier'] == tier]
        for year in ['ALL', 2024, 2025, 2026]:
            g = base if year == 'ALL' else base[base['year'] == year]
            m = g[g['mature_30d_recalc']]
            if m.empty: continue
            fill = m[m['limit_filled']]
            rows.append({
                'tier': tier, 'year': year, 'signals': len(m),
                'straight_target_rate': (m['straight_result_recalc']=='TARGET').mean(),
                'straight_stop_rate': (m['straight_result_recalc']=='STOP').mean(),
                'straight_avg_net_per_signal': m['straight_net_return'].mean(),
                'limit_fill_rate': m['limit_filled'].mean(),
                'limit_target_rate_per_filled': (fill['limit_result']=='TARGET').mean() if len(fill) else np.nan,
                'limit_stop_rate_per_filled': (fill['limit_result']=='STOP').mean() if len(fill) else np.nan,
                'limit_avg_net_per_filled': fill['limit_net_return_per_filled_trade'].mean() if len(fill) else np.nan,
                'limit_avg_net_per_original_signal': m['limit_net_return_per_signal'].fillna(0).mean(),
                'median_limit_fill_hours': fill['limit_fill_hours'].median() if len(fill) else np.nan,
                'ambiguous_fill_target_candles': int(fill['fill_candle_target_ambiguous'].sum()) if len(fill) else 0,
                'direct_plus10': int((m['path_group']=='DIRECT_PLUS10').sum()),
                'dip_then_plus10': int((m['path_group']=='DIP_THEN_PLUS10').sum()),
                'no_plus10': int((m['path_group']=='NO_PLUS10_30D').sum()),
            })
    return pd.DataFrame(rows)


def feature_comparison(z):
    m = z[z['mature_30d_recalc'] & z['path_group'].isin(['DIRECT_PLUS10','DIP_THEN_PLUS10'])].copy()
    m['direct'] = (m['path_group']=='DIRECT_PLUS10').astype(int)
    exclude = {'current_liquidity_rank','precursor_alerts','signal_checkpoint_h','entry_price','forward_hours_available',
               'mature_30d','mature_30d_recalc','straight_exit_hours_from_signal','straight_gross_return','straight_net_return',
               'limit_fill_hours','limit_fill_price','limit_exit_hours_from_signal','limit_gross_return','limit_net_return_per_signal',
               'limit_net_return_per_filled_trade','exact_hours_to_signal_plus10','exact_hours_to_signal_minus5'}
    feats = [c for c in m.columns if (c.startswith('sig_') or c in ['candidate_open_to_signal_close','candidate_mfe_to_signal','candidate_mae_to_signal','precursor_alerts','signal_checkpoint_h','current_liquidity_rank','signals_same_hour','signals_prev6h','signals_prev24h'])]
    rows=[]
    for f in feats:
        x = pd.to_numeric(m[f], errors='coerce').to_numpy(float); y=m['direct'].to_numpy(int)
        ok=np.isfinite(x)
        if ok.sum()<50 or len(np.unique(y[ok]))<2: continue
        try: auc=roc_auc_score(y[ok], x[ok])
        except ValueError: continue
        d=x[ok & (y==1)]; q=x[ok & (y==0)]
        direction='HIGHER_DIRECT' if np.nanmedian(d)>=np.nanmedian(q) else 'LOWER_DIRECT'
        sep=max(auc,1-auc)
        yr_dirs=[]; yr_aucs=[]
        years=pd.to_datetime(m['signal_time'],utc=True).dt.year.to_numpy()
        for yr in [2024,2025,2026]:
            oy=ok & (years==yr)
            if oy.sum()<20 or len(np.unique(y[oy]))<2: continue
            a=roc_auc_score(y[oy],x[oy]); dd=x[oy & (y==1)]; qq=x[oy & (y==0)]
            yr_dirs.append('HIGHER_DIRECT' if np.nanmedian(dd)>=np.nanmedian(qq) else 'LOWER_DIRECT')
            yr_aucs.append(max(a,1-a))
        rows.append({'feature':f,'direct_n':len(d),'dip_then_plus10_n':len(q),
                     'direct_median':np.nanmedian(d),'dip_median':np.nanmedian(q),
                     'direction':direction,'separation_auc':sep,
                     'same_direction_all_available_years': len(yr_dirs)>1 and len(set(yr_dirs))==1,
                     'min_year_auc': min(yr_aucs) if yr_aucs else np.nan})
    return pd.DataFrame(rows).sort_values(['same_direction_all_available_years','min_year_auc','separation_auc'],ascending=[False,False,False])


def categorical_breakdown(z):
    m=z[z['mature_30d_recalc']].copy(); rows=[]
    for col in ['candidate_family','signal_checkpoint_h','replay_tier']:
        for val,g in m.groupby(col):
            rows.append({'dimension':col,'value':val,'n':len(g),
                         'direct_plus10_rate':(g.path_group=='DIRECT_PLUS10').mean(),
                         'dip_then_plus10_rate':(g.path_group=='DIP_THEN_PLUS10').mean(),
                         'limit_fill_rate':g.limit_filled.mean(),
                         'straight_target_rate':(g.straight_result_recalc=='TARGET').mean(),
                         'limit_target_rate_per_filled':(g.loc[g.limit_filled,'limit_result']=='TARGET').mean() if g.limit_filled.any() else np.nan})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sig=pd.read_csv(SRC/'signals.csv.gz',compression='gzip')
    for c in ['signal_time','candidate_start_time']: sig[c]=pd.to_datetime(sig[c],utc=True)
    end=pd.Timestamp(sig['signal_time'].max())+pd.Timedelta(hours=HORIZON_H+2)
    now=pd.Timestamp.now(tz='UTC').floor('h')-pd.Timedelta(hours=1)
    end=min(end,now)
    btc=r8.load_extended('BTCUSDT',end)
    rows=[]; failures=[]
    for symbol,g in sig.groupby('symbol',sort=False):
        try:
            rr,ff=process_symbol(symbol,g,btc,end); rows.extend(rr); failures.extend(ff)
            print(symbol,'done',len(rr),flush=True)
        except Exception as e:
            failures.append({'symbol':symbol,'error':repr(e)}); print('ERROR',symbol,repr(e),flush=True)
    z=pd.DataFrame(rows)
    z=add_cross_sectional_features(z)
    z.to_csv(OUT/'signal_strategy_paths.csv.gz',index=False,compression='gzip')
    pd.DataFrame(failures).to_csv(OUT/'failures.csv',index=False)
    ss=strategy_summary(z); ss.to_csv(OUT/'strategy_summary.csv',index=False)
    fc=feature_comparison(z); fc.to_csv(OUT/'direct_vs_dip_features.csv',index=False)
    cb=categorical_breakdown(z); cb.to_csv(OUT/'path_group_breakdown.csv',index=False)

    allrow=ss[(ss.tier=='ALL_TOP200') & (ss.year.astype(str)=='ALL')].iloc[0]
    lines=['# Explosive Move Entry Route Round 9','',
           '**RESEARCH ONLY. No live strategy or automation changes.**','',
           '## Design','',
           '- Same Round 8 full-population signals. Primary horizon is 30 days from the original signal for both routes.',
           '- Route A: buy at signal close, OCO +10% / -5%.',
           '- Route B: place buy limit 5% below signal; if filled, OCO +10% / -5% from fill.',
           '- 0.30% round-trip costs on executed trades. Unfilled limits have zero return.',
           '- Conservative 1h OHLC ordering: stop wins same-candle ties. A target on the limit-fill candle is not credited when the candle opened above the limit, because the high could precede the fill.',
           '- Signal-time features are stored for DIRECT_PLUS10 vs DIP_THEN_PLUS10 analysis; no future features are used as predictors.','',
           '## Overall mature-signal result','',
           f"- Mature signals: **{int(allrow.signals)}**.",
           f"- Straight OCO target rate: **{100*allrow.straight_target_rate:.1f}%**; average net per signal **{100*allrow.straight_avg_net_per_signal:.2f}%**.",
           f"- -5% limit fill rate: **{100*allrow.limit_fill_rate:.1f}%**.",
           f"- After a fill, limit-route OCO target rate: **{100*allrow.limit_target_rate_per_filled:.1f}%**; average net per filled trade **{100*allrow.limit_avg_net_per_filled:.2f}%**.",
           f"- Limit-route average net per original signal including unfilled orders: **{100*allrow.limit_avg_net_per_original_signal:.2f}%**.",
           f"- Median wait for limit fill: **{allrow.median_limit_fill_hours:.1f}h**.",
           f"- Path groups: DIRECT_PLUS10 **{int(allrow.direct_plus10)}**, DIP_THEN_PLUS10 **{int(allrow.dip_then_plus10)}**, NO_PLUS10_30D **{int(allrow.no_plus10)}**.",'',
           '## Commonality dataset','',
           f'- Numeric signal-time comparison features saved: **{len(fc)}**.',
           '- `direct_vs_dip_features.csv` ranks contemporaneous differences and checks whether direction is consistent across available years.',
           '- `signal_strategy_paths.csv.gz` retains per-signal outcomes, fill timing, path class, candidate context, detector features and market signal breadth for later split-rule research.',
           '- These comparisons are descriptive; 2024-2026 has already influenced earlier research and is not a pristine untouched holdout.']
    (OUT/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    (OUT/'manifest.json').write_text(json.dumps({'study':'Round 9 entry route','branch':'round-3','horizon_hours':HORIZON_H,'cost':ROUND_TRIP_COST,'source':'Round 8 full-population signals','live_changes':False},indent=2)+'\n')
    print((OUT/'ANALYSIS.md').read_text(),flush=True)
    print('\nTOP FEATURES\n',fc.head(20).to_string(index=False),flush=True)

if __name__=='__main__': main()
