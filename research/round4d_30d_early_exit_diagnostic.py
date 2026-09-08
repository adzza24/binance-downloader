from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from binance_data import load_symbol
from round2b_validation import add_live_features, controlled_activity
from round4d_open_position_bear_response import build_bear_state, simulate
from round3a_exit_architecture import net_pnl

VARIANT = 'SELL_30D_BEFORE_CONFIRMATION_HINDSIGHT'


def bear_starts(bear_daily: pd.Series):
    return list(bear_daily[(bear_daily) & (~bear_daily.shift(1, fill_value=False))].index)


def force_30d_early(df, sig, normal, starts, cfg):
    entry_time = pd.Timestamp(sig['entry_time']).tz_convert('UTC')
    normal_exit = pd.Timestamp(normal['exit_time']).tz_convert('UTC')
    candidates = []
    for s in starts:
        t = pd.Timestamp(s).tz_convert('UTC') - pd.Timedelta(days=30)
        if entry_time < t < normal_exit:
            candidates.append((t, s))
    if not candidates:
        return None
    t, confirm = min(candidates, key=lambda z: z[0])
    q = df[pd.to_datetime(df.time, utc=True) >= t]
    if q.empty:
        return None
    b = q.iloc[0]
    px = float(b.open)
    pnl = net_pnl(float(sig['entry_price']), [(1.0, px)], cfg)
    return {
        'signal_id': sig['signal_id'], 'symbol': sig['symbol'], 'entry_time': sig['entry_time'],
        'normal_exit_time': normal['exit_time'], 'normal_pnl': float(normal['pnl_usdt']),
        'early_exit_time': b.time, 'early_exit_price': px, 'future_bear_confirm_time': confirm,
        'early_exit_pnl': float(pnl), 'delta_vs_normal': float(pnl-normal['pnl_usdt'])
    }


def process_symbol(symbol, btc, bear_daily, starts, cfg):
    df = btc.copy() if symbol == 'BTCUSDT' else load_symbol(symbol, cfg['interval'], cfg['start'], cfg['end'])
    if len(df) < 800: return []
    x = add_live_features(df, btc); sigs = controlled_activity(symbol, x, cfg)
    rows=[]
    for sig in sigs:
        normal = simulate(df, sig, 'ENTRY_GATE_ONLY', bear_daily, cfg, None)
        if normal['exit_reason']=='ENTRY_GATE':
            continue
        r = force_30d_early(df, sig, normal, starts, cfg)
        if r: rows.append(r)
    return rows


def main():
    cfg=json.loads(Path('research/config.json').read_text())
    out=Path('research/results/round4d_30d'); out.mkdir(parents=True, exist_ok=True)
    btc=load_symbol('BTCUSDT',cfg['interval'],cfg['start'],cfg['end'])
    bear_daily=build_bear_state(btc); starts=bear_starts(bear_daily)
    all_rows=[]
    with ThreadPoolExecutor(max_workers=6) as ex:
        fut={ex.submit(process_symbol,s,btc,bear_daily,starts,cfg):s for s in cfg['symbols']}
        for f in as_completed(fut):
            s=fut[f]
            try:
                rows=f.result(); all_rows += rows; print(s,len(rows),flush=True)
            except Exception as e: print('ERROR',s,repr(e),flush=True)
    q=pd.DataFrame(all_rows)
    if q.empty:
        print('NO ROWS'); return
    q['entry_year']=pd.to_datetime(q.entry_time,utc=True).dt.year
    q['exit_year']=pd.to_datetime(q.early_exit_time,utc=True).dt.year
    q.to_csv(out/'affected_positions.csv',index=False)
    summary=pd.DataFrame([{
        'variant':VARIANT,
        'affected_positions':len(q),
        'normal_pnl_on_affected':q.normal_pnl.sum(),
        'early_exit_pnl':q.early_exit_pnl.sum(),
        'delta_vs_normal':q.delta_vs_normal.sum(),
        'improved_positions':int((q.delta_vs_normal>0).sum()),
        'worsened_positions':int((q.delta_vs_normal<0).sum()),
        'median_delta':q.delta_vs_normal.median(),
    }])
    summary.to_csv(out/'summary.csv',index=False)
    y=q.groupby('exit_year').agg(affected_positions=('signal_id','size'),normal_pnl=('normal_pnl','sum'),early_exit_pnl=('early_exit_pnl','sum'),delta_vs_normal=('delta_vs_normal','sum')).reset_index()
    y.to_csv(out/'year_summary.csv',index=False)
    manifest={
        'study':'30-day pre-confirmation hindsight diagnostic',
        'warning':'NOT CAUSAL. Exit date is defined relative to a bear confirmation 30 days in the future and therefore cannot be traded directly. Purpose is only to test whether the confirmed regime tends to arrive after meaningful gains have already been lost.',
        'base':'ENTRY_GATE_ONLY from Round 4D, frozen MA200 AND regime and frozen Round 3H exit.',
        'action':'For every base trade still open exactly 30 days before a future bear-confirmation start, force full exit at the first hourly bar open on that diagnostic date.'
    }
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(summary.to_string(index=False)); print(y.to_string(index=False)); print(q.sort_values('delta_vs_normal',ascending=False).to_string(index=False))

if __name__=='__main__': main()
