from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from binance_data import load_symbol
from round2b_validation import add_live_features, controlled_activity
from round4e_180d_warning_trail import daily_states, simulate


def process_symbol(symbol, btc, warn_daily, bear_daily, cfg):
    df = btc.copy() if symbol == 'BTCUSDT' else load_symbol(symbol, cfg['interval'], cfg['start'], cfg['end'])
    if len(df) < 800:
        return []
    x = add_live_features(df, btc)
    sigs = controlled_activity(symbol, x, cfg)
    rows = []
    for sig in sigs:
        base = simulate(df, sig, 'CONTROL_STAGE_AWARE', warn_daily, bear_daily, cfg, None)
        if not base.get('warn_seen') or base.get('exit_reason') == 'ENTRY_GATE':
            continue
        entry_time = pd.Timestamp(sig['entry_time'])
        exit_time = pd.Timestamp(base['exit_time'])
        start = int(sig['entry_index'])
        live = df.iloc[start:].copy()
        live = live[(pd.to_datetime(live.time, utc=True) >= entry_time) & (pd.to_datetime(live.time, utc=True) <= exit_time)]
        if live.empty:
            continue
        warn_mask = pd.to_datetime(live.time, utc=True).map(lambda t: bool(warn_daily.get(t.floor('D'), False)))
        if not bool(warn_mask.any()):
            continue
        first_warn_idx = np.flatnonzero(warn_mask.to_numpy())[0]
        first_warn_time = pd.Timestamp(live.iloc[first_warn_idx].time)
        peak_i = int(np.argmax(live.high.astype(float).to_numpy()))
        peak_row = live.iloc[peak_i]
        peak_time = pd.Timestamp(peak_row.time)
        peak_price = float(peak_row.high)
        entry_price = float(sig['entry_price'])
        peak_gain = peak_price / entry_price - 1.0
        warn_price = float(live.iloc[first_warn_idx].open)
        warn_gain = warn_price / entry_price - 1.0
        warn_minus_peak_days = (first_warn_time - peak_time).total_seconds() / 86400.0
        thresholds = {}
        for frac in (0.80, 0.90, 0.95):
            target = entry_price * (1.0 + peak_gain * frac)
            hit = live[live.high.astype(float) >= target]
            thresholds[f't{int(frac*100)}_time'] = pd.Timestamp(hit.iloc[0].time) if len(hit) else pd.NaT
            thresholds[f'warn_minus_t{int(frac*100)}_days'] = ((first_warn_time - thresholds[f't{int(frac*100)}_time']).total_seconds()/86400.0) if len(hit) else np.nan
        rows.append({
            'signal_id': sig['signal_id'], 'symbol': symbol,
            'entry_time': entry_time, 'entry_price': entry_price,
            'warning_time': first_warn_time, 'warning_price': warn_price, 'warning_gain_pct': warn_gain*100,
            'peak_time': peak_time, 'peak_price': peak_price, 'peak_gain_pct': peak_gain*100,
            'warning_minus_peak_days': warn_minus_peak_days,
            'exit_time': exit_time, 'exit_reason': base['exit_reason'], 'normal_pnl_usdt': float(base['pnl_usdt']),
            **thresholds,
        })
    return rows


def main():
    cfg = json.loads(Path('research/config.json').read_text())
    out = Path('research/results/round4e_peak_timing'); out.mkdir(parents=True, exist_ok=True)
    btc = load_symbol('BTCUSDT', cfg['interval'], cfg['start'], cfg['end'])
    warn_daily, bear_daily = daily_states(btc)
    rows = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(process_symbol, s, btc, warn_daily, bear_daily, cfg): s for s in cfg['symbols']}
        for f in as_completed(futs):
            s = futs[f]
            try:
                r = f.result(); rows += r; print(s, len(r), flush=True)
            except Exception as e:
                print('ERROR', s, repr(e), flush=True)
    q = pd.DataFrame(rows).sort_values(['warning_time','symbol'])
    q.to_csv(out/'warning_peak_timing.csv', index=False)
    if len(q):
        summary = {
            'trades': int(len(q)),
            'warning_after_peak_count': int((q.warning_minus_peak_days > 0).sum()),
            'warning_before_or_at_peak_count': int((q.warning_minus_peak_days <= 0).sum()),
            'median_warning_minus_peak_days': float(q.warning_minus_peak_days.median()),
            'mean_warning_minus_peak_days': float(q.warning_minus_peak_days.mean()),
            'median_warning_minus_t80_days': float(q.warn_minus_t80_days.median()),
            'median_warning_minus_t90_days': float(q.warn_minus_t90_days.median()),
            'median_warning_minus_t95_days': float(q.warn_minus_t95_days.median()),
            'late_gt7d': int((q.warning_minus_peak_days > 7).sum()),
            'late_gt30d': int((q.warning_minus_peak_days > 30).sum()),
            'late_gt60d': int((q.warning_minus_peak_days > 60).sum()),
        }
        pd.DataFrame([summary]).to_csv(out/'summary.csv', index=False)
        print('\nSUMMARY\n', pd.DataFrame([summary]).to_string(index=False))
        print('\nDETAIL\n', q[['symbol','entry_time','warning_time','peak_time','warning_minus_peak_days','peak_gain_pct','warning_gain_pct','normal_pnl_usdt']].to_string(index=False))

if __name__ == '__main__':
    main()
