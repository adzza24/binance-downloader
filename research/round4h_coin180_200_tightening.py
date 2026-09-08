from __future__ import annotations

import json, math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from binance_data import load_symbol
from round2b_validation import add_live_features, controlled_activity
from round3a_exit_architecture import capped_stop, net_pnl
from round4g_coin_regime_actions import daily_state, at_state, trail_stop, capital_sim

BASE_GIVEBACK = 0.60
MAs = [180, 200]
TIGHTEN = [0.50, 0.40, 0.30, 0.20, 0.10]
VARIANTS = []
for w in MAs:
    VARIANTS.append(f'COIN{w}_GATE')
    VARIANTS += [f'COIN{w}_TIGHTEN{int(g*100)}' for g in TIGHTEN]
    VARIANTS.append(f'COIN{w}_STAGE_AWARE')


def parse_variant(v):
    w = 180 if v.startswith('COIN180_') else 200
    if v.endswith('_GATE'): return w, 'GATE', None
    if v.endswith('_STAGE_AWARE'): return w, 'STAGE_AWARE', None
    g = int(v.split('TIGHTEN')[-1]) / 100.0
    return w, 'TIGHTEN', g


def res(sig, variant, exit_time, reason, pnl, bars, event=False, event_time=None, event_stage=None, event_peak_gain=None, is_open=False):
    return {
        'signal_id': sig['signal_id'], 'symbol': sig['symbol'], 'entry_time': sig['entry_time'],
        'variant': variant, 'exit_time': exit_time, 'exit_reason': reason, 'pnl_usdt': float(pnl),
        'bars_held': int(bars), 'event': bool(event), 'event_time': event_time,
        'event_stage': event_stage, 'event_peak_gain': event_peak_gain, 'is_open': bool(is_open)
    }


def simulate(df, sig, variant, states, cfg):
    w, action, giveback = parse_variant(variant)
    state = states[w]
    start = int(sig['entry_index']); entry = float(sig['entry_price'])
    init_stop = capped_stop(entry, float(sig['structural_stop']), 0.10)
    entry_day = pd.Timestamp(sig['entry_time']).tz_convert('UTC').floor('D')
    if bool(state.get(entry_day, False)):
        return res(sig, variant, sig['entry_time'], 'ENTRY_GATE', 0.0, 0)

    stop = init_stop; activated = False; peak = entry
    last_state = at_state(df.iloc[start].time, state); event_seen = False

    for j in range(start, len(df)):
        b = df.iloc[j]
        low, high, close, op = map(float, (b.low, b.high, b.close, b.open))
        current_state = at_state(b.time, state)
        state_start = current_state and not last_state
        last_state = current_state
        peak = max(peak, high); pg = peak / entry - 1
        stage = 'INITIAL_PROTECTION' if not activated else ('TRAIL_30PLUS' if pg >= 0.30 else 'BREAKEVEN_PROTECTION')

        if action != 'GATE' and state_start and j > start and not event_seen:
            event_seen = True
            if action == 'TIGHTEN' and activated:
                cand = trail_stop(entry, peak, close, giveback)
                if cand is not None: stop = max(stop, cand)
            elif action == 'STAGE_AWARE':
                if stage == 'INITIAL_PROTECTION':
                    return res(sig, variant, b.time, f'COIN{w}_STAGE_INITIAL_EXIT', net_pnl(entry, [(1.0, op)], cfg), j-start+1,
                               True, b.time, stage, pg)
                elif stage == 'BREAKEVEN_PROTECTION':
                    stop = max(stop, op * 0.95)
                else:
                    cand = trail_stop(entry, peak, close, 0.50)
                    if cand is not None: stop = max(stop, cand)

        if not activated:
            if low <= init_stop:
                return res(sig, variant, b.time, 'STOP', net_pnl(entry, [(1.0, init_stop)], cfg), j-start+1,
                           event_seen, b.time if event_seen else None, stage if event_seen else None, pg if event_seen else None)
            if high >= entry * 1.05:
                activated = True; stop = max(stop, entry); continue
        else:
            if low <= stop:
                return res(sig, variant, b.time, 'GAIN_TRAIL_STOP', net_pnl(entry, [(1.0, stop)], cfg), j-start+1,
                           event_seen, b.time if event_seen else None, stage if event_seen else None, pg if event_seen else None)
            if high >= entry * 11.0:
                px = entry * 11.0
                return res(sig, variant, b.time, 'TAKE_PROFIT', net_pnl(entry, [(1.0, px)], cfg), j-start+1,
                           event_seen, b.time if event_seen else None, stage if event_seen else None, pg if event_seen else None)
            if pg >= 0.30: stop = max(stop, entry * 1.10)
            cand = trail_stop(entry, peak, close, BASE_GIVEBACK)
            if cand is not None: stop = max(stop, cand)

    close = float(df.iloc[-1].close)
    return res(sig, variant, df.iloc[-1].time, 'DATA_END_OPEN', net_pnl(entry, [(1.0, close)], cfg), len(df)-start,
               event_seen, is_open=True)


def process_symbol(symbol, btc, cfg):
    df = btc.copy() if symbol == 'BTCUSDT' else load_symbol(symbol, cfg['interval'], cfg['start'], cfg['end'])
    if len(df) < 800: return [], []
    states = {w: daily_state(df, w) for w in MAs}
    x = add_live_features(df, btc); sigs = controlled_activity(symbol, x, cfg); rows = []
    for sig in sigs:
        for v in VARIANTS: rows.append(simulate(df, sig, v, states, cfg))
    return sigs, rows


def pf(vals):
    vals = np.asarray(vals, float); pos = vals[vals > 0]; neg = vals[vals < 0]
    return float(pos.sum() / abs(neg.sum())) if len(neg) and neg.sum() else math.inf


def summarise(trades):
    q = trades.copy(); q['entry_time'] = pd.to_datetime(q.entry_time, utc=True); q['exit_time'] = pd.to_datetime(q.exit_time, utc=True)
    rows = []; years = []
    for v, g in q.groupby('variant'):
        vals = g.pnl_usdt.to_numpy(float); active = g[g.exit_reason != 'ENTRY_GATE']; cap, dd, skip = capital_sim(g)
        rows.append({
            'variant': v, 'signals': len(g), 'active_trades': len(active), 'entry_gate_skips': int((g.exit_reason == 'ENTRY_GATE').sum()),
            'open_position_events': int(g.event.sum()), 'combined_pnl': float(vals.sum()), 'profit_factor': pf(vals[vals != 0]),
            'win_rate_active': float((active.pnl_usdt > 0).mean()) if len(active) else np.nan,
            'capital_profit_proxy': float(cap), 'capital_max_realised_dd': float(dd), 'skipped_due_cash': int(skip)
        })
        for y, gy in g.groupby(g.entry_time.dt.year):
            years.append({'variant': v, 'year': int(y), 'pnl_usdt': float(gy.pnl_usdt.sum())})
    return pd.DataFrame(rows), pd.DataFrame(years)


def main():
    cfg = json.loads(Path('research/config.json').read_text()); out = Path('research/results/round4h'); out.mkdir(parents=True, exist_ok=True)
    btc = load_symbol('BTCUSDT', cfg['interval'], cfg['start'], cfg['end']); all_s = []; all_t = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        fut = {ex.submit(process_symbol, s, btc, cfg): s for s in cfg['symbols']}
        for f in as_completed(fut):
            s = fut[f]
            try:
                sig, rows = f.result(); all_s += sig; all_t += rows; print(s, len(sig), flush=True)
            except Exception as e: print('ERROR', s, repr(e), flush=True)
    trades = pd.DataFrame(all_t); pd.DataFrame(all_s).to_csv(out/'signals.csv', index=False); trades.to_csv(out/'trades.csv', index=False)
    sm, yr = summarise(trades); sm.to_csv(out/'summary.csv', index=False); yr.to_csv(out/'year_summary.csv', index=False)
    manifest = {
        'study': 'Round 4H coin 180D vs 200D open-position tightening comparison',
        'status': 'RESEARCH ONLY - no live strategy changes',
        'signal': 'Coin close below own MA AND own MA falling vs 20 completed days earlier, shifted one full day causally.',
        'ma_windows': MAs, 'tighten_givebacks': TIGHTEN,
        'base_exit': 'Frozen Round 3H 60% accumulated-profit giveback, +5 breakeven, +30 +10 floor, +1000% take profit.',
        'stage_aware': 'INITIAL exits at signal open; BREAKEVEN tightens to 5% below signal open; TRAIL_30PLUS tightens to 50% giveback.',
        'purpose': 'Test whether earlier coin180 confirmation reaches more still-open positions and whether stronger tightening at coin180/coin200 protects more peak profit without clipping leaders.',
        'capital_proxy': '1750 start, 300 per trade, realised/cost-basis DD only; not true mark-to-market.'
    }
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('\nSUMMARY\n', sm.sort_values('combined_pnl', ascending=False).to_string(index=False)); print('\nYEARS\n', yr.to_string(index=False))

if __name__ == '__main__': main()
