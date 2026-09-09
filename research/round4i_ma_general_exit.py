from __future__ import annotations

import json, math, heapq
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from binance_data import load_symbol
from round2b_validation import add_live_features, controlled_activity
from round3a_exit_architecture import capped_stop, net_pnl
from round4g_coin_regime_actions import daily_state, at_state

START_CAPITAL = 1750.0
POSITION_USDT = 300.0
BASE_GIVEBACK = 0.60
VARIANTS = ['ROUND3H_TRAIL60', 'COIN180_HARD_EXIT', 'COIN200_HARD_EXIT']


def trail_stop(entry, peak, close, giveback):
    pg = peak / entry - 1
    if pg < 0.30:
        return None
    protected_gain = pg * (1 - giveback)
    return min(entry * (1 + protected_gain), close * 0.999)


def result(sig, variant, exit_time, reason, pnl, bars, peak, exit_px, reached30, is_open=False):
    entry = float(sig['entry_price'])
    peak_gain = peak / entry - 1
    exit_gain = exit_px / entry - 1 if exit_px is not None else np.nan
    retained = exit_gain / peak_gain if reached30 and peak_gain > 0 else np.nan
    return {
        'signal_id': sig['signal_id'], 'symbol': sig['symbol'], 'entry_time': sig['entry_time'],
        'variant': variant, 'exit_time': exit_time, 'exit_reason': reason, 'pnl_usdt': float(pnl),
        'bars_held': int(bars), 'peak_price': float(peak), 'peak_gain': float(peak_gain),
        'exit_price': float(exit_px) if exit_px is not None else np.nan, 'exit_gain': float(exit_gain) if exit_px is not None else np.nan,
        'reached_30': bool(reached30), 'peak_profit_retained': float(retained) if pd.notna(retained) else np.nan,
        'is_open': bool(is_open)
    }


def simulate(df, sig, variant, states, cfg):
    start = int(sig['entry_index'])
    entry = float(sig['entry_price'])
    init_stop = capped_stop(entry, float(sig['structural_stop']), 0.10)

    # Keep the same frozen global BTC200 entry gate for every exit variant.
    entry_day = pd.Timestamp(sig['entry_time']).tz_convert('UTC').floor('D')
    if bool(states['btc200'].get(entry_day, False)):
        return result(sig, variant, sig['entry_time'], 'ENTRY_GATE', 0.0, 0, entry, entry, False)

    stop = init_stop
    activated = False
    peak = entry
    reached30 = False

    ma_key = None
    if variant == 'COIN180_HARD_EXIT': ma_key = 'coin180'
    if variant == 'COIN200_HARD_EXIT': ma_key = 'coin200'
    last_ma = at_state(df.iloc[start].time, states[ma_key]) if ma_key else False

    for j in range(start, len(df)):
        b = df.iloc[j]
        low, high, close, op = map(float, (b.low, b.high, b.close, b.open))

        # A confirmed MA deterioration exit is acted on at the first hourly OPEN
        # of the causal completed-day state transition, before using this bar's high/low.
        if ma_key:
            current_ma = at_state(b.time, states[ma_key])
            ma_start = current_ma and not last_ma
            last_ma = current_ma
            if ma_start and j > start:
                return result(sig, variant, b.time, f'{ma_key.upper()}_HARD_EXIT',
                              net_pnl(entry, [(1.0, op)], cfg), j-start+1, peak, op, reached30)

        peak = max(peak, high)
        pg = peak / entry - 1
        reached30 = reached30 or pg >= 0.30

        if not activated:
            if low <= init_stop:
                return result(sig, variant, b.time, 'STOP', net_pnl(entry, [(1.0, init_stop)], cfg),
                              j-start+1, peak, init_stop, reached30)
            if high >= entry * 1.05:
                activated = True
                stop = max(stop, entry)
                continue
        else:
            if low <= stop:
                reason = 'GAIN_TRAIL_STOP' if variant == 'ROUND3H_TRAIL60' and reached30 else 'PROFIT_FLOOR_STOP'
                return result(sig, variant, b.time, reason, net_pnl(entry, [(1.0, stop)], cfg),
                              j-start+1, peak, stop, reached30)
            if high >= entry * 11.0:
                px = entry * 11.0
                return result(sig, variant, b.time, 'TAKE_PROFIT', net_pnl(entry, [(1.0, px)], cfg),
                              j-start+1, peak, px, reached30)

            # All variants retain the +10% minimum profit floor after +30% peak gain.
            if pg >= 0.30:
                stop = max(stop, entry * 1.10)

            # Only the control retains the dynamic 60% accumulated-profit giveback trail.
            if variant == 'ROUND3H_TRAIL60':
                cand = trail_stop(entry, peak, close, BASE_GIVEBACK)
                if cand is not None:
                    stop = max(stop, cand)

    close = float(df.iloc[-1].close)
    return result(sig, variant, df.iloc[-1].time, 'DATA_END_OPEN', net_pnl(entry, [(1.0, close)], cfg),
                  len(df)-start, peak, close, reached30, is_open=True)


def process_symbol(symbol, btc, btc200, cfg):
    df = btc.copy() if symbol == 'BTCUSDT' else load_symbol(symbol, cfg['interval'], cfg['start'], cfg['end'])
    if len(df) < 800:
        return [], []
    states = {
        'btc200': btc200,
        'coin180': daily_state(df, 180),
        'coin200': daily_state(df, 200),
    }
    x = add_live_features(df, btc)
    sigs = controlled_activity(symbol, x, cfg)
    rows = []
    for sig in sigs:
        for v in VARIANTS:
            rows.append(simulate(df, sig, v, states, cfg))
    return sigs, rows


def pf(vals):
    vals = np.asarray(vals, float)
    pos = vals[vals > 0]
    neg = vals[vals < 0]
    return float(pos.sum() / abs(neg.sum())) if len(neg) and neg.sum() else math.inf


def capital_sim(g):
    x = g.sort_values(['entry_time','symbol','signal_id']).copy()
    cash = START_CAPITAL; heap = []; open_principal = 0.0; peak = START_CAPITAL; max_dd = 0.0; skipped = 0
    for _, r in x.iterrows():
        t = r.entry_time
        while heap and heap[0][0] <= t:
            et, principal, pnl = heapq.heappop(heap)
            cash += principal + pnl; open_principal -= principal
            eq = cash + open_principal; peak = max(peak, eq); max_dd = min(max_dd, eq - peak)
        if r.exit_reason == 'ENTRY_GATE':
            continue
        if cash + 1e-9 < POSITION_USDT:
            skipped += 1; continue
        cash -= POSITION_USDT; open_principal += POSITION_USDT
        heapq.heappush(heap, (r.exit_time, POSITION_USDT, float(r.pnl_usdt)))
    while heap:
        et, principal, pnl = heapq.heappop(heap)
        cash += principal + pnl; open_principal -= principal
        eq = cash + open_principal; peak = max(peak, eq); max_dd = min(max_dd, eq - peak)
    return cash - START_CAPITAL, max_dd, skipped


def summarise(trades):
    q = trades.copy()
    q['entry_time'] = pd.to_datetime(q.entry_time, utc=True)
    q['exit_time'] = pd.to_datetime(q.exit_time, utc=True)
    rows = []; years = []
    for v, g in q.groupby('variant'):
        vals = g.pnl_usdt.to_numpy(float)
        active = g[g.exit_reason != 'ENTRY_GATE']
        winners30 = active[active.reached_30]
        cap, dd, skip = capital_sim(g)
        rows.append({
            'variant': v, 'signals': len(g), 'active_trades': len(active), 'entry_gate_skips': int((g.exit_reason == 'ENTRY_GATE').sum()),
            'combined_pnl': float(vals.sum()), 'profit_factor': pf(vals[vals != 0]),
            'win_rate_active': float((active.pnl_usdt > 0).mean()) if len(active) else np.nan,
            'capital_profit_proxy': float(cap), 'capital_max_realised_dd': float(dd), 'skipped_due_cash': int(skip),
            'trades_reached_30': int(len(winners30)),
            'median_peak_gain_30plus': float(winners30.peak_gain.median()) if len(winners30) else np.nan,
            'median_exit_gain_30plus': float(winners30.exit_gain.median()) if len(winners30) else np.nan,
            'median_peak_profit_retained': float(winners30.peak_profit_retained.median()) if len(winners30) else np.nan,
            'mean_peak_profit_retained': float(winners30.peak_profit_retained.mean()) if len(winners30) else np.nan,
        })
        for y, gy in g.groupby(g.entry_time.dt.year):
            years.append({'variant': v, 'year': int(y), 'pnl_usdt': float(gy.pnl_usdt.sum())})
    return pd.DataFrame(rows), pd.DataFrame(years)


def main():
    cfg = json.loads(Path('research/config.json').read_text())
    out = Path('research/results/round4i'); out.mkdir(parents=True, exist_ok=True)
    btc = load_symbol('BTCUSDT', cfg['interval'], cfg['start'], cfg['end'])
    btc200 = daily_state(btc, 200)
    all_s = []; all_t = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        fut = {ex.submit(process_symbol, s, btc, btc200, cfg): s for s in cfg['symbols']}
        for f in as_completed(fut):
            s = fut[f]
            try:
                sig, rows = f.result(); all_s += sig; all_t += rows; print(s, len(sig), flush=True)
            except Exception as e:
                print('ERROR', s, repr(e), flush=True)
    trades = pd.DataFrame(all_t)
    pd.DataFrame(all_s).to_csv(out/'signals.csv', index=False)
    trades.to_csv(out/'trades.csv', index=False)
    sm, yr = summarise(trades)
    sm.to_csv(out/'summary.csv', index=False)
    yr.to_csv(out/'year_summary.csv', index=False)
    manifest = {
        'study': 'Round 4I MA as general exit strategy',
        'status': 'RESEARCH ONLY - no live strategy changes',
        'entry_gate': 'Same BTC200 AND falling-vs-20d causal gate for every variant.',
        'control': 'Frozen Round 3H: structural stop, +5 breakeven, +30 +10 floor, 60% accumulated-profit giveback, +1000% take profit.',
        'ma_exit_variants': 'After +30 keep only +10% floor; remove dynamic giveback trail; hard exit at first confirmed own-coin 180D or 200D AND deterioration transition at hourly open.',
        'ma_signal': 'Coin close below own MA AND own MA falling vs 20 completed days earlier, shifted one full day causally.',
        'peak_metrics': 'For every trade that reaches +30%, record peak gain, exit gain and exit_gain/peak_gain retention.',
        'capital_proxy': '1750 start, 300 per trade, realised/cost-basis drawdown proxy only; not true mark-to-market.'
    }
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('\nSUMMARY\n', sm.sort_values('combined_pnl', ascending=False).to_string(index=False))
    print('\nYEARS\n', yr.to_string(index=False))

if __name__ == '__main__':
    main()
