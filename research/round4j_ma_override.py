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
VARIANTS = [
    'CONTROL_TRAIL60',
    'COIN180_ALL_STAGE_OVERRIDE',
    'COIN200_ALL_STAGE_OVERRIDE',
    'COIN180_POST30_OVERRIDE',
    'COIN200_POST30_OVERRIDE',
]


def trail_stop(entry, peak, close):
    pg = peak / entry - 1
    if pg < 0.30:
        return None
    protected_gain = pg * (1 - BASE_GIVEBACK)
    return min(entry * (1 + protected_gain), close * 0.999)


def variant_ma(variant):
    if '180' in variant:
        return 'coin180'
    if '200' in variant:
        return 'coin200'
    return None


def result(sig, variant, exit_time, reason, pnl, bars, peak, exit_px, reached30,
           ma_events=None, is_open=False):
    entry = float(sig['entry_price'])
    peak_gain = peak / entry - 1
    exit_gain = exit_px / entry - 1 if exit_px is not None else np.nan
    retained = exit_gain / peak_gain if reached30 and peak_gain > 0 else np.nan
    ma_events = ma_events or []
    return {
        'signal_id': sig['signal_id'], 'symbol': sig['symbol'], 'entry_time': sig['entry_time'],
        'variant': variant, 'exit_time': exit_time, 'exit_reason': reason, 'pnl_usdt': float(pnl),
        'bars_held': int(bars), 'peak_price': float(peak), 'peak_gain': float(peak_gain),
        'exit_price': float(exit_px) if exit_px is not None else np.nan,
        'exit_gain': float(exit_gain) if exit_px is not None else np.nan,
        'reached_30': bool(reached30),
        'peak_profit_retained': float(retained) if pd.notna(retained) else np.nan,
        'first_ma_time': ma_events[0]['time'] if ma_events else pd.NaT,
        'first_ma_gain_stage': ma_events[0]['stage'] if ma_events else None,
        'first_ma_peak_gain': ma_events[0]['peak_gain'] if ma_events else np.nan,
        'ma_event_count': len(ma_events), 'is_open': bool(is_open),
    }


def stage_from_peak(pg):
    if pg >= 0.30:
        return '30PLUS'
    if pg >= 0.05:
        return '5_TO_30'
    return 'UNDER_5'


def simulate(df, sig, variant, states, cfg):
    start = int(sig['entry_index'])
    entry = float(sig['entry_price'])
    init_stop = capped_stop(entry, float(sig['structural_stop']), 0.10)

    entry_day = pd.Timestamp(sig['entry_time']).tz_convert('UTC').floor('D')
    if bool(states['btc200'].get(entry_day, False)):
        return result(sig, variant, sig['entry_time'], 'ENTRY_GATE', 0.0, 0, entry, entry, False)

    stop = init_stop
    activated = False
    peak = entry
    reached30 = False
    ma_key = variant_ma(variant)
    last_ma = at_state(df.iloc[start].time, states[ma_key]) if ma_key else False
    ma_events = []

    for j in range(start, len(df)):
        b = df.iloc[j]
        low, high, close, op = map(float, (b.low, b.high, b.close, b.open))

        # MA state is based only on completed prior-day data. Act on a new confirmed
        # deterioration transition at this hourly open before observing this bar high/low.
        if ma_key:
            current_ma = at_state(b.time, states[ma_key])
            ma_start = current_ma and not last_ma
            last_ma = current_ma
            if ma_start and j > start:
                pg_before_bar = peak / entry - 1
                event = {'time': b.time, 'stage': stage_from_peak(pg_before_bar), 'peak_gain': pg_before_bar}
                ma_events.append(event)
                all_stage = 'ALL_STAGE' in variant
                post30 = 'POST30' in variant and reached30
                if all_stage or post30:
                    return result(sig, variant, b.time, f'{ma_key.upper()}_HARD_EXIT',
                                  net_pnl(entry, [(1.0, op)], cfg), j-start+1, peak, op,
                                  reached30, ma_events)

        peak = max(peak, high)
        pg = peak / entry - 1
        reached30 = reached30 or pg >= 0.30

        if not activated:
            if low <= init_stop:
                return result(sig, variant, b.time, 'STOP', net_pnl(entry, [(1.0, init_stop)], cfg),
                              j-start+1, peak, init_stop, reached30, ma_events)
            if high >= entry * 1.05:
                activated = True
                stop = max(stop, entry)
                continue
        else:
            if low <= stop:
                reason = 'GAIN_TRAIL_STOP' if reached30 else 'BREAKEVEN_STOP'
                return result(sig, variant, b.time, reason, net_pnl(entry, [(1.0, stop)], cfg),
                              j-start+1, peak, stop, reached30, ma_events)
            if high >= entry * 11.0:
                px = entry * 11.0
                return result(sig, variant, b.time, 'TAKE_PROFIT', net_pnl(entry, [(1.0, px)], cfg),
                              j-start+1, peak, px, reached30, ma_events)

            if pg >= 0.30:
                stop = max(stop, entry * 1.10)

            # Exact current mature-winner architecture retained in every variant:
            # 60% accumulated-profit giveback trail from peak.
            cand = trail_stop(entry, peak, close)
            if cand is not None:
                stop = max(stop, cand)

    close = float(df.iloc[-1].close)
    return result(sig, variant, df.iloc[-1].time, 'DATA_END_OPEN', net_pnl(entry, [(1.0, close)], cfg),
                  len(df)-start, peak, close, reached30, ma_events, is_open=True)


def process_symbol(symbol, btc, btc200, cfg):
    df = btc.copy() if symbol == 'BTCUSDT' else load_symbol(symbol, cfg['interval'], cfg['start'], cfg['end'])
    if len(df) < 800:
        return [], []
    states = {'btc200': btc200, 'coin180': daily_state(df, 180), 'coin200': daily_state(df, 200)}
    x = add_live_features(df, btc)
    sigs = controlled_activity(symbol, x, cfg)
    rows = []
    for sig in sigs:
        for v in VARIANTS:
            rows.append(simulate(df, sig, v, states, cfg))
    return sigs, rows


def pf(vals):
    vals = np.asarray(vals, float)
    pos, neg = vals[vals > 0], vals[vals < 0]
    return float(pos.sum() / abs(neg.sum())) if len(neg) and neg.sum() else math.inf


def capital_sim(g):
    x = g.sort_values(['entry_time','symbol','signal_id']).copy()
    cash = START_CAPITAL; heap = []; open_principal = 0.0
    peak = START_CAPITAL; max_dd = 0.0; skipped = 0
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
            'variant': v, 'signals': len(g), 'active_trades': len(active),
            'entry_gate_skips': int((g.exit_reason == 'ENTRY_GATE').sum()),
            'combined_pnl': float(vals.sum()), 'profit_factor': pf(vals[vals != 0]),
            'win_rate_active': float((active.pnl_usdt > 0).mean()) if len(active) else np.nan,
            'capital_profit_proxy': float(cap), 'capital_max_realised_dd': float(dd),
            'skipped_due_cash': int(skip), 'trades_reached_30': int(len(winners30)),
            'ma_hard_exits': int(g.exit_reason.str.contains('HARD_EXIT', na=False).sum()),
            'median_peak_gain_30plus': float(winners30.peak_gain.median()) if len(winners30) else np.nan,
            'median_exit_gain_30plus': float(winners30.exit_gain.median()) if len(winners30) else np.nan,
            'median_peak_profit_retained': float(winners30.peak_profit_retained.median()) if len(winners30) else np.nan,
        })
        for y, gy in g.groupby(g.entry_time.dt.year):
            years.append({'variant': v, 'year': int(y), 'pnl_usdt': float(gy.pnl_usdt.sum())})
    return pd.DataFrame(rows), pd.DataFrame(years)


def comparisons(trades):
    control = trades[trades.variant == 'CONTROL_TRAIL60'][['signal_id','pnl_usdt','exit_reason','exit_time','peak_gain','exit_gain']].copy()
    control = control.rename(columns={c: f'control_{c}' for c in control.columns if c != 'signal_id'})
    rows = []
    for v in VARIANTS[1:]:
        g = trades[trades.variant == v].merge(control, on='signal_id', how='left')
        g['delta_pnl_vs_control'] = g.pnl_usdt - g.control_pnl_usdt
        g['outcome'] = np.where(g.delta_pnl_vs_control > 1e-9, 'IMPROVED',
                       np.where(g.delta_pnl_vs_control < -1e-9, 'HARMED', 'UNCHANGED'))
        rows.append(g)
    comp = pd.concat(rows, ignore_index=True)
    sm = comp.groupby('variant').agg(
        improved=('outcome', lambda s: int((s == 'IMPROVED').sum())),
        harmed=('outcome', lambda s: int((s == 'HARMED').sum())),
        unchanged=('outcome', lambda s: int((s == 'UNCHANGED').sum())),
        total_delta_pnl=('delta_pnl_vs_control','sum'),
        median_delta_pnl=('delta_pnl_vs_control','median'),
        max_improvement=('delta_pnl_vs_control','max'),
        max_harm=('delta_pnl_vs_control','min'),
    ).reset_index()
    return comp, sm


def main():
    cfg = json.loads(Path('research/config.json').read_text())
    out = Path('research/results/round4j'); out.mkdir(parents=True, exist_ok=True)
    btc = load_symbol('BTCUSDT', cfg['interval'], cfg['start'], cfg['end'])
    btc200 = daily_state(btc, 200)
    all_s, all_t = [], []
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
    comp, comp_sm = comparisons(trades)
    comp.to_csv(out/'control_comparisons.csv', index=False)
    comp_sm.to_csv(out/'comparison_summary.csv', index=False)
    stage = (comp[comp.first_ma_gain_stage.notna()]
             .groupby(['variant','first_ma_gain_stage'])
             .agg(trades=('signal_id','count'), delta_pnl=('delta_pnl_vs_control','sum'),
                  improved=('outcome', lambda s: int((s == 'IMPROVED').sum())),
                  harmed=('outcome', lambda s: int((s == 'HARMED').sum())))
             .reset_index())
    stage.to_csv(out/'ma_stage_summary.csv', index=False)
    manifest = {
        'study': 'Round 4J MA hard-exit override added to frozen 60% trailing architecture',
        'status': 'RESEARCH ONLY - no live strategy changes',
        'control': 'Frozen current setup: structural stop capped ~-10%, +5% breakeven, +30% +10% floor plus 60% accumulated-profit giveback trail, +1000% take-profit.',
        'variants': VARIANTS,
        'all_stage_override': 'Hard exit on first confirmed own-coin MA deterioration transition at hourly open, whether trade is below +5%, +5-30%, or 30%+.',
        'post30_override': 'Hard exit only when a new confirmed MA deterioration transition occurs after the trade has already reached +30%; pre-30 transitions are recorded but not carried forward as active exits.',
        'ma_signal': 'Coin close below own MA AND own MA falling vs 20 completed days earlier, shifted one full day causally.',
        'trail': '60% accumulated-profit giveback remains active in every variant; no alternative trail percentages tested.',
        'position_model': 'Primary combined P&L uses fixed 300 USDT per signal without constraining overlaps. Existing 1750 USDT realised/cost-basis proxy retained only as secondary diagnostic.'
    }
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('\nSUMMARY\n', sm.sort_values('combined_pnl', ascending=False).to_string(index=False))
    print('\nCOMPARISON\n', comp_sm.sort_values('total_delta_pnl', ascending=False).to_string(index=False))
    print('\nMA STAGES\n', stage.to_string(index=False))

if __name__ == '__main__':
    main()
