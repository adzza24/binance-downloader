from __future__ import annotations

import json, math, heapq
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from binance_data import load_symbol
from round2b_validation import add_live_features, controlled_activity
from round3a_exit_architecture import capped_stop, net_pnl

START_CAPITAL = 1750.0
POSITION_USDT = 300.0
BASE_SCHEDULE = [(m, 0.60) for m in [0.30, 0.50, 1.00, 2.00, 3.00, 5.00, 7.50]]
VARIANTS = [
    'ENTRY_GATE_ONLY',
    'FULL_LIQUIDATION',
    'PROFIT_ONLY_LIQUIDATION',
    'PARTIAL_50',
    'TIGHTEN_5PCT',
    'STAGE_AWARE',
]


def build_bear_state(btc: pd.DataFrame) -> pd.Series:
    x = btc[['time', 'close']].copy()
    x['time'] = pd.to_datetime(x.time, utc=True)
    d = x.set_index('time').close.astype(float).resample('1D').last().dropna()
    ma = d.rolling(200, min_periods=180).mean()
    raw = (d < ma) & (ma / ma.shift(20) - 1 < 0)
    # Causal: day D uses only completed information through D-1.
    return raw.shift(1).fillna(False).astype(bool)


def bar_bear_state(bar_time, bear_daily: pd.Series) -> bool:
    d = pd.Timestamp(bar_time).tz_convert('UTC').floor('D')
    return bool(bear_daily.get(d, False))


def result(sig, variant, exit_time, exit_reason, pnl, bars, event_time=None, event_price=None,
           event_stage=None, normal_pnl=None, intervention_fraction=0.0, is_open=False):
    normal_pnl = pnl if normal_pnl is None else normal_pnl
    return {
        'signal_id': sig['signal_id'], 'symbol': sig['symbol'], 'entry_time': sig['entry_time'],
        'variant': variant, 'exit_time': exit_time, 'exit_reason': exit_reason,
        'pnl_usdt': float(pnl), 'normal_pnl_usdt': float(normal_pnl),
        'counterfactual_delta_usdt': float(pnl - normal_pnl), 'bars_held': int(bars),
        'bear_event_time': event_time, 'bear_event_price': event_price, 'bear_event_stage': event_stage,
        'intervention_fraction': float(intervention_fraction), 'is_open': bool(is_open),
    }


def simulate(df: pd.DataFrame, sig: dict, variant: str, bear_daily: pd.Series, cfg: dict, normal_pnl: float | None = None):
    start = int(sig['entry_index']); entry = float(sig['entry_price'])
    init_stop = capped_stop(entry, float(sig['structural_stop']), 0.10)
    entry_day = pd.Timestamp(sig['entry_time']).tz_convert('UTC').floor('D')
    if bool(bear_daily.get(entry_day, False)):
        return result(sig, variant, sig['entry_time'], 'ENTRY_GATE', 0.0, 0, normal_pnl=0.0)

    stop = init_stop; activated = False; peak = entry; stage_gb = None
    partial_exit = None; remaining = 1.0; intervention_done = False
    last_bear = False

    for j in range(start, len(df)):
        b = df.iloc[j]
        low, high, close, op = map(float, (b.low, b.high, b.close, b.open))
        current_bear = bar_bear_state(b.time, bear_daily)
        bear_start = current_bear and not last_bear
        last_bear = current_bear

        peak = max(peak, high)
        peak_gain_pre = peak / entry - 1
        stage = 'INITIAL_PROTECTION' if not activated else ('TRAIL_30PLUS' if peak_gain_pre >= 0.30 else 'BREAKEVEN_PROTECTION')

        if bear_start and not intervention_done and j > start:
            intervention_done = True
            event_time, event_price, event_stage = b.time, op, stage
            if variant == 'FULL_LIQUIDATION':
                pnl = net_pnl(entry, [(1.0, op)], cfg)
                return result(sig, variant, b.time, 'BEAR_FULL_LIQUIDATION', pnl, j-start+1,
                              event_time, op, stage, normal_pnl, 1.0)
            if variant == 'PROFIT_ONLY_LIQUIDATION' and op > entry:
                pnl = net_pnl(entry, [(1.0, op)], cfg)
                return result(sig, variant, b.time, 'BEAR_PROFIT_LIQUIDATION', pnl, j-start+1,
                              event_time, op, stage, normal_pnl, 1.0)
            if variant == 'PARTIAL_50':
                partial_exit = op; remaining = 0.5
            elif variant == 'TIGHTEN_5PCT':
                stop = max(stop, op * 0.95)
            elif variant == 'STAGE_AWARE':
                if stage == 'INITIAL_PROTECTION':
                    pnl = net_pnl(entry, [(1.0, op)], cfg)
                    return result(sig, variant, b.time, 'BEAR_STAGE_INITIAL_LIQUIDATION', pnl, j-start+1,
                                  event_time, op, stage, normal_pnl, 1.0)
                elif stage == 'BREAKEVEN_PROTECTION':
                    stop = max(stop, op * 0.95)
                # 30%+ runners keep the frozen Round 3H trail unchanged.

        # Frozen Round 3H ordering and trail.
        if not activated:
            if low <= init_stop:
                exits = [(0.5, partial_exit), (0.5, init_stop)] if partial_exit is not None else [(1.0, init_stop)]
                pnl = net_pnl(entry, exits, cfg)
                return result(sig, variant, b.time, 'STOP', pnl, j-start+1,
                              locals().get('event_time'), locals().get('event_price'), locals().get('event_stage'), normal_pnl,
                              0.5 if partial_exit is not None else 0.0)
            if high >= entry * 1.05:
                activated = True; stop = max(stop, entry); continue
        else:
            if low <= stop:
                exits = [(0.5, partial_exit), (0.5, stop)] if partial_exit is not None else [(1.0, stop)]
                pnl = net_pnl(entry, exits, cfg)
                return result(sig, variant, b.time, 'GAIN_TRAIL_STOP', pnl, j-start+1,
                              locals().get('event_time'), locals().get('event_price'), locals().get('event_stage'), normal_pnl,
                              0.5 if partial_exit is not None else 0.0)
            if high >= entry * 11.0:
                px = entry * 11.0
                exits = [(0.5, partial_exit), (0.5, px)] if partial_exit is not None else [(1.0, px)]
                pnl = net_pnl(entry, exits, cfg)
                return result(sig, variant, b.time, 'TAKE_PROFIT', pnl, j-start+1,
                              locals().get('event_time'), locals().get('event_price'), locals().get('event_stage'), normal_pnl,
                              0.5 if partial_exit is not None else 0.0)
            peak_gain = peak / entry - 1
            for threshold, gb in BASE_SCHEDULE:
                if peak_gain >= threshold: stage_gb = gb
            if peak_gain >= 0.30:
                stop = max(stop, entry * 1.10)
            if stage_gb is not None:
                floor_gain = peak_gain * (1 - stage_gb)
                candidate = min(entry * (1 + floor_gain), close * 0.999)
                stop = max(stop, candidate)

    close = float(df.iloc[-1].close)
    exits = [(0.5, partial_exit), (0.5, close)] if partial_exit is not None else [(1.0, close)]
    pnl = net_pnl(entry, exits, cfg)
    return result(sig, variant, df.iloc[-1].time, 'DATA_END_OPEN', pnl, len(df)-start,
                  locals().get('event_time'), locals().get('event_price'), locals().get('event_stage'), normal_pnl,
                  0.5 if partial_exit is not None else 0.0, True)


def process_symbol(symbol, btc, bear_daily, cfg):
    df = btc.copy() if symbol == 'BTCUSDT' else load_symbol(symbol, cfg['interval'], cfg['start'], cfg['end'])
    if len(df) < 800: return [], []
    x = add_live_features(df, btc); sigs = controlled_activity(symbol, x, cfg)
    rows = []
    for sig in sigs:
        base = simulate(df, sig, 'ENTRY_GATE_ONLY', bear_daily, cfg, None)
        base_pnl = float(base['pnl_usdt'])
        base['normal_pnl_usdt'] = base_pnl; base['counterfactual_delta_usdt'] = 0.0
        rows.append(base)
        for v in VARIANTS[1:]:
            rows.append(simulate(df, sig, v, bear_daily, cfg, base_pnl))
    return sigs, rows


def pf(vals):
    vals = np.asarray(vals, float); pos = vals[vals > 0]; neg = vals[vals < 0]
    return float(pos.sum() / abs(neg.sum())) if len(neg) and neg.sum() else math.inf


def capital_sim(g):
    x = g.sort_values(['entry_time', 'symbol', 'signal_id']).copy()
    cash = START_CAPITAL; open_heap = []; open_principal = 0.0; peak = START_CAPITAL; max_dd = 0.0; skipped_cash = 0
    for _, r in x.iterrows():
        t = r.entry_time
        while open_heap and open_heap[0][0] <= t:
            et, principal, pnl = heapq.heappop(open_heap); cash += principal + pnl; open_principal -= principal
            eq = cash + open_principal; peak = max(peak, eq); max_dd = min(max_dd, eq - peak)
        if r.exit_reason == 'ENTRY_GATE': continue
        if cash + 1e-9 < POSITION_USDT:
            skipped_cash += 1; continue
        cash -= POSITION_USDT; open_principal += POSITION_USDT
        heapq.heappush(open_heap, (r.exit_time, POSITION_USDT, float(r.pnl_usdt)))
    while open_heap:
        et, principal, pnl = heapq.heappop(open_heap); cash += principal + pnl; open_principal -= principal
        eq = cash + open_principal; peak = max(peak, eq); max_dd = min(max_dd, eq - peak)
    return cash - START_CAPITAL, max_dd, skipped_cash


def summarise(trades):
    rows = []; years = []
    q = trades.copy(); q['entry_time'] = pd.to_datetime(q.entry_time, utc=True); q['exit_time'] = pd.to_datetime(q.exit_time, utc=True)
    for v, g in q.groupby('variant'):
        vals = g.pnl_usdt.to_numpy(float); active = g[g.exit_reason != 'ENTRY_GATE']
        cap_pnl, cap_dd, cash_skip = capital_sim(g)
        rows.append({
            'variant': v, 'signals': len(g), 'active_trades': len(active), 'entry_gate_skips': int((g.exit_reason == 'ENTRY_GATE').sum()),
            'bear_interventions': int(g.bear_event_time.notna().sum()), 'combined_pnl': float(vals.sum()), 'profit_factor': pf(vals[vals != 0]),
            'win_rate_active': float((active.pnl_usdt > 0).mean()) if len(active) else np.nan,
            'counterfactual_delta': float(g.counterfactual_delta_usdt.sum()),
            'capital_profit_proxy': float(cap_pnl), 'capital_max_realised_dd': float(cap_dd), 'skipped_due_cash': int(cash_skip),
        })
        for y, gy in g.groupby(g.entry_time.dt.year):
            years.append({'variant': v, 'year': int(y), 'pnl_usdt': float(gy.pnl_usdt.sum()),
                          'counterfactual_delta': float(gy.counterfactual_delta_usdt.sum()),
                          'bear_interventions': int(gy.bear_event_time.notna().sum())})
    return pd.DataFrame(rows), pd.DataFrame(years)


def main():
    cfg = json.loads(Path('research/config.json').read_text())
    out = Path('research/results/round4d'); out.mkdir(parents=True, exist_ok=True)
    btc = load_symbol('BTCUSDT', cfg['interval'], cfg['start'], cfg['end'])
    bear_daily = build_bear_state(btc)
    all_s, all_t = [], []
    with ThreadPoolExecutor(max_workers=6) as ex:
        fut = {ex.submit(process_symbol, s, btc, bear_daily, cfg): s for s in cfg['symbols']}
        for f in as_completed(fut):
            s = fut[f]
            try:
                sig, rows = f.result(); all_s += sig; all_t += rows; print(s, len(sig), flush=True)
            except Exception as e:
                print('ERROR', s, repr(e), flush=True)
    signals = pd.DataFrame(all_s); trades = pd.DataFrame(all_t)
    sm, yr = summarise(trades)
    signals.to_csv(out/'signals.csv', index=False); trades.to_csv(out/'trades.csv', index=False)
    sm.to_csv(out/'summary.csv', index=False); yr.to_csv(out/'year_summary.csv', index=False); bear_daily.rename('bear').to_csv(out/'bear_state.csv')
    manifest = {
        'study': 'Round 4D open-position response to confirmed bear regime',
        'status': 'RESEARCH ONLY - does not alter live strategy',
        'frozen_regime': 'BTC below 200D MA AND 200D MA falling versus 20 completed days earlier; state shifted one full day before use.',
        'entry_rule': 'All variants use the same frozen bear regime as a new-entry gate.',
        'base_exit': 'Frozen Round 3H FIXED60 accumulated-profit giveback architecture.',
        'variants': {
            'ENTRY_GATE_ONLY': 'No action on positions already open when bear confirms.',
            'FULL_LIQUIDATION': 'Exit 100% at first hourly bar open after bear confirmation.',
            'PROFIT_ONLY_LIQUIDATION': 'At bear confirmation, exit 100% only if bar open is above entry; otherwise keep normal exit.',
            'PARTIAL_50': 'Exit 50% at bear-confirmation bar open; remaining 50% keeps normal exit.',
            'TIGHTEN_5PCT': 'At bear confirmation, ratchet stop to max(existing strategy stop, 5% below confirmation bar open). One fixed probe, not optimised.',
            'STAGE_AWARE': 'INITIAL: full exit; BREAKEVEN stage: tighten to 5% below confirmation open; 30%+ trail stage: leave frozen trail unchanged.'
        },
        'event_execution': 'Bear confirmation action occurs at the first hourly bar open of the causal bear day, avoiding use of that bar high/low to decide the action.',
        'counterfactual': 'Each intervention trade is compared against ENTRY_GATE_ONLY for the same signal.',
        'capital_proxy': '1750 USDT starting cash, 300 USDT per trade, entry refused without free cash; drawdown is realised/cost-basis proxy, not exact mark-to-market.',
        'warning': 'TIGHTEN_5PCT and STAGE_AWARE are deliberately single fixed probes. If either family wins, parameters should be tested separately rather than tuned in this round.'
    }
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('\nSUMMARY\n', sm.sort_values('combined_pnl', ascending=False).to_string(index=False))
    print('\nYEARS\n', yr.to_string(index=False))

if __name__ == '__main__': main()
