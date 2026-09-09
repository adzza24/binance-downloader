from __future__ import annotations

import json, math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from binance_data import load_symbol
from round2b_validation import add_live_features, controlled_activity
from round3a_exit_architecture import capped_stop, net_pnl
from round4g_coin_regime_actions import daily_state

START_WALLET = 2000.0
POSITION_USDT = 300.0
BASE_GIVEBACK = 0.60
VARIANTS = ['NO_GATE', 'BTC200_GATE', 'COIN200_GATE']


def pf(vals):
    vals = np.asarray(vals, float)
    pos = vals[vals > 0]
    neg = vals[vals < 0]
    return float(pos.sum() / abs(neg.sum())) if len(neg) and neg.sum() else math.inf


def trail_stop(entry, peak, close):
    pg = peak / entry - 1.0
    if pg < 0.30:
        return None
    protected_gain = pg * (1.0 - BASE_GIVEBACK)
    return min(entry * (1.0 + protected_gain), close * 0.999)


def simulate_base_exit(df, sig, cfg):
    start = int(sig['entry_index'])
    entry = float(sig['entry_price'])
    init_stop = capped_stop(entry, float(sig['structural_stop']), 0.10)
    stop = init_stop
    activated = False
    peak = entry
    reached30 = False

    for j in range(start, len(df)):
        b = df.iloc[j]
        low, high, close = map(float, (b.low, b.high, b.close))
        peak = max(peak, high)
        pg = peak / entry - 1.0
        reached30 = reached30 or pg >= 0.30

        if not activated:
            if low <= init_stop:
                return b.time, 'STOP', net_pnl(entry, [(1.0, init_stop)], cfg), init_stop, reached30
            if high >= entry * 1.05:
                activated = True
                stop = max(stop, entry)
                continue
        else:
            if low <= stop:
                reason = 'GAIN_TRAIL_STOP' if reached30 else 'BREAKEVEN_STOP'
                return b.time, reason, net_pnl(entry, [(1.0, stop)], cfg), stop, reached30
            if high >= entry * 11.0:
                px = entry * 11.0
                return b.time, 'TAKE_PROFIT', net_pnl(entry, [(1.0, px)], cfg), px, reached30
            if pg >= 0.30:
                stop = max(stop, entry * 1.10)
            cand = trail_stop(entry, peak, close)
            if cand is not None:
                stop = max(stop, cand)

    px = float(df.iloc[-1].close)
    return df.iloc[-1].time, 'DATA_END_OPEN', net_pnl(entry, [(1.0, px)], cfg), px, reached30


def process_symbol(symbol, btc, btc200, cfg):
    df = btc.copy() if symbol == 'BTCUSDT' else load_symbol(symbol, cfg['interval'], cfg['start'], cfg['end'])
    if len(df) < 800:
        return [], [], None
    coin200 = daily_state(df, 200)
    x = add_live_features(df, btc)
    sigs = controlled_activity(symbol, x, cfg)
    rows = []
    for sig in sigs:
        et = pd.Timestamp(sig['entry_time']).tz_convert('UTC')
        entry_day = et.floor('D')
        exit_time, exit_reason, pnl, exit_price, reached30 = simulate_base_exit(df, sig, cfg)
        rows.append({
            'signal_id': sig['signal_id'], 'symbol': symbol,
            'entry_time': et, 'entry_price': float(sig['entry_price']),
            'exit_time': pd.Timestamp(exit_time).tz_convert('UTC'),
            'exit_reason': exit_reason, 'exit_price': float(exit_price),
            'pnl_usdt': float(pnl), 'reached_30': bool(reached30),
            'btc200_blocked': bool(btc200.get(entry_day, False)),
            'coin200_blocked': bool(coin200.get(entry_day, False)),
        })
    closes = df[['time', 'close']].copy()
    closes['time'] = pd.to_datetime(closes.time, utc=True)
    closes = closes.drop_duplicates('time').set_index('time').close.astype(float).sort_index()
    return sigs, rows, closes


def gate_allowed(r, variant):
    if variant == 'NO_GATE':
        return True
    if variant == 'BTC200_GATE':
        return not bool(r.btc200_blocked)
    if variant == 'COIN200_GATE':
        return not bool(r.coin200_blocked)
    raise ValueError(variant)


def independent_summary(base):
    rows = []
    years = []
    for variant in VARIANTS:
        g = base[base.apply(lambda r: gate_allowed(r, variant), axis=1)].copy()
        vals = g.pnl_usdt.to_numpy(float)
        rows.append({
            'variant': variant,
            'signals_total': int(len(base)),
            'active_trades': int(len(g)),
            'gate_skips': int(len(base) - len(g)),
            'combined_pnl': float(vals.sum()),
            'profit_factor': pf(vals),
            'win_rate': float((g.pnl_usdt > 0).mean()) if len(g) else np.nan,
            'take_profit_1000_count': int((g.exit_reason == 'TAKE_PROFIT').sum()),
        })
        gy = g.copy()
        gy['year'] = gy.entry_time.dt.year
        for year, y in gy.groupby('year'):
            years.append({'variant': variant, 'year': int(year), 'pnl_usdt': float(y.pnl_usdt.sum()), 'trades': int(len(y))})
    return pd.DataFrame(rows), pd.DataFrame(years)


def liquidation_value(entry_price, close_price, cfg):
    qty = POSITION_USDT / entry_price
    px = close_price * (1.0 - float(cfg['slippage_rate']))
    proceeds = qty * px
    return proceeds * (1.0 - float(cfg['fee_rate']))


def portfolio_sim(base, closes_by_symbol, variant, cfg):
    candidates = base[base.apply(lambda r: gate_allowed(r, variant), axis=1)].copy()
    candidates = candidates.sort_values(['entry_time', 'symbol', 'signal_id']).reset_index(drop=True)
    if candidates.empty:
        return {}, pd.DataFrame(), pd.DataFrame()

    entry_fee = POSITION_USDT * float(cfg['fee_rate'])
    cash = START_WALLET
    open_pos = {}
    admitted = []
    skipped_cash = []

    entries_by_time = {}
    exits_by_time = {}
    for idx, r in candidates.iterrows():
        entries_by_time.setdefault(r.entry_time, []).append((idx, r))

    start = candidates.entry_time.min().floor('h')
    end = max(candidates.exit_time.max(), max(s.index.max() for s in closes_by_symbol.values())).floor('h')
    timeline = pd.date_range(start, end, freq='1h', tz='UTC')

    equity_rows = []
    peak_equity = START_WALLET
    max_dd_abs = 0.0
    max_dd_pct = 0.0
    max_dd_time = None
    peak_time = start
    underwater_start = None
    max_underwater_hours = 0.0
    exposure_sum = 0.0
    exposure_obs = 0
    max_open = 0

    for t in timeline:
        # Realise exits first so capital released this hour can fund same-hour signals.
        for key, p in list(open_pos.items()):
            if p['exit_time'] <= t:
                # net_pnl already includes the entry fee. Because entry fee was deducted
                # from cash at admission, add it back here alongside principal + net P&L
                # so final wallet change equals exactly net_pnl.
                cash += POSITION_USDT + float(p['pnl_usdt']) + entry_fee
                admitted[p['admitted_index']]['realised_wallet_after_exit'] = cash
                del open_pos[key]

        for idx, r in entries_by_time.get(t, []):
            required = POSITION_USDT + entry_fee
            if cash + 1e-9 < required:
                skipped_cash.append({
                    'variant': variant, 'signal_id': r.signal_id, 'symbol': r.symbol,
                    'entry_time': r.entry_time, 'wallet_cash': cash,
                    'required_cash': required, 'reason': 'INSUFFICIENT_CASH'
                })
                continue
            cash -= required
            rec = {
                'variant': variant, 'signal_id': r.signal_id, 'symbol': r.symbol,
                'entry_time': r.entry_time, 'entry_price': float(r.entry_price),
                'exit_time': r.exit_time, 'exit_reason': r.exit_reason,
                'exit_price': float(r.exit_price), 'pnl_usdt': float(r.pnl_usdt),
                'wallet_cash_after_entry': cash,
                'realised_wallet_after_exit': np.nan,
            }
            admitted.append(rec)
            ai = len(admitted) - 1
            open_pos[r.signal_id] = {**rec, 'admitted_index': ai}

        open_value = 0.0
        for p in open_pos.values():
            s = closes_by_symbol[p['symbol']]
            if t in s.index:
                close_px = float(s.loc[t])
            else:
                prior = s.loc[:t]
                close_px = float(prior.iloc[-1]) if len(prior) else float(p['entry_price'])
            open_value += liquidation_value(float(p['entry_price']), close_px, cfg)

        equity = cash + open_value
        n_open = len(open_pos)
        max_open = max(max_open, n_open)
        exposure = open_value / equity if equity > 0 else np.nan
        if np.isfinite(exposure):
            exposure_sum += exposure
            exposure_obs += 1

        if equity > peak_equity:
            peak_equity = equity
            peak_time = t
            underwater_start = None
        else:
            if underwater_start is None and equity < peak_equity:
                underwater_start = t
            if underwater_start is not None:
                max_underwater_hours = max(max_underwater_hours, (t - underwater_start).total_seconds() / 3600.0)

        dd_abs = equity - peak_equity
        dd_pct = dd_abs / peak_equity if peak_equity > 0 else np.nan
        if dd_abs < max_dd_abs:
            max_dd_abs = dd_abs
            max_dd_pct = dd_pct
            max_dd_time = t

        equity_rows.append({
            'variant': variant, 'time': t, 'cash': cash,
            'open_liquidation_value': open_value, 'equity': equity,
            'peak_equity': peak_equity, 'drawdown_usdt': dd_abs,
            'drawdown_pct': dd_pct, 'open_positions': n_open,
            'exposure_pct': exposure,
        })

    eq = pd.DataFrame(equity_rows)
    adm = pd.DataFrame(admitted)
    skips = pd.DataFrame(skipped_cash)

    # Rolling losses from hourly MTM equity.
    eqs = eq.set_index('time').equity
    rolling = {}
    for hours, label in [(24, '1d'), (24*7, '7d'), (24*30, '30d')]:
        prev = eqs.shift(hours)
        ret = eqs / prev - 1.0
        rolling[f'worst_{label}_return_pct'] = float(ret.min()) if ret.notna().any() else np.nan

    final_equity = float(eq.iloc[-1].equity)
    total_return = final_equity / START_WALLET - 1.0
    mtm_dd_pct = abs(max_dd_pct)
    summary = {
        'variant': variant,
        'starting_wallet': START_WALLET,
        'final_equity': final_equity,
        'portfolio_pnl': final_equity - START_WALLET,
        'total_return_pct': total_return,
        'admitted_trades': int(len(adm)),
        'gate_skips': int(len(base) - len(candidates)),
        'cash_capacity_skips': int(len(skips)),
        'max_open_positions': int(max_open),
        'average_exposure_pct': float(exposure_sum / exposure_obs) if exposure_obs else np.nan,
        'max_mtm_drawdown_usdt': float(max_dd_abs),
        'max_mtm_drawdown_pct': float(max_dd_pct),
        'max_drawdown_time': max_dd_time,
        'max_underwater_hours': float(max_underwater_hours),
        'return_to_maxdd': float(total_return / mtm_dd_pct) if mtm_dd_pct > 0 else math.inf,
        **rolling,
    }
    return summary, eq, adm, skips


def main():
    cfg = json.loads(Path('research/config.json').read_text())
    out = Path('research/results/round4k')
    out.mkdir(parents=True, exist_ok=True)

    btc = load_symbol('BTCUSDT', cfg['interval'], cfg['start'], cfg['end'])
    btc200 = daily_state(btc, 200)
    all_sigs, all_rows = [], []
    closes_by_symbol = {}

    with ThreadPoolExecutor(max_workers=6) as ex:
        fut = {ex.submit(process_symbol, s, btc, btc200, cfg): s for s in cfg['symbols']}
        for f in as_completed(fut):
            symbol = fut[f]
            try:
                sigs, rows, closes = f.result()
                all_sigs += sigs
                all_rows += rows
                if closes is not None:
                    closes_by_symbol[symbol] = closes
                print(symbol, len(sigs), flush=True)
            except Exception as e:
                print('ERROR', symbol, repr(e), flush=True)

    base = pd.DataFrame(all_rows)
    base['entry_time'] = pd.to_datetime(base.entry_time, utc=True)
    base['exit_time'] = pd.to_datetime(base.exit_time, utc=True)
    base = base.sort_values(['entry_time', 'symbol', 'signal_id']).reset_index(drop=True)
    pd.DataFrame(all_sigs).to_csv(out / 'signals.csv', index=False)
    base.to_csv(out / 'base_trades.csv', index=False)

    indep, years = independent_summary(base)
    indep.to_csv(out / 'independent_summary.csv', index=False)
    years.to_csv(out / 'independent_year_summary.csv', index=False)

    portfolio_summaries = []
    equity_curves = []
    admitted_all = []
    skips_all = []
    for v in VARIANTS:
        sm, eq, adm, skips = portfolio_sim(base, closes_by_symbol, v, cfg)
        portfolio_summaries.append(sm)
        equity_curves.append(eq)
        admitted_all.append(adm)
        if len(skips):
            skips_all.append(skips)

    ps = pd.DataFrame(portfolio_summaries)
    ps.to_csv(out / 'portfolio_mtm_summary.csv', index=False)
    pd.concat(equity_curves, ignore_index=True).to_csv(out / 'portfolio_equity_hourly.csv', index=False)
    pd.concat(admitted_all, ignore_index=True).to_csv(out / 'portfolio_admitted_trades.csv', index=False)
    if skips_all:
        pd.concat(skips_all, ignore_index=True).to_csv(out / 'portfolio_cash_skips.csv', index=False)
    else:
        pd.DataFrame(columns=['variant','signal_id','symbol','entry_time','wallet_cash','required_cash','reason']).to_csv(out / 'portfolio_cash_skips.csv', index=False)

    comp = ps.set_index('variant')
    rows = []
    for v in ['BTC200_GATE', 'COIN200_GATE']:
        r = comp.loc[v]
        rows.append({
            'variant': v,
            'vs_no_gate_portfolio_pnl': float(r.portfolio_pnl - comp.loc['NO_GATE'].portfolio_pnl),
            'vs_no_gate_maxdd_pct_points': float(r.max_mtm_drawdown_pct - comp.loc['NO_GATE'].max_mtm_drawdown_pct),
            'vs_no_gate_final_equity': float(r.final_equity - comp.loc['NO_GATE'].final_equity),
        })
    # direct candidate comparison, positive values favour coin200 where stated.
    coin = comp.loc['COIN200_GATE']; btcg = comp.loc['BTC200_GATE']
    rows.append({
        'variant': 'COIN200_MINUS_BTC200',
        'vs_no_gate_portfolio_pnl': float(coin.portfolio_pnl - btcg.portfolio_pnl),
        'vs_no_gate_maxdd_pct_points': float(coin.max_mtm_drawdown_pct - btcg.max_mtm_drawdown_pct),
        'vs_no_gate_final_equity': float(coin.final_equity - btcg.final_equity),
    })
    pd.DataFrame(rows).to_csv(out / 'gate_comparison.csv', index=False)

    manifest = {
        'study': 'Round 4K true mark-to-market validation of BTC200 versus coin200 entry regime gate',
        'status': 'RESEARCH ONLY - no live strategy changes',
        'variants': VARIANTS,
        'entry_gate': '200D AND deterioration: prior completed daily close below 200D MA AND 200D MA falling versus 20 completed days earlier.',
        'base_entry': 'Frozen Controlled Activity signal family from Round 2B.',
        'base_exit': 'Frozen current architecture: structural stop capped ~-10%, +5% breakeven, +30% +10% floor plus 60% accumulated-profit giveback trail, +1000% take-profit.',
        'independent_view': 'Fixed 300 USDT per signal, overlaps unconstrained, retained for continuity and pure gate comparison.',
        'portfolio_view': 'One starting wallet of 2000 USDT. Each admitted trade uses fixed 300 USDT plus entry fee. Realised gains/losses permanently change future available cash. Signals are chronological first-come-first-served and skipped if cash is insufficient. Capital is never reset.',
        'mtm': 'Hourly equity = cash + current liquidation value of every open position. Liquidation value applies configured slippage and exit fee. Entry fee is deducted at admission. Actual exits reconcile to the existing net_pnl result.',
        'costs': {'fee_rate': cfg['fee_rate'], 'slippage_rate': cfg['slippage_rate']},
        'decision_question': 'Does coin200 produce a materially better true MTM risk profile than BTC200 while sacrificing little enough return to justify preferring coin-specific gating?'
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))

    print('\nINDEPENDENT SUMMARY\n', indep.to_string(index=False))
    print('\nPORTFOLIO MTM SUMMARY\n', ps.to_string(index=False))
    print('\nGATE COMPARISON\n', pd.read_csv(out / 'gate_comparison.csv').to_string(index=False))


if __name__ == '__main__':
    main()
