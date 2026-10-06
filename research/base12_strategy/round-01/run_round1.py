from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from binance_data import CACHE_ROOT, COLS, normalise_ts
import explosive_move_discovery_round1 as disc
import explosive_move_real_world_replay_round8 as r8

ROOT = Path('research/base12_strategy/round-01')
ROUND_TRIP_COST = 0.003
TARGET = 0.10
STOP = 0.02
HORIZON_H = 720
MIN_SEGMENT_ROWS = 3000
GAP_LIMIT = pd.Timedelta(minutes=90)
SLEEVES = [1, 2, 3]


def valid_symbol(symbol: str) -> bool:
    if not symbol.endswith('USDT'):
        return False
    base = symbol[:-4]
    if base in disc.STABLE_BASES or symbol in disc.EXCL:
        return False
    return re.search(r'(UP|DOWN|BULL|BEAR)USDT$', symbol) is None


def cache_inventory() -> pd.DataFrame:
    rows = []
    if not CACHE_ROOT.exists():
        return pd.DataFrame(columns=['symbol', 'files'])
    for d in sorted(CACHE_ROOT.iterdir()):
        if not d.is_dir() or not valid_symbol(d.name):
            continue
        files = sorted((d / '1h').glob(f'{d.name}-1h-*.zip'))
        months = []
        for f in files:
            m = re.search(r'(\d{4}-\d{2})\.zip$', f.name)
            if m:
                months.append(m.group(1))
        if months:
            rows.append({'symbol': d.name, 'files': len(files), 'first_cached_month': min(months), 'last_cached_month': max(months)})
    return pd.DataFrame(rows)


def read_cached_symbol(symbol: str) -> pd.DataFrame:
    files = sorted((CACHE_ROOT / symbol / '1h').glob(f'{symbol}-1h-*.zip'))
    frames = []
    bad_files = 0
    for f in files:
        try:
            with zipfile.ZipFile(io.BytesIO(f.read_bytes())) as zf:
                names = zf.namelist()
                if not names:
                    bad_files += 1
                    continue
                with zf.open(names[0]) as fh:
                    q = pd.read_csv(fh, header=None, names=COLS)
            if not q.empty:
                frames.append(q)
        except Exception:
            bad_files += 1
    if not frames:
        return pd.DataFrame(columns=COLS + ['time'])
    df = pd.concat(frames, ignore_index=True)
    df['time'] = normalise_ts(df['open_time'])
    for c in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_base']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['time', 'open', 'high', 'low', 'close']).sort_values('time').drop_duplicates('time').reset_index(drop=True)
    df.attrs['bad_files'] = bad_files
    return df


def split_contiguous(df: pd.DataFrame):
    if df.empty:
        return []
    gaps = df['time'].diff()
    seg_id = (gaps.isna() | (gaps > GAP_LIMIT)).cumsum()
    return [g.reset_index(drop=True) for _, g in df.groupby(seg_id) if len(g)]


def coverage_row(symbol: str, df: pd.DataFrame, invrow) -> dict:
    gaps = df['time'].diff() if len(df) else pd.Series(dtype='timedelta64[ns]')
    segments = split_contiguous(df)
    usable = [g for g in segments if len(g) >= MIN_SEGMENT_ROWS]
    years = sorted(set(int(y) for y in pd.to_datetime(df['time'], utc=True).dt.year)) if len(df) else []
    usable_years = sorted(set(int(y) for g in usable for y in pd.to_datetime(g['time'], utc=True).dt.year))
    return {
        'symbol': symbol, 'cached_files': int(invrow.files), 'first_cached_month': invrow.first_cached_month,
        'last_cached_month': invrow.last_cached_month, 'rows': len(df),
        'first_time': df.time.min() if len(df) else pd.NaT, 'last_time': df.time.max() if len(df) else pd.NaT,
        'large_gaps': int((gaps > GAP_LIMIT).sum()) if len(df) else 0, 'segments': len(segments),
        'usable_segments': len(usable), 'calendar_years_present': ','.join(map(str, years)),
        'usable_years_present': ','.join(map(str, usable_years)), 'bad_cache_files': int(df.attrs.get('bad_files', 0)),
    }


def first_touch_trade(df: pd.DataFrame, signal_idx: int) -> dict:
    high = df['high'].to_numpy(float); low = df['low'].to_numpy(float); close = df['close'].to_numpy(float)
    entry = float(close[signal_idx]); avail = len(df) - signal_idx - 1; end_idx = min(len(df) - 1, signal_idx + HORIZON_H)
    if avail <= 0:
        return {'mature_30d': False, 'trade_result': 'UNRESOLVED', 'exit_time': pd.NaT, 'holding_hours': np.nan, 'gross_return': np.nan, 'net_return': np.nan}
    result = None; exit_idx = None
    for k in range(signal_idx + 1, end_idx + 1):
        hs = low[k] <= entry * (1 - STOP); ht = high[k] >= entry * (1 + TARGET)
        if hs:
            result, exit_idx = 'STOP', k; break
        if ht:
            result, exit_idx = 'TARGET', k; break
    mature = avail >= HORIZON_H
    if result == 'TARGET': gross = TARGET
    elif result == 'STOP': gross = -STOP
    elif mature:
        result = 'TIME_EXIT'; exit_idx = signal_idx + HORIZON_H; gross = float(close[exit_idx] / entry - 1)
    else:
        return {'mature_30d': False, 'trade_result': 'UNRESOLVED', 'exit_time': pd.NaT, 'holding_hours': np.nan, 'gross_return': np.nan, 'net_return': np.nan}
    return {'mature_30d': bool(mature), 'trade_result': result, 'exit_time': pd.Timestamp(df.iloc[exit_idx].time),
            'holding_hours': int(exit_idx - signal_idx), 'gross_return': float(gross), 'net_return': float(gross - ROUND_TRIP_COST)}


def generate_segment_signals(symbol: str, seg: pd.DataFrame, btc: pd.DataFrame, models, thresholds, r5rules, r7rules, segment_id: int):
    r8.START_REPLAY = pd.Timestamp('1900-01-01T00:00:00Z')
    alerts, ff = r8.build_base_alerts(seg, btc, models, thresholds)
    if alerts.empty:
        return []
    alerts['symbol'] = symbol
    selected = r8.apply_round5_filter(alerts, ff, r5rules)
    episodes = r8.build_episodes(selected)
    if episodes.empty:
        return []
    btc_aligned = btc.set_index('time')['close'].reindex(pd.DatetimeIndex(seg['time'])).to_numpy(float)
    rows = []
    for e in episodes.itertuples(index=False):
        if str(e.family) != 'BASE':
            continue
        i = int(e.start_idx); first_checkpoint = None; first_vals = None; first_j = None
        for h in [3, 6, 12]:
            j = i + h
            if j >= len(seg):
                break
            vals = r8.evolution_values(i, j, e.family, seg, ff, btc_aligned)
            if vals['mfe_since_open'] > 0.05:
                continue
            if r8.checkpoint_pass(vals, e.family, r7rules[h]):
                first_checkpoint, first_vals, first_j = h, vals, j
                break
        if first_checkpoint != 12:
            continue
        rows.append({
            'symbol': symbol, 'segment_id': segment_id, 'candidate_start_time': pd.Timestamp(e.start_time),
            'candidate_family': str(e.family), 'precursor_alerts': int(e.precursor_alerts), 'signal_checkpoint_h': 12,
            'signal_time': pd.Timestamp(seg.iloc[first_j].time), 'entry_price': float(seg.iloc[first_j].close),
            'candidate_open_to_signal_close': float(seg.iloc[first_j].close / seg.iloc[i].close - 1),
            'candidate_mfe_to_signal': float(first_vals['mfe_since_open']), 'candidate_mae_to_signal': float(first_vals['mae_since_open']),
            **first_touch_trade(seg, first_j),
        })
    return rows


def concurrency_stats(trades: pd.DataFrame):
    t = trades[trades.trade_result != 'UNRESOLVED'].copy()
    if t.empty:
        return pd.DataFrame(), pd.DataFrame()
    t['signal_time'] = pd.to_datetime(t.signal_time, utc=True); t['exit_time'] = pd.to_datetime(t.exit_time, utc=True)
    active = []; rows = []
    for r in t.sort_values(['signal_time', 'symbol']).itertuples(index=False):
        active = [x for x in active if x > r.signal_time]
        before = len(active); active.append(r.exit_time)
        rows.append({'symbol': r.symbol, 'signal_time': r.signal_time, 'year': r.signal_time.year,
                     'concurrent_before_entry': before, 'concurrent_after_entry': len(active)})
    pt = pd.DataFrame(rows); summary = []
    for key, g in list(pt.groupby('year')) + [('ALL', pt)]:
        summary.append({'year': key, 'trades': len(g), 'overlap_trade_rate': float((g.concurrent_before_entry > 0).mean()),
                        'max_concurrency_observed': int(g.concurrent_after_entry.max()),
                        'median_concurrent_before_entry': float(g.concurrent_before_entry.median())})
    return pd.DataFrame(summary), pt


def simulate_year(year, g: pd.DataFrame, n_sleeves: int):
    initial = 1000.0; equities = [initial / n_sleeves] * n_sleeves
    free_at = [pd.Timestamp.min.tz_localize('UTC')] * n_sleeves; ledger = []; curve = [(pd.Timestamp(f'{int(year)}-01-01', tz='UTC'), initial)]
    for r in g.sort_values(['signal_time', 'symbol'], kind='stable').itertuples(index=False):
        st = pd.Timestamp(r.signal_time); et = pd.Timestamp(r.exit_time)
        if pd.isna(et) or not np.isfinite(r.net_return):
            ledger.append({'year': year, 'sleeves': n_sleeves, 'symbol': r.symbol, 'signal_time': st, 'exit_time': et,
                           'accepted': False, 'skip_reason': 'unresolved', 'sleeve': np.nan, 'capital_before': np.nan, 'capital_after': np.nan, 'net_return': r.net_return}); continue
        candidates = [i for i, ft in enumerate(free_at) if ft <= st]
        if not candidates:
            ledger.append({'year': year, 'sleeves': n_sleeves, 'symbol': r.symbol, 'signal_time': st, 'exit_time': et,
                           'accepted': False, 'skip_reason': 'capacity', 'sleeve': np.nan, 'capital_before': np.nan, 'capital_after': np.nan, 'net_return': r.net_return}); continue
        i = min(candidates); before = equities[i]; after = before * (1 + float(r.net_return)); equities[i] = after; free_at[i] = et
        ledger.append({'year': year, 'sleeves': n_sleeves, 'symbol': r.symbol, 'signal_time': st, 'exit_time': et,
                       'accepted': True, 'skip_reason': '', 'sleeve': i + 1, 'capital_before': before, 'capital_after': after, 'net_return': r.net_return})
        curve.append((et, float(sum(equities))))
    led = pd.DataFrame(ledger); c = pd.DataFrame(curve, columns=['time', 'equity']).sort_values('time'); peak = c.equity.cummax(); dd = c.equity / peak - 1
    ending = float(sum(equities))
    return {'year': int(year), 'sleeves': n_sleeves, 'starting_equity': initial, 'ending_equity': ending,
            'annual_return': ending / initial - 1, 'accepted_trades': int(led.accepted.sum()) if len(led) else 0,
            'capacity_skipped_trades': int((led.skip_reason == 'capacity').sum()) if len(led) else 0,
            'max_realised_drawdown': float(dd.min()) if len(dd) else 0.0}, led


def portfolio_simulations(trades: pd.DataFrame):
    mature = trades[trades.trade_result != 'UNRESOLVED'].copy(); mature['signal_time'] = pd.to_datetime(mature.signal_time, utc=True); mature['exit_time'] = pd.to_datetime(mature.exit_time, utc=True); mature['entry_year'] = mature.signal_time.dt.year
    summaries = []; ledgers = []
    for year, g in mature.groupby('entry_year'):
        for n in SLEEVES:
            s, l = simulate_year(year, g, n); summaries.append(s)
            if len(l): ledgers.append(l)
    return pd.DataFrame(summaries), pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()


def year_summary(trades: pd.DataFrame):
    z = trades.copy(); z['signal_time'] = pd.to_datetime(z.signal_time, utc=True); z['year'] = z.signal_time.dt.year; z['week'] = z.signal_time.dt.strftime('%G-W%V')
    rows = []
    for key, g in list(z.groupby('year')) + [('ALL', z)]:
        r = g[g.trade_result != 'UNRESOLVED']; net = pd.to_numeric(r.net_return, errors='coerce').dropna()
        rows.append({'year': key, 'signals': len(g), 'resolved_trades': len(r), 'symbols': g.symbol.nunique(), 'distinct_weeks': g.week.nunique(),
                     'targets': int((r.trade_result == 'TARGET').sum()), 'stops': int((r.trade_result == 'STOP').sum()), 'time_exits': int((r.trade_result == 'TIME_EXIT').sum()),
                     'unresolved': int((g.trade_result == 'UNRESOLVED').sum()), 'target_rate': float((r.trade_result == 'TARGET').mean()) if len(r) else np.nan,
                     'avg_net_return': float(net.mean()) if len(net) else np.nan, 'median_net_return': float(net.median()) if len(net) else np.nan,
                     'p25_net_return': float(net.quantile(.25)) if len(net) else np.nan, 'p75_net_return': float(net.quantile(.75)) if len(net) else np.nan,
                     'median_holding_hours': float(pd.to_numeric(r.holding_hours, errors='coerce').median()) if len(r) else np.nan})
    return pd.DataFrame(rows)


def symbol_summary(trades: pd.DataFrame):
    z = trades[trades.trade_result != 'UNRESOLVED'].copy()
    if z.empty: return pd.DataFrame()
    return z.groupby('symbol', as_index=False).agg(trades=('symbol', 'size'), target_rate=('trade_result', lambda s: float((s == 'TARGET').mean())),
        avg_net_return=('net_return', 'mean'), median_net_return=('net_return', 'median'), total_net_return_points=('net_return', 'sum'),
        first_signal=('signal_time', 'min'), last_signal=('signal_time', 'max')).sort_values(['total_net_return_points', 'trades'], ascending=[False, False])


def write_analysis(years, overlap, portfolios, coverage, failures):
    lines = ['# BASE +12h Strategy Research - Round 01 Analysis', '', '**Research only. No live strategy, portfolio, log or automation changes.**', '',
             '## Frozen strategy', '', '- BASE candidate family.', '- First accepted checkpoint exactly +12h.', '- Enter at signal close.',
             '- OCO +10% target / -2% stop.', f'- {ROUND_TRIP_COST*100:.2f}% round-trip cost.', '- Stop-first on same 1h candle ties.',
             '- 30-day maximum research holding period; positions still open at 30 days are marked to market and closed.', '',
             '## Universe and coverage', '', f'- Cached USDT symbols inventoried: **{len(coverage):,}**.',
             f'- Symbols with at least one usable continuous segment: **{int((coverage.usable_segments > 0).sum()) if len(coverage) else 0:,}**.',
             f'- Processing failures: **{len(failures):,}**.', '', '## By year', '',
             '| Year | Signals | Trades | Symbols | Weeks | Target rate | Avg net | Median net | Median hold |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in years.itertuples(index=False):
        pc = lambda v: 'n/a' if pd.isna(v) else f'{100*v:.2f}%'
        lines.append(f'| {r.year} | {int(r.signals)} | {int(r.resolved_trades)} | {int(r.symbols)} | {int(r.distinct_weeks)} | {pc(r.target_rate)} | {pc(r.avg_net_return)} | {pc(r.median_net_return)} | {r.median_holding_hours:.1f}h |')
    lines += ['', '## Overlap', '']
    if len(overlap):
        a = overlap[overlap.year.astype(str) == 'ALL'].iloc[0]
        lines += [f'- Trades overlapping an already-open trade: **{100*a.overlap_trade_rate:.1f}%**.', f'- Maximum observed simultaneous open trades: **{int(a.max_concurrency_observed)}**.', '']
    lines += ['## Annual $1,000 portfolio simulations', '', 'Each year resets to $1,000. Sleeve capital compounds within that year. Signals that arrive while every sleeve is occupied are skipped.', '',
              '| Year | Sleeves | Ending equity | Annual return | Accepted | Capacity skipped | Max realised DD |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in portfolios.itertuples(index=False):
        lines.append(f'| {r.year} | {r.sleeves} | ${r.ending_equity:,.2f} | {100*r.annual_return:.2f}% | {r.accepted_trades} | {r.capacity_skipped_trades} | {100*r.max_realised_drawdown:.2f}% |')
    lines += ['', '## Interpretation guardrails', '', '- This is a broad cached-universe robustness replay, not a pristine holdout.',
              '- The cache defines the universe, so symbol/listing survivorship and incomplete historical coverage remain limitations.',
              '- Portfolio results are capacity models, not proof of executable live performance.', '- No result from this round is promoted to the live strategy automatically.']
    (ROOT / 'ANALYSIS.md').write_text('\n'.join(lines) + '\n')


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    inv = cache_inventory()
    if inv.empty: raise RuntimeError(f'No cached 1h symbols found at {CACHE_ROOT}')
    btc = read_cached_symbol('BTCUSDT')
    if btc.empty: raise RuntimeError('BTCUSDT cached 1h history is required')
    models, thresholds, _ = r8.train_frozen_detector(); r5rules, r7rules = r8.load_frozen_rules()
    all_signals = []; coverage = []; failures = []
    for invrow in inv.itertuples(index=False):
        symbol = invrow.symbol
        try:
            df = read_cached_symbol(symbol); coverage.append(coverage_row(symbol, df, invrow)); segments = split_contiguous(df); symbol_rows = []
            for sid, seg in enumerate(segments, start=1):
                if len(seg) < MIN_SEGMENT_ROWS: continue
                try: symbol_rows.extend(generate_segment_signals(symbol, seg, btc, models, thresholds, r5rules, r7rules, sid))
                except Exception as e: failures.append({'symbol': symbol, 'segment_id': sid, 'error': repr(e)})
            all_signals.extend(symbol_rows); print(symbol, 'rows', len(df), 'segments', len(segments), 'signals', len(symbol_rows), flush=True)
        except Exception as e:
            failures.append({'symbol': symbol, 'segment_id': '', 'error': repr(e)}); print('ERROR', symbol, repr(e), flush=True)
    cov = pd.DataFrame(coverage); fail = pd.DataFrame(failures); sig = pd.DataFrame(all_signals)
    cov.to_csv(ROOT / 'data_coverage.csv', index=False); fail.to_csv(ROOT / 'failures.csv', index=False)
    if sig.empty:
        sig.to_csv(ROOT / 'signals.csv.gz', index=False, compression='gzip'); raise RuntimeError('No BASE +12h signals found')
    sig = sig.sort_values(['signal_time', 'symbol']).reset_index(drop=True); sig.to_csv(ROOT / 'signals.csv.gz', index=False, compression='gzip'); sig.to_csv(ROOT / 'trades.csv.gz', index=False, compression='gzip')
    ys = year_summary(sig); osum, otrades = concurrency_stats(sig); psum, pledger = portfolio_simulations(sig); ssum = symbol_summary(sig)
    ys.to_csv(ROOT / 'year_summary.csv', index=False); osum.to_csv(ROOT / 'overlap_summary.csv', index=False); otrades.to_csv(ROOT / 'overlap_trades.csv.gz', index=False, compression='gzip')
    psum.to_csv(ROOT / 'portfolio_years.csv', index=False); pledger.to_csv(ROOT / 'portfolio_trades.csv.gz', index=False, compression='gzip'); ssum.to_csv(ROOT / 'symbol_summary.csv', index=False)
    write_analysis(ys, osum, psum, cov, fail); print((ROOT / 'ANALYSIS.md').read_text(), flush=True)


if __name__ == '__main__':
    main()
