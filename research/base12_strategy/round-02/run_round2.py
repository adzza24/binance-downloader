from __future__ import annotations

import argparse
import io
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from binance_data import CACHE_ROOT, COLS, normalise_ts
import explosive_move_discovery_round1 as disc
import explosive_move_real_world_replay_round8 as r8

ROOT = Path('research/base12_strategy/round-02')
S3_ENDPOINT = 'https://s3-ap-northeast-1.amazonaws.com/data.binance.vision'
VISION_BASE = 'https://data.binance.vision'
ROUND_TRIP_COST = 0.003
TARGET = 0.10
STOP = 0.02
HORIZON_H = 720
MIN_SEGMENT_ROWS = 3000
GAP_LIMIT = pd.Timedelta(minutes=90)
SLEEVES = [1, 2, 3]
ROUND1_AVG_NET = 0.035
ROUND1_TRADES = 60

STABLE_OR_FIAT_BASES = set(disc.STABLE_BASES)
TOKENISED_SYMBOLS = {
    'SPCXBUSDT','MSTRBUSDT','SNDKBUSDT','CRCLUSDT','NVDABUSDT','SKHYUSDT',
    'SOXLBUSDT','SNXXUSDT','INTCBUSDT','TSLABUSDT','MUBUSDT','CBRSUSDT',
    'BNCUSDT','QQQBUSDT','REUSDT','WDCBUSDT','GOOGLBUSDT','KORUUSDT',
    'SOXSBUSDT','SPYBUSDT','AVGOBUSDT','COINBUSDT','MSFTBUSDT',
}
LEVERAGED_RE = re.compile(r'(UP|DOWN|BULL|BEAR)USDT$')
MONTH_RE = re.compile(r'-(\d{4}-\d{2})\.zip$')


def get_bytes(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={'User-Agent': 'base12-round2-validation/1.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def get_json(url: str, timeout: int = 45):
    return json.loads(get_bytes(url, timeout=timeout).decode())


def s3_list(prefix: str, delimiter: str | None = None):
    token = None
    keys, prefixes = [], []
    while True:
        params = {'list-type': '2', 'prefix': prefix, 'max-keys': '1000'}
        if delimiter:
            params['delimiter'] = delimiter
        if token:
            params['continuation-token'] = token
        url = S3_ENDPOINT + '?' + urllib.parse.urlencode(params)
        root = ET.fromstring(get_bytes(url, timeout=90))
        keys.extend([x.text for x in root.findall('.//{*}Contents/{*}Key') if x.text])
        prefixes.extend([x.text for x in root.findall('.//{*}CommonPrefixes/{*}Prefix') if x.text])
        truncated = (root.findtext('.//{*}IsTruncated') or '').lower() == 'true'
        if not truncated:
            break
        token = root.findtext('.//{*}NextContinuationToken')
        if not token:
            break
    return keys, prefixes


def universe_exclusion_reason(symbol: str) -> str | None:
    if not symbol.endswith('USDT'):
        return 'not_usdt'
    base = symbol[:-4]
    if base in STABLE_OR_FIAT_BASES:
        return 'stable_or_fiat_base'
    if LEVERAGED_RE.search(symbol):
        return 'leveraged_token_name'
    if symbol in TOKENISED_SYMBOLS:
        return 'tokenised_equity_or_etf'
    return None


def list_symbol_months(symbol: str):
    prefix = f'data/spot/monthly/klines/{symbol}/1h/'
    keys, _ = s3_list(prefix)
    months = []
    for key in keys:
        m = MONTH_RE.search(key)
        if m:
            months.append(m.group(1))
    return sorted(set(months))


def current_spot_usdt_symbols():
    try:
        info = get_json('https://data-api.binance.vision/api/v3/exchangeInfo')
        return {
            s.get('symbol') for s in info.get('symbols', [])
            if s.get('quoteAsset') == 'USDT'
            and s.get('status') == 'TRADING'
            and s.get('isSpotTradingAllowed', True)
        }
    except Exception:
        return set()


def discover_universe():
    ROOT.mkdir(parents=True, exist_ok=True)
    _, prefixes = s3_list('data/spot/monthly/klines/', delimiter='/')
    raw_symbols = sorted({
        p.rstrip('/').split('/')[-1]
        for p in prefixes
        if p.rstrip('/').split('/')[-1]
    })
    excluded = []
    candidates = []
    for symbol in raw_symbols:
        reason = universe_exclusion_reason(symbol)
        if reason:
            excluded.append({'symbol': symbol, 'reason': reason})
        else:
            candidates.append(symbol)

    current = current_spot_usdt_symbols()
    rows, failures = [], []

    def work(symbol):
        months = list_symbol_months(symbol)
        return symbol, months

    with ThreadPoolExecutor(max_workers=20) as ex:
        futs = {ex.submit(work, s): s for s in candidates}
        for fut in as_completed(futs):
            symbol = futs[fut]
            try:
                _, months = fut.result()
                if not months:
                    excluded.append({'symbol': symbol, 'reason': 'no_monthly_1h_archives'})
                    continue
                rows.append({
                    'symbol': symbol,
                    'first_archive_month': months[0],
                    'last_archive_month': months[-1],
                    'archive_months': len(months),
                    'months': ';'.join(months),
                    'currently_listed_spot_usdt': symbol in current if current else np.nan,
                })
                print('DISCOVER', symbol, len(months), months[0], months[-1], flush=True)
            except Exception as e:
                failures.append({'symbol': symbol, 'stage': 'discover', 'error': repr(e)})
                print('DISCOVER_ERROR', symbol, repr(e), flush=True)

    universe = pd.DataFrame(rows).sort_values('symbol').reset_index(drop=True)
    pd.DataFrame(excluded).sort_values(['reason', 'symbol']).to_csv(ROOT / 'excluded_universe.csv', index=False)
    universe.to_csv(ROOT / 'historical_universe.csv', index=False)
    pd.DataFrame(failures).to_csv(ROOT / 'discovery_failures.csv', index=False)
    print('DISCOVERY_SUMMARY', 'raw', len(raw_symbols), 'included', len(universe), 'excluded', len(excluded), 'failures', len(failures), flush=True)
    if universe.empty:
        raise RuntimeError('Historical universe discovery returned no usable USDT symbols')


def archive_key(symbol: str, month: str):
    return f'data/spot/monthly/klines/{symbol}/1h/{symbol}-1h-{month}.zip'


def fetch_archive_month(symbol: str, month: str):
    cache = CACHE_ROOT / symbol / '1h' / f'{symbol}-1h-{month}.zip'
    raw = None
    if cache.exists():
        raw = cache.read_bytes()
    else:
        key = archive_key(symbol, month)
        try:
            raw = get_bytes(f'{VISION_BASE}/{key}', timeout=90)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(raw)
        except Exception:
            return None
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names = zf.namelist()
            if not names:
                raise zipfile.BadZipFile('empty archive')
            with zf.open(names[0]) as fh:
                return pd.read_csv(fh, header=None, names=COLS)
    except Exception:
        cache.unlink(missing_ok=True)
        return None


def load_archive_symbol(symbol: str, months: list[str]):
    frames = []
    missing = []
    for month in months:
        q = fetch_archive_month(symbol, month)
        if q is None or q.empty:
            missing.append(month)
        else:
            frames.append(q)
    if not frames:
        return pd.DataFrame(columns=COLS + ['time']), missing
    df = pd.concat(frames, ignore_index=True)
    df['time'] = normalise_ts(df['open_time'])
    for c in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_base']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = (
        df.dropna(subset=['time', 'open', 'high', 'low', 'close'])
          .sort_values('time')
          .drop_duplicates('time')
          .reset_index(drop=True)
    )
    return df, missing


def split_contiguous(df: pd.DataFrame):
    if df.empty:
        return []
    gaps = df['time'].diff()
    seg_id = (gaps.isna() | (gaps > GAP_LIMIT)).cumsum()
    return [g.reset_index(drop=True) for _, g in df.groupby(seg_id) if len(g)]


def first_touch_trade(df: pd.DataFrame, signal_idx: int):
    high = df['high'].to_numpy(float)
    low = df['low'].to_numpy(float)
    close = df['close'].to_numpy(float)
    entry = float(close[signal_idx])
    avail = len(df) - signal_idx - 1
    end_idx = min(len(df) - 1, signal_idx + HORIZON_H)
    if avail <= 0:
        return {
            'trade_result': 'UNRESOLVED', 'exit_time': pd.NaT,
            'holding_hours': np.nan, 'gross_return': np.nan, 'net_return': np.nan,
        }

    result = None
    exit_idx = None
    for k in range(signal_idx + 1, end_idx + 1):
        hit_stop = low[k] <= entry * (1 - STOP)
        hit_target = high[k] >= entry * (1 + TARGET)
        if hit_stop:
            result, exit_idx = 'STOP', k
            break
        if hit_target:
            result, exit_idx = 'TARGET', k
            break

    if result == 'TARGET':
        gross = TARGET
    elif result == 'STOP':
        gross = -STOP
    elif avail >= HORIZON_H:
        result = 'TIME_EXIT'
        exit_idx = signal_idx + HORIZON_H
        gross = float(close[exit_idx] / entry - 1)
    else:
        return {
            'trade_result': 'UNRESOLVED', 'exit_time': pd.NaT,
            'holding_hours': np.nan, 'gross_return': np.nan, 'net_return': np.nan,
        }

    return {
        'trade_result': result,
        'exit_time': pd.Timestamp(df.iloc[exit_idx].time),
        'holding_hours': int(exit_idx - signal_idx),
        'gross_return': float(gross),
        'net_return': float(gross - ROUND_TRIP_COST),
    }


def generate_segment_signals(symbol: str, seg: pd.DataFrame, btc: pd.DataFrame,
                             models, thresholds, r5rules, r7rules, segment_id: int,
                             currently_listed, first_month: str, last_month: str):
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
    qv = pd.to_numeric(seg['quote_volume'], errors='coerce')
    qv24 = qv.rolling(24, min_periods=12).sum()
    qv7d = qv.rolling(168, min_periods=72).sum() / 7.0
    qv30d = qv.rolling(720, min_periods=168).sum() / 30.0

    rows = []
    for e in episodes.itertuples(index=False):
        if str(e.family) != 'BASE':
            continue
        i = int(e.start_idx)
        first_checkpoint = None
        first_vals = None
        first_j = None
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
            'symbol': symbol,
            'segment_id': segment_id,
            'candidate_start_time': pd.Timestamp(e.start_time),
            'candidate_family': 'BASE',
            'precursor_alerts': int(e.precursor_alerts),
            'signal_checkpoint_h': 12,
            'signal_time': pd.Timestamp(seg.iloc[first_j].time),
            'entry_price': float(seg.iloc[first_j].close),
            'candidate_open_to_signal_close': float(seg.iloc[first_j].close / seg.iloc[i].close - 1),
            'candidate_mfe_to_signal': float(first_vals['mfe_since_open']),
            'candidate_mae_to_signal': float(first_vals['mae_since_open']),
            'quote_volume_24h': float(qv24.iloc[first_j]) if pd.notna(qv24.iloc[first_j]) else np.nan,
            'quote_volume_7d_avg_daily': float(qv7d.iloc[first_j]) if pd.notna(qv7d.iloc[first_j]) else np.nan,
            'quote_volume_30d_avg_daily': float(qv30d.iloc[first_j]) if pd.notna(qv30d.iloc[first_j]) else np.nan,
            'currently_listed_spot_usdt': currently_listed,
            'first_archive_month': first_month,
            'last_archive_month': last_month,
            **first_touch_trade(seg, first_j),
        })
    return rows


def run_shard(shard_index: int, total_shards: int, out_dir: Path):
    universe = pd.read_csv(ROOT / 'historical_universe.csv')
    universe = universe.sort_values('symbol').reset_index(drop=True)
    universe = universe[universe.index % total_shards == shard_index].copy()
    if universe.empty:
        raise RuntimeError(f'No symbols assigned to shard {shard_index}')

    btcrow = pd.read_csv(ROOT / 'historical_universe.csv')
    btcrow = btcrow[btcrow.symbol == 'BTCUSDT']
    if btcrow.empty:
        raise RuntimeError('BTCUSDT not present in historical universe')
    btc_months = str(btcrow.iloc[0].months).split(';')
    btc, btc_missing = load_archive_symbol('BTCUSDT', btc_months)
    if btc.empty:
        raise RuntimeError('Could not load BTCUSDT historical archive')

    models, thresholds, _ = r8.train_frozen_detector()
    r5rules, r7rules = r8.load_frozen_rules()

    all_signals = []
    coverage = []
    failures = []
    for u in universe.itertuples(index=False):
        symbol = str(u.symbol)
        months = str(u.months).split(';')
        try:
            df, missing = load_archive_symbol(symbol, months)
            segments = split_contiguous(df)
            usable = [g for g in segments if len(g) >= MIN_SEGMENT_ROWS]
            coverage.append({
                'symbol': symbol,
                'expected_archive_months': len(months),
                'loaded_archive_months': len(months) - len(missing),
                'missing_archive_months': ';'.join(missing),
                'rows': len(df),
                'first_time': df.time.min() if len(df) else pd.NaT,
                'last_time': df.time.max() if len(df) else pd.NaT,
                'segments': len(segments),
                'usable_segments': len(usable),
                'currently_listed_spot_usdt': u.currently_listed_spot_usdt,
            })
            symbol_rows = []
            for sid, seg in enumerate(segments, start=1):
                if len(seg) < MIN_SEGMENT_ROWS:
                    continue
                try:
                    symbol_rows.extend(generate_segment_signals(
                        symbol, seg, btc, models, thresholds, r5rules, r7rules, sid,
                        u.currently_listed_spot_usdt, u.first_archive_month, u.last_archive_month,
                    ))
                except Exception as e:
                    failures.append({'symbol': symbol, 'segment_id': sid, 'stage': 'signal_generation', 'error': repr(e)})
            all_signals.extend(symbol_rows)
            print('SHARD', shard_index, symbol, 'rows', len(df), 'segments', len(segments), 'signals', len(symbol_rows), flush=True)
        except Exception as e:
            failures.append({'symbol': symbol, 'segment_id': '', 'stage': 'symbol', 'error': repr(e)})
            print('SHARD_ERROR', shard_index, symbol, repr(e), flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_signals).to_csv(out_dir / f'signals_shard_{shard_index}.csv.gz', index=False, compression='gzip')
    pd.DataFrame(coverage).to_csv(out_dir / f'coverage_shard_{shard_index}.csv', index=False)
    pd.DataFrame(failures).to_csv(out_dir / f'failures_shard_{shard_index}.csv', index=False)
    pd.DataFrame([{
        'shard': shard_index,
        'symbols': len(universe),
        'signals': len(all_signals),
        'failures': len(failures),
        'btc_missing_months': ';'.join(btc_missing),
    }]).to_csv(out_dir / f'shard_summary_{shard_index}.csv', index=False)


def concat_csvs(paths, compression=None):
    frames = []
    for p in paths:
        try:
            q = pd.read_csv(p, compression=compression)
            if len(q) or len(q.columns):
                frames.append(q)
        except pd.errors.EmptyDataError:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def resolved_trades(signals: pd.DataFrame):
    return signals[(signals.trade_result != 'UNRESOLVED') & pd.to_numeric(signals.net_return, errors='coerce').notna()].copy()


def year_summary(signals: pd.DataFrame):
    z = signals.copy()
    z['signal_time'] = pd.to_datetime(z.signal_time, utc=True)
    z['year'] = z.signal_time.dt.year
    z['week'] = z.signal_time.dt.strftime('%G-W%V')
    rows = []
    for key, g in list(z.groupby('year')) + [('ALL', z)]:
        r = resolved_trades(g)
        net = pd.to_numeric(r.net_return, errors='coerce')
        rows.append({
            'year': key,
            'signals': len(g),
            'resolved_trades': len(r),
            'symbols': g.symbol.nunique(),
            'distinct_weeks': g.week.nunique(),
            'targets': int((r.trade_result == 'TARGET').sum()),
            'stops': int((r.trade_result == 'STOP').sum()),
            'time_exits': int((r.trade_result == 'TIME_EXIT').sum()),
            'unresolved': int((g.trade_result == 'UNRESOLVED').sum()),
            'target_rate': float((r.trade_result == 'TARGET').mean()) if len(r) else np.nan,
            'avg_net_return': float(net.mean()) if len(r) else np.nan,
            'median_net_return': float(net.median()) if len(r) else np.nan,
            'median_holding_hours': float(pd.to_numeric(r.holding_hours, errors='coerce').median()) if len(r) else np.nan,
        })
    return pd.DataFrame(rows)


def concurrency_stats(signals: pd.DataFrame):
    t = resolved_trades(signals)
    if t.empty:
        return pd.DataFrame()
    t['signal_time'] = pd.to_datetime(t.signal_time, utc=True)
    t['exit_time'] = pd.to_datetime(t.exit_time, utc=True)
    active = []
    rows = []
    for r in t.sort_values(['signal_time', 'symbol'], kind='stable').itertuples(index=False):
        active = [x for x in active if x > r.signal_time]
        before = len(active)
        active.append(r.exit_time)
        rows.append({'year': r.signal_time.year, 'concurrent_before_entry': before, 'concurrent_after_entry': len(active)})
    p = pd.DataFrame(rows)
    out = []
    for key, g in list(p.groupby('year')) + [('ALL', p)]:
        out.append({
            'year': key,
            'trades': len(g),
            'overlap_trade_rate': float((g.concurrent_before_entry > 0).mean()),
            'max_concurrency_observed': int(g.concurrent_after_entry.max()),
            'median_concurrent_before_entry': float(g.concurrent_before_entry.median()),
        })
    return pd.DataFrame(out)


def simulate_book(g: pd.DataFrame, n_sleeves: int, starting_equity: float = 1000.0):
    equities = [starting_equity / n_sleeves] * n_sleeves
    free_at = [pd.Timestamp.min.tz_localize('UTC')] * n_sleeves
    ledger = []
    curve = []
    if len(g):
        start_time = pd.to_datetime(g.signal_time, utc=True).min()
        curve.append((start_time, starting_equity))

    for r in g.sort_values(['signal_time', 'symbol'], kind='stable').itertuples(index=False):
        st = pd.Timestamp(r.signal_time)
        et = pd.Timestamp(r.exit_time) if pd.notna(r.exit_time) else pd.NaT
        if pd.isna(et) or not np.isfinite(r.net_return):
            ledger.append({'symbol': r.symbol, 'signal_time': st, 'accepted': False, 'skip_reason': 'unresolved'})
            continue
        candidates = [i for i, ft in enumerate(free_at) if ft <= st]
        if not candidates:
            ledger.append({'symbol': r.symbol, 'signal_time': st, 'accepted': False, 'skip_reason': 'capacity'})
            continue
        i = min(candidates)
        before = equities[i]
        after = before * (1 + float(r.net_return))
        equities[i] = after
        free_at[i] = et
        ledger.append({
            'symbol': r.symbol, 'signal_time': st, 'exit_time': et, 'accepted': True,
            'skip_reason': '', 'sleeve': i + 1, 'capital_before': before,
            'capital_after': after, 'net_return': float(r.net_return),
        })
        curve.append((et, float(sum(equities))))

    led = pd.DataFrame(ledger)
    c = pd.DataFrame(curve, columns=['time', 'equity']).sort_values('time') if curve else pd.DataFrame(columns=['time', 'equity'])
    if len(c):
        dd = c.equity / c.equity.cummax() - 1
        max_dd = float(dd.min())
    else:
        max_dd = 0.0
    ending = float(sum(equities))
    return {
        'starting_equity': starting_equity,
        'ending_equity': ending,
        'return': ending / starting_equity - 1,
        'accepted_trades': int(led.accepted.sum()) if 'accepted' in led else 0,
        'capacity_skipped_trades': int((led.skip_reason == 'capacity').sum()) if 'skip_reason' in led else 0,
        'max_realised_drawdown': max_dd,
    }, led


def portfolio_outputs(signals: pd.DataFrame):
    z = resolved_trades(signals)
    z['signal_time'] = pd.to_datetime(z.signal_time, utc=True)
    z['exit_time'] = pd.to_datetime(z.exit_time, utc=True)
    z['year'] = z.signal_time.dt.year
    annual, annual_ledgers = [], []
    for year, g in z.groupby('year'):
        for sleeves in SLEEVES:
            s, led = simulate_book(g, sleeves, 1000.0)
            annual.append({'year': int(year), 'sleeves': sleeves, **s})
            if len(led):
                led['year'] = int(year)
                led['sleeves'] = sleeves
                annual_ledgers.append(led)

    continuous, continuous_ledgers = [], []
    for sleeves in SLEEVES:
        s, led = simulate_book(z, sleeves, 1000.0)
        continuous.append({'sleeves': sleeves, **s})
        if len(led):
            led['sleeves'] = sleeves
            continuous_ledgers.append(led)

    return (
        pd.DataFrame(annual),
        pd.concat(annual_ledgers, ignore_index=True) if annual_ledgers else pd.DataFrame(),
        pd.DataFrame(continuous),
        pd.concat(continuous_ledgers, ignore_index=True) if continuous_ledgers else pd.DataFrame(),
    )


def status_summary(signals: pd.DataFrame):
    z = resolved_trades(signals)
    if z.empty:
        return pd.DataFrame()
    z['universe_status'] = np.where(
        z.currently_listed_spot_usdt.astype(str).str.lower().isin(['true', '1']),
        'CURRENT', 'DELISTED_OR_NOT_CURRENT'
    )
    rows = []
    for status, g in z.groupby('universe_status'):
        rows.append({
            'universe_status': status,
            'trades': len(g),
            'symbols': g.symbol.nunique(),
            'target_rate': float((g.trade_result == 'TARGET').mean()),
            'avg_net_return': float(pd.to_numeric(g.net_return, errors='coerce').mean()),
            'median_net_return': float(pd.to_numeric(g.net_return, errors='coerce').median()),
        })
    return pd.DataFrame(rows)


def liquidity_summary(signals: pd.DataFrame):
    z = resolved_trades(signals)
    if z.empty:
        return pd.DataFrame()
    z['signal_time'] = pd.to_datetime(z.signal_time, utc=True)
    z['year'] = z.signal_time.dt.year
    z['liq'] = pd.to_numeric(z.quote_volume_30d_avg_daily, errors='coerce')
    rows = []

    def add_group(label, g):
        q = g[g.liq.notna()].copy()
        if len(q) < 4:
            return
        try:
            q['quartile'] = pd.qcut(q.liq.rank(method='first'), 4, labels=['Q1_LOW', 'Q2', 'Q3', 'Q4_HIGH'])
        except ValueError:
            return
        for quartile, h in q.groupby('quartile', observed=True):
            rows.append({
                'period': label,
                'liquidity_quartile': str(quartile),
                'trades': len(h),
                'median_30d_avg_daily_quote_volume': float(h.liq.median()),
                'target_rate': float((h.trade_result == 'TARGET').mean()),
                'avg_net_return': float(pd.to_numeric(h.net_return, errors='coerce').mean()),
            })

    add_group('ALL', z)
    for year, g in z.groupby('year'):
        add_group(str(year), g)
    return pd.DataFrame(rows)


def early_history_summary(signals: pd.DataFrame):
    z = signals.copy()
    z['signal_time'] = pd.to_datetime(z.signal_time, utc=True)
    z['period'] = np.select(
        [z.signal_time.dt.year <= 2018, z.signal_time.dt.year <= 2022],
        ['PRE_2019', '2019_2022_DEVELOPMENT_ERA'],
        default='2023_PLUS',
    )
    rows = []
    for period, g in z.groupby('period'):
        r = resolved_trades(g)
        rows.append({
            'period': period,
            'signals': len(g),
            'trades': len(r),
            'symbols': g.symbol.nunique(),
            'target_rate': float((r.trade_result == 'TARGET').mean()) if len(r) else np.nan,
            'avg_net_return': float(pd.to_numeric(r.net_return, errors='coerce').mean()) if len(r) else np.nan,
        })
    return pd.DataFrame(rows)


def write_analysis(universe, excluded, years, status, overlap, annual, continuous, failures, early):
    all_row = years[years.year.astype(str) == 'ALL'].iloc[0]
    lines = [
        '# BASE +12h Strategy Research — Round 02 Analysis', '',
        '**Research only. No live strategy, portfolio, log or automation changes.**', '',
        '## Frozen strategy', '',
        '- BASE candidate family.',
        '- First accepted checkpoint exactly +12h.',
        '- Enter at signal close.',
        '- OCO +10% / -2%.',
        f'- {ROUND_TRIP_COST*100:.2f}% round-trip cost.',
        '- No Round 02 parameter optimisation.', '',
        '## Historical universe', '',
        f'- Historical USDT archive symbols included: **{len(universe):,}**.',
        f'- Archive symbols explicitly excluded: **{len(excluded):,}**.',
        f'- Processing failures: **{len(failures):,}**.',
        '- Universe comes from Binance Vision historical monthly 1h archive listings, not current exchange ranking.', '',
        '## Headline validation result', '',
        f'- Signals: **{int(all_row.signals):,}**.',
        f'- Resolved trades: **{int(all_row.resolved_trades):,}** across **{int(all_row.symbols):,}** symbols and **{int(all_row.distinct_weeks):,}** weeks.',
        f'- Target rate: **{100*all_row.target_rate:.2f}%**.',
        f'- Average net return/trade: **{100*all_row.avg_net_return:.2f}%**.',
        f'- Round 01 benchmark: {ROUND1_TRADES} trades at **{100*ROUND1_AVG_NET:.2f}%** average net/trade.', '',
        '## By year', '',
        '| Year | Signals | Trades | Symbols | Weeks | Target rate | Avg net | Median net | Median hold |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in years.itertuples(index=False):
        pc = lambda x: 'n/a' if pd.isna(x) else f'{100*x:.2f}%'
        hold = 'n/a' if pd.isna(r.median_holding_hours) else f'{r.median_holding_hours:.1f}h'
        lines.append(f'| {r.year} | {int(r.signals)} | {int(r.resolved_trades)} | {int(r.symbols)} | {int(r.distinct_weeks)} | {pc(r.target_rate)} | {pc(r.avg_net_return)} | {pc(r.median_net_return)} | {hold} |')

    lines += ['', '## Current survivors vs historical/delisted symbols', '',
              '| Status | Trades | Symbols | Target rate | Avg net |',
              '|---|---:|---:|---:|---:|']
    for r in status.itertuples(index=False):
        lines.append(f'| {r.universe_status} | {int(r.trades)} | {int(r.symbols)} | {100*r.target_rate:.2f}% | {100*r.avg_net_return:.2f}% |')

    lines += ['', '## Early-history check', '',
              '| Period | Signals | Trades | Symbols | Target rate | Avg net |',
              '|---|---:|---:|---:|---:|---:|']
    for r in early.itertuples(index=False):
        tr = 'n/a' if pd.isna(r.target_rate) else f'{100*r.target_rate:.2f}%'
        av = 'n/a' if pd.isna(r.avg_net_return) else f'{100*r.avg_net_return:.2f}%'
        lines.append(f'| {r.period} | {int(r.signals)} | {int(r.trades)} | {int(r.symbols)} | {tr} | {av} |')

    lines += ['', '## Overlap', '']
    if len(overlap):
        a = overlap[overlap.year.astype(str) == 'ALL'].iloc[0]
        lines += [
            f'- Trades overlapping an already-open trade: **{100*a.overlap_trade_rate:.1f}%**.',
            f'- Maximum observed simultaneous open trades: **{int(a.max_concurrency_observed)}**.',
        ]

    lines += ['', '## Annual-reset 1,000 USDT portfolio', '',
              '| Year | Sleeves | Ending equity | Return | Accepted | Capacity skipped | Max realised DD |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for _, r in annual.iterrows():
        lines.append(f'| {int(r["year"])} | {int(r["sleeves"])} | {r["ending_equity"]:,.2f} | {100*r["return"]:.2f}% | {int(r["accepted_trades"])} | {int(r["capacity_skipped_trades"])} | {100*r["max_realised_drawdown"]:.2f}% |')

    lines += ['', '## Continuous 1,000 USDT compound simulation', '',
              '| Sleeves | Ending equity | Total return | Accepted | Capacity skipped | Max realised DD |',
              '|---:|---:|---:|---:|---:|---:|']
    for _, r in continuous.iterrows():
        lines.append(f'| {int(r["sleeves"])} | {r["ending_equity"]:,.2f} | {100*r["return"]:.2f}% | {int(r["accepted_trades"])} | {int(r["capacity_skipped_trades"])} | {100*r["max_realised_drawdown"]:.2f}% |')

    lines += ['', '## Interpretation guardrails', '',
              '- This test reduces current-symbol survivorship bias by discovering the universe from historical Binance Vision archives, including symbols no longer currently listed.',
              '- It is still not a pristine prospective test because the signal was originally developed using overlapping historical periods.',
              '- Liquidity is analysed descriptively; no new Round 02 liquidity cutoff was fitted.',
              '- Portfolio capacity ordering is deterministic for same-time signals and should not be mistaken for an optimised selector.',
              '- No result from this round is promoted to the live strategy automatically.']
    (ROOT / 'ANALYSIS.md').write_text('\n'.join(lines) + '\n')


def combine(shards_dir: Path):
    signal_files = sorted(shards_dir.rglob('signals_shard_*.csv.gz'))
    coverage_files = sorted(shards_dir.rglob('coverage_shard_*.csv'))
    failure_files = sorted(shards_dir.rglob('failures_shard_*.csv'))
    if not signal_files:
        raise RuntimeError(f'No shard signal files found under {shards_dir}')

    signals = concat_csvs(signal_files, compression='gzip')
    coverage = concat_csvs(coverage_files)
    failures = concat_csvs(failure_files)
    universe = pd.read_csv(ROOT / 'historical_universe.csv')
    excluded = pd.read_csv(ROOT / 'excluded_universe.csv')

    if signals.empty:
        raise RuntimeError('Round 02 produced no BASE +12h signals')

    signals['signal_time'] = pd.to_datetime(signals.signal_time, utc=True)
    signals['exit_time'] = pd.to_datetime(signals.exit_time, utc=True)
    signals = signals.sort_values(['signal_time', 'symbol'], kind='stable').reset_index(drop=True)
    signals.to_csv(ROOT / 'signals.csv.gz', index=False, compression='gzip')
    coverage.to_csv(ROOT / 'data_coverage.csv', index=False)
    failures.to_csv(ROOT / 'failures.csv', index=False)

    years = year_summary(signals)
    overlap = concurrency_stats(signals)
    annual, annual_ledger, continuous, continuous_ledger = portfolio_outputs(signals)
    status = status_summary(signals)
    liquidity = liquidity_summary(signals)
    early = early_history_summary(signals)

    years.to_csv(ROOT / 'year_summary.csv', index=False)
    overlap.to_csv(ROOT / 'overlap_summary.csv', index=False)
    annual.to_csv(ROOT / 'portfolio_annual.csv', index=False)
    annual_ledger.to_csv(ROOT / 'portfolio_annual_trades.csv.gz', index=False, compression='gzip')
    continuous.to_csv(ROOT / 'portfolio_continuous.csv', index=False)
    continuous_ledger.to_csv(ROOT / 'portfolio_continuous_trades.csv.gz', index=False, compression='gzip')
    status.to_csv(ROOT / 'status_summary.csv', index=False)
    liquidity.to_csv(ROOT / 'liquidity_summary.csv', index=False)
    early.to_csv(ROOT / 'early_history_summary.csv', index=False)

    write_analysis(universe, excluded, years, status, overlap, annual, continuous, failures, early)
    print((ROOT / 'ANALYSIS.md').read_text(), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['discover', 'shard', 'combine'], required=True)
    p.add_argument('--shard-index', type=int)
    p.add_argument('--total-shards', type=int, default=8)
    p.add_argument('--out-dir', type=Path)
    p.add_argument('--shards-dir', type=Path)
    args = p.parse_args()

    if args.mode == 'discover':
        discover_universe()
    elif args.mode == 'shard':
        if args.shard_index is None or args.out_dir is None:
            p.error('--shard-index and --out-dir are required for shard mode')
        run_shard(args.shard_index, args.total_shards, args.out_dir)
    else:
        if args.shards_dir is None:
            p.error('--shards-dir is required for combine mode')
        combine(args.shards_dir)


if __name__ == '__main__':
    main()
