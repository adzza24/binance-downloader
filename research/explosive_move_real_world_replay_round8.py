from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from binance_data import COLS, load_symbol
import explosive_move_discovery_round1 as disc
import explosive_move_recall_round3 as recall
import explosive_move_recall_round3_retry as retry

OUT = Path('research/results/explosive_move_real_world_replay_round8')
CHECKPOINT_DIR = OUT / 'checkpoints'
START_REPLAY = pd.Timestamp('2024-01-01T00:00:00Z')
LOOKBACK_START = '2023-09-01'
DEDUPE_HOURS = 96
ALERT_COOLDOWN_HOURS = 24
ROUND_TRIP_COST = 0.003
UNION_CAPS = {3: 1, 6: 0, 12: 2, 24: 1, 48: 1}
CHECKPOINTS = sorted(UNION_CAPS)


def get_json(url: str):
    req = urllib.request.Request(url, headers={'User-Agent': 'explosive-real-world-replay/1.0'})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.loads(r.read().decode())


def current_top200() -> pd.DataFrame:
    info = get_json('https://data-api.binance.vision/api/v3/exchangeInfo')
    ticker = get_json('https://data-api.binance.vision/api/v3/ticker/24hr')
    ok = {}
    for s in info['symbols']:
        sym = s.get('symbol', '')
        base = s.get('baseAsset', '')
        if s.get('quoteAsset') != 'USDT' or s.get('status') != 'TRADING' or not s.get('isSpotTradingAllowed', True):
            continue
        if base in disc.STABLE_BASES or sym in disc.EXCL:
            continue
        if re.search(r'(UP|DOWN|BULL|BEAR)USDT$', sym):
            continue
        ok[sym] = base
    vols = {x['symbol']: float(x.get('quoteVolume', 0) or 0) for x in ticker if x.get('symbol') in ok}
    ranked = sorted(vols.items(), key=lambda kv: kv[1], reverse=True)[:200]
    return pd.DataFrame([
        {
            'symbol': sym,
            'current_quote_volume_24h': vol,
            'current_liquidity_rank': i + 1,
            'replay_tier': 'ORIGINAL_TOP100' if i < 100 else 'EXTERNAL_101_200',
        }
        for i, (sym, vol) in enumerate(ranked)
    ])


def fetch_rest_klines(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while cursor <= end_ms:
        params = urllib.parse.urlencode({
            'symbol': symbol,
            'interval': '1h',
            'startTime': cursor,
            'endTime': end_ms,
            'limit': 1000,
        })
        data = get_json(f'https://data-api.binance.vision/api/v3/klines?{params}')
        if not data:
            break
        rows.extend(data)
        nxt = int(data[-1][0]) + 3600_000
        if nxt <= cursor:
            break
        cursor = nxt
        if len(data) < 1000:
            break
    if not rows:
        return pd.DataFrame(columns=COLS + ['time'])
    z = pd.DataFrame(rows, columns=COLS)
    z['time'] = pd.to_datetime(pd.to_numeric(z['open_time'], errors='coerce'), unit='ms', utc=True)
    for c in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_base']:
        z[c] = pd.to_numeric(z[c], errors='coerce')
    return z.dropna(subset=['time', 'open', 'high', 'low', 'close'])


def load_extended(symbol: str, end: pd.Timestamp) -> pd.DataFrame:
    # Monthly archives provide the bulk history. REST only fills the tail not yet in monthly archives.
    z = load_symbol(symbol, '1h', LOOKBACK_START, end.strftime('%Y-%m-%d'))
    last = pd.Timestamp(z['time'].max()) if len(z) else pd.Timestamp(LOOKBACK_START, tz='UTC') - pd.Timedelta(hours=1)
    tail_start = max(last + pd.Timedelta(hours=1), pd.Timestamp(LOOKBACK_START, tz='UTC'))
    if tail_start <= end:
        tail = fetch_rest_klines(symbol, tail_start, end)
        if len(tail):
            z = pd.concat([z, tail], ignore_index=True)
    if z.empty:
        return z
    z['time'] = pd.to_datetime(z['time'], utc=True)
    return z.sort_values('time').drop_duplicates('time').reset_index(drop=True)


def train_frozen_detector():
    events, samples, _ = recall.load_inputs()
    assignments = recall.build_families(events, samples)
    models, threshold_table, _ = retry.train_family_models(assignments, samples)
    thresholds = recall.selected_thresholds(threshold_table)
    return models, thresholds, threshold_table


def load_frozen_rules():
    r5 = pd.read_csv('research/results/explosive_move_intersection_veto_round5/selected_rules.csv')
    r5 = r5[np.isclose(r5['target_recall'], 0.90)].sort_values('step').reset_index(drop=True)
    r7 = pd.read_csv('research/results/explosive_move_candidate_evolution_round7/evolution_frontier_rules.csv')
    by_checkpoint = {}
    for h, cap in UNION_CAPS.items():
        by_checkpoint[h] = r7[(r7['checkpoint_h'] == h) & (r7['development_noise_cap'] == cap)].sort_values('step').reset_index(drop=True)
    return r5, by_checkpoint


def rule_pass(value, direction: str, threshold: float) -> bool:
    if value is None or not np.isfinite(value):
        return False
    return bool(value >= threshold) if direction == 'GE' else bool(value <= threshold)


def apply_round5_filter(alerts: pd.DataFrame, ff: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    if alerts.empty:
        return alerts
    vals = []
    for r in alerts.itertuples(index=False):
        i = int(r.candle_idx)
        row = {
            'ret_6h_delta_24h': np.nan,
            'ret_24h_delta_12h': np.nan,
            'ret_24h_delta_3h': np.nan,
        }
        if i >= 24:
            row['ret_6h_delta_24h'] = float(ff.iloc[i]['ret_6h'] - ff.iloc[i - 24]['ret_6h'])
        if i >= 12:
            row['ret_24h_delta_12h'] = float(ff.iloc[i]['ret_24h'] - ff.iloc[i - 12]['ret_24h'])
        if i >= 3:
            row['ret_24h_delta_3h'] = float(ff.iloc[i]['ret_24h'] - ff.iloc[i - 3]['ret_24h'])
        vals.append(row)
    v = pd.DataFrame(vals)
    out = pd.concat([alerts.reset_index(drop=True), v], axis=1)
    keep = np.ones(len(out), dtype=bool)
    for rr in rules.itertuples(index=False):
        x = pd.to_numeric(out[rr.feature], errors='coerce').to_numpy(float)
        m = np.isfinite(x) & ((x >= rr.threshold) if rr.direction == 'GE' else (x <= rr.threshold))
        if rr.scope != 'ALL':
            scoped = out['family'].astype(str).eq(rr.scope).to_numpy()
            m = (~scoped) | m
        keep &= m
    return out[keep].copy()


def build_base_alerts(df: pd.DataFrame, btc: pd.DataFrame, models, thresholds) -> tuple[pd.DataFrame, pd.DataFrame]:
    ff = disc.build_features(df, btc)
    ff['time'] = pd.to_datetime(ff['time'], utc=True)
    valid = ff[disc.FEATURES].notna().mean(axis=1).to_numpy() >= 0.80
    times = ff['time']
    rows = []
    for fam, (imp, rf) in models.items():
        if fam not in thresholds:
            continue
        scores = rf.predict_proba(imp.transform(ff[disc.FEATURES]))[:, 1]
        idxs = np.flatnonzero(valid & (scores >= thresholds[fam]))
        last = -10**9
        for i in idxs:
            if int(i) - last < ALERT_COOLDOWN_HOURS:
                continue
            last = int(i)
            ts = times.iloc[int(i)]
            if ts < START_REPLAY:
                continue
            rows.append({
                'symbol': '', 'time': ts, 'family': fam, 'score': float(scores[int(i)]),
                'threshold': float(thresholds[fam]), 'candle_idx': int(i),
            })
    return pd.DataFrame(rows), ff


def build_episodes(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    s = selected.sort_values(['time', 'family'], kind='stable').copy()
    s['prev_time'] = s['time'].shift()
    gap = (s['time'] - s['prev_time']).dt.total_seconds() / 3600.0
    s['new_episode'] = s['prev_time'].isna() | (gap > DEDUPE_HOURS)
    s['episode_num'] = s['new_episode'].cumsum().astype(int)
    return s.groupby('episode_num', as_index=False).agg(
        start_time=('time', 'first'),
        family=('family', 'first'),
        start_idx=('candle_idx', 'first'),
        precursor_alerts=('time', 'size'),
        episode_last_alert=('time', 'last'),
    )


def evolution_values(i: int, j: int, family: str, df: pd.DataFrame, ff: pd.DataFrame, btc_close: np.ndarray) -> dict:
    close = df['close'].to_numpy(float)
    high = df['high'].to_numpy(float)
    low = df['low'].to_numpy(float)
    base = close[i]
    close_j = close[j]
    seg_hi = float(np.nanmax(high[i + 1:j + 1])) if j > i else base
    seg_lo = float(np.nanmin(low[i + 1:j + 1])) if j > i else base
    coin_ret = close_j / base - 1
    btc_ret = btc_close[j] / btc_close[i] - 1 if np.isfinite(btc_close[i]) and np.isfinite(btc_close[j]) and btc_close[i] > 0 else np.nan
    out = {
        'mfe_since_open': seg_hi / base - 1,
        'mae_since_open': seg_lo / base - 1,
        'relative_return_since_open': coin_ret - btc_ret if np.isfinite(btc_ret) else np.nan,
    }
    cur = ff.iloc[j]
    basef = ff.iloc[i]
    for f in disc.FEATURES:
        cv = cur.get(f, np.nan)
        bv = basef.get(f, np.nan)
        out[f'cur_{f}'] = float(cv) if pd.notna(cv) else np.nan
        out[f'delta_{f}'] = float(cv - bv) if pd.notna(cv) and pd.notna(bv) else np.nan
    return out


def checkpoint_pass(values: dict, family: str, rules: pd.DataFrame) -> bool:
    for r in rules.itertuples(index=False):
        if r.scope != 'ALL' and str(family) != str(r.scope):
            continue
        if not rule_pass(values.get(r.feature, np.nan), r.direction, float(r.threshold)):
            return False
    return True


def first_touch(high: np.ndarray, low: np.ndarray, start: int, end: int, entry: float, target: float, stop: float):
    for k in range(start, end + 1):
        hit_stop = low[k] <= entry * (1 - stop)
        hit_target = high[k] >= entry * (1 + target)
        if hit_stop and hit_target:
            return 'STOP', k
        if hit_stop:
            return 'STOP', k
        if hit_target:
            return 'TARGET', k
    return None, None


def hours_to_level(high: np.ndarray, low: np.ndarray, start: int, end: int, entry: float, pct: float, side: str):
    for k in range(start, end + 1):
        if side == 'up' and high[k] >= entry * (1 + pct):
            return k - (start - 1)
        if side == 'down' and low[k] <= entry * (1 - pct):
            return k - (start - 1)
    return np.nan


def outcome(signal_idx: int, df: pd.DataFrame) -> dict:
    close = df['close'].to_numpy(float); high = df['high'].to_numpy(float); low = df['low'].to_numpy(float)
    entry = close[signal_idx]
    avail = len(df) - signal_idx - 1
    end30 = min(len(df) - 1, signal_idx + 720)
    mature = avail >= 720
    row = {'entry_price': entry, 'forward_hours_available': int(avail), 'mature_30d': bool(mature)}
    for hours, name in [(24, '24h'), (72, '72h'), (168, '7d'), (720, '30d')]:
        if avail < hours:
            row[f'mfe_{name}'] = np.nan; row[f'mae_{name}'] = np.nan
            continue
        e = signal_idx + hours
        row[f'mfe_{name}'] = float(np.nanmax(high[signal_idx + 1:e + 1]) / entry - 1)
        row[f'mae_{name}'] = float(np.nanmin(low[signal_idx + 1:e + 1]) / entry - 1)
    for pct in [0.05, 0.10, 0.15, 0.20, 0.30]:
        row[f'hours_to_up_{int(pct*100)}'] = hours_to_level(high, low, signal_idx + 1, end30, entry, pct, 'up')
    for pct in [0.05, 0.10, 0.15]:
        row[f'hours_to_down_{int(pct*100)}'] = hours_to_level(high, low, signal_idx + 1, end30, entry, pct, 'down')
    for target, stop, label in [(0.10, 0.05, 't10_s5'), (0.20, 0.05, 't20_s5'), (0.20, 0.10, 't20_s10')]:
        result, k = first_touch(high, low, signal_idx + 1, end30, entry, target, stop)
        row[f'{label}_result'] = result or 'OPEN'
        row[f'{label}_hours'] = (k - signal_idx) if k is not None else np.nan
        if mature:
            if result == 'TARGET': gross = target
            elif result == 'STOP': gross = -stop
            else: gross = close[signal_idx + 720] / entry - 1
            row[f'{label}_gross_return'] = float(gross)
            row[f'{label}_net_return'] = float(gross - ROUND_TRIP_COST)
        else:
            row[f'{label}_gross_return'] = np.nan; row[f'{label}_net_return'] = np.nan
    return row


def process_symbol(meta, btc: pd.DataFrame, models, thresholds, r5rules, r7rules, end: pd.Timestamp):
    symbol = meta.symbol
    cp = CHECKPOINT_DIR / f'{symbol}.csv.gz'
    if cp.exists():
        z = pd.read_csv(cp, compression='gzip')
        for c in ['candidate_start_time', 'signal_time']:
            if c in z: z[c] = pd.to_datetime(z[c], utc=True)
        print(symbol, 'checkpoint reused', len(z), flush=True)
        return z

    df = load_extended(symbol, end)
    if len(df) < 3000:
        raise RuntimeError(f'insufficient_history:{len(df)}')
    btc_aligned = btc.set_index('time')['close'].reindex(pd.DatetimeIndex(df['time'])).to_numpy(float)
    alerts, ff = build_base_alerts(df, btc, models, thresholds)
    alerts['symbol'] = symbol
    selected = apply_round5_filter(alerts, ff, r5rules)
    episodes = build_episodes(selected)
    rows = []
    if not episodes.empty:
        time_to_idx = {pd.Timestamp(t): i for i, t in enumerate(pd.to_datetime(df['time'], utc=True))}
        for e in episodes.itertuples(index=False):
            i = int(e.start_idx)
            signal = None
            for h in CHECKPOINTS:
                j = i + h
                if j >= len(df):
                    continue
                vals = evolution_values(i, j, e.family, df, ff, btc_aligned)
                if vals['mfe_since_open'] > 0.05:
                    continue
                if checkpoint_pass(vals, e.family, r7rules[h]):
                    signal = (h, j, vals)
                    break
            if signal is None:
                continue
            h, j, vals = signal
            r = {
                'symbol': symbol,
                'current_liquidity_rank': int(meta.current_liquidity_rank),
                'replay_tier': meta.replay_tier,
                'candidate_start_time': pd.Timestamp(e.start_time),
                'candidate_family': e.family,
                'precursor_alerts': int(e.precursor_alerts),
                'signal_checkpoint_h': int(h),
                'signal_time': pd.Timestamp(df.iloc[j].time),
                'candidate_open_to_signal_close': float(df.iloc[j].close / df.iloc[i].close - 1),
                'candidate_mfe_to_signal': float(vals['mfe_since_open']),
                'candidate_mae_to_signal': float(vals['mae_since_open']),
            }
            r.update(outcome(j, df))
            rows.append(r)
    z = pd.DataFrame(rows)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    z.to_csv(cp, index=False, compression='gzip')
    print(symbol, 'complete', 'alerts', len(alerts), 'selected', len(selected), 'episodes', len(episodes), 'signals', len(z), flush=True)
    return z


def pct_nonnull(s):
    x = pd.to_numeric(s, errors='coerce').dropna()
    return float(x.mean()) if len(x) else np.nan


def summarise(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    signals = signals.copy()
    signals['year'] = pd.to_datetime(signals['signal_time'], utc=True).dt.year
    signals['week'] = pd.to_datetime(signals['signal_time'], utc=True).dt.strftime('%G-W%V')
    groups = []
    for tier in ['ORIGINAL_TOP100', 'EXTERNAL_101_200', 'ALL_TOP200']:
        base = signals if tier == 'ALL_TOP200' else signals[signals['replay_tier'] == tier]
        for year in ['ALL', 2024, 2025, 2026]:
            g = base if year == 'ALL' else base[base['year'] == year]
            if g.empty:
                continue
            m = g[g['mature_30d'].astype(bool)]
            row = {
                'tier': tier, 'year': year, 'signals': len(g), 'symbols': g['symbol'].nunique(),
                'weeks_with_signal': g['week'].nunique(), 'mature_30d_signals': len(m),
                'median_mfe_24h': pd.to_numeric(g['mfe_24h'], errors='coerce').median(),
                'median_mfe_7d': pd.to_numeric(g['mfe_7d'], errors='coerce').median(),
                'median_mfe_30d': pd.to_numeric(m['mfe_30d'], errors='coerce').median(),
                'median_mae_30d': pd.to_numeric(m['mae_30d'], errors='coerce').median(),
                'hit10_30d_rate': float(m['hours_to_up_10'].notna().mean()) if len(m) else np.nan,
                'hit20_30d_rate': float(m['hours_to_up_20'].notna().mean()) if len(m) else np.nan,
                'hit30_30d_rate': float(m['hours_to_up_30'].notna().mean()) if len(m) else np.nan,
                't10_s5_target_rate': float((m['t10_s5_result'] == 'TARGET').mean()) if len(m) else np.nan,
                't20_s5_target_rate': float((m['t20_s5_result'] == 'TARGET').mean()) if len(m) else np.nan,
                't20_s10_target_rate': float((m['t20_s10_result'] == 'TARGET').mean()) if len(m) else np.nan,
                't10_s5_avg_net_return': pd.to_numeric(m['t10_s5_net_return'], errors='coerce').mean(),
                't20_s5_avg_net_return': pd.to_numeric(m['t20_s5_net_return'], errors='coerce').mean(),
                't20_s10_avg_net_return': pd.to_numeric(m['t20_s10_net_return'], errors='coerce').mean(),
            }
            groups.append(row)
    return pd.DataFrame(groups)


def write_analysis(summary: pd.DataFrame, signals: pd.DataFrame, universe: pd.DataFrame, end: pd.Timestamp):
    lines = [
        '# Explosive Move Full-Population Replay Round 8', '',
        '**Status:** RESEARCH ONLY on isolated `round-3`. No live strategy or automation changes.', '',
        '## Test design', '',
        f'- Replay period: **2024-01-01 through {end.isoformat()}** using real Binance 1h candles.',
        '- Signal generation is causal and does not consult the explosive-event catalogue or wanted/noise labels.',
        '- Frozen pipeline: Round 3 family detector -> Round 5 90% precursor intersection -> 96h same-symbol candidate dedupe -> Round 7 checkpoint-union rules.',
        '- Frozen union is the Round 7 winner-maximising development-noise-cap-5 combination: `3h=1 | 6h=0 | 12h=2 | 24h=1 | 48h=1`.',
        '- Candidates that exceeded +5% from candidate open before a checkpoint are not eligible at that checkpoint.',
        '- Outcomes are measured directly from subsequent candles after the signal, not from catalogue membership.',
        '- Current-liquidity ranks 1-100 are reported separately from ranks 101-200, which were outside the original top-100 event universe.',
        f'- Simple target/stop diagnostics subtract **{ROUND_TRIP_COST*100:.2f}%** round-trip costs and use conservative stop-first ordering if target and stop touch in the same 1h candle.', '',
        '## Results', '',
        '| Tier | Year | Signals | Weeks | Mature 30d | Median 30d MFE | Median 30d MAE | +10% hit | +20% hit | +20/-5 target rate | +20/-5 avg net |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    if not summary.empty:
        for r in summary.itertuples(index=False):
            def fpc(x): return 'n/a' if pd.isna(x) else f'{100*x:.1f}%'
            lines.append(f'| {r.tier} | {r.year} | {int(r.signals):,} | {int(r.weeks_with_signal):,} | {int(r.mature_30d_signals):,} | {fpc(r.median_mfe_30d)} | {fpc(r.median_mae_30d)} | {fpc(r.hit10_30d_rate)} | {fpc(r.hit20_30d_rate)} | {fpc(r.t20_s5_target_rate)} | {fpc(r.t20_s5_avg_net_return)} |')
    lines += ['', '## Interpretation guardrails', '',
              '- This is substantially less sample-conditioned than the previous event/noise studies, but it is not a pristine prospective test because the frozen rules were developed using portions of 2024-2026 history.',
              '- The rank 101-200 tier is a useful symbol-level distribution-shift check, but the universe is still based on symbols/liquidity available at replay-run time, so delisted-coin survivorship bias remains.',
              '- Recent signals without a full 30 days of forward candles are retained but excluded from 30-day target/stop return averages.',
              '- No result is promoted to the live strategy automatically.']
    (OUT / 'ANALYSIS.md').write_text('\n'.join(lines) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp.now(tz='UTC').floor('h') - pd.Timedelta(hours=1)
    universe = current_top200()
    universe.to_csv(OUT / 'universe.csv', index=False)
    models, thresholds, threshold_table = train_frozen_detector()
    threshold_table.to_csv(OUT / 'frozen_detector_thresholds.csv', index=False)
    r5rules, r7rules = load_frozen_rules()
    btc = load_extended('BTCUSDT', end)

    parts = []
    failures = []
    for meta in universe.itertuples(index=False):
        try:
            z = process_symbol(meta, btc, models, thresholds, r5rules, r7rules, end)
            if len(z): parts.append(z)
        except Exception as exc:
            failures.append({'symbol': meta.symbol, 'rank': meta.current_liquidity_rank, 'tier': meta.replay_tier, 'error': repr(exc)})
            print('ERROR', meta.symbol, repr(exc), flush=True)
    signals = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(signals):
        signals['signal_time'] = pd.to_datetime(signals['signal_time'], utc=True)
        signals = signals.sort_values(['signal_time', 'symbol']).reset_index(drop=True)
    signals.to_csv(OUT / 'signals.csv.gz', index=False, compression='gzip')
    pd.DataFrame(failures).to_csv(OUT / 'failures.csv', index=False)
    summary = summarise(signals)
    summary.to_csv(OUT / 'summary.csv', index=False)
    write_analysis(summary, signals, universe, end)
    manifest = {
        'study': 'Explosive Move Full-Population Replay Round 8',
        'branch': 'round-3',
        'replay_start': START_REPLAY.isoformat(),
        'replay_end': end.isoformat(),
        'universe': 'current eligible Binance USDT spot pairs ranked 1-200 by current 24h quote volume',
        'tiers': ['ORIGINAL_TOP100', 'EXTERNAL_101_200'],
        'signal': 'Round3 detector -> Round5 target_recall=0.90 rules -> 96h candidate -> Round7 union caps 3h=1,6h=0,12h=2,24h=1,48h=1',
        'evaluation': 'direct forward candle paths; no event catalogue labels used',
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print((OUT / 'ANALYSIS.md').read_text(), flush=True)


if __name__ == '__main__':
    main()
