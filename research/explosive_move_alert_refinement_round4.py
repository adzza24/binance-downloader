from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from binance_data import load_symbol
from explosive_move_discovery_round1 import FEATURES, build_features

SRC = Path('research/results/explosive_move_recall_round3')
OUT = Path('research/results/explosive_move_alert_refinement_round4')
CHECKPOINTS = OUT / 'checkpoints'
START = '2019-01-01'
END = '2026-08-31'
TARGETS = [0.95, 0.90, 0.85, 0.80, 0.70]
MAX_RULES = 5
MIN_SCOPE_POS = 40
MIN_SCOPE_NEG = 100

DELTA_FEATURES = [
    'ret_6h', 'ret_24h', 'ret_72h', 'range_24h', 'atr24_pct',
    'dist_30d_high', 'rebound_30d_low', 'volume_ratio_6h',
    'trade_ratio_6h', 'rs_24h'
]


def load_inputs():
    alerts = pd.read_csv(SRC / 'causal_alerts.csv.gz', compression='gzip')
    alerts['time'] = pd.to_datetime(alerts['time'], utc=True)
    alerts = alerts.reset_index(drop=True)
    alerts['alert_id'] = np.arange(len(alerts), dtype=np.int64)
    alerts['score_margin'] = pd.to_numeric(alerts['score'], errors='coerce') - pd.to_numeric(alerts['threshold'], errors='coerce')

    events = pd.read_csv(SRC / 'event_coverage.csv.gz', compression='gzip')
    for c in ['start_time', 'window_start', 'window_end']:
        events[c] = pd.to_datetime(events[c], utc=True)
    return alerts, events


def add_cross_sectional_features(alerts: pd.DataFrame) -> pd.DataFrame:
    out = alerts.copy()
    hourly = out.groupby('time').size().sort_index()
    rolling6 = hourly.rolling('6h', min_periods=1).sum()
    out['global_alerts_same_hour'] = out['time'].map(hourly).astype(float)
    out['global_alerts_6h'] = out['time'].map(rolling6).astype(float)

    fam_hourly = out.groupby(['family', 'time']).size().rename('n').reset_index()
    out = out.merge(fam_hourly.rename(columns={'n': 'family_alerts_same_hour'}), on=['family', 'time'], how='left')
    pieces = []
    for fam, g in fam_hourly.groupby('family'):
        s = g.set_index('time')['n'].sort_index().rolling('6h', min_periods=1).sum()
        z = pd.DataFrame({'family': fam, 'time': s.index, 'family_alerts_6h': s.values})
        pieces.append(z)
    if pieces:
        out = out.merge(pd.concat(pieces, ignore_index=True), on=['family', 'time'], how='left')
    else:
        out['family_alerts_6h'] = np.nan

    out = out.sort_values(['symbol', 'family', 'time'])
    out['hours_since_prev_alert'] = out.groupby(['symbol', 'family'])['time'].diff().dt.total_seconds() / 3600.0
    out['score_change_prev_alert'] = out.groupby(['symbol', 'family'])['score'].diff()
    return out.sort_values('alert_id').reset_index(drop=True)


def checkpoint_path(symbol: str) -> Path:
    safe = ''.join(ch for ch in symbol if ch.isalnum() or ch in {'-', '_'})
    return CHECKPOINTS / f'{safe}.csv.gz'


def enrich_symbol(symbol: str, rows: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    cp = checkpoint_path(symbol)
    if cp.exists():
        z = pd.read_csv(cp, compression='gzip')
        z['time'] = pd.to_datetime(z['time'], utc=True)
        print(symbol, 'feature checkpoint reused', flush=True)
        return z

    df = load_symbol(symbol, '1h', START, END)
    ff = build_features(df, btc).copy()
    ff['time'] = pd.to_datetime(ff['time'], utc=True)
    ff = ff.drop_duplicates('time').set_index('time').sort_index()

    base_cols = [c for c in FEATURES if c in ff.columns]
    target_times = pd.DatetimeIndex(rows['time'])
    x = ff.reindex(target_times)[base_cols].reset_index().rename(columns={'index': 'time'})
    x.insert(0, 'alert_id', rows['alert_id'].to_numpy())
    x.insert(1, 'symbol', symbol)

    for f in DELTA_FEATURES:
        if f not in ff.columns:
            continue
        for h in [3, 6, 12, 24]:
            shifted = ff[f].shift(h)
            vals = ff[f].reindex(target_times).to_numpy() - shifted.reindex(target_times).to_numpy()
            x[f'{f}_delta_{h}h'] = vals

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    x.to_csv(cp, index=False, compression='gzip')
    print(symbol, 'feature checkpoint saved', flush=True)
    return x


def reconstruct_features(alerts: pd.DataFrame) -> pd.DataFrame:
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    btc = load_symbol('BTCUSDT', '1h', START, END)
    parts = []
    failures = []
    for symbol, g in alerts.groupby('symbol', sort=False):
        try:
            parts.append(enrich_symbol(symbol, g.sort_values('alert_id'), btc))
        except Exception as e:
            failures.append({'symbol': symbol, 'error': repr(e)})
            print('ERROR', symbol, repr(e), flush=True)
    pd.DataFrame(failures).to_csv(OUT / 'feature_failures.csv', index=False)
    if not parts:
        raise RuntimeError('No feature checkpoints could be created')
    feats = pd.concat(parts, ignore_index=True)
    return alerts.merge(feats.drop(columns=['symbol'], errors='ignore'), on='alert_id', how='left', suffixes=('', '_feature'))


def build_alert_event_map(alerts: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by_symbol_alerts = {s: g.sort_values('time') for s, g in alerts.groupby('symbol')}
    for symbol, ev in events.groupby('symbol'):
        a = by_symbol_alerts.get(symbol)
        if a is None or a.empty:
            continue
        times = a['time'].to_numpy(dtype='datetime64[ns]')
        ids = a['alert_id'].to_numpy(dtype=np.int64)
        for r in ev.itertuples(index=False):
            lo = np.searchsorted(times, np.datetime64(r.window_start.tz_convert('UTC').tz_localize(None)), side='left')
            hi = np.searchsorted(times, np.datetime64(r.window_end.tz_convert('UTC').tz_localize(None)), side='right')
            if hi > lo:
                for aid in ids[lo:hi]:
                    rows.append((int(aid), r.event_id))
    m = pd.DataFrame(rows, columns=['alert_id', 'event_id']).drop_duplicates()
    return m


def mark_positive_alerts(alerts: pd.DataFrame, amap: pd.DataFrame) -> pd.DataFrame:
    positive_ids = set(amap['alert_id'].unique())
    out = alerts.copy()
    out['is_event_alert'] = out['alert_id'].isin(positive_ids)
    return out


def numeric_features(df: pd.DataFrame) -> list[str]:
    exclude = {
        'alert_id', 'year', 'threshold', 'matched_event_window', 'is_event_alert'
    }
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            usable = pd.to_numeric(df[c], errors='coerce').replace([np.inf, -np.inf], np.nan).notna().sum()
            if usable >= 500:
                cols.append(c)
    return cols


def safe_auc(y, x):
    mask = np.isfinite(x)
    if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y[mask], x[mask]))
    except Exception:
        return np.nan


def feature_contrasts(alerts: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for split, g in alerts.groupby('split2'):
        y = g['is_event_alert'].astype(int).to_numpy()
        for f in features:
            x = pd.to_numeric(g[f], errors='coerce').replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
            pos = x[y == 1]
            neg = x[y == 0]
            pos = pos[np.isfinite(pos)]
            neg = neg[np.isfinite(neg)]
            if len(pos) < 20 or len(neg) < 20:
                continue
            auc = safe_auc(y, x)
            direction = 'HIGHER' if pd.Series(pos).median() >= pd.Series(neg).median() else 'LOWER'
            separation = max(auc, 1 - auc) if np.isfinite(auc) else np.nan
            rows.append({
                'split': split, 'feature': f, 'positive_n': len(pos), 'noise_n': len(neg),
                'positive_median': float(np.nanmedian(pos)), 'noise_median': float(np.nanmedian(neg)),
                'auc_positive': auc, 'separation_auc': separation, 'direction': direction,
            })
    return pd.DataFrame(rows)


def stable_discriminators(contrasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    needed = ['DISCOVERY_2019_2022', 'VALIDATION_2023_2024', 'CONFIRMATION_2025_2026']
    for f, g in contrasts.groupby('feature'):
        d = {r.split: r for r in g.itertuples()}
        if not all(s in d for s in needed):
            continue
        dirs = [d[s].direction for s in needed]
        same = len(set(dirs)) == 1
        rows.append({
            'feature': f, 'direction': dirs[0] if same else 'MIXED', 'same_direction_all_splits': same,
            'discovery_auc': d['DISCOVERY_2019_2022'].separation_auc,
            'validation_auc': d['VALIDATION_2023_2024'].separation_auc,
            'confirmation_auc': d['CONFIRMATION_2025_2026'].separation_auc,
            'min_auc': min(d[s].separation_auc for s in needed),
            'mean_auc': np.mean([d[s].separation_auc for s in needed]),
        })
    return pd.DataFrame(rows).sort_values(['same_direction_all_splits', 'min_auc', 'mean_auc'], ascending=[False, False, False])


def event_index(events: pd.DataFrame):
    return events.set_index('event_id')['split2'].to_dict()


def evaluate_mask(mask: np.ndarray, alerts: pd.DataFrame, amap: pd.DataFrame, events: pd.DataFrame) -> dict:
    selected_ids = set(alerts.loc[mask, 'alert_id'].astype(int))
    if selected_ids:
        captured = set(amap.loc[amap['alert_id'].isin(selected_ids), 'event_id'])
    else:
        captured = set()
    out = {}
    for split, eg in events.groupby('split2'):
        eids = set(eg['event_id'])
        split_alerts = alerts['split2'].eq(split).to_numpy()
        noise = (~alerts['is_event_alert']).to_numpy() & split_alerts
        selected_noise = noise & mask
        out[split] = {
            'event_recall': len(captured & eids) / len(eids) if eids else np.nan,
            'captured_events': len(captured & eids),
            'events': len(eids),
            'noise_retention': selected_noise.sum() / max(noise.sum(), 1),
            'noise_alerts': int(selected_noise.sum()),
            'selected_alerts': int((mask & split_alerts).sum()),
        }
    return out


def candidate_rules(alerts: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    disc = alerts[alerts['split2'] == 'DISCOVERY_2019_2022']
    scopes = [('ALL', disc)] + [(fam, g) for fam, g in disc.groupby('family')]
    rows = []
    for scope, g in scopes:
        posg = g[g['is_event_alert']]
        negg = g[~g['is_event_alert']]
        if len(posg) < MIN_SCOPE_POS or len(negg) < MIN_SCOPE_NEG:
            continue
        for f in features:
            p = pd.to_numeric(posg[f], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
            n = pd.to_numeric(negg[f], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
            if len(p) < MIN_SCOPE_POS or len(n) < MIN_SCOPE_NEG:
                continue
            direction = 'GE' if p.median() >= n.median() else 'LE'
            qs = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50] if direction == 'GE' else [0.95, 0.90, 0.80, 0.70, 0.60, 0.50]
            for q in qs:
                t = float(p.quantile(q))
                if not np.isfinite(t):
                    continue
                rows.append({'scope': scope, 'feature': f, 'direction': direction, 'threshold': t, 'source_quantile': q})
    return pd.DataFrame(rows).drop_duplicates(['scope', 'feature', 'direction', 'threshold'])


def apply_rule(alerts: pd.DataFrame, r) -> np.ndarray:
    x = pd.to_numeric(alerts[r.feature], errors='coerce').to_numpy(dtype=float)
    keep = x >= r.threshold if r.direction == 'GE' else x <= r.threshold
    keep &= np.isfinite(x)
    if r.scope != 'ALL':
        scope_mask = alerts['family'].eq(r.scope).to_numpy()
        keep = (~scope_mask) | keep
    return keep


def greedy_ladder(alerts: pd.DataFrame, amap: pd.DataFrame, events: pd.DataFrame, candidates: pd.DataFrame):
    rows = []
    selected_rule_rows = []
    for target in TARGETS:
        current = np.ones(len(alerts), dtype=bool)
        chosen = []
        base_eval = evaluate_mask(current, alerts, amap, events)
        for step in range(1, MAX_RULES + 1):
            best = None
            best_mask = None
            best_eval = None
            best_noise = math.inf
            for idx, r in candidates.iterrows():
                if idx in chosen:
                    continue
                trial = current & apply_rule(alerts, r)
                ev = evaluate_mask(trial, alerts, amap, events)
                drec = ev['DISCOVERY_2019_2022']['event_recall']
                vrec = ev['VALIDATION_2023_2024']['event_recall']
                if drec < max(target - 0.02, 0) or vrec < target:
                    continue
                noise = ev['VALIDATION_2023_2024']['noise_retention']
                if noise < best_noise:
                    best = idx
                    best_mask = trial
                    best_eval = ev
                    best_noise = noise
            if best is None:
                break
            prev_noise = evaluate_mask(current, alerts, amap, events)['VALIDATION_2023_2024']['noise_retention']
            if prev_noise - best_noise < 0.005:
                break
            current = best_mask
            chosen.append(best)
            rr = candidates.loc[best]
            selected_rule_rows.append({
                'target_recall': target, 'step': step, 'scope': rr.scope, 'feature': rr.feature,
                'direction': rr.direction, 'threshold': rr.threshold,
            })

        final_eval = evaluate_mask(current, alerts, amap, events)
        for split, metrics in final_eval.items():
            rows.append({
                'target_recall': target, 'rules': len(chosen), 'split': split,
                **metrics,
            })
    return pd.DataFrame(rows), pd.DataFrame(selected_rule_rows)


def write_analysis(stable: pd.DataFrame, ladder: pd.DataFrame, rules: pd.DataFrame, alerts: pd.DataFrame, events: pd.DataFrame):
    lines = [
        '# Explosive Move Alert Refinement Round 4', '',
        '**Status:** RESEARCH ONLY on isolated `round-3` branch. No live strategy or automation changes.', '',
        '## Question', '',
        'Starting from the very-high-recall Round 3 detector, which contemporaneous factors distinguish alerts that occur inside one of the original 8,683 explosive-event windows from unmatched/noise alerts?', '',
        f'- Alert population analysed: **{len(alerts):,}**.',
        f'- Original event population: **{len(events):,}**.',
        '- Discovery candidate thresholds are derived from 2019-2022 only.',
        '- The filter ladder is selected using 2023-2024 validation subject to explicit event-recall floors.',
        '- 2025-2026 is reported as previously exposed confirmation data and is never used to choose rules.', '',
        '## Most stable individual discriminators', '',
        '| Feature | Direction | Discovery AUC | Validation AUC | Confirmation AUC |',
        '|---|---|---:|---:|---:|'
    ]
    top = stable[stable.same_direction_all_splits].head(15)
    for _, r in top.iterrows():
        lines.append(f"| {r.feature} | {r.direction} | {r.discovery_auc:.3f} | {r.validation_auc:.3f} | {r.confirmation_auc:.3f} |")

    lines += ['', '## Recall / noise Pareto ladder', '',
              '| Target | Split | Rules | Event recall | Noise retained | Selected alerts |',
              '|---:|---|---:|---:|---:|---:|']
    for _, r in ladder.iterrows():
        lines.append(f"| {100*r.target_recall:.0f}% | {r['split']} | {int(r.rules)} | {100*r.event_recall:.1f}% | {100*r.noise_retention:.1f}% | {int(r.selected_alerts):,} |")

    lines += ['', '## Selected interpretable rules', '']
    if rules.empty:
        lines.append('No conjunction improved validation noise by at least 0.5 percentage points while respecting the requested recall floors.')
    else:
        for target, g in rules.groupby('target_recall'):
            lines.append(f'### {100*target:.0f}% recall target')
            lines.append('')
            for r in g.sort_values('step').itertuples():
                op = '>=' if r.direction == 'GE' else '<='
                prefix = 'all alerts' if r.scope == 'ALL' else f'{r.scope} alerts only'
                lines.append(f'- Step {r.step}: for {prefix}, require `{r.feature} {op} {r.threshold:.6g}`.')
            lines.append('')

    lines += ['## Interpretation guardrails', '',
              '- Positive alert rows are not independent observations. Event recall, not alert-level accuracy, is therefore the primary selection metric.',
              '- Several alerts may map to one event; retaining any qualifying alert retains that event.',
              '- AUC is descriptive only. Rules are selected using event-level recall and noise-alert retention.',
              '- The ladder is a research filter study, not an entry or exit strategy.',
              '- No result is promoted to the live strategy automatically.', '']
    (OUT / 'ANALYSIS.md').write_text('\n'.join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    alerts, events = load_inputs()
    alerts = add_cross_sectional_features(alerts)
    enriched = reconstruct_features(alerts)
    amap = build_alert_event_map(enriched, events)
    enriched = mark_positive_alerts(enriched, amap)

    feats = numeric_features(enriched)
    contrasts = feature_contrasts(enriched, feats)
    stable = stable_discriminators(contrasts)
    candidates = candidate_rules(enriched, feats)
    ladder, selected_rules = greedy_ladder(enriched, amap, events, candidates)

    enriched.to_csv(OUT / 'enriched_alerts.csv.gz', index=False, compression='gzip')
    amap.to_csv(OUT / 'alert_event_map.csv.gz', index=False, compression='gzip')
    contrasts.to_csv(OUT / 'feature_contrasts.csv', index=False)
    stable.to_csv(OUT / 'stable_discriminators.csv', index=False)
    candidates.to_csv(OUT / 'candidate_rules.csv.gz', index=False, compression='gzip')
    ladder.to_csv(OUT / 'filter_ladder.csv', index=False)
    selected_rules.to_csv(OUT / 'selected_rules.csv', index=False)

    manifest = {
        'study': 'Explosive Move Alert Refinement Round 4',
        'branch': 'round-3',
        'status': 'RESEARCH ONLY - no live strategy or automation edits',
        'source_alerts': str(SRC / 'causal_alerts.csv.gz'),
        'source_events': str(SRC / 'event_coverage.csv.gz'),
        'alert_count': int(len(enriched)),
        'event_count': int(len(events)),
        'feature_count': int(len(feats)),
        'targets': TARGETS,
        'max_rules': MAX_RULES,
        'selection': 'Candidate threshold directions and values from 2019-2022 discovery; greedy conjunction selected on 2023-2024 validation while preserving explicit event-recall floors; 2025-2026 confirmation reporting only.',
        'checkpointing': 'Per-symbol reconstructed feature snapshots are checkpointed and reusable after failure.',
        'caveats': [
            '2025-2026 is previously exposed confirmation data, not a pristine holdout',
            'Round 1 current-universe survivorship bias remains',
            'alert observations are clustered within events and market episodes; event recall is the primary metric',
        ],
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    write_analysis(stable, ladder, selected_rules, enriched, events)

    print(ladder.to_string(index=False), flush=True)
    print('\nTOP STABLE DISCRIMINATORS', flush=True)
    print(stable.head(20).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
