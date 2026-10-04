from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

import explosive_move_recall_round3 as base

CHECKPOINT_DIR = base.OUT / 'checkpoints'
ALERT_COLUMNS = ['symbol', 'time', 'year', 'split2', 'family', 'score', 'threshold']


def train_family_models(assignments: pd.DataFrame, samples: pd.DataFrame):
    family_by_event = assignments.set_index('event_id')['family']
    work = samples[samples.label.isin(['WINNER', 'HARD_FAIL', 'RANDOM_CONTROL'])].copy()
    work['family'] = work['event_id'].map(family_by_event)
    models = {}
    rows = []
    importances = []

    for fam in ['CAPITULATION', 'BASE', 'MOMENTUM']:
        tr = work[(work.family == fam) & (work.split2 == 'DISCOVERY_2019_2022')].copy()
        tr['y'] = (tr.label == 'WINNER').astype(int)
        if tr.empty or tr.y.nunique() < 2:
            continue

        imp = SimpleImputer(strategy='median')
        X = imp.fit_transform(tr[base.FEATURES])
        y = tr.y.to_numpy()
        rf = RandomForestClassifier(
            n_estimators=350, max_depth=8, min_samples_leaf=20,
            max_features='sqrt', class_weight={0: 1.0, 1: 2.0},
            random_state=73, n_jobs=-1
        )
        rf.fit(X, y)
        models[fam] = (imp, rf)

        for f, v in zip(base.FEATURES, rf.feature_importances_):
            importances.append({'family': fam, 'feature': f, 'importance': float(v)})

        va = work[(work.family == fam) & (work.split2 == 'VALIDATION_2023_2024')].copy()
        selection_source = 'VALIDATION_2023_2024'
        score_frame = va
        if va.empty or not (va.label == 'WINNER').any():
            selection_source = 'DISCOVERY_2019_2022_FALLBACK'
            score_frame = tr.copy()

        score_frame['score'] = rf.predict_proba(imp.transform(score_frame[base.FEATURES]))[:, 1]
        winner_scores = score_frame[score_frame.label == 'WINNER'].groupby('event_id').score.max()
        control_scores = score_frame[score_frame.label != 'WINNER'].groupby(['symbol', 'pseudo_start_time']).score.max()

        family_rows = []
        for t in base.THRESHOLDS:
            recall = float((winner_scores >= t).mean()) if len(winner_scores) else np.nan
            control_fire = float((control_scores >= t).mean()) if len(control_scores) else 1.0
            family_rows.append({
                'family': fam,
                'threshold': float(t),
                'selection_source': selection_source,
                'validation_events': int(len(winner_scores)),
                'validation_recall_proxy': recall,
                'validation_control_fire_rate': control_fire,
            })

        cand = pd.DataFrame(family_rows)
        eligible = cand[cand.validation_recall_proxy >= base.TARGET_VALIDATION_RECALL]
        if len(eligible):
            best = eligible.sort_values(
                ['validation_control_fire_rate', 'threshold'], ascending=[True, False]
            ).iloc[0]
        else:
            cand['score_obj'] = cand.validation_recall_proxy.fillna(0) - 0.25 * cand.validation_control_fire_rate.fillna(1)
            best = cand.sort_values(['score_obj', 'validation_recall_proxy'], ascending=False).iloc[0]

        for r in family_rows:
            r['selected'] = bool(np.isclose(r['threshold'], float(best.threshold)))
            rows.append(r)

    return models, pd.DataFrame(rows), pd.DataFrame(importances)


def checkpoint_path(symbol: str) -> Path:
    safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', symbol)
    digest = hashlib.sha1(symbol.encode('utf-8')).hexdigest()[:8]
    return CHECKPOINT_DIR / f'{safe}-{digest}.csv.gz'


def load_checkpoint_parts():
    parts = []
    for path in sorted(CHECKPOINT_DIR.glob('*.csv.gz')):
        try:
            part = pd.read_csv(path)
            if len(part):
                part['time'] = pd.to_datetime(part['time'], utc=True)
            parts.append(part.reindex(columns=ALERT_COLUMNS))
        except Exception as exc:
            print('CHECKPOINT_READ_ERROR', path.name, repr(exc), flush=True)
    return parts


def causal_replay(universe: pd.DataFrame, models, thresholds):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    btc = base.load_symbol('BTCUSDT', '1h', base.START, base.END)
    alert_parts = load_checkpoint_parts()
    failures = []

    for _, meta in universe.iterrows():
        symbol = meta.symbol
        cp = checkpoint_path(symbol)
        if cp.exists():
            print(symbol, 'checkpoint reused', flush=True)
            continue
        try:
            df = base.load_symbol(symbol, '1h', base.START, base.END)
            if len(df) < 3000:
                failures.append({'symbol': symbol, 'error': f'insufficient_history:{len(df)}'})
                continue
            ff = base.build_features(df, btc)
            valid = ff[base.FEATURES].notna().mean(axis=1).to_numpy() >= 0.80
            times = pd.to_datetime(ff.time, utc=True)
            symbol_rows = []
            for fam, (imp, rf) in models.items():
                if fam not in thresholds:
                    continue
                scores = rf.predict_proba(imp.transform(ff[base.FEATURES]))[:, 1]
                idx = np.flatnonzero(valid & (scores >= thresholds[fam]))
                last = -10**9
                for i in idx:
                    if int(i) - last < base.ALERT_COOLDOWN_HOURS:
                        continue
                    last = int(i)
                    ts = times.iloc[int(i)]
                    symbol_rows.append({
                        'symbol': symbol, 'time': ts, 'year': int(ts.year),
                        'split2': base.split_for_year(int(ts.year)),
                        'family': fam, 'score': float(scores[int(i)]),
                        'threshold': thresholds[fam]
                    })
            part = pd.DataFrame(symbol_rows, columns=ALERT_COLUMNS)
            part.to_csv(cp, index=False, compression='gzip')
            alert_parts.append(part)
            print(symbol, 'complete checkpoint saved', flush=True)
        except Exception as e:
            failures.append({'symbol': symbol, 'error': repr(e)})
            print('ERROR', symbol, repr(e), flush=True)

    alerts = pd.concat(alert_parts, ignore_index=True) if alert_parts else pd.DataFrame(columns=ALERT_COLUMNS)
    if len(alerts):
        alerts['time'] = pd.to_datetime(alerts['time'], utc=True)
        alerts = alerts.drop_duplicates(['symbol', 'time', 'family']).sort_values(['time', 'symbol', 'family']).reset_index(drop=True)
    return alerts, pd.DataFrame(failures)


def attach_event_coverage(assignments: pd.DataFrame, alerts: pd.DataFrame):
    out = assignments.copy()
    out['start_time'] = pd.to_datetime(out['start_time'], utc=True)
    alerts = alerts.copy()
    alerts['time'] = pd.to_datetime(alerts['time'], utc=True)
    out['window_start'] = out.start_time - pd.to_timedelta(base.PRE_EVENT_HOURS, unit='h')
    end_h = pd.to_numeric(out.hours_to_5pct, errors='coerce').fillna(out.hours_to_20pct)
    out['window_end'] = out.start_time + pd.to_timedelta(end_h, unit='h')
    out['captured'] = False
    out['capture_family'] = ''
    out['first_alert_time'] = pd.Series(pd.NaT, index=out.index, dtype='datetime64[ns, UTC]')
    out['hours_from_event_start'] = np.nan

    matched_alert_ids = set()
    alerts = alerts.reset_index(drop=True)
    by_symbol = {s: g for s, g in alerts.groupby('symbol')}

    for i, e in out.iterrows():
        g = by_symbol.get(e.symbol)
        if g is None:
            continue
        m = g[(g.time >= e.window_start) & (g.time <= e.window_end)]
        if m.empty:
            continue
        first = m.sort_values('time').iloc[0]
        first_time = pd.Timestamp(first['time'])
        out.at[i, 'captured'] = True
        out.at[i, 'capture_family'] = first['family']
        out.at[i, 'first_alert_time'] = first_time
        out.at[i, 'hours_from_event_start'] = (first_time - e.start_time).total_seconds() / 3600.0
        matched_alert_ids.update(m.index.tolist())

    alerts['matched_event_window'] = alerts.index.isin(matched_alert_ids)
    return out, alerts


_original_write_analysis = base.write_analysis


def write_analysis(assignments, thresholds, coverage, episode_summary):
    _original_write_analysis(assignments, thresholds, coverage, episode_summary)
    selected = thresholds[thresholds.selected]
    fallback = selected[selected.selection_source != 'VALIDATION_2023_2024'] if 'selection_source' in selected else pd.DataFrame()
    path = base.OUT / 'ANALYSIS.md'
    with path.open('a') as f:
        f.write('\n## Replay checkpointing\n\n')
        f.write('Causal replay is checkpointed per symbol under `checkpoints/`. A rerun reuses completed symbol files and retries only unfinished symbols. Checkpoints are persisted to the isolated `round-3` branch even if a later analysis step fails.\n')
        if not fallback.empty:
            f.write('\n## Threshold-selection fallback\n\n')
            f.write('One or more signal families had no usable 2023-2024 validation winner rows. Those families used discovery-era data only for threshold selection rather than touching 2025-2026 confirmation data.\n\n')
            for _, r in fallback.iterrows():
                f.write(f"- {r.family}: {r.selection_source}, threshold {r.threshold:.2f}.\n")


def patch_manifest():
    path = base.OUT / 'manifest.json'
    if not path.exists():
        return
    manifest = json.loads(path.read_text())
    threshold_path = base.OUT / 'threshold_selection.csv'
    if threshold_path.exists():
        t = pd.read_csv(threshold_path)
        selected = t[t.selected.astype(str).str.lower().isin(['true', '1'])]
        if 'selection_source' in selected.columns:
            manifest['threshold_selection_sources'] = selected[['family', 'selection_source', 'threshold']].to_dict('records')
    manifest['threshold_selection'] = 'Prefer 2023-2024 validation with >=80% event recall proxy then lowest control fire rate. If a family has no usable validation winner rows, use discovery-era threshold selection for that family only; never use 2025-2026 confirmation for selection.'
    manifest['checkpointing'] = 'Per-symbol causal replay checkpoints are written after each successful symbol and persisted to the isolated round-3 branch even if a later stage fails. Reruns reuse completed symbols.'
    path.write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    base.train_family_models = train_family_models
    base.causal_replay = causal_replay
    base.attach_event_coverage = attach_event_coverage
    base.write_analysis = write_analysis
    base.main()
    patch_manifest()
