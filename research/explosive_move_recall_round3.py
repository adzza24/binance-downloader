from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from binance_data import load_symbol
from explosive_move_discovery_round1 import FEATURES, build_features

SRC = Path('research/results/explosive_move_discovery_round1')
OUT = Path('research/results/explosive_move_recall_round3')
START = '2019-01-01'
END = '2026-08-31'
LEADS = [0, 6, 12, 24, 72]
PRE_EVENT_HOURS = 72
ALERT_COOLDOWN_HOURS = 24
TARGET_VALIDATION_RECALL = 0.80
THRESHOLDS = np.round(np.arange(0.10, 0.951, 0.05), 2)

CLUSTER_FEATURES = [
    'ret_6h', 'ret_24h', 'ret_72h', 'ret_30d',
    'range_24h', 'range_7d', 'range_30d',
    'atr24_pct', 'atr7d_pct', 'dist_30d_high',
    'rebound_30d_low', 'volume_ratio_6h', 'trade_ratio_6h', 'rs_24h'
]


def split_for_year(y: int) -> str:
    if y <= 2022:
        return 'DISCOVERY_2019_2022'
    if y <= 2024:
        return 'VALIDATION_2023_2024'
    return 'CONFIRMATION_2025_2026'


def load_inputs():
    events = pd.read_csv(SRC / 'events.csv', parse_dates=['start_time', 'target_time'])
    events['split2'] = events['year'].astype(int).map(split_for_year)
    universe = pd.read_csv(SRC / 'universe.csv')
    usecols = ['event_id', 'label', 'symbol', 'lead_hours', 'pseudo_start_time', 'split'] + FEATURES
    samples = pd.read_csv(SRC / 'precursor_samples.csv.gz', usecols=usecols, compression='gzip')
    samples = samples[samples['lead_hours'].isin(LEADS)].copy()
    samples['pseudo_start_time'] = pd.to_datetime(samples['pseudo_start_time'], utc=True)
    samples['split2'] = samples['split'].replace({'HOLDOUT_2025_2026': 'CONFIRMATION_2025_2026'})
    return events, samples, universe


def build_families(events: pd.DataFrame, samples: pd.DataFrame):
    lead0 = samples[(samples.label == 'WINNER') & (samples.lead_hours == 0)]
    lead0 = lead0.drop_duplicates('event_id').set_index('event_id')
    disc_ids = events.loc[events.split2 == 'DISCOVERY_2019_2022', 'event_id']
    train = lead0.reindex(disc_ids).dropna(how='all', subset=CLUSTER_FEATURES)

    imp = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    x = scaler.fit_transform(imp.fit_transform(train[CLUSTER_FEATURES]))
    km = KMeans(n_clusters=3, random_state=73, n_init=30)
    train_clusters = km.fit_predict(x)

    train_named = train[['ret_30d']].copy()
    train_named['cluster'] = train_clusters
    med = train_named.groupby('cluster')['ret_30d'].median().sort_values()
    low = int(med.index[0])
    high = int(med.index[-1])
    middle = int([c for c in med.index if c not in {low, high}][0])
    names = {low: 'CAPITULATION', middle: 'BASE', high: 'MOMENTUM'}

    all0 = lead0.reindex(events.event_id)
    xa = scaler.transform(imp.transform(all0[CLUSTER_FEATURES]))
    assigned = km.predict(xa)
    out = events[['event_id', 'symbol', 'start_time', 'year', 'split2', 'hours_to_5pct', 'hours_to_20pct']].copy()
    out['family_cluster'] = assigned
    out['family'] = [names[int(c)] for c in assigned]
    return out


def family_profiles(assignments: pd.DataFrame, samples: pd.DataFrame):
    start = samples[(samples.label == 'WINNER') & (samples.lead_hours == 0)].drop_duplicates('event_id')
    z = assignments[['event_id', 'family', 'split2']].merge(start[['event_id'] + FEATURES], on='event_id', how='left')
    rows = []
    profile_features = ['ret_6h','ret_24h','ret_72h','ret_30d','range_24h','range_7d','atr24_pct','atr7d_pct','dist_30d_high','rebound_30d_low','volume_ratio_6h','trade_ratio_6h','rs_24h']
    for fam, g in z.groupby('family'):
        row = {'family': fam, 'events': len(g)}
        for f in profile_features:
            row[f'median_{f}'] = pd.to_numeric(g[f], errors='coerce').median()
        rows.append(row)
    return pd.DataFrame(rows)


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
        X = imp.fit_transform(tr[FEATURES])
        y = tr.y.to_numpy()
        rf = RandomForestClassifier(
            n_estimators=350, max_depth=8, min_samples_leaf=20,
            max_features='sqrt', class_weight={0: 1.0, 1: 2.0},
            random_state=73, n_jobs=-1
        )
        rf.fit(X, y)
        models[fam] = (imp, rf)

        for f, v in zip(FEATURES, rf.feature_importances_):
            importances.append({'family': fam, 'feature': f, 'importance': float(v)})

        va = work[(work.family == fam) & (work.split2 == 'VALIDATION_2023_2024')].copy()
        va['score'] = rf.predict_proba(imp.transform(va[FEATURES]))[:, 1]
        winner_scores = va[va.label == 'WINNER'].groupby('event_id').score.max()
        control_scores = va[va.label != 'WINNER'].groupby(['symbol', 'pseudo_start_time']).score.max()

        family_rows = []
        for t in THRESHOLDS:
            recall = float((winner_scores >= t).mean()) if len(winner_scores) else np.nan
            control_fire = float((control_scores >= t).mean()) if len(control_scores) else np.nan
            family_rows.append({
                'family': fam, 'threshold': float(t),
                'validation_events': int(len(winner_scores)),
                'validation_recall_proxy': recall,
                'validation_control_fire_rate': control_fire
            })
        cand = pd.DataFrame(family_rows)
        eligible = cand[cand.validation_recall_proxy >= TARGET_VALIDATION_RECALL]
        if len(eligible):
            best = eligible.sort_values(['validation_control_fire_rate', 'threshold'], ascending=[True, False]).iloc[0]
        else:
            cand['score_obj'] = cand.validation_recall_proxy - 0.25 * cand.validation_control_fire_rate
            best = cand.sort_values(['score_obj', 'validation_recall_proxy'], ascending=False).iloc[0]
        for r in family_rows:
            r['selected'] = bool(np.isclose(r['threshold'], float(best.threshold)))
            rows.append(r)

    return models, pd.DataFrame(rows), pd.DataFrame(importances)


def selected_thresholds(thresholds: pd.DataFrame):
    return {r.family: float(r.threshold) for _, r in thresholds[thresholds.selected].iterrows()}


def causal_replay(universe: pd.DataFrame, models, thresholds):
    btc = load_symbol('BTCUSDT', '1h', START, END)
    alert_rows = []
    failures = []

    for _, meta in universe.iterrows():
        symbol = meta.symbol
        try:
            df = load_symbol(symbol, '1h', START, END)
            if len(df) < 3000:
                failures.append({'symbol': symbol, 'error': f'insufficient_history:{len(df)}'})
                continue
            ff = build_features(df, btc)
            valid = ff[FEATURES].notna().mean(axis=1).to_numpy() >= 0.80
            times = pd.to_datetime(ff.time, utc=True)
            for fam, (imp, rf) in models.items():
                if fam not in thresholds:
                    continue
                scores = rf.predict_proba(imp.transform(ff[FEATURES]))[:, 1]
                idx = np.flatnonzero(valid & (scores >= thresholds[fam]))
                last = -10**9
                for i in idx:
                    if int(i) - last < ALERT_COOLDOWN_HOURS:
                        continue
                    last = int(i)
                    ts = times.iloc[int(i)]
                    alert_rows.append({
                        'symbol': symbol, 'time': ts, 'year': int(ts.year),
                        'split2': split_for_year(int(ts.year)),
                        'family': fam, 'score': float(scores[int(i)]),
                        'threshold': thresholds[fam]
                    })
            print(symbol, 'complete', flush=True)
        except Exception as e:
            failures.append({'symbol': symbol, 'error': repr(e)})
            print('ERROR', symbol, repr(e), flush=True)

    return pd.DataFrame(alert_rows), pd.DataFrame(failures)


def attach_event_coverage(assignments: pd.DataFrame, alerts: pd.DataFrame):
    out = assignments.copy()
    out['window_start'] = out.start_time - pd.to_timedelta(PRE_EVENT_HOURS, unit='h')
    end_h = pd.to_numeric(out.hours_to_5pct, errors='coerce').fillna(out.hours_to_20pct)
    out['window_end'] = out.start_time + pd.to_timedelta(end_h, unit='h')
    out['captured'] = False
    out['capture_family'] = ''
    out['first_alert_time'] = pd.NaT
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
        out.at[i, 'captured'] = True
        out.at[i, 'capture_family'] = first.family
        out.at[i, 'first_alert_time'] = first.time
        out.at[i, 'hours_from_event_start'] = (first.time - e.start_time).total_seconds() / 3600.0
        matched_alert_ids.update(m.index.tolist())

    alerts['matched_event_window'] = alerts.index.isin(matched_alert_ids)
    return out, alerts


def episode_table(covered: pd.DataFrame):
    z = covered.sort_values('start_time').copy()
    episode = 0
    episode_start = None
    ids = []
    for t in z.start_time:
        if episode_start is None or t > episode_start + pd.Timedelta(hours=48):
            episode += 1
            episode_start = t
        ids.append(episode)
    z['episode_id'] = ids
    rows = []
    for eid, g in z.groupby('episode_id'):
        rows.append({
            'episode_id': int(eid), 'start_time': g.start_time.min(), 'end_time': g.start_time.max(),
            'split2': g.iloc[0].split2, 'events': len(g), 'symbols': g.symbol.nunique(),
            'captured_events': int(g.captured.sum()), 'event_recall': float(g.captured.mean()),
            'captured_any': bool(g.captured.any()),
            'market_wide': bool(g.symbol.nunique() >= 3)
        })
    return pd.DataFrame(rows)


def summaries(covered: pd.DataFrame, alerts: pd.DataFrame, episodes: pd.DataFrame):
    cov_rows = []
    for split, g in covered.groupby('split2'):
        a = alerts[alerts.split2 == split]
        weeks = max((g.start_time.max() - g.start_time.min()).total_seconds() / (7 * 86400), 1)
        cov_rows.append({
            'split': split, 'events': len(g), 'captured_events': int(g.captured.sum()),
            'event_recall': float(g.captured.mean()),
            'alerts': len(a), 'alerts_per_week': len(a) / weeks,
            'false_alerts': int((~a.matched_event_window).sum()),
            'false_alerts_per_week': int((~a.matched_event_window).sum()) / weeks,
            'median_hours_from_event_start': pd.to_numeric(g.loc[g.captured, 'hours_from_event_start'], errors='coerce').median()
        })
    coverage = pd.DataFrame(cov_rows)

    year_rows = []
    for y, g in covered.groupby('year'):
        year_rows.append({'year': int(y), 'events': len(g), 'captured': int(g.captured.sum()), 'event_recall': float(g.captured.mean())})
    year_summary = pd.DataFrame(year_rows)

    ep_rows = []
    for split, g in episodes.groupby('split2'):
        major = g[g.market_wide]
        ep_rows.append({
            'split': split, 'episodes': len(g), 'episodes_with_any_capture': int(g.captured_any.sum()),
            'episode_any_recall': float(g.captured_any.mean()) if len(g) else np.nan,
            'market_wide_episodes': len(major),
            'market_wide_any_recall': float(major.captured_any.mean()) if len(major) else np.nan,
            'median_episode_event_recall': float(g.event_recall.median()) if len(g) else np.nan
        })
    return coverage, year_summary, pd.DataFrame(ep_rows)


def write_analysis(assignments, thresholds, coverage, episode_summary):
    selected = thresholds[thresholds.selected].copy()
    counts = assignments.family.value_counts()
    lines = [
        '# Explosive Move Recall Round 3', '',
        '**Status:** RESEARCH ONLY on isolated `round-3` branch. No live strategy or automation changes.', '',
        '## Purpose', '',
        'Reset the entry research around recall. The fixed target population is the original Round 1 catalogue of 8,683 de-duplicated +20%-before--5 events. Three precursor families are learned from discovery-era winner conditions, then separate family classifiers are trained to recognise those conditions. Thresholds are selected on 2023-2024 with an explicit preference for at least 80% family-level event recall before reducing control fire rate.', '',
        'An event is counted as captured when a causal alert appears on the same symbol from 72 hours before the original event start through that event\'s +5% milestone. This evaluation window uses hindsight only for scoring recall; the alert itself is generated from contemporaneous features.', '',
        '## Learned families', '',
        '| Family | Events |', '|---|---:|'
    ]
    for fam in ['CAPITULATION','BASE','MOMENTUM']:
        lines.append(f'| {fam} | {int(counts.get(fam, 0))} |')

    lines += ['', '## Validation-selected family thresholds', '',
              '| Family | Threshold | Validation event recall proxy | Validation control fire |',
              '|---|---:|---:|---:|']
    for _, r in selected.sort_values('family').iterrows():
        lines.append(f"| {r.family} | {r.threshold:.2f} | {100*r.validation_recall_proxy:.1f}% | {100*r.validation_control_fire_rate:.1f}% |")

    lines += ['', '## End-to-end causal recall', '',
              '| Split | Events | Captured | Event recall | Alerts/week | False alerts/week | Median timing vs event start |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for _, r in coverage.iterrows():
        lines.append(f"| {r['split']} | {int(r.events)} | {int(r.captured_events)} | {100*r.event_recall:.1f}% | {r.alerts_per_week:.1f} | {r.false_alerts_per_week:.1f} | {r.median_hours_from_event_start:.1f}h |")

    lines += ['', '## Independent episode check', '',
              'Events are also grouped into fixed 48-hour market episodes so simultaneous altcoin moves are not mistaken for independent evidence.', '',
              '| Split | Episodes | Any captured | Market-wide episodes (3+ symbols) | Market-wide any captured | Median event recall inside episode |',
              '|---|---:|---:|---:|---:|---:|']
    for _, r in episode_summary.iterrows():
        lines.append(f"| {r['split']} | {int(r.episodes)} | {100*r.episode_any_recall:.1f}% | {int(r.market_wide_episodes)} | {100*r.market_wide_any_recall:.1f}% | {100*r.median_episode_event_recall:.1f}% |")

    lines += ['', '## Guardrails', '',
              '- This is recall-first discovery, not a deployable strategy.',
              '- False-alert rate, timing and drawdown remain diagnostics; they are not allowed to silently shrink the target population to a tiny high-precision subset.',
              '- The original event catalogue itself is defined as +20% before -5%; this round does not use -5% as an exit or entry-selection rule.',
              '- 2025-2026 has been seen in prior research and is confirmation data, not a pristine holdout.',
              '- Current-universe survivorship bias from Round 1 remains.',
              '- No result is promoted to the live strategy automatically.', '']
    (OUT / 'ANALYSIS.md').write_text('\n'.join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    events, samples, universe = load_inputs()
    assignments = build_families(events, samples)
    profiles = family_profiles(assignments, samples)
    models, thresholds, importances = train_family_models(assignments, samples)
    chosen = selected_thresholds(thresholds)

    alerts, failures = causal_replay(universe, models, chosen)
    if alerts.empty:
        raise RuntimeError('Causal replay produced no alerts')
    alerts['time'] = pd.to_datetime(alerts.time, utc=True)
    covered, alerts = attach_event_coverage(assignments, alerts)
    episodes = episode_table(covered)
    coverage, year_summary, episode_summary = summaries(covered, alerts, episodes)

    assignments.to_csv(OUT / 'event_family_assignments.csv', index=False)
    profiles.to_csv(OUT / 'family_profiles.csv', index=False)
    thresholds.to_csv(OUT / 'threshold_selection.csv', index=False)
    importances.sort_values(['family','importance'], ascending=[True,False]).to_csv(OUT / 'model_feature_importance.csv', index=False)
    alerts.to_csv(OUT / 'causal_alerts.csv.gz', index=False, compression='gzip')
    covered.to_csv(OUT / 'event_coverage.csv.gz', index=False, compression='gzip')
    coverage.to_csv(OUT / 'coverage_summary.csv', index=False)
    year_summary.to_csv(OUT / 'year_summary.csv', index=False)
    episodes.to_csv(OUT / 'episode_detail.csv', index=False)
    episode_summary.to_csv(OUT / 'episode_summary.csv', index=False)
    failures.to_csv(OUT / 'failures.csv', index=False)
    write_analysis(assignments, thresholds, coverage, episode_summary)

    manifest = {
        'study': 'Explosive Move Recall Round 3',
        'branch': 'round-3',
        'status': 'RESEARCH ONLY - no live strategy or automation edits',
        'source_events': 'research/results/explosive_move_discovery_round1/events.csv',
        'source_event_count': int(len(events)),
        'families': 3,
        'family_method': 'KMeans(3) fit only on 2019-2022 winner start features; CAPITULATION/BASE/MOMENTUM names assigned by discovery median 30d return.',
        'model': 'One RandomForestClassifier per family trained on 2019-2022 winner precursor snapshots vs same-family HARD_FAIL/RANDOM_CONTROL samples.',
        'positive_leads_hours': LEADS,
        'threshold_selection': f'2023-2024 validation; prefer >= {TARGET_VALIDATION_RECALL:.0%} event recall proxy then lowest control fire rate.',
        'causal_event_capture_window': f'{PRE_EVENT_HOURS}h before original event start through +5% milestone.',
        'alert_cooldown_hours_per_symbol_family': ALERT_COOLDOWN_HOURS,
        'episode_definition': 'Fixed 48h windows starting from the first event in each episode; market-wide means >=3 symbols.',
        'history': {'start': START, 'end': END, 'interval': '1h'},
        'caveats': ['Round 1 current-universe survivorship bias remains', '2025-2026 is previously exposed confirmation data', 'event scoring windows use hindsight only to measure recall, never to generate alerts']
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(coverage.to_string(index=False))
    print(episode_summary.to_string(index=False))


if __name__ == '__main__':
    main()
