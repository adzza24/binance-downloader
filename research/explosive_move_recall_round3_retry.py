from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

import explosive_move_recall_round3 as base


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


_original_write_analysis = base.write_analysis


def write_analysis(assignments, thresholds, coverage, episode_summary):
    _original_write_analysis(assignments, thresholds, coverage, episode_summary)
    selected = thresholds[thresholds.selected]
    fallback = selected[selected.selection_source != 'VALIDATION_2023_2024'] if 'selection_source' in selected else pd.DataFrame()
    if fallback.empty:
        return
    path = base.OUT / 'ANALYSIS.md'
    with path.open('a') as f:
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
    path.write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    base.train_family_models = train_family_models
    base.write_analysis = write_analysis
    base.main()
    patch_manifest()
