from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SRC4 = Path('research/results/explosive_move_alert_refinement_round4')
SRC3 = Path('research/results/explosive_move_recall_round3')
OUT = Path('research/results/explosive_move_intersection_veto_round5')

DISC = 'DISCOVERY_2019_2022'
VAL = 'VALIDATION_2023_2024'
CONF = 'CONFIRMATION_2025_2026'
TARGETS = [0.90, 0.92, 0.95]
MAX_DEPTH = 4
BEAM_WIDTH = 80
MAX_BASE_CANDIDATES = 450
MIN_SCOPE_DISC_EVENTS = 50
MIN_NOISE_ALERTS = 200


@dataclass(frozen=True)
class Rule:
    scope: str
    feature: str
    direction: str
    threshold: float
    origin: str

    def key(self):
        return (self.scope, self.feature, self.direction, round(float(self.threshold), 12))

    def describe(self):
        op = '>=' if self.direction == 'GE' else '<='
        prefix = '' if self.scope == 'ALL' else f'{self.scope} only: '
        return f'{prefix}{self.feature} {op} {self.threshold:.8g}'


class Evaluator:
    def __init__(self, alerts: pd.DataFrame, amap: pd.DataFrame, events: pd.DataFrame):
        self.alerts = alerts.reset_index(drop=True)
        self.n = len(alerts)
        self.noise = ~self.alerts['is_event_alert'].astype(bool).to_numpy()
        self.alert_split = self.alerts['split2'].astype(str).to_numpy()

        ev = events[['event_id', 'split2']].drop_duplicates('event_id').reset_index(drop=True)
        self.events = ev
        self.event_id_to_idx = {eid: i for i, eid in enumerate(ev['event_id'])}
        self.event_split = ev['split2'].astype(str).to_numpy()
        self.n_events = len(ev)

        aid_to_row = pd.Series(np.arange(self.n, dtype=np.int64), index=self.alerts['alert_id'].astype(int)).to_dict()
        m = amap[['alert_id', 'event_id']].drop_duplicates().copy()
        m['alert_row'] = m['alert_id'].map(aid_to_row)
        m['event_idx'] = m['event_id'].map(self.event_id_to_idx)
        m = m.dropna(subset=['alert_row', 'event_idx'])
        self.map_alert = m['alert_row'].astype(np.int64).to_numpy()
        self.map_event = m['event_idx'].astype(np.int64).to_numpy()

        self.split_event_idx = {s: np.flatnonzero(self.event_split == s) for s in [DISC, VAL, CONF]}
        self.split_alert_mask = {s: (self.alert_split == s) for s in [DISC, VAL, CONF]}

    def metrics(self, mask: np.ndarray) -> dict[str, dict[str, float]]:
        selected_map = mask[self.map_alert].astype(np.int8)
        captured_counts = np.bincount(self.map_event, weights=selected_map, minlength=self.n_events)
        captured = captured_counts > 0
        out = {}
        for s in [DISC, VAL, CONF]:
            eidx = self.split_event_idx[s]
            am = self.split_alert_mask[s]
            noise = self.noise & am
            kept_noise = noise & mask
            out[s] = {
                'event_recall': float(captured[eidx].mean()) if len(eidx) else np.nan,
                'events': int(len(eidx)),
                'captured_events': int(captured[eidx].sum()),
                'noise_retention': float(kept_noise.sum() / max(noise.sum(), 1)),
                'noise_alerts': int(kept_noise.sum()),
                'selected_alerts': int((mask & am).sum()),
            }
        return out


def load_inputs():
    alerts = pd.read_csv(SRC4 / 'enriched_alerts.csv.gz', compression='gzip')
    alerts['is_event_alert'] = alerts['is_event_alert'].astype(bool)
    alerts['alert_id'] = alerts['alert_id'].astype(int)
    amap = pd.read_csv(SRC4 / 'alert_event_map.csv.gz', compression='gzip')
    events = pd.read_csv(SRC3 / 'event_coverage.csv.gz', compression='gzip')
    return alerts, amap, events


def numeric_features(alerts: pd.DataFrame) -> list[str]:
    exclude = {
        'alert_id', 'threshold', 'is_event_alert', 'matched_event_window', 'year'
    }
    cols = []
    for c in alerts.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(alerts[c]):
            x = pd.to_numeric(alerts[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
            if x.notna().sum() >= 1000 and x.nunique(dropna=True) >= 8:
                cols.append(c)
    return cols


def rule_mask(alerts: pd.DataFrame, r: Rule) -> np.ndarray:
    x = pd.to_numeric(alerts[r.feature], errors='coerce').to_numpy(dtype=float)
    valid = np.isfinite(x)
    keep = valid & ((x >= r.threshold) if r.direction == 'GE' else (x <= r.threshold))
    if r.scope != 'ALL':
        sm = alerts['family'].astype(str).eq(r.scope).to_numpy()
        keep = (~sm) | keep
    return keep


def discovery_scopes(alerts: pd.DataFrame, events: pd.DataFrame):
    d = alerts[alerts['split2'] == DISC]
    scopes = ['ALL']
    disc_event_symbols = events[events['split2'] == DISC]
    for fam, g in d.groupby('family'):
        # Require enough distinct positive alert-linked events before allowing family-specific tuning.
        if g['is_event_alert'].sum() < MIN_SCOPE_DISC_EVENTS:
            continue
        if (~g['is_event_alert']).sum() < MIN_NOISE_ALERTS:
            continue
        scopes.append(str(fam))
    return scopes


def generate_candidates(alerts: pd.DataFrame, events: pd.DataFrame, features: list[str]) -> list[Rule]:
    disc = alerts[alerts['split2'] == DISC]
    scopes = discovery_scopes(alerts, events)
    rules: dict[tuple, Rule] = {}

    for scope in scopes:
        g = disc if scope == 'ALL' else disc[disc['family'].astype(str) == scope]
        pos = g[g['is_event_alert']]
        neg = g[~g['is_event_alert']]
        if len(pos) < 50 or len(neg) < MIN_NOISE_ALERTS:
            continue

        for f in features:
            p = pd.to_numeric(pos[f], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
            n = pd.to_numeric(neg[f], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
            if len(p) < 50 or len(n) < MIN_NOISE_ALERTS:
                continue

            specs = []
            # Positive-boundary candidates: deliberately keep most positive alerts.
            for q in [0.01, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15]:
                specs.append(('GE', float(p.quantile(q)), f'positive_q{q:g}'))
                specs.append(('LE', float(p.quantile(1 - q)), f'positive_q{1-q:g}'))
            # Noise-derived veto boundaries. Intersections may remove noise missed by positive quantiles.
            for q in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
                specs.append(('GE', float(n.quantile(q)), f'noise_q{q:g}'))
                specs.append(('LE', float(n.quantile(q)), f'noise_q{q:g}'))

            for direction, threshold, origin in specs:
                if not np.isfinite(threshold):
                    continue
                r = Rule(scope, f, direction, threshold, origin)
                rules[r.key()] = r
    return list(rules.values())


def prefilter_candidates(alerts: pd.DataFrame, evaluator: Evaluator, rules: list[Rule]):
    rows = []
    for i, r in enumerate(rules):
        m = rule_mask(alerts, r)
        ev = evaluator.metrics(m)
        dr = ev[DISC]['event_recall']
        dn = ev[DISC]['noise_retention']
        if dr < 0.90 or dn >= 0.985:
            continue
        rows.append((i, dr, dn, dr - dn))
    if not rows:
        raise RuntimeError('No discovery candidates survived prefilter')
    z = pd.DataFrame(rows, columns=['idx', 'disc_recall', 'disc_noise', 'utility'])
    # Prefer noise removal while preserving a little recall cushion.
    z['rank_score'] = z['disc_noise'] + 0.8 * np.maximum(0, 0.94 - z['disc_recall'])
    z = z.sort_values(['rank_score', 'disc_noise', 'disc_recall'], ascending=[True, True, False]).head(MAX_BASE_CANDIDATES)
    kept = [rules[int(i)] for i in z['idx']]
    return kept, z


def dedupe_rule_paths(paths):
    seen = set()
    out = []
    for p in paths:
        k = tuple(sorted(r.key() for r in p['rules']))
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


def beam_search(alerts: pd.DataFrame, evaluator: Evaluator, rules: list[Rule], target: float):
    masks = [rule_mask(alerts, r) for r in rules]
    full = np.ones(len(alerts), dtype=bool)
    beam = [{'rules': [], 'mask': full, 'last_idx': -1, 'metrics': evaluator.metrics(full)}]
    all_viable = []

    for depth in range(1, MAX_DEPTH + 1):
        expanded = []
        for state in beam:
            used = {r.key() for r in state['rules']}
            for idx, r in enumerate(rules):
                if r.key() in used:
                    continue
                # Canonical ordering prevents permutations of the same intersection.
                if idx <= state['last_idx']:
                    continue
                trial = state['mask'] & masks[idx]
                ev = evaluator.metrics(trial)
                if ev[DISC]['event_recall'] < target or ev[VAL]['event_recall'] < target:
                    continue
                expanded.append({
                    'rules': state['rules'] + [r],
                    'mask': trial,
                    'last_idx': idx,
                    'metrics': ev,
                })
        if not expanded:
            break
        expanded = dedupe_rule_paths(expanded)
        expanded.sort(key=lambda s: (
            s['metrics'][VAL]['noise_retention'],
            s['metrics'][DISC]['noise_retention'],
            -s['metrics'][VAL]['event_recall'],
            len(s['rules']),
        ))
        beam = expanded[:BEAM_WIDTH]
        all_viable.extend(beam)

    if not all_viable:
        return None, []
    # Validation-only selection after discovery candidate generation. Confirmation never participates.
    all_viable.sort(key=lambda s: (
        s['metrics'][VAL]['noise_retention'],
        len(s['rules']),
        s['metrics'][DISC]['noise_retention'],
        -s['metrics'][VAL]['event_recall'],
    ))
    return all_viable[0], all_viable[:100]


def family_breakdown(mask: np.ndarray, alerts: pd.DataFrame, amap: pd.DataFrame, events: pd.DataFrame):
    selected_ids = set(alerts.loc[mask, 'alert_id'].astype(int))
    m = amap[amap['alert_id'].isin(selected_ids)]
    captured = set(m['event_id'])
    fam_by_event = {}
    # Use family of positive alerts mapping to each event where possible; event may have multiple families.
    pos = alerts[alerts['is_event_alert']][['alert_id', 'family']]
    mm = amap.merge(pos, on='alert_id', how='left')
    for eid, g in mm.groupby('event_id'):
        fam_by_event[eid] = '|'.join(sorted(set(g['family'].dropna().astype(str))))
    rows = []
    for split in [DISC, VAL, CONF]:
        e = events[events['split2'] == split]
        for fam in sorted(alerts['family'].dropna().astype(str).unique()):
            # Approximate event family ownership: event has at least one precursor alert of this family.
            eids = [eid for eid in e['event_id'] if fam in fam_by_event.get(eid, '').split('|')]
            if not eids:
                continue
            rows.append({
                'split': split, 'family': fam, 'events': len(eids),
                'captured_events': sum(eid in captured for eid in eids),
                'event_recall': sum(eid in captured for eid in eids) / len(eids),
            })
    return pd.DataFrame(rows)


def yearly_breakdown(mask: np.ndarray, alerts: pd.DataFrame, amap: pd.DataFrame, events: pd.DataFrame):
    selected_ids = set(alerts.loc[mask, 'alert_id'].astype(int))
    captured = set(amap.loc[amap['alert_id'].isin(selected_ids), 'event_id'])
    ev = events.copy()
    if 'start_time' in ev.columns:
        ev['year'] = pd.to_datetime(ev['start_time'], utc=True).dt.year
    elif 'year' not in ev.columns:
        return pd.DataFrame()
    rows = []
    for year, g in ev.groupby('year'):
        aids = alerts['split2'].eq(g['split2'].iloc[0]).to_numpy() if 'split2' in g else np.ones(len(alerts), bool)
        # Noise by calendar year if alert timestamps are available.
        if 'time' in alerts.columns:
            ay = pd.to_datetime(alerts['time'], utc=True).dt.year.to_numpy() == int(year)
            noise = (~alerts['is_event_alert'].to_numpy(dtype=bool)) & ay
            nr = ((noise & mask).sum() / max(noise.sum(), 1))
        else:
            nr = np.nan
        eids = set(g['event_id'])
        rows.append({
            'year': int(year), 'events': len(eids), 'captured_events': len(eids & captured),
            'event_recall': len(eids & captured) / len(eids) if eids else np.nan,
            'noise_retention': nr,
        })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    alerts, amap, events = load_inputs()
    evaluator = Evaluator(alerts, amap, events)
    features = numeric_features(alerts)
    raw_rules = generate_candidates(alerts, events, features)
    rules, prefilter = prefilter_candidates(alerts, evaluator, raw_rules)

    prefilter.to_csv(OUT / 'candidate_prefilter.csv', index=False)
    pd.DataFrame([{
        'scope': r.scope, 'feature': r.feature, 'direction': r.direction,
        'threshold': r.threshold, 'origin': r.origin
    } for r in rules]).to_csv(OUT / 'candidate_rules.csv', index=False)

    eval_rows = []
    rule_rows = []
    frontier_rows = []
    best_by_target = {}

    for target in TARGETS:
        best, frontier = beam_search(alerts, evaluator, rules, target)
        if best is None:
            continue
        best_by_target[target] = best
        for step, r in enumerate(best['rules'], 1):
            rule_rows.append({
                'target_recall': target, 'step': step, 'scope': r.scope,
                'feature': r.feature, 'direction': r.direction,
                'threshold': r.threshold, 'origin': r.origin,
            })
        for split, m in best['metrics'].items():
            eval_rows.append({'target_recall': target, 'split': split, 'rules': len(best['rules']), **m})
        for rank, state in enumerate(frontier[:30], 1):
            frontier_rows.append({
                'target_recall': target, 'rank': rank, 'rules': len(state['rules']),
                'validation_event_recall': state['metrics'][VAL]['event_recall'],
                'validation_noise_retention': state['metrics'][VAL]['noise_retention'],
                'discovery_event_recall': state['metrics'][DISC]['event_recall'],
                'discovery_noise_retention': state['metrics'][DISC]['noise_retention'],
                'confirmation_event_recall': state['metrics'][CONF]['event_recall'],
                'confirmation_noise_retention': state['metrics'][CONF]['noise_retention'],
                'rule_text': ' AND '.join(r.describe() for r in state['rules']),
            })

    eval_df = pd.DataFrame(eval_rows)
    rules_df = pd.DataFrame(rule_rows)
    frontier_df = pd.DataFrame(frontier_rows)
    eval_df.to_csv(OUT / 'evaluation.csv', index=False)
    rules_df.to_csv(OUT / 'selected_rules.csv', index=False)
    frontier_df.to_csv(OUT / 'frontier.csv', index=False)

    # Primary 90% result diagnostics.
    primary = best_by_target.get(0.90)
    if primary is not None:
        fam = family_breakdown(primary['mask'], alerts, amap, events)
        fam.to_csv(OUT / 'family_breakdown_90.csv', index=False)
        yearly_breakdown(primary['mask'], alerts, amap, events).to_csv(OUT / 'year_breakdown_90.csv', index=False)
        selected_alerts = alerts.loc[primary['mask'], ['alert_id', 'symbol', 'time', 'family', 'split2', 'is_event_alert']].copy()
        selected_alerts.to_csv(OUT / 'selected_alerts_90.csv.gz', index=False, compression='gzip')

    lines = [
        '# Explosive Move Intersection / Veto Round 5', '',
        '**Status:** RESEARCH ONLY on isolated `round-3` branch. No live strategy or automation changes.', '',
        '## Question', '',
        'Can a small intersection of contemporaneous conditions, including noise-veto rules, remove substantially more broad-detector noise while retaining at least ~90% of the original explosive events?', '',
        f'- Input alerts: **{len(alerts):,}** from Round 4.',
        f'- Original events: **{events.event_id.nunique():,}**.',
        f'- Numeric features considered: **{len(features)}**.',
        f'- Raw one-sided candidate conditions generated from 2019-2022 only: **{len(raw_rules):,}**.',
        f'- Discovery-prefiltered conditions entering beam search: **{len(rules):,}**.',
        '- Candidate thresholds are generated from discovery only.',
        '- Intersections are selected on 2023-2024 validation while requiring the same recall floor in discovery.',
        '- 2025-2026 confirmation is frozen and never participates in rule selection.', '',
        '## Best small intersections', '',
        '| Target | Split | Rules | Event recall | Noise retained | Noise removed | Selected alerts |',
        '|---:|---|---:|---:|---:|---:|---:|',
    ]
    for _, r in eval_df.iterrows():
        lines.append(f"| {r.target_recall:.0%} | {r.split} | {int(r.rules)} | {r.event_recall:.1%} | {r.noise_retention:.1%} | {1-r.noise_retention:.1%} | {int(r.selected_alerts):,} |")

    for target in TARGETS:
        rr = rules_df[rules_df.target_recall == target]
        if rr.empty:
            continue
        lines += ['', f'### {target:.0%} target rule set', '']
        for _, r in rr.iterrows():
            op = '>=' if r.direction == 'GE' else '<='
            scope = '' if r.scope == 'ALL' else f' for {r.scope} alerts only'
            lines.append(f"- `{r.feature} {op} {r.threshold:.8g}`{scope} ({r.origin}).")

    if primary is not None:
        m = primary['metrics']
        lines += ['', '## Primary 90% interpretation', '']
        lines.append(f"Validation retains **{m[VAL]['event_recall']:.1%}** of events while retaining **{m[VAL]['noise_retention']:.1%}** of unmatched alerts, removing **{1-m[VAL]['noise_retention']:.1%}** of validation noise.")
        lines.append(f"Frozen 2025-2026 confirmation retains **{m[CONF]['event_recall']:.1%}** of events while retaining **{m[CONF]['noise_retention']:.1%}** of unmatched alerts, removing **{1-m[CONF]['noise_retention']:.1%}** of confirmation noise.")

    lines += ['', '## Guardrails', '',
        '- Event recall, not alert-level accuracy, is the binding constraint because multiple alerts can map to one event.',
        '- Family-specific conditions are only allowed for families with enough discovery observations; the tiny capitulation cluster is not independently tuned.',
        '- This is a filter study, not an entry/exit strategy.',
        '- Survivorship bias in the original Round 1 universe remains.',
        '- 2025-2026 is previously exposed confirmation data, not a pristine holdout.',
        '- No result is promoted to the live strategy automatically.',
    ]
    (OUT / 'ANALYSIS.md').write_text('\n'.join(lines))

    manifest = {
        'study': 'Explosive Move Intersection Veto Round 5',
        'branch': 'round-3',
        'status': 'RESEARCH ONLY - no live strategy or automation edits',
        'input_alerts': int(len(alerts)),
        'source_events': int(events.event_id.nunique()),
        'targets': TARGETS,
        'max_rules': MAX_DEPTH,
        'beam_width': BEAM_WIDTH,
        'candidate_generation': 'Discovery 2019-2022 positive-boundary and noise-quantile one-sided conditions, global and sufficiently populated family scopes.',
        'selection': 'Beam search selected on 2023-2024 validation subject to same event-recall floor in discovery; minimise validation unmatched-alert retention, then rule count.',
        'confirmation': '2025-2026 frozen reporting only; never used in selection.',
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    print((OUT / 'ANALYSIS.md').read_text(), flush=True)


if __name__ == '__main__':
    main()
