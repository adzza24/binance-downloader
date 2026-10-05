from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SRC3 = Path('research/results/explosive_move_recall_round3')
SRC4 = Path('research/results/explosive_move_alert_refinement_round4')
SRC5 = Path('research/results/explosive_move_intersection_veto_round5')
OUT = Path('research/results/explosive_move_zero_noise_frontier_round6')

DISC = 'DISCOVERY_2019_2022'
VAL = 'VALIDATION_2023_2024'
CONF = 'CONFIRMATION_2025_2026'
SPLITS = [DISC, VAL, CONF]
DEDUPE_HOURS = 96
MAX_DEPTH = 8
MAX_RULES = 500
BEAM_WIDTH = 700
NOISE_CAPS = [4000, 3000, 2500, 2000, 1500, 1000, 750, 500, 300, 200, 100, 50, 20, 10, 5, 2, 1, 0]
RETAIN_FRACS = [0.99, 0.98, 0.95, 0.92, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05]


@dataclass(frozen=True)
class Rule:
    scope: str
    feature: str
    direction: str
    threshold: float
    origin: str

    def describe(self) -> str:
        op = '>=' if self.direction == 'GE' else '<='
        prefix = '' if self.scope == 'ALL' else f'{self.scope} only: '
        return f'{prefix}{self.feature} {op} {self.threshold:.8g}'


def bool_to_int(arr: np.ndarray) -> int:
    x = 0
    for i, v in enumerate(arr):
        if bool(v):
            x |= 1 << i
    return x


def load_inputs():
    selected = pd.read_csv(SRC5 / 'selected_alerts_90.csv.gz', compression='gzip')
    selected['time'] = pd.to_datetime(selected['time'], utc=True)
    selected['alert_id'] = selected['alert_id'].astype(int)
    enriched = pd.read_csv(SRC4 / 'enriched_alerts.csv.gz', compression='gzip')
    enriched['time'] = pd.to_datetime(enriched['time'], utc=True)
    enriched['alert_id'] = enriched['alert_id'].astype(int)
    amap = pd.read_csv(SRC4 / 'alert_event_map.csv.gz', compression='gzip')
    amap['alert_id'] = amap['alert_id'].astype(int)
    events = pd.read_csv(SRC3 / 'event_coverage.csv.gz', compression='gzip')
    events['start_time'] = pd.to_datetime(events['start_time'], utc=True)
    events['week'] = events['start_time'].dt.strftime('%G-W%V')
    return selected, enriched, amap, events


def build_episodes(selected: pd.DataFrame, enriched: pd.DataFrame, amap: pd.DataFrame):
    s = selected.sort_values(['symbol', 'time']).copy()
    s['prev_time'] = s.groupby('symbol')['time'].shift()
    gap = (s['time'] - s['prev_time']).dt.total_seconds() / 3600.0
    s['new_episode'] = s['prev_time'].isna() | (gap > DEDUPE_HOURS)
    s['episode_num'] = s.groupby('symbol')['new_episode'].cumsum().astype(int)
    s['episode_id'] = s['symbol'].astype(str) + '|' + s['episode_num'].astype(str)

    ep = s.groupby('episode_id', as_index=False).agg(
        symbol=('symbol', 'first'),
        start_time=('time', 'first'),
        end_time=('time', 'last'),
        split2=('split2', 'first'),
        alert_count=('alert_id', 'count'),
    )

    epmap = amap.merge(s[['alert_id', 'episode_id']], on='alert_id', how='inner')[['episode_id', 'event_id']].drop_duplicates()
    ec = epmap.groupby('episode_id')['event_id'].nunique().rename('wanted_events').reset_index()
    ep = ep.merge(ec, on='episode_id', how='left')
    ep['wanted_events'] = ep['wanted_events'].fillna(0).astype(int)
    ep['is_wanted_episode'] = ep['wanted_events'] > 0

    # First-alert snapshot only: every condition is known causally when the 96h candidate opens.
    first_ids = s.sort_values(['episode_id', 'time']).groupby('episode_id', as_index=False).first()[['episode_id', 'alert_id', 'family']]
    feature_cols = [c for c in enriched.columns if c not in {'symbol', 'time', 'year', 'split2', 'family', 'matched_event_window', 'is_event_alert'}]
    first = first_ids.merge(enriched[feature_cols], on='alert_id', how='left')
    ep = ep.merge(first.drop(columns=['alert_id']), on='episode_id', how='left')
    return ep, epmap, s


def numeric_features(ep: pd.DataFrame) -> list[str]:
    exclude = {'wanted_events', 'is_wanted_episode', 'alert_count', 'threshold', 'alert_id'}
    out = []
    for c in ep.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(ep[c]):
            x = pd.to_numeric(ep[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
            if x.notna().sum() >= 500 and x.nunique(dropna=True) >= 8:
                out.append(c)
    return out


def rule_bool(g: pd.DataFrame, r: Rule) -> np.ndarray:
    x = pd.to_numeric(g[r.feature], errors='coerce').to_numpy(dtype=float)
    valid = np.isfinite(x)
    keep = valid & ((x >= r.threshold) if r.direction == 'GE' else (x <= r.threshold))
    if r.scope != 'ALL':
        sm = g['family'].astype(str).eq(r.scope).to_numpy()
        keep = (~sm) | keep
    return keep


def generate_rules(ep: pd.DataFrame, features: list[str]) -> list[Rule]:
    d = ep[ep['split2'] == DISC]
    scopes = ['ALL']
    for fam, g in d.groupby('family'):
        if g['is_wanted_episode'].sum() >= 80 and (~g['is_wanted_episode']).sum() >= 80:
            scopes.append(str(fam))

    rules: dict[tuple, Rule] = {}
    for scope in scopes:
        g = d if scope == 'ALL' else d[d['family'].astype(str) == scope]
        pos = g[g['is_wanted_episode']]
        neg = g[~g['is_wanted_episode']]
        if len(pos) < 50 or len(neg) < 50:
            continue
        for f in features:
            p = pd.to_numeric(pos[f], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
            n = pd.to_numeric(neg[f], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
            if len(p) < 50 or len(n) < 50:
                continue
            direction = 'GE' if p.median() >= n.median() else 'LE'
            for retain in RETAIN_FRACS:
                q = 1 - retain if direction == 'GE' else retain
                t = float(p.quantile(q))
                if np.isfinite(t):
                    r = Rule(scope, f, direction, t, f'positive_retain_{retain:g}')
                    rules[(scope, f, direction, round(t, 12))] = r
            # Noise boundaries help find strong veto intersections.
            for q in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
                t = float(n.quantile(q))
                for dr in ['GE', 'LE']:
                    if np.isfinite(t):
                        r = Rule(scope, f, dr, t, f'noise_q{q:g}')
                        rules[(scope, f, dr, round(t, 12))] = r
    return list(rules.values())


def build_rule_masks(ep_by_split: dict[str, pd.DataFrame], rules: list[Rule]):
    out = []
    seen = set()
    for r in rules:
        masks = {s: bool_to_int(rule_bool(ep_by_split[s], r)) for s in SPLITS}
        key = (masks[DISC], masks[VAL])
        if key in seen:
            continue
        seen.add(key)
        out.append((r, masks))
    return out


def prefilter(ep_by_split, rule_masks, wanted_bits, noise_bits):
    rows = []
    d_w = wanted_bits[DISC].bit_count()
    d_n = noise_bits[DISC].bit_count()
    for i, (r, masks) in enumerate(rule_masks):
        dw = (masks[DISC] & wanted_bits[DISC]).bit_count()
        dn = (masks[DISC] & noise_bits[DISC]).bit_count()
        pret = dw / max(d_w, 1)
        nret = dn / max(d_n, 1)
        if pret < 0.045 or nret >= 0.995:
            continue
        lift = pret / max(nret, 1e-9)
        if lift < 1.02 and nret > 0.85:
            continue
        rows.append({'idx': i, 'pret': pret, 'nret': nret, 'lift': lift})
    z = pd.DataFrame(rows)
    keep = set()
    for floor in [0.99, 0.98, 0.95, 0.92, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05]:
        q = z[z['pret'] >= floor].sort_values(['nret', 'pret'], ascending=[True, False]).head(35)
        keep.update(q['idx'].astype(int))
    keep.update(z.sort_values('lift', ascending=False).head(80)['idx'].astype(int))
    keep.update(z.sort_values(['nret', 'pret'], ascending=[True, False]).head(80)['idx'].astype(int))
    chosen = sorted(keep)
    if len(chosen) > MAX_RULES:
        chosen = sorted(chosen, key=lambda i: (
            float(z.loc[z['idx'] == i, 'nret'].iloc[0]) - 0.25 * float(z.loc[z['idx'] == i, 'pret'].iloc[0])
        ))[:MAX_RULES]
    return chosen, z


def counts(mask: int, split: str, wanted_bits, noise_bits):
    return (mask & wanted_bits[split]).bit_count(), (mask & noise_bits[split]).bit_count()


def search_frontier(ep_by_split, rule_masks, candidate_idx, wanted_bits, noise_bits, full_bits):
    # State: discovery mask, validation mask, ordered rule indexes, used (scope, feature) pairs.
    states = [(full_bits[DISC], full_bits[VAL], tuple(), frozenset())]
    best_by_cap = {}
    all_best_zero = None

    base_dw = wanted_bits[DISC].bit_count()
    base_dn = noise_bits[DISC].bit_count()
    base_vw = wanted_bits[VAL].bit_count()
    base_vn = noise_bits[VAL].bit_count()

    def update_best(dm, vm, ridxs):
        nonlocal all_best_zero
        dw, dn = counts(dm, DISC, wanted_bits, noise_bits)
        vw, vn = counts(vm, VAL, wanted_bits, noise_bits)
        dev_noise = dn + vn
        dev_wanted = dw + vw
        for cap in NOISE_CAPS:
            if dev_noise <= cap:
                prev = best_by_cap.get(cap)
                key = (dev_wanted, vw, dw, -len(ridxs), -dev_noise)
                if prev is None or key > prev[0]:
                    best_by_cap[cap] = (key, (dm, vm, ridxs, dw, dn, vw, vn))
        if dev_noise == 0:
            key = (dev_wanted, vw, dw, -len(ridxs))
            if all_best_zero is None or key > all_best_zero[0]:
                all_best_zero = (key, (dm, vm, ridxs, dw, dn, vw, vn))

    update_best(full_bits[DISC], full_bits[VAL], tuple())

    for depth in range(1, MAX_DEPTH + 1):
        expanded = {}
        for dm, vm, ridxs, used in states:
            last = ridxs[-1] if ridxs else -1
            for ci in candidate_idx:
                if ci <= last:
                    continue
                r, masks = rule_masks[ci]
                sf = (r.scope, r.feature)
                if sf in used:
                    continue
                ndm = dm & masks[DISC]
                nvm = vm & masks[VAL]
                if ndm == dm and nvm == vm:
                    continue
                dw, dn = counts(ndm, DISC, wanted_bits, noise_bits)
                vw, vn = counts(nvm, VAL, wanted_bits, noise_bits)
                if dw < 10 or vw < 5:
                    continue
                keymask = (ndm, nvm)
                if keymask not in expanded:
                    expanded[keymask] = (ndm, nvm, ridxs + (ci,), used | {sf})
        cand = list(expanded.values())
        if not cand:
            break
        for dm, vm, ridxs, _ in cand:
            update_best(dm, vm, ridxs)

        # Multi-objective pruning: retain strong states at every development-noise cap,
        # strong precision states at many winner floors, and several trade-off slopes.
        keep_keys = set()
        enriched = []
        for st in cand:
            dm, vm, ridxs, used = st
            dw, dn = counts(dm, DISC, wanted_bits, noise_bits)
            vw, vn = counts(vm, VAL, wanted_bits, noise_bits)
            enriched.append((st, dw, dn, vw, vn))

        for cap in NOISE_CAPS:
            feasible = [x for x in enriched if x[2] + x[4] <= cap]
            feasible.sort(key=lambda x: (x[1] + x[3], x[3], x[1], -len(x[0][2])), reverse=True)
            keep_keys.update((x[0][0], x[0][1]) for x in feasible[:25])

        for floor in [3000, 2500, 2000, 1600, 1200, 1000, 800, 600, 400, 300, 200, 150, 100, 75, 50, 25]:
            feasible = [x for x in enriched if x[1] + x[3] >= floor]
            feasible.sort(key=lambda x: (x[2] + x[4], -(x[1] + x[3]), len(x[0][2])))
            keep_keys.update((x[0][0], x[0][1]) for x in feasible[:20])

        for lam in [0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]:
            ranked = sorted(enriched, key=lambda x: (
                min(x[1] / max(base_dw, 1), x[3] / max(base_vw, 1))
                - lam * max(x[2] / max(base_dn, 1), x[4] / max(base_vn, 1)),
                x[1] + x[3]
            ), reverse=True)
            keep_keys.update((x[0][0], x[0][1]) for x in ranked[:25])

        states = [x[0] for x in enriched if (x[0][0], x[0][1]) in keep_keys]
        if len(states) > BEAM_WIDTH:
            scored = []
            for st in states:
                dw, dn = counts(st[0], DISC, wanted_bits, noise_bits)
                vw, vn = counts(st[1], VAL, wanted_bits, noise_bits)
                precision = (dw + vw) / max(dw + vw + dn + vn, 1)
                retention = min(dw / max(base_dw, 1), vw / max(base_vw, 1))
                scored.append((precision + 0.45 * retention, st))
            scored.sort(key=lambda x: x[0], reverse=True)
            states = [x[1] for x in scored[:BEAM_WIDTH]]
        print('depth', depth, 'expanded', len(cand), 'beam', len(states), flush=True)

    return best_by_cap, all_best_zero


def selected_mask_for_split(ridxs, split, rule_masks, full_bits):
    m = full_bits[split]
    for i in ridxs:
        m &= rule_masks[i][1][split]
    return m


def split_metrics(mask: int, split: str, ep_by_split, epmap, events):
    g = ep_by_split[split]
    selected_idx = [i for i in range(len(g)) if (mask >> i) & 1]
    chosen = g.iloc[selected_idx]
    wanted_eps = int(chosen['is_wanted_episode'].sum())
    noise_eps = int((~chosen['is_wanted_episode']).sum())
    eids = set(epmap[epmap['episode_id'].isin(chosen['episode_id'])]['event_id'])
    em = events[events['event_id'].isin(eids)]
    return {
        'selected_candidates': int(len(chosen)),
        'wanted_candidate_episodes': wanted_eps,
        'noise_candidate_episodes': noise_eps,
        'unique_wanted_events': int(len(eids)),
        'weeks_covered': int(em['week'].nunique()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    selected, enriched, amap, events = load_inputs()
    ep, epmap, selected_with_episode = build_episodes(selected, enriched, amap)

    # Preserve real chronological 96h episodes. A handful cross research split boundaries;
    # evaluation assigns the whole episode to the split in which the candidate opened.
    ep_by_split = {s: ep[ep['split2'] == s].reset_index(drop=True) for s in SPLITS}
    wanted_bits = {}
    noise_bits = {}
    full_bits = {}
    for s, g in ep_by_split.items():
        y = g['is_wanted_episode'].to_numpy(dtype=bool)
        wanted_bits[s] = bool_to_int(y)
        noise_bits[s] = bool_to_int(~y)
        full_bits[s] = (1 << len(g)) - 1

    features = numeric_features(ep)
    raw_rules = generate_rules(ep, features)
    rule_masks = build_rule_masks(ep_by_split, raw_rules)
    candidate_idx, prefilter = prefilter(ep_by_split, rule_masks, wanted_bits, noise_bits)

    best_by_cap, best_zero = search_frontier(ep_by_split, rule_masks, candidate_idx, wanted_bits, noise_bits, full_bits)

    base_rows = []
    for s in SPLITS:
        m = split_metrics(full_bits[s], s, ep_by_split, epmap, events)
        base_rows.append({'split': s, **m})
    pd.DataFrame(base_rows).to_csv(OUT / 'baseline_96h.csv', index=False)

    frontier_rows = []
    selected_rule_rows = []
    for cap in NOISE_CAPS:
        if cap not in best_by_cap:
            continue
        _, state = best_by_cap[cap]
        dm, vm, ridxs, dw, dn, vw, vn = state
        for s in SPLITS:
            mask = selected_mask_for_split(ridxs, s, rule_masks, full_bits)
            metrics = split_metrics(mask, s, ep_by_split, epmap, events)
            frontier_rows.append({
                'development_noise_cap': cap,
                'rules': len(ridxs),
                'split': s,
                **metrics,
            })
        for step, i in enumerate(ridxs, 1):
            r = rule_masks[i][0]
            selected_rule_rows.append({
                'development_noise_cap': cap,
                'step': step,
                'scope': r.scope,
                'feature': r.feature,
                'direction': r.direction,
                'threshold': r.threshold,
                'origin': r.origin,
                'description': r.describe(),
            })

    frontier = pd.DataFrame(frontier_rows)
    rules_out = pd.DataFrame(selected_rule_rows).drop_duplicates()
    frontier.to_csv(OUT / 'frontier.csv', index=False)
    rules_out.to_csv(OUT / 'frontier_rules.csv', index=False)
    prefilter.to_csv(OUT / 'candidate_prefilter.csv', index=False)

    ep[['episode_id', 'symbol', 'start_time', 'end_time', 'split2', 'alert_count', 'wanted_events', 'is_wanted_episode']].to_csv(
        OUT / 'candidate_episodes_96h.csv.gz', index=False, compression='gzip'
    )

    # Human-readable report.
    lines = [
        '# Explosive Move Zero-Noise Frontier Round 6',
        '',
        '**Status:** RESEARCH ONLY on isolated `round-3` branch. No live strategy or automation changes.',
        '',
        '## Objective',
        '',
        'Starting from the Round 5 90%-recall filtered alerts, collapse repeated same-symbol alerts into 96h candidate episodes and progressively intersect contemporaneous first-alert conditions. Minimise candidate-level noise toward zero while retaining as many genuine wanted candidate episodes/events as possible. The ~381 active-week figure is a floor/coverage diagnostic, never a cap on winners.',
        '',
        f'- Dedupe window: **{DEDUPE_HOURS}h**.',
        f'- Candidate first-alert numeric features: **{len(features)}**.',
        f'- Raw candidate conditions: **{len(raw_rules)}**.',
        f'- Prefiltered conditions entering search: **{len(candidate_idx)}**.',
        '- Thresholds are generated from 2019-2022 discovery only.',
        '- 2019-2024 is the development frontier; 2025-2026 is frozen confirmation and never participates in rule selection.',
        '- All intersection features are from the first alert that opened the 96h candidate. No future episode information is used.',
        '',
        '## Baseline 96h candidates',
        '',
        '| Split | Wanted candidate episodes | Noise candidate episodes | Unique wanted events | Weeks covered |',
        '|---|---:|---:|---:|---:|',
    ]
    for r in base_rows:
        lines.append(f"| {r['split']} | {r['wanted_candidate_episodes']:,} | {r['noise_candidate_episodes']:,} | {r['unique_wanted_events']:,} | {r['weeks_covered']:,} |")

    lines += ['', '## Progressive noise frontier', '',
              '| Dev noise cap | Rules | Dev wanted episodes | Dev noise episodes | Confirmation wanted episodes | Confirmation noise episodes | Confirmation unique events | Confirmation weeks |',
              '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cap in NOISE_CAPS:
        q = frontier[frontier['development_noise_cap'] == cap] if not frontier.empty else pd.DataFrame()
        if q.empty:
            continue
        d = q[q['split'].isin([DISC, VAL])]
        c = q[q['split'] == CONF].iloc[0]
        lines.append(
            f"| {cap:,} | {int(q['rules'].iloc[0])} | {int(d['wanted_candidate_episodes'].sum()):,} | {int(d['noise_candidate_episodes'].sum()):,} | "
            f"{int(c['wanted_candidate_episodes']):,} | {int(c['noise_candidate_episodes']):,} | {int(c['unique_wanted_events']):,} | {int(c['weeks_covered']):,} |"
        )

    if best_zero is not None:
        _, z = best_zero
        ridxs = z[2]
        lines += ['', '## Least-restrictive development zero-noise solution found', '']
        for step, i in enumerate(ridxs, 1):
            lines.append(f'- Step {step}: `{rule_masks[i][0].describe()}`.')
        lines += ['', 'This solution has zero candidate-level noise across 2019-2024 development data. Its frozen 2025-2026 confirmation outcome is shown in the frontier table and is the important generalisation check.']
    else:
        lines += ['', '## Zero-noise result', '', 'No development zero-noise intersection was found within the searched eight-rule frontier while retaining the minimum search population.']

    lines += ['', '## Guardrails', '',
              '- This is a candidate-filter study, not an entry/exit strategy.',
              '- Zero noise in fitted/development history is not proof of future zero noise; confirmation performance is decisive.',
              '- The original current-universe survivorship bias remains.',
              '- 2025-2026 has been seen in earlier research and is confirmation, not a pristine untouched holdout.',
              '- No result is promoted to the live strategy automatically.']
    (OUT / 'ANALYSIS.md').write_text('\n'.join(lines) + '\n')

    manifest = {
        'study': 'Explosive Move Zero-Noise Frontier Round 6',
        'branch': 'round-3',
        'status': 'RESEARCH ONLY - no live strategy or automation edits',
        'dedupe_hours': DEDUPE_HOURS,
        'source_selected_alerts': str(SRC5 / 'selected_alerts_90.csv.gz'),
        'max_intersection_depth': MAX_DEPTH,
        'confirmation_never_used_for_selection': True,
        'first_alert_features_only': True,
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print((OUT / 'ANALYSIS.md').read_text(), flush=True)


if __name__ == '__main__':
    main()
