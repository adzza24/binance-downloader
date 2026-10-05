from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from binance_data import load_symbol
from explosive_move_discovery_round1 import FEATURES, build_features

SRC3 = Path('research/results/explosive_move_recall_round3')
SRC4 = Path('research/results/explosive_move_alert_refinement_round4')
SRC5 = Path('research/results/explosive_move_intersection_veto_round5')
OUT = Path('research/results/explosive_move_candidate_evolution_round7')
CHECKPOINT_DIR = OUT / 'checkpoints'

DISC = 'DISCOVERY_2019_2022'
VAL = 'VALIDATION_2023_2024'
CONF = 'CONFIRMATION_2025_2026'
SPLITS = [DISC, VAL, CONF]
CHECKPOINTS = [3, 6, 12, 24, 48]
DEDUPE_HOURS = 96
START = '2019-01-01'
END = '2026-09-03'
MAX_ACTIONABLE_MFE = 0.05
MAX_DEPTH = 7
BEAM_WIDTH = 500
MAX_RULES = 420
NOISE_CAPS = [5000, 4000, 3000, 2500, 2000, 1500, 1000, 750, 500, 300, 200, 100, 50, 20, 10, 5, 2, 1, 0]
RETAIN_FRACS = [0.99, 0.98, 0.95, 0.92, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05]


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


def bool_to_int(arr: np.ndarray) -> int:
    x = 0
    for i, v in enumerate(arr):
        if bool(v):
            x |= 1 << i
    return x


def build_episodes():
    selected = pd.read_csv(SRC5 / 'selected_alerts_90.csv.gz', compression='gzip')
    selected['time'] = pd.to_datetime(selected['time'], utc=True)
    selected['alert_id'] = selected['alert_id'].astype(int)
    selected = selected.sort_values(['symbol', 'time']).copy()
    selected['prev_time'] = selected.groupby('symbol')['time'].shift()
    gap = (selected['time'] - selected['prev_time']).dt.total_seconds() / 3600.0
    selected['new_episode'] = selected['prev_time'].isna() | (gap > DEDUPE_HOURS)
    selected['episode_num'] = selected.groupby('symbol')['new_episode'].cumsum().astype(int)
    selected['episode_id'] = selected['symbol'].astype(str) + '|' + selected['episode_num'].astype(str)

    ep = selected.groupby('episode_id', as_index=False).agg(
        symbol=('symbol', 'first'),
        start_time=('time', 'first'),
        end_time=('time', 'last'),
        split2=('split2', 'first'),
        family=('family', 'first'),
        alert_count=('alert_id', 'count'),
    )

    amap = pd.read_csv(SRC4 / 'alert_event_map.csv.gz', compression='gzip')
    amap['alert_id'] = amap['alert_id'].astype(int)
    epmap = amap.merge(selected[['alert_id', 'episode_id']], on='alert_id', how='inner')[['episode_id', 'event_id']].drop_duplicates()
    counts = epmap.groupby('episode_id')['event_id'].nunique().rename('wanted_events').reset_index()
    ep = ep.merge(counts, on='episode_id', how='left')
    ep['wanted_events'] = ep['wanted_events'].fillna(0).astype(int)
    ep['is_wanted_episode'] = ep['wanted_events'] > 0

    events = pd.read_csv(SRC3 / 'event_coverage.csv.gz', compression='gzip')
    events['start_time'] = pd.to_datetime(events['start_time'], utc=True)
    events['week'] = events['start_time'].dt.strftime('%G-W%V')
    return ep, epmap, events


def checkpoint_path(symbol: str) -> Path:
    safe = ''.join(ch for ch in symbol if ch.isalnum() or ch in {'-', '_'})
    return CHECKPOINT_DIR / f'{safe}.csv.gz'


def safe_div(a, b):
    if b is None or not np.isfinite(b) or b == 0:
        return np.nan
    return a / b


def evolution_for_symbol(symbol: str, eps: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    cp = checkpoint_path(symbol)
    if cp.exists():
        z = pd.read_csv(cp, compression='gzip')
        z['start_time'] = pd.to_datetime(z['start_time'], utc=True)
        print(symbol, 'checkpoint reused', len(z), flush=True)
        return z

    df = load_symbol(symbol, '1h', START, END)
    if df.empty:
        raise RuntimeError('no candle data')
    ff = build_features(df, btc).set_index('time').sort_index()
    raw = df.set_index('time').sort_index()
    times = raw.index
    closes = raw['close'].to_numpy(float)
    highs = raw['high'].to_numpy(float)
    lows = raw['low'].to_numpy(float)
    volumes = raw['volume'].to_numpy(float)
    trades = raw['trades'].to_numpy(float)
    taker = raw['taker_buy_base'].to_numpy(float)

    btc_raw = btc.set_index('time').sort_index()
    btc_close = btc_raw['close'].reindex(times).to_numpy(float)

    rows = []
    for e in eps.itertuples(index=False):
        ts = pd.Timestamp(e.start_time)
        try:
            i = int(times.get_loc(ts))
        except KeyError:
            continue
        if i < 2160:
            continue
        base_close = closes[i]
        if not np.isfinite(base_close) or base_close <= 0:
            continue
        base_feat = ff.loc[ts] if ts in ff.index else None
        pre_lo = max(0, i - 24)
        pre_vol = float(np.nanmean(volumes[pre_lo:i])) if i > pre_lo else np.nan
        pre_trades = float(np.nanmean(trades[pre_lo:i])) if i > pre_lo else np.nan
        btc_base = btc_close[i] if i < len(btc_close) else np.nan

        for h in CHECKPOINTS:
            j = i + h
            if j >= len(raw):
                continue
            seg = slice(i + 1, j + 1)
            seg_inc = slice(i, j + 1)
            hi = float(np.nanmax(highs[seg])) if j > i else base_close
            lo = float(np.nanmin(lows[seg])) if j > i else base_close
            close_j = closes[j]
            mfe = hi / base_close - 1
            mae = lo / base_close - 1
            path_hi = float(np.nanmax(highs[seg_inc]))
            path_lo = float(np.nanmin(lows[seg_inc]))
            path_range = path_hi / path_lo - 1 if path_lo > 0 else np.nan
            rebound_low = close_j / path_lo - 1 if path_lo > 0 else np.nan
            dd_high = close_j / path_hi - 1 if path_hi > 0 else np.nan
            pos_range = (close_j - path_lo) / (path_hi - path_lo) if path_hi > path_lo else np.nan
            changes = np.diff(closes[i:j + 1])
            green_frac = float(np.mean(changes > 0)) if len(changes) else np.nan

            win3 = max(i + 1, j - 2)
            win6 = max(i + 1, j - 5)
            v3 = float(np.nanmean(volumes[win3:j + 1]))
            v6 = float(np.nanmean(volumes[win6:j + 1]))
            tr3 = float(np.nanmean(trades[win3:j + 1]))
            tr6 = float(np.nanmean(trades[win6:j + 1]))
            tv = volumes[i + 1:j + 1]
            tb = taker[i + 1:j + 1]
            taker_share = float(np.nansum(tb) / np.nansum(tv)) if np.nansum(tv) > 0 else np.nan

            ret_1h_local = close_j / closes[j - 1] - 1 if j - 1 >= i and closes[j - 1] > 0 else np.nan
            ret_3h_local = close_j / closes[max(i, j - 3)] - 1 if closes[max(i, j - 3)] > 0 else np.nan
            ret_6h_local = close_j / closes[max(i, j - 6)] - 1 if closes[max(i, j - 6)] > 0 else np.nan
            mid = i + max(1, h // 2)
            first_half = closes[mid] / base_close - 1 if closes[mid] > 0 else np.nan
            second_half = close_j / closes[mid] - 1 if closes[mid] > 0 else np.nan

            btc_ret = btc_close[j] / btc_base - 1 if np.isfinite(btc_close[j]) and np.isfinite(btc_base) and btc_base > 0 else np.nan
            coin_ret = close_j / base_close - 1

            row = {
                'episode_id': e.episode_id,
                'symbol': symbol,
                'start_time': ts,
                'split2': e.split2,
                'family': e.family,
                'checkpoint_h': h,
                'wanted_events': int(e.wanted_events),
                'is_wanted_episode': bool(e.is_wanted_episode),
                'close_progress': coin_ret,
                'mfe_since_open': mfe,
                'mae_since_open': mae,
                'path_range': path_range,
                'rebound_from_path_low': rebound_low,
                'drawdown_from_path_high': dd_high,
                'close_position_in_path_range': pos_range,
                'green_hour_fraction': green_frac,
                'ret_1h_local': ret_1h_local,
                'ret_3h_local': ret_3h_local,
                'ret_6h_local': ret_6h_local,
                'first_half_return': first_half,
                'second_half_return': second_half,
                'return_acceleration': second_half - first_half if np.isfinite(first_half) and np.isfinite(second_half) else np.nan,
                'volume_last3_vs_pre24': safe_div(v3, pre_vol),
                'volume_last6_vs_pre24': safe_div(v6, pre_vol),
                'trades_last3_vs_pre24': safe_div(tr3, pre_trades),
                'trades_last6_vs_pre24': safe_div(tr6, pre_trades),
                'taker_buy_share_since_open': taker_share,
                'btc_return_since_open': btc_ret,
                'relative_return_since_open': coin_ret - btc_ret if np.isfinite(btc_ret) else np.nan,
                'actionable_under_5pct': bool(mfe <= MAX_ACTIONABLE_MFE),
            }

            tstamp = times[j]
            if tstamp in ff.index:
                cur = ff.loc[tstamp]
                for f in FEATURES:
                    cv = cur.get(f, np.nan)
                    row[f'cur_{f}'] = float(cv) if pd.notna(cv) else np.nan
                    if base_feat is not None:
                        bv = base_feat.get(f, np.nan)
                        row[f'delta_{f}'] = float(cv - bv) if pd.notna(cv) and pd.notna(bv) else np.nan
            rows.append(row)

    z = pd.DataFrame(rows)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    z.to_csv(cp, index=False, compression='gzip')
    print(symbol, 'checkpoint saved', len(z), flush=True)
    return z


def reconstruct_evolution(ep: pd.DataFrame) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    btc = load_symbol('BTCUSDT', '1h', START, END)
    parts = []
    failures = []
    for symbol, g in ep.groupby('symbol', sort=False):
        try:
            parts.append(evolution_for_symbol(symbol, g.sort_values('start_time'), btc))
        except Exception as exc:
            failures.append({'symbol': symbol, 'error': repr(exc)})
            print('ERROR', symbol, repr(exc), flush=True)
    pd.DataFrame(failures).to_csv(OUT / 'feature_failures.csv', index=False)
    if not parts:
        raise RuntimeError('No evolution rows created')
    return pd.concat(parts, ignore_index=True)


def numeric_features(df: pd.DataFrame) -> list[str]:
    exclude = {'checkpoint_h', 'wanted_events', 'is_wanted_episode'}
    out = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            x = pd.to_numeric(df[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
            if x.notna().sum() >= 300 and x.nunique(dropna=True) >= 8:
                out.append(c)
    return out


def feature_contrasts(evo: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for h in CHECKPOINTS:
        for split in SPLITS:
            g = evo[(evo['checkpoint_h'] == h) & (evo['split2'] == split) & evo['actionable_under_5pct']]
            y = g['is_wanted_episode'].astype(int).to_numpy()
            if len(np.unique(y)) < 2:
                continue
            for f in features:
                x = pd.to_numeric(g[f], errors='coerce').replace([np.inf, -np.inf], np.nan).to_numpy(float)
                mask = np.isfinite(x)
                if mask.sum() < 50 or len(np.unique(y[mask])) < 2:
                    continue
                try:
                    auc = float(roc_auc_score(y[mask], x[mask]))
                except ValueError:
                    continue
                sep = max(auc, 1 - auc)
                pos = x[mask & (y == 1)]
                neg = x[mask & (y == 0)]
                direction = 'HIGHER' if np.nanmedian(pos) >= np.nanmedian(neg) else 'LOWER'
                rows.append({
                    'checkpoint_h': h, 'split': split, 'feature': f, 'direction': direction,
                    'separation_auc': sep, 'wanted_median': float(np.nanmedian(pos)),
                    'noise_median': float(np.nanmedian(neg)), 'wanted_n': len(pos), 'noise_n': len(neg),
                })
    return pd.DataFrame(rows)


def stable_contrasts(ct: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (h, f), g in ct.groupby(['checkpoint_h', 'feature']):
        d = {r.split: r for r in g.itertuples()}
        if not all(s in d for s in SPLITS):
            continue
        dirs = [d[s].direction for s in SPLITS]
        same = len(set(dirs)) == 1
        rows.append({
            'checkpoint_h': h, 'feature': f, 'direction': dirs[0] if same else 'MIXED',
            'same_direction_all_splits': same,
            'discovery_auc': d[DISC].separation_auc,
            'validation_auc': d[VAL].separation_auc,
            'confirmation_auc': d[CONF].separation_auc,
            'min_auc': min(d[s].separation_auc for s in SPLITS),
            'mean_auc': float(np.mean([d[s].separation_auc for s in SPLITS])),
        })
    return pd.DataFrame(rows).sort_values(
        ['same_direction_all_splits', 'min_auc', 'mean_auc'], ascending=[False, False, False]
    )


def rule_bool(g: pd.DataFrame, r: Rule) -> np.ndarray:
    x = pd.to_numeric(g[r.feature], errors='coerce').to_numpy(float)
    valid = np.isfinite(x)
    keep = valid & ((x >= r.threshold) if r.direction == 'GE' else (x <= r.threshold))
    if r.scope != 'ALL':
        sm = g['family'].astype(str).eq(r.scope).to_numpy()
        keep = (~sm) | keep
    return keep


def generate_rules(gdisc: pd.DataFrame, features: list[str]) -> list[Rule]:
    scopes = ['ALL']
    for fam, fg in gdisc.groupby('family'):
        if fg['is_wanted_episode'].sum() >= 80 and (~fg['is_wanted_episode']).sum() >= 80:
            scopes.append(str(fam))
    rules = {}
    for scope in scopes:
        g = gdisc if scope == 'ALL' else gdisc[gdisc['family'].astype(str) == scope]
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
                    rules[r.key()] = r
            for q in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
                t = float(n.quantile(q))
                if not np.isfinite(t):
                    continue
                for dr in ['GE', 'LE']:
                    r = Rule(scope, f, dr, t, f'noise_q{q:g}')
                    rules[r.key()] = r
    return list(rules.values())


def prefilter_rules(by_split, rules):
    base_w = int(by_split[DISC]['is_wanted_episode'].sum())
    base_n = int((~by_split[DISC]['is_wanted_episode']).sum())
    rows = []
    masks = []
    seen = set()
    for r in rules:
        rm = {s: bool_to_int(rule_bool(by_split[s], r)) for s in SPLITS}
        key = (rm[DISC], rm[VAL])
        if key in seen:
            continue
        seen.add(key)
        dm = rule_bool(by_split[DISC], r)
        dw = int((dm & by_split[DISC]['is_wanted_episode'].to_numpy(bool)).sum())
        dn = int((dm & (~by_split[DISC]['is_wanted_episode'].to_numpy(bool))).sum())
        pret = dw / max(base_w, 1)
        nret = dn / max(base_n, 1)
        if pret < 0.04 or nret >= 0.995:
            continue
        lift = pret / max(nret, 1e-9)
        if lift < 1.015 and nret > 0.9:
            continue
        idx = len(masks)
        masks.append((r, rm))
        rows.append({'idx': idx, 'pret': pret, 'nret': nret, 'lift': lift, 'feature': r.feature, 'scope': r.scope})
    z = pd.DataFrame(rows)
    if z.empty:
        return [], pd.DataFrame()
    keep = set()
    for floor in [0.99, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05]:
        q = z[z['pret'] >= floor].sort_values(['nret', 'pret'], ascending=[True, False]).head(30)
        keep.update(q['idx'].astype(int))
    keep.update(z.sort_values('lift', ascending=False).head(80)['idx'].astype(int))
    keep.update(z.sort_values(['nret', 'pret'], ascending=[True, False]).head(80)['idx'].astype(int))
    chosen = sorted(keep)
    if len(chosen) > MAX_RULES:
        score_map = z.set_index('idx')
        chosen = sorted(chosen, key=lambda i: (
            float(score_map.loc[i, 'nret']) - 0.22 * float(score_map.loc[i, 'pret'])
        ))[:MAX_RULES]
    return [masks[i] for i in chosen], z


def bit_counts(mask: int, wanted_bits: int, noise_bits: int):
    return (mask & wanted_bits).bit_count(), (mask & noise_bits).bit_count()


def search_frontier(by_split, rule_masks):
    wanted_bits = {}
    noise_bits = {}
    full_bits = {}
    for s in SPLITS:
        y = by_split[s]['is_wanted_episode'].to_numpy(bool)
        wanted_bits[s] = bool_to_int(y)
        noise_bits[s] = bool_to_int(~y)
        full_bits[s] = (1 << len(y)) - 1

    states = [(full_bits[DISC], full_bits[VAL], tuple(), frozenset())]
    best_by_cap = {}

    def update(dm, vm, ridxs):
        dw, dn = bit_counts(dm, wanted_bits[DISC], noise_bits[DISC])
        vw, vn = bit_counts(vm, wanted_bits[VAL], noise_bits[VAL])
        dev_noise = dn + vn
        dev_wanted = dw + vw
        for cap in NOISE_CAPS:
            if dev_noise <= cap:
                key = (dev_wanted, vw, dw, -dev_noise, -len(ridxs))
                prev = best_by_cap.get(cap)
                if prev is None or key > prev[0]:
                    best_by_cap[cap] = (key, (dm, vm, ridxs))

    update(full_bits[DISC], full_bits[VAL], tuple())

    for depth in range(1, MAX_DEPTH + 1):
        expanded = []
        for dm, vm, ridxs, used in states:
            start_idx = ridxs[-1] + 1 if ridxs else 0
            for i in range(start_idx, len(rule_masks)):
                r, masks = rule_masks[i]
                pair = (r.scope, r.feature)
                if pair in used:
                    continue
                ndm = dm & masks[DISC]
                nvm = vm & masks[VAL]
                if ndm == dm and nvm == vm:
                    continue
                dw, dn = bit_counts(ndm, wanted_bits[DISC], noise_bits[DISC])
                vw, vn = bit_counts(nvm, wanted_bits[VAL], noise_bits[VAL])
                if dw + vw < 10:
                    continue
                nr = ridxs + (i,)
                update(ndm, nvm, nr)
                expanded.append((ndm, nvm, nr, used | {pair}, dw + vw, dn + vn, vw, vn))
        if not expanded:
            break

        # Preserve a broad Pareto set: best winner count inside logarithmic-ish noise bands,
        # plus globally strongest precision/coverage states.
        dedup = {}
        for x in expanded:
            key = (x[0], x[1])
            rank = (x[4], -x[5], x[6], -len(x[2]))
            if key not in dedup or rank > dedup[key][0]:
                dedup[key] = (rank, x)
        ex = [v[1] for v in dedup.values()]
        selected = []
        bands = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 200),
                 (201, 400), (401, 750), (751, 1200), (1201, 2000), (2001, 4000), (4001, 999999)]
        for lo, hi in bands:
            b = [x for x in ex if lo <= x[5] <= hi]
            b.sort(key=lambda x: (x[4], x[6], -x[5]), reverse=True)
            selected.extend(b[:45])
        ex.sort(key=lambda x: (x[4] / max(x[5] + 1, 1), x[4], -x[5]), reverse=True)
        selected.extend(ex[:100])
        uniq = {}
        for x in selected:
            uniq[(x[0], x[1])] = x
        states = [(x[0], x[1], x[2], x[3]) for x in list(uniq.values())[:BEAM_WIDTH]]

    return best_by_cap, wanted_bits, noise_bits, full_bits


def selected_mask(ridxs, split, rule_masks, full_bits):
    m = full_bits[split]
    for i in ridxs:
        m &= rule_masks[i][1][split]
    return m


def mask_to_episode_ids(mask: int, g: pd.DataFrame) -> list[str]:
    return [eid for i, eid in enumerate(g['episode_id']) if (mask >> i) & 1]


def metrics_for_mask(mask: int, split: str, by_split, epmap, events):
    g = by_split[split]
    ids = mask_to_episode_ids(mask, g)
    sel = g[g['episode_id'].isin(ids)]
    wanted_eps = int(sel['is_wanted_episode'].sum())
    noise_eps = int((~sel['is_wanted_episode']).sum())
    event_ids = set(epmap.loc[epmap['episode_id'].isin(ids), 'event_id'])
    ev = events[events['event_id'].isin(event_ids)]
    return {
        'selected_candidates': len(sel),
        'wanted_candidate_episodes': wanted_eps,
        'noise_candidate_episodes': noise_eps,
        'unique_wanted_events': len(event_ids),
        'weeks_covered': int(ev['week'].nunique()),
        'precision': wanted_eps / max(wanted_eps + noise_eps, 1),
    }


def analyse_checkpoint(evo: pd.DataFrame, h: int, epmap: pd.DataFrame, events: pd.DataFrame, features: list[str]):
    g = evo[(evo['checkpoint_h'] == h) & evo['actionable_under_5pct']].copy()
    by_split = {s: g[g['split2'] == s].reset_index(drop=True) for s in SPLITS}
    if any(x.empty for x in by_split.values()):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rules = generate_rules(by_split[DISC], features)
    rule_masks, pre = prefilter_rules(by_split, rules)
    if not rule_masks:
        return pd.DataFrame(), pd.DataFrame(), pre
    best, wanted_bits, noise_bits, full_bits = search_frontier(by_split, rule_masks)

    rows = []
    rule_rows = []
    for cap in NOISE_CAPS:
        if cap not in best:
            continue
        _, (_, _, ridxs) = best[cap]
        for s in SPLITS:
            m = selected_mask(ridxs, s, rule_masks, full_bits)
            rows.append({'checkpoint_h': h, 'development_noise_cap': cap, 'rules': len(ridxs), 'split': s,
                         **metrics_for_mask(m, s, by_split, epmap, events)})
        for step, i in enumerate(ridxs, 1):
            r = rule_masks[i][0]
            rule_rows.append({'checkpoint_h': h, 'development_noise_cap': cap, 'step': step,
                              'scope': r.scope, 'feature': r.feature, 'direction': r.direction,
                              'threshold': r.threshold, 'origin': r.origin, 'description': r.describe()})
    return pd.DataFrame(rows), pd.DataFrame(rule_rows), pre


def write_analysis(evo, stable, frontier, rules, epmap, events):
    lines = [
        '# Explosive Move Candidate Evolution Round 7', '',
        '**Status:** RESEARCH ONLY on isolated `round-3` branch. No live strategy or automation changes.', '',
        '## Objective', '',
        'Starting from the Round 5 90%-recall alerts collapsed into 96h same-symbol candidate episodes, test whether causal developments after candidate open separate wanted explosive candidates from noise more effectively than the first-alert snapshot.', '',
        f'- Evolution checkpoints: **{", ".join(str(x) + "h" for x in CHECKPOINTS)}**.',
        f'- Actionability gate: candidate has not traded more than **+{MAX_ACTIONABLE_MFE*100:.0f}%** above its opening price by that checkpoint.',
        '- Rule thresholds are generated from 2019-2022 discovery only.',
        '- Intersections are selected using 2019-2024 development data. 2025-2026 is frozen confirmation only.',
        '- The search maximises retained wanted candidate episodes for progressively lower candidate-level noise caps; weekly breadth is reported, never imposed as a cap on winners.', '',
        '## Actionable population before evolution rules', '',
        '| Checkpoint | Split | Wanted episodes | Noise episodes | Wanted share |',
        '|---:|---|---:|---:|---:|',
    ]
    for h in CHECKPOINTS:
        for s in SPLITS:
            g = evo[(evo['checkpoint_h'] == h) & (evo['split2'] == s) & evo['actionable_under_5pct']]
            w = int(g['is_wanted_episode'].sum()); n = int((~g['is_wanted_episode']).sum())
            lines.append(f'| {h}h | {s} | {w:,} | {n:,} | {100*w/max(w+n,1):.1f}% |')

    lines += ['', '## Strongest stable evolution discriminators', '',
              '| Checkpoint | Feature | Direction | Discovery AUC | Validation AUC | Confirmation AUC |',
              '|---:|---|---|---:|---:|---:|']
    top = stable[stable['same_direction_all_splits']].sort_values(['min_auc', 'mean_auc'], ascending=False).head(20)
    for r in top.itertuples(index=False):
        lines.append(f'| {int(r.checkpoint_h)}h | `{r.feature}` | {r.direction} | {r.discovery_auc:.3f} | {r.validation_auc:.3f} | {r.confirmation_auc:.3f} |')

    lines += ['', '## Best candidate-level frontiers by checkpoint', '',
              '| Checkpoint | Dev noise cap | Dev wanted | Dev noise | Conf wanted | Conf noise | Conf events | Conf weeks | Precision |',
              '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    if not frontier.empty:
        for h in CHECKPOINTS:
            fh = frontier[frontier['checkpoint_h'] == h]
            for cap in [1000, 500, 200, 100, 50, 20, 10, 5, 2, 1, 0]:
                z = fh[fh['development_noise_cap'] == cap]
                if z.empty:
                    continue
                d = z[z['split'].isin([DISC, VAL])]
                c = z[z['split'] == CONF]
                if c.empty:
                    continue
                devw = int(d['wanted_candidate_episodes'].sum()); devn = int(d['noise_candidate_episodes'].sum())
                cr = c.iloc[0]
                lines.append(f'| {h}h | {cap:,} | {devw:,} | {devn:,} | {int(cr.wanted_candidate_episodes):,} | {int(cr.noise_candidate_episodes):,} | {int(cr.unique_wanted_events):,} | {int(cr.weeks_covered):,} | {100*float(cr.precision):.1f}% |')

    # Best confirmation precision among solutions retaining at least 50 confirmation winners is descriptive only.
    if not frontier.empty:
        conf = frontier[(frontier['split'] == CONF) & (frontier['wanted_candidate_episodes'] >= 50)].copy()
        if not conf.empty:
            conf = conf.sort_values(['precision', 'wanted_candidate_episodes'], ascending=[False, False])
            r = conf.iloc[0]
            lines += ['', '## Descriptive confirmation highlight', '',
                      f'Among searched solutions that still retain at least 50 confirmation wanted episodes, the highest observed confirmation precision is **{100*r.precision:.1f}%** at the **+{int(r.checkpoint_h)}h** checkpoint: **{int(r.wanted_candidate_episodes)} wanted vs {int(r.noise_candidate_episodes)} noise**, covering **{int(r.unique_wanted_events)} unique wanted events across {int(r.weeks_covered)} weeks**.',
                      '', 'This is descriptive only: confirmation data was not used to choose the rules.']

    lines += ['', '## Guardrails', '',
              '- Future checkpoint information is used only at the checkpoint when it would actually be known.',
              '- Candidates that already traded >+5% from candidate-open before a checkpoint are excluded from that checkpoint as no longer early/actionable under this conservative proxy.',
              '- Wanted/noise labels are still defined from the historical explosive-event catalogue, so this remains research rather than a deployable strategy.',
              '- 2025-2026 has been exposed in prior research and is confirmation, not a pristine untouched holdout.',
              '- No result is promoted to the live strategy automatically.']
    (OUT / 'ANALYSIS.md').write_text('\n'.join(lines) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ep, epmap, events = build_episodes()
    evo = reconstruct_evolution(ep)
    evo.to_csv(OUT / 'candidate_evolution.csv.gz', index=False, compression='gzip')

    features = numeric_features(evo[evo['actionable_under_5pct']])
    ct = feature_contrasts(evo, features)
    ct.to_csv(OUT / 'feature_contrasts.csv', index=False)
    stable = stable_contrasts(ct)
    stable.to_csv(OUT / 'stable_evolution_features.csv', index=False)

    frontier_parts = []
    rule_parts = []
    pre_parts = []
    for h in CHECKPOINTS:
        g = evo[(evo['checkpoint_h'] == h) & evo['actionable_under_5pct']]
        hfeatures = numeric_features(g)
        f, rr, pre = analyse_checkpoint(evo, h, epmap, events, hfeatures)
        if not f.empty:
            frontier_parts.append(f)
        if not rr.empty:
            rule_parts.append(rr)
        if not pre.empty:
            pre = pre.copy(); pre['checkpoint_h'] = h; pre_parts.append(pre)

    frontier = pd.concat(frontier_parts, ignore_index=True) if frontier_parts else pd.DataFrame()
    rule_df = pd.concat(rule_parts, ignore_index=True) if rule_parts else pd.DataFrame()
    pre_df = pd.concat(pre_parts, ignore_index=True) if pre_parts else pd.DataFrame()
    frontier.to_csv(OUT / 'evolution_frontier.csv', index=False)
    rule_df.to_csv(OUT / 'evolution_frontier_rules.csv', index=False)
    pre_df.to_csv(OUT / 'candidate_prefilter.csv.gz', index=False, compression='gzip')

    write_analysis(evo, stable, frontier, rule_df, epmap, events)
    manifest = {
        'study': 'Explosive Move Candidate Evolution Round 7',
        'branch': 'round-3',
        'status': 'RESEARCH ONLY - no live strategy or automation edits',
        'dedupe_hours': DEDUPE_HOURS,
        'checkpoints_hours': CHECKPOINTS,
        'actionable_max_mfe': MAX_ACTIONABLE_MFE,
        'selection': 'thresholds discovery only; frontier selection 2019-2024; 2025-2026 frozen confirmation',
        'candidate_source': str(SRC5 / 'selected_alerts_90.csv.gz'),
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print((OUT / 'ANALYSIS.md').read_text(), flush=True)


if __name__ == '__main__':
    main()
