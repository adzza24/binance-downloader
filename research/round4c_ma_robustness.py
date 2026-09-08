from __future__ import annotations

import json, math, heapq
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from binance_data import load_symbol
from round4_regime_gating import BASE_SCHEDULE, daily_close, symbol_work

START_CAPITAL = 1750.0
BASE_SIZE = 300.0
MA_LENGTHS = [50, 100, 150, 180, 200, 220]
SLOPE_LOOKBACKS = [10, 20, 30]
CONFIRM_DAYS = [1, 3, 5]
MODES = ['BELOW', 'FALLING', 'AND']
FORWARD_HORIZONS = [7, 14, 30, 60]


def build_ma_features(btc_daily: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame(index=btc_daily.index)
    f['btc_close'] = btc_daily
    for ma in MA_LENGTHS:
        m = btc_daily.rolling(ma, min_periods=max(35, int(ma * 0.85))).mean()
        f[f'ma{ma}'] = m
        f[f'dist{ma}'] = btc_daily / m - 1
        for lb in SLOPE_LOOKBACKS:
            f[f'slope{ma}_{lb}'] = m / m.shift(lb) - 1
    # Strictly causal: an intraday signal on day D uses only completed information through D-1.
    return f.shift(1)


def attach_features(trades: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    q = trades.copy()
    q['entry_time'] = pd.to_datetime(q.entry_time, utc=True)
    q['exit_time'] = pd.to_datetime(q.exit_time, utc=True)
    dates = q.entry_time.dt.floor('D')
    rows = []
    for d in dates:
        rows.append(features.loc[d] if d in features.index else None)
    rf = pd.DataFrame(rows, index=q.index)
    for c in rf.columns:
        q[c] = rf[c]
    return q.dropna(subset=['dist220', 'slope220_30'])


def state_series(features: pd.DataFrame, ma: int, slope_lb: int, mode: str) -> pd.Series:
    below = features[f'dist{ma}'] < 0
    falling = features[f'slope{ma}_{slope_lb}'] < 0
    if mode == 'BELOW':
        return below.fillna(False)
    if mode == 'FALLING':
        return falling.fillna(False)
    if mode == 'AND':
        return (below & falling).fillna(False)
    raise KeyError(mode)


def confirmed_state(raw: pd.Series, days: int) -> pd.Series:
    if days <= 1:
        return raw.astype(bool)
    return raw.rolling(days, min_periods=days).sum().eq(days)


def pf(vals: np.ndarray) -> float:
    pos = vals[vals > 0]
    neg = vals[vals < 0]
    if not len(neg) or neg.sum() == 0:
        return math.inf
    return float(pos.sum() / abs(neg.sum()))


def capital_sim(g: pd.DataFrame) -> dict:
    x = g.sort_values(['entry_time', 'symbol', 'signal_id']).copy()
    cash = START_CAPITAL
    open_heap = []
    open_principal = 0.0
    peak = START_CAPITAL
    min_eq = START_CAPITAL
    max_dd = 0.0
    skipped_cash = 0
    executed = 0

    def settle_until(t):
        nonlocal cash, open_principal, peak, min_eq, max_dd
        while open_heap and open_heap[0][0] <= t:
            et, principal, pnl = heapq.heappop(open_heap)
            cash += principal + pnl
            open_principal -= principal
            eq = cash + open_principal
            peak = max(peak, eq)
            min_eq = min(min_eq, eq)
            max_dd = min(max_dd, eq - peak)

    for _, r in x.iterrows():
        settle_until(r.entry_time)
        if bool(r.pause):
            continue
        if cash + 1e-9 < BASE_SIZE:
            skipped_cash += 1
            continue
        cash -= BASE_SIZE
        open_principal += BASE_SIZE
        executed += 1
        heapq.heappush(open_heap, (r.exit_time, BASE_SIZE, float(r.pnl_usdt)))
        eq = cash + open_principal
        peak = max(peak, eq)
        min_eq = min(min_eq, eq)
        max_dd = min(max_dd, eq - peak)

    settle_until(pd.Timestamp.max.tz_localize('UTC'))
    return {
        'capital_profit_proxy': cash - START_CAPITAL,
        'capital_max_realised_dd': max_dd,
        'capital_min_equity_proxy': min_eq,
        'executed_capital_trades': executed,
        'skipped_due_cash': skipped_cash,
        'capital_ruin': bool(min_eq <= 0),
    }


def variant_metrics(q: pd.DataFrame, pause: pd.Series, name: str) -> tuple[dict, pd.DataFrame]:
    g = q.copy()
    g['pause'] = pause.reindex(g.index).fillna(False).astype(bool)
    g['scaled_pnl'] = np.where(g.pause, 0.0, g.pnl_usdt.astype(float))
    closed = g[~g.is_open]
    vals = closed.scaled_pnl.to_numpy(float)
    active = vals[vals != 0]
    ordered = closed.sort_values('exit_time')
    curve = START_CAPITAL + ordered.scaled_pnl.cumsum()
    dd = curve - curve.cummax()
    m = {
        'variant': name,
        'signals': len(g),
        'active': int((~g.pause).sum()),
        'skipped': int(g.pause.sum()),
        'realised_pnl': float(closed.scaled_pnl.sum()),
        'open_mtm': float(g[g.is_open].scaled_pnl.sum()),
        'combined_pnl': float(g.scaled_pnl.sum()),
        'profit_factor': pf(active),
        'win_rate_active': float((active > 0).mean()) if len(active) else np.nan,
        'max_pseudo_dd': float(dd.min()) if len(dd) else 0.0,
        'min_pseudo_equity': float(curve.min()) if len(curve) else START_CAPITAL,
    }
    m.update(capital_sim(g))
    return m, g


def chronological_summary(q: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_pause = pd.Series(False, index=q.index)
    m, _ = variant_metrics(q, base_pause, 'BASE')
    m.update({'ma': 0, 'slope_lb': 0, 'mode': 'BASE', 'confirm_days': 0})
    rows.append(m)
    dates = q.entry_time.dt.floor('D')
    for ma in MA_LENGTHS:
        for lb in SLOPE_LOOKBACKS:
            for mode in MODES:
                raw = state_series(features, ma, lb, mode)
                for cd in CONFIRM_DAYS:
                    conf = confirmed_state(raw, cd)
                    pause = dates.map(conf).fillna(False)
                    name = f'{mode}_MA{ma}_S{lb}_C{cd}'
                    m, _ = variant_metrics(q, pd.Series(pause.to_numpy(), index=q.index), name)
                    m.update({'ma': ma, 'slope_lb': lb, 'mode': mode, 'confirm_days': cd})
                    rows.append(m)
    return pd.DataFrame(rows)


def subperiod_summary(q: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    # Predefined chronological slices only; no retrospective market labels.
    periods = [('FULL', None, None), ('EARLY', 2019, 2022), ('LATE', 2023, 2026), ('FORWARD_2025_2026', 2025, 2026)]
    out = []
    for label, y0, y1 in periods:
        z = q.copy()
        if y0 is not None:
            z = z[z.entry_time.dt.year >= y0]
        if y1 is not None:
            z = z[z.entry_time.dt.year <= y1]
        s = chronological_summary(z, features)
        s['period'] = label
        out.append(s)
    return pd.concat(out, ignore_index=True)


def episodes(s: pd.Series) -> pd.DataFrame:
    s = s.fillna(False).astype(bool)
    grp = s.ne(s.shift()).cumsum()
    rows = []
    for _, g in s[s].groupby(grp[s]):
        rows.append({'start': g.index.min(), 'end': g.index.max(), 'days': len(g)})
    return pd.DataFrame(rows)


def warning_diagnostics(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Use one common 20d slope and AND state for cross-MA lead/escalation diagnostics.
    states = {}
    for ma in MA_LENGTHS:
        states[ma] = confirmed_state(state_series(features, ma, 20, 'AND'), 3)

    ep_rows = []
    for ma, s in states.items():
        e = episodes(s)
        for _, r in e.iterrows():
            ep_rows.append({'ma': ma, **r.to_dict()})
    ep = pd.DataFrame(ep_rows)

    long = states[200]
    long_starts = list(episodes(long)['start']) if long.any() else []
    warn_rows = []
    for ma in [50, 100, 150, 180, 220]:
        for _, r in episodes(states[ma]).iterrows():
            start = r['start']
            future = [x for x in long_starts if x >= start]
            next_long = future[0] if future else pd.NaT
            lead = (next_long - start).days if pd.notna(next_long) else np.nan
            escalates_90 = bool(pd.notna(next_long) and 0 <= lead <= 90)
            warn_rows.append({'ma': ma, 'warning_start': start, 'warning_days': r['days'], 'next_ma200_start': next_long, 'lead_days': lead, 'escalates_to_ma200_within_90d': escalates_90})
    warn = pd.DataFrame(warn_rows)
    return ep, warn


def forward_signal_returns(features: pd.DataFrame) -> pd.DataFrame:
    # BTC forward returns are diagnostics only, never inputs to states.
    raw_close = features['btc_close'].shift(-1)  # undo feature shift to recover same-day completed close approximately
    rows = []
    for ma in MA_LENGTHS:
        state = confirmed_state(state_series(features, ma, 20, 'AND'), 3)
        starts = state & ~state.shift(1, fill_value=False)
        for d in features.index[starts]:
            row = {'ma': ma, 'signal_date': d}
            for h in FORWARD_HORIZONS:
                if d in raw_close.index:
                    loc = raw_close.index.get_loc(d)
                    j = loc + h
                    row[f'btc_fwd_{h}d'] = float(raw_close.iloc[j] / raw_close.iloc[loc] - 1) if j < len(raw_close) and pd.notna(raw_close.iloc[loc]) and pd.notna(raw_close.iloc[j]) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def cascade_diagnostics(features: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    # Descriptive risk ladder only: count how many MA horizons are in causal 3-day-confirmed AND deterioration.
    st = pd.DataFrame(index=features.index)
    for ma in MA_LENGTHS:
        st[str(ma)] = confirmed_state(state_series(features, ma, 20, 'AND'), 3)
    st['count'] = st.sum(axis=1)
    dates = q.entry_time.dt.floor('D')
    counts = dates.map(st['count']).fillna(0).astype(int)
    x = q.copy()
    x['cascade_count'] = counts.to_numpy()
    rows = []
    for c, g in x.groupby('cascade_count'):
        rows.append({'cascade_count': int(c), 'signals': len(g), 'sum_pnl': float(g.pnl_usdt.sum()), 'mean_pnl': float(g.pnl_usdt.mean()), 'win_rate': float((g.pnl_usdt > 0).mean())})
    return pd.DataFrame(rows)


def main():
    cfg = json.loads(Path('research/config.json').read_text())
    out = Path('research/results/round4c')
    out.mkdir(parents=True, exist_ok=True)

    btc = load_symbol('BTCUSDT', cfg['interval'], cfg['start'], cfg['end'])
    all_t, closes = [], []
    with ThreadPoolExecutor(max_workers=6) as ex:
        fut = {ex.submit(symbol_work, s, btc, cfg): s for s in cfg['symbols']}
        for f in as_completed(fut):
            s = fut[f]
            try:
                sig, tr, dc = f.result()
                all_t += tr
                if dc is not None:
                    closes.append(dc)
                print(s, len(sig), flush=True)
            except Exception as e:
                print('ERROR', s, repr(e), flush=True)

    p = pd.concat(closes, axis=1).sort_index()
    btc_daily = p['BTCUSDT']
    features = build_ma_features(btc_daily)
    q = attach_features(pd.DataFrame(all_t), features)

    summary = subperiod_summary(q, features)
    summary.to_csv(out / 'summary.csv', index=False)

    ep, warn = warning_diagnostics(features)
    ep.to_csv(out / 'ma_episodes.csv', index=False)
    warn.to_csv(out / 'warning_escalation.csv', index=False)
    forward_signal_returns(features).to_csv(out / 'warning_forward_returns.csv', index=False)
    cascade_diagnostics(features, q).to_csv(out / 'cascade_diagnostics.csv', index=False)
    features.to_csv(out / 'daily_ma_features.csv')
    q.to_csv(out / 'trades_with_ma_features.csv', index=False)

    # Robustness ranking: reward performance and drawdown while penalising isolated optima.
    full = summary[summary.period == 'FULL'].copy()
    nonbase = full[full.variant != 'BASE'].copy()
    nonbase['pnl_rank'] = nonbase.combined_pnl.rank(pct=True)
    nonbase['pf_rank'] = nonbase.profit_factor.replace([np.inf, -np.inf], np.nan).rank(pct=True)
    nonbase['dd_rank'] = (-nonbase.capital_max_realised_dd.abs()).rank(pct=True)
    nonbase['robust_score'] = nonbase[['pnl_rank', 'pf_rank', 'dd_rank']].mean(axis=1)
    nonbase.sort_values('robust_score', ascending=False).to_csv(out / 'robustness_ranking.csv', index=False)

    manifest = {
        'study': 'Round 4C multi-horizon moving-average regime robustness and early-warning study',
        'scope': 'Entry gating only. Frozen Controlled Activity entries and Round 3H FIXED60 exits are unchanged. No regime-triggered exits are tested here.',
        'causality': 'All moving-average features are daily and shifted one full day before use. Confirmation requires consecutive causal daily states. No retrospective bull/bear labels are used in rule construction.',
        'ma_lengths': MA_LENGTHS,
        'slope_lookbacks': SLOPE_LOOKBACKS,
        'confirmation_days': CONFIRM_DAYS,
        'modes': MODES,
        'design_principle': 'Seek a stable plateau across nearby parameters, not the single highest-return variant.',
        'chronological_slices': ['FULL', 'EARLY_2019_2022', 'LATE_2023_2026', 'FORWARD_2025_2026'],
        'warning_diagnostics': 'Shorter MAs are also analysed as possible early-warning signals. Lead/escalation and forward-return diagnostics are descriptive and do not alter the tested entry gates.',
        'cascade_diagnostics': 'Counts how many MA horizons (50/100/150/180/200/220) are simultaneously in 3-day-confirmed below+falling deterioration using 20d slope; descriptive only.',
        'capital_proxy': 'Starts with 1,750 USDT, max 300 USDT per signal, refuses entries without free cash, settles at strategy exit. Drawdown is realised/cost-basis proxy, not exact mark-to-market.',
        'validation_warning': 'Underlying entry/exit strategy was previously developed using full history. This is robustness analysis, not pristine independent out-of-sample validation.'
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    print('\nTOP ROBUSTNESS\n')
    print(nonbase.sort_values('robust_score', ascending=False).head(30).to_string(index=False))
    print('\nCASCADE\n')
    print(cascade_diagnostics(features, q).to_string(index=False))


if __name__ == '__main__':
    main()
