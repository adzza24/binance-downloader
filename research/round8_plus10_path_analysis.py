from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from explosive_move_real_world_replay_round8 import load_extended

SRC = Path('research/results/explosive_move_real_world_replay_round8/signals.csv.gz')
OUT = Path('research/results/explosive_move_real_world_replay_round8/plus10_path_analysis.csv')
DETAIL = Path('research/results/explosive_move_real_world_replay_round8/plus10_path_detail.csv.gz')


def main():
    sig = pd.read_csv(SRC, compression='gzip')
    sig['signal_time'] = pd.to_datetime(sig['signal_time'], utc=True)
    hit = sig[pd.to_numeric(sig['hours_to_up_10'], errors='coerce').notna()].copy()
    rows = []
    for symbol, g in hit.groupby('symbol', sort=False):
        end = g['signal_time'].max() + pd.Timedelta(hours=float(pd.to_numeric(g['hours_to_up_10']).max()) + 2)
        df = load_extended(symbol, end)
        if df.empty:
            continue
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df.sort_values('time').drop_duplicates('time').reset_index(drop=True)
        idx = {pd.Timestamp(t): i for i, t in enumerate(df['time'])}
        lows = pd.to_numeric(df['low'], errors='coerce').to_numpy(float)
        for r in g.itertuples(index=False):
            i = idx.get(pd.Timestamp(r.signal_time))
            if i is None:
                continue
            h = int(round(float(r.hours_to_up_10)))
            j = i + h
            if j >= len(df):
                continue
            entry = float(r.entry_price)
            # Strictly pre-target completed candles. This cannot accidentally include a low that occurred after the +10% touch within the target hour.
            if j > i + 1:
                strict_low = float(np.nanmin(lows[i+1:j]))
            else:
                strict_low = entry
            # Conservative variant includes the full target candle; intrahour ordering is unknown.
            incl_low = float(np.nanmin(lows[i+1:j+1])) if j > i else entry
            rows.append({
                'symbol': r.symbol,
                'replay_tier': r.replay_tier,
                'signal_time': r.signal_time,
                'hours_to_up_10': h,
                'mae_before_up10_strict': strict_low / entry - 1,
                'mae_before_up10_including_target_hour': incl_low / entry - 1,
            })
    d = pd.DataFrame(rows)
    d.to_csv(DETAIL, index=False, compression='gzip')
    out = []
    for tier in ['ORIGINAL_TOP100','EXTERNAL_101_200','ALL_TOP200']:
        g = d if tier == 'ALL_TOP200' else d[d.replay_tier == tier]
        if g.empty:
            continue
        x = pd.to_numeric(g['mae_before_up10_strict'], errors='coerce').dropna()
        y = pd.to_numeric(g['mae_before_up10_including_target_hour'], errors='coerce').dropna()
        t = pd.to_numeric(g['hours_to_up_10'], errors='coerce').dropna()
        out.append({
            'tier': tier,
            'plus10_hits': len(g),
            'avg_downside_before_plus10_strict': x.mean(),
            'median_downside_before_plus10_strict': x.median(),
            'p25_downside_before_plus10_strict': x.quantile(.25),
            'p75_downside_before_plus10_strict': x.quantile(.75),
            'avg_downside_including_target_hour': y.mean(),
            'median_downside_including_target_hour': y.median(),
            'median_hours_to_plus10': t.median(),
            'mean_hours_to_plus10': t.mean(),
            'p25_hours_to_plus10': t.quantile(.25),
            'p75_hours_to_plus10': t.quantile(.75),
        })
    pd.DataFrame(out).to_csv(OUT, index=False)
    print(pd.DataFrame(out).to_csv(index=False))

if __name__ == '__main__':
    main()
