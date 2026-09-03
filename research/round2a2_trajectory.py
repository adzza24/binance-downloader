from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from binance_data import load_symbol
from round2a_discovery import FEATURES, feature_frame, discover_events

OFFSETS = [-72, -48, -24, -12, -6, 0, 6, 12, 24]
TRAJ_FEATURES = [
    "price_rel", "ret_6h", "ret_24h", "rs_24h", "range_24h", "atr24_pct",
    "vol_ratio_6h", "trade_ratio_6h", "taker_6h", "taker_trend_24h",
    "close_location_24h", "dist_high_72h", "higher_lows_24h", "green_share_24h",
    "btc_ret_24h", "btc_atr24_pct",
]


def build_symbol(symbol: str, cfg: dict, btc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = btc.copy() if symbol == "BTCUSDT" else load_symbol(symbol, cfg["interval"], cfg["start"], cfg["end"])
    if len(df) < 500:
        return pd.DataFrame(), pd.DataFrame()
    ff = feature_frame(df, btc)
    events = discover_events(symbol, df, ff)
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    ff = ff.reset_index(drop=True)
    index_by_time = pd.Series(ff.index.to_numpy(), index=ff.time).to_dict()
    traj = []
    for eid, ev in events.reset_index(drop=True).iterrows():
        idx = index_by_time.get(ev.time)
        if idx is None:
            continue
        anchor = ff.close.iloc[idx]
        for off in OFFSETS:
            j = idx + off
            if j < 0 or j >= len(ff):
                continue
            row = {
                "event_id": f"{symbol}:{eid}", "symbol": symbol, "kind": ev.kind,
                "event_time": ev.time, "offset_h": off,
                "future_upside_7d": ev.future_upside_7d,
                "future_adverse_3d": ev.future_adverse_3d,
                "price_rel": ff.close.iloc[j] / anchor - 1,
            }
            for f in TRAJ_FEATURES:
                if f == "price_rel":
                    continue
                row[f] = ff[f].iloc[j]
            traj.append(row)
    return events, pd.DataFrame(traj)


def std_diff(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna(); b = b.dropna()
    if len(a) < 5 or len(b) < 5:
        return np.nan
    pooled = np.sqrt((a.var() + b.var()) / 2)
    return (a.mean() - b.mean()) / pooled if pooled else np.nan


def trajectory_summary(traj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for off in OFFSETS:
        z = traj[traj.offset_h == off]
        for f in TRAJ_FEATURES:
            a = z.loc[z.kind == "RALLY", f]
            b = z.loc[z.kind == "CONTROL", f]
            rows.append({
                "offset_h": off, "feature": f,
                "rally_median": a.median(), "control_median": b.median(),
                "median_diff": a.median() - b.median(), "standardised_diff": std_diff(a, b),
                "rallies": a.notna().sum(), "controls": b.notna().sum(),
            })
    return pd.DataFrame(rows)


def transition_summary(traj: pd.DataFrame) -> pd.DataFrame:
    wide = traj.pivot_table(index=["event_id", "symbol", "kind", "event_time"], columns="offset_h", values=TRAJ_FEATURES)
    rows = []
    windows = [(-72, -24), (-48, 0), (-24, 0), (-12, 0), (-12, 6), (0, 6), (0, 12), (0, 24)]
    meta = wide.reset_index()[["event_id", "symbol", "kind", "event_time"]]
    for f in TRAJ_FEATURES:
        for aoff, boff in windows:
            if (f, aoff) not in wide.columns or (f, boff) not in wide.columns:
                continue
            vals = wide[(f, boff)] - wide[(f, aoff)]
            tmp = meta.copy(); tmp["value"] = vals.to_numpy()
            ra = tmp.loc[tmp.kind == "RALLY", "value"]
            co = tmp.loc[tmp.kind == "CONTROL", "value"]
            rows.append({
                "feature": f, "from_h": aoff, "to_h": boff,
                "rally_median_change": ra.median(), "control_median_change": co.median(),
                "median_change_diff": ra.median() - co.median(),
                "standardised_diff": std_diff(ra, co),
                "rallies": ra.notna().sum(), "controls": co.notna().sum(),
            })
    return pd.DataFrame(rows).sort_values("standardised_diff", key=lambda s: s.abs(), ascending=False)


def year_robustness(traj: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    # Recompute the strongest pre/at-start contrasts by year, avoiding post-start leakage for eventual validation hypotheses.
    rows = []
    traj = traj.copy(); traj["year"] = pd.to_datetime(traj.event_time).dt.year
    candidates = trajectory_summary(traj)
    candidates = candidates[candidates.offset_h <= 0].sort_values("standardised_diff", key=lambda s: s.abs(), ascending=False).head(25)
    for _, c in candidates.iterrows():
        signs=[]; diffs=[]; years=0
        for y, z in traj[traj.offset_h == c.offset_h].groupby("year"):
            a=z.loc[z.kind=="RALLY",c.feature]; b=z.loc[z.kind=="CONTROL",c.feature]
            d=std_diff(a,b)
            if pd.notna(d):
                years += 1; diffs.append(d); signs.append(np.sign(d))
        rows.append({"type":"level","feature":c.feature,"from_h":c.offset_h,"to_h":c.offset_h,
                     "overall_std_diff":c.standardised_diff,"years":years,
                     "same_direction_years":int(sum(s==np.sign(c.standardised_diff) for s in signs)),
                     "median_year_std_diff":float(np.nanmedian(diffs)) if diffs else np.nan})
    return pd.DataFrame(rows).sort_values(["same_direction_years","overall_std_diff"], ascending=[False,False])


def main():
    cfg=json.loads(Path("research/config.json").read_text())
    out=Path("research/results/round2a2"); out.mkdir(parents=True,exist_ok=True)
    btc=load_symbol("BTCUSDT",cfg["interval"],cfg["start"],cfg["end"])
    all_events=[]; all_traj=[]
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(build_symbol,s,cfg,btc):s for s in cfg["symbols"]}
        for fut in as_completed(futs):
            s=futs[fut]
            try:
                ev,tr=fut.result()
                print("Trajectory",s,"events",len(ev),flush=True)
                if not ev.empty: all_events.append(ev)
                if not tr.empty: all_traj.append(tr)
            except Exception as e:
                print("ERROR",s,repr(e),flush=True)
    events=pd.concat(all_events,ignore_index=True)
    traj=pd.concat(all_traj,ignore_index=True)
    ts=trajectory_summary(traj)
    trans=transition_summary(traj)
    robust=year_robustness(traj,trans)
    events.to_csv(out/"events.csv",index=False)
    traj.to_csv(out/"trajectories.csv",index=False)
    ts.to_csv(out/"trajectory_summary.csv",index=False)
    trans.to_csv(out/"transition_contrasts.csv",index=False)
    robust.to_csv(out/"prestart_robustness.csv",index=False)
    # Earliest pre-start separation for each feature.
    early=[]
    for f,z in ts[ts.offset_h<=0].groupby("feature"):
        z=z.assign(absdiff=z.standardised_diff.abs()).sort_values("offset_h")
        strong=z[z.absdiff>=0.20]
        chosen=strong.iloc[0] if len(strong) else z.sort_values("absdiff",ascending=False).iloc[0]
        early.append(chosen.drop(labels=["absdiff"]).to_dict())
    pd.DataFrame(early).sort_values("standardised_diff",key=lambda s:s.abs(),ascending=False).to_csv(out/"earliest_separation.csv",index=False)
    Path(out/"manifest.json").write_text(json.dumps({
        "study":"Round 2A-2 retrospective rally phase/trajectory discovery",
        "labels":"same hindsight rally/control labels as Round 2A",
        "offset_hours":OFFSETS,
        "trajectory_features":TRAJ_FEATURES,
        "important":"post-start offsets are discovery-only and must not be used directly as entry predictors; Round 2B may use only features observable by its decision timestamp"
    },indent=2))
    print("TOP PRESTART LEVEL CONTRASTS")
    print(ts[ts.offset_h<=0].sort_values("standardised_diff",key=lambda s:s.abs(),ascending=False).head(25).to_string(index=False))
    print("TOP TRANSITIONS")
    print(trans.head(25).to_string(index=False))
    print("ROBUST")
    print(robust.head(20).to_string(index=False))

if __name__=="__main__": main()
