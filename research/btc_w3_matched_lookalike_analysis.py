from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

IN = Path("research/input/btc_forensic_2024")
OUT = Path("research/results/daily_return_harvest/btc_w3_matched_lookalikes_2024")
OUT.mkdir(parents=True, exist_ok=True)

N_CONTROLS = 5
EXCLUDE_HOURS = 24
MAX_CONTROL_REUSE = 1
RNG = np.random.default_rng(20241007)

MATCH_FEATURES = [
    "rsi6", "rsi14", "bb_z", "stoch_k", "cci20",
    "close_vs_ema6", "close_vs_ema12", "close_vs_ema24", "close_vs_vwap24",
    "macd_hist_pct", "atr14_pct", "log_ret",
]

SHAPE_TOKENS = (
    "__d1", "__d3", "__d6", "__d12", "__d24",
    "__delta", "__diff_mean", "__diff_std", "__up_share",
    "__late_vs_prev", "__accel", "__z", "__pos",
)


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    vals = p[ok]
    order = np.argsort(vals)
    ranked = vals[order]
    n = len(ranked)
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    tmp = np.empty_like(q)
    tmp[order] = q
    out[np.flatnonzero(ok)] = tmp
    return out


def pooled_smd(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    den = math.sqrt((np.nanvar(a, ddof=1) + np.nanvar(b, ddof=1)) / 2)
    return float((np.nanmean(a) - np.nanmean(b)) / den) if den > 1e-12 else np.nan


def robust_scale_frame(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = df[cols].replace([np.inf, -np.inf], np.nan).astype(float)
    med = x.median().to_numpy(float)
    q25 = x.quantile(.25).to_numpy(float); q75 = x.quantile(.75).to_numpy(float)
    scale = q75 - q25
    std = x.std().to_numpy(float)
    scale = np.where(np.isfinite(scale) & (scale > 1e-10), scale, std)
    scale = np.where(np.isfinite(scale) & (scale > 1e-10), scale, 1.0)
    arr = x.to_numpy(float)
    arr = np.where(np.isfinite(arr), arr, med)
    return (arr - med) / scale, med, scale


def quarter(ts: pd.Series) -> pd.Series:
    return ts.dt.quarter.astype(int)


def build_matches(meta: pd.DataFrame, static: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in MATCH_FEATURES if c in static.columns]
    if len(cols) < 8:
        raise RuntimeError(f"Too few matching features found: {cols}")

    all_state = static[cols].copy()
    z, _, _ = robust_scale_frame(all_state, cols)
    rowpos = pd.Series(np.arange(len(meta)), index=meta.row_idx.astype(int)).to_dict()
    a = anchors.copy()
    a["decision_pos"] = a.row_idx.astype(int).map(rowpos)
    a = a.dropna(subset=["decision_pos"]).copy()
    a["decision_pos"] = a.decision_pos.astype(int)
    a["time"] = pd.to_datetime(a.time, utc=True)
    a["quarter"] = quarter(a.time)

    times = pd.to_datetime(meta.time, utc=True)
    blocked = np.zeros(len(meta), dtype=bool)
    for t in a.time:
        blocked |= (times.sub(t).abs() <= pd.Timedelta(hours=EXCLUDE_HOURS)).to_numpy()

    neg = meta.w3_2_24h.ne(1).to_numpy() & (~blocked)
    neg_pos = np.flatnonzero(neg)
    q_all = quarter(times).to_numpy()
    use_count = np.zeros(len(meta), dtype=int)

    # Process hardest anchors first so unique-control matching does not punish rare states.
    candidate_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    hardness = []
    for r in a.itertuples():
        pool = neg_pos[q_all[neg_pos] == int(r.quarter)]
        d = np.sqrt(np.nanmean((z[pool] - z[int(r.decision_pos)]) ** 2, axis=1))
        order = np.argsort(d)
        pool, d = pool[order], d[order]
        candidate_cache[int(r.event_id)] = (pool, d)
        h = d[min(N_CONTROLS - 1, len(d) - 1)] if len(d) else np.inf
        hardness.append((float(h), int(r.event_id)))

    rows = []
    for _, eid in sorted(hardness, reverse=True):
        ar = a[a.event_id.eq(eid)].iloc[0]
        pool, dist = candidate_cache[eid]
        picked = []
        for cap in [MAX_CONTROL_REUSE, 2, 5]:
            for p, d in zip(pool, dist):
                if use_count[p] >= cap or p in picked:
                    continue
                picked.append(int(p)); use_count[p] += 1
                rows.append({
                    "event_id": eid,
                    "anchor_row_idx": int(ar.row_idx),
                    "anchor_time": ar.time,
                    "anchor_decision_pos": int(ar.decision_pos),
                    "control_row_idx": int(meta.row_idx.iloc[p]),
                    "control_time": times.iloc[p],
                    "control_decision_pos": int(p),
                    "match_rank": len(picked),
                    "state_distance": float(d),
                    "quarter": int(ar.quarter),
                })
                if len(picked) >= N_CONTROLS:
                    break
            if len(picked) >= N_CONTROLS:
                break
        if len(picked) < N_CONTROLS:
            raise RuntimeError(f"Could only match {len(picked)} controls for event {eid}")
    return pd.DataFrame(rows).sort_values(["event_id", "match_rank"]).reset_index(drop=True)


def balance_table(meta: pd.DataFrame, static: pd.DataFrame, anchors: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    rowpos = pd.Series(np.arange(len(meta)), index=meta.row_idx.astype(int)).to_dict()
    apos = anchors.row_idx.astype(int).map(rowpos).dropna().astype(int).to_numpy()
    mpos = matches.control_decision_pos.astype(int).to_numpy()
    negpos = np.flatnonzero(meta.w3_2_24h.ne(1).to_numpy())
    rows = []
    for f in MATCH_FEATURES:
        if f not in static.columns:
            continue
        av = pd.to_numeric(static.iloc[apos][f], errors="coerce").to_numpy(float)
        bv = pd.to_numeric(static.iloc[negpos][f], errors="coerce").to_numpy(float)
        mv = pd.to_numeric(static.iloc[mpos][f], errors="coerce").to_numpy(float)
        rows.append({
            "feature": f,
            "anchor_mean": float(np.nanmean(av)),
            "all_negative_mean": float(np.nanmean(bv)),
            "matched_control_mean": float(np.nanmean(mv)),
            "smd_before": pooled_smd(av, bv),
            "smd_after": pooled_smd(av, mv),
            "abs_smd_after": abs(pooled_smd(av, mv)),
        })
    return pd.DataFrame(rows).sort_values("abs_smd_after")


def failure_types(meta: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p in matches.control_decision_pos.astype(int):
        r = meta.iloc[p]
        t3 = float(r.target3_hour) if pd.notna(r.target3_hour) else np.nan
        s2 = float(r.stop2_hour) if pd.notna(r.stop2_hour) else np.nan
        if np.isfinite(t3) and np.isfinite(s2) and t3 == s2 and t3 < 24:
            typ = "SAME_HOUR_TARGET_STOP"
        elif np.isfinite(s2) and s2 < 24 and (not np.isfinite(t3) or s2 < t3):
            typ = "STOP_FIRST"
        elif np.isfinite(t3) and t3 < 24:
            typ = "OTHER_NON_W3_WITH_TARGET"
        elif np.isfinite(t3):
            typ = "TARGET_AFTER_24H"
        elif np.isfinite(s2) and s2 < 24:
            typ = "STOP_WITHOUT_TARGET"
        else:
            typ = "NO_TARGET_WITHIN_DATA"
        rows.append({"failure_type": typ})
    z = pd.DataFrame(rows).value_counts("failure_type").rename("controls").reset_index()
    z["share"] = z.controls / z.controls.sum()
    return z


def event_control_positions(matches: pd.DataFrame) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    apos = matches.groupby("event_id").anchor_decision_pos.first().astype(int).to_numpy()
    controls = {int(eid): g.control_decision_pos.astype(int).to_numpy() for eid, g in matches.groupby("event_id")}
    return apos, controls


def paired_temporal_comparison(temp: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    event_ids = sorted(matches.event_id.unique())
    records = []
    for col in temp.columns:
        if col.endswith("__now"):
            continue
        diffs = []
        wins = []
        ctrls = []
        for eid in event_ids:
            g = matches[matches.event_id.eq(eid)]
            ap = int(g.anchor_decision_pos.iloc[0]); cp = g.control_decision_pos.astype(int).to_numpy()
            w = float(temp.iloc[ap][col]) if pd.notna(temp.iloc[ap][col]) else np.nan
            c = pd.to_numeric(temp.iloc[cp][col], errors="coerce").to_numpy(float)
            cm = float(np.nanmean(c)) if np.isfinite(c).any() else np.nan
            if np.isfinite(w) and np.isfinite(cm):
                wins.append(w); ctrls.append(cm); diffs.append(w - cm)
        d = np.asarray(diffs, float)
        if len(d) < 70:
            continue
        sd = np.nanstd(d, ddof=1)
        eff = float(np.nanmean(d) / sd) if sd > 1e-12 else np.nan
        p = float(ttest_1samp(d, 0.0, nan_policy="omit").pvalue) if len(d) >= 3 else np.nan
        pos = float(np.mean(d > 0)); neg = float(np.mean(d < 0))
        records.append({
            "feature": col,
            "base_indicator": col.split("__", 1)[0],
            "n_events": int(len(d)),
            "winner_mean": float(np.nanmean(wins)),
            "control_mean": float(np.nanmean(ctrls)),
            "mean_paired_diff": float(np.nanmean(d)),
            "median_paired_diff": float(np.nanmedian(d)),
            "paired_effect": eff,
            "abs_paired_effect": abs(eff) if np.isfinite(eff) else np.nan,
            "direction": "WINNER_HIGH" if np.nanmean(d) > 0 else "WINNER_LOW",
            "sign_consistency": max(pos, neg),
            "p_value": p,
            "shape_only": any(tok in col for tok in SHAPE_TOKENS),
        })
    out = pd.DataFrame(records)
    out["fdr_q"] = bh_fdr(out.p_value.to_numpy(float))
    return out.sort_values(["abs_paired_effect", "sign_consistency"], ascending=[False, False])


def raw_sequence_profiles(npz, matches: pd.DataFrame) -> pd.DataFrame:
    channels = [str(x) for x in npz["channels"]]
    specs = [(24, npz["seq24"], 1), (72, npz["seq72"], 3), (240, npz["seq240"], 10)]
    event_ids = sorted(matches.event_id.unique())
    rows = []
    for window, arr, step in specs:
        for ci, ch in enumerate(channels):
            for j in range(arr.shape[1]):
                wd, cd, shape_d = [], [], []
                for eid in event_ids:
                    g = matches[matches.event_id.eq(eid)]
                    ap = int(g.anchor_decision_pos.iloc[0]); cp = g.control_decision_pos.astype(int).to_numpy()
                    w = float(arr[ap, j, ci]); c = float(np.mean(arr[cp, j, ci]))
                    wd.append(w); cd.append(c)
                    wshape = float(arr[ap, j, ci] - arr[ap, -1, ci])
                    cshape = float(np.mean(arr[cp, j, ci] - arr[cp, -1, ci]))
                    shape_d.append(wshape - cshape)
                d = np.asarray(wd) - np.asarray(cd)
                sd = np.std(d, ddof=1); seff = np.mean(d) / sd if sd > 1e-12 else np.nan
                sh = np.asarray(shape_d); shsd = np.std(sh, ddof=1); sheff = np.mean(sh) / shsd if shsd > 1e-12 else np.nan
                rows.append({
                    "window_h": window,
                    "offset_h": -int((arr.shape[1] - 1 - j) * step),
                    "channel": ch,
                    "winner_mean_z": float(np.mean(wd)),
                    "control_mean_z": float(np.mean(cd)),
                    "paired_effect": float(seff),
                    "shape_paired_effect": float(sheff),
                    "abs_shape_effect": abs(float(sheff)) if np.isfinite(sheff) else np.nan,
                })
    return pd.DataFrame(rows)


def make_classification_rows(matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos = []; y = []; groups = []; event_order = []
    for eid, g in matches.groupby("event_id"):
        pos.append(int(g.anchor_decision_pos.iloc[0])); y.append(1); groups.append(int(eid)); event_order.append(int(eid))
        for p in g.control_decision_pos.astype(int):
            pos.append(int(p)); y.append(0); groups.append(int(eid)); event_order.append(int(eid))
    return np.asarray(pos), np.asarray(y), np.asarray(groups), np.asarray(event_order)


def group_cv_temporal(temp: pd.DataFrame, matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pos, y, groups, _ = make_classification_rows(matches)
    cols = [c for c in temp.columns if not c.endswith("__now") and any(tok in c for tok in SHAPE_TOKENS)]
    X = temp.iloc[pos][cols].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=6)
    for tr, te in gkf.split(X, y, groups):
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X[tr]); Xte = imp.transform(X[te])
        k = min(80, Xtr.shape[1])
        sel = SelectKBest(f_classif, k=k)
        Xtr = sel.fit_transform(Xtr, y[tr]); Xte = sel.transform(Xte)
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        model = LogisticRegression(C=.35, max_iter=3000, class_weight="balanced", solver="liblinear", random_state=42)
        model.fit(Xtr, y[tr])
        oof[te] = model.predict_proba(Xte)[:, 1]
    return ranking_metrics("TEMPORAL_CHANGE", y, groups, oof), pd.DataFrame({"event_id": groups, "label": y, "score": oof, "model": "TEMPORAL_CHANGE"})


def group_cv_sequence(npz, matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pos, y, groups, _ = make_classification_rows(matches)
    blocks = []
    for key in ["seq24", "seq72", "seq240"]:
        a = np.asarray(npz[key][pos], np.float32)
        # Remove the final level: the model sees how the path arrived at t0, not t0 itself.
        a = a - a[:, -1:, :]
        blocks.append(a.reshape(len(a), -1))
    X = np.concatenate(blocks, axis=1)
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=6)
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        ncomp = min(30, Xtr.shape[1], Xtr.shape[0] - 2)
        pca = PCA(n_components=ncomp, random_state=42)
        Xtr = pca.fit_transform(Xtr); Xte = pca.transform(Xte)
        model = LogisticRegression(C=.5, max_iter=3000, class_weight="balanced", solver="liblinear", random_state=42)
        model.fit(Xtr, y[tr])
        oof[te] = model.predict_proba(Xte)[:, 1]
    return ranking_metrics("RAW_SEQUENCE_SHAPE", y, groups, oof), pd.DataFrame({"event_id": groups, "label": y, "score": oof, "model": "RAW_SEQUENCE_SHAPE"})


def ranking_metrics(name: str, y: np.ndarray, groups: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    auc = roc_auc_score(y, score); ap = average_precision_score(y, score)
    ranks = []
    for eid in np.unique(groups):
        m = groups == eid
        s = score[m]; yy = y[m]
        order = np.argsort(-s)
        winner_rank = int(np.flatnonzero(yy[order] == 1)[0]) + 1
        ranks.append(winner_rank)
    ranks = np.asarray(ranks)
    return pd.DataFrame([{
        "model": name,
        "rows": len(y),
        "events": len(np.unique(groups)),
        "class_precision_baseline": float(y.mean()),
        "auc": float(auc),
        "average_precision": float(ap),
        "winner_top1_rate": float(np.mean(ranks <= 1)),
        "winner_top2_rate": float(np.mean(ranks <= 2)),
        "winner_top3_rate": float(np.mean(ranks <= 3)),
        "median_winner_rank": float(np.median(ranks)),
        "random_top1": 1 / (N_CONTROLS + 1),
        "random_top2": 2 / (N_CONTROLS + 1),
    }])


def top_unique_features(comp: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    z = comp[comp.shape_only & comp.paired_effect.notna()].copy()
    z = z.sort_values(["abs_paired_effect", "sign_consistency"], ascending=[False, False])
    return z.drop_duplicates("base_indicator").head(n)


def main():
    meta = pd.read_parquet(IN / "hourly_outcomes_2024.parquet")
    static = pd.read_parquet(IN / "static_indicators_2024.parquet")
    temp = pd.read_parquet(IN / "temporal_features_2024.parquet")
    anchors = pd.read_csv(IN / "w3_event_anchors.csv")
    npz = np.load(IN / "raw_sequences_2024.npz", allow_pickle=False)
    meta["time"] = pd.to_datetime(meta.time, utc=True)
    anchors["time"] = pd.to_datetime(anchors.time, utc=True)

    matches = build_matches(meta, static, anchors)
    matches.to_csv(OUT / "matched_lookalikes.csv", index=False)
    balance = balance_table(meta, static, anchors, matches)
    balance.to_csv(OUT / "current_state_balance.csv", index=False)
    failures = failure_types(meta, matches)
    failures.to_csv(OUT / "control_failure_types.csv", index=False)

    comp = paired_temporal_comparison(temp, matches)
    comp.to_csv(OUT / "trajectory_feature_comparison.csv", index=False)
    top = top_unique_features(comp, 30)
    top.to_csv(OUT / "top_unique_trajectory_features.csv", index=False)

    raw = raw_sequence_profiles(npz, matches)
    raw.to_csv(OUT / "raw_sequence_matched_profiles.csv", index=False)
    raw_top = raw[raw.offset_h.lt(0)].sort_values("abs_shape_effect", ascending=False).drop_duplicates("channel").head(20)
    raw_top.to_csv(OUT / "top_raw_shape_differences.csv", index=False)

    mt, pred_t = group_cv_temporal(temp, matches)
    ms, pred_s = group_cv_sequence(npz, matches)
    metrics = pd.concat([mt, ms], ignore_index=True)
    metrics.to_csv(OUT / "matched_cv_metrics.csv", index=False)
    pd.concat([pred_t, pred_s], ignore_index=True).to_csv(OUT / "matched_cv_predictions.csv", index=False)

    summary = {
        "study": "BTC 2024 matched W3 lookalike trajectory comparison",
        "research_only": True,
        "2025_2026_touched": False,
        "winner_events": int(matches.event_id.nunique()),
        "controls_per_event": N_CONTROLS,
        "matched_controls": int(len(matches)),
        "matching_features": [c for c in MATCH_FEATURES if c in static.columns],
        "median_state_distance": float(matches.state_distance.median()),
        "p90_state_distance": float(matches.state_distance.quantile(.90)),
        "max_abs_smd_after": float(balance.abs_smd_after.max()),
        "median_abs_smd_after": float(balance.abs_smd_after.median()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# BTC 2024 Matched W3 Lookalike Trajectory Comparison", "",
        "**RESEARCH ONLY. No live strategy or automation changes. 2025/2026 were not read.**", "",
        f"- Winner event anchors: **{summary['winner_events']}**",
        f"- Matched non-W3 controls: **{summary['matched_controls']}** ({N_CONTROLS} per event)",
        f"- Controls excluded within ±{EXCLUDE_HOURS}h of any W3 anchor and matched within the same quarter.",
        f"- Median state-space distance: **{summary['median_state_distance']:.3f}**; p90 **{summary['p90_state_distance']:.3f}**.",
        f"- Median absolute post-match SMD across matching variables: **{summary['median_abs_smd_after']:.3f}**; max **{summary['max_abs_smd_after']:.3f}**.", "",
        "## Current-state matching balance", "", balance.to_markdown(index=False, floatfmt=".4f"), "",
        "## What happened to matched failures", "", failures.to_markdown(index=False, floatfmt=".4f"), "",
        "## Strongest *trajectory* differences after matching current state", "", top.head(20).to_markdown(index=False, floatfmt=".4f"), "",
        "## Strongest raw sequence-shape differences", "", raw_top.head(15).to_markdown(index=False, floatfmt=".4f"), "",
        "## Can trajectory pick the winner from five state-lookalikes?", "", metrics.to_markdown(index=False, floatfmt=".4f"), "",
        "Random reference: one winner among six candidates gives 16.7% top-1 and 33.3% top-2.", "",
        "These CV diagnostics are discovery-only within 2024; they are not a trading backtest and are not evidence of 2025/2026 performance.",
    ]
    (OUT / "ANALYSIS.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
