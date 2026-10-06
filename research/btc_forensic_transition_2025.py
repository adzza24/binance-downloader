from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import average_precision_score, roc_auc_score

import btc_forensic_transition_2024 as base
from binance_data import load_symbol

OUT = Path("research/results/daily_return_harvest/btc_forensic_transition_2025")
OUT.mkdir(parents=True, exist_ok=True)
IN2024 = Path("research/input/btc_forensic_2024")

SYMBOL = "BTCUSDT"
INTERVAL = "1h"
LOAD_START = "2024-11-01"
LOAD_END = "2025-12-31"
DECISION_START = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
DECISION_END = pd.Timestamp("2025-12-28 23:00:00", tz="UTC")

TARGETS = [0.015, 0.020, 0.025, 0.030, 0.040, 0.050]
STOPS = [0.0075, 0.010, 0.015, 0.020, 0.025, 0.030]
MAX_OCO_HOURS = 168


def fit_models_2025(meta: pd.DataFrame, static: pd.DataFrame, temporal: pd.DataFrame, seq24, seq72, seq240, y):
    t = pd.to_datetime(meta.time, utc=True).reset_index(drop=True)
    train = t < pd.Timestamp("2025-07-01", tz="UTC")
    dev = (t >= pd.Timestamp("2025-07-08", tz="UTC")) & (t < pd.Timestamp("2025-10-01", tz="UTC"))
    hold = t >= pd.Timestamp("2025-10-08", tz="UTC")
    sets = {
        "STATIC": static.reset_index(drop=True),
        "TEMPORAL": temporal.reset_index(drop=True),
        "TEMPORAL_SEQ": pd.concat([
            temporal.reset_index(drop=True),
            pd.DataFrame(seq24.reshape(len(seq24), -1), columns=[f"s24_{i}" for i in range(seq24.shape[1] * seq24.shape[2])]),
            pd.DataFrame(seq72.reshape(len(seq72), -1), columns=[f"s72_{i}" for i in range(seq72.shape[1] * seq72.shape[2])]),
            pd.DataFrame(seq240.reshape(len(seq240), -1), columns=[f"s240_{i}" for i in range(seq240.shape[1] * seq240.shape[2])]),
        ], axis=1),
    }
    metrics, frontier, importances = [], [], []
    for name, X in sets.items():
        X = X.replace([np.inf, -np.inf], np.nan).astype(np.float32)
        usable = X.columns[X[train].notna().mean() > .70]
        X = X[usable]
        model = LGBMClassifier(
            n_estimators=1200, learning_rate=.025, num_leaves=31, max_depth=-1,
            min_child_samples=40, subsample=.85, colsample_bytree=.70,
            reg_lambda=5, reg_alpha=1, random_state=42, verbosity=-1, n_jobs=2,
        )
        model.fit(X[train], y[train], eval_set=[(X[dev], y[dev])], eval_metric="binary_logloss", callbacks=[early_stopping(80, verbose=False)])
        scores = np.full(len(X), np.nan)
        scores[dev] = model.predict_proba(X[dev])[:, 1]
        scores[hold] = model.predict_proba(X[hold])[:, 1]
        for split, mask in [("DEV", dev), ("HOLDOUT", hold)]:
            metrics.append({
                "model": name, "split": split, "rows": int(mask.sum()),
                "w3_rate": float(y[mask].mean()),
                "auc": roc_auc_score(y[mask], scores[mask]),
                "average_precision": average_precision_score(y[mask], scores[mask]),
            })
        dev_scores, dev_y, dev_t = scores[dev], y[dev], t[dev].reset_index(drop=True)
        candidates = np.unique(np.quantile(dev_scores[np.isfinite(dev_scores)], np.linspace(.5, .999, 250)))
        for minrate in [.10, .25, .50, 1.0]:
            best = None
            for th in candidates:
                m = base.onset_metrics(dev_t, dev_y, dev_scores, th)
                if m["signals_per_day"] + 1e-9 < minrate:
                    continue
                if best is None or (m["precision"], m["wins_per_day"]) > (best["precision"], best["wins_per_day"]):
                    best = {"threshold": float(th), **m}
            if best is None:
                continue
            hm = base.onset_metrics(t[hold].reset_index(drop=True), y[hold], scores[hold], best["threshold"])
            frontier.append({"model": name, "min_dev_signals_per_day": minrate, "threshold": best["threshold"], "split": "DEV", **{k: v for k, v in best.items() if k != "threshold"}})
            frontier.append({"model": name, "min_dev_signals_per_day": minrate, "threshold": best["threshold"], "split": "HOLDOUT", **hm})
        for f, imp in zip(X.columns, model.feature_importances_):
            importances.append({"model": name, "feature": f, "importance": int(imp)})
    pd.DataFrame(metrics).to_csv(OUT / "model_metrics.csv", index=False)
    pd.DataFrame(frontier).to_csv(OUT / "precision_frequency_frontier.csv", index=False)
    pd.DataFrame(importances).sort_values(["model", "importance"], ascending=[True, False]).to_csv(OUT / "model_feature_importance.csv", index=False)


def oco_trade(df: pd.DataFrame, decision_row: int, target: float, stop: float, max_hours: int = MAX_OCO_HOURS):
    if decision_row + 1 >= len(df):
        return None
    raw_entry = float(df.open.iloc[decision_row + 1])
    if not np.isfinite(raw_entry) or raw_entry <= 0:
        return None
    entry_cost = raw_entry * (1 + base.SLIPPAGE_RATE) * (1 + base.FEE_RATE)
    exit_factor = (1 - base.SLIPPAGE_RATE) * (1 - base.FEE_RATE)
    target_px = entry_cost * (1 + target) / exit_factor
    stop_px = entry_cost * (1 - stop) / exit_factor
    end = min(len(df), decision_row + 1 + max_hours)
    for j in range(decision_row + 1, end):
        hit_t = float(df.high.iloc[j]) >= target_px
        hit_s = float(df.low.iloc[j]) <= stop_px
        age = j - (decision_row + 1)
        if hit_t and hit_s:
            return -stop, "STOP_SAME_CANDLE", age
        if hit_s:
            return -stop, "STOP", age
        if hit_t:
            return target, "TARGET", age
    j = end - 1
    exit_net = float(df.close.iloc[j]) * exit_factor
    return exit_net / entry_cost - 1, "MARK_168H", j - (decision_row + 1)


def oco_surface(df: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    anchor_rows = anchors.row_idx.astype(int).to_numpy()
    for target in TARGETS:
        for stop in STOPS:
            trades = [oco_trade(df, r, target, stop) for r in anchor_rows]
            trades = [x for x in trades if x is not None]
            rets = np.asarray([x[0] for x in trades], float)
            outcomes = [x[1] for x in trades]
            ages = np.asarray([x[2] for x in trades], float)
            rows.append({
                "target": target, "stop": stop, "events": len(trades),
                "mean_return": float(np.mean(rets)),
                "median_return": float(np.median(rets)),
                "geomean_return": float(np.exp(np.mean(np.log1p(rets))) - 1),
                "positive_rate": float(np.mean(rets > 0)),
                "target_rate": float(np.mean(np.asarray(outcomes) == "TARGET")),
                "stop_rate": float(np.mean(np.isin(outcomes, ["STOP", "STOP_SAME_CANDLE"]))),
                "marked_rate": float(np.mean(np.asarray(outcomes) == "MARK_168H")),
                "resolved_24h": float(np.mean(ages < 24)),
                "resolved_48h": float(np.mean(ages < 48)),
                "resolved_72h": float(np.mean(ages < 72)),
                "resolved_168h": float(np.mean(np.asarray(outcomes) != "MARK_168H")),
                "median_hours": float(np.median(ages)),
                "p90_hours": float(np.percentile(ages, 90)),
            })
    return pd.DataFrame(rows).sort_values(["geomean_return", "mean_return"], ascending=False)


def cross_year_comparison():
    p24 = IN2024 / "aligned_divergence.csv"
    p25 = OUT / "aligned_divergence.csv"
    if not p24.exists() or not p25.exists():
        return {}, pd.DataFrame(), pd.DataFrame()
    a = pd.read_csv(p24).rename(columns={
        "effect_size": "effect_2024", "winner_median": "winner_median_2024", "control_median": "control_median_2024",
        "abs_effect": "abs_effect_2024",
    })
    b = pd.read_csv(p25).rename(columns={
        "effect_size": "effect_2025", "winner_median": "winner_median_2025", "control_median": "control_median_2025",
        "abs_effect": "abs_effect_2025",
    })
    m = a.merge(b, on=["offset_h", "feature"], how="inner")
    m["same_direction"] = np.sign(m.effect_2024) == np.sign(m.effect_2025)
    m["min_abs_effect"] = np.minimum(m.abs_effect_2024, m.abs_effect_2025)
    m["mean_abs_effect"] = (m.abs_effect_2024 + m.abs_effect_2025) / 2
    stable = m[(m.offset_h <= 0) & m.same_direction & (m.abs_effect_2024 >= .25) & (m.abs_effect_2025 >= .25)].copy()
    stable = stable.sort_values(["min_abs_effect", "mean_abs_effect"], ascending=False)
    m.to_csv(OUT / "cross_year_all_divergence.csv", index=False)
    stable.to_csv(OUT / "cross_year_stable_divergence.csv", index=False)

    onset = pd.DataFrame()
    o24 = IN2024 / "divergence_onset.csv"
    o25 = OUT / "divergence_onset.csv"
    if o24.exists() and o25.exists():
        x24 = pd.read_csv(o24).add_suffix("_2024").rename(columns={"feature_2024": "feature"})
        x25 = pd.read_csv(o25).add_suffix("_2025").rename(columns={"feature_2025": "feature"})
        onset = x24.merge(x25, on="feature", how="inner")
        onset.to_csv(OUT / "cross_year_onset_comparison.csv", index=False)

    summary = {
        "comparable_offset_features": int(len(m)),
        "same_direction_share_pre_event": float(m[m.offset_h <= 0].same_direction.mean()),
        "stable_effect_ge_0_25_both_years": int(len(stable)),
    }
    return summary, stable, onset


def motif_cross_year():
    p24 = IN2024 / "motif_enrichment.csv"
    p25 = OUT / "motif_enrichment.csv"
    if not p24.exists() or not p25.exists():
        return pd.DataFrame()
    rows = []
    for year, path in [(2024, p24), (2025, p25)]:
        z = pd.read_csv(path)
        for w, g in z.groupby("motif_window"):
            g = g.sort_values(["lift", "signals"], ascending=[False, False])
            r = g.iloc[0]
            rows.append({
                "year": year, "window_h": int(w), "best_precision": float(r.precision),
                "best_lift": float(r.lift), "signals": int(r.signals), "wins": int(r.wins),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "cross_year_motif_summary.csv", index=False)
    return out


def main():
    base.OUT = OUT
    print("Loading BTC 2025 hourly history", flush=True)
    raw = load_symbol(SYMBOL, INTERVAL, LOAD_START, LOAD_END)
    if len(raw) < 10000:
        raise RuntimeError(f"Insufficient BTC history: {len(raw)}")
    x = base.build_indicators(raw)
    paths = base.classify_paths(x)
    for col in paths.columns:
        x[col] = paths[col].to_numpy()
    extra = base.future_extra_labels(x)
    for col in extra.columns:
        x[col] = extra[col].to_numpy()

    feature_cols = [c for c in base.PROFILE_COLS + [
        "abs_body_pct", "avg_trade_quote_rel24", "avg_trade_quote_rel72", "avg_trade_quote_rel240", "obv_delta",
        "volume_rel24", "volume_rel72", "volume_rel240", "quote_volume_rel24", "quote_volume_rel72", "quote_volume_rel240",
        "trades_rel24", "trades_rel72", "trades_rel240", "close_vs_ema168", "close_vs_ema240", "close_vs_sma24", "close_vs_sma72", "close_vs_sma168", "close_vs_sma240",
        "ema168_slope_24h", "ema_gap_6_24", "ema_gap_24_72", "ema_gap_72_168", "atr6_pct", "atr24_pct", "atr72_pct",
        "close_vs_vwap240", "dist_high24", "dist_high72", "dist_high240", "rebound_low24", "rebound_low72", "rebound_low240",
    ] if c in x.columns]
    feature_cols = list(dict.fromkeys(feature_cols))
    temporal, _ = base.temporal_feature_frame(x, feature_cols)

    decision_mask = (x.time >= DECISION_START) & (x.time <= DECISION_END) & x.path_valid.eq(1)
    decision_idx = np.flatnonzero(decision_mask.to_numpy())
    decision_idx = decision_idx[decision_idx >= 239]
    meta_cols = [
        "time", "w3_2_24h", "w5_2_48h", "w5_3_48h", "w7_3_72h", "w10_4_72h",
        "target3_hour", "target5_hour", "stop2_hour", "mfe_24h", "mae_24h", "mfe_48h", "mae_48h",
        "mfe_72h", "mae_72h", "end_net_24h", "end_net_48h", "end_net_72h",
        "w3_2_48h", "w5_2_24h", "t_up1", "t_up2", "t_up3", "t_up5", "t_dn1", "t_dn2", "t_dn3",
    ]
    meta = x.loc[decision_idx, meta_cols].copy()
    meta["row_idx"] = decision_idx
    meta.reset_index(drop=True, inplace=True)
    y = meta.w3_2_24h.to_numpy(int)
    print("Decision hours", len(meta), "W3", int(y.sum()), "rate", float(y.mean()), flush=True)

    static = x.loc[decision_idx, feature_cols].reset_index(drop=True).astype(np.float32)
    temp = temporal.loc[decision_idx].reset_index(drop=True)
    seq24, seq72, seq240 = base.make_sequences(x, decision_idx)

    meta.to_parquet(OUT / "hourly_outcomes_2025.parquet", index=False, compression="zstd")
    static.to_parquet(OUT / "static_indicators_2025.parquet", index=False, compression="zstd")
    temp.to_parquet(OUT / "temporal_features_2025.parquet", index=False, compression="zstd")
    np.savez_compressed(OUT / "raw_sequences_2025.npz", seq24=seq24, seq72=seq72, seq240=seq240, channels=np.asarray(base.SEQ_CHANNELS))

    decisions = meta[["row_idx", "time", "w3_2_24h"]].copy()
    anchors = base.event_anchors(decisions)
    controls = base.matched_controls(decisions, anchors, x)
    anchors.to_csv(OUT / "w3_event_anchors.csv", index=False)
    controls.to_csv(OUT / "matched_controls.csv", index=False)
    base.aligned_profiles(x, anchors, controls)
    base.univariate_scan(temp, y)
    anchor_rows = set(anchors.row_idx.astype(int))
    anchor_mask = np.array([int(r) in anchor_rows for r in decision_idx])
    base.motif_scan(seq24, seq72, seq240, y, anchor_mask)
    fit_models_2025(meta, static, temp, seq24, seq72, seq240, y)

    oco = oco_surface(x, anchors)
    oco.to_csv(OUT / "oco_winner_geometry_2025.csv", index=False)
    cross_summary, stable, onset = cross_year_comparison()
    motifs_cross = motif_cross_year()

    mm = pd.read_csv(OUT / "model_metrics.csv")
    fr = pd.read_csv(OUT / "precision_frequency_frontier.csv")
    uni = pd.read_csv(OUT / "univariate_temporal_features.csv")
    div = pd.read_csv(OUT / "divergence_onset.csv")
    motifs = pd.read_csv(OUT / "motif_enrichment.csv")
    summary = {
        "study": "BTC forensic transition 2025",
        "status": "RESEARCH ONLY - no live strategy or automation changes",
        "decision_start": str(DECISION_START), "decision_end": str(DECISION_END),
        "hours": int(len(meta)), "primary_w3_hours": int(y.sum()), "primary_w3_rate": float(y.mean()),
        "w3_event_anchors": int(len(anchors)), "windows_hours": list(base.WINDOWS),
        "sequence_channels": base.SEQ_CHANNELS, "temporal_input_features": int(temp.shape[1]),
        "2026_touched": False, "cross_year": cross_summary,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# BTC 2025 Forensic Transition Study", "",
        "**RESEARCH ONLY. No live strategy or automation changes. 2026 was not read.**", "",
        f"- Decision hours: **{len(meta):,}**",
        f"- W3 (+3% before -2% within 24h, net of modelled costs) hours: **{int(y.sum()):,} ({y.mean():.2%})**",
        f"- W3 event anchors after 12h episode clustering: **{len(anchors):,}**",
        f"- Temporal engineered features: **{temp.shape[1]:,}**",
        "- Rolling contexts: **24h, 72h, 240h**",
        "- 2025 was analysed independently before cross-year comparison.",
        "- 2026 was **not read or evaluated**.", "",
        "## Internal 2025 model comparison", "", mm.to_markdown(index=False, floatfmt=".4f"), "",
        "## Precision/frequency frontier", "", fr.to_markdown(index=False, floatfmt=".4f"), "",
        "## Strongest 2025 temporal features", "", uni.head(25).to_markdown(index=False, floatfmt=".4f"), "",
        "## Strongest 2025 pre-event divergence timing", "", div.head(25).to_markdown(index=False, floatfmt=".4f"), "",
        "## Highest-enrichment 2025 sequence motifs", "", motifs.head(25).to_markdown(index=False, floatfmt=".4f"), "",
        "## 2025 W3-winner OCO geometry", "", oco.head(15).to_markdown(index=False, floatfmt=".4f"), "",
        "## Stable 2024 + 2025 pre-event divergences", "",
        (stable.head(30).to_markdown(index=False, floatfmt=".4f") if len(stable) else "No features met the predeclared stable-effect criterion."), "",
        "## Cross-year motif summary", "", (motifs_cross.to_markdown(index=False, floatfmt=".4f") if len(motifs_cross) else "Unavailable"), "",
    ]
    (OUT / "ANALYSIS.md").write_text("\n".join(lines))
    print("\n".join(lines[:18]), flush=True)


if __name__ == "__main__":
    main()
