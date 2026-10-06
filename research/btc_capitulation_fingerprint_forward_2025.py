from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from binance_data import load_symbol
from btc_forensic_transition_2024 import build_indicators, temporal_feature_frame, FEE_RATE, SLIPPAGE_RATE

IN2024 = Path("research/input/btc_forensic_2024")
TIGHT = Path("research/results/daily_return_harvest/btc_w3_matched_lookalikes_tight_2024/top_unique_trajectory_features.csv")
OUT = Path("research/results/daily_return_harvest/btc_capitulation_fingerprint_forward_2025")
OUT.mkdir(parents=True, exist_ok=True)

STATE_FEATURES = [
    "rsi6", "rsi14", "bb_z", "stoch_k", "cci20", "macd_hist_pct",
    "close_vs_ema6", "close_vs_ema12", "close_vs_ema24", "close_vs_vwap24",
    "atr14_pct", "log_ret",
]
TARGETS = [0.015, 0.020, 0.025, 0.030, 0.040, 0.050]
STOPS = [0.0075, 0.010, 0.015, 0.020, 0.025, 0.030]
HORIZON_H = 168
MODEL_C = 0.20


def make_pipe() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=MODEL_C, penalty="l2", max_iter=5000, solver="lbfgs")),
    ])


def select_trajectory_features(temp: pd.DataFrame, top: pd.DataFrame, max_features: int = 18) -> list[str]:
    z = top.copy()
    for c in ["abs_paired_effect", "sign_consistency", "fdr_q"]:
        z[c] = pd.to_numeric(z[c], errors="coerce")
    z = z[
        z.feature.isin(temp.columns)
        & z.shape_only.astype(bool)
        & z.fdr_q.le(0.01)
        & z.sign_consistency.ge(0.62)
        & z.abs_paired_effect.ge(0.30)
    ].sort_values(["abs_paired_effect", "sign_consistency"], ascending=[False, False])

    # Greedy de-correlation makes the fingerprint compact rather than retaining
    # many near-identical moving-average displacement features.
    selected: list[str] = []
    clean = temp.replace([np.inf, -np.inf], np.nan)
    for f in z.feature.astype(str):
        if f in selected:
            continue
        if not selected:
            selected.append(f)
        else:
            c = clean[selected + [f]].corr(method="spearman").loc[f, selected].abs()
            if c.max(skipna=True) < 0.88:
                selected.append(f)
        if len(selected) >= max_features:
            break
    return selected


def onset_mask(times: pd.Series, score: np.ndarray, threshold: float) -> np.ndarray:
    on = np.isfinite(score) & (score >= threshold)
    prev = np.r_[False, on[:-1]]
    gap = np.r_[True, times.iloc[1:].reset_index(drop=True).sub(times.iloc[:-1].reset_index(drop=True)).dt.total_seconds().to_numpy() != 3600]
    return on & (~prev | gap)


def threshold_frontier(times: pd.Series, y: np.ndarray, scores: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict]:
    days = max(1.0, (times.iloc[-1] - times.iloc[0]).total_seconds() / 86400.0)
    rows = []
    for model_name, s in scores.items():
        vals = s[np.isfinite(s)]
        thresholds = np.unique(np.quantile(vals, np.linspace(0.50, 0.999, 500)))
        for min_rate in [0.10, 0.25, 0.50, 1.00]:
            best = None
            for th in thresholds:
                om = onset_mask(times, s, float(th))
                n = int(om.sum())
                if n == 0:
                    continue
                rate = n / days
                if rate + 1e-12 < min_rate:
                    continue
                wins = int(y[om].sum())
                precision = wins / n
                cand = (precision, wins / days, -n, float(th), n, wins, rate)
                if best is None or cand[:3] > best[:3]:
                    best = cand
            if best is not None:
                precision, wins_day, _, th, n, wins, rate = best
                rows.append({
                    "model": model_name, "min_signals_per_day": min_rate, "threshold": th,
                    "signals": n, "wins": wins, "precision": precision,
                    "signals_per_day": rate, "wins_per_day": wins_day,
                })
    fr = pd.DataFrame(rows).sort_values(["min_signals_per_day", "precision", "wins_per_day"], ascending=[True, False, False])
    primary_pool = fr[fr.min_signals_per_day.eq(0.10)].copy()
    if primary_pool.empty:
        raise RuntimeError("No 0.10/day fingerprint threshold found")
    primary = primary_pool.sort_values(["precision", "wins_per_day"], ascending=False).iloc[0].to_dict()
    return fr, primary


def fit_oof(meta: pd.DataFrame, static: pd.DataFrame, temp: pd.DataFrame, traj: list[str]):
    times = pd.to_datetime(meta.time, utc=True).reset_index(drop=True)
    y = meta.w3_2_24h.astype(int).to_numpy()
    sets = {
        "STATE_ONLY": static[STATE_FEATURES].copy(),
        "TRAJECTORY_ONLY": temp[traj].copy(),
        "COMBINED": pd.concat([static[STATE_FEATURES].reset_index(drop=True), temp[traj].reset_index(drop=True)], axis=1),
    }
    scores = {k: np.full(len(meta), np.nan, dtype=float) for k in sets}
    for q in [1, 2, 3, 4]:
        te = times.dt.quarter.eq(q).to_numpy()
        tr = ~te
        for name, X in sets.items():
            pipe = make_pipe()
            pipe.fit(X.loc[tr], y[tr])
            scores[name][te] = pipe.predict_proba(X.loc[te])[:, 1]
    frontier, primary = threshold_frontier(times, y, scores)
    return times, y, sets, scores, frontier, primary


def signal_times(times: pd.Series, score: np.ndarray, threshold: float) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(times[onset_mask(times, score, threshold)])


def index_by_time(raw: pd.DataFrame) -> dict[pd.Timestamp, int]:
    return {pd.Timestamp(t): i for i, t in enumerate(pd.to_datetime(raw.time, utc=True))}


def simulate_oco(raw: pd.DataFrame, decision_time: pd.Timestamp, target: float, stop: float, horizon_h: int = HORIZON_H) -> dict | None:
    idx = index_by_time(raw).get(pd.Timestamp(decision_time))
    if idx is None or idx + 1 >= len(raw):
        return None
    ei = idx + 1
    open_px = float(raw.open.iloc[ei])
    if not np.isfinite(open_px) or open_px <= 0:
        return None

    # Same net-of-cost convention used by the W3 research labels.
    entry_cost = open_px * (1 + SLIPPAGE_RATE) * (1 + FEE_RATE)
    exit_factor = (1 - SLIPPAGE_RATE) * (1 - FEE_RATE)
    target_raw = entry_cost * (1 + target) / exit_factor
    stop_raw = entry_cost * (1 - stop) / exit_factor
    end = min(len(raw), ei + horizon_h)

    for j in range(ei, end):
        hi = float(raw.high.iloc[j]); lo = float(raw.low.iloc[j])
        hit_t = hi >= target_raw
        hit_s = lo <= stop_raw
        h = j - ei
        if hit_t and hit_s:
            return {"return": -stop, "reason": "SAME_BAR_STOP", "hours": h, "resolved": 1}
        if hit_s:
            return {"return": -stop, "reason": "STOP", "hours": h, "resolved": 1}
        if hit_t:
            return {"return": target, "reason": "TARGET", "hours": h, "resolved": 1}

    j = end - 1
    close_px = float(raw.close.iloc[j])
    ret = close_px * exit_factor / entry_cost - 1
    return {"return": ret, "reason": "UNRESOLVED_MTM", "hours": max(0, j - ei), "resolved": 0}


def oco_grid(raw: pd.DataFrame, sig_times: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for tp in TARGETS:
        for sl in STOPS:
            trades = [simulate_oco(raw, t, tp, sl) for t in sig_times]
            trades = [x for x in trades if x is not None]
            if not trades:
                continue
            r = np.asarray([x["return"] for x in trades], float)
            hrs = np.asarray([x["hours"] for x in trades], int)
            resolved = np.asarray([x["resolved"] for x in trades], int)
            reasons = pd.Series([x["reason"] for x in trades])
            geo = float(np.exp(np.mean(np.log1p(np.clip(r, -0.999, None)))) - 1)
            rows.append({
                "target": tp, "stop": sl, "trades": len(trades),
                "mean_return": float(np.mean(r)), "median_return": float(np.median(r)), "geomean_return": geo,
                "positive_rate": float(np.mean(r > 0)), "target_rate": float(reasons.eq("TARGET").mean()),
                "stop_rate": float(reasons.isin(["STOP", "SAME_BAR_STOP"]).mean()),
                "resolved_24h": float(np.mean((resolved == 1) & (hrs < 24))),
                "resolved_48h": float(np.mean((resolved == 1) & (hrs < 48))),
                "resolved_72h": float(np.mean((resolved == 1) & (hrs < 72))),
                "resolved_168h": float(np.mean(resolved == 1)),
                "median_hours": float(np.median(hrs)), "p90_hours": float(np.quantile(hrs, .90)),
                "worst_return": float(np.min(r)),
            })
    out = pd.DataFrame(rows)
    viable = out[out.resolved_168h.ge(0.80)]
    if viable.empty:
        viable = out
    best_idx = viable.sort_values(["geomean_return", "mean_return", "resolved_168h"], ascending=False).index[0]
    out["selected_2024"] = False
    out.loc[best_idx, "selected_2024"] = True
    return out.sort_values(["selected_2024", "geomean_return"], ascending=[False, False])


def build_forward_features(start: str, end: str, decision_start: pd.Timestamp, decision_end: pd.Timestamp, static_cols: list[str], traj: list[str]):
    raw = load_symbol("BTCUSDT", "1h", start, end)
    x = build_indicators(raw)
    temp, _ = temporal_feature_frame(x, static_cols)
    mask = (pd.to_datetime(x.time, utc=True) >= decision_start) & (pd.to_datetime(x.time, utc=True) <= decision_end)
    idx = np.flatnonzero(mask.to_numpy())
    idx = idx[idx >= 239]
    times = pd.to_datetime(x.time.iloc[idx], utc=True).reset_index(drop=True)
    static = x.loc[idx, static_cols].reset_index(drop=True).astype(np.float32)
    temp = temp.loc[idx].reset_index(drop=True)
    return raw, times, static, temp


def fixed_w3_labels(raw: pd.DataFrame, times: pd.Series) -> np.ndarray:
    out = []
    for t in times:
        z = simulate_oco(raw, pd.Timestamp(t), 0.03, 0.02, horizon_h=24)
        out.append(1 if z is not None and z["reason"] == "TARGET" else 0)
    return np.asarray(out, int)


def evaluate_forward(raw: pd.DataFrame, times: pd.Series, score: np.ndarray, threshold: float, target: float, stop: float):
    om = onset_mask(times, score, threshold)
    st = pd.DatetimeIndex(times[om])
    w3 = fixed_w3_labels(raw, pd.Series(st))
    chosen = [simulate_oco(raw, t, target, stop) for t in st]
    original = [simulate_oco(raw, t, 0.03, 0.02) for t in st]
    rows = []
    for i, t in enumerate(st):
        c = chosen[i]; o = original[i]
        rows.append({
            "time": t, "score": float(score[om][i]), "w3_3_2_24h": int(w3[i]),
            "chosen_oco_return": c["return"], "chosen_oco_reason": c["reason"], "chosen_oco_hours": c["hours"],
            "original_3_2_return": o["return"], "original_3_2_reason": o["reason"], "original_3_2_hours": o["hours"],
        })
    detail = pd.DataFrame(rows)
    days = max(1.0, (times.iloc[-1] - times.iloc[0]).total_seconds() / 86400.0)
    def summarise(prefix: str):
        if detail.empty:
            return {}
        r = detail[f"{prefix}_return"].to_numpy(float)
        reason = detail[f"{prefix}_reason"]
        return {
            f"{prefix}_mean_return": float(np.mean(r)),
            f"{prefix}_geomean_return": float(np.exp(np.mean(np.log1p(np.clip(r, -0.999, None)))) - 1),
            f"{prefix}_positive_rate": float(np.mean(r > 0)),
            f"{prefix}_target_rate": float(reason.eq("TARGET").mean()),
            f"{prefix}_stop_rate": float(reason.isin(["STOP", "SAME_BAR_STOP"]).mean()),
        }
    summary = {
        "signals": int(len(detail)), "signals_per_day": float(len(detail) / days),
        "w3_wins": int(detail.w3_3_2_24h.sum()) if len(detail) else 0,
        "w3_precision": float(detail.w3_3_2_24h.mean()) if len(detail) else np.nan,
        **summarise("chosen_oco"), **summarise("original_3_2"),
    }
    return detail, summary


def main():
    print("Loading preserved BTC 2024 forensic dataset", flush=True)
    meta24 = pd.read_parquet(IN2024 / "hourly_outcomes_2024.parquet")
    static24 = pd.read_parquet(IN2024 / "static_indicators_2024.parquet")
    temp24 = pd.read_parquet(IN2024 / "temporal_features_2024.parquet")
    top = pd.read_csv(TIGHT)
    meta24["time"] = pd.to_datetime(meta24.time, utc=True)

    traj = select_trajectory_features(temp24, top)
    if len(traj) < 8:
        raise RuntimeError(f"Too few trajectory fingerprint features selected: {traj}")
    pd.DataFrame({"feature": STATE_FEATURES + traj, "type": ["STATE"] * len(STATE_FEATURES) + ["TRAJECTORY"] * len(traj)}).to_csv(OUT / "fingerprint_features.csv", index=False)

    times24, y24, sets24, scores24, frontier, primary = fit_oof(meta24, static24, temp24, traj)
    frontier.to_csv(OUT / "fingerprint_2024_oof_frontier.csv", index=False)
    model_name = str(primary["model"]); threshold = float(primary["threshold"])
    sig24 = signal_times(times24, scores24[model_name], threshold)

    # Fit the final frozen 2024 model before loading any 2025 rows.
    final_pipe = make_pipe()
    final_pipe.fit(sets24[model_name], y24)
    joblib.dump(final_pipe, OUT / "fingerprint_model_2024.joblib")

    # OCO discovery uses 2024 only. End signal eligibility on Dec 24 so the
    # seven-day analytical horizon never needs 2025 candles.
    raw24 = load_symbol("BTCUSDT", "1h", "2023-11-01", "2024-12-31")
    sig24_oco = sig24[sig24 <= pd.Timestamp("2024-12-24 23:00:00", tz="UTC")]
    grid = oco_grid(raw24, sig24_oco)
    grid.to_csv(OUT / "oco_grid_2024.csv", index=False)
    best = grid[grid.selected_2024].iloc[0]
    chosen_target, chosen_stop = float(best.target), float(best.stop)

    manifest = {
        "status": "RESEARCH ONLY - no live strategy or automation changes",
        "discovery_year": 2024,
        "forward_year": 2025,
        "2026_touched": False,
        "model": model_name,
        "model_c": MODEL_C,
        "threshold": threshold,
        "state_features": STATE_FEATURES,
        "trajectory_features": traj,
        "2024_oof_primary": {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer,)) else v) for k, v in primary.items()},
        "oco_grid_targets": TARGETS,
        "oco_grid_stops": STOPS,
        "oco_analysis_horizon_h": HORIZON_H,
        "selected_oco_target": chosen_target,
        "selected_oco_stop": chosen_stop,
        "selected_oco_2024_metrics": {k: (bool(v) if isinstance(v, (np.bool_, bool)) else float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer,)) else v) for k, v in best.to_dict().items()},
    }
    (OUT / "fingerprint_manifest_frozen_before_2025.json").write_text(json.dumps(manifest, indent=2, default=str))
    print("FROZEN 2024 fingerprint and OCO", json.dumps({"model": model_name, "threshold": threshold, "target": chosen_target, "stop": chosen_stop}), flush=True)

    # First forward exposure: 2025. Load only through 2025-12-31; 2026 remains untouched.
    raw25, times25, static25, temp25 = build_forward_features(
        "2024-11-01", "2025-12-31",
        pd.Timestamp("2025-01-01 00:00:00", tz="UTC"),
        pd.Timestamp("2025-12-24 23:00:00", tz="UTC"),
        list(static24.columns), traj,
    )
    if model_name == "STATE_ONLY":
        X25 = static25[STATE_FEATURES]
    elif model_name == "TRAJECTORY_ONLY":
        X25 = temp25[traj]
    else:
        X25 = pd.concat([static25[STATE_FEATURES].reset_index(drop=True), temp25[traj].reset_index(drop=True)], axis=1)
    score25 = final_pipe.predict_proba(X25)[:, 1]
    detail25, sum25 = evaluate_forward(raw25, times25, score25, threshold, chosen_target, chosen_stop)
    detail25.to_csv(OUT / "forward_2025_signals.csv", index=False)
    pd.DataFrame([sum25]).to_csv(OUT / "forward_2025_summary.csv", index=False)

    baseline25 = fixed_w3_labels(raw25, times25)
    summary = {
        **manifest,
        "2024_signal_count_for_oco": int(len(sig24_oco)),
        "2024_w3_baseline_rate": float(y24.mean()),
        "2025_decision_hours": int(len(times25)),
        "2025_full_stream_w3_baseline_rate": float(baseline25.mean()),
        "2025_forward": {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer,)) else v) for k, v in sum25.items()},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    original_row = grid[np.isclose(grid.target, .03) & np.isclose(grid.stop, .02)].iloc[0]
    lines = [
        "# BTC Capitulation Fingerprint -> 2025 Forward Test", "",
        "**RESEARCH ONLY. No live strategy or automation changes. 2026 was not read.**", "",
        "## Frozen fingerprint", "",
        f"- Primary model selected on 2024 OOF full-stream frontier: **{model_name}**",
        f"- Fingerprint features: **{len(STATE_FEATURES)} state + {len(traj)} trajectory** (model uses the subset implied by {model_name}).",
        f"- Frozen probability threshold: **{threshold:.6f}**",
        f"- 2024 OOF W3 precision: **{float(primary['precision']):.2%}** across **{int(primary['signals'])}** signal onsets ({float(primary['signals_per_day']):.3f}/day).", "",
        "## 2024 OCO discovery", "",
        f"- Pre-declared grid: {len(TARGETS)} targets x {len(STOPS)} stops = **{len(TARGETS)*len(STOPS)} pairs**.",
        f"- Selected on 2024 only: **+{chosen_target:.2%} / -{chosen_stop:.2%}**.",
        f"- Selected pair 2024 geometric mean/trade: **{float(best.geomean_return):.3%}**; mean: **{float(best.mean_return):.3%}**; target rate: **{float(best.target_rate):.2%}**; 168h resolved: **{float(best.resolved_168h):.2%}**.",
        f"- Original +3%/-2% on same 2024 signals: geometric mean/trade **{float(original_row.geomean_return):.3%}**; mean **{float(original_row.mean_return):.3%}**; target rate **{float(original_row.target_rate):.2%}**.",
        "- The 168h horizon is analytical only; it is not a proposed forced exit.", "",
        "## First untouched-year exposure: 2025", "",
        f"- Eligible decision hours: **{len(times25):,}**; raw W3 baseline: **{baseline25.mean():.2%}**.",
        f"- Frozen fingerprint signal onsets: **{sum25['signals']}** ({sum25['signals_per_day']:.3f}/day).",
        f"- W3 (+3/-2 within 24h) precision: **{sum25['w3_precision']:.2%}** ({sum25['w3_wins']}/{sum25['signals']}).",
        f"- Frozen selected OCO mean return/trade: **{sum25['chosen_oco_mean_return']:.3%}**; geometric mean **{sum25['chosen_oco_geomean_return']:.3%}**; positive rate **{sum25['chosen_oco_positive_rate']:.2%}**.",
        f"- Original +3/-2 mean return/trade: **{sum25['original_3_2_mean_return']:.3%}**; geometric mean **{sum25['original_3_2_geomean_return']:.3%}**; positive rate **{sum25['original_3_2_positive_rate']:.2%}**.", "",
        "## 2024 full-stream threshold frontier", "", frontier.to_markdown(index=False, floatfmt=".4f"), "",
        "## Top 2024 OCO pairs", "", grid.head(12).to_markdown(index=False, floatfmt=".4f"), "",
    ]
    (OUT / "ANALYSIS.md").write_text("\n".join(lines))
    print("\n".join(lines[:32]), flush=True)


if __name__ == "__main__":
    main()
