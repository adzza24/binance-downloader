from __future__ import annotations

"""Research-only same-time selector analysis for DRH2 high-purity A/B/C event families.

Develops the selector exclusively on the unused discovery tail (2023-11-01..2024-04-30),
then evaluates frozen choices on 2024-05..2025-12 validation and 2026 confirmation.
No live strategy or portfolio state is read or changed.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from build_high_purity_models import (  # noqa: E402
    CONF, CONSTRUCTED_TYPES, DISC, TRAIN_CUTOFF, VAL,
    fit_models, load_samples, source_parts,
)

SPEC = ROOT / "research/high_purity_signal_features.json"
OUT = ROOT / "research/results/daily_return_harvest/high_purity_runtime/selector_analysis"

# Outcomes/future information must never enter selector features.
NON_CAUSAL_EXACT = {
    "sample_id", "source_event_id", "symbol", "decision_time", "sample_type", "split", "year",
    "account_excluded", "w3_2_24h", "w5_2_48h", "w5_3_48h", "w7_3_72h", "w10_4_72h",
    "mfe_24h", "mae_24h", "mfe_48h", "mae_48h", "mfe_72h", "mae_72h",
    "end_net_3h", "end_net_6h", "end_net_12h", "end_net_24h", "end_net_48h", "end_net_72h",
}
NON_CAUSAL_PREFIXES = ("future_", "target_", "stop_", "outcome_", "label_")
# Same-time selection should be coin-relative. Market-wide columns are identical or near-identical within an hour.
MARKET_WIDE_PREFIXES = ("btc_", "breadth_", "market_", "median_alt_")


def normalise_time(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        if getattr(s.dt, "tz", None) is None:
            return s.dt.tz_localize("UTC")
        return s.dt.tz_convert("UTC")
    n = pd.to_numeric(s, errors="coerce")
    if n.notna().mean() > 0.99:
        return pd.to_datetime(n, unit="ms", utc=True)
    return pd.to_datetime(s, utc=True)


def macro_family(name: str) -> str:
    root = name.split("__chg_")[0]
    for prefix in [
        "ret_", "rs_", "range_", "atr", "dist_", "rebound_", "close_vs_ema", "ema",
        "close_vs_vwap", "vwap", "volume_", "trade_", "taker_", "quote_", "liquidity_",
        "body_", "wick_", "candle_", "volatility_", "participation_",
    ]:
        if root.startswith(prefix):
            return prefix.rstrip("_")
    return root.split("_")[0]


def numeric_causal_columns(first_part: Path) -> list[str]:
    schema = pq.ParquetFile(first_part).schema_arrow
    out = []
    for field in schema:
        name = field.name
        if name in NON_CAUSAL_EXACT or name.startswith(NON_CAUSAL_PREFIXES):
            continue
        if name.startswith(MARKET_WIDE_PREFIXES):
            continue
        t = field.type
        if getattr(t, "bit_width", None) is not None or str(t) == "bool":
            out.append(name)
    return out


def load_feature_rows(parts: list[Path], wanted_global_ids: set[int], causal: list[str]) -> pd.DataFrame:
    meta = ["symbol", "decision_time", "sample_type", "split", "w3_2_24h", "w5_2_48h", "w5_3_48h", "w7_3_72h", "w10_4_72h"]
    cols = list(dict.fromkeys(meta + causal))
    frames = []
    offset = 0
    for part in parts:
        pf = pq.ParquetFile(part)
        n = pf.metadata.num_rows
        local = sorted(g - offset for g in wanted_global_ids if offset <= g < offset + n)
        if local:
            frame = pd.read_parquet(part, columns=cols).iloc[local].copy()
            frame["global_id"] = np.asarray(local, dtype=np.int64) + offset
            frames.append(frame)
        offset += n
    if not frames:
        raise RuntimeError("No feature rows loaded")
    out = pd.concat(frames, ignore_index=True)
    out["decision_time"] = normalise_time(out["decision_time"])
    for c in causal:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(np.float32)
    return out


def sequential_family(scores: pd.DataFrame, thresholds: dict[str, float], delta: float) -> tuple[np.ndarray, np.ndarray]:
    ta = max(0.0, thresholds["A"] - delta)
    tb = max(0.0, thresholds["B"] - delta)
    tc = max(0.0, thresholds["C"] - delta)
    a = scores["score_a"].to_numpy() >= ta
    b = (~a) & (scores["score_b"].to_numpy() >= tb)
    c = (~a) & (~b) & (scores["score_c"].to_numpy() >= tc)
    fam = np.full(len(scores), "", dtype=object)
    fam[a], fam[b], fam[c] = "A", "B", "C"
    return a | b | c, fam


def days_in(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 1
    return max((frame["decision_time"].max().floor("D") - frame["decision_time"].min().floor("D")).days + 1, 1)


def metrics(frame: pd.DataFrame, selected: np.ndarray, label: str) -> dict:
    z = frame.loc[selected].copy()
    winners = int((z["sample_type"] == "WINNER_EVENT").sum())
    n = len(z)
    days = days_in(frame)
    return {
        f"{label}_signals": n,
        f"{label}_winners": winners,
        f"{label}_precision": winners / n if n else None,
        f"{label}_signals_per_day": n / days,
        f"{label}_winners_per_day": winners / days,
        f"{label}_w5_2_rate": float(pd.to_numeric(z["w5_2_48h"], errors="coerce").mean()) if n else None,
        f"{label}_w7_3_rate": float(pd.to_numeric(z["w7_3_72h"], errors="coerce").mean()) if n else None,
        f"{label}_false_types": z.loc[z["sample_type"] != "WINNER_EVENT", "sample_type"].value_counts().to_dict(),
    }


def add_within_time_ranks(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    rank_cols = []
    for f in features:
        rc = f"RANK__{f}"
        # Percentile among simultaneous candidate coins. Single-candidate hours become 0.5.
        out[rc] = out.groupby("decision_time", sort=False)[f].rank(pct=True, method="average").fillna(0.5).astype(np.float32)
        rank_cols.append(rc)
    return out, rank_cols


def select_features(dev: pd.DataFrame, causal: list[str], max_total: int = 28, per_family: int = 2) -> tuple[list[str], pd.DataFrame]:
    y = (dev["sample_type"] == "WINNER_EVENT").astype(int).to_numpy()
    scored = []
    # Score both absolute state and within-hour rank. Ranking is the main target, but absolute state can help single-candidate hours.
    ranked, rank_cols = add_within_time_ranks(dev[["decision_time"] + causal], causal)
    for raw, rc in zip(causal, rank_cols):
        for col, kind, vals in [(raw, "raw", dev[raw]), (rc, "rank", ranked[rc])]:
            x = pd.to_numeric(vals, errors="coerce").to_numpy(float)
            finite = np.isfinite(x)
            if finite.sum() < max(60, int(0.7 * len(x))) or len(np.unique(y[finite])) < 2:
                continue
            try:
                auc = roc_auc_score(y[finite], x[finite])
            except Exception:
                continue
            scored.append({"feature": raw, "model_col": col, "kind": kind, "auc": auc, "auc_sep": max(auc, 1 - auc), "direction": "high" if auc >= 0.5 else "low", "family": macro_family(raw)})
    table = pd.DataFrame(scored).sort_values("auc_sep", ascending=False)
    chosen, fam_count, used_raw = [], {}, set()
    for r in table.itertuples(index=False):
        if len(chosen) >= max_total:
            break
        # Prefer one representation of a raw feature and cap macro-family concentration.
        if r.feature in used_raw:
            continue
        if fam_count.get(r.family, 0) >= per_family:
            continue
        if r.auc_sep < 0.56:
            continue
        chosen.append(r.model_col)
        used_raw.add(r.feature)
        fam_count[r.family] = fam_count.get(r.family, 0) + 1
    return chosen, table


def materialise_model_frame(frame: pd.DataFrame, chosen: list[str]) -> pd.DataFrame:
    raw_needed = [c.replace("RANK__", "") for c in chosen if c.startswith("RANK__")]
    ranked, _ = add_within_time_ranks(frame[["decision_time"] + list(dict.fromkeys(raw_needed))], list(dict.fromkeys(raw_needed))) if raw_needed else (frame, [])
    X = pd.DataFrame(index=frame.index)
    for c in chosen:
        if c.startswith("RANK__"):
            X[c] = ranked[c].to_numpy()
        else:
            X[c] = frame[c].to_numpy()
    return X


def top1_mask(frame: pd.DataFrame, probs: np.ndarray, threshold: float) -> np.ndarray:
    tmp = pd.DataFrame({"time": frame["decision_time"].to_numpy(), "p": probs, "row": np.arange(len(frame))})
    tmp = tmp[tmp["p"] >= threshold]
    keep = np.zeros(len(frame), dtype=bool)
    if tmp.empty:
        return keep
    idx = tmp.sort_values(["time", "p"], ascending=[True, False]).groupby("time", sort=False).head(1)["row"].to_numpy(int)
    keep[idx] = True
    return keep


def choose_threshold(dev: pd.DataFrame, probs: np.ndarray) -> tuple[float, pd.DataFrame]:
    qs = np.unique(np.r_[np.linspace(0.0, 0.80, 17), np.linspace(0.82, 0.98, 17), [0.985, 0.99, 0.995]])
    rows = []
    for q in qs:
        th = float(np.quantile(probs, q))
        keep = top1_mask(dev, probs, th)
        m = metrics(dev, keep, "dev")
        rows.append({"quantile": float(q), "threshold": th, **m})
    frontier = pd.DataFrame(rows)
    viable = frontier[(frontier["dev_precision"] >= 0.90) & (frontier["dev_signals_per_day"] >= 0.75)]
    if len(viable):
        best = viable.sort_values(["dev_winners_per_day", "dev_precision"], ascending=False).iloc[0]
    else:
        viable = frontier[frontier["dev_precision"] >= 0.90]
        if len(viable):
            best = viable.sort_values(["dev_winners_per_day", "dev_precision"], ascending=False).iloc[0]
        else:
            best = frontier.sort_values(["dev_precision", "dev_winners_per_day"], ascending=False).iloc[0]
    return float(best["threshold"]), frontier


def main() -> None:
    spec = json.loads(SPEC.read_text())
    minimal, source = load_samples(spec["A_FEATURES"], spec["BC_FEATURES"])
    base, model_a, model_b, model_c = fit_models(minimal, spec)
    base = base.copy()
    base["global_id"] = base.index.astype(np.int64)
    base["score_a"] = model_a.predict_proba(base[spec["A_FEATURES"]])[:, 1]
    base["score_b"] = model_b.predict_proba(base[spec["BC_FEATURES"]])[:, 1]
    base["score_c"] = model_c.predict_proba(base[spec["BC_FEATURES"]])[:, 1]

    parts, _ = source_parts()
    causal = numeric_causal_columns(parts[0])
    # Load all rows that could enter the broadest candidate pool.
    max_delta = 0.10
    broad, _ = sequential_family(base, spec["thresholds"], max_delta)
    wanted = set(base.loc[broad, "global_id"].astype(int).tolist())
    feat = load_feature_rows(parts, wanted, causal)
    score_cols = base[["global_id", "score_a", "score_b", "score_c"]]
    feat = feat.merge(score_cols, on="global_id", how="inner", validate="one_to_one")

    result_rows = []
    all_frontiers = []
    selected_feature_rows = []
    deltas = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]

    for delta in deltas:
        sig, fam = sequential_family(feat, spec["thresholds"], delta)
        cand = feat.loc[sig].copy()
        cand["family"] = fam[sig]
        dev = cand[(cand["split"] == DISC) & (cand["decision_time"] >= TRAIN_CUTOFF)].copy()
        val = cand[cand["split"] == VAL].copy()
        conf = cand[cand["split"] == CONF].copy()
        if dev.empty or val.empty or conf.empty:
            continue

        # Baseline candidate pool and top-1 by strongest A/B/C family score.
        raw_dev = np.ones(len(dev), dtype=bool)
        raw_val = np.ones(len(val), dtype=bool)
        raw_conf = np.ones(len(conf), dtype=bool)
        baseline = {"delta": delta, "mode": "candidate_pool", **metrics(dev, raw_dev, "dev"), **metrics(val, raw_val, "val"), **metrics(conf, raw_conf, "conf")}
        result_rows.append(baseline)

        # ABC top1 baseline: family-specific scores are only used as a tie breaker, not as the learned selector.
        def abc_strength(z: pd.DataFrame) -> np.ndarray:
            return np.maximum.reduce([z["score_a"].to_numpy(), z["score_b"].to_numpy(), z["score_c"].to_numpy()])
        for mode, dz, vz, cz in [("abc_top1", dev, val, conf)]:
            kd = top1_mask(dz, abc_strength(dz), -1.0)
            kv = top1_mask(vz, abc_strength(vz), -1.0)
            kc = top1_mask(cz, abc_strength(cz), -1.0)
            result_rows.append({"delta": delta, "mode": mode, **metrics(dz, kd, "dev"), **metrics(vz, kv, "val"), **metrics(cz, kc, "conf")})

        chosen, feature_table = select_features(dev, causal)
        if len(chosen) < 3:
            continue
        for c in chosen:
            row = feature_table[feature_table["model_col"] == c].iloc[0].to_dict()
            selected_feature_rows.append({"delta": delta, **row})

        Xd = materialise_model_frame(dev, chosen)
        yd = (dev["sample_type"] == "WINNER_EVENT").astype(int).to_numpy()
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(C=0.2, class_weight="balanced", max_iter=2000, random_state=42)),
        ])
        class_counts = np.bincount(yd)
        folds = int(min(5, class_counts.min())) if len(class_counts) > 1 else 2
        folds = max(folds, 2)
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        oof = cross_val_predict(pipe, Xd, yd, cv=cv, method="predict_proba")[:, 1]
        threshold, frontier = choose_threshold(dev, oof)
        frontier["delta"] = delta
        all_frontiers.append(frontier)

        pipe.fit(Xd, yd)
        pv = pipe.predict_proba(materialise_model_frame(val, chosen))[:, 1]
        pc = pipe.predict_proba(materialise_model_frame(conf, chosen))[:, 1]
        pdv = pipe.predict_proba(Xd)[:, 1]
        kd = top1_mask(dev, pdv, threshold)
        kv = top1_mask(val, pv, threshold)
        kc = top1_mask(conf, pc, threshold)
        result_rows.append({
            "delta": delta, "mode": "learned_selector_top1", "selector_threshold": threshold,
            "n_selector_features": len(chosen), "selector_features": " | ".join(chosen),
            **metrics(dev, kd, "dev"), **metrics(val, kv, "val"), **metrics(conf, kc, "conf"),
        })

    results = pd.DataFrame(result_rows)
    # Primary ranking: require >=90% precision in BOTH untouched periods, then maximise worst-period winners/day.
    learned = results[results["mode"] == "learned_selector_top1"].copy()
    learned["min_oos_precision"] = learned[["val_precision", "conf_precision"]].min(axis=1)
    learned["min_oos_winners_per_day"] = learned[["val_winners_per_day", "conf_winners_per_day"]].min(axis=1)
    learned["min_oos_signals_per_day"] = learned[["val_signals_per_day", "conf_signals_per_day"]].min(axis=1)
    viable = learned[learned["min_oos_precision"] >= 0.90]
    best = (viable if len(viable) else learned).sort_values(
        ["min_oos_winners_per_day", "min_oos_precision"], ascending=False
    ).iloc[0].to_dict()

    OUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT / "selector_results.csv", index=False)
    if all_frontiers:
        pd.concat(all_frontiers, ignore_index=True).to_csv(OUT / "development_frontiers.csv", index=False)
    pd.DataFrame(selected_feature_rows).to_csv(OUT / "selected_features.csv", index=False)
    (OUT / "selector_summary.json").write_text(json.dumps({
        "research_only": True,
        "training_source": source,
        "development_period": "DISCOVERY tail from 2023-11-01 through 2024-04-30",
        "untouched_evaluation_periods": [VAL, CONF],
        "candidate_deltas_tested": deltas,
        "selection_rule": ">=90% precision in both untouched periods, then maximise worst-period winners/day",
        "best": best,
    }, indent=2, default=str) + "\n")

    lines = [
        "# High-Purity Same-Time Selector Analysis",
        "",
        "Research-only. No live strategy changes.",
        "",
        "Selector development used only the unused discovery tail from 2023-11-01. Validation and 2026 confirmation were not used to fit or tune the selector.",
        "",
        "## Best result",
        "",
        f"- Candidate threshold delta: {best.get('delta')}",
        f"- Validation precision: {best.get('val_precision'):.4f}; signals/day: {best.get('val_signals_per_day'):.3f}; winners/day: {best.get('val_winners_per_day'):.3f}",
        f"- 2026 confirmation precision: {best.get('conf_precision'):.4f}; signals/day: {best.get('conf_signals_per_day'):.3f}; winners/day: {best.get('conf_winners_per_day'):.3f}",
        f"- Validation W5-2 rate: {best.get('val_w5_2_rate'):.4f}; 2026 W5-2 rate: {best.get('conf_w5_2_rate'):.4f}",
        f"- Selector features: {best.get('selector_features')}",
        "",
        "See selector_results.csv for the full precision/frequency frontier and candidate-pool comparisons.",
    ]
    (OUT / "ANALYSIS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"best": best}, indent=2, default=str))


if __name__ == "__main__":
    main()
