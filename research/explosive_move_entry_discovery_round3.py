from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

import explosive_move_entry_discovery_round2 as r2
from binance_data import load_symbol

OUT = Path("research/results/explosive_move_entry_discovery_round3_final")
START = r2.START
END = r2.END
MAX_HOURS = r2.MAX_HOURS
MAX_ENTRY_ADVANCE = 0.05
WATCH_MAX = {"CAPITULATION": 72, "BASE": 168}
LEVELS = [5, 10, 20, 30, 50]
PRIMARY_TARGET = "hit20_any_30d"
BASE_MODEL_FEATURES = list(r2.MODEL_FEATURES)
DYNAMIC_FEATURES = [
    "watch_age_hours", "watch_return", "watch_low_drawdown",
    "rebound_from_watch_low", "watch_high_advance", "distance_from_watch_high",
]
MODEL_FEATURES = [f for f in BASE_MODEL_FEATURES if f != "hours_since_anchor"] + DYNAMIC_FEATURES


def period_name(year: int) -> str:
    if year <= 2022:
        return "DISCOVERY_2019_2022"
    if year <= 2024:
        return "VALIDATION_2023_2024"
    return "CONFIRMATION_2025_2026"


def entry_path_metrics_full(df: pd.DataFrame, entry_i: int, entry: float) -> dict:
    end = min(len(df), entry_i + MAX_HOURS)
    min_low = entry
    max_high = entry
    hit_idx = {p: None for p in LEVELS}
    mae_before_20 = np.nan
    first_adverse5 = None
    first_adverse10 = None

    for j in range(entry_i, end):
        lo = float(df.low.iloc[j])
        hi = float(df.high.iloc[j])
        min_low = min(min_low, lo)
        max_high = max(max_high, hi)

        if first_adverse5 is None and lo <= entry * 0.95:
            first_adverse5 = j
        if first_adverse10 is None and lo <= entry * 0.90:
            first_adverse10 = j

        for p in LEVELS:
            if hit_idx[p] is None and hi >= entry * (1 + p / 100):
                hit_idx[p] = j
                if p == 20:
                    mae_before_20 = min_low / entry - 1

    mfe = max_high / entry - 1
    mae = min_low / entry - 1
    h20 = hit_idx[20]
    before5 = int(h20 is not None and (first_adverse5 is None or h20 < first_adverse5))
    before10 = int(h20 is not None and (first_adverse10 is None or h20 < first_adverse10))
    out = {
        "mfe_30d": mfe,
        "mae_30d": mae,
        "capped_upside20": min(max(mfe, 0.0), 0.20),
        "hit20_any_30d": int(h20 is not None),
        "hit20_before_adverse5": before5,
        "hit20_before_adverse10": before10,
        "mae_before_20": mae_before_20,
    }
    for p in LEVELS:
        out[f"hit{p}_any_30d"] = int(hit_idx[p] is not None)
        out[f"hours_to_{p}"] = (hit_idx[p] - entry_i) if hit_idx[p] is not None else np.nan
    return out


def precompute_forward_extrema(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    high = pd.Series(df.high.to_numpy(dtype=float))
    low = pd.Series(df.low.to_numpy(dtype=float))
    future_high = high.iloc[::-1].rolling(MAX_HOURS, min_periods=1).max().iloc[::-1].to_numpy()
    future_low = low.iloc[::-1].rolling(MAX_HOURS, min_periods=1).min().iloc[::-1].to_numpy()
    return future_high, future_low


def entry_path_metrics_fast(entry_i: int, entry: float, forward_high: np.ndarray, forward_low: np.ndarray) -> dict:
    mfe = float(forward_high[entry_i] / entry - 1)
    mae = float(forward_low[entry_i] / entry - 1)
    out = {
        "mfe_30d": mfe,
        "mae_30d": mae,
        "capped_upside20": min(max(mfe, 0.0), 0.20),
        "hit20_any_30d": int(mfe >= 0.20),
        "hit20_before_adverse5": np.nan,
        "hit20_before_adverse10": np.nan,
        "mae_before_20": np.nan,
    }
    for p in LEVELS:
        out[f"hit{p}_any_30d"] = int(mfe >= p / 100)
        out[f"hours_to_{p}"] = np.nan
    return out


def dynamic_values(df: pd.DataFrame, decision_i: int, watch_i: int, watch_low: float, watch_high: float) -> dict:
    close = float(df.close.iloc[decision_i])
    anchor = float(df.close.iloc[watch_i])
    return {
        "watch_age_hours": int(decision_i - watch_i),
        "watch_return": close / anchor - 1,
        "watch_low_drawdown": watch_low / anchor - 1,
        "rebound_from_watch_low": close / watch_low - 1,
        "watch_high_advance": watch_high / anchor - 1,
        "distance_from_watch_high": close / watch_high - 1,
    }


def make_candidate_row(symbol: str, arch: str, split: str, df: pd.DataFrame, ff: pd.DataFrame,
                       watch_i: int, decision_i: int, watch_low: float, watch_high: float,
                       cfg: dict, episode_id: str, path_cache: tuple[np.ndarray, np.ndarray]) -> dict | None:
    if decision_i + 1 >= len(df):
        return None
    slip = float(cfg["slippage_rate"])
    entry_i = decision_i + 1
    entry = float(df.open.iloc[entry_i]) * (1 + slip)
    anchor = float(df.close.iloc[watch_i])
    advance = entry / anchor - 1
    if advance > MAX_ENTRY_ADVANCE:
        return None

    src = ff.iloc[decision_i]
    row = {
        "episode_id": episode_id,
        "symbol": symbol,
        "archetype": arch,
        "split": split,
        "year": int(pd.Timestamp(df.time.iloc[decision_i]).year),
        "watch_start": pd.Timestamp(df.time.iloc[watch_i]),
        "decision_time": pd.Timestamp(df.time.iloc[decision_i]),
        "entry_time": pd.Timestamp(df.time.iloc[entry_i]),
        "watch_anchor_price": anchor,
        "entry_price": entry,
        "entry_progress_from_watch_pct": advance,
    }
    for f in BASE_MODEL_FEATURES:
        if f == "hours_since_anchor":
            continue
        v = src.get(f, np.nan)
        row[f] = float(v) if pd.notna(v) else np.nan
    row.update(dynamic_values(df, decision_i, watch_i, watch_low, watch_high))
    row.update(entry_path_metrics_fast(entry_i, entry, path_cache[0], path_cache[1]))
    return row


def build_causal_watch_candidates(universe: pd.DataFrame, data: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
                                  arch_model, gate_params: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    episodes = []
    pmap = {r.archetype: r for _, r in gate_params.iterrows()}

    for symbol in universe.symbol:
        if symbol not in data:
            continue
        df, ff0 = data[symbol]
        ff = r2.apply_arch_to_feature_frame(ff0, arch_model)
        path_cache = precompute_forward_extrema(df)
        years = pd.to_datetime(df.time, utc=True).dt.year.to_numpy()

        for arch in ["CAPITULATION", "BASE"]:
            if arch not in pmap:
                continue
            params = pmap[arch]
            max_watch = WATCH_MAX[arch]
            i = 2160
            episode_no = 0
            while i < len(df) - 2:
                year = int(years[i])
                if year < 2019 or year > 2026 or not r2.watch_gate(ff.iloc[i], arch, params):
                    i += 1
                    continue

                episode_no += 1
                watch_i = i
                episode_id = f"{symbol}:{arch}:{pd.Timestamp(df.time.iloc[i]).isoformat()}:{episode_no}"
                deadline = min(len(df) - 2, watch_i + max_watch)
                watch_low = float(df.low.iloc[watch_i])
                watch_high = float(df.high.iloc[watch_i])
                candidate_count = 0
                first_valid_time = None
                last_valid_time = None

                for j in range(watch_i, deadline + 1):
                    watch_low = min(watch_low, float(df.low.iloc[j]))
                    watch_high = max(watch_high, float(df.high.iloc[j]))
                    split = period_name(int(years[j]))
                    row = make_candidate_row(symbol, arch, split, df, ff, watch_i, j, watch_low, watch_high, cfg, episode_id, path_cache)
                    if row is None:
                        if float(df.close.iloc[j]) / float(df.close.iloc[watch_i]) - 1 > MAX_ENTRY_ADVANCE:
                            break
                        continue
                    rows.append(row)
                    candidate_count += 1
                    if first_valid_time is None:
                        first_valid_time = row["decision_time"]
                    last_valid_time = row["decision_time"]

                episodes.append({
                    "episode_id": episode_id,
                    "symbol": symbol,
                    "archetype": arch,
                    "split": period_name(year),
                    "watch_start": pd.Timestamp(df.time.iloc[watch_i]),
                    "candidate_hours": candidate_count,
                    "first_candidate_time": first_valid_time,
                    "last_candidate_time": last_valid_time,
                })
                i = deadline + 1

    return pd.DataFrame(rows), pd.DataFrame(episodes)


def simple_trigger(row: pd.Series, arch: str, rule: str) -> bool:
    if rule == "REBOUND2_VOL11":
        return bool(row.rebound_12h_low >= 0.02 and row.volume_ratio_3h >= 1.10)
    if rule == "WATCH_REBOUND2_VOL11":
        return bool(row.rebound_from_watch_low >= 0.02 and row.volume_ratio_3h >= 1.10)
    if rule == "WATCH_REBOUND1_VOL11":
        return bool(row.rebound_from_watch_low >= 0.01 and row.volume_ratio_3h >= 1.10)
    if rule == "REBOUND2_TRADE11":
        return bool(row.rebound_from_watch_low >= 0.02 and row.trade_ratio_3h >= 1.10)
    if rule == "EMA6_RECLAIM_VOL11":
        return bool(row.close_vs_ema6 >= 0 and row.volume_ratio_3h >= 1.10)
    if rule == "BREAK3_VOL12":
        return bool(row.close_vs_prev3h_high >= 0 and row.volume_ratio_current >= 1.20)
    if rule == "EXPAND3_VOL12":
        return bool(row.ret_3h >= 0.01 and row.volume_ratio_current >= 1.20)
    if rule == "BREAK6_VOL13":
        return bool(row.close_vs_prev6h_high >= 0 and row.volume_ratio_current >= 1.30)
    if rule == "VOL_TRADE13":
        return bool(row.volume_ratio_current >= 1.30 and row.trade_ratio_current >= 1.30 and row.ret_1h > 0)
    return False


SIMPLE_RULES = {
    "CAPITULATION": ["REBOUND2_VOL11", "WATCH_REBOUND2_VOL11", "WATCH_REBOUND1_VOL11", "REBOUND2_TRADE11", "EMA6_RECLAIM_VOL11", "BREAK3_VOL12"],
    "BASE": ["EXPAND3_VOL12", "BREAK3_VOL12", "BREAK6_VOL13", "VOL_TRADE13", "EMA6_RECLAIM_VOL11"],
}


def first_alerts_for_rule(candidates: pd.DataFrame, arch: str, rule: str) -> pd.DataFrame:
    z = candidates[candidates.archetype == arch].sort_values(["episode_id", "decision_time"]).copy()
    if z.empty:
        return z
    m = z.apply(lambda r: simple_trigger(r, arch, rule), axis=1)
    return z[m].groupby("episode_id", as_index=False).first()


def metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "alerts": 0, "hit20_any_rate": np.nan, "mean_capped_upside20": np.nan,
            "median_mfe_30d": np.nan, "median_mae_30d": np.nan,
            "median_mae_before_20": np.nan, "hit20_before_adverse5_rate": np.nan,
            "hit20_before_adverse10_rate": np.nan, "median_hours_to_20": np.nan,
            "median_entry_progress_pct": np.nan,
        }
    return {
        "alerts": len(frame),
        "hit5_any_rate": float(frame.hit5_any_30d.mean()),
        "hit10_any_rate": float(frame.hit10_any_30d.mean()),
        "hit20_any_rate": float(frame.hit20_any_30d.mean()),
        "hit30_any_rate": float(frame.hit30_any_30d.mean()),
        "hit50_any_rate": float(frame.hit50_any_30d.mean()),
        "mean_capped_upside20": float(frame.capped_upside20.mean()),
        "median_mfe_30d": float(frame.mfe_30d.median()),
        "median_mae_30d": float(frame.mae_30d.median()),
        "median_mae_before_20": float(frame.loc[frame.hit20_any_30d == 1, "mae_before_20"].median()) if (frame.hit20_any_30d == 1).any() else np.nan,
        "hit20_before_adverse5_rate": float(frame.hit20_before_adverse5.mean()),
        "hit20_before_adverse10_rate": float(frame.hit20_before_adverse10.mean()),
        "median_hours_to_20": float(frame.loc[frame.hit20_any_30d == 1, "hours_to_20"].median()) if (frame.hit20_any_30d == 1).any() else np.nan,
        "median_entry_progress_pct": float(frame.entry_progress_from_watch_pct.median()),
    }


def evaluate_simple(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    rows = []
    alert_sets = {}
    for arch, rules in SIMPLE_RULES.items():
        for rule in rules:
            all_alerts = first_alerts_for_rule(candidates, arch, rule)
            alert_sets[(arch, rule)] = all_alerts
            for split, g in all_alerts.groupby("split"):
                row = {"method": "SIMPLE", "archetype": arch, "rule": rule, "split": split}
                row.update(metrics(g))
                rows.append(row)
    table = pd.DataFrame(rows)

    selected = []
    for arch in ["CAPITULATION", "BASE"]:
        v = table[(table.archetype == arch) & (table.split == "VALIDATION_2023_2024") & (table.alerts >= 25)].copy()
        if v.empty:
            continue
        v = v.sort_values(["hit20_any_rate", "mean_capped_upside20", "hit20_before_adverse10_rate"], ascending=False)
        r = v.iloc[0]
        selected.append({
            "archetype": arch,
            "rule": r.rule,
            "validation_alerts": int(r.alerts),
            "validation_hit20_any_rate": float(r.hit20_any_rate),
            "validation_mean_capped_upside20": float(r.mean_capped_upside20),
            "validation_median_mae_30d": float(r.median_mae_30d),
        })
    return table, pd.DataFrame(selected), alert_sets


@dataclass
class RFBundle:
    archetype: str
    imputer: SimpleImputer
    classifier: RandomForestClassifier
    threshold: float


def fit_rf(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, RFBundle]]:
    rows = []
    imps = []
    selected = []
    bundles = {}

    for arch in ["CAPITULATION", "BASE"]:
        z = candidates[candidates.archetype == arch].copy()
        tr = z[z.split == "DISCOVERY_2019_2022"].copy()
        va = z[z.split == "VALIDATION_2023_2024"].copy()
        co = z[z.split == "CONFIRMATION_2025_2026"].copy()
        if min(len(tr), len(va)) < 200 or tr[PRIMARY_TARGET].nunique() < 2:
            continue

        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(tr[MODEL_FEATURES])
        clf = RandomForestClassifier(
            n_estimators=500, max_depth=6, min_samples_leaf=40,
            class_weight="balanced", random_state=43, n_jobs=-1
        )
        clf.fit(Xtr, tr[PRIMARY_TARGET])

        pva = clf.predict_proba(imp.transform(va[MODEL_FEATURES]))[:, 1]
        qs = np.unique(np.quantile(pva, np.linspace(0.50, 0.995, 45)))
        threshold_candidates = []
        for th in qs:
            tmp = va.assign(_p=pva)
            hits = tmp[tmp._p >= th].sort_values(["episode_id", "decision_time"]).groupby("episode_id", as_index=False).first()
            if len(hits) < 25:
                continue
            mm = metrics(hits)
            threshold_candidates.append((mm["hit20_any_rate"], mm["mean_capped_upside20"], float(th), len(hits)))
        if not threshold_candidates:
            continue
        threshold_candidates.sort(reverse=True)
        _, _, th, _ = threshold_candidates[0]
        bundle = RFBundle(arch, imp, clf, th)
        bundles[arch] = bundle

        for split, frame in [("VALIDATION_2023_2024", va), ("CONFIRMATION_2025_2026", co)]:
            if frame.empty:
                continue
            pp = clf.predict_proba(imp.transform(frame[MODEL_FEATURES]))[:, 1]
            alerts = frame.assign(model_probability=pp)
            alerts = alerts[alerts.model_probability >= th].sort_values(["episode_id", "decision_time"]).groupby("episode_id", as_index=False).first()
            row = {"method": "RF", "archetype": arch, "split": split, "threshold": th, "candidate_hours": len(frame)}
            row.update(metrics(alerts))
            rows.append(row)

        for f, v in zip(MODEL_FEATURES, clf.feature_importances_):
            imps.append({"archetype": arch, "feature": f, "importance": float(v)})

        sv = [r for r in rows if r["archetype"] == arch and r["split"] == "VALIDATION_2023_2024"][-1]
        selected.append({
            "archetype": arch,
            "threshold": th,
            "validation_alerts": int(sv["alerts"]),
            "validation_hit20_any_rate": float(sv["hit20_any_rate"]),
            "validation_mean_capped_upside20": float(sv["mean_capped_upside20"]),
            "validation_median_mae_30d": float(sv["median_mae_30d"]),
        })

    return pd.DataFrame(rows), pd.DataFrame(imps), pd.DataFrame(selected), bundles


def selected_alerts(candidates: pd.DataFrame, selected_simple: pd.DataFrame, selected_models: pd.DataFrame,
                    bundles: dict[str, RFBundle]) -> pd.DataFrame:
    frames = []
    for _, s in selected_simple.iterrows():
        a = first_alerts_for_rule(candidates, s.archetype, s.rule).copy()
        a["method"] = "SIMPLE"
        a["trigger"] = s.rule
        a["model_probability"] = np.nan
        frames.append(a)

    for _, s in selected_models.iterrows():
        arch = s.archetype
        b = bundles.get(arch)
        if b is None:
            continue
        z = candidates[candidates.archetype == arch].copy()
        pp = b.classifier.predict_proba(b.imputer.transform(z[MODEL_FEATURES]))[:, 1]
        a = z.assign(model_probability=pp)
        a = a[a.model_probability >= b.threshold].sort_values(["episode_id", "decision_time"]).groupby("episode_id", as_index=False).first()
        a["method"] = "RF"
        a["trigger"] = f"RF>={b.threshold:.4f}"
        frames.append(a)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()



def enrich_selected_alerts(alerts: pd.DataFrame, data: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    if alerts.empty:
        return alerts
    out = alerts.copy()
    time_maps = {}
    for symbol in out.symbol.unique():
        if symbol not in data:
            continue
        df, _ = data[symbol]
        time_maps[symbol] = (df, pd.Series(df.index.to_numpy(), index=pd.to_datetime(df.time, utc=True)).to_dict())

    detailed = []
    for _, row in out.iterrows():
        symbol = row.symbol
        if symbol not in time_maps:
            detailed.append({})
            continue
        df, idx = time_maps[symbol]
        ei = idx.get(pd.Timestamp(row.entry_time))
        if ei is None:
            detailed.append({})
            continue
        detailed.append(entry_path_metrics_full(df, int(ei), float(row.entry_price)))

    for i, d in enumerate(detailed):
        for k, v in d.items():
            out.at[out.index[i], k] = v
    return out


def summary_with_frequency(alerts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if alerts.empty:
        return pd.DataFrame()
    spans = {
        "DISCOVERY_2019_2022": (pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2022-12-31 23:00", tz="UTC")),
        "VALIDATION_2023_2024": (pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2024-12-31 23:00", tz="UTC")),
        "CONFIRMATION_2025_2026": (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-08-31 23:00", tz="UTC")),
    }
    for (method, arch, trigger, split), g in alerts.groupby(["method", "archetype", "trigger", "split"]):
        start, end = spans[split]
        weeks = (end - start).total_seconds() / (7 * 24 * 3600)
        row = {"method": method, "archetype": arch, "trigger": trigger, "split": split, "alerts_per_week": len(g) / weeks}
        row.update(metrics(g))
        row["max_coin_alert_share"] = float(g.symbol.value_counts(normalize=True).iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def write_analysis(universe: pd.DataFrame, episodes: pd.DataFrame, selected_simple: pd.DataFrame,
                   selected_models: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Explosive Move Entry Discovery Round 3",
        "",
        "**Status:** RESEARCH ONLY. No Early Breakout automation or Crypto Live Strategy Specification changes.",
        "",
        "## Purpose",
        "",
        "Round 3 isolates entry quality from exit design. Watches are generated causally from past-only archetype/context gates. Once a watch starts, candidate entry conditions are evaluated every completed hour until the watch expires or an entry would be more than 5% advanced from the watch anchor.",
        "",
        "The primary model target is simply whether +20% is reached within 30 days from the next-hour entry, regardless of interim drawdown. No stop or trailing exit is used to train or select the entry model.",
        "",
        "Primary reporting therefore uses forward opportunity/path metrics: hit rates to +5/+10/+20/+30/+50, 30d MFE, 30d MAE, time to target, MAE before +20 and mean upside opportunity capped at +20%. The +20-before--5 and +20-before--10 fields are secondary diagnostics only and do not choose the model.",
        "",
        "Because 2025-2026 results were already inspected in earlier rounds, this study labels that period CONFIRMATION rather than claiming it is a pristine untouched holdout.",
        "",
        f"Universe: {len(universe)} frozen eligible Binance USDT pairs. Watch episodes: {len(episodes)}.",
        "",
        "## Validation-selected triggers",
        "",
        "| Method | Archetype | Trigger | Validation alerts | +20 any | Mean capped upside | Median 30d MAE |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, r in selected_simple.iterrows():
        lines.append(f"| SIMPLE | {r.archetype} | {r.rule} | {int(r.validation_alerts)} | {100*r.validation_hit20_any_rate:.1f}% | {100*r.validation_mean_capped_upside20:.1f}% | {100*r.validation_median_mae_30d:.1f}% |")
    for _, r in selected_models.iterrows():
        lines.append(f"| RF | {r.archetype} | p>={r.threshold:.4f} | {int(r.validation_alerts)} | {100*r.validation_hit20_any_rate:.1f}% | {100*r.validation_mean_capped_upside20:.1f}% | {100*r.validation_median_mae_30d:.1f}% |")

    lines += [
        "",
        "## End-to-end causal results",
        "",
        "| Method | Archetype | Split | Alerts/wk | +10 any | +20 any | +30 any | Mean capped +20 opportunity | Median MFE | Median MAE | +20 before -5 | +20 before -10 | Median h to +20 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if summary.empty:
        lines.append("| - | - | - | - | - | - | - | - | - | - | - | - | - |")
    else:
        for _, r in summary.sort_values(["method", "archetype", "split"]).iterrows():
            lines.append(
                f"| {r.method} | {r.archetype} | {r.split} | {r.alerts_per_week:.2f} | "
                f"{100*r.hit10_any_rate:.1f}% | {100*r.hit20_any_rate:.1f}% | {100*r.hit30_any_rate:.1f}% | "
                f"{100*r.mean_capped_upside20:.1f}% | {100*r.median_mfe_30d:.1f}% | {100*r.median_mae_30d:.1f}% | "
                f"{100*r.hit20_before_adverse5_rate:.1f}% | {100*r.hit20_before_adverse10_rate:.1f}% | "
                f"{r.median_hours_to_20:.1f} |"
            )

    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- Mean capped +20 opportunity is NOT a realised trading return. It answers: if upside could be harvested perfectly but never credited beyond +20%, how much upside did the entry expose us to on average?",
        "- MAE is reported beside upside so entries that eventually rally only after severe drawdown are visible rather than being labelled unambiguously good.",
        "- +20-before--5 and +20-before--10 are path-quality diagnostics, not exit rules.",
        "- No trailing stop, profit taking or runner logic is optimised in this round.",
        "- 2025-2026 is confirmation data, not a pristine holdout, because earlier research already exposed results from that period.",
        "- The frozen current-liquidity universe retains survivorship bias.",
        "- No result is promoted to the live strategy by this workflow.",
        "",
    ]
    (OUT / "ANALYSIS.md").write_text("\n".join(lines))


def main():
    cfg = json.loads(Path("research/config.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    universe = r2.load_frozen_top50()
    universe.to_csv(OUT / "universe.csv", index=False)
    btc = load_symbol("BTCUSDT", "1h", START, END)

    data = {}
    failures = []

    def load_one(symbol: str):
        df = load_symbol(symbol, "1h", START, END)
        if len(df) < 3000:
            return symbol, None, None, f"insufficient_history:{len(df)}"
        return symbol, df, r2.build_features(df, btc), None

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(load_one, s): s for s in universe.symbol}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                symbol, df, ff, err = fut.result()
                if err:
                    failures.append({"symbol": symbol, "kind": "load", "error": err})
                else:
                    data[symbol] = (df, ff)
                print(symbol, "rows", 0 if df is None else len(df), err or "OK", flush=True)
            except Exception as e:
                failures.append({"symbol": s, "kind": "load", "error": repr(e)})
                print("ERROR", s, repr(e), flush=True)

    anchors, recheck_fail = r2.build_anchor_cohort(universe, data)
    failures.extend(recheck_fail)
    arch_model, arch_summary = r2.fit_archetypes(anchors)
    anchors = r2.assign_archetypes(anchors, arch_model)
    gates = r2.discovery_watch_gate_params(anchors, arch_model)

    arch_summary.to_csv(OUT / "archetype_summary.csv", index=False)
    gates.to_csv(OUT / "watch_gate_params.csv", index=False)

    candidates, episodes = build_causal_watch_candidates(universe, data, arch_model, gates, cfg)
    candidates.to_csv(OUT / "watch_candidates.csv.gz", index=False, compression="gzip")
    episodes.to_csv(OUT / "watch_episodes.csv", index=False)

    simple_metrics, selected_simple, _ = evaluate_simple(candidates)
    simple_metrics.to_csv(OUT / "simple_rule_metrics.csv", index=False)
    selected_simple.to_csv(OUT / "selected_simple.csv", index=False)

    rf_metrics, importances, selected_models, bundles = fit_rf(candidates)
    rf_metrics.to_csv(OUT / "model_quality.csv", index=False)
    importances.to_csv(OUT / "model_feature_importance.csv", index=False)
    selected_models.to_csv(OUT / "selected_models.csv", index=False)

    alerts = selected_alerts(candidates, selected_simple, selected_models, bundles)
    alerts = enrich_selected_alerts(alerts, data)
    alerts.to_csv(OUT / "causal_entry_alerts.csv.gz", index=False, compression="gzip")
    summary = summary_with_frequency(alerts)
    summary.to_csv(OUT / "causal_entry_summary.csv", index=False)

    if not alerts.empty:
        alerts.assign(year=pd.to_datetime(alerts.entry_time, utc=True).dt.year).groupby(
            ["method", "archetype", "trigger", "split", "year"]
        ).agg(
            alerts=("symbol", "size"),
            hit20_any_rate=("hit20_any_30d", "mean"),
            mean_capped_upside20=("capped_upside20", "mean"),
            median_mfe_30d=("mfe_30d", "median"),
            median_mae_30d=("mae_30d", "median"),
            hit20_before_adverse5_rate=("hit20_before_adverse5", "mean"),
            hit20_before_adverse10_rate=("hit20_before_adverse10", "mean"),
        ).reset_index().to_csv(OUT / "year_summary.csv", index=False)

        alerts.groupby(["method", "archetype", "trigger", "split", "symbol"]).agg(
            alerts=("symbol", "size"),
            hit20_any_rate=("hit20_any_30d", "mean"),
            mean_capped_upside20=("capped_upside20", "mean"),
            median_mae_30d=("mae_30d", "median"),
        ).reset_index().to_csv(OUT / "coin_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(OUT / "year_summary.csv", index=False)
        pd.DataFrame().to_csv(OUT / "coin_summary.csv", index=False)

    pd.DataFrame(failures).to_csv(OUT / "failures.csv", index=False)
    write_analysis(universe, episodes, selected_simple, selected_models, summary)

    manifest = {
        "study": "Explosive Move Entry Discovery Round 3",
        "status": "RESEARCH ONLY - no live strategy or automation edits",
        "purpose": "Causal watch-to-entry discovery with entry quality separated from exit optimisation.",
        "universe": "Frozen Round 2 eligible top-50 universe; survivorship bias remains.",
        "history": {"start": START, "end": END, "interval": "1h"},
        "watch_logic": "Discovery-derived CAPITULATION/BASE gates. Evaluate entry every completed hour while watch remains alive; next-hour open plus configured slippage; reject entries >5% above watch anchor.",
        "watch_max_hours": WATCH_MAX,
        "primary_target": "+20% reached at any point within 30d, irrespective of interim drawdown.",
        "primary_metrics": ["hit5/10/20/30/50_any_30d", "mfe_30d", "mae_30d", "capped_upside20", "mae_before_20", "hours_to_targets"],
        "secondary_path_diagnostics": ["hit20_before_adverse5", "hit20_before_adverse10"],
        "exit_policy": "None for model selection. No stop, trailing stop or profit-taking logic is optimised.",
        "model": {
            "type": "RandomForestClassifier",
            "n_estimators": 500,
            "max_depth": 6,
            "min_samples_leaf": 40,
            "class_weight": "balanced",
            "fit": "2019-2022 candidate watch-hours",
            "threshold_selection": "2023-2024 validation; maximise +20-any hit rate then capped-upside opportunity, minimum 25 first-alert watch episodes",
        },
        "period_note": "2025-2026 is labelled confirmation, not untouched holdout, because earlier rounds already exposed that period.",
        "costs": {"fee_rate": cfg["fee_rate"], "slippage_rate": cfg["slippage_rate"]},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("UNIVERSE", len(universe), "LOADED", len(data), "EPISODES", len(episodes), "CANDIDATES", len(candidates), "FAILURES", len(failures))
    print("SELECTED SIMPLE\n", selected_simple.to_string(index=False) if not selected_simple.empty else "none")
    print("SELECTED MODELS\n", selected_models.to_string(index=False) if not selected_models.empty else "none")
    print("SUMMARY\n", summary.to_string(index=False) if not summary.empty else "none")


if __name__ == "__main__":
    main()
