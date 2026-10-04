from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import explosive_move_entry_discovery_round2 as r2
import explosive_move_entry_discovery_round3_final as r3
from binance_data import load_symbol

OUT = Path("research/results/explosive_move_entry_timing_round4")
START = r2.START
END = r2.END
TIMING_RULES = [
    "IMMEDIATE",
    "GREEN_2H",
    "EMA6_RECLAIM",
    "EMA12_RECLAIM",
    "BREAK3",
    "BREAK6",
    "POSTQUAL_REBOUND1",
    "POSTQUAL_REBOUND2",
    "POSTQUAL_REBOUND2_VOL11",
    "POSTQUAL_REBOUND2_BREAK3",
    "POSTQUAL_REBOUND1_EMA6",
    "RF_STRENGTHEN_005",
    "RF_STRENGTHEN_010",
]


def load_round3_thresholds() -> dict[str, float]:
    p = Path("research/results/explosive_move_entry_discovery_round3_entry_only/selected_triggers.csv")
    s = pd.read_csv(p)
    s = s[s.method == "RF"].copy()
    out = {}
    for _, r in s.iterrows():
        out[str(r.archetype)] = float(str(r.trigger).split(">=")[-1])
    return out


def timing_pass(rule: str, row: pd.Series, prob: float, threshold: float,
                postqual_low: float, close: float) -> bool:
    if rule == "IMMEDIATE":
        return True
    if rule == "GREEN_2H":
        return bool(row.ret_1h > 0 and row.ret_2h > 0)
    if rule == "EMA6_RECLAIM":
        return bool(row.close_vs_ema6 >= 0)
    if rule == "EMA12_RECLAIM":
        return bool(row.close_vs_ema12 >= 0)
    if rule == "BREAK3":
        return bool(row.close_vs_prev3h_high >= 0)
    if rule == "BREAK6":
        return bool(row.close_vs_prev6h_high >= 0)
    rebound = close / postqual_low - 1
    if rule == "POSTQUAL_REBOUND1":
        return bool(rebound >= .01)
    if rule == "POSTQUAL_REBOUND2":
        return bool(rebound >= .02)
    if rule == "POSTQUAL_REBOUND2_VOL11":
        return bool(rebound >= .02 and row.volume_ratio_3h >= 1.10)
    if rule == "POSTQUAL_REBOUND2_BREAK3":
        return bool(rebound >= .02 and row.close_vs_prev3h_high >= 0)
    if rule == "POSTQUAL_REBOUND1_EMA6":
        return bool(rebound >= .01 and row.close_vs_ema6 >= 0)
    if rule == "RF_STRENGTHEN_005":
        return bool(prob >= threshold + .05)
    if rule == "RF_STRENGTHEN_010":
        return bool(prob >= threshold + .10)
    return False


def evaluate_episode(rows: list[dict], arch: str, threshold: float,
                     model, df: pd.DataFrame, cfg: dict) -> list[dict]:
    ep = pd.DataFrame(rows).sort_values("decision_time").reset_index(drop=True)
    imp, clf = model
    probs = clf.predict_proba(imp.transform(ep[r3.MODEL_FEATURES]))[:, 1]
    q = np.flatnonzero(probs >= threshold)
    if len(q) == 0:
        return []
    qi = int(q[0])
    qrow = ep.iloc[qi]
    q_decision_i = int(qrow.entry_index) - 1
    q_price = float(df.close.iloc[q_decision_i])
    postqual_low = float(df.low.iloc[q_decision_i])
    fired = set()
    out = []

    for k in range(qi, len(ep)):
        row = ep.iloc[k]
        decision_i = int(row.entry_index) - 1
        postqual_low = min(postqual_low, float(df.low.iloc[decision_i]))
        close = float(df.close.iloc[decision_i])
        prob = float(probs[k])

        for rule in TIMING_RULES:
            if rule in fired:
                continue
            if not timing_pass(rule, row, prob, threshold, postqual_low, close):
                continue
            entry_i = decision_i + 1
            if entry_i >= len(df):
                continue
            entry = float(df.open.iloc[entry_i]) * (1 + float(cfg["slippage_rate"]))
            from_low = entry / postqual_low - 1
            if rule != "IMMEDIATE" and from_low > .05:
                continue
            d = {
                "episode_id": row.episode_id,
                "symbol": row.symbol,
                "archetype": arch,
                "split": row.split,
                "rule": rule,
                "rf_threshold": threshold,
                "rf_probability": prob,
                "qualify_time": pd.Timestamp(qrow.decision_time),
                "decision_time": pd.Timestamp(row.decision_time),
                "entry_time": pd.Timestamp(df.time.iloc[entry_i]),
                "entry_price": entry,
                "hours_after_qualification": int(decision_i - q_decision_i),
                "entry_from_postqual_low_pct": from_low,
                "entry_from_qualification_price_pct": entry / q_price - 1,
            }
            d.update(r3.full_path(df, entry_i, entry))
            out.append(d)
            fired.add(rule)
    return out


def metric_row(g: pd.DataFrame, baseline_alerts: int) -> dict:
    if g.empty:
        return {"alerts": 0}
    return {
        "alerts": len(g),
        "capture_vs_immediate": len(g) / baseline_alerts if baseline_alerts else np.nan,
        "hit10_any_rate": float(g.hit10_any_30d.mean()),
        "hit20_any_rate": float(g.hit20_any_30d.mean()),
        "hit30_any_rate": float(g.hit30_any_30d.mean()),
        "mean_capped_upside20": float(g.capped_upside20.mean()),
        "median_mfe_30d": float(g.mfe_30d.median()),
        "median_mae_30d": float(g.mae_30d.median()),
        "hit20_before_adverse5_rate": float(g.hit20_before_adverse5.mean()),
        "hit20_before_adverse10_rate": float(g.hit20_before_adverse10.mean()),
        "median_mae_before_20": float(g.loc[g.hit20_any_30d == 1, "mae_before_20"].median()) if (g.hit20_any_30d == 1).any() else np.nan,
        "median_hours_to_20": float(g.loc[g.hit20_any_30d == 1, "hours_to_20"].median()) if (g.hit20_any_30d == 1).any() else np.nan,
        "median_hours_after_qualification": float(g.hours_after_qualification.median()),
        "median_entry_from_postqual_low_pct": float(g.entry_from_postqual_low_pct.median()),
    }


def summarize(alerts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arch, split), z in alerts.groupby(["archetype", "split"]):
        baseline = int((z.rule == "IMMEDIATE").sum())
        for rule, g in z.groupby("rule"):
            r = {"archetype": arch, "split": split, "rule": rule}
            r.update(metric_row(g, baseline))
            rows.append(r)
    return pd.DataFrame(rows)


def select_rules(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arch in ["CAPITULATION", "BASE"]:
        v = summary[(summary.archetype == arch) & (summary.split == "VALIDATION_2023_2024")].copy()
        if v.empty:
            continue
        b = v[v.rule == "IMMEDIATE"]
        if b.empty:
            continue
        baseline = b.iloc[0]
        min_alerts = max(8, int(math.ceil(float(baseline.alerts) * .40)))
        floor_hit20 = max(0.0, float(baseline.hit20_any_rate) - .10)
        c = v[(v.rule != "IMMEDIATE") & (v.alerts >= min_alerts) & (v.hit20_any_rate >= floor_hit20)].copy()
        if c.empty:
            c = v[(v.rule != "IMMEDIATE") & (v.alerts >= min_alerts)].copy()
        if c.empty:
            continue
        c = c.sort_values(
            ["hit20_before_adverse5_rate", "median_mae_30d", "hit20_any_rate", "capture_vs_immediate"],
            ascending=[False, False, False, False],
        )
        r = c.iloc[0]
        rows.append({
            "archetype": arch,
            "rule": r.rule,
            "validation_alerts": int(r.alerts),
            "validation_capture_vs_immediate": float(r.capture_vs_immediate),
            "validation_hit20_any_rate": float(r.hit20_any_rate),
            "validation_hit20_before_adverse5_rate": float(r.hit20_before_adverse5_rate),
            "validation_median_mae_30d": float(r.median_mae_30d),
            "validation_median_delay_hours": float(r.median_hours_after_qualification),
            "baseline_hit20_any_rate": float(baseline.hit20_any_rate),
            "baseline_hit20_before_adverse5_rate": float(baseline.hit20_before_adverse5_rate),
            "baseline_median_mae_30d": float(baseline.median_mae_30d),
        })
    return pd.DataFrame(rows)


def selected_comparison(summary: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    chunks = []
    for _, s in selected.iterrows():
        for split in ["VALIDATION_2023_2024", "CONFIRMATION_2025_2026"]:
            q = summary[(summary.archetype == s.archetype) & (summary.split == split) &
                        (summary.rule.isin(["IMMEDIATE", s.rule]))].copy()
            chunks.append(q)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def write_analysis(selected: pd.DataFrame, comparison: pd.DataFrame) -> None:
    lines = [
        "# Explosive Move Entry Timing Round 4",
        "",
        "**Status:** RESEARCH ONLY. No live strategy or automation changes.",
        "",
        "Round 4 keeps the Round 3 Random Forest qualification fixed (CAPITULATION and BASE) and tests only when to enter after the RF first qualifies a causal watch. No exit, trailing-stop or profit-taking rule is used for selection.",
        "",
        "Timing rules are evaluated hourly after RF qualification. Delayed entries must be within 5% of the lowest price observed since qualification. Selection uses 2023-2024 validation and prioritises +20-before--5 path quality while requiring +20-any to stay within 10 percentage points of the immediate-entry baseline where possible. The -5% measure is used only as an ordering/path-quality metric, not as an exit assumption.",
        "",
        "2025-2026 remains previously exposed confirmation data, not a pristine holdout.",
        "",
        "## Selected timing rules",
        "",
        "| Archetype | Rule | Validation alerts | Capture vs immediate | +20 any | +20 before -5 | Median MAE | Median delay |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in selected.iterrows():
        lines.append(f"| {r.archetype} | {r.rule} | {int(r.validation_alerts)} | {100*r.validation_capture_vs_immediate:.1f}% | {100*r.validation_hit20_any_rate:.1f}% | {100*r.validation_hit20_before_adverse5_rate:.1f}% | {100*r.validation_median_mae_30d:.1f}% | {r.validation_median_delay_hours:.1f}h |")

    lines += [
        "",
        "## Immediate vs selected timing",
        "",
        "| Archetype | Split | Rule | Alerts | +20 any | +20 before -5 | +20 before -10 | Median MAE | Median h to +20 | Median delay |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in comparison.sort_values(["archetype", "split", "rule"]).iterrows():
        lines.append(f"| {r.archetype} | {r.split} | {r.rule} | {int(r.alerts)} | {100*r.hit20_any_rate:.1f}% | {100*r.hit20_before_adverse5_rate:.1f}% | {100*r.hit20_before_adverse10_rate:.1f}% | {100*r.median_mae_30d:.1f}% | {r.median_hours_to_20:.1f} | {r.median_hours_after_qualification:.1f}h |")

    lines += [
        "",
        "## Guardrails",
        "",
        "- Round 3 RF models and qualification thresholds are held fixed; this round changes timing only.",
        "- +20-before--5 is a path-ordering metric, not a simulated stop-loss return.",
        "- MFE/MAE and +20-any remain visible so a timing rule cannot look good merely by filtering out difficult paths.",
        "- 2025-2026 has been seen in earlier research and is confirmation only.",
        "- No result is promoted to the live strategy automatically.",
        "",
    ]
    (OUT / "ANALYSIS.md").write_text("\n".join(lines))


def main():
    cfg = json.loads(Path("research/config.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    thresholds = load_round3_thresholds()
    pd.DataFrame([{"archetype": k, "rf_threshold": v} for k, v in thresholds.items()]).to_csv(OUT / "fixed_rf_thresholds.csv", index=False)

    universe = r2.load_frozen_top50()
    universe.to_csv(OUT / "universe.csv", index=False)
    btc = load_symbol("BTCUSDT", "1h", START, END)

    data = {}
    failures = []
    def load_one(symbol):
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
                print("LOAD", symbol, 0 if df is None else len(df), err or "OK", flush=True)
            except Exception as e:
                failures.append({"symbol": s, "kind": "load", "error": repr(e)})

    anchors, recheck = r2.build_anchor_cohort(universe, data)
    failures.extend(recheck)
    arch_model, arch_summary = r2.fit_archetypes(anchors)
    anchors = r2.assign_archetypes(anchors, arch_model)
    gates = r2.discovery_watch_gate_params(anchors, arch_model)
    arch_summary.to_csv(OUT / "archetype_summary.csv", index=False)
    gates.to_csv(OUT / "watch_gate_params.csv", index=False)

    train = r3.sample_training_rows(universe, data, arch_model, gates, cfg)
    models, imps = r3.fit_models(train)
    imps.to_csv(OUT / "model_feature_importance.csv", index=False)

    pmap = {r.archetype: r for _, r in gates.iterrows()}
    all_alerts = []
    for split in ["VALIDATION_2023_2024", "CONFIRMATION_2025_2026"]:
        for symbol in universe.symbol:
            if symbol not in data:
                continue
            df, ff0 = data[symbol]
            ff = r2.apply_arch_to_feature_frame(ff0, arch_model)
            fh, fl = r3.precompute_forward_extrema(df)
            for arch in ["CAPITULATION", "BASE"]:
                if arch not in thresholds or arch not in models:
                    continue
                for rows in r3.iter_episode_rows(symbol, arch, split, df, ff, pmap[arch], cfg, fh, fl):
                    all_alerts.extend(evaluate_episode(rows, arch, thresholds[arch], models[arch], df, cfg))
            print("EVAL", split, symbol, flush=True)

    alerts = pd.DataFrame(all_alerts)
    alerts.to_csv(OUT / "timing_alerts.csv.gz", index=False, compression="gzip")
    summary = summarize(alerts)
    summary.to_csv(OUT / "timing_rule_summary.csv", index=False)
    selected = select_rules(summary)
    selected.to_csv(OUT / "selected_timing_rules.csv", index=False)
    comparison = selected_comparison(summary, selected)
    comparison.to_csv(OUT / "selected_vs_immediate.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT / "failures.csv", index=False)
    write_analysis(selected, comparison)

    manifest = {
        "study": "Explosive Move Entry Timing Round 4",
        "status": "RESEARCH ONLY - no live strategy or automation edits",
        "fixed_round3_rf_thresholds": thresholds,
        "timing_rules": TIMING_RULES,
        "entry_constraint": "Delayed entry must be <=5% above the lowest low observed since RF qualification.",
        "selection": "2023-2024 validation; require >=40% of immediate alert count (minimum 8) and preferably +20-any within 10pp of immediate; maximise +20-before--5, then less-negative MAE, +20-any, capture.",
        "exit_policy": "No exit optimisation. +20-before--5 and +20-before--10 are path-ordering diagnostics only.",
        "confirmation_note": "2025-2026 is previously exposed confirmation data, not pristine holdout.",
        "costs": {"fee_rate": cfg["fee_rate"], "slippage_rate": cfg["slippage_rate"]},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("SELECTED\n", selected.to_string(index=False) if not selected.empty else "none", flush=True)
    print("COMPARISON\n", comparison.to_string(index=False) if not comparison.empty else "none", flush=True)


if __name__ == "__main__":
    main()
