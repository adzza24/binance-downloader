from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import daily_return_harvest_round2_event_discovery as drh
import daily_return_harvest_round2_parallel as par

STAGED = drh.OUT / "staged_merge"
CANONICAL = drh.OUT / "dataset_v1"
MICRO_SYMBOLS = 5


def read_parts(root: Path, name: str) -> pd.DataFrame:
    parts = []
    for p in sorted(root.glob(f"*/{name}")):
        if not p.exists() or p.stat().st_size == 0:
            continue
        try:
            z = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        except pd.errors.EmptyDataError:
            continue
        if len(z):
            parts.append(z)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def context(shard_root: Path, universe_dir: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    panel_raw = read_parts(shard_root, "panel.parquet")
    events = read_parts(shard_root, "events.parquet")
    candidates = read_parts(shard_root, "event_candidates.parquet")
    local = read_parts(shard_root, "local_samples.parquet")
    failures = read_parts(shard_root, "failures.csv")
    if panel_raw.empty or events.empty or local.empty:
        raise RuntimeError("Missing completed shard inputs")

    panel = drh.add_cross_section_context(panel_raw)
    del panel_raw
    gc.collect()
    same_time = drh.choose_same_time_controls(events, panel)
    sample_index = (
        pd.concat([local, same_time], ignore_index=True)
        .drop_duplicates("sample_id")
        .sort_values(["decision_time", "sample_type", "symbol"])
        .reset_index(drop=True)
    )

    panel.to_parquet(out / "panel_context.parquet", index=False, compression="zstd")
    events.to_parquet(out / "events.parquet", index=False, compression="zstd")
    candidates.to_parquet(out / "event_candidates.parquet", index=False, compression="zstd")
    sample_index.to_parquet(out / "sample_index.parquet", index=False, compression="zstd")
    if failures.empty:
        pd.DataFrame(columns=["symbol", "error"]).to_csv(out / "failures.csv", index=False)
    else:
        failures.to_csv(out / "failures.csv", index=False)

    for name in ["universe.csv", "candidate_pool_liquidity.csv", "prepare_manifest.json"]:
        src = universe_dir / name
        if src.exists():
            shutil.copy2(src, out / name)

    (out / "context_manifest.json").write_text(json.dumps({
        "dataset_version": par.DATASET_VERSION,
        "schema_hash": par.schema_hash(),
        "panel_rows": len(panel),
        "events": len(events),
        "samples_indexed": len(sample_index),
        "micro_partition_symbols": MICRO_SYMBOLS,
    }, indent=2))
    print("CONTEXT", len(panel), "panel rows", len(events), "events", len(sample_index), "sample refs", flush=True)


def build_shard(shard_dir: Path, context_dir: Path, out: Path) -> None:
    done = json.loads((shard_dir / "DONE.json").read_text())
    if done.get("schema_hash") != par.schema_hash():
        raise RuntimeError("Shard schema hash mismatch")
    symbols = list(done["symbols"])
    out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(context_dir / "panel_context.parquet")
    sample_index = pd.read_parquet(context_dir / "sample_index.parquet")
    sample_index = sample_index[sample_index.symbol.isin(symbols)].copy()
    feature_dest = drh.TMP / "features"
    feature_dest.mkdir(parents=True, exist_ok=True)
    for src in (shard_dir / "features").glob("*.pkl.gz"):
        shutil.copy2(src, feature_dest / src.name)

    built_samples = 0
    built_history = 0
    written = []
    for n in range(0, len(symbols), MICRO_SYMBOLS):
        chunk = symbols[n:n + MICRO_SYMBOLS]
        refs = sample_index[sample_index.symbol.isin(chunk)].copy()
        if refs.empty:
            continue
        samples, history = drh.build_sample_outputs(refs, panel)
        if samples.empty:
            continue
        part = n // MICRO_SYMBOLS
        sp = out / f"samples_part_{part:02d}.parquet"
        hp = out / f"history_part_{part:02d}.parquet"
        samples.to_parquet(sp, index=False, compression="zstd")
        history.to_parquet(hp, index=False, compression="zstd")
        built_samples += len(samples)
        built_history += len(history)
        written.append({"part": part, "symbols": chunk, "samples": len(samples), "history_rows": len(history)})
        print("PART", part, chunk, len(samples), "samples", len(history), "history", flush=True)
        del samples, history, refs
        gc.collect()

    if not written:
        raise RuntimeError("No canonical parts built")
    (out / "DONE.json").write_text(json.dumps({
        "dataset_version": par.DATASET_VERSION,
        "schema_hash": par.schema_hash(),
        "source_shard": done.get("shard_index"),
        "symbols": symbols,
        "samples": built_samples,
        "history_rows": built_history,
        "parts": written,
    }, indent=2))


def load_canonical_samples(root: Path) -> pd.DataFrame:
    parts = []
    for p in sorted(root.glob("*/samples_part_*.parquet")):
        z = pd.read_parquet(p)
        if len(z):
            parts.append(z)
    if not parts:
        raise RuntimeError("No canonical sample partitions found")
    return pd.concat(parts, ignore_index=True)


def persist_partitions(root: Path) -> None:
    if CANONICAL.exists():
        shutil.rmtree(CANONICAL)
    (CANONICAL / "samples").mkdir(parents=True, exist_ok=True)
    (CANONICAL / "history").mkdir(parents=True, exist_ok=True)
    si = hi = 0
    for p in sorted(root.glob("*/samples_part_*.parquet")):
        shutil.copy2(p, CANONICAL / "samples" / f"part_{si:02d}.parquet")
        si += 1
    for p in sorted(root.glob("*/history_part_*.parquet")):
        shutil.copy2(p, CANONICAL / "history" / f"part_{hi:02d}.parquet")
        hi += 1


def analyse(canonical_root: Path, context_dir: Path) -> None:
    drh.OUT.mkdir(parents=True, exist_ok=True)
    samples = load_canonical_samples(canonical_root)
    events = pd.read_parquet(context_dir / "events.parquet")
    event_candidates = pd.read_parquet(context_dir / "event_candidates.parquet")
    sample_index = pd.read_parquet(context_dir / "sample_index.parquet")
    failures_path = context_dir / "failures.csv"
    try:
        failures = pd.read_csv(failures_path) if failures_path.exists() and failures_path.stat().st_size else pd.DataFrame(columns=["symbol", "error"])
    except pd.errors.EmptyDataError:
        failures = pd.DataFrame(columns=["symbol", "error"])
    universe = pd.read_csv(context_dir / "universe.csv")

    defs, conds = drh.make_condition_library(samples)
    uni, intersections, baselines, top_conditions = drh.mine_intersections(samples, defs, conds)
    fcmp = drh.feature_comparison(samples)
    condition_matrix = samples[["sample_id", "source_event_id", "symbol", "decision_time", "sample_type", "split"]].copy()
    for name in top_conditions[:60]:
        condition_matrix[name] = drh.apply_condition(samples, conds[name]).fillna(False).astype(np.int8)
    winners = samples[samples.sample_type == "WINNER_EVENT"].copy()
    winner_cond = condition_matrix[condition_matrix.sample_type == "WINNER_EVENT"].drop(
        columns=["source_event_id", "symbol", "decision_time", "sample_type", "split"]
    )
    enriched = (
        events.merge(winners, left_on="event_id", right_on="source_event_id", how="left", suffixes=("", "_sample"))
        .merge(winner_cond, on="sample_id", how="left")
    )

    events.to_csv(drh.OUT / "events.csv", index=False)
    enriched.to_csv(drh.OUT / "events_enriched.csv.gz", index=False, compression="gzip")
    event_candidates.to_csv(drh.OUT / "event_candidates.csv.gz", index=False, compression="gzip")
    sample_index.to_csv(drh.OUT / "sample_index.csv.gz", index=False, compression="gzip")
    condition_matrix.to_csv(drh.OUT / "condition_matrix.csv.gz", index=False, compression="gzip")
    defs.to_csv(drh.OUT / "condition_dictionary.csv", index=False)
    uni.to_csv(drh.OUT / "univariate_conditions.csv", index=False)
    intersections.to_csv(drh.OUT / "intersections.csv", index=False)
    fcmp.to_csv(drh.OUT / "feature_comparison.csv", index=False)
    failures.to_csv(drh.OUT / "failures.csv", index=False)

    baseline_rows = []
    base = samples[samples.sample_type == "BASELINE_RANDOM"]
    for split_name, g in base.groupby("split"):
        baseline_rows.append({
            "split": split_name,
            "samples": len(g),
            "symbols": g.symbol.nunique(),
            "w3_2_24h_rate": g.w3_2_24h.mean(),
            "w5_2_48h_rate": g.w5_2_48h.mean(),
            "w5_3_48h_rate": g.w5_3_48h.mean(),
            "w7_3_72h_rate": g.w7_3_72h.mean(),
            "w10_4_72h_rate": g.w10_4_72h.mean(),
        })
    pd.DataFrame(baseline_rows).to_csv(drh.OUT / "baseline_rates.csv", index=False)

    persist_partitions(canonical_root)
    events.to_parquet(CANONICAL / "events.parquet", index=False, compression="zstd")
    event_candidates.to_parquet(CANONICAL / "event_candidates.parquet", index=False, compression="zstd")
    sample_index.to_parquet(CANONICAL / "sample_index.parquet", index=False, compression="zstd")
    condition_matrix.to_parquet(CANONICAL / "condition_matrix.parquet", index=False, compression="zstd")
    universe.to_parquet(CANONICAL / "universe.parquet", index=False, compression="zstd")

    history_rows = 0
    stage_manifests = []
    for p in sorted(canonical_root.glob("*/DONE.json")):
        m = json.loads(p.read_text())
        stage_manifests.append(m)
        history_rows += int(m.get("history_rows", 0))
    manifest = {
        "study": "Daily Return Harvest Round 2 - event-first historical winner discovery",
        "status": "RESEARCH ONLY - no live strategy, portfolio or automation changes",
        "dataset_version": par.DATASET_VERSION,
        "schema_hash": par.schema_hash(),
        "pipeline": "staged low-memory merge: global context -> four canonical shard builders -> final analysis",
        "discovery_period": ["2021-11-01", "2024-04-30"],
        "temporal_validation_period": ["2024-05-01", "2025-12-31"],
        "confirmation_period": ["2026-01-01", "2026-09-27"],
        "costs": {"fee_rate_each_side": drh.FEE_RATE, "slippage_rate_each_side": drh.SLIPPAGE_RATE},
        "feature_history_offsets_hours": drh.SNAPSHOT_OFFSETS,
        "wide_trajectory_deltas_hours": drh.DELTA_HOURS,
        "canonical_partition_symbols": MICRO_SYMBOLS,
        "canonical_shards": len(stage_manifests),
        "samples": len(samples),
        "history_rows": history_rows,
        "events": len(events),
        "cache_policy": "Expensive per-symbol feature/path work is reused from completed four-shard artifacts; canonical wide/history partitions are persisted for future analysis without candle reprocessing.",
    }
    (drh.OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (CANONICAL / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    drh.write_analysis(events, samples, intersections, baselines, universe, failures.to_dict("records"))
    print("ANALYSED", len(events), "events", len(samples), "samples", history_rows, "history rows", flush=True)
    if len(intersections):
        print("TOP INTERSECTIONS\n", intersections.head(25).to_string(index=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("context")
    c.add_argument("--shard-root", type=Path, required=True)
    c.add_argument("--universe-dir", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("build-shard")
    b.add_argument("--shard-dir", type=Path, required=True)
    b.add_argument("--context-dir", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("analyse")
    a.add_argument("--canonical-root", type=Path, required=True)
    a.add_argument("--context-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.mode == "context":
        context(args.shard_root, args.universe_dir, args.out)
    elif args.mode == "build-shard":
        build_shard(args.shard_dir, args.context_dir, args.out)
    else:
        analyse(args.canonical_root, args.context_dir)


if __name__ == "__main__":
    main()
