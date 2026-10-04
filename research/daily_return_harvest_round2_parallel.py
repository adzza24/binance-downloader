from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import daily_return_harvest_round2_event_discovery as drh

DATASET_VERSION = "drh2-event-v1-20261004"
PARALLEL_OUT = drh.OUT / "parallel"
CANONICAL = drh.OUT / "dataset_v1"


def schema_hash() -> str:
    spec = {
        "dataset_version": DATASET_VERSION,
        "data_start": drh.DATA_START,
        "data_end": drh.DATA_END,
        "discovery_start": str(drh.DISCOVERY_START),
        "discovery_end": str(drh.DISCOVERY_END),
        "validation_end": str(drh.VALIDATION_END),
        "confirmation_end": str(drh.CONFIRMATION_END),
        "universe_size": drh.UNIVERSE_SIZE,
        "min_quote_vol_24h": drh.MIN_QUOTE_VOL_24H,
        "event_gap_hours": drh.EVENT_GAP_HOURS,
        "winner_buffer_hours": drh.WINNER_BUFFER_HOURS,
        "fee_rate": drh.FEE_RATE,
        "slippage_rate": drh.SLIPPAGE_RATE,
        "max_forward_hours": drh.MAX_FORWARD_HOURS,
        "snapshot_offsets": drh.SNAPSHOT_OFFSETS,
        "delta_hours": drh.DELTA_HOURS,
        "base_features": drh.BASE_FEATURES,
        "context_features": drh.CONTEXT_FEATURES,
        "path_cols": drh.PATH_COLS,
    }
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:20]


def prepare(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pool = drh.load_candidate_pool()
    if pool.empty:
        raise RuntimeError("Candidate universe empty")
    liq = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(drh.scan_liquidity, r): r.symbol for _, r in pool.iterrows()}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                row = fut.result()
                liq.append(row)
                print("LIQ", sym, row["active_days"], row["median_daily_quote_volume"], flush=True)
            except Exception as e:
                liq.append({
                    "symbol": sym,
                    "base_asset": sym[:-4],
                    "account_excluded": sym in drh.ACCOUNT_EXCLUSIONS,
                    "hours": 0,
                    "active_days": 0,
                    "median_daily_quote_volume": np.nan,
                    "median_trailing_24h_quote_volume": np.nan,
                    "error": repr(e),
                })
                print("LIQ ERROR", sym, repr(e), flush=True)
    liq = pd.DataFrame(liq).sort_values("median_daily_quote_volume", ascending=False, na_position="last")
    universe = liq[(liq.active_days >= drh.MIN_DISCOVERY_DAYS) & liq.median_daily_quote_volume.notna()].head(drh.UNIVERSE_SIZE).copy()
    universe["historical_liquidity_rank"] = np.arange(1, len(universe) + 1)
    if len(universe) < 50:
        raise RuntimeError(f"Only {len(universe)} symbols have sufficient discovery history")
    liq.to_csv(out / "candidate_pool_liquidity.csv", index=False)
    universe.to_csv(out / "universe.csv", index=False)
    (out / "prepare_manifest.json").write_text(json.dumps({
        "dataset_version": DATASET_VERSION,
        "schema_hash": schema_hash(),
        "symbols": len(universe),
    }, indent=2))


def process_symbol_fast(meta: pd.Series, btc: pd.DataFrame, feature_dir: Path):
    symbol = meta.symbol
    df = btc.copy() if symbol == "BTCUSDT" else drh.load_symbol(symbol, drh.INTERVAL, drh.DATA_START, drh.DATA_END)
    if len(df) < drh.MIN_DISCOVERY_DAYS * 24:
        return symbol, None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"insufficient_history:{len(df)}"
    ff = drh.build_features(df, btc).reset_index(drop=True)
    p = drh.classify_paths(ff).reset_index(drop=True)
    ff = pd.concat([ff, p], axis=1).copy()
    ff["eligible"] = drh.eligible_mask(ff)
    ff["account_excluded"] = bool(meta.account_excluded)
    ff["historical_liquidity_rank"] = int(meta.historical_liquidity_rank)
    events, candidates, blocked = drh.detect_events(symbol, ff)
    samples = drh.make_local_samples(symbol, ff, events, blocked)
    keep = list(dict.fromkeys([
        "time", "open", "high", "low", "close", "eligible", "account_excluded", "historical_liquidity_rank"
    ] + drh.BASE_FEATURES + drh.PATH_COLS))
    feature_dir.mkdir(parents=True, exist_ok=True)
    ff[keep].to_pickle(feature_dir / f"{symbol}.pkl.gz", compression="gzip")
    panel_cols = [
        "time", "eligible", "ret_1h", "ret_3h", "ret_6h", "ret_24h", "rs_6h", "rs_24h",
        "volume_ratio_3h", "trade_ratio_3h", "quote_volume_24h", "w3_2_24h",
    ]
    panel = ff[panel_cols].copy()
    panel["symbol"] = symbol
    return symbol, panel, events, candidates, samples, None


def shard(universe_path: Path, shard_index: int, shard_count: int, out: Path) -> None:
    done = out / "DONE.json"
    if done.exists():
        existing = json.loads(done.read_text())
        if existing.get("schema_hash") == schema_hash():
            print("CHECKPOINT HIT", out, flush=True)
            return
    out.mkdir(parents=True, exist_ok=True)
    feature_dir = out / "features"
    universe = pd.read_csv(universe_path)
    part = universe.iloc[shard_index::shard_count].copy()
    if part.empty:
        raise RuntimeError("Empty shard")
    btc = drh.load_symbol("BTCUSDT", drh.INTERVAL, drh.DATA_START, drh.DATA_END)
    if len(btc) < 1000:
        raise RuntimeError("BTC history unavailable")
    panels, events_parts, cand_parts, sample_parts, failures = [], [], [], [], []
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(process_symbol_fast, r, btc, feature_dir): r.symbol for _, r in part.iterrows()}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                symbol, panel, ev, cand, samp, err = fut.result()
                if err:
                    failures.append({"symbol": symbol, "error": err})
                if panel is not None and len(panel):
                    panels.append(panel)
                if len(ev):
                    events_parts.append(ev)
                if len(cand):
                    cand_parts.append(cand)
                if len(samp):
                    sample_parts.append(samp)
                print("PROCESS", symbol, "events", len(ev), "samples", len(samp), flush=True)
            except Exception as e:
                failures.append({"symbol": sym, "error": repr(e)})
                print("PROCESS ERROR", sym, repr(e), flush=True)
    if not panels:
        raise RuntimeError("No panels produced")
    pd.concat(panels, ignore_index=True).to_parquet(out / "panel.parquet", index=False, compression="zstd")
    (pd.concat(events_parts, ignore_index=True) if events_parts else pd.DataFrame()).to_parquet(out / "events.parquet", index=False, compression="zstd")
    (pd.concat(cand_parts, ignore_index=True) if cand_parts else pd.DataFrame()).to_parquet(out / "event_candidates.parquet", index=False, compression="zstd")
    (pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame()).to_parquet(out / "local_samples.parquet", index=False, compression="zstd")
    pd.DataFrame(failures).to_csv(out / "failures.csv", index=False)
    done.write_text(json.dumps({
        "dataset_version": DATASET_VERSION,
        "schema_hash": schema_hash(),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "symbols": part.symbol.tolist(),
    }, indent=2))


def read_parts(shard_root: Path, name: str) -> pd.DataFrame:
    parts = []
    for p in sorted(shard_root.glob(f"*/{name}")):
        if p.suffix == ".parquet":
            z = pd.read_parquet(p)
        else:
            z = pd.read_csv(p)
        if len(z):
            parts.append(z)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def persist_partitions(samples: pd.DataFrame, history: pd.DataFrame) -> None:
    if CANONICAL.exists():
        shutil.rmtree(CANONICAL)
    (CANONICAL / "samples").mkdir(parents=True, exist_ok=True)
    (CANONICAL / "history").mkdir(parents=True, exist_ok=True)
    symbols = sorted(samples.symbol.dropna().unique())
    for n in range(0, len(symbols), 5):
        chunk = symbols[n:n + 5]
        samples[samples.symbol.isin(chunk)].to_parquet(CANONICAL / "samples" / f"part_{n//5:02d}.parquet", index=False, compression="zstd")
        history[history.symbol.isin(chunk)].to_parquet(CANONICAL / "history" / f"part_{n//5:02d}.parquet", index=False, compression="zstd")


def merge(shard_root: Path, universe_path: Path) -> None:
    drh.OUT.mkdir(parents=True, exist_ok=True)
    (drh.TMP / "features").mkdir(parents=True, exist_ok=True)
    shard_dirs = [p for p in shard_root.iterdir() if p.is_dir() and (p / "DONE.json").exists()]
    if not shard_dirs:
        raise RuntimeError("No completed shard checkpoints found")
    for sd in shard_dirs:
        meta = json.loads((sd / "DONE.json").read_text())
        if meta.get("schema_hash") != schema_hash():
            raise RuntimeError(f"Schema mismatch in {sd}")
        for src in (sd / "features").glob("*.pkl.gz"):
            shutil.copy2(src, drh.TMP / "features" / src.name)
    panel = drh.add_cross_section_context(read_parts(shard_root, "panel.parquet"))
    events = read_parts(shard_root, "events.parquet")
    event_candidates = read_parts(shard_root, "event_candidates.parquet")
    local = read_parts(shard_root, "local_samples.parquet")
    failures = read_parts(shard_root, "failures.csv")
    if events.empty:
        raise RuntimeError("No winning events discovered")
    same_time = drh.choose_same_time_controls(events, panel)
    sample_index = pd.concat([local, same_time], ignore_index=True).drop_duplicates("sample_id").sort_values(["decision_time", "sample_type", "symbol"]).reset_index(drop=True)
    samples, history = drh.build_sample_outputs(sample_index, panel)
    if samples.empty:
        raise RuntimeError("No sample features produced")
    defs, conds = drh.make_condition_library(samples)
    uni, intersections, baselines, top_conditions = drh.mine_intersections(samples, defs, conds)
    fcmp = drh.feature_comparison(samples)
    condition_matrix = samples[["sample_id", "source_event_id", "symbol", "decision_time", "sample_type", "split"]].copy()
    for name in top_conditions[:60]:
        condition_matrix[name] = drh.apply_condition(samples, conds[name]).fillna(False).astype(np.int8)
    winners = samples[samples.sample_type == "WINNER_EVENT"].copy()
    winner_cond = condition_matrix[condition_matrix.sample_type == "WINNER_EVENT"].drop(columns=["source_event_id", "symbol", "decision_time", "sample_type", "split"])
    enriched = events.merge(winners, left_on="event_id", right_on="source_event_id", how="left", suffixes=("", "_sample")).merge(winner_cond, on="sample_id", how="left")

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

    persist_partitions(samples, history)
    events.to_parquet(CANONICAL / "events.parquet", index=False, compression="zstd")
    event_candidates.to_parquet(CANONICAL / "event_candidates.parquet", index=False, compression="zstd")
    sample_index.to_parquet(CANONICAL / "sample_index.parquet", index=False, compression="zstd")
    condition_matrix.to_parquet(CANONICAL / "condition_matrix.parquet", index=False, compression="zstd")
    universe = pd.read_csv(universe_path)
    universe.to_parquet(CANONICAL / "universe.parquet", index=False, compression="zstd")
    manifest = {
        "study": "Daily Return Harvest Round 2 - event-first historical winner discovery",
        "status": "RESEARCH ONLY - no live strategy, portfolio or automation changes",
        "dataset_version": DATASET_VERSION,
        "schema_hash": schema_hash(),
        "cache_policy": "Raw Binance candles and per-symbol engineered frames use Actions cache. Canonical sample/event/history datasets are partitioned in repo so analysis can be rerun without candle processing.",
        "discovery_period": ["2021-11-01", "2024-04-30"],
        "temporal_validation_period": ["2024-05-01", "2025-12-31"],
        "confirmation_period": ["2026-01-01", "2026-09-27"],
        "costs": {"fee_rate_each_side": drh.FEE_RATE, "slippage_rate_each_side": drh.SLIPPAGE_RATE},
        "feature_history_offsets_hours": drh.SNAPSHOT_OFFSETS,
        "wide_trajectory_deltas_hours": drh.DELTA_HOURS,
        "canonical_partition_symbols": 5,
        "shards_completed": len(shard_dirs),
        "samples": len(samples),
        "history_rows": len(history),
        "events": len(events),
    }
    (drh.OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (CANONICAL / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    drh.write_analysis(events, samples, intersections, baselines, universe, failures.to_dict("records"))
    print("MERGED", len(events), "events", len(samples), "samples", len(history), "history rows", flush=True)
    if len(intersections):
        print("TOP INTERSECTIONS\n", intersections.head(25).to_string(index=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", type=Path, default=PARALLEL_OUT)
    s = sub.add_parser("shard")
    s.add_argument("--universe", type=Path, required=True)
    s.add_argument("--shard-index", type=int, required=True)
    s.add_argument("--shard-count", type=int, default=4)
    s.add_argument("--out", type=Path, required=True)
    m = sub.add_parser("merge")
    m.add_argument("--shard-root", type=Path, required=True)
    m.add_argument("--universe", type=Path, required=True)
    args = ap.parse_args()
    if args.mode == "prepare":
        prepare(args.out)
    elif args.mode == "shard":
        shard(args.universe, args.shard_index, args.shard_count, args.out)
    else:
        merge(args.shard_root, args.universe)


if __name__ == "__main__":
    main()
