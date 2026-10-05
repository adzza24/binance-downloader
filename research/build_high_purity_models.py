from __future__ import annotations

# Rebuilds the frozen research-only A/B/C classifier artifacts.
# IMPORTANT: preserve the original stage-3 shard/partition row order used during discovery.

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "research/results/daily_return_harvest/round2_event_discovery/dataset_v1/samples"
ORIGINAL_SHARDS = ROOT / "research/cache/high_purity_model_source"
SPEC = ROOT / "research/high_purity_signal_features.json"
OUT = ROOT / "research/results/daily_return_harvest/high_purity_runtime/models"

DISC = "DISCOVERY_2021-11_2024-04"
VAL = "VALIDATION_2024-05_2025-12"
CONF = "CONFIRMATION_2026"
TRAIN_CUTOFF = pd.Timestamp("2023-11-01", tz="UTC")

CONSTRUCTED_TYPES = [
    "WINNER_EVENT",
    "HARD_NEGATIVE",
    "SAME_COIN_RANDOM",
    "SAME_TIME_CONTROL",
]

EXPECTED = {
    DISC: {"signals": 975, "winners": 938},
    VAL: {"signals": 645, "winners": 587},
    CONF: {"signals": 254, "winners": 231},
}


def source_parts() -> tuple[list[Path], str]:
    # Exact discovery analysis loaded shard folders lexicographically (0,1,2,3),
    # then samples_part_*.parquet within each folder. Preserve that row order because
    # HistGradientBoostingClassifier's internal validation split is row-order-sensitive.
    shard_parts = []
    for shard in range(4):
        d = ORIGINAL_SHARDS / f"drh2-canonical-{shard}"
        shard_parts.extend(sorted(d.glob("samples_part_*.parquet")))
    if shard_parts:
        return shard_parts, "stage3-canonical-shards-0-3"

    parts = sorted(CANONICAL.glob("part_*.parquet"))
    if parts:
        return parts, "consolidated-dataset-fallback"
    raise RuntimeError("No Round 2 sample partitions found")


def load_samples(a_features: list[str], bc_features: list[str]) -> tuple[pd.DataFrame, str]:
    cols = list(dict.fromkeys(
        ["sample_type", "split", "decision_time"] + a_features + bc_features
    ))
    parts, source = source_parts()
    frames = [pd.read_parquet(path, columns=cols) for path in parts]
    df = pd.concat(frames, ignore_index=True)

    if not pd.api.types.is_datetime64_any_dtype(df["decision_time"]):
        numeric = pd.to_numeric(df["decision_time"], errors="coerce")
        if numeric.notna().mean() > 0.99:
            df["decision_time"] = pd.to_datetime(numeric, unit="ms", utc=True)
        else:
            df["decision_time"] = pd.to_datetime(df["decision_time"], utc=True)
    elif getattr(df["decision_time"].dt, "tz", None) is None:
        df["decision_time"] = df["decision_time"].dt.tz_localize("UTC")
    else:
        df["decision_time"] = df["decision_time"].dt.tz_convert("UTC")

    print(f"Loaded {len(df)} rows from {len(parts)} partitions using source={source}")
    return df, source


def fit_models(df: pd.DataFrame, spec: dict):
    a_features = spec["A_FEATURES"]
    bc_features = spec["BC_FEATURES"]
    th = spec["thresholds"]

    base = df[df["sample_type"].isin(CONSTRUCTED_TYPES)].copy()
    base["iswin"] = (base["sample_type"] == "WINNER_EVENT").astype(np.int8)

    train_a = base[(base["split"] == DISC) & (base["decision_time"] < TRAIN_CUTOFF)]
    family_a = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=30, l2_regularization=3, random_state=1,
    )
    family_a.fit(train_a[a_features], train_a["iswin"])
    base["score_a"] = family_a.predict_proba(base[a_features])[:, 1]

    train_b = base[
        (base["split"] == DISC)
        & (base["decision_time"] < TRAIN_CUTOFF)
        & (base["score_a"] < th["A"])
    ]
    family_b = HistGradientBoostingClassifier(
        max_iter=320, learning_rate=0.04, max_leaf_nodes=31,
        min_samples_leaf=35, l2_regularization=5, random_state=231,
    )
    family_b.fit(train_b[bc_features], train_b["iswin"])
    base["score_b"] = family_b.predict_proba(base[bc_features])[:, 1]

    covered_a = base["score_a"] >= th["A"]
    covered_b = (~covered_a) & (base["score_b"] >= th["B"])
    train_c = base[
        (base["split"] == DISC)
        & (base["decision_time"] < TRAIN_CUTOFF)
        & (~covered_a)
        & (~covered_b)
    ]
    family_c = HistGradientBoostingClassifier(
        max_iter=320, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=4, random_state=315,
    )
    family_c.fit(train_c[bc_features], train_c["iswin"])
    base["score_c"] = family_c.predict_proba(base[bc_features])[:, 1]

    return base, family_a, family_b, family_c


def verify(base: pd.DataFrame, thresholds: dict) -> dict:
    a = base["score_a"].to_numpy() >= thresholds["A"]
    b = (~a) & (base["score_b"].to_numpy() >= thresholds["B"]
    c = (~a) & (~b) & (base["score_c"].to_numpy() >= thresholds["C"]
    signal = a | b | c
    win = base["iswin"].to_numpy().astype(bool)

    summary = {}
    mismatches = []
    for split, expected in EXPECTED.items():
        m = base["split"].to_numpy() == split
        signals = int((m & signal).sum())
        winners = int((m & signal & win).sum())
        summary[split] = {
            "signals": signals,
            "winners": winners,
            "precision": winners / signals if signals else None,
            "family_a": int((m & a).sum()),
            "family_b": int((m & b).sum()),
            "family_c": int((m & c).sum()),
        }
        if signals != expected["signals"] or winners != expected["winners"]:
            mismatches.append(
                f"{split}: got {signals}/{winners}, expected "
                f"{expected['signals']}/{expected['winners']}"
            )
    print(json.dumps(summary, indent=2))
    if mismatches:
        raise RuntimeError("Model integrity check failed: " + "; ".join(mismatches))
    return summary


def main():
    spec = json.loads(SPEC.read_text())
    df, source = load_samples(spec["A_FEATURES"], spec["BC_FEATURES"])
    base, family_a, family_b, family_c = fit_models(df, spec)
    summary = verify(base, spec["thresholds"])

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "event_hgb_model.pkl").open("wb") as f:
        pickle.dump((family_a, spec["A_FEATURES"]), f, protocol=pickle.HIGHEST_PROTOCOL)
    with (OUT / "familyB_model.pkl").open("wb") as f:
        pickle.dump((family_b, spec["BC_FEATURES"]), f, protocol=pickle.HIGHEST_PROTOCOL)
    with (OUT / "familyC15_model.pkl").open("wb") as f:
        pickle.dump((family_c, spec["BC_FEATURES"]), f, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "model_set": "high-purity-abc-v1",
        "dataset_version": "drh2-event-v1-20261004",
        "schema_hash": "8a9a01e2478311b051a1",
        "training_source": source,
        "sklearn_version": sklearn.__version__,
        "training_cutoff_utc": str(TRAIN_CUTOFF),
        "thresholds": spec["thresholds"],
        "verification": summary,
        "note": "Research-only frozen model artifacts. Runtime must not retrain.",
    }
    (OUT / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
