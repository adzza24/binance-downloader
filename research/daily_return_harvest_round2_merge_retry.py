from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import daily_return_harvest_round2_parallel as drh2


def read_parts_safe(shard_root: Path, name: str) -> pd.DataFrame:
    parts = []
    for path in sorted(shard_root.glob(f"*/{name}")):
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if len(frame):
            parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--universe", type=Path, required=True)
    args = parser.parse_args()
    drh2.read_parts = read_parts_safe
    drh2.merge(args.shard_root, args.universe)


if __name__ == "__main__":
    main()
