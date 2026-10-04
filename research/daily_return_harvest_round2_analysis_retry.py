from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import daily_return_harvest_round2_event_discovery as drh
import daily_return_harvest_round2_staged_merge as staged


def qsafe_float(s: pd.Series, q: float) -> float:
    """Quantile helper that safely treats bool/int/float features as numeric floats."""
    z = pd.to_numeric(s, errors="coerce").astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
    return float(z.quantile(q)) if len(z) else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical-root", type=Path, required=True)
    ap.add_argument("--context-dir", type=Path, required=True)
    args = ap.parse_args()

    # make_condition_library resolves qsafe from the drh module at runtime.
    drh.qsafe = qsafe_float
    staged.analyse(args.canonical_root, args.context_dir)


if __name__ == "__main__":
    main()
