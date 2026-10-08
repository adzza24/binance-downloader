#!/usr/bin/env python3
"""Convert existing frozen sklearn family_models.joblib to the exact BASE12 core JSON schema.
No training, refitting, retraining, or strategy changes are permitted."""
import json
import sys
from pathlib import Path
import joblib
import numpy as np
from explosive_move_discovery_round1 import FEATURES

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
payload = joblib.load(input_path)
assert sorted(payload["models"]) == ["BASE", "CAPITULATION", "MOMENTUM"], "Missing family"
assert len(FEATURES) == 34, "Unexpected trained feature count"
models = {}
for family, (imputer, forest) in payload["models"].items():
    median = [float(v) for v in imputer.statistics_]
    assert len(median) == len(FEATURES)
    trees = []
    for estimator in forest.estimators_:
        tr = estimator.tree_
        # sklearn tree_.value may store normalised fractions or counts.
        probabilities = []
        for v in tr.value[:, 0, :]:
            total = float(np.sum(v))
            probabilities.append(float(v[1] / total) if len(v) > 1 and total > 0 else 0.0)
        trees.append({
            "f": [int(v) for v in tr.feature],
            "x": [float(v) for v in tr.threshold],
            "l": [int(v) for v in tr.children_left],
            "r": [int(v) for v in tr.children_right],
            "p": probabilities,
        })
    assert len(trees) == 350, (family, len(trees))
    models[family] = {"imputer_median": median, "trees": trees}
    # Check converted traversal against the original frozen sklearn forest.
    probes = np.tile(np.asarray(median, dtype=float), (6, 1))
    for i in range(1, 6):
        probes[i, :] *= 1.0 + 0.013 * i
    wanted = forest.predict_proba(imputer.transform(probes))[:, 1]
    actual = []
    for vec in probes:
        total = 0.0
        for tree in trees:
            node = 0
            while tree["f"][node] >= 0:
                node = tree["l"][node] if vec[tree["f"][node]] <= tree["x"][node] else tree["r"][node]
            total += tree["p"][node]
        actual.append(total / len(trees))
    assert np.allclose(wanted, actual, rtol=0, atol=1e-12), f"{family} model conversion did not match"
    print("Validated frozen model", family, len(trees), "trees", flush=True)
out = {
    "feature_order": list(FEATURES),
    "thresholds": {str(k): float(v) for k, v in payload["thresholds"].items()},
    "models": models
}
assert out["thresholds"]["CAPITULATION"] == 0.70
assert out["thresholds"]["BASE"] == 0.65
assert out["thresholds"]["MOMENTUM"] == 0.35
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(out, separators=(",", ":"), allow_nan=False))
print("Frozen model export ready; family models unchanged", flush=True)
