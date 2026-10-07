from pathlib import Path
import joblib
import explosive_move_real_world_replay_round8 as r8

out = Path("research/base12_strategy/monitor_export")
out.mkdir(parents=True, exist_ok=True)
models, thresholds, _ = r8.train_frozen_detector()
joblib.dump({"base_model": models["BASE"], "threshold": thresholds["BASE"]}, out / "base_model.joblib")
print("exported", thresholds["BASE"])
