"""v4.0: record feel-script demonstrations and clone them into a PPO policy (training/bc.py).

Usage:
  uv run python scripts/train_door_bc.py --name door_bc_v1_seed0 [--episodes 3000] [--seed 0]
Writes runs/<name>/ like a training run (best_/final_model.zip, best_/final_vecnormalize.pkl,
metadata.yaml, bc_log.json), so PPO can warm-start from it (init_from) and evaluate_door.py can
run it (--policy run:runs/<name>). The demonstrations are cached in runs/door_demos/.
"""

import argparse
import json
import shutil
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from polypanda.environments.door import OBS_SCHEMA, DoorConfig, DoorEnv
from polypanda.training.bc import CloneConfig, RecordConfig, clone, record, save_vecnormalize
from polypanda.training.config import TrainConfig
from polypanda.training.runner import _git_state, _lock_hash


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--config", type=Path, default=Path("configs/door_v1_ppo.yaml"))
    p.add_argument("--episodes", type=int, default=RecordConfig.episodes)
    p.add_argument("--noise", type=float, default=RecordConfig.noise)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=CloneConfig.epochs)
    p.add_argument("--log-std", type=float, default=CloneConfig.log_std)
    p.add_argument("--jobs", type=int, default=RecordConfig.jobs)
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = p.parse_args()
    git = _git_state()
    train_cfg = replace(TrainConfig.from_yaml(args.config), seed=args.seed)
    env_config = DoorConfig.from_yaml(train_cfg.env_config)
    rec = RecordConfig(episodes=args.episodes, noise=args.noise, jobs=args.jobs)

    cache = args.runs_dir / "door_demos" / f"feel_{rec.episodes}_{rec.seed}_{rec.noise}.npz"
    start = time.perf_counter()
    if cache.exists():
        demos = dict(np.load(cache))
        print(f"demonstrations from {cache}")
    else:
        demos = record(rec)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **demos)
        print(f"recorded {demos['kept']}/{demos['total']} successful episodes, "
              f"{len(demos['obs'])} steps ({time.perf_counter() - start:.0f} s) -> {cache}")
    model, venv, log = clone(demos, train_cfg, env_config,
                             CloneConfig(epochs=args.epochs, log_std=args.log_std,
                                         seed=args.seed))
    run_dir = args.runs_dir / args.name
    run_dir.mkdir(parents=True, exist_ok=False)
    model.save(run_dir / "best_model.zip")
    save_vecnormalize(venv, run_dir / "best_vecnormalize.pkl")
    shutil.copy(run_dir / "best_model.zip", run_dir / "final_model.zip")
    shutil.copy(run_dir / "best_vecnormalize.pkl", run_dir / "final_vecnormalize.pkl")
    (run_dir / "bc_log.json").write_text(json.dumps(log, indent=1))
    meta = {
        "policy_id": args.name, "skill": "door", "algorithm": "BC (into a PPO policy)",
        "seed": args.seed, "observation_schema": OBS_SCHEMA,
        "action_schema": DoorEnv.ACTION_SCHEMA,
        "demonstrations": {"script": "feel (control/door_baselines.py)", "file": str(cache),
                           "kept": int(demos["kept"]), "total": int(demos["total"]),
                           "steps": int(len(demos["obs"])), **asdict(rec)},
        "clone": asdict(CloneConfig(epochs=args.epochs, log_std=args.log_std,
                                    seed=args.seed)),
        "training_config": str(args.config), "env_config": train_cfg.env_config,
        "checkpoints": {"best": "best_model.zip", "final": "final_model.zip"},
        "normalization": {"best": "best_vecnormalize.pkl", "final": "final_vecnormalize.pkl"},
        "package_lock_hash": _lock_hash(), "git": git,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (run_dir / "metadata.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    val = log['val_mse'][-1] if log['val_mse'] else float('nan')
    print(f"cloned: train mse {log['train_mse'][-1]:.4f}, held-out mse {val:.4f}, "
          f"value mse {log['value_mse'][-1]:.3f} -> {run_dir}")


if __name__ == "__main__":
    main()
