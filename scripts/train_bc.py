"""v4.1: clone recorded demonstrations into a PPO policy, from a committed clone config
(training/bc.py, BCRunConfig; configs/*_bc_r*.yaml), for any task.

Usage:
  uv run python scripts/train_bc.py --config configs/insertion_bc_r1.yaml --seed 0
Writes runs/<config stem>_seed<seed>/ (or --name) like a training run (best_/final_model.zip, the
same network in both, best_/final_vecnormalize.pkl, config.yaml, env_config.yaml, metadata.yaml,
bc_log.json), so PPO can warm-start from it (init_from) and the evaluators can run it
(--policy run:runs/<name>@final). The demonstrations must exist (scripts/record_insertion_demos.py,
scripts/record_door_demos.py). v4.0's door clone (scripts/train_door_bc.py) is unchanged.
"""

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from polypanda.environments.tasks import task_spec
from polypanda.training.bc import BCRunConfig, clone, save_vecnormalize
from polypanda.training.config import TrainConfig
from polypanda.training.runner import _git_state, _lock_hash, plain


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--name")
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--epochs", type=int, help="override (smoke runs only)")
    args = p.parse_args()
    git = _git_state()            # recorded before cloning, so later edits cannot mislabel it
    bc = BCRunConfig.from_yaml(args.config)
    ccfg = replace(bc.clone, seed=args.seed)
    if args.epochs is not None:
        ccfg = replace(ccfg, epochs=args.epochs, value_epochs=min(ccfg.value_epochs, args.epochs))
    train_cfg = replace(TrainConfig.from_yaml(bc.training_config), seed=args.seed)
    task = task_spec(train_cfg.task)
    env_config = task.config_cls.from_yaml(train_cfg.env_config)
    name = args.name or f"{args.config.stem}_seed{args.seed}"
    run_dir = args.runs_dir / name
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists")
    demo_path = Path(bc.demos.format(seed=args.seed))     # (per clone seed with a teacher)
    start = time.perf_counter()
    demos = dict(np.load(demo_path))
    print(f"{demo_path}: {int(demos['kept'])}/{int(demos['total'])} episodes kept, "
          f"{len(demos['obs'])} steps")
    model, venv, log = clone(demos, train_cfg, env_config, ccfg)
    wall = time.perf_counter() - start
    run_dir.mkdir(parents=True)
    model.save(run_dir / "best_model.zip")
    save_vecnormalize(venv, run_dir / "best_vecnormalize.pkl")
    shutil.copy(run_dir / "best_model.zip", run_dir / "final_model.zip")
    shutil.copy(run_dir / "best_vecnormalize.pkl", run_dir / "final_vecnormalize.pkl")
    (run_dir / "bc_log.json").write_text(json.dumps(log, indent=1))
    # (as a training run: the task's config, so evaluation/policies.run_env_cls finds the env;
    # its warm start is the clone's own, not the PPO round's)
    run_cfg = replace(train_cfg, init_from=ccfg.init_from, init_checkpoint=ccfg.init_checkpoint,
                      total_timesteps=0)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(plain(run_cfg), sort_keys=False))
    (run_dir / "env_config.yaml").write_text(yaml.safe_dump(plain(env_config), sort_keys=False))
    meta = {
        "policy_id": name, "skill": train_cfg.task, "algorithm": "BC (into a PPO policy)",
        "seed": args.seed, "observation_schema": task.obs_schema,
        "action_schema": getattr(task.env_cls, "ACTION_SCHEMA", None),
        "demonstrations": {"file": str(demo_path), "sha256_12": file_digest(demo_path),
                           "kept": int(demos["kept"]), "total": int(demos["total"]),
                           "steps": int(len(demos["obs"])), "source": bc.demos_source},
        "clone": asdict(ccfg), "clone_config": str(args.config),
        "training_config": bc.training_config, "env_config": train_cfg.env_config,
        "checkpoints": {"best": "best_model.zip", "final": "final_model.zip"},
        "normalization": {"best": "best_vecnormalize.pkl", "final": "final_vecnormalize.pkl"},
        "log_std": [round(float(x), 4) for x in model.policy.log_std.detach()],
        "wall_clock_s": round(wall, 1),
        "package_lock_hash": _lock_hash(), "git": git,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **({"init_from": dict(log["init"])} if "init" in log else {}),
    }
    (run_dir / "metadata.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    val = log["val_mse"][-1] if log["val_mse"] else float("nan")
    first = (f"; source at first: held-out mse {log['init'].get('start_val_mse', float('nan')):.4f}"
             if "init" in log else "")
    print(f"cloned: train mse {log['train_mse'][-1]:.4f}, held-out mse {val:.4f}, "
          f"value mse {log['value_mse'][-1]:.3f}{first} ({wall:.0f} s) -> {run_dir}")


if __name__ == "__main__":
    main()
