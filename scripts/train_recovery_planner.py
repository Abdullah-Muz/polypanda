"""Train the learned recovery planner (v3 f) with PPO in the symbolic cell.

Usage: uv run python scripts/train_recovery_planner.py --seed 0 [--steps 500000]
  ... --blind situation | --blind clock | --perfect      # ablations (docs/v3_targets.md)
Writes runs/recovery_planner[_<ablation>]_seed<k>/model.zip and meta.json (git state, task,
calibration odds and action times, settings). Each step is one rescue-or-replace decision; the
reward, 1 for an order completed by its deadline, comes at the end of the order, so there is no
discount (gamma 1). The observation is already scaled (SymbolicCell.features): no normalization.
"""

import argparse
import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from polypanda.environments.recovery_planner import RecoveryPlannerEnv
from polypanda.supervisor.symbolic import DEFAULT_TASK, Calibration


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--blind", choices=("situation", "clock"), action="append", default=[])
    parser.add_argument("--perfect", action="store_true",
                        help="train with every skill always succeeding (ablation)")
    parser.add_argument("--envs", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    git = git_state()
    cal = Calibration.from_calibration_run()
    train_cal = cal.perfect() if args.perfect else cal
    tag = "".join(f"_blind-{b}" for b in args.blind) + ("_perfect" if args.perfect else "")
    run_dir = Path(f"runs/recovery_planner{tag}_seed{args.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(lambda: RecoveryPlannerEnv(train_cal, DEFAULT_TASK, tuple(args.blind)),
                       n_envs=args.envs, seed=args.seed)
    settings = {"n_steps": 256, "batch_size": 512, "gamma": 1.0, "gae_lambda": 0.95,
                "learning_rate": 3e-4, "ent_coef": 0.01, "n_epochs": 10,
                "policy_kwargs": {"net_arch": [64, 64]}}
    model = PPO("MlpPolicy", env, seed=args.seed, verbose=0,
                tensorboard_log=str(run_dir / "tb"), **settings)
    model.learn(total_timesteps=args.steps)
    model.save(run_dir / "model.zip")
    (run_dir / "meta.json").write_text(json.dumps({
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"), "git": git,
        "seed": args.seed, "steps": args.steps, "envs": args.envs, "blind": args.blind,
        "perfect_skills": args.perfect, "task": asdict(DEFAULT_TASK), "ppo": settings,
        "calibration": {"odds": {str(k): v for k, v in cal.odds.items()},
                        "action_times_s": cal.times}}, indent=2) + "\n")
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
