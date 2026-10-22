"""Train a reach policy.

Usage: uv run python scripts/train.py --config configs/reach_ppo.yaml [--seed 0]
                                      [--timesteps N] [--name RUN_NAME]
Creates runs/<name>/ (gitignored). Watch progress with: uv run tensorboard --logdir runs
"""

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import yaml

from polypanda.environments.reach import ReachConfig
from polypanda.training.config import TrainConfig
from polypanda.training.runner import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--name")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    if args.timesteps is not None:
        cfg = replace(cfg, total_timesteps=args.timesteps)
    env_config = ReachConfig.from_yaml(cfg.env_config)
    name = args.name or (f"reach_{cfg.algorithm}_seed{cfg.seed}_"
                         f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    print(f"training {cfg.algorithm.upper()} seed {cfg.seed} for {cfg.total_timesteps:,} steps "
          f"with {cfg.n_envs} {cfg.vec_env} envs -> {args.runs_dir / name}")
    run_dir = train(cfg, env_config, args.runs_dir / name)
    meta = yaml.safe_load((run_dir / "metadata.yaml").read_text())
    print(f"done in {meta['wall_clock_s']:.0f} s ({meta['steps_per_s']:.0f} steps/s); "
          f"git dirty: {meta['git']['dirty']}")
    print(f"evaluate: uv run python scripts/evaluate.py --policy run:{run_dir}")


if __name__ == "__main__":
    main()
