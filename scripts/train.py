"""Train a policy for a registered task (reach, grasp).

Usage: uv run python scripts/train.py --config configs/reach_ppo.yaml [--seed 0]
                                      [--timesteps N] [--name RUN_NAME]
       uv run python scripts/train.py --config configs/grasp_ppo.yaml
Creates runs/<name>/ (gitignored). Watch progress with: uv run tensorboard --logdir runs
"""

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import yaml

from polypanda.environments.tasks import task_spec
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
    env_config = task_spec(cfg.task).config_cls.from_yaml(cfg.env_config)
    name = args.name or (f"{cfg.task}_{cfg.algorithm}_seed{cfg.seed}_"
                         f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    print(f"training {cfg.algorithm.upper()} on {cfg.task} seed {cfg.seed} for "
          f"{cfg.total_timesteps:,} steps with {cfg.n_envs} {cfg.vec_env} envs "
          f"-> {args.runs_dir / name}")
    run_dir = train(cfg, env_config, args.runs_dir / name)
    meta = yaml.safe_load((run_dir / "metadata.yaml").read_text())
    print(f"done in {meta['wall_clock_s']:.0f} s ({meta['steps_per_s']:.0f} steps/s); "
          f"git dirty: {meta['git']['dirty']}")
    script = {"grasp": "evaluate_grasp.py", "transport": "evaluate_transport.py",
              "transport_residual": "evaluate_transport.py"}.get(cfg.task, "evaluate.py")
    print(f"evaluate: uv run python scripts/{script} --policy run:{run_dir}")


if __name__ == "__main__":
    main()
