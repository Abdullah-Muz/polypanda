"""Probe a stacking place policy in its training scene: precision, bias and how much it nudges.

Usage: uv run python scripts/probe_place_stack.py --place runs/<place run> [--config ...]
For towers of 0, 1 and 3 cubes, 40 episodes each from fixed seeds, the policy sets a cube down
and the probe reports:
- placed: the env's own success check;
- off-centre: the settled cube's distance from the cube below (on the table: from the target),
  median and 90th percentile;
- bias: the mean settled offset from the target and its spread, in the training frame (x away
  from the robot, y to its left), over the placed episodes;
- nudge: the most any tower cube moved during the episode (the episode is not ended at the
  config's limit, so the probe sees the whole placement).
The config defaults to the run's own task config (env_config.yaml in the run directory).
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from polypanda.environments.place import PlaceConfig, PlaceEnv
from polypanda.policies.loader import load_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--place", type=Path, required=True, help="place run directory")
    parser.add_argument("--config", type=Path, help="place env config (default: the run's)")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=5000)
    args = parser.parse_args()
    cfg = PlaceConfig.from_yaml(args.config or args.place / "env_config.yaml")
    cfg = replace(cfg, stack=replace(cfg.stack, disturb_tol=1.0))  # see the whole placement
    env, policy = PlaceEnv(cfg), load_run(str(args.place))
    print(f"{args.place.name} ({args.episodes} episodes per tower height)")
    for level in (0, 1, 3):
        placed, off, offset, nudge = 0, [], [], []
        for k in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + k, options={"level": level})
            target = env.target[:2].copy()
            worst = 0.0
            for _ in range(cfg.horizon):
                action, _ = policy.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                worst = max(worst, info["tower_moved"])
                if terminated or truncated:
                    break
            cube = env.data.xpos[env.cube.body][:2]
            below = env.data.xpos[env.tower[level - 1].body][:2] if level else target
            off.append(np.linalg.norm(cube - below) * 1000)
            nudge.append(worst * 1000)
            if info["is_success"]:
                placed += 1
                offset.append((cube - target) * 1000)
        off, nudge, offset = np.array(off), np.array(nudge), np.array(offset).reshape(-1, 2)
        mean, sd = offset.mean(0), offset.std(0)
        line = (f"  on {level} cubes: placed {placed}/{args.episodes}; off-centre median "
                f"{np.median(off):.1f} mm (p90 {np.percentile(off, 90):.1f}); bias "
                f"({mean[0]:+.1f}, {mean[1]:+.1f}) mm, spread ({sd[0]:.1f}, {sd[1]:.1f})")
        if level:
            line += (f"; nudge median {np.median(nudge):.1f} mm, over 2 mm in "
                     f"{(nudge > 2).mean():.0%}")
        print(line, flush=True)


if __name__ == "__main__":
    main()
