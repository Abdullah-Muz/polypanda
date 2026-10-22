"""Watch a reach policy in the MuJoCo viewer.

Usage: uv run python scripts/demo_reach.py --policy ik [--episodes 10] [--speed 1.0]
       uv run python scripts/demo_reach.py --policy run:runs/<run_dir>

Targets come from the held-out set. The green sphere is the target, drawn at the 2 cm success
tolerance; the red dot is the TCP. After success the episode keeps running for --hold seconds,
so you can see whether the policy stays on the target. Close the window to stop.
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from polypanda.control.classical_baseline import JacobianIKPolicy, RandomPolicy
from polypanda.environments.reach import ReachConfig, ReachEnv
from polypanda.evaluation.reach import evaluation_config
from polypanda.evaluation.sets import ReachEvalSet
from polypanda.policies.loader import load_run


def load_policy(spec: str):
    if spec == "ik":
        return JacobianIKPolicy()
    if spec == "random":
        return RandomPolicy(seed=0)
    if spec.startswith("run:"):
        run_dir, _, which = spec[4:].partition("@")
        return load_run(run_dir, which or "best")
    raise SystemExit(f"unknown policy {spec!r}; use ik, random or run:DIR[@best|final]")


def draw_target(viewer, target: np.ndarray, radius: float) -> None:
    scn = viewer.user_scn
    mujoco.mjv_initGeom(scn.geoms[0], type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[radius, 0, 0], pos=target, mat=np.eye(3).flatten(),
                        rgba=[0.1, 0.9, 0.2, 0.45])
    scn.ngeom = 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="ik")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    parser.add_argument("--hold", type=float, default=2.0, help="seconds to watch after success")
    parser.add_argument("--set", type=Path, default=Path("eval_sets/reach_nominal_v1.json"))
    args = parser.parse_args()

    policy = load_policy(args.policy)
    eval_set = ReachEvalSet.load(args.set)
    env = ReachEnv(evaluation_config(ReachConfig()))  # success does not end the episode
    tolerance = env.config.success.tolerance
    frame = env.adapter.control_dt / args.speed
    hold_steps = round(args.hold / env.adapter.control_dt)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.opt.sitegroup[4] = 1  # show the TCP site
        for i, target in enumerate(eval_set.targets[:args.episodes]):
            obs, info = env.reset(options={"target": target})
            draw_target(viewer, target, tolerance)
            viewer.sync()
            time.sleep(0.5 / args.speed)
            reached_at = None
            for _ in range(env.config.episode.horizon):
                if not viewer.is_running():
                    return
                start = time.perf_counter()
                action, _ = policy.predict(obs, deterministic=True)
                obs, _, _, truncated, info = env.step(action)
                viewer.sync()
                time.sleep(max(0.0, frame - (time.perf_counter() - start)))
                if reached_at is None and info["is_success"]:
                    reached_at = env.elapsed_steps
                held_long_enough = (reached_at is not None
                                    and env.elapsed_steps - reached_at >= hold_steps)
                if truncated or held_long_enough:
                    break
            dt = env.adapter.control_dt
            outcome = "timed out" if reached_at is None else f"reached in {reached_at * dt:.2f} s"
            print(f"target {i + 1}/{args.episodes} ({eval_set.regions[i]}): {outcome}; "
                  f"{args.hold:.1f} s later distance {info['distance'] * 1e3:.1f} mm")
            time.sleep(0.7 / args.speed)


if __name__ == "__main__":
    main()
