"""Watch a grasp controller pick up cubes from the held-out placement set.

Usage: uv run python scripts/demo_grasp.py [--policy scripted] [--episodes 5] [--speed 1.0]
       uv run python scripts/demo_grasp.py --policy run:runs/grasp_ppo_curr_seed0
Runs through GraspEnv (the env's gate closes the fingers), keeping each episode going for
--hold seconds after the first verified grasp. Close the window to stop.
"""

import argparse
import time
from pathlib import Path

import mujoco.viewer

from polypanda.environments.grasp import GraspEnv
from polypanda.evaluation.grasp import PolicyDriver, ScriptedDriver
from polypanda.evaluation.sets import GraspEvalSet
from polypanda.policies.loader import load_run
from polypanda.visualization.viewer import fast_rendering


def make_driver(spec: str):
    if spec == "scripted":
        return ScriptedDriver()
    if spec.startswith("run:"):
        run_dir, _, which = spec[4:].partition("@")
        return PolicyDriver(load_run(run_dir, which or "best"))
    raise SystemExit(f"unknown policy {spec!r}; use scripted or run:DIR[@best|final]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="scripted")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    parser.add_argument("--shadows", action="store_true", help="render shadows and reflections")
    parser.add_argument("--hold", type=float, default=2.0, help="seconds to watch after success")
    parser.add_argument("--set", type=Path, default=Path("eval_sets/grasp_nominal_v1.json"))
    args = parser.parse_args()

    env = GraspEnv()
    driver = make_driver(args.policy)
    eval_set = GraspEvalSet.load(args.set)
    dt = env.adapter.control_dt
    frame = dt / args.speed
    hold_steps = round(args.hold / dt)

    if not args.shadows:
        fast_rendering(env.model)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.opt.sitegroup[4] = 1  # show the TCP site
        for i, (x, y, yaw) in enumerate(eval_set.placements[:args.episodes]):
            obs, _ = env.reset(options={"cube": (x, y, yaw)})
            driver.reset(env)
            viewer.sync()
            time.sleep(0.5 / args.speed)
            closed_at = success_at = None
            info = {}
            for _ in range(env.config.episode.horizon):
                if not viewer.is_running():
                    return
                start = time.perf_counter()
                obs, _, terminated, truncated, info = env.step(driver(env, obs))
                viewer.sync()
                if closed_at is None and info["gripper_closed"]:
                    closed_at = env.elapsed_steps * dt
                if success_at is None and info["is_success"]:
                    success_at = env.elapsed_steps * dt
                time.sleep(max(0.0, frame - (time.perf_counter() - start)))
                if terminated or truncated or (
                        success_at is not None
                        and env.elapsed_steps * dt - success_at >= hold_steps * dt):
                    break
            if info.get("is_success"):
                outcome = (f"GRASPED: fingers closed at {closed_at:.1f} s, "
                           f"verified at {success_at:.1f} s, lifted {info['lift'] * 100:.1f} cm")
            elif info.get("failure") == "knocked_over":
                outcome = "failed: knocked the cube over"
            elif closed_at is None:
                outcome = "failed: never lined up well enough for the fingers to close"
            else:
                outcome = (f"failed: closed at {closed_at:.1f} s "
                           f"but lift only {info['lift'] * 100:.1f} cm")
            print(f"cube {i + 1}/{args.episodes} at ({x:.2f}, {y:+.2f}), yaw {yaw:+.2f}: {outcome}")
            time.sleep(1.0 / args.speed)


if __name__ == "__main__":
    main()
