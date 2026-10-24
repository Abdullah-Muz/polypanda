"""Watch a grasp controller pick up cubes from the held-out placement set.

Usage: uv run python scripts/demo_grasp.py [--policy scripted] [--episodes 5] [--speed 1.0]
       uv run python scripts/demo_grasp.py --policy run:runs/grasp_ppo_curr_seed0
       uv run python scripts/demo_grasp.py --policy run:runs/<training run> --watch
Runs through the policy's own task environment (GraspEnv, or GraspV2Env for v2 runs; the env's
gate closes the fingers), keeping each episode going for --hold seconds after the first verified
grasp. --watch reloads the run's newest best checkpoint before every episode, so you can watch a
policy improve while it trains (episodes then cycle through the set indefinitely). Close the
window to stop.
"""

import argparse
import itertools
import time
from pathlib import Path

import mujoco.viewer

from polypanda.evaluation.grasp import PolicyDriver
from polypanda.evaluation.pickplace import grasp_driver_from_spec, grasp_env_for
from polypanda.evaluation.sets import GraspEvalSet
from polypanda.policies.loader import load_run
from polypanda.visualization.viewer import fast_rendering


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="scripted", help="scripted | run:DIR[@best|final]")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--watch", action="store_true",
                        help="reload the newest best checkpoint every episode (live training)")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    parser.add_argument("--shadows", action="store_true", help="render shadows and reflections")
    parser.add_argument("--hold", type=float, default=2.0, help="seconds to watch after success")
    parser.add_argument("--set", type=Path, default=Path("eval_sets/grasp_nominal_v1.json"))
    args = parser.parse_args()

    env = grasp_env_for(args.policy)
    run_dir = Path(args.policy[4:].partition("@")[0]) if args.policy.startswith("run:") else None
    best_file = run_dir / "best_model.zip" if run_dir else None
    if args.watch and best_file is not None and not best_file.exists():
        print("waiting for the run's first validated checkpoint (every 50k training steps)...")
        while not best_file.exists():
            time.sleep(5)
        time.sleep(2)  # let the normalization statistics finish writing
    driver = grasp_driver_from_spec(args.policy)[0]
    loaded_mtime = best_file.stat().st_mtime if best_file and best_file.exists() else None
    eval_set = GraspEvalSet.load(args.set)
    dt = env.adapter.control_dt
    frame = dt / args.speed
    hold_steps = round(args.hold / dt)

    if not args.shadows:
        fast_rendering(env.model)
    order = itertools.cycle(range(len(eval_set))) if args.watch else range(args.episodes)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.opt.sitegroup[4] = 1  # show the TCP site
        for n, i in enumerate(order):
            if args.watch and best_file is not None and best_file.exists():
                mtime = best_file.stat().st_mtime
                if mtime != loaded_mtime:
                    driver = PolicyDriver(load_run(run_dir, "best"))
                    loaded_mtime = mtime
                    print(f"--- loaded newer best checkpoint "
                          f"({time.strftime('%H:%M:%S', time.localtime(mtime))})")
            x, y, yaw = eval_set.placements[i]
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
            elif info.get("failure") == "pushed_away":
                outcome = "failed: pushed the cube more than 5 cm"
            elif closed_at is None:
                outcome = "failed: never lined up well enough for the fingers to close"
            else:
                outcome = (f"failed: closed at {closed_at:.1f} s "
                           f"but lift only {info['lift'] * 100:.1f} cm")
            label = f"{n + 1}" if args.watch else f"{n + 1}/{args.episodes}"
            print(f"cube {label} at ({x:.2f}, {y:+.2f}), yaw {yaw:+.2f}: {outcome}")
            time.sleep(1.0 / args.speed)


if __name__ == "__main__":
    main()
