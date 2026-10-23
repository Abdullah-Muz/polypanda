"""Watch a transport policy carry grasped cubes to goals from the held-out set.

Usage: uv run python scripts/demo_transport.py --policy scripted [--episodes 6]
       uv run python scripts/demo_transport.py --policy run:runs/transport_ppo_pilot_seed0
       uv run python scripts/demo_transport.py --policy run:runs/<training run> --watch
The green sphere is the goal at the 2 cm success tolerance. --watch reloads the run's newest
best checkpoint before every episode, so you can watch a policy improve while it trains
(episodes then cycle through the set indefinitely). Close the window to stop.
"""

import argparse
import itertools
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from polypanda.environments.handoff import HandoffPool
from polypanda.environments.transport import TransportEnv
from polypanda.evaluation.grasp import PolicyDriver
from polypanda.evaluation.transport import ScriptedCarryDriver
from polypanda.policies.loader import load_run
from polypanda.visualization.viewer import fast_rendering


def draw_goal(viewer, goal: np.ndarray, radius: float) -> None:
    scn = viewer.user_scn
    mujoco.mjv_initGeom(scn.geoms[0], type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[radius, 0, 0],
                        pos=goal, mat=np.eye(3).flatten(), rgba=[0.1, 0.9, 0.2, 0.4])
    scn.ngeom = 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="scripted", help="scripted | run:DIR[@best|final]")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--watch", action="store_true",
                        help="reload the newest best checkpoint every episode (live training)")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    parser.add_argument("--shadows", action="store_true", help="render shadows and reflections")
    parser.add_argument("--set", type=Path, default=Path("eval_sets/transport_nominal_v1.npz"))
    args = parser.parse_args()

    env = TransportEnv()
    if not args.shadows:
        fast_rendering(env.model)
    eval_set = HandoffPool.load(args.set, env.model)
    run_dir = None
    if args.policy == "scripted":
        driver = ScriptedCarryDriver()
    elif args.policy.startswith("run:"):
        run_dir, _, which = args.policy[4:].partition("@")
        driver = PolicyDriver(load_run(run_dir, which or "best"))
    else:
        raise SystemExit(f"unknown policy {args.policy!r}")
    frame = env.adapter.control_dt / args.speed
    best_file = Path(run_dir) / "best_model.zip" if run_dir else None
    loaded_mtime = best_file.stat().st_mtime if best_file and best_file.exists() else None

    order = itertools.cycle(range(len(eval_set))) if args.watch else range(args.episodes)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.opt.sitegroup[4] = 1
        for n, i in enumerate(order):
            if args.watch and best_file is not None and best_file.exists():
                mtime = best_file.stat().st_mtime
                if mtime != loaded_mtime:
                    driver = PolicyDriver(load_run(run_dir, "best"))
                    loaded_mtime = mtime
                    print(f"--- loaded newer best checkpoint "
                          f"({time.strftime('%H:%M:%S', time.localtime(mtime))})")
            obs, info = env.reset(options={"state": eval_set.states[i], "goal": eval_set.goals[i]})
            driver.reset(env)
            draw_goal(viewer, env.goal, env.config.success.tolerance)
            viewer.sync()
            time.sleep(0.5 / args.speed)
            first = None
            for _ in range(env.config.episode.horizon):
                if not viewer.is_running():
                    return
                start = time.perf_counter()
                obs, _, terminated, truncated, info = env.step(driver(env, obs))
                viewer.sync()
                if first is None and info["is_success"]:
                    first = env.elapsed_steps * env.adapter.control_dt
                time.sleep(max(0.0, frame - (time.perf_counter() - start)))
                if terminated or truncated:
                    break
            if info["is_success"]:
                outcome = f"DELIVERED at {first:.1f} s and held"
            elif info["failure"] == "lost_grip":
                outcome = f"LOST GRIP after {env.elapsed_steps * env.adapter.control_dt:.1f} s"
            elif info["failure"] == "lowered_onto_table":
                outcome = (f"LOWERED ONTO TABLE after "
                           f"{env.elapsed_steps * env.adapter.control_dt:.1f} s (still holding)")
            else:
                outcome = f"not delivered: {info['distance'] * 100:.1f} cm from goal"
            print(f"episode {n + 1} (start from {eval_set.sources[i]}): {outcome}")
            time.sleep(1.0 / args.speed)


if __name__ == "__main__":
    main()
