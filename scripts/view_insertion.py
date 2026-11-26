"""Watch one insertion episode in MuJoCo's viewer, again and again (slow motion by default).

Usage:
  uv run python scripts/view_insertion.py                                  # straight-down's worst
  uv run python scripts/view_insertion.py --actor search --index 285       # the search, same nest
  uv run python scripts/view_insertion.py --actor random                   # untrained exploration
  uv run python scripts/view_insertion.py --actor run:runs/insertion_ppo_r1_seed0@best
Actors: straight_down, search (the frozen scripts, control/scripted_insertion.py), random (an
untrained PPO policy's exploration: Gaussian noise of round 1's initial std), hold, or a trained
run (run:<dir>[@best|@final]). The episode is `--index` of `--set` (development set by default:
episode 285 is straight-down's worst, 202 N on the nest). After each pass it prints the outcome and
the peak nest force (the damage limit is 50 N). Close the window to stop. Mouse: left-drag
rotates, right-drag moves, scroll zooms; double-click a body to centre on it.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from polypanda.environments.insertion import InsertionEnv
from polypanda.evaluation.insertion import NOISE_SEED, run_policy, run_script
from polypanda.visualization.viewer import fast_rendering


class Closed(Exception):
    pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--actor", default="straight_down")
    p.add_argument("--set", type=Path, default=Path("eval_sets/insertion_dev_v1.json"))
    p.add_argument("--index", type=int, default=285)
    p.add_argument("--speed", type=float, default=0.25, help="playback speed (1 = real time)")
    p.add_argument("--once", action="store_true", help="play the episode once")
    args = p.parse_args()
    if "_final_" in args.set.name or "stretch" in args.set.name:
        p.error("one-shot sets are not for viewing; use a development set")
    ep = json.loads(args.set.read_text())["episodes"][args.index]

    env = InsertionEnv()
    fast_rendering(env.model)
    act = None
    if args.actor.startswith("run:"):
        from polypanda.evaluation.policies import load_run_policy
        spec = args.actor[4:]
        act = load_run_policy(spec if "@" in spec else f"{spec}@final")
    elif args.actor == "random":
        rng = np.random.default_rng(args.index)
        std = np.array([0.1, 0.1, 0.1, 0.1, 0.3])
        act = lambda obs: np.clip(rng.normal(0.0, std), -1, 1)       # noqa: E731
    elif args.actor == "hold":
        act = lambda obs: np.array([0, 0, 0, 0, -1.0])               # noqa: E731
    elif args.actor not in ("straight_down", "search"):
        p.error(f"unknown actor {args.actor!r}")

    dt = 25 * env.model.opt.timestep
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        clock = {"t": time.perf_counter()}

        def show():
            if not viewer.is_running():
                raise Closed
            viewer.sync()
            clock["t"] += dt / args.speed
            time.sleep(max(0.0, clock["t"] - time.perf_counter()))

        try:
            while True:
                env.reset(seed=NOISE_SEED + args.index, options={"episode": ep})
                x, y, z = env.data.xpos[env.part.body]
                cam = viewer.cam                     # close on the nest, from the side
                cam.lookat[:] = [x, y, z - 0.12]
                cam.distance, cam.elevation, cam.azimuth = 0.55, -12.0, 150.0
                env_hook = env.ctl.control_hook      # the env's own (force watch, logs): kept

                def hook(env_hook=env_hook):
                    if env_hook is not None:
                        env_hook()
                    show()
                env.ctl.control_hook = hook
                clock["t"] = time.perf_counter()
                out = run_script(env, args.actor) if act is None else run_policy(env, act)
                f = max(out.get("peak_force") or 0.0, env.stats["peak_force"])
                print(f"{args.actor} on episode {args.index} ({ep['tester']}): "
                      f"{'SEATED' if out['success'] else 'FAILED: ' + str(out['failure'])}, "
                      f"peak nest force {f:.0f} N (limit 50 N), {out.get('time_s', 0):.1f} s")
                if args.once:
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(0.05)
                    return
                for _ in range(int(2.0 / 0.05)):     # a pause before the replay
                    if not viewer.is_running():
                        raise Closed
                    viewer.sync()
                    time.sleep(0.05)
        except Closed:
            pass


if __name__ == "__main__":
    main()
