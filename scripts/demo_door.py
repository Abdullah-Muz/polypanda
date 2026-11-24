"""Watch a door policy on episodes of a door set, e.g. the ones it slammed in an evaluation.

Usage:
  uv run python scripts/demo_door.py --slams        # round 2's slams on door_dev_v1, slowed
  uv run python scripts/demo_door.py --policy run:runs/door_v1_bc_ppo_r2_seed0 --episodes 8 26
  uv run python scripts/demo_door.py --policy feel --failures --speed 0.5
Each episode is replayed exactly as scripts/evaluate_door.py ran it (same machine, same sensor
noise seed), so --slams / --failures pick episodes from that evaluation's results file
(results/v4/door/<policy>_<set>.json). The lid's speed is printed when it hits a stop, and the
view holds there for --hold seconds. Close the window to stop.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco.viewer
import numpy as np

from polypanda.environments.door import DoorEnv
from polypanda.visualization.viewer import fast_rendering

NOISE_SEED = 7_000_000           # scripts/evaluate_door.py


def actor(policy: str, env):
    if policy.startswith("run:"):
        from polypanda.evaluation.policies import load_run_policy
        act = load_run_policy(policy[4:])
        return (lambda obs: act(obs)), (lambda: None)
    from polypanda.control.door_baselines import BaselineDoor
    from polypanda.control.scripted_door import DoorExpert
    script = DoorExpert(env) if policy == "expert" else BaselineDoor(env, policy)
    return (lambda obs: script.act()), script.reset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="run:runs/door_v1_bc_ppo_r2_seed0")
    p.add_argument("--set", type=Path, default=Path("eval_sets/door_dev_v1.json"))
    p.add_argument("--episodes", type=int, nargs="*", help="set indices to play")
    p.add_argument("--slams", action="store_true", help="the episodes it slammed (results file)")
    p.add_argument("--failures", action="store_true", help="the episodes it failed")
    p.add_argument("--speed", type=float, default=0.3, help="playback speed (1 = real time)")
    p.add_argument("--hold", type=float, default=2.0, help="seconds held at a slam")
    args = p.parse_args()

    data = json.loads(args.set.read_text())
    episodes = data["episodes"] if isinstance(data, dict) else data
    indices = list(args.episodes or [])
    if args.slams or args.failures:
        name = args.policy.replace("run:", "").replace("runs/", "runs_").replace("/", "_")
        rows = json.loads((Path("results/v4/door") / f"{name}_{args.set.stem}.json")
                          .read_text())["rows"]
        indices += [r["index"] for r in rows
                    if (args.slams and r["slams"]) or (args.failures and not r["success"])]
    if not indices:
        indices = [0, 1, 2]
    env = DoorEnv()
    act, reset = actor(args.policy, env)
    fast_rendering(env.model)
    dt = env.adapter.control_dt
    r = env.config.reward
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for n, i in enumerate(indices):
            ep = episodes[i]
            obs, _ = env.reset(seed=NOISE_SEED + i, options={"episode": ep})
            reset()
            viewer.cam.lookat[:] = env.handle_xyz()
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.1, 200.0, -30.0
            print(f"[{n + 1}/{len(indices)}] episode {i}: {ep['command']} a {ep['slide']} lid"
                  f"{' with a spring' if ep['spring'] > 0 else ''}, travel {ep['travel']:.3f} m")
            viewer.sync()
            time.sleep(1.0)
            done, info = False, {}
            while not done:
                if not viewer.is_running():
                    return
                start = time.perf_counter()
                q_prev = env.machine.lid
                obs, _, term, trunc, info = env.step(act(obs))
                done = term or trunc
                viewer.sync()
                speed = abs(info["lid"] - q_prev) / dt
                if info["reward_terms"]["slam"] < 0 or (
                        speed > 0 and (info["lid"] >= env.machine.cfg.travel - 0.001 > q_prev
                                       or info["lid"] <= 0.001 < q_prev)):
                    limit = r.slam_open_speed if info["lid"] > q_prev else r.slam_close_speed
                    tag = "SLAM" if speed > limit else "hit the stop"
                    print(f"    step {env.elapsed_steps}: {tag} at {speed:.3f} m/s "
                          f"(allowed {limit:.2f})")
                    if speed > limit:
                        time.sleep(args.hold)
                time.sleep(max(0.0, dt / args.speed - (time.perf_counter() - start)))
            st = info["stats"]
            print(f"    {'success' if info['is_success'] else 'failed: ' + str(info['failure'])}"
                  f" in {env.elapsed_steps} steps, peak force {st['peak_force']:.0f} N, "
                  f"lid {np.round(info['lid'], 3)} m")
            time.sleep(1.0)


if __name__ == "__main__":
    main()
