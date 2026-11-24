"""Watch a door actor inside the line's tending cycle (T5), replayed as evaluate_line_door.py runs
it: drive to M1, open, pick, load, close.

Usage:
  uv run python scripts/demo_line_door.py --actor run:runs/door_v2_bc_ppo_r5_seed0@final \
      --set eval_sets/door_final_v1.json --episodes 4 14
  ... --start lowered     # the hand lowered to the door skill's trained start height first
  ... --failures          # the machines this actor failed in its T5 results file
--start line (default) is the cycle as T5 ran it: the door move starts from the line's carry
posture, about 31 cm above the handle; the door policy was trained from about 8 cm.
Close the window to stop. --headless runs without a window and prints the outcome.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from polypanda.evaluation import line_door

START_Z = 0.32          # DoorStartConfig.carry_height: where every training episode starts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--actor", default="run:runs/door_v2_bc_ppo_r5_seed0@final")
    p.add_argument("--set", type=Path, default=Path("eval_sets/door_final_v1.json"))
    p.add_argument("--episodes", type=int, nargs="*")
    p.add_argument("--failures", action="store_true")
    p.add_argument("--start", choices=("line", "lowered"), default="line")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    p.add_argument("--headless", action="store_true")
    args = p.parse_args()
    data = json.loads(args.set.read_text())
    indices = list(args.episodes or [])
    if args.failures:
        name = args.actor.replace("run:", "").replace("/", "_").replace("@", "_")
        rows = json.loads((Path("results/v4/line") / f"{name}_{data['name']}_n100.json")
                          .read_text())["rows"]
        indices += [r["index"] for r in rows if not r["success"]]
    indices = indices or [0]

    viewer = None
    real_step = mujoco.mj_step
    clock = {"sim": 0.0, "wall": 0.0}

    def step(m, d, nstep=1):          # every physics step of the cycle goes through here
        real_step(m, d, nstep=nstep)
        if viewer is None:
            return
        if not viewer.is_running():
            raise SystemExit(0)
        if d.time - clock["sim"] >= 0.05:       # 20 frames per simulated second
            viewer.sync()
            ahead = (d.time - clock["sim"]) / args.speed - (time.perf_counter() - clock["wall"])
            time.sleep(max(0.0, ahead))
            clock["sim"], clock["wall"] = d.time, time.perf_counter()

    mujoco.mj_step = step
    orig_build = line_door.Line.build_line
    orig_move = line_door._door_move

    def build(cfg):
        nonlocal viewer
        line = orig_build(cfg)
        if not args.headless:
            if viewer is not None:
                viewer.close()
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(line.model, line.data)
            viewer.cam.lookat[:] = [-0.6, -0.35, 0.25]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.6, 135.0, -30.0
            clock["sim"], clock["wall"] = 0.0, time.perf_counter()
        return line

    def door_move(ctl, door, command, seed, ep):
        if args.start == "lowered":
            b = ctl.beliefs["M1"]
            ctl.vertical(b.origin[:2], START_Z, ctl.carry_yaw(ctl.angle_of(b.origin[:2])))
        out = orig_move(ctl, door, command, seed, ep)
        print(f"    {command}: {'done' if out['success'] else 'FAILED: ' + str(out['failure'])}"
              f" in {out['steps']} steps, peak force {out['peak_force']:.0f} N")
        return out

    line_door.Line.build_line = staticmethod(build)
    line_door._door_move = door_move
    for n, i in enumerate(indices):
        ep = data["episodes"][i]
        print(f"[{n + 1}/{len(indices)}] machine {i}: lid slides {ep['slide']}"
              f"{', spring' if ep['spring'] > 0 else ''}, travel {ep['travel']:.3f} m, "
              f"start {args.start}")
        row = line_door.run_cycle(args.actor, i, ep)
        state = "RUNNING" if row["success"] else f"not running: {row['failure']}"
        print(f"    machine {state} (lid opened {row['lid_opened']:.3f} m, "
              f"cycle {row['cycle_s']:.0f} s)")
        if viewer is not None:
            time.sleep(1.5)
    if viewer is not None:
        viewer.close()
    np.seterr()


if __name__ == "__main__":
    main()
