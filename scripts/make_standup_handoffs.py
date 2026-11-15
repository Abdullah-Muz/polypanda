"""v3.1 skill round: the grips the learned lying-cube grasp really hands the stand-up.

Usage:
  uv run python scripts/make_standup_handoffs.py --grasp runs/grasp_anypose5_seed0 [--n 300]
      [--seed 20261207] [--out data/handoff/standup_handoffs_<run>.npz]

In the work cell (LearnedCell, the learned grasp only), a cube lies on a side where a knock left
a lying cube in the v3.0 calibration run (radius and angle of a recoverable lying knock row, a
random yaw); the learned grasp picks it and lifts it to the carry posture, as the cell does
before a stand-up. For every pick it records whether the cube is held and, if it is, the cube's
pose in the hand (TCP site frame): its centre (p_rel) and rotation (R_rel), as the stand-up's
start (StandUpEnv._start) builds them. Also printed, against the synthetic grip the stand-up was
trained from (the long axis exactly along the hand's x, slid at most 1 cm, its centre 0.5 cm
below to 1 cm above the TCP): how far the long axis turns off the hand's x axis, how far the
cube rolls in the fingers, and the slide along and across the long axis.
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from polypanda.environments.grasp import GRIPPER_OPEN
from polypanda.simulation.workcell import Workcell, WorkOrder
from polypanda.supervisor.cell import LearnedCell, LearnedSkill
from polypanda.supervisor.symbolic import CALIBRATION_RUNS

ORDER = WorkOrder((("red", "green", "blue"), ("yellow",), ("red",)),
                  tuple((c, None) for c in ("red", "green", "yellow", "blue", "red")))
LYING = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])  # 90 deg about x


def lying_rests() -> np.ndarray:
    """(radius, angle deg) of every recoverable lying cube a v3.0 calibration knock left."""
    rows = [r for p in sorted(Path().glob(CALIBRATION_RUNS)) for r in csv.DictReader(open(p))
            if r["skill"] == "knock" and r["pose"] == "lying" and r["recoverable"] == "True"]
    return np.array([[float(r["radius"]), float(r["angle"])] for r in rows])


def deg_between_axes(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two lines (either direction), in degrees."""
    return float(np.degrees(np.arccos(min(1.0, abs(float(a @ b))))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grasp", type=Path, required=True)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261207)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or Path(f"data/handoff/standup_handoffs_{args.grasp.name}.npz")

    rests = lying_rests()
    cell = Workcell.build()
    ctl = LearnedCell(cell, grasp=LearnedSkill(args.grasp))
    cube = cell.cubes[3]
    held, p_rel, q_rel, where = [], [], [], []
    for i in range(args.n):
        rng = np.random.default_rng([args.seed, i])
        cell.reset(ORDER, rng)
        r, a = rests[rng.integers(len(rests))]
        yaw = rng.uniform(-np.pi, np.pi)
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]), LYING)
        x, y = r * np.cos(np.radians(a)), r * np.sin(np.radians(a))
        ctl.d.qpos[cube.qpos:cube.qpos + 7] = [x, y, cube.half_size[0], *q]
        ctl.d.qvel[:] = 0.0
        mujoco.mj_forward(ctl.m, ctl.d)
        ctl.q_cmd, ctl.grip = ctl.home.copy(), GRIPPER_OPEN   # (open, as before any cell pick)
        ctl.to_carry()
        ctl.step(20)
        ctl.source, ctl.order_index = "loose", i
        ctl.pick(cube)
        ok = cell.attached(cube)
        held.append(ok)
        where.append((r, a, yaw))
        if ok:
            site = ctl.d.site_xpos[ctl.arm.tcp_site]
            R_site = ctl.d.site_xmat[ctl.arm.tcp_site].reshape(3, 3)
            p_rel.append(R_site.T @ (ctl.d.xpos[cube.body] - site))
            R_rel = R_site.T @ ctl.d.xmat[cube.body].reshape(3, 3)
            qr = np.zeros(4)
            mujoco.mju_mat2Quat(qr, R_rel.flatten())
            q_rel.append(qr)
        print(f"{i}: r {r:.2f} angle {a:6.1f}: {'held' if ok else 'not held'}", flush=True)

    p_rel, q_rel = np.array(p_rel), np.array(q_rel)
    turn, roll = [], []
    for qr in q_rel:
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, qr)
        R = R.reshape(3, 3)
        turn.append(deg_between_axes(R[:, 2], np.array([1.0, 0.0, 0.0])))  # long axis vs x
        z = np.array([0.0, 0.0, 1.0])     # (square section: either side face may face up)
        roll.append(min(deg_between_axes(R[:, 0], z), deg_between_axes(R[:, 1], z)))
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    meta = {"grasp": str(args.grasp), "n": args.n, "seed": args.seed,
            "held": int(sum(held)), "git": git.stdout.strip(),
            "rests": "recoverable lying knock rows of the v3.0 calibration run"}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, p_rel=p_rel, q_rel=q_rel, held=np.array(held), where=np.array(where),
             meta=json.dumps(meta))

    def pct(v):
        return " / ".join(f"{np.percentile(v, q):.1f}" for q in (50, 90, 99))
    print(f"\n{args.grasp.name}: held {sum(held)}/{args.n}; wrote {out}")
    print(f"  long axis off the hand's x (deg, median / 90th / 99th): {pct(turn)}"
          "   synthetic: 0")
    print(f"  roll in the fingers (deg): {pct(roll)}   synthetic: 0")
    print(f"  slide along x (cm): {pct(100 * np.abs(p_rel[:, 0]))}   synthetic: up to 1.0")
    print(f"  across, y (cm): {pct(100 * np.abs(p_rel[:, 1]))}   synthetic: 0")
    print(f"  centre z in the hand (cm, min / median / max): {100 * p_rel[:, 2].min():.1f} / "
          f"{100 * np.median(p_rel[:, 2]):.1f} / {100 * p_rel[:, 2].max():.1f}"
          "   synthetic: -1.0 to 0.5")


if __name__ == "__main__":
    main()
