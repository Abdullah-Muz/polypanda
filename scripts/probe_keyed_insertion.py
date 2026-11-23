"""v4.0 probe: does a keyed part go into the keyed pocket only the right way round?
(docs/keyed_part.md, docs/v4_plan.md section 3.4)

A measurement of the keyed fixture, not a skill evaluation. The line's parts carry a rib on one
face (LineConfig.keyed_parts) and M1's pocket has the matching slot in its back wall
(MachineConfig.keyed). The scripted tending controller picks a part standing at the machine's
stop (the infeed's jitter: +-1.5 cm, and any yaw, so the rib starts anywhere) with `pick_for`
and loads it with `load`, which turns the rib to where the script believes the slot is. It
answers:

1. Right way round: the script believes the slot is where it is. Does the part seat, with the
   place aimed at the pocket's centre plus an error of 0-3 mm in a random direction (the realistic
   camera error of docs/pose_error.md)?
2. Wrong way round: the script believes the slot is in another wall (90, 180 or 270 deg round),
   so it sets the part in with its rib facing a plain wall. Does the pocket refuse it, i.e. does
   the part rest on its rib, raised and tilted, so that the machine does not count it as seated?

The pocket is Phase 0's (2 mm clearance per side, 15 mm deep, 45 deg lead-in) with the slot cut
for a 10 x 8 x 12 mm rib with 2 mm clearance. Every condition runs the same trials (the same draws
per trial index). Physical outcome, besides the machine's verdict: the gap under the part and its
tilt; contacts between the part and the pocket (peak force) and between the gripper and the
machine.

Usage:
  uv run python scripts/probe_keyed_insertion.py               # 4 orientations x 4 errors x 30
  uv run python scripts/probe_keyed_insertion.py --n 2 --jobs 2 --out-dir /tmp/probe  # smoke
Writes <out-dir>/keyed_insertion_probe.json (per condition, provenance) and _trials.csv.
"""

import argparse
import csv
import json
import platform
import subprocess
import time
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

from polypanda.control.scripted_tending import ScriptedTending, true_belief
from polypanda.evaluation.reach import wilson_interval
from polypanda.simulation.line import Line, LineConfig, MachineSpot
from polypanda.simulation.machine import MachineConfig, rib_geom
from polypanda.simulation.workcell import WorkcellConfig, _yaw_quat

SEED = 20261123
STOP, MACHINE_ANGLE, PART_ANGLE = -0.6, -90.0, -30.0     # m, deg, deg (as evaluation/tending.py)
JITTER_XY = 0.015                                        # m either way, as the infeed's jitter
WRONG = (0, 90, 180, 270)                                # deg the believed slot is turned by


def build() -> Line:
    """The line (without the visual-only dressing) with keyed parts and a keyed M1."""
    cfg = MachineConfig(keyed=True, cycle_s=1e6)
    return Line.build_line(LineConfig(cell=WorkcellConfig(tower_angles=(-40.0,), dressing=False),
                                      machines=(MachineSpot("M1", STOP, MACHINE_ANGLE, cfg),),
                                      keyed_parts=True))


class ContactMonitor:
    """Called every control step: the part against the pocket (peak force), and the gripper
    against the machine's body or pocket."""

    def __init__(self, line: Line, part) -> None:
        m = line.model
        self.line = line
        machine = {m.body("M1").id} | {m.body(f"M1_pocket{k}").id for k in range(4)}
        self.pocket = {g for g in range(m.ngeom) if m.geom_bodyid[g] in machine}
        self.part = {part.geom, rib_geom(m, part)}
        self.gripper = {m.body(b).id for b in ("hand", "left_finger", "right_finger")}
        self.reset()

    def reset(self) -> None:
        self.peak_force, self.gripper_touch, self.rib_touch = 0.0, False, False

    def __call__(self) -> None:
        self.line.update_machines()
        m, d = self.line.model, self.line.data
        f6 = np.zeros(6)
        for i in range(d.ncon):
            con = d.contact[i]
            if con.geom1 in self.pocket:
                other = con.geom2
            elif con.geom2 in self.pocket:
                other = con.geom1
            else:
                continue
            if other in self.part:
                mujoco.mj_contactForce(m, d, i, f6)
                self.peak_force = max(self.peak_force, float(f6[0]))
            elif m.geom_bodyid[other] in self.gripper:
                self.gripper_touch = True


def run_trial(line: Line, wrong: int, error_mm: float, trial: int) -> dict:
    """Pick a keyed part at the machine's stop and load it with the script believing the slot is
    `wrong` deg round from where it is, aimed `error_mm` off the pocket's centre."""
    m, d = line.model, line.data
    rng = np.random.default_rng([SEED, trial])             # the same draws in every condition
    line.reset_line()
    m1 = line.machines["M1"]
    d.qpos[line.rail.joint_qpos] = STOP                    # the robot already at M1's stop
    d.ctrl[line.rail.actuator] = STOP
    d.qpos[m1.lid_qpos] = m1.cfg.travel - ScriptedTending.open_margin   # the lid open
    part = line.cubes[0]
    spot = line.line.station_xy(STOP, PART_ANGLE) + rng.uniform(-JITTER_XY, JITTER_XY, 2)
    part_yaw = rng.uniform(-np.pi, np.pi)
    direction = rng.uniform(0.0, 2 * np.pi)
    d.qpos[part.qpos:part.qpos + 7] = [*spot, part.half_size[2], *_yaw_quat(part_yaw)]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    truth = true_belief(line, "M1")
    belief = replace(truth, origin=truth.origin + error_mm / 1000 * np.array(
        [np.cos(direction), np.sin(direction), 0.0]),
        config=replace(truth.config, key_wall=(truth.config.key_wall + wrong // 90) % 4))
    ctl = ScriptedTending(line, beliefs={"M1": belief})
    monitor = ContactMonitor(line, part)
    ctl.on_step = monitor
    row = {"wrong_deg": wrong, "error_mm": error_mm, "trial": trial,
           "part_yaw_deg": float(np.degrees(part_yaw)), "outcome": "pick_failed"}
    try:
        ctl.step(10)
        ctl.swing_to(np.radians(PART_ANGLE))
        ctl.pick_for(part, "M1")
        if not line.attached(part):
            return row
        ctl.swing_to(np.radians(MACHINE_ANGLE))
        turn = belief.yaw + belief.config.key_yaw - ctl.rib_yaw(part)
        row["load_turn_deg"] = float(np.degrees((turn + np.pi) % (2 * np.pi) - np.pi))
        monitor.reset()
        ctl.load(part, "M1")
    except RuntimeError as e:          # a crashed trial is a failed trial, recorded
        return row | {"outcome": "error", "error": repr(e)[:200]}
    ctl.step(20)                                           # 1 s to settle
    near, seated = m1.in_pocket(part)
    R_m = d.xmat[m1.body].reshape(3, 3)
    rel = R_m.T @ (d.xpos[part.body] - d.xpos[m1.body])
    tilt = float(np.degrees(np.arccos(np.clip(d.xmat[part.body][8], -1.0, 1.0))))
    gap = float(rel[2] - part.half_size[2])
    # the physical seat alone, without the machine's orientation check
    down = bool(gap < 0.0015 and tilt < 3.0 and np.abs(rel[:2]).max() <= m1.cfg.clearance + 0.0005)
    before = d.xpos[part.body].copy()
    ctl.step(100)                                          # 5 s more: does it creep down?
    late = R_m.T @ (d.xpos[part.body] - d.xpos[m1.body])
    row |= {"outcome": "seated" if seated else "refused" if near else "outside",
            "seated": seated, "physically_down": down, "key_error_deg": m1.key_error(part),
            "offset_mm": float(np.abs(rel[:2]).max() * 1000),
            "gap_mm": gap * 1000, "tilt_deg": tilt,
            "gap_after_5s_mm": float((late[2] - part.half_size[2]) * 1000),
            "seated_after_5s": m1.in_pocket(part)[1],
            "drift_mm": float(np.linalg.norm(d.xpos[part.body] - before) * 1000),
            "peak_force_n": monitor.peak_force, "gripper_touched_machine": monitor.gripper_touch,
            "regrasps": ctl.regrasps, "ik_failures": ctl.ik_failures}
    return row


def run_condition(job: tuple[int, float, int]) -> list[dict]:
    wrong, error_mm, n = job
    line = build()
    return [run_trial(line, wrong, error_mm, k) for k in range(n)]


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    placed = [r for r in rows if r["outcome"] not in ("pick_failed", "error")]
    seated = [r for r in placed if r["seated"]]
    lo, hi = wilson_interval(len(seated), n)

    def stats(key: str, rs: list[dict]) -> dict | None:
        if not rs:
            return None
        v = np.array([abs(r[key]) for r in rs], dtype=float)
        return {"median": round(float(np.median(v)), 3), "min": round(float(v.min()), 3),
                "p95": round(float(np.percentile(v, 95)), 3), "max": round(float(v.max()), 3)}

    refused = [r for r in placed if not r["seated"]]
    return {"n": n, "seated": len(seated), "rate": len(seated) / n, "ci95": [lo, hi],
            "outcomes": dict(Counter(r["outcome"] for r in rows)),
            "physically_down": sum(r["physically_down"] for r in placed),
            "seated_after_5s": sum(r["seated_after_5s"] for r in placed),
            "refused_gap_after_5s_mm": stats("gap_after_5s_mm", refused),
            "gripper_touched_machine": sum(r["gripper_touched_machine"] for r in placed),
            "regrasps": sum(r["regrasps"] for r in placed),
            "ik_failures": sum(r["ik_failures"] for r in placed),
            "peak_force_n": stats("peak_force_n", placed),
            "seated_key_error_deg": stats("key_error_deg", seated),
            "seated_offset_mm": stats("offset_mm", seated),
            "seated_drift_mm": stats("drift_mm", seated),
            "refused_gap_mm": stats("gap_mm", refused),
            "refused_tilt_deg": stats("tilt_deg", refused)}


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=30, help="trials per condition")
    parser.add_argument("--errors", type=float, nargs="+", default=[0, 1, 2, 3],
                        help="aiming errors, mm")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=Path("results/v4"))
    args = parser.parse_args()

    git = git_state()   # recorded before running, so later commits cannot mislabel the result
    jobs = [(w, float(e), args.n) for w in WRONG for e in args.errors]
    t0 = time.perf_counter()
    with Pool(args.jobs) as pool:
        results = pool.map(run_condition, jobs, chunksize=1)
    rows = [r for rs in results for r in rs]
    conditions: dict = {}
    for (wrong, error, _), rs in zip(jobs, results, strict=True):
        tag = "right" if wrong == 0 else f"wrong{wrong}"
        conditions.setdefault(tag, {})[f"{error:g}"] = summarize(rs)
    mc, rib = MachineConfig(keyed=True), MachineConfig().rib
    record = {"name": "keyed_insertion_probe",
              "what": "scripted load of a keyed part into the keyed pocket, right way round and "
                      "each wrong way round",
              "seed": SEED, "n_per_condition": args.n, "errors_mm": args.errors,
              "pocket": {"clearance_mm": 1000 * mc.clearance, "depth_mm": 1000 * mc.pocket_depth,
                         "straight_mm": 1000 * mc.pocket_straight,
                         "lead_in_deg": mc.lead_in_deg, "key_wall": mc.key_wall,
                         "slot_width_mm": 1000 * (rib.width + 2 * mc.key_clearance),
                         "slot_depth_mm": 1000 * (rib.proud + mc.key_clearance)},
              "rib_mm": {"width": 1000 * rib.width, "proud": 1000 * rib.proud,
                         "height": 1000 * rib.height},
              "seated_means": "Machine.in_pocket: gap under 1.5 mm, tilt under 3 deg, centre "
                              "within clearance + 0.5 mm, and the rib within "
                              f"{mc.key_tolerance:g} deg of the slot",
              "conditions": conditions,
              "created_utc": datetime.now(UTC).isoformat(timespec="seconds"), "git": git,
              "versions": {"python": platform.python_version(), "mujoco": mujoco.__version__,
                           "numpy": np.__version__},
              "wall_s": round(time.perf_counter() - t0, 1)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "keyed_insertion_probe.json").write_text(json.dumps(record, indent=2) + "\n")
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.out_dir / "keyed_insertion_probe_trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"{'orientation':12s} {'error':>5s}  seated        gap mm (refused, min/med)  "
          f"tilt deg (min/med)  peak N (med/max)  gripper  regrasps")
    for tag, by_error in conditions.items():
        for error, s in by_error.items():
            g, t, f = s["refused_gap_mm"], s["refused_tilt_deg"], s["peak_force_n"]
            gap = f"{g['min']:5.1f} / {g['median']:5.1f}" if g else "    -        "
            tilt = f"{t['min']:5.1f} / {t['median']:5.1f}" if t else "    -        "
            force = f"{f['median']:6.1f} / {f['max']:6.1f}" if f else "-"
            print(f"{tag:12s} {error:>4s}mm  {s['seated']:3d}/{s['n']:<3d} {100 * s['rate']:5.1f}%"
                  f"   {gap}             {tilt}      {force}    {s['gripper_touched_machine']:4d}"
                  f"     {s['regrasps']:4d}   down {s['physically_down']}, "
                  f"seated after 5 s {s['seated_after_5s']}")
    print(f"wrote {args.out_dir}/keyed_insertion_probe.json and _trials.csv "
          f"({record['wall_s']} s)")


if __name__ == "__main__":
    main()
