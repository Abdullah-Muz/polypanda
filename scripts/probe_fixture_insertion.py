"""v4 Phase 0 probe: can this simulator seat a held cube in a tight pocket? (docs/v4_plan.md)

A feasibility measurement made before any v4 design is fixed; not a skill evaluation and not a
target. A machine fixture is a pocket the part must be set into with a millimetre or two of
clearance. This probe adds such a pocket to the cell at tower B's station and has the scripted
controller pick an upright cube from the bench (the infeed's jitter: +-1.5 cm, +-15 deg), carry
it over and set it into the pocket with `ScriptedCell.place`, unchanged. It answers:

1. Does the contact behave? Seating rate with exact positions, peak wall forces, penetration,
   and whether a seated cube stays still.
2. What does a position error cost the script? The place is aimed at the pocket's centre plus
   an error of 0-6 mm in a random direction, as a camera estimate would be off. Up to the
   clearance the cube still fits; beyond it only the lead-in and the arm's give can help.
3. Can the seated cube be taken out again with the scripted pick (fingers above the rim)?

Pocket: square, 40 mm + 2 x clearance wide, 15 mm deep: 10 mm of straight wall, then a 45 deg
lead-in 5 mm high and 5 mm wide. Two wall frictions: "cube" (the walls' own coefficient is never
used: MuJoCo takes the larger of the two, the cube's 1.0) and "steel" (0.3, enforced with geom
priority). Every condition runs the same trials (the same draws per trial index).

Usage:
  uv run python scripts/probe_fixture_insertion.py                # 4 pockets x 6 errors x 50
  uv run python scripts/probe_fixture_insertion.py --n 5 --jobs 4 --out-dir /tmp/probe  # smoke
Writes <out-dir>/fixture_insertion_probe.json (per condition, provenance) and _trials.csv.
"""

import argparse
import csv
import json
import platform
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

from polypanda.control.scripted_cell import ScriptedCell
from polypanda.evaluation.cell_picks import wrap45
from polypanda.evaluation.reach import wilson_interval
from polypanda.simulation.scenes import CubeIndices
from polypanda.simulation.workcell import (
    CUBE_NAME,
    Conveyor,
    Workcell,
    WorkcellConfig,
    WorkOrder,
    _yaw_quat,
    build_workcell_spec,
    station_xy,
)

SEED = 20261122
POCKET_ANGLE = -90.0          # deg about the base axis: tower B's station
SOURCE_ANGLE = 10.0           # deg: buffer slot 1, where the cube is picked from
CUBE_HALF = 0.02              # m, half the cube's 4 cm side
DEPTH, STRAIGHT, WALL = 0.015, 0.010, 0.010   # m: rim height, straight part, wall thickness
LEAD_IN_DEG = 45.0            # the lead-in's slope from the horizontal
STEEL_FRICTION = 0.3
JITTER_XY, JITTER_YAW = 0.015, 15.0            # m and deg either way, as the infeed's jitter
SEATED_GAP, SEATED_TILT, SEATED_SLACK = 0.0015, 3.0, 0.0005   # m, deg, m


@dataclass(frozen=True)
class Pocket:
    clearance: float          # m per side
    wall_friction: str        # "cube" (max rule: 1.0) or "steel" (0.3, by priority)

    @property
    def tag(self) -> str:
        return f"clearance{self.clearance * 1000:g}mm_{self.wall_friction}"


def add_pocket(spec: mujoco.MjSpec, cfg: WorkcellConfig, pocket: Pocket,
               angle_deg: float = POCKET_ANGLE) -> None:
    """Four walls around a square opening on the bench at a station, square to the station's
    direction: a straight part, a sloped lead-in above it and a block behind the lead-in."""
    a = CUBE_HALF + pocket.clearance                 # half the opening
    theta = np.radians(LEAD_IN_DEG)
    rise = DEPTH - STRAIGHT
    run = rise / np.tan(theta)                       # how far the lead-in widens the opening
    length, half_thick = rise / np.sin(theta), 0.003
    along = np.array([np.cos(theta), 0.0, np.sin(theta)])     # up the slope
    normal = np.array([-np.sin(theta), 0.0, np.cos(theta)])   # out of the slope, into the opening
    lead_pos = np.array([a, 0.0, STRAIGHT]) + along * length / 2 - normal * half_thick
    lead_quat = [np.cos(theta / 2), 0.0, -np.sin(theta / 2), 0.0]   # x axis onto `along`
    extra: dict = {"solref": [0.01, 1.0], "rgba": [0.55, 0.57, 0.6, 1.0]}
    if pocket.wall_friction == "steel":
        extra |= {"friction": [STEEL_FRICTION, 0.005, 0.0001], "priority": 1}
    body = spec.worldbody.add_body(name="pocket", pos=[*station_xy(angle_deg, cfg.radius), 0.0],
                                   quat=_yaw_quat(np.radians(angle_deg)))
    box = mujoco.mjtGeom.mjGEOM_BOX
    for k in range(4):
        wall = body.add_body(name=f"pocket_wall{k}", quat=_yaw_quat(k * np.pi / 2))
        wall.add_geom(name=f"pocket_wall{k}", type=box, size=[WALL / 2, a + WALL, STRAIGHT / 2],
                      pos=[a + WALL / 2, 0.0, STRAIGHT / 2], **extra)
        wall.add_geom(name=f"pocket_lead{k}", type=box, size=[length / 2, a + WALL, half_thick],
                      pos=lead_pos.tolist(), quat=lead_quat, **extra)
        wall.add_geom(name=f"pocket_back{k}", type=box,
                      size=[(WALL - run) / 2, a + WALL, rise / 2],
                      pos=[a + run + (WALL - run) / 2, 0.0, STRAIGHT + rise / 2], **extra)


def build_cell(pocket: Pocket) -> Workcell:
    """The cell (without the visual-only factory hall) with a pocket at tower B's station."""
    cfg = WorkcellConfig(dressing=False)
    spec = build_workcell_spec(cfg)
    add_pocket(spec, cfg, pocket)
    model = spec.compile()
    cell = Workcell(model, mujoco.MjData(model), cfg)
    cell.cubes = [CubeIndices.from_model(model, CUBE_NAME.format(k)) for k in range(cfg.n_cubes)]
    cell.conveyor = Conveyor(cell)
    return cell


def pocket_frame(cell: Workcell, cube: CubeIndices) -> tuple[np.ndarray, float, float, float]:
    """The cube in the pocket's frame: xy offset from the centre (m), the gap under its bottom
    (m, as if upright), its tilt (deg) and its yaw (deg, wrapped to +-45)."""
    d = cell.data
    p, R = d.xpos[cube.body], d.xmat[cube.body].reshape(3, 3)
    a = np.radians(POCKET_ANGLE)
    rel = p[:2] - station_xy(POCKET_ANGLE, cell.config.radius)
    local = np.array([np.cos(a) * rel[0] + np.sin(a) * rel[1],
                      -np.sin(a) * rel[0] + np.cos(a) * rel[1]])
    tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))
    yaw = wrap45(float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))) - POCKET_ANGLE)
    return local, float(p[2] - cube.half_size[2]), tilt, yaw


def is_seated(local: np.ndarray, gap: float, tilt: float, clearance: float) -> bool:
    """On the pocket's floor, upright, inside the opening."""
    return bool(gap < SEATED_GAP and tilt < SEATED_TILT
                and np.abs(local).max() <= clearance + SEATED_SLACK)


class ContactMonitor:
    """Called every control step: what touches the pocket, and how hard."""

    def __init__(self, cell: Workcell, cube: CubeIndices) -> None:
        m = cell.model
        self.cell, self.cube = cell, cube
        self.pocket = {g for g in range(m.ngeom)
                       if m.body(m.geom_bodyid[g]).name.startswith("pocket")}
        self.gripper = {m.body(b).id for b in ("hand", "left_finger", "right_finger")}
        self.reset()

    def reset(self) -> None:
        self.peak_force, self.peak_depth, self.contact_steps = 0.0, 0.0, 0
        self.gripper_touch = False

    def __call__(self) -> None:
        m, d = self.cell.model, self.cell.data
        f6, touching = np.zeros(6), False
        for i in range(d.ncon):
            con = d.contact[i]
            if con.geom1 in self.pocket:
                other = con.geom2
            elif con.geom2 in self.pocket:
                other = con.geom1
            else:
                continue
            if other == self.cube.geom:
                mujoco.mj_contactForce(m, d, i, f6)
                self.peak_force = max(self.peak_force, float(f6[0]))
                self.peak_depth = max(self.peak_depth, float(-con.dist))
                touching = True
            elif m.geom_bodyid[other] in self.gripper:
                self.gripper_touch = True
        self.contact_steps += touching


def run_trial(cell: Workcell, pocket: Pocket, error_mm: float, trial: int) -> dict:
    """Pick an upright cube from the bench, carry it to the pocket and place it, aimed at the
    pocket's centre plus `error_mm` in a random direction; then try to take it out again."""
    cfg, m, d = cell.config, cell.model, cell.data
    rng = np.random.default_rng([SEED, trial])            # the same draws in every condition
    cell.reset(WorkOrder(recipes=((), (), ()), arrivals=()))
    cube = cell.cubes[0]
    spot = station_xy(SOURCE_ANGLE, cfg.radius) + rng.uniform(-JITTER_XY, JITTER_XY, 2)
    yaw = np.arctan2(spot[1], spot[0]) + np.radians(rng.uniform(-JITTER_YAW, JITTER_YAW))
    direction = rng.uniform(0.0, 2 * np.pi)
    d.qpos[cube.qpos:cube.qpos + 7] = [*spot, cube.half_size[2], *_yaw_quat(yaw)]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    monitor = ContactMonitor(cell, cube)
    ctl = ScriptedCell(cell, on_step=monitor)
    row = {"pocket": pocket.tag, "clearance_mm": pocket.clearance * 1000,
           "wall_friction": pocket.wall_friction, "error_mm": error_mm, "trial": trial,
           "outcome": "pick_failed"}
    ctl.step(10)
    ctl.pick(cube)
    if not cell.attached(cube):
        return row
    ctl.swing_to(np.radians(POCKET_ANGLE))
    target = (station_xy(POCKET_ANGLE, cfg.radius)
              + error_mm / 1000 * np.array([np.cos(direction), np.sin(direction)]))
    tcp = d.site_xpos[ctl.arm.tcp_site]
    row["grip_offset_mm"] = float(np.linalg.norm(d.xpos[cube.body][:2] - tcp[:2]) * 1000)
    row["yaw_before_deg"] = pocket_frame(cell, cube)[3]
    monitor.reset()
    ctl.place(cube, target, level=0)
    ctl.step(20)                                          # 1 s to settle
    local, gap, tilt, yaw_after = pocket_frame(cell, cube)
    before = d.xpos[cube.body].copy()
    ctl.step(20)
    seated = is_seated(local, gap, tilt, pocket.clearance)
    row |= {"outcome": ("seated" if seated else
                        "on_rim" if np.abs(local).max() < 0.05 else "outside"),
            "offset_mm": float(np.abs(local).max() * 1000), "gap_mm": gap * 1000,
            "tilt_deg": tilt, "yaw_after_deg": yaw_after,
            "drift_mm": float(np.linalg.norm(d.xpos[cube.body] - before) * 1000),
            "peak_force_n": monitor.peak_force, "peak_depth_mm": monitor.peak_depth * 1000,
            "wall_contact_steps": monitor.contact_steps,
            "gripper_touched_rim": monitor.gripper_touch,
            "ik_failures": ctl.ik_failures}
    if seated:                                            # take it out again
        monitor.reset()
        ctl.pick(cube)
        row |= {"unload_ok": bool(cell.attached(cube) and d.xpos[cube.body][2] > 0.15),
                "unload_gripper_touched_rim": monitor.gripper_touch}
    return row


def run_condition(job: tuple[Pocket, float, int]) -> list[dict]:
    pocket, error_mm, n = job
    cell = build_cell(pocket)
    return [run_trial(cell, pocket, error_mm, k) for k in range(n)]


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    placed = [r for r in rows if r["outcome"] != "pick_failed"]
    seated = [r for r in placed if r["outcome"] == "seated"]
    lo, hi = wilson_interval(len(seated), n)

    def stats(key: str, rs: list[dict]) -> dict | None:
        if not rs:
            return None
        v = np.array([r[key] for r in rs], dtype=float)
        return {"median": round(float(np.median(v)), 3),
                "p95": round(float(np.percentile(v, 95)), 3), "max": round(float(v.max()), 3)}

    return {"n": n, "seated": len(seated), "rate": len(seated) / n, "ci95": [lo, hi],
            "outcomes": dict(Counter(r["outcome"] for r in rows)),
            "touched_walls": sum(r["wall_contact_steps"] > 0 for r in placed),
            "gripper_touched_rim": sum(r["gripper_touched_rim"] for r in placed),
            "peak_force_n": stats("peak_force_n", placed),
            "peak_depth_mm": stats("peak_depth_mm", placed),
            "seated_offset_mm": stats("offset_mm", seated),
            "seated_drift_mm": stats("drift_mm", seated),
            "yaw_before_abs_deg": stats("yaw_before_deg", [
                {**r, "yaw_before_deg": abs(r["yaw_before_deg"])} for r in placed]),
            "unload_ok": sum(r.get("unload_ok", False) for r in seated),
            "unload_gripper_touched_rim": sum(r.get("unload_gripper_touched_rim", False)
                                              for r in seated)}


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=50, help="trials per condition")
    parser.add_argument("--errors", type=float, nargs="+", default=[0, 1, 2, 3, 4, 6],
                        help="position errors, mm")
    parser.add_argument("--clearances", type=float, nargs="+", default=[2, 1],
                        help="pocket clearances per side, mm")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=Path("results/v4"))
    args = parser.parse_args()

    git = git_state()   # recorded before running, so later commits cannot mislabel the result
    pockets = [Pocket(c / 1000, f) for c in args.clearances for f in ("cube", "steel")]
    jobs = [(p, float(e), args.n) for p in pockets for e in args.errors]
    t0 = time.perf_counter()
    with Pool(args.jobs) as pool:
        results = pool.map(run_condition, jobs, chunksize=1)
    rows = [r for rs in results for r in rs]
    conditions: dict = {}
    for (pocket, error, _), rs in zip(jobs, results, strict=True):
        conditions.setdefault(pocket.tag, {})[f"{error:g}"] = summarize(rs)
    record = {"name": "fixture_insertion_probe",
              "what": "v4 Phase 0 feasibility probe: scripted place of a held cube into a pocket",
              "seed": SEED, "n_per_condition": args.n, "errors_mm": args.errors,
              "pocket": {"cube_side_mm": 2000 * CUBE_HALF, "depth_mm": 1000 * DEPTH,
                         "straight_mm": 1000 * STRAIGHT, "lead_in_deg": LEAD_IN_DEG,
                         "wall_mm": 1000 * WALL, "steel_friction": STEEL_FRICTION,
                         "station_deg": POCKET_ANGLE},
              "seated_means": {"gap_under_mm": 1000 * SEATED_GAP, "tilt_under_deg": SEATED_TILT,
                               "slack_mm": 1000 * SEATED_SLACK},
              "conditions": conditions,
              "created_utc": datetime.now(UTC).isoformat(timespec="seconds"), "git": git,
              "versions": {"python": platform.python_version(), "mujoco": mujoco.__version__,
                           "numpy": np.__version__},
              "wall_s": round(time.perf_counter() - t0, 1)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "fixture_insertion_probe.json").write_text(json.dumps(record, indent=2) + "\n")
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.out_dir / "fixture_insertion_probe_trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"{'pocket':26s} {'error':>5s}  seated        touched  peak N (med/max)  "
          f"depth mm (max)  gripper on rim  unload")
    for tag, by_error in conditions.items():
        for error, s in by_error.items():
            f, p = s["peak_force_n"], s["peak_depth_mm"]
            print(f"{tag:26s} {error:>4s}mm  {s['seated']:3d}/{s['n']:<3d} "
                  f"{100 * s['rate']:5.1f}%  "
                  f"{s['touched_walls']:5d}    {f['median']:6.1f} / {f['max']:6.1f}   "
                  f"{p['max']:8.3f}       {s['gripper_touched_rim']:5d}       "
                  f"{s['unload_ok']}/{s['seated']}"
                  + (f" ({s['unload_gripper_touched_rim']} touched rim)"
                     if s["unload_gripper_touched_rim"] else ""))
    print(f"wrote {args.out_dir}/fixture_insertion_probe.json and _trials.csv "
          f"({record['wall_s']} s)")


if __name__ == "__main__":
    main()
