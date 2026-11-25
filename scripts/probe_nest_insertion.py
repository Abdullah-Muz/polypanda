"""v4.1 Phase 0 probe: does a scripted straight-down insertion seat a part in a TIGHT test nest?

A feasibility measurement made before v4.1's learned insertion is designed; not a skill
evaluation and not a target. v4.0's Phase 0 (scripts/probe_fixture_insertion.py) found the
scripted place seats a cube in a pocket with 1-2 mm clearance and a 5 mm 45-degree lead-in up to
4 mm of aiming error, so a learned place had nothing to win there. An electronics test nest is
tighter: a few tenths of a millimetre around the part and only a small edge chamfer. This probe
asks whether that changes the answer, and whether the simulator's contacts can be trusted at that
scale:

1. Contact validity: how deep do the part and the nest walls interpenetrate (peak depth)? If the
   depth is comparable with the clearance, sub-millimetre results in this simulator are not
   trustworthy.
2. What error does a straight-down insertion tolerate? The nest is rotated by 0-2 degrees from
   where the script believes it is (a camera's yaw error and fixture variation) and the place is
   aimed with a 0-3 mm position error in a random direction.

The part is the cell's 4 cm cube (held by the cell's grasp assist, a rigid weld: no give in the
fingers, which makes jams harsher than with a real compliant grasp). The nest: a square opening,
40 mm + 2 x clearance wide, 10 mm deep, whose top edge has a 45-degree chamfer of `chamfer` mm.
The scripted cell picks the cube from the bench and sets it in with the unchanged
`ScriptedCell.place`. Every condition runs the same trials (the same draws per trial index).

Usage:
  uv run python scripts/probe_nest_insertion.py                  # 3 x 2 x 4 x 3 conditions x 20
  uv run python scripts/probe_nest_insertion.py --n 3 --jobs 4 --out-dir /tmp/probe   # smoke
Writes <out-dir>/nest_insertion_probe.json (per condition, provenance) and _trials.csv.
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

SEED = 20261124
NEST_ANGLE = -90.0            # deg about the base axis: tower B's station (where it is believed)
SOURCE_ANGLE = 10.0           # deg: buffer slot 1, where the cube is picked from
CUBE_HALF = 0.02              # m, half the cube's 4 cm side
DEPTH, WALL = 0.010, 0.010    # m: the nest's depth (rim height), wall thickness
JITTER_XY, JITTER_YAW = 0.015, 15.0            # m and deg either way, as the infeed's jitter
SEATED_GAP, SEATED_TILT, SEATED_SLACK = 0.001, 2.0, 0.0002   # m, deg, m


# Contact settings for the part and the nest: the cell's default, and the stiffest stable one at
# the 2 ms timestep (time constant >= 2 timesteps) with a narrower, harder impedance zone.
CONTACTS = {"default": {"solref": [0.01, 1.0], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0]},
            "stiff": {"solref": [0.004, 1.0], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0]}}


@dataclass(frozen=True)
class Nest:
    clearance: float          # m per side
    chamfer: float            # m: the 45-degree chamfer on the rim's inner edge
    yaw_error: float          # deg: the nest is turned this much from where it is believed
    contact: str = "default"  # CONTACTS key, for the part and the nest

    @property
    def tag(self) -> str:
        return (f"c{self.clearance * 1000:g}mm_ch{self.chamfer * 1000:g}mm_"
                f"yaw{self.yaw_error:g}deg_{self.contact}")

    @property
    def angle(self) -> float:
        return NEST_ANGLE + self.yaw_error


def add_nest(spec: mujoco.MjSpec, cfg: WorkcellConfig, nest: Nest) -> None:
    """Four walls around a square opening on the bench, each a straight part under a 45-degree
    chamfer of `nest.chamfer` (and a block behind the chamfer)."""
    a = CUBE_HALF + nest.clearance                   # half the opening
    rise = run = nest.chamfer                        # 45 degrees
    straight = DEPTH - rise
    theta = np.radians(45.0)
    length, half_thick = rise / np.sin(theta), 0.0005
    along = np.array([np.cos(theta), 0.0, np.sin(theta)])
    normal = np.array([-np.sin(theta), 0.0, np.cos(theta)])
    lead_pos = np.array([a, 0.0, straight]) + along * length / 2 - normal * half_thick
    lead_quat = [np.cos(theta / 2), 0.0, -np.sin(theta / 2), 0.0]
    extra: dict = {**CONTACTS[nest.contact], "rgba": [0.55, 0.57, 0.6, 1.0]}
    body = spec.worldbody.add_body(name="nest", pos=[*station_xy(NEST_ANGLE, cfg.radius), 0.0],
                                   quat=_yaw_quat(np.radians(nest.angle)))
    box = mujoco.mjtGeom.mjGEOM_BOX
    for k in range(4):
        wall = body.add_body(name=f"nest_wall{k}", quat=_yaw_quat(k * np.pi / 2))
        wall.add_geom(name=f"nest_wall{k}", type=box, size=[WALL / 2, a + WALL, straight / 2],
                      pos=[a + WALL / 2, 0.0, straight / 2], **extra)
        if rise > 0:
            wall.add_geom(name=f"nest_chamfer{k}", type=box,
                          size=[length / 2, a + WALL, half_thick], pos=lead_pos.tolist(),
                          quat=lead_quat, **extra)
            wall.add_geom(name=f"nest_back{k}", type=box, size=[(WALL - run) / 2, a + WALL,
                                                                 rise / 2],
                          pos=[a + run + (WALL - run) / 2, 0.0, straight + rise / 2], **extra)


def build_cell(nest: Nest) -> Workcell:
    cfg = WorkcellConfig(dressing=False)
    spec = build_workcell_spec(cfg)
    add_nest(spec, cfg, nest)
    for k in range(cfg.n_cubes):                     # the parts get the same contact setting
        g = spec.geom(CUBE_NAME.format(k))
        g.solref, g.solimp = CONTACTS[nest.contact]["solref"], CONTACTS[nest.contact]["solimp"]
    model = spec.compile()
    cell = Workcell(model, mujoco.MjData(model), cfg)
    cell.cubes = [CubeIndices.from_model(model, CUBE_NAME.format(k)) for k in range(cfg.n_cubes)]
    cell.conveyor = Conveyor(cell)
    return cell


def nest_frame(cell: Workcell, cube: CubeIndices, nest: Nest):
    """The cube in the nest's (true) frame: xy offset from the centre (m), the gap under its
    bottom (m), its tilt (deg) and its yaw relative to the nest (deg, wrapped to +-45)."""
    d = cell.data
    p, R = d.xpos[cube.body], d.xmat[cube.body].reshape(3, 3)
    a = np.radians(nest.angle)
    rel = p[:2] - station_xy(NEST_ANGLE, cell.config.radius)
    local = np.array([np.cos(a) * rel[0] + np.sin(a) * rel[1],
                      -np.sin(a) * rel[0] + np.cos(a) * rel[1]])
    tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))
    yaw = wrap45(float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))) - nest.angle)
    return local, float(p[2] - cube.half_size[2]), tilt, yaw


class ContactMonitor:
    """Called every control step: what touches the nest, how hard and how deep."""

    def __init__(self, cell: Workcell, cube: CubeIndices) -> None:
        m = cell.model
        self.cell, self.cube = cell, cube
        self.nest = {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("nest")}
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
            if con.geom1 in self.nest:
                other = con.geom2
            elif con.geom2 in self.nest:
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


def run_trial(cell: Workcell, nest: Nest, error_mm: float, trial: int) -> dict:
    cfg, m, d = cell.config, cell.model, cell.data
    rng = np.random.default_rng([SEED, trial])
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
    row = {"nest": nest.tag, "clearance_mm": nest.clearance * 1000,
           "chamfer_mm": nest.chamfer * 1000, "yaw_error_deg": nest.yaw_error,
           "error_mm": error_mm, "trial": trial, "outcome": "pick_failed"}
    ctl.step(10)
    ctl.pick(cube)
    if not cell.attached(cube):
        return row
    ctl.swing_to(np.radians(NEST_ANGLE))
    target = (station_xy(NEST_ANGLE, cfg.radius)
              + error_mm / 1000 * np.array([np.cos(direction), np.sin(direction)]))
    monitor.reset()
    ctl.place(cube, target, level=0)
    ctl.step(20)
    local, gap, tilt, yaw_after = nest_frame(cell, cube, nest)
    before = d.xpos[cube.body].copy()
    ctl.step(20)
    seated = bool(gap < SEATED_GAP and tilt < SEATED_TILT
                  and np.abs(local).max() <= nest.clearance + SEATED_SLACK)
    outcome = ("seated" if seated else
               "jammed_on_rim" if np.abs(local).max() < CUBE_HALF + nest.clearance + WALL
               else "outside")
    row |= {"outcome": outcome, "offset_mm": float(np.abs(local).max() * 1000),
            "gap_mm": gap * 1000, "tilt_deg": tilt, "yaw_after_deg": yaw_after,
            "drift_mm": float(np.linalg.norm(d.xpos[cube.body] - before) * 1000),
            "peak_force_n": monitor.peak_force, "peak_depth_mm": monitor.peak_depth * 1000,
            "wall_contact_steps": monitor.contact_steps,
            "gripper_touched_rim": monitor.gripper_touch, "ik_failures": ctl.ik_failures}
    return row


def run_condition(job) -> list[dict]:
    nest, error_mm, n = job
    cell = build_cell(nest)
    return [run_trial(cell, nest, error_mm, k) for k in range(n)]


def summarize(rows: list[dict]) -> dict:
    placed = [r for r in rows if r["outcome"] != "pick_failed"]
    k = sum(r["outcome"] == "seated" for r in placed)

    def stats(key, rs):
        v = np.array([r[key] for r in rs if key in r], dtype=float)
        if not len(v):
            return None
        return {"median": round(float(np.median(v)), 3),
                "p95": round(float(np.percentile(v, 95)), 3), "max": round(float(v.max()), 3)}
    lo, hi = wilson_interval(k, len(placed)) if placed else (0.0, 0.0)
    return {"n": len(rows), "placed": len(placed), "seated": k,
            "rate": k / len(placed) if placed else None, "ci95": [lo, hi],
            "outcomes": dict(Counter(r["outcome"] for r in rows)),
            "peak_force_n": stats("peak_force_n", placed),
            "peak_depth_mm": stats("peak_depth_mm", placed),
            "gap_mm": stats("gap_mm", placed), "tilt_deg": stats("tilt_deg", placed),
            "gripper_touched_rim": sum(bool(r.get("gripper_touched_rim")) for r in placed)}


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clearances", type=float, nargs="+", default=[0.1, 0.25, 0.5], help="mm")
    p.add_argument("--chamfers", type=float, nargs="+", default=[0.5, 1.0], help="mm")
    p.add_argument("--yaw-errors", type=float, nargs="+", default=[0.0, 1.0, 2.0], help="deg")
    p.add_argument("--errors", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0], help="mm")
    p.add_argument("--contacts", nargs="+", default=["default", "stiff"], choices=list(CONTACTS))
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--out-dir", type=Path, default=Path("results/v4"))
    args = p.parse_args()
    git = git_state()
    nests = [Nest(c / 1000, ch / 1000, y, ct) for ct in args.contacts for c in args.clearances
             for ch in args.chamfers for y in args.yaw_errors]
    jobs = [(nest, e, args.n) for nest in nests for e in args.errors]
    start = time.perf_counter()
    with Pool(args.jobs) as pool:
        rows = [r for rs in pool.map(run_condition, jobs) for r in rs]
    wall = time.perf_counter() - start
    conditions: dict = {}
    for nest in nests:
        conditions[nest.tag] = {f"{e:g}mm": summarize([r for r in rows if r["nest"] == nest.tag
                                                       and r["error_mm"] == e])
                                for e in args.errors}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "nest_insertion_probe.json"
    out.write_text(json.dumps({
        "name": "nest_insertion_probe", "what": __doc__.split("\n")[0], "seed": SEED,
        "n_per_condition": args.n, "errors_mm": args.errors, "depth_mm": DEPTH * 1000,
        "seated_means": {"gap_mm": SEATED_GAP * 1000, "tilt_deg": SEATED_TILT,
                         "offset_within_clearance_plus_mm": SEATED_SLACK * 1000},
        "conditions": conditions, "created_utc": datetime.now(UTC).isoformat(), "git": git,
        "versions": {"mujoco": mujoco.__version__, "python": platform.python_version()},
        "wall_s": round(wall, 1)}, indent=1))
    with open(args.out_dir / "nest_insertion_probe_trials.csv", "w", newline="") as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} trials, {wall:.0f} s)")
    for tag, by_err in conditions.items():
        def fmt(e, s):
            force = s["peak_force_n"]["median"] if s["peak_force_n"] else "-"
            depth = s["peak_depth_mm"]["p95"] if s["peak_depth_mm"] else "-"
            return f"{e}: {s['seated']}/{s['placed']} (F {force} N, depth p95 {depth} mm)"
        print(f"  {tag:36s} " + "  ".join(fmt(e, s) for e, s in by_err.items()))


if __name__ == "__main__":
    main()
