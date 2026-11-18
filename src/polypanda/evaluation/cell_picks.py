"""v3.2: cell-like pick checks for a grasp, run through the cell's own pick (docs/v3_2_targets.md).

The v3.1 grasp met its one-shot target (lying cubes in the grasp's own scene) and still knocked
over upright cubes in the cell, on one seed of three: cubes on the belt turned 5-15 deg one way,
and cubes standing near the base turned close to 45 deg (docs/recovery.md, v3.1). These sets
put one upright cube where the cell picks cubes, at a known rotation, in a fresh cell, and call
`LearnedCell.pick` (or `ScriptedCell.pick` for the scripted baseline). That is the same base
swing, station view, gates and grasp assist as in a work order, and takes about 2 s a case.

- belt: the cube stands on the belt at the pick point (lateral offset and yaw drawn as the
  infeed draws them, +-1.5 cm and +-15 deg), with 0-2 more cubes queued behind it as a batch
  leaves them, and partly built towers.
- floor: the cube stands on the table near the base, at one of the cell's staging spots' angles
  (where the stand-up sets cubes upright, and where they are picked again), radius
  0.31-0.40 m, turned uniformly -45..+45 deg relative to the hand, with partly built towers.

A case's `yaw_rel` is the cube's rotation relative to the hand's starting direction (the hand
faces the cube's centre from the base axis), wrapped to -45..+45 deg (the cube's cross-section
is square). `size` scales the test cube at constant density (CubePhysics); the sets themselves
hold nominal cubes, and the size condition is applied on top of a set when asked.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from polypanda.control.scripted_cell import CellControllerConfig, ScriptedCell
from polypanda.simulation.randomization import CubePhysics
from polypanda.simulation.workcell import (
    Workcell,
    WorkcellConfig,
    WorkOrder,
    _yaw_quat,
    belt_direction,
)

KINDS = ("belt", "floor")
BELT_LATERAL = 0.015          # m either way, the infeed's jitter (CellEvalConfig.lateral)
BELT_YAW = 15.0               # deg either way, the infeed's jitter
FLOOR_RADIUS = (0.31, 0.40)   # m from the base axis
TEST_CUBE = 0                 # the cube picked; the others are towers and queued belt cubes


@dataclass(frozen=True)
class PickCase:
    kind: str                 # "belt" or "floor"
    angle: float              # deg about the base axis, of the cube's nominal spot
    radius: float             # m from the base axis, of the cube's nominal spot
    lateral: float            # m, sideways offset across the belt (belt cases; 0 on the floor)
    yaw_rel: float            # deg, the cube's rotation relative to the hand, -45..+45
    queued: int               # cubes waiting on the belt behind it (belt cases)
    towers: tuple[int, int, int]  # cubes already stacked on towers A, B, C


@dataclass
class PickSet:
    name: str
    kind: str
    seed: int
    cases: list[PickCase]

    def __len__(self) -> int:
        return len(self.cases)

    @property
    def digest(self) -> str:
        raw = json.dumps([asdict(c) for c in self.cases], sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()[:12]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(
            {"name": self.name, "kind": self.kind, "seed": self.seed, "n": len(self),
             "digest": self.digest, "cases": [asdict(c) for c in self.cases]}, indent=1) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> PickSet:
        payload = json.loads(Path(path).read_text())
        cases = [PickCase(**{**c, "towers": tuple(c["towers"])}) for c in payload["cases"]]
        loaded = cls(payload["name"], payload["kind"], payload["seed"], cases)
        if loaded.digest != payload["digest"]:
            raise ValueError(f"{path}: case digest mismatch (file edited or corrupted)")
        return loaded


def generate_pick_set(name: str, kind: str, n: int, seed: int,
                      cell: WorkcellConfig | None = None,
                      staging: tuple[tuple[float, float], ...] | None = None) -> PickSet:
    """`n` cases from their own random stream (`seed`)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    cfg = cell or WorkcellConfig()
    angles = [a for _, a in (staging or CellControllerConfig().staging)]
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n):
        towers = tuple(int(h) for h in rng.integers(0, 4, size=3))
        if kind == "belt":
            cases.append(PickCase(
                kind, float(cfg.infeed_angle), float(cfg.radius),
                float(rng.uniform(-BELT_LATERAL, BELT_LATERAL)),
                float(rng.uniform(-BELT_YAW, BELT_YAW)), int(rng.integers(0, 3)), towers))
        else:
            cases.append(PickCase(
                kind, float(angles[int(rng.integers(len(angles)))]),
                float(rng.uniform(*FLOOR_RADIUS)), 0.0, float(rng.uniform(-45.0, 45.0)), 0,
                towers))
    return PickSet(name, kind, seed, cases)


def wrap45(deg: float) -> float:
    return (deg + 45.0) % 90.0 - 45.0


def stage_case(case: PickCase, size: float = 1.0, config: WorkcellConfig | None = None
               ) -> Workcell:
    """A fresh cell holding the case: robot at home, belt stopped, every cube at rest."""
    cfg = config or WorkcellConfig()
    cell = Workcell.build(cfg)
    cell.reset(WorkOrder(recipes=((), (), ()), arrivals=()))   # all cubes parked, belt empty
    m, d = cell.model, cell.data
    cube = cell.cubes[TEST_CUBE]
    if size != 1.0:
        CubePhysics(m, cube).apply(size_scale=size)
    spot = np.array([np.cos(np.radians(case.angle)), np.sin(np.radians(case.angle))]) * case.radius
    travel = belt_direction(cfg)
    normal = np.array([-travel[1], travel[0]])
    lift = cfg.belt_lift if case.kind == "belt" else 0.0
    xy = spot + case.lateral * normal
    facing = np.arctan2(xy[1], xy[0])               # the hand faces the cube's centre
    yaw = facing + np.radians(case.yaw_rel)
    d.qpos[cube.qpos:cube.qpos + 7] = [*xy, cube.half_size[2] + lift, *_yaw_quat(yaw)]
    spare = iter(cell.cubes[1:])
    heading = float(np.arctan2(travel[1], travel[0]))
    for k in range(case.queued):                    # queued behind it, as a batch leaves them
        c = next(spare)
        p = spot - (k + 1) * cfg.cube_spacing * travel
        d.qpos[c.qpos:c.qpos + 7] = [*p, c.half_size[2] + cfg.belt_lift, *_yaw_quat(heading)]
    for t, height in enumerate(case.towers):
        txy = cell.tower_xy(t)
        for level in range(height):
            c = next(spare)
            d.qpos[c.qpos:c.qpos + 7] = [*txy, c.half_size[2] * (2 * level + 1),
                                         *_yaw_quat(np.radians(cfg.tower_angles[t]))]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    return cell


def run_case(case: PickCase, make_controller, size: float = 1.0,
             config: WorkcellConfig | None = None) -> dict:
    """Stage the case and pick the test cube with `make_controller(cell)` (a LearnedCell or a
    ScriptedCell); the outcome, with the cube's rotation relative to the hand as measured when
    the pick starts."""
    cell = stage_case(case, size, config)
    ctl: ScriptedCell = make_controller(cell)
    ctl.source = "belt" if case.kind == "belt" else "loose"
    ctl.step(10)                                     # settle on the belt or table
    cube = cell.cubes[TEST_CUBE]
    p, R = cell.data.xpos[cube.body].copy(), cell.data.xmat[cube.body].reshape(3, 3)
    yaw_rel = wrap45(float(np.degrees(np.arctan2(R[1, 0], R[0, 0]) - np.arctan2(p[1], p[0]))))
    tilt0 = float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))
    learned = hasattr(ctl, "picks") and getattr(ctl, "grasp", None) is not None
    n_before = len(ctl.picks) if learned else 0
    ctl.pick(cube)
    if learned:
        r = ctl.picks[n_before]
        success, failure, steps = r["success"], r["failure"], r["steps"]
        knocked_others = r.get("loose_after", 0) > r.get("loose_before", 0)
    else:                                            # scripted: held in the hand after the pick
        success = cell.attached(cube) or cell.grasp_verified(cube)
        failure, steps, knocked_others = (None if success else "not_held"), None, None
    tilt = float(np.degrees(np.arccos(np.clip(cell.data.xmat[cube.body][8], -1.0, 1.0))))
    if not success and failure is None and not cell.attached(cube) and tilt > 45.0:
        failure = "knocked_over"
    return {**asdict(case), "size": size, "yaw_rel_measured": yaw_rel, "tilt_start": tilt0,
            "success": bool(success), "failure": failure, "steps": steps,
            "knocked_over": failure == "knocked_over", "knocked_others": knocked_others}


def summarize(rows: list[dict]) -> dict:
    """Success, knock-overs and failures, overall and by 10-deg rotation bin."""
    from collections import Counter

    from polypanda.evaluation.reach import wilson_interval

    def block(rs):
        n, ok = len(rs), sum(r["success"] for r in rs)
        lo, hi = wilson_interval(ok, n) if n else (0.0, 0.0)
        return {"n": n, "success": ok, "rate": ok / n if n else None, "ci95": [lo, hi],
                "knocked_over": sum(r["knocked_over"] for r in rs),
                "failures": dict(Counter(r["failure"] or "timeout" for r in rs
                                         if not r["success"]))}

    bins = {}
    for lo in range(-45, 45, 10):
        rs = [r for r in rows if lo <= r["yaw_rel"] < lo + 10 or (lo == 35 and r["yaw_rel"] >= 45)]
        if rs:
            bins[f"{lo:+d}..{lo + 10:+d}"] = block(rs)
    return {**block(rows), "by_rotation": bins}
