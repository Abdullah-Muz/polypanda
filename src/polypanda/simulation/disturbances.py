"""Disturbances for the work-order cell (v3): knocking a tower over, and judging whether the
fallen cubes can be recovered (docs/v3_targets.md).

A knock is one of three pushes, in a random horizontal direction:
- a shove on the top cube, or on a middle cube, at 0.6-1.5 m/s;
- a steady 1.0-2.5 N push on the top cube, as a hand would push, stopping once the cube has
  moved 3 cm (pushing on would fling it across the bench).

The outcome is classified once everything has settled: D1 exactly one cube left the tower, D2
some but not all, D3 all of them. A knock is recoverable when every fallen cube rests flat
(upright, or lying on a side, not leaning on anything), lies within the arm's reach from the base
axis and inside its base rotation range, and is not on the belt or in the reject bin.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from polypanda.simulation.scenes import CubeIndices
from polypanda.simulation.workcell import Workcell

PUSHES = ("shove_top", "shove_middle", "push_top")
SHOVE_SPEED = (0.6, 1.5)     # m/s
PUSH_FORCE = (1.0, 2.5)      # N
PUSH_LIMIT = 0.03            # m the pushed cube may move before the push stops
PUSH_STEPS = 16              # control steps the push may last (0.8 s)
SETTLE_STEPS = 60            # control steps to settle afterwards (3 s)
MOVED = 0.01                 # m: a cube further than this from where it stood has left the tower
REACH = (0.30, 0.78)         # m from the base axis where a cube on the table can be grasped
BASE_RANGE = 162.0           # deg either way (joint 1's limit is 166 deg)
UPRIGHT_TILT, FLAT_TILT = 10.0, 10.0  # deg


@dataclass(frozen=True)
class Knock:
    push: str            # one of PUSHES
    angle: float         # rad, direction of the push in the horizontal plane
    strength: float      # m/s for a shove, N for the steady push

    @classmethod
    def sample(cls, rng: np.random.Generator) -> "Knock":
        push = str(rng.choice(PUSHES))
        lo, hi = PUSH_FORCE if push == "push_top" else SHOVE_SPEED
        return cls(push, float(rng.uniform(0, 2 * np.pi)), float(rng.uniform(lo, hi)))

    def target(self, tower: list[CubeIndices]) -> CubeIndices:
        """The cube pushed (`tower` bottom first)."""
        return tower[len(tower) // 2] if self.push == "shove_middle" else tower[-1]


def stand_tower(cell: Workcell, t: int, cubes: list[CubeIndices], rng: np.random.Generator,
                jitter: float = 0.002, yaw_jitter: float = 0.03) -> None:
    """Stand `cubes` (bottom first) as a tower on fixture `t`, each a little off centre and
    turned, as the robot leaves them; the scene is not stepped."""
    import mujoco
    d = cell.data
    xy, yaw0 = cell.tower_xy(t), np.radians(cell.config.tower_angles[t])
    for k, c in enumerate(cubes):
        p = xy + rng.uniform(-jitter, jitter, 2)
        a = yaw0 + rng.uniform(-yaw_jitter, yaw_jitter)
        d.qpos[c.qpos:c.qpos + 7] = [*p, c.half_size[2] * (2 * k + 1),
                                     np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)]
        d.qvel[c.qvel:c.qvel + 6] = 0.0
    mujoco.mj_forward(cell.model, d)


def apply_knock(cell: Workcell, tower: list[CubeIndices], knock: Knock,
                step: Callable[[], None], settle: int = SETTLE_STEPS) -> None:
    """Push one cube of `tower` (bottom first) and let everything settle. `step` advances the
    simulation by one control step (the controller's own step while an order runs)."""
    d = cell.data
    cube = knock.target(tower)
    u = np.array([np.cos(knock.angle), np.sin(knock.angle)])
    if knock.push == "push_top":
        start = d.xpos[cube.body].copy()
        for _ in range(PUSH_STEPS):
            moved = np.linalg.norm(d.xpos[cube.body] - start)
            d.xfrc_applied[cube.body, :2] = knock.strength * u if moved < PUSH_LIMIT else 0.0
            step()
        d.xfrc_applied[cube.body, :] = 0.0
    else:
        d.qvel[cube.qvel:cube.qvel + 2] = knock.strength * u
    for _ in range(settle):
        step()


def rest_state(cell: Workcell, cube: CubeIndices) -> dict:
    """How a cube on the table rests: its pose (upright, lying or leaning), where it is, and
    whether the arm can get at it."""
    d = cell.data
    p = d.xpos[cube.body]
    up = abs(float(d.xmat[cube.body][8]))  # |cube z axis . world z|
    half = cube.half_size
    if up > np.cos(np.radians(UPRIGHT_TILT)) and abs(p[2] - half[2]) < 0.005:
        pose = "upright"
    elif up < np.sin(np.radians(FLAT_TILT)) and abs(p[2] - half[0]) < 0.004:
        pose = "lying"
    else:
        pose = "leaning"
    radius = float(np.hypot(p[0], p[1]))
    angle = float(np.degrees(np.arctan2(p[1], p[0])))
    in_bin = bool(np.abs(p[:2] - cell.reject_xy).max() < cell.config.bin_size / 2)
    blocked = cell.conveyor.on_belt(cube) or in_bin
    reachable = REACH[0] <= radius <= REACH[1] and abs(angle) <= BASE_RANGE and not blocked
    return {"pose": pose, "radius": radius, "angle": angle, "reachable": reachable,
            "recoverable": reachable and pose != "leaning"}


def knock_outcome(cell: Workcell, tower: list[CubeIndices], before: list[np.ndarray]) -> dict:
    """Classify a settled knock: which cubes left the tower (`before`: their positions just
    before it), D1/D2/D3, and whether every fallen cube can be recovered."""
    fallen = [c for c, p in zip(tower, before, strict=True)
              if np.linalg.norm(cell.data.xpos[c.body] - p) > MOVED]
    kind = (None if not fallen else "D1" if len(fallen) == 1
            else "D3" if len(fallen) == len(tower) else "D2")
    states = [rest_state(cell, c) for c in fallen]
    xy = [cell.data.xpos[c.body][:2] for c in fallen]
    touching = any(np.linalg.norm(a - b) < 0.065 for i, a in enumerate(xy) for b in xy[i + 1:])
    return {"kind": kind, "fallen": fallen, "states": states, "touching": touching,
            "recoverable": bool(fallen) and all(s["recoverable"] for s in states)}
