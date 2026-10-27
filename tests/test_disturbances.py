"""Knocking towers over, and judging whether the fallen cubes can be recovered (v3)."""

import mujoco
import numpy as np
import pytest

from polypanda.simulation.disturbances import (
    PUSH_FORCE,
    SHOVE_SPEED,
    Knock,
    apply_knock,
    knock_outcome,
    rest_state,
    stand_tower,
)
from polypanda.simulation.workcell import Workcell, WorkcellConfig, WorkOrder


@pytest.fixture(scope="module")
def cell():
    cell = Workcell.build(WorkcellConfig(dressing=False))
    cell.reset(WorkOrder((("red",) * 4,) * 3, tuple(("red", 0) for _ in range(12))))
    return cell


def _step(cell):
    return lambda: mujoco.mj_step(cell.model, cell.data, nstep=25)


def test_a_standing_tower_stays_put(cell):
    tower = cell.cubes[:4]
    stand_tower(cell, 1, tower, np.random.default_rng(0))
    before = [cell.data.xpos[c.body].copy() for c in tower]
    for _ in range(20):
        _step(cell)()
    assert all(np.linalg.norm(cell.data.xpos[c.body] - p) < 0.001
               for c, p in zip(tower, before, strict=True))
    assert knock_outcome(cell, tower, before)["kind"] is None


def test_a_hard_shove_on_the_top_cube_knocks_it_off(cell):
    tower = cell.cubes[:2]
    stand_tower(cell, 0, tower, np.random.default_rng(1))
    for _ in range(10):
        _step(cell)()
    before = [cell.data.xpos[c.body].copy() for c in tower]
    tangent = np.radians(cell.config.tower_angles[0]) + np.pi / 2  # along the station circle
    apply_knock(cell, tower, Knock("shove_top", tangent, 1.2), _step(cell))
    out = knock_outcome(cell, tower, before)
    assert out["kind"] == "D1" and out["fallen"] == [tower[1]]
    assert out["states"][0]["pose"] in ("upright", "lying", "leaning")


def test_rest_states(cell):
    d, cube = cell.data, cell.cubes[5]

    def put(xy, z, quat):
        d.qpos[cube.qpos:cube.qpos + 7] = [*xy, z, *quat]
        mujoco.mj_forward(cell.model, d)

    upright = [1.0, 0.0, 0.0, 0.0]
    on_side = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]  # 90 deg about x
    put([0.0, -0.55], 0.03, upright)
    s = rest_state(cell, cube)
    assert s["pose"] == "upright" and s["reachable"] and s["recoverable"]
    put([0.0, -0.55], 0.02, on_side)
    s = rest_state(cell, cube)
    assert s["pose"] == "lying" and s["recoverable"]
    put([0.0, -0.9], 0.03, upright)
    assert not rest_state(cell, cube)["reachable"]             # beyond the arm's reach
    put(cell.reject_xy, 0.03, upright)
    assert not rest_state(cell, cube)["reachable"]             # in the reject bin
    tilted = [np.cos(np.pi / 8), np.sin(np.pi / 8), 0.0, 0.0]  # 45 deg
    put([0.0, -0.55], 0.04, tilted)
    s = rest_state(cell, cube)
    assert s["pose"] == "leaning" and not s["recoverable"]


def test_knock_sampling_covers_every_push_within_its_range():
    rng = np.random.default_rng(0)
    knocks = [Knock.sample(rng) for _ in range(300)]
    assert {k.push for k in knocks} == {"shove_top", "shove_middle", "push_top"}
    for k in knocks:
        lo, hi = PUSH_FORCE if k.push == "push_top" else SHOVE_SPEED
        assert lo <= k.strength <= hi and 0 <= k.angle < 2 * np.pi
    tower = list(range(4))
    assert Knock("shove_middle", 0, 1).target(tower) == 2
    assert Knock("push_top", 0, 1).target(tower) == 3
