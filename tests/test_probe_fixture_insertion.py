"""scripts/probe_fixture_insertion.py: the pocket's geometry, the seated check, one trial."""

import importlib
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from polypanda.control.scripted_cell import ScriptedCell
from polypanda.simulation.workcell import WorkOrder, _yaw_quat, station_xy

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def probe():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield importlib.import_module("probe_fixture_insertion")
    finally:
        sys.path.remove(str(SCRIPTS))


@pytest.fixture(scope="module")
def cell(probe):
    return probe.build_cell(probe.Pocket(0.002, "steel"))


def _rest(probe, cell, local_xy, bottom: float):
    """Set cube 0 upright at `local_xy` in the pocket's frame, `bottom` m above the bench, square
    to the pocket, and let it come to rest for 1 s (the arm holds its home pose)."""
    cell.reset(WorkOrder(recipes=((), (), ()), arrivals=()))
    cube = cell.cubes[0]
    a = np.radians(probe.POCKET_ANGLE)
    x, y = local_xy
    xy = station_xy(probe.POCKET_ANGLE, cell.config.radius) + [
        np.cos(a) * x - np.sin(a) * y, np.sin(a) * x + np.cos(a) * y]
    cell.data.qpos[cube.qpos:cube.qpos + 7] = [*xy, cube.half_size[2] + bottom, *_yaw_quat(a)]
    cell.data.qvel[:] = 0.0
    mujoco.mj_forward(cell.model, cell.data)
    ScriptedCell(cell).step(20)
    return probe.pocket_frame(cell, cube)


def test_the_opening_is_the_cube_plus_the_clearance_on_every_side(probe, cell):
    m = cell.model
    for k in range(4):
        g = m.geom(f"pocket_wall{k}")
        assert g.pos[0] - g.size[0] == pytest.approx(0.022)      # the inner face, wall frame
        assert 2 * g.size[2] == pytest.approx(probe.STRAIGHT)
        back = m.geom(f"pocket_back{k}")
        assert back.pos[2] + back.size[2] == pytest.approx(probe.DEPTH)   # the rim's top


def test_steel_walls_set_the_friction_and_cube_walls_leave_it_to_the_cube(probe, cell):
    m = cell.model
    assert m.geom("pocket_lead0").friction[0] == pytest.approx(probe.STEEL_FRICTION)
    assert m.geom("pocket_lead0").priority[0] == 1
    plain = probe.build_cell(probe.Pocket(0.002, "cube")).model
    assert plain.geom("pocket_lead0").priority[0] == 0
    assert plain.geom("pocket_lead0").friction[0] == pytest.approx(1.0)


def test_a_cube_lowered_into_the_pocket_is_seated_and_stays_inside(probe, cell):
    local, gap, tilt, yaw = _rest(probe, cell, (0.0015, -0.001), bottom=0.004)
    assert probe.is_seated(local, gap, tilt, 0.002)
    assert abs(yaw) < 1.0


def test_a_cube_set_down_on_the_rim_is_not_seated(probe, cell):
    local, gap, tilt, _ = _rest(probe, cell, (0.028, 0.0), bottom=probe.DEPTH + 0.002)
    assert not probe.is_seated(local, gap, tilt, 0.002)


def test_a_cube_standing_beside_the_pocket_is_not_seated(probe, cell):
    local, gap, tilt, _ = _rest(probe, cell, (0.09, 0.0), bottom=0.0)
    assert gap < probe.SEATED_GAP and tilt < probe.SEATED_TILT      # upright on the bench
    assert not probe.is_seated(local, gap, tilt, 0.002)


def test_the_scripted_place_seats_the_cube_when_positions_are_exact(probe, cell):
    row = probe.run_trial(cell, probe.Pocket(0.002, "steel"), 0.0, 0)
    assert row["outcome"] == "seated"
    assert row["wall_contact_steps"] == 0 and not row["gripper_touched_rim"]
    assert row["unload_ok"]


def test_every_condition_runs_the_same_trial(probe, cell):
    a = probe.run_trial(cell, probe.Pocket(0.002, "steel"), 0.0, 3)
    b = probe.run_trial(cell, probe.Pocket(0.002, "steel"), 0.0, 3)
    assert a["grip_offset_mm"] == pytest.approx(b["grip_offset_mm"], abs=1e-6)
    assert a["yaw_before_deg"] == pytest.approx(b["yaw_before_deg"], abs=1e-6)
