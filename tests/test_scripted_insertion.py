"""control/scripted_insertion.py: the wrist force-torque reading and the force-guided search,
in scripts/probe_nest_insertion.py's tight nest (stiff contacts, 0.25 mm clearance)."""

import importlib
import sys
from pathlib import Path

import mujoco
import pytest

from polypanda.control.scripted_cell import ScriptedCell
from polypanda.control.scripted_insertion import part_force, part_wrench
from polypanda.simulation.workcell import WorkOrder, _yaw_quat, station_xy

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def probe():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield importlib.import_module("probe_nest_insertion")
    finally:
        sys.path.remove(str(SCRIPTS))


def test_a_resting_part_feels_its_weight_upward_and_no_twist(probe):
    cell = probe.build_cell(probe.Nest(0.00025, 0.0005, 0.0, "stiff"))
    cell.reset(WorkOrder(recipes=((), (), ()), arrivals=()))
    cube = cell.cubes[0]
    xy = station_xy(probe.SOURCE_ANGLE, cell.config.radius)
    cell.data.qpos[cube.qpos:cube.qpos + 7] = [*xy, cube.half_size[2], *_yaw_quat(0.3)]
    cell.data.qvel[:] = 0.0
    mujoco.mj_forward(cell.model, cell.data)
    ScriptedCell(cell).step(20)
    w = part_wrench(cell.model, cell.data, cube)
    weight = cell.model.body_subtreemass[cube.body] * 9.81
    assert w[2] == pytest.approx(weight, rel=0.02)
    assert abs(w[0]) < 0.02 * weight and abs(w[1]) < 0.02 * weight
    assert abs(w[5]) < 1e-3
    assert part_force(cell.model, cell.data, cube) == pytest.approx(weight, rel=0.02)


@pytest.mark.parametrize("error_mm, nest_yaw", [(0.0, 1.0), (2.0, 0.0)])
def test_the_search_seats_a_part_the_straight_down_place_cannot(probe, error_mm, nest_yaw):
    nest = probe.Nest(0.00025, 0.0005, nest_yaw, "stiff")
    row = probe.run_trial(probe.build_cell(nest), nest, error_mm, 0, "search", 30.0)
    assert row["outcome"] == "seated"
    assert row["dropped_in"] and row["peak_force_n"] < 15.0
