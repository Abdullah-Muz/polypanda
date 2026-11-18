"""v3.2 cell-like pick checks (evaluation/cell_picks.py)."""

import numpy as np
import pytest

from polypanda.control.scripted_cell import ScriptedCell
from polypanda.evaluation.cell_picks import (
    PickCase,
    PickSet,
    generate_pick_set,
    run_case,
    stage_case,
    summarize,
    wrap45,
)
from polypanda.simulation.disturbances import rest_state


def test_sets_are_reproducible_and_checked(tmp_path):
    a = generate_pick_set("t", "floor", 20, seed=5)
    b = generate_pick_set("t", "floor", 20, seed=5)
    assert a.digest == b.digest and a.cases == b.cases
    assert all(-45 <= c.yaw_rel <= 45 and 0.31 <= c.radius <= 0.40 for c in a.cases)
    belt = generate_pick_set("t", "belt", 20, seed=5)
    assert all(abs(c.yaw_rel) <= 15 and abs(c.lateral) <= 0.015 and c.angle == 90.0
               for c in belt.cases)
    path = tmp_path / "s.json"
    a.save(path)
    assert PickSet.load(path).cases == a.cases
    path.write_text(path.read_text().replace('"radius": 0.3', '"radius": 0.2', 1))
    with pytest.raises(ValueError, match="digest"):
        PickSet.load(path)
    with pytest.raises(ValueError):
        generate_pick_set("t", "shelf", 1, seed=0)


def test_wrap45():
    assert wrap45(50.0) == pytest.approx(-40.0)
    assert wrap45(-50.0) == pytest.approx(40.0)
    assert wrap45(90.0 + 12.0) == pytest.approx(12.0)


@pytest.mark.parametrize("kind", ["belt", "floor"])
def test_staged_case_rests_where_and_as_asked(kind):
    case = (PickCase("belt", 90.0, 0.5, 0.01, -9.0, 2, (1, 0, 3)) if kind == "belt"
            else PickCase("floor", -65.0, 0.35, 0.0, 33.0, 0, (2, 3, 0)))
    cell = stage_case(case, size=0.9 if kind == "floor" else 1.0)
    ctl = ScriptedCell(cell)
    ctl.step(20)
    cube = cell.cubes[0]
    p, R = cell.data.xpos[cube.body], cell.data.xmat[cube.body].reshape(3, 3)
    yaw_rel = wrap45(np.degrees(np.arctan2(R[1, 0], R[0, 0]) - np.arctan2(p[1], p[0])))
    assert yaw_rel == pytest.approx(case.yaw_rel, abs=0.5)
    assert abs(np.hypot(p[0], p[1]) - case.radius) < 0.015
    if kind == "belt":
        assert cell.conveyor.on_belt(cube)
        assert sum(cell.conveyor.on_belt(c) for c in cell.cubes) == 1 + case.queued
    else:
        assert rest_state(cell, cube)["pose"] == "upright"
        assert cube.half_size[2] == pytest.approx(0.03 * 0.9)
    # tower cubes stand stacked and still
    assert all(np.linalg.norm(cell.data.qvel[c.qvel:c.qvel + 6]) < 1e-2 for c in cell.cubes)


def test_scripted_pick_of_a_belt_case():
    r = run_case(PickCase("belt", 90.0, 0.5, -0.012, 11.0, 1, (0, 2, 1)), ScriptedCell)
    assert r["success"] and not r["knocked_over"]
    s = summarize([r, {**r, "success": False, "failure": "knocked_over", "knocked_over": True,
                       "yaw_rel": -40.0}])
    assert s["n"] == 2 and s["success"] == 1 and s["knocked_over"] == 1
    assert s["by_rotation"]["-45..-35"]["knocked_over"] == 1
