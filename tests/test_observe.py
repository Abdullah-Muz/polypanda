"""Reading the cell as it physically is (v3, closed loop)."""

import mujoco
import numpy as np

from polypanda.control.scripted_cell import ScriptedCell
from polypanda.simulation.disturbances import Knock, apply_knock, stand_tower
from polypanda.simulation.workcell import Workcell, WorkOrder
from polypanda.supervisor.observe import observe


def test_the_view_matches_a_completed_order_with_buffer_and_reject():
    cell = Workcell.build()
    ctl = ScriptedCell(cell)
    order = WorkOrder((("red", "green"), ("blue",), ("yellow",)),
                      (("green", None), ("yellow", None), ("red", None), ("blue", None),
                       ("red", None)))
    towers = ctl.run_mixed_order(order, np.random.default_rng(0))
    view = observe(cell)
    assert view.towers == towers == [[2, 0], [3], [1]]
    assert view.parked == [None, None, None] and view.loose == {}  # the surplus red is in the bin


def test_an_order_with_a_knock_records_it_and_fails_without_recovery():
    from polypanda.simulation.disturbances import Disturbance
    cell = Workcell.build()
    ctl = ScriptedCell(cell)
    order = WorkOrder((("red", "green", "blue"), ("yellow",), ("red",)),
                      tuple((c, None) for c in ("red", "green", "yellow", "blue", "red")))
    tangent = np.radians(cell.config.tower_angles[0]) + np.pi / 2
    log = []
    towers = ctl.run_mixed_order(order, np.random.default_rng(0),
                                 disturbance=Disturbance(0, 2, Knock("shove_top", tangent, 1.2)),
                                 log=log.append)
    out = ctl.disturbance_outcome
    assert out["kind"] == "D1" and out["fallen"] == [1]  # the green, knocked off tower A
    assert any(line.startswith("knock (shove_top): D1") for line in log)
    assert ctl.check(towers)[0] is False                 # v2 does not notice: tower A is wrong


def test_a_knocked_tower_shows_its_fallen_cubes_as_loose():
    cell = Workcell.build()
    order = WorkOrder((("red",) * 4, ("blue",), ("green",)), tuple(("red", 0) for _ in range(6)))
    cell.reset(order)
    cell.conveyor.next_arrival = 6                   # (as if all six had arrived)
    tower, rng = cell.cubes[:4], np.random.default_rng(0)
    stand_tower(cell, 0, tower, rng)
    for k, s in ((4, 0), (5, 2)):                    # two cubes parked in buffer slots 1 and 3
        c = cell.cubes[k]
        cell.data.qpos[c.qpos:c.qpos + 7] = [*cell.buffer_xy(s), c.half_size[2], 1, 0, 0, 0]
    mujoco.mj_forward(cell.model, cell.data)

    def step():
        mujoco.mj_step(cell.model, cell.data, nstep=25)

    for _ in range(10):
        step()
    view = observe(cell)
    assert view.towers[0] == [0, 1, 2, 3] and view.parked == [4, None, 5] and view.loose == {}
    tangent = np.radians(cell.config.tower_angles[0]) + np.pi / 2
    apply_knock(cell, tower, Knock("shove_top", tangent, 1.2), step)
    view = observe(cell)
    assert view.height(0) < 4 and view.towers[0] == [0, 1, 2, 3][:view.height(0)]
    assert set(view.loose) == set(range(view.height(0), 4))  # exactly the cubes that fell
    assert all({"pose", "reachable", "recoverable"} <= set(s) for s in view.loose.values())
    assert view.parked == [4, None, 5]
