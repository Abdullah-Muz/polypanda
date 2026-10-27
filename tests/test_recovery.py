"""Closed-loop recovery in the work-order cell (v3): after a knock the controller reads the cell,
brings its plan in line with the towers as they stand, and recovers the loose cubes it can."""

import mujoco
import numpy as np
import pytest

from polypanda.control.scripted_cell import ScriptedCell
from polypanda.simulation.disturbances import Disturbance, Knock
from polypanda.simulation.workcell import Workcell, WorkOrder

UPRIGHT, UPSIDE_DOWN = [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]
LYING = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]  # 90 deg about x


class MovesTheTopCube(ScriptedCell):
    """A stand-in for a knock: the top cube of the tower is set down on the table beside it, at
    a known spot and in a known pose (real knocks are random)."""

    pose, radius = UPRIGHT, 0.5

    def knock(self, tower, knock, log=None):
        cube = self.cell.cubes[tower[-1]]
        a = np.radians(self.cell.config.tower_angles[0] + 15)  # beside tower A
        z = cube.half_size[0] if self.pose is LYING else cube.half_size[2]
        r = self.radius
        self.d.qpos[cube.qpos:cube.qpos + 7] = [r * np.cos(a), r * np.sin(a), z, *self.pose]
        self.d.qvel[cube.qvel:cube.qvel + 6] = 0.0
        mujoco.mj_forward(self.m, self.d)
        self.step(10)
        self.disturbance_outcome = {"kind": "D1", "recoverable": True, "fallen": [tower[-1]]}


ORDER = WorkOrder((("red", "green", "blue"), ("yellow",), ("red",)),
                  tuple((c, None) for c in ("red", "green", "yellow", "blue", "red")))
KNOCK = Disturbance(0, 2, Knock("shove_top", 0.0, 1.0))  # after the green goes onto tower A


@pytest.mark.parametrize("pose", [UPRIGHT, UPSIDE_DOWN, LYING])
def test_a_fallen_cube_is_recovered_and_the_tower_rebuilt(pose):
    ctl = MovesTheTopCube(Workcell.build())
    ctl.pose = pose
    log = []
    towers = ctl.run_mixed_order(ORDER, np.random.default_rng(0), disturbance=KNOCK,
                                 recover=True, log=log.append)
    assert ctl.check(towers) == [True, True, True]  # upside down counts as standing
    assert towers[0] == [0, 1, 3]                    # the green went back on, then blue
    assert ctl.stats["recovered"] == 1 and ctl.stats["written_off"] == 0
    assert any("tower A: 1 cubes fell" in line for line in log)
    assert ctl.ik_failures == 0


def test_without_recovery_the_order_fails_and_an_unreachable_cube_is_written_off():
    ctl = MovesTheTopCube(Workcell.build())
    towers = ctl.run_mixed_order(ORDER, np.random.default_rng(0), disturbance=KNOCK)
    assert ctl.check(towers)[0] is False             # v2: does not notice
    ctl.radius = 0.85                                # beyond the arm's reach
    log = []
    towers = ctl.run_mixed_order(ORDER, np.random.default_rng(0), disturbance=KNOCK,
                                 recover=True, log=log.append)
    assert ctl.stats["written_off"] == 1 and ctl.stats["recovered"] == 0
    assert any("written off (upright, out of reach)" in line for line in log)
    assert ctl.check(towers)[0] is False
