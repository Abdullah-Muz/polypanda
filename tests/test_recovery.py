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


def test_a_knock_asked_for_from_outside_hits_the_next_tower_two_high():
    """The viewer's knock key: the request waits for a placement on a tower at least 2 cubes
    tall (tower A after its green cube), knocks it once, and the cell recovers."""
    ctl = MovesTheTopCube(Workcell.build())
    ctl.knock_request = KNOCK.knock
    log = []
    towers = ctl.run_mixed_order(ORDER, np.random.default_rng(0), recover=True, log=log.append)
    assert ctl.knock_request is None and ctl.stats["recovered"] == 1
    assert ctl.check(towers) == [True, True, True]
    assert any("tower A: 1 cubes fell" in line for line in log)


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


def test_a_spare_replaces_a_cube_lost_for_good_and_the_planner_can_choose_one():
    """v3 f: a written-off cube the towers need is replaced by a spare of its colour, which
    comes up the belt as one more arrival; a recovery planner may also leave a cube it could
    rescue for a spare. It is asked with the features the symbolic cell gives it."""
    from polypanda.simulation.workcell import WorkcellConfig
    from polypanda.supervisor.symbolic import SymbolicCell

    times = {"pick_belt": 16.0, "place_tower": 2.0, "overhead": 9.0}
    ctl = MovesTheTopCube(Workcell.build(WorkcellConfig(n_cubes=7)))
    ctl.radius = 0.85                                # beyond the arm's reach
    log = []
    towers = ctl.run_mixed_order(ORDER, np.random.default_rng(0), disturbance=KNOCK,
                                 recover=True, log=log.append, spares=2, times=times)
    assert ctl.check(towers) == [True, True, True]
    assert ctl.stats["spares_used"] == 1 and towers[0] == [0, 5, 3]   # the spare is arrival 5
    assert ctl.cell.cube_colour(ctl.cell.cubes[5]) == "green"
    ctl.radius, asked = 0.5, []                      # in reach: the planner is asked

    def replace(d):
        asked.append(d)
        return True
    towers = ctl.run_mixed_order(ORDER, np.random.default_rng(0), disturbance=KNOCK,
                                 recover=True, spares=2, replace=replace, deadline=400.0,
                                 times=times)
    assert ctl.check(towers) == [True, True, True] and ctl.stats["replaced"] == 1
    assert len(asked) == 1 and not asked[0].retry and asked[0].cube.pose == "upright"
    assert asked[0].features.shape == (SymbolicCell.N_FEATURES,) and asked[0].features[12] == 1


def test_the_learned_stand_up_env_runs_in_the_cell_on_the_held_cube():
    """d+: the stand-up env on the workcell's own scene, started from the scripted stand-up's
    hand-over point (the lying cube held at the carry height over a staging spot): its arm is
    the robot only, its obstacles are the other cubes (the tower), not the held one."""
    from dataclasses import replace
    from pathlib import Path

    from polypanda.environments.standup import StandUpConfig, StandUpV3Env
    from polypanda.simulation.disturbances import stand_tower
    from polypanda.supervisor.cell import LearnedCell
    from polypanda.supervisor.stations import StationView

    cell = Workcell.build()
    ctl = LearnedCell(cell)
    rng = np.random.default_rng(0)
    cell.reset(ORDER, rng)
    tower = cell.cubes[:3]
    stand_tower(cell, 0, tower, rng)
    cube = cell.cubes[3]
    a = np.radians(cell.config.tower_angles[0] + 15)
    ctl.d.qpos[cube.qpos:cube.qpos + 7] = [0.5 * np.cos(a), 0.5 * np.sin(a), cube.half_size[0],
                                           *LYING]
    mujoco.mj_forward(ctl.m, ctl.d)
    ctl.to_carry()
    ctl.step(20)
    held, c, yaw = ctl.grasp_lying(cube)
    assert held
    spot = ctl.staging_xy(0)
    angle, _, _ = ctl.carry_over(cube, spot, c, yaw)
    cfg = StandUpConfig.from_yaml(Path(__file__).parents[1] / "configs" / "standup_v3_r11.yaml")
    cfg = cfg.for_evaluation()
    cfg = replace(cfg, command=replace(cfg.command, chooser=None))
    env = StandUpV3Env(cfg, scene=(cell.model, cell.data), cube=cell.model.body(cube.body).name)
    view = StationView(env, angle)
    top = ctl.nearest_stack(spot, exclude=cube)
    assert np.allclose(top[:2], cell.tower_xy(0), atol=0.01)
    assert top[2] == pytest.approx(3 * 2 * cube.half_size[2], abs=0.005)
    obs, _ = view.reset(spot=spot, tower=top)
    hand = cell.model.body("hand").id
    assert hand in env.arm_bodies and cube.body not in env.arm_bodies
    assert {c.body for c in tower} <= env._obstacle_bodies
    assert not env._obstacle_bodies & (env.arm_bodies | {cube.body, 0})
    assert env._tower_gap()[0] > 0.0 and np.isfinite(obs).all()   # (carried high: clear)
    for _ in range(5):
        obs, _, terminated, _, info = view.step(np.zeros(6, dtype=np.float32))
        assert not terminated and np.isfinite(obs).all()
    assert cell.attached(cube) and env.released is False


def test_the_learned_cell_records_each_attempt_with_its_situation():
    """Step e (skill outcome rates by situation): an attempt is logged with where its cube comes
    from, how it rests, where it is, how far the nearest stack is, the loose cubes around it
    before and after, and its start and end times; a knock records where each cube came to rest."""
    from polypanda.simulation.disturbances import stand_tower
    from polypanda.supervisor.cell import LearnedCell

    cell = Workcell.build()
    ctl = LearnedCell(cell)
    rng = np.random.default_rng(0)
    cell.reset(ORDER, rng)
    stand_tower(cell, 0, cell.cubes[:3], rng)
    cube = cell.cubes[3]
    a = np.radians(cell.config.tower_angles[0] + 15)
    ctl.d.qpos[cube.qpos:cube.qpos + 7] = [0.5 * np.cos(a), 0.5 * np.sin(a), cube.half_size[0],
                                           *LYING]
    mujoco.mj_forward(ctl.m, ctl.d)
    ctl.source, ctl.order_index = "loose", 7
    xy = ctl.d.xpos[cube.body][:2].copy()
    s = ctl._situation(cube, xy)
    assert s["order"] == 7 and s["source"] == "loose" and s["pose"] == "lying"
    assert s["radius"] == pytest.approx(0.5, abs=0.01)
    assert s["angle"] == pytest.approx(cell.config.tower_angles[0] + 15, abs=0.5)
    assert s["near"] == pytest.approx(np.linalg.norm(xy - cell.tower_xy(0)), abs=0.01)
    ctl.step(4)
    result = ctl._finish({"success": True}, s, cube)
    assert result["loose_after"] == result["loose_before"] and result["t1"] > result["t0"]
    ctl.knock([0, 1, 2], KNOCK.knock)
    out = ctl.disturbance_outcome
    assert len(out["states"]) == len(out["fallen"])
    assert all({"pose", "radius", "angle", "recoverable"} <= set(s) for s in out["states"])
