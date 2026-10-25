"""Milestone 9 stacking cell: layout, conveyor indexing, reach, grip, and a scripted tower."""

import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

from polypanda.control.ik import PoseIK, top_down_quat
from polypanda.simulation.panda import PandaIndices
from polypanda.simulation.workcell import (
    BELT_CONTACT,
    STACK_LIGHT,
    Workcell,
    WorkcellConfig,
    queue_xy,
    station_xy,
)

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def cell():
    return Workcell.build()


def _run_belt(cell, steps=400):
    arm = PandaIndices.from_model(cell.model)
    home = cell.data.qpos[arm.arm_qpos].copy()
    for k in range(steps):
        cell.conveyor.update()
        cell.data.ctrl[arm.arm_actuators] = home
        mujoco.mj_step(cell.model, cell.data, nstep=25)
        if k > 5 and not cell.conveyor.moving:
            return


def test_stations_sit_on_the_circle_around_the_base(cell):
    cfg = cell.config
    for xy, angle in ((cell.pick_xy, cfg.infeed_angle), (cell.tower_xy, cfg.tower_angle),
                      (cell.reject_xy, cfg.reject_angle)):
        assert np.linalg.norm(xy) == pytest.approx(cfg.radius)
        assert np.degrees(np.arctan2(xy[1], xy[0])) == pytest.approx(angle)
        assert abs(angle) < 166  # inside joint 1's range
    mujoco.mj_forward(cell.model, cell.data)
    np.testing.assert_allclose(cell.data.site_xpos[cell.model.site("tower_base").id][:2],
                               cell.tower_xy, atol=1e-9)


def test_conveyor_indexes_each_cube_to_the_pick_point(cell):
    cell.reset()
    for k, cube in enumerate(cell.cubes):
        assert cell.conveyor.index()
        _run_belt(cell)
        p = cell.data.xpos[cube.body]
        assert np.linalg.norm(p[:2] - cell.pick_xy) < 0.002, k
        assert p[2] == pytest.approx(cube.half_size[2] + cell.config.belt_lift, abs=1e-3)
        assert np.degrees(np.arccos(cell.data.xmat[cube.body].reshape(3, 3)[2, 2])) < 1.0
        cell.data.qpos[cube.qpos:cube.qpos + 3] = [0.6, -0.6 + 0.1 * k, 0.03]  # "picked"
        mujoco.mj_forward(cell.model, cell.data)
    assert not cell.conveyor.index()  # nothing left on the belt


def test_belt_carries_cubes_but_touches_nothing_else(cell):
    m = cell.model
    belt, floor = m.geom("belt").id, m.geom("floor").id
    cube = m.geom("cube0").id

    def collide(a, b):
        return bool(m.geom_contype[a] & m.geom_conaffinity[b]
                    or m.geom_contype[b] & m.geom_conaffinity[a])

    assert collide(belt, cube) and not collide(belt, floor)
    fingers = [g for g in range(m.ngeom)
               if m.body(m.geom_bodyid[g]).name in ("left_finger", "right_finger")]
    assert fingers and not any(collide(belt, g) for g in fingers)
    assert m.geom_conaffinity[cube] & BELT_CONTACT and collide(cube, floor)


def test_dressing_and_stack_light_never_collide(cell):
    m = cell.model
    for name in ("factory_floor", "fence_0", "hazard_0", *(f"lamp_{c}" for c in STACK_LIGHT)):
        g = m.geom(name).id
        assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0, name


def test_stack_light_switches_lamps(cell):
    m = cell.model
    cell.set_stack_light("green")
    green, red = m.geom_rgba[m.geom("lamp_green").id], m.geom_rgba[m.geom("lamp_red").id]
    assert green[1] > 0.9 and red[0] < 0.3
    cell.set_stack_light("red")
    assert m.geom_rgba[m.geom("lamp_red").id][0] > 0.9
    assert m.geom_rgba[m.geom("lamp_green").id][1] < 0.3


def test_every_station_is_reachable_top_down_facing_it(cell):
    ik = PoseIK()
    home = ik.data.qpos[ik.idx.arm_qpos].copy()
    targets = [(cell.pick_xy, 0.032)] + [(cell.tower_xy, 0.03 + 0.06 * k) for k in range(3)] + [
        (cell.tower_xy, 0.27), (cell.reject_xy, 0.15)]
    for xy, z in targets:
        angle = np.arctan2(xy[1], xy[0])
        seed = home.copy()
        seed[0] = angle
        for yaw in np.linspace(-np.pi / 4, np.pi / 4, 5):
            assert ik.solve(seed, [*xy, z], top_down_quat(angle + yaw))[1], (xy, z, yaw)


def test_grip_is_stiffer_but_opens_as_wide(cell):
    m = cell.model
    act = m.actuator("actuator8").id
    stiffness = -m.actuator_biasprm[act, 1]
    assert stiffness == pytest.approx(1000.0)
    assert m.actuator_gainprm[act, 0] * 255 / stiffness == pytest.approx(0.04)  # full opening


def test_queue_positions_follow_the_belt():
    cfg = WorkcellConfig()
    q = queue_xy(cfg)
    gaps = [np.linalg.norm(q[k + 1] - q[k]) for k in range(len(q) - 1)]
    np.testing.assert_allclose(gaps, cfg.cube_spacing)
    assert np.linalg.norm(q[0] - station_xy(cfg.infeed_angle, cfg.radius)) == pytest.approx(
        cfg.first_cube)


def test_scripted_preview_builds_a_tower():
    spec = importlib.util.spec_from_file_location("view_workcell",
                                                  ROOT / "scripts" / "view_workcell.py")
    view = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(view)

    class Headless:
        def is_running(self):
            return True

        def sync(self):
            pass

    cell = Workcell.build()
    preview = view.Preview(cell, Headless(), speed=1e9)
    preview.run_cycle(np.random.default_rng(3), jitter=True)
    d = cell.data
    heights = sorted(d.xpos[c.body][2] for c in cell.cubes)
    np.testing.assert_allclose(heights, [0.03, 0.09, 0.15], atol=0.005)
    for c in cell.cubes:
        assert np.degrees(np.arccos(d.xmat[c.body].reshape(3, 3)[2, 2])) < 5
        assert np.linalg.norm(d.xpos[c.body][:2] - cell.tower_xy) < 0.02
