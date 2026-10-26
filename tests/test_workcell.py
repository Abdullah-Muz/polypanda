"""Milestone 9 work-order cell: layout, recipes and signs, conveyor feeding, reach, grip,
factory dressing, and a scripted work order."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from polypanda.control.ik import PoseIK, top_down_quat
from polypanda.control.scripted_cell import ScriptedCell
from polypanda.simulation.panda import PandaIndices
from polypanda.simulation.workcell import (
    BELT_CONTACT,
    COLOURS,
    STACK_LIGHT,
    TOWERS,
    Workcell,
    WorkcellConfig,
    WorkOrder,
    queue_xy,
    station_xy,
)

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def cell():
    return Workcell.build()


def _run_belt(cell, steps=600):
    arm = PandaIndices.from_model(cell.model)
    home = cell.data.qpos[arm.arm_qpos].copy()
    for _ in range(steps):
        cell.conveyor.update()
        cell.data.ctrl[arm.arm_actuators] = home
        mujoco.mj_step(cell.model, cell.data, nstep=25)
        if cell.conveyor.ready:
            return True
    return False


def test_stations_sit_on_the_circle_around_the_base(cell):
    cfg = cell.config
    mujoco.mj_forward(cell.model, cell.data)
    stations = [(cell.pick_xy, cfg.infeed_angle), (cell.reject_xy, cfg.reject_angle)] + [
        (cell.tower_xy(t), a) for t, a in enumerate(cfg.tower_angles)] + [
        (cell.buffer_xy(s), a) for s, a in enumerate(cfg.buffer_angles)]
    for xy, angle in stations:
        assert np.linalg.norm(xy) == pytest.approx(cfg.radius)
        assert np.degrees(np.arctan2(xy[1], xy[0])) == pytest.approx(angle)
        assert abs(angle) < 166  # inside joint 1's range
    for t in range(3):
        site = cell.model.site(f"tower{TOWERS[t]}_base").id
        np.testing.assert_allclose(cell.data.site_xpos[site][:2], cell.tower_xy(t), atol=1e-9)


def test_work_orders_are_always_placeable_in_arrival_order():
    rng = np.random.default_rng(0)
    for _ in range(50):
        order = WorkOrder.random(rng)
        assert len(order.recipes) == 3
        assert all(1 <= len(r) <= 4 and set(r) <= set(COLOURS) for r in order.recipes)
        placed = [[] for _ in order.recipes]
        for colour, t in order.arrivals:
            assert order.recipes[t][len(placed[t])] == colour  # next needed by that tower
            placed[t].append(colour)
        assert [tuple(p) for p in placed] == list(order.recipes)


def test_reset_paints_cubes_and_signs(cell):
    order = WorkOrder(recipes=(("red", "blue"), ("yellow",), ("green", "green", "red")),
                      arrivals=(("red", 0), ("green", 2), ("blue", 0), ("yellow", 1),
                                ("green", 2), ("red", 2)))
    cell.reset(order)
    m = cell.model
    for k, (colour, _) in enumerate(order.arrivals):
        np.testing.assert_allclose(m.geom_rgba[cell.cubes[k].geom], COLOURS[colour])
        assert cell.cube_colour(cell.cubes[k]) == colour
    for t, recipe in enumerate(order.recipes):
        for k in range(cell.config.max_height):
            rgba = m.geom_rgba[m.geom(f"sign{TOWERS[t]}_{k}").id]
            if k < len(recipe):
                np.testing.assert_allclose(rgba, COLOURS[recipe[k]])
            else:
                assert max(rgba[:3]) < 0.3  # unlit square


def test_conveyor_feeds_every_cube_of_the_order_to_the_pick_point(cell):
    order = WorkOrder.random(np.random.default_rng(4), heights=(2, 3))
    cell.reset(order)
    assert len(order.arrivals) > cell.config.batch  # needs a rewind and a second batch
    assert cell.conveyor.index()
    for k in range(len(order.arrivals)):
        assert _run_belt(cell), k
        cube = cell.conveyor.lead()
        assert cell.cubes.index(cube) == k  # arrives in order
        p = cell.data.xpos[cube.body]
        assert np.linalg.norm(p[:2] - cell.pick_xy) < 0.005
        assert np.degrees(np.arccos(cell.data.xmat[cube.body].reshape(3, 3)[2, 2])) < 1.0
        cell.data.qpos[cube.qpos:cube.qpos + 3] = [-0.6, -0.8 + 0.08 * k, 0.03]  # "picked"
        mujoco.mj_forward(cell.model, cell.data)
        more = cell.conveyor.index()
        assert more == (k < len(order.arrivals) - 1)


def test_belt_carries_cubes_but_touches_nothing_else(cell):
    m = cell.model
    belt, table = m.geom("belt").id, m.geom("floor").id
    cube = m.geom("cube0").id

    def collide(a, b):
        return bool(m.geom_contype[a] & m.geom_conaffinity[b]
                    or m.geom_contype[b] & m.geom_conaffinity[a])

    assert collide(belt, cube) and not collide(belt, table)
    fingers = [g for g in range(m.ngeom)
               if m.body(m.geom_bodyid[g]).name in ("left_finger", "right_finger")]
    assert fingers and not any(collide(belt, g) for g in fingers)
    assert m.geom_conaffinity[cube] & BELT_CONTACT and collide(cube, table)


def test_dressing_never_collides(cell):
    m = cell.model
    cell_parts = {"belt", "floor"} | {f"cube{k}" for k in range(cell.config.n_cubes)}
    bin_body = m.body("reject_bin").id
    robot = {m.body(n).id for n in ("link0", "link1", "link2", "link3", "link4", "link5",
                                    "link6", "link7", "hand", "left_finger", "right_finger")}
    for g in range(m.ngeom):
        body = m.geom_bodyid[g]
        if body in robot or body == bin_body or m.geom(g).name in cell_parts:
            continue
        assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0, m.geom(g).name
    for name in ("factory_floor", "control_cabinet", "hmi_screen", "tunnel", "hopper",
                 *(f"lamp_{c}" for c in STACK_LIGHT)):
        m.geom(name)  # exists


def test_stack_light_switches_lamps(cell):
    m = cell.model
    cell.set_stack_light("green")
    assert m.geom_rgba[m.geom("lamp_green").id][1] > 0.9
    assert m.geom_rgba[m.geom("lamp_red").id][0] < 0.3
    cell.set_stack_light("red")
    assert m.geom_rgba[m.geom("lamp_red").id][0] > 0.9
    assert m.geom_rgba[m.geom("lamp_green").id][1] < 0.3


def test_every_station_is_reachable_top_down_facing_it(cell):
    ik = PoseIK()
    home = ik.data.qpos[ik.idx.arm_qpos].copy()
    targets = [(cell.pick_xy, 0.032), (cell.reject_xy, 0.15)] + [
        (cell.tower_xy(t), z) for t in range(3) for z in (0.03, 0.21, 0.36)] + [
        (cell.buffer_xy(s), 0.03) for s in range(3)]
    for xy, z in targets:
        angle = np.arctan2(xy[1], xy[0])
        seed = home.copy()
        seed[0] = angle
        for yaw in np.linspace(-np.pi / 4, np.pi / 4, 3):
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
    assert len(q) == cfg.batch
    gaps = [np.linalg.norm(q[k + 1] - q[k]) for k in range(len(q) - 1)]
    np.testing.assert_allclose(gaps, cfg.cube_spacing)
    assert np.linalg.norm(q[0] - station_xy(cfg.infeed_angle, cfg.radius)) == pytest.approx(
        cfg.first_cube)


def test_grasp_assist_attaches_only_a_verified_grip():
    cell = Workcell.build()
    order = WorkOrder(recipes=(("red",), ("blue",), ("green",)),
                      arrivals=(("red", 0), ("blue", 1), ("green", 2)))
    cell.reset(order)
    cube = cell.cubes[0]
    assert not cell.attach(cube) and not cell.attached(cube)  # nothing is holding it
    ctl = ScriptedCell(cell)
    cell.conveyor.index()
    ctl.to_carry()
    ctl.swing_to(np.radians(cell.config.infeed_angle))
    while not cell.conveyor.ready:
        ctl.step()
    ctl.pick(cube)
    assert cell.attached(cube) and cell.grasp_verified(cube)
    cell.detach(cube)
    ctl.grip = 255.0  # open: the released cube falls back to the table
    ctl.step(40)
    assert not cell.attached(cube)
    assert cell.data.xpos[cube.body][2] < 0.05


def test_grasp_assist_can_be_switched_off():
    from dataclasses import replace

    cell = Workcell.build(replace(WorkcellConfig(), grasp_assist=False))
    cell.reset(WorkOrder.random(np.random.default_rng(0)))
    assert not cell.attach(cell.cubes[0])
    names = [cell.model.equality(i).name for i in range(cell.model.neq)]
    assert not any(n.startswith("assist") for n in names)  # (the finger coupling remains)


def test_scripted_cell_fulfils_work_orders():
    cell = Workcell.build()
    ctl = ScriptedCell(cell)
    rng = np.random.default_rng(3)
    order = WorkOrder(recipes=(("red", "blue", "green", "yellow"), ("yellow", "green"),
                               ("blue",)),
                      arrivals=(("red", 0), ("yellow", 1), ("blue", 0), ("blue", 2),
                                ("green", 0), ("green", 1), ("yellow", 0)))
    towers = ctl.run_order(order, rng, lateral=0.015, yaw=np.radians(15))
    assert ctl.check(towers) == [True, True, True]
    assert ctl.ik_failures == 0
    assert not any(cell.attached(c) for c in cell.cubes)  # everything released


def test_scripted_cell_routes_cubes_through_the_buffer_and_the_reject_bin():
    cell = Workcell.build()
    ctl = ScriptedCell(cell)
    # green arrives before the red it goes on (buffer), the last red is surplus (reject bin)
    order = WorkOrder((("red", "green"), ("blue",), ("yellow",)),
                      (("green", None), ("yellow", None), ("red", None), ("blue", None),
                       ("red", None)))
    log = []
    towers = ctl.run_mixed_order(order, np.random.default_rng(0), lateral=0.015,
                                 yaw=np.radians(15), log=log.append)
    assert towers == [[2, 0], [3], [1]] and ctl.check(towers) == [True, True, True]
    assert ctl.stats == {"buffered": 1, "from_buffer": 1, "rejected": 1, "needed_rejected": 0}
    assert any("buffer slot 1" in line for line in log)
    surplus = cell.data.xpos[cell.cubes[4].body]
    assert np.abs(surplus[:2] - cell.reject_xy).max() < cell.config.bin_size / 2  # in the bin
    assert ctl.ik_failures == 0
