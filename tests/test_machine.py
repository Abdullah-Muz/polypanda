"""v4.0: the benchtop machine (simulation/machine.py), the line (simulation/line.py) and the
scripted tending controller (control/scripted_tending.py)."""

import mujoco
import numpy as np
import pytest

from polypanda.control.scripted_tending import ScriptedTending
from polypanda.simulation.line import Line, LineConfig, MachineSpot
from polypanda.simulation.machine import DONE, EMPTY, FAULT, LOADED, RUNNING, MachineConfig
from polypanda.simulation.workcell import Workcell, WorkcellConfig, _yaw_quat


def make_line(**machine) -> Line:
    cfg = MachineConfig(**{"cycle_s": 2.0, **machine})
    return Line.build_line(LineConfig(machines=(MachineSpot("M1", -0.6, config=cfg),)))


def seat(line: Line, cube_k: int = 0):
    """Put a cube upright in M1's pocket, square to it, at rest."""
    m1 = line.machines["M1"]
    cube = line.cubes[cube_k]
    p = m1.pocket_xyz()
    yaw = np.arctan2(*line.data.xmat[m1.body].reshape(3, 3)[1::-1, 0])
    line.data.qpos[cube.qpos:cube.qpos + 7] = [p[0], p[1], cube.half_size[2] + 0.0005,
                                               *_yaw_quat(yaw)]
    line.data.qvel[:] = 0.0
    mujoco.mj_forward(line.model, line.data)
    return cube


def run(line: Line, seconds: float) -> None:
    """Physics with the machines updated every control step (25 physics steps), arm held."""
    ctl = ScriptedTending(line)
    for _ in range(int(seconds / ctl.dt)):
        ctl.step()


def test_the_fixed_cell_is_built_without_machines_or_rail():
    cell = Workcell.build(WorkcellConfig(dressing=False))
    for name in ("M1", "rail"):
        assert mujoco.mj_name2id(cell.model, mujoco.mjtObj.mjOBJ_BODY, name) < 0


def test_the_machine_faces_the_robot_at_its_rail_stop():
    line = make_line()
    m1 = line.machines["M1"]
    np.testing.assert_allclose(m1.pocket_xyz()[:2], [-0.6, -0.5], atol=1e-9)
    R = line.data.xmat[m1.body].reshape(3, 3)
    np.testing.assert_allclose(R[:, 0], [0.0, -1.0, 0.0], atol=1e-9)   # +x away from the robot


def test_an_unknown_slide_direction_is_refused():
    with pytest.raises(ValueError):
        MachineConfig(slide="up")


@pytest.mark.parametrize("slide,travel", [("right", 0.12), ("left", 0.12), ("out", 0.25)])
def test_the_lid_slides_open_when_pulled_and_stays_put_when_let_go(slide, travel):
    line = make_line(slide=slide, travel=travel)
    m1, d = line.machines["M1"], line.data
    lid = line.model.body("M1_lid").id
    axis = d.xmat[m1.body].reshape(3, 3) @ m1.cfg.slide_axis
    d.xfrc_applied[lid, :3] = 6.0 * axis                 # pull the lid open
    run(line, 1.5)
    assert m1.lid == pytest.approx(travel, abs=0.003)    # against its open stop
    d.xfrc_applied[lid, :3] = 0.0
    q = m1.lid
    run(line, 1.0)
    assert m1.lid == pytest.approx(q, abs=0.002)         # friction holds it


def _close_lid(line: Line, force: float) -> None:
    """Open the lid a little, then push it shut with `force` N."""
    m1, d = line.machines["M1"], line.data
    lid = line.model.body("M1_lid").id
    axis = d.xmat[m1.body].reshape(3, 3) @ m1.cfg.slide_axis
    d.qpos[m1.lid_qpos] = 0.04
    mujoco.mj_forward(line.model, d)
    run(line, 0.1)                                       # the latch arms: the lid was opened
    d.xfrc_applied[lid, :3] = -force * axis
    run(line, 1.5)
    d.xfrc_applied[lid, :3] = 0.0


def test_a_firm_push_latches_the_lid_and_a_gentle_one_does_not():
    firm = make_line(latch_speed=0.1)
    _close_lid(firm, 8.0)
    assert firm.machines["M1"].latched
    assert firm.machines["M1"].lid == pytest.approx(0.0, abs=0.001)
    # 0.6 N against 0.4 N of friction and 4 N s/m of damping: about 0.05 m/s, under the latch's
    # 0.1 m/s, so the lid shuts but does not latch
    gentle = make_line(latch_speed=0.1)
    _close_lid(gentle, 0.6)
    assert gentle.machines["M1"].lid < 0.003
    assert not gentle.machines["M1"].latched


def test_the_machine_starts_only_with_a_seated_part_and_a_latched_lid():
    line = make_line()
    m1 = line.machines["M1"]
    _close_lid(line, 8.0)                    # latched but empty: nothing to run
    assert m1.latched and m1.state == EMPTY
    line2 = make_line()
    seat(line2)
    run(line2, 0.3)
    assert line2.machines["M1"].state == LOADED      # seated, lid not latched: waits


def test_a_full_cycle_finishes_the_part_and_unlocks_the_lid():
    line = make_line(cycle_s=2.0)
    m1 = line.machines["M1"]
    cube = seat(line)
    run(line, 0.3)
    assert m1.state == LOADED
    _close_lid(line, 8.0)
    assert m1.state in (RUNNING, DONE)
    run(line, 2.5)
    assert m1.state == DONE and not m1.latched
    np.testing.assert_allclose(line.model.geom_rgba[cube.geom], m1.finished_rgba)
    # the lid unlocked does not re-latch by itself
    run(line, 0.5)
    assert not m1.latched


def test_losing_the_part_mid_cycle_is_a_fault():
    line = make_line(cycle_s=5.0)
    m1 = line.machines["M1"]
    cube = seat(line)
    run(line, 0.3)
    _close_lid(line, 8.0)
    assert m1.state == RUNNING
    line.data.qpos[cube.qpos + 2] += 0.05                 # the part lifted off its seat
    mujoco.mj_forward(line.model, line.data)
    run(line, 0.2)
    assert m1.state == FAULT


@pytest.mark.parametrize("slide,travel", [("right", 0.12), ("out", 0.25)])
def test_the_scripted_controller_tends_the_machine_through_a_whole_cycle(slide, travel):
    # (a cycle longer than letting go of the handle and rising, about 3 s, so that it is still
    # running when the controller is back at the carry height)
    line = make_line(slide=slide, travel=travel, cycle_s=8.0)
    m, d = line.model, line.data
    m1 = line.machines["M1"]
    ctl = ScriptedTending(line)
    arm_bodies = {b for b in range(m.nbody) if m.body(b).name.startswith(("link", "hand"))
                  or "finger" in m.body(b).name}
    machine_body = {m1.body}       # walls, roof and pocket (not the lid or its handle)
    hits = []

    def watch():
        line.update_machines()
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if {b1, b2} & arm_bodies and {b1, b2} & machine_body:
                hits.append((m.geom(c.geom1).name, m.geom(c.geom2).name))

    ctl.on_step = watch
    cube = line.cubes[0]
    spot = line.line.station_xy(-0.6, -30.0)
    d.qpos[cube.qpos:cube.qpos + 7] = [*spot, cube.half_size[2], *_yaw_quat(np.radians(-30))]
    mujoco.mj_forward(m, d)
    ctl.go_to("M1")
    ctl.move_lid("M1", opening=True)
    assert m1.lid == pytest.approx(travel - ctl.open_margin, abs=0.004)   # stops short of the stop
    ctl.swing_to(np.radians(-30.0))
    ctl.pick(cube)
    assert line.attached(cube)
    ctl.swing_to(np.radians(-90.0))
    ctl.load(cube, "M1")
    ctl.step(10)
    assert m1.state == LOADED
    ctl.move_lid("M1", opening=False)
    ctl.step(5)
    assert m1.latched and m1.state == RUNNING
    while m1.state == RUNNING:
        ctl.step()
    assert m1.state == DONE
    ctl.move_lid("M1", opening=True)
    ctl.swing_to(np.radians(-90.0))
    ctl.unload(cube)
    assert line.attached(cube) and d.xpos[cube.body][2] > 0.25
    ctl.step(5)
    assert m1.state == EMPTY
    assert ctl.ik_failures == 0
    assert not hits, f"the arm touched the machine's body: {sorted(set(hits))[:5]}"
