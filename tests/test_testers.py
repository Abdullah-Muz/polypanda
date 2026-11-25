"""v4.1: the three tester kinds (docs/v4_1_plan.md, section 1): A, v4.0's sliding lid; B, a drawer
that slides toward the robot (simulation/machine.py, the door env and its scripts); C, a pneumatic
lid that opens and closes itself. (That the lid stays exactly v4.0's: test_lid_identity.py.)"""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from polypanda.control.scripted_tending import ScriptedTending
from polypanda.simulation.line import Line, LineConfig, MachineSpot
from polypanda.simulation.machine import (
    DONE,
    DRAWER_SLIDE,
    LOADED,
    RUNNING,
    MachineConfig,
    add_machine,
)
from polypanda.simulation.machine_variation import (
    MachineLayout,
    draw_pneumatic,
    needed_opening,
    variant,
)
from polypanda.simulation.workcell import _yaw_quat


def make_line(kind: str, **machine) -> Line:
    cfg = MachineConfig(**{"kind": kind, "cycle_s": 2.0, **machine})
    return Line.build_line(LineConfig(machines=(MachineSpot("M1", -0.6, config=cfg),)))


def run(line: Line, seconds: float) -> None:
    """Physics with the machines updated every control step, arm held (as tests/test_machine)."""
    ctl = ScriptedTending(line)
    for _ in range(int(seconds / ctl.dt)):
        ctl.step()


def seat(line: Line):
    """A cube upright in M1's nest, square to it, at rest."""
    m1, d = line.machines["M1"], line.data
    cube = line.cubes[0]
    p = m1.pocket_xyz()
    R = d.xmat[m1.nest_body].reshape(3, 3)
    d.qpos[cube.qpos:cube.qpos + 7] = [p[0], p[1], p[2] + cube.half_size[2] + 0.0005,
                                       *_yaw_quat(np.arctan2(R[1, 0], R[0, 0]))]
    d.qvel[:] = 0.0
    mujoco.mj_forward(line.model, d)
    return cube


def machine_contacts(model, data, name: str) -> list[tuple[str, str]]:
    """Contacts between two of the machine's own geoms."""
    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        a, b = model.geom(c.geom1).name, model.geom(c.geom2).name
        if a.startswith(f"{name}_") and b.startswith(f"{name}_"):
            out.append((a, b))
    return out


# --- configs -------------------------------------------------------------------------------
def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError):
        MachineConfig(kind="carousel")


def test_a_drawer_slides_toward_the_robot_only():
    cfg = MachineConfig(kind="drawer")
    assert cfg.slide == DRAWER_SLIDE and cfg.nest_inset == pytest.approx(cfg.travel)
    np.testing.assert_allclose(cfg.slide_axis, [-1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        MachineConfig(kind="drawer", slide="left")
    with pytest.raises(ValueError):
        MachineConfig(slide=DRAWER_SLIDE)                 # a lid never slides toward the robot


def test_a_drawer_that_does_not_fit_is_refused():
    with pytest.raises(ValueError):
        MachineConfig(kind="drawer", drawer_width=0.06)  # the nest does not fit in the tray
    with pytest.raises(ValueError):
        MachineConfig(kind="drawer", height=0.07)        # a part does not fit under the roof


# --- the drawer ----------------------------------------------------------------------------
def test_open_the_drawer_brings_its_nest_out_to_the_machines_place():
    line = make_line("drawer")
    m1, d = line.machines["M1"], line.data
    shut = m1.pocket_xyz()
    d.qpos[m1.door_qpos] = m1.cfg.travel
    mujoco.mj_forward(line.model, d)
    # at the open stop the nest is where a lid machine's pocket would be: the station
    np.testing.assert_allclose(m1.pocket_xyz()[:2], [-0.6, -0.5], atol=1e-9)
    # it came toward the robot (the robot stands at y = 0)
    assert shut[1] < m1.pocket_xyz()[1]
    np.testing.assert_allclose(np.linalg.norm(shut - m1.pocket_xyz()), m1.cfg.travel, atol=1e-9)


@pytest.mark.parametrize("q", [0.0, 0.05, 0.12])
def test_the_drawer_handle_site_matches_its_config(q):
    line = make_line("drawer")
    m1, d = line.machines["M1"], line.data
    d.qpos[m1.door_qpos] = q
    mujoco.mj_forward(line.model, d)
    R, p = d.xmat[m1.body].reshape(3, 3), d.xpos[m1.body]
    np.testing.assert_allclose(m1.handle_xyz(), p + R @ m1.cfg.handle_centre(q), atol=1e-9)


@pytest.mark.parametrize("q", [0.0, 0.001, 0.06, 0.12])
def test_the_drawer_slides_without_touching_its_box(q):
    line = make_line("drawer")
    m1, d = line.machines["M1"], line.data
    d.qpos[m1.door_qpos] = q
    mujoco.mj_forward(line.model, d)
    touching = machine_contacts(line.model, d, "M1")
    if q == 0.0:          # shut, the front plate rests against the box's front (its stop)
        assert all("M1_drawer_face" in pair for pair in touching)
    else:
        assert touching == []


def test_the_drawer_needs_its_nest_clear_of_the_box():
    cfg = MachineConfig(kind="drawer")
    need = needed_opening(cfg)
    assert need < cfg.travel
    # opened by `need`, the nest's back edge is drawer_room in front of the box's front
    front = cfg.nest_inset - cfg.drawer_front
    nest_back = cfg.nest_inset - need + cfg.nest_half
    assert front - nest_back == pytest.approx(cfg.drawer_room)


def test_a_part_rides_in_the_drawer_and_runs_a_cycle():
    line = make_line("drawer")
    m1, d = line.machines["M1"], line.data
    d.qpos[m1.door_qpos] = m1.cfg.travel
    mujoco.mj_forward(line.model, d)
    cube = seat(line)
    run(line, 0.5)
    assert m1.state == LOADED and m1.in_pocket(cube)[1]
    # push the drawer shut at about 7 cm/s, firmly enough to latch: the part goes in with it
    for _ in range(200):
        d.qfrc_applied[m1.door_dof] = -50.0 * (d.qvel[m1.door_dof] + 0.08)
        mujoco.mj_step(line.model, d, nstep=5)
        m1.update()
        if m1.latched:
            break
    d.qfrc_applied[m1.door_dof] = 0.0
    assert m1.latched and m1.lid < 0.003
    run(line, 0.3)
    assert m1.state == RUNNING and m1.in_pocket(cube)[1]
    run(line, 2.2)
    assert m1.state == DONE and not m1.latched


def _push_shut(line: Line, force: float, seconds: float) -> None:
    """Push the drawer toward shut with `force` (N), physics only."""
    m1, d = line.machines["M1"], line.data
    for _ in range(int(seconds / line.model.opt.timestep)):
        d.qfrc_applied[m1.door_dof] = -force
        mujoco.mj_step(line.model, d)
    d.qfrc_applied[m1.door_dof] = 0.0


def test_a_drawers_index_pin_holds_it_at_its_open_stop():
    """A free slide gives way to any push on the nest (a part pressed on its chamfer shoves it
    shut); the index pin, engaged by the cell near the open stop, pulls the drawer onto the stop
    and holds it there within a couple of millimetres under 40 N."""
    line = make_line("drawer")
    m1, d = line.machines["M1"], line.data
    travel = m1.cfg.travel
    assert not m1.indexed
    d.qpos[m1.door_qpos] = travel - 0.05
    mujoco.mj_forward(line.model, d)
    assert not m1.index_open() and not m1.indexed        # too far from the stop to catch
    d.qpos[m1.door_qpos] = travel - 0.006
    mujoco.mj_forward(line.model, d)
    assert m1.index_open() and m1.indexed
    _push_shut(line, 0.0, 0.3)
    assert m1.lid == pytest.approx(travel, abs=0.0005)    # pulled onto its stop
    _push_shut(line, 40.0, 0.5)
    assert m1.lid > travel - 0.002
    m1.release_index()
    assert not m1.indexed
    _push_shut(line, 40.0, 0.2)
    assert m1.lid < travel - 0.02                         # free again: shoved shut
    m1.index_open(catch=1.0)
    m1.reset()
    assert not m1.indexed                                 # a reset withdraws it
    lid = make_line("lid").machines["M1"]
    with pytest.raises(ValueError):
        lid.index_open()
    lid.release_index()                                   # (no-op without a pin)
    assert not lid.indexed


def test_a_drawer_is_reached_into_across_it_and_its_nest_stands_on_its_tray():
    """The hand (about 200 x 65 mm) passes a lid's opening lengthwise (fingers closing along the
    machine's x) but must reach into a drawer with its fingers closing across it (its y), between
    the front plate and the box; a drawer's nest stands on its tray, not on the bench."""
    lid, drawer = MachineConfig(), MachineConfig(kind="drawer")
    assert lid.hand_axis == 0.0 and MachineConfig(kind="pneumatic").hand_axis == 0.0
    assert drawer.hand_axis == pytest.approx(np.pi / 2)
    assert lid.nest_floor == 0.0 and MachineConfig(kind="pneumatic").nest_floor == 0.0
    assert drawer.nest_floor == drawer.drawer_floor > 0.0
    line = make_line("drawer")
    m1 = line.machines["M1"]
    assert m1.pocket_xyz()[2] == pytest.approx(drawer.nest_floor)


# --- the pneumatic tester ------------------------------------------------------------------
def test_a_pneumatic_tester_has_no_handle_and_opens_on_a_signal():
    line = make_line("pneumatic")
    m1 = line.machines["M1"]
    with pytest.raises(ValueError):
        m1.handle_xyz()
    assert m1.request_open()
    run(line, 0.6)
    assert 0.02 < m1.lid < m1.cfg.travel                # it runs at a steady, limited speed
    run(line, 1.4)
    assert m1.lid == pytest.approx(m1.cfg.travel, abs=0.003)
    assert m1.request_close()
    run(line, 2.0)
    assert m1.latched and m1.lid < 0.003


def test_a_pneumatic_tester_runs_a_cycle_and_opens_itself():
    line = make_line("pneumatic")
    m1 = line.machines["M1"]
    m1.request_open()
    run(line, 2.0)
    cube = seat(line)
    run(line, 0.3)
    assert m1.state == LOADED
    m1.request_close()
    run(line, 1.8)                                       # 0.12 m at 0.08 m/s, then it latches
    assert m1.state == RUNNING
    assert not m1.request_open()                         # the interlock: refused while testing
    run(line, 2.0)                                       # (cycle_s 2 s since it latched)
    assert m1.state == DONE
    run(line, 2.0)
    assert m1.lid == pytest.approx(m1.cfg.travel, abs=0.003) and m1.in_pocket(cube)[1]


def test_drawn_pneumatic_testers_open_and_latch():
    rng = np.random.default_rng(0)
    for _ in range(2):
        cfg = draw_pneumatic(rng)
        assert cfg.kind == "pneumatic" and 0.06 <= cfg.pneumatic_speed <= 0.1
        line = Line.build_line(LineConfig(machines=(MachineSpot("M1", -0.6, config=cfg),)))
        m1 = line.machines["M1"]
        m1.request_open()
        run(line, 2.6)
        assert m1.lid == pytest.approx(cfg.travel, abs=0.003)
        m1.request_close()
        run(line, 2.6)
        assert m1.latched


def test_a_lid_machine_has_no_cylinder():
    m1 = make_line("lid").machines["M1"]
    with pytest.raises(ValueError):
        m1.request_open()


# --- runtime variants ----------------------------------------------------------------------
@pytest.mark.parametrize("kind,changes", [
    ("drawer", {"travel": 0.135, "handle_offset": 0.02, "handle_shift": -0.004,
                "handle_height": 0.045, "lid_friction": 2.0}),
    ("pneumatic", {"travel": 0.13, "lid_damping": 6.0}),
])
def test_a_variant_applied_at_runtime_matches_one_compiled(kind, changes):
    compiled = MachineConfig(kind=kind)
    spec = mujoco.MjSpec()
    add_machine(spec, "M", (0.5, 0.0), 0.0, compiled)
    model = spec.compile()
    cfg = variant(compiled, **changes)
    MachineLayout(model, "M", compiled).apply(model, cfg)
    spec2 = mujoco.MjSpec()
    add_machine(spec2, "M", (0.5, 0.0), 0.0, cfg)
    want = spec2.compile()
    for attr in ("geom_pos", "geom_quat", "geom_size", "site_pos", "dof_frictionloss",
                 "dof_damping", "jnt_range", "actuator_actrange"):
        np.testing.assert_allclose(getattr(model, attr), getattr(want, attr), atol=1e-12,
                                   err_msg=attr)


# --- the door env and its scripts on drawers ------------------------------------------------
def drawer_env(kinds=("drawer",)):
    from polypanda.environments.door import DoorConfig, DoorEnv
    cfg = DoorConfig()
    return DoorEnv(replace(cfg, variation=replace(cfg.variation, kinds=kinds)))


def test_a_lid_only_env_draws_no_kind_and_refuses_a_drawer_episode():
    from polypanda.environments.door import DoorEnv
    env = DoorEnv()
    ep = env.draw_episode(np.random.default_rng(0))
    assert "kind" not in ep
    drawer = drawer_env().draw_episode(np.random.default_rng(0))
    assert drawer["kind"] == "drawer" and drawer["slide"] == DRAWER_SLIDE
    with pytest.raises(ValueError):
        env.reset(seed=0, options={"episode": drawer})


def test_a_mixed_env_draws_both_kinds_and_parks_the_other():
    env = drawer_env(("lid", "drawer"))
    rng = np.random.default_rng(1)
    eps = [env.draw_episode(rng) for _ in range(40)]
    kinds = {e["kind"] for e in eps}
    assert kinds == {"lid", "drawer"}
    for ep in (next(e for e in eps if e["kind"] == k) for k in ("lid", "drawer")):
        env.reset(seed=0, options={"episode": ep})
        assert env.kind == ep["kind"]
        near = env.machine.pocket_xyz()
        other = env.machines["drawer" if ep["kind"] == "lid" else "lid"]
        assert np.linalg.norm(other.pocket_xyz()[:2] - near[:2]) > 3.0


def test_a_drawer_focus_cell_is_drawn_and_checked():
    from polypanda.environments.door import DoorConfig, DoorEnv, DoorFocusConfig
    focus = DoorFocusConfig(prob=1.0, cells=("open/toward/plain",))
    env = drawer_env(("lid", "drawer"))
    env2 = DoorEnv(replace(env.config, focus=focus))
    ep = env2.draw_episode(np.random.default_rng(0))
    assert ep["kind"] == "drawer" and ep["command"] == "open" and ep["spring"] == 0.0
    with pytest.raises(ValueError):                       # a drawer cell, but lids only
        DoorEnv(replace(DoorConfig(), focus=focus))


@pytest.mark.parametrize("script,command", [
    ("expert", "open"), ("expert", "close"), ("feel_v2", "open"), ("feel_v2", "close"),
    ("camera", "open"),       # (the camera script closes at the nominal latch speed, no retry:
])                            # it fails on a stiff latch, as on lids)
def test_the_door_scripts_open_and_close_a_drawer(script, command):
    from polypanda.control.door_baselines import BaselineDoor
    from polypanda.control.scripted_door import DoorExpert
    env = drawer_env()
    actor = DoorExpert(env) if script == "expert" else BaselineDoor(env, script)
    ep = env.draw_episode(np.random.default_rng(3), command)
    obs, _ = env.reset(seed=3, options={"episode": ep})
    actor.reset()
    info = {}
    for _ in range(env.config.horizon):
        obs, _, term, trunc, info = env.step(actor.act())
        if term or trunc:
            break
    assert info["is_success"], info["failure"]
    assert info["stats"]["collision_steps"] == 0 and info["stats"]["slams"] == 0
