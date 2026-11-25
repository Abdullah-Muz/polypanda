"""v4.1: the electronics look (simulation/electronics.py) is visual only: the line's physics is the
same with either theme."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from polypanda.control.scripted_tending import ScriptedTending
from polypanda.simulation.line import Line, LineConfig, MachineSpot, build_line_spec
from polypanda.simulation.machine import EMPTY, RUNNING, MachineConfig


def _colliding(model):
    return sorted(model.geom(g).name for g in range(model.ngeom)
                  if model.geom_contype[g] or model.geom_conaffinity[g])


def test_the_electronics_theme_adds_no_physics():
    plain = build_line_spec(LineConfig()).compile()
    themed = build_line_spec(LineConfig(theme="electronics")).compile()
    assert themed.ngeom > plain.ngeom                     # it does add things to look at
    for attr in ("nq", "nv", "nu", "nbody", "njnt", "neq"):
        assert getattr(themed, attr) == getattr(plain, attr), attr
    assert _colliding(themed) == _colliding(plain)
    np.testing.assert_allclose(themed.body_mass, plain.body_mass)


def test_the_line_moves_the_same_with_either_theme():
    a = Line.build_line(LineConfig())
    b = Line.build_line(LineConfig(theme="electronics"))
    rng = np.random.default_rng(0)
    for _ in range(40):
        ctrl = a.data.ctrl.copy()
        ctrl[: a.model.nu] += rng.normal(0.0, 0.01, a.model.nu)
        a.data.ctrl[:] = ctrl
        b.data.ctrl[:] = ctrl
        mujoco.mj_step(a.model, a.data, nstep=10)
        mujoco.mj_step(b.model, b.data, nstep=10)
    np.testing.assert_allclose(b.data.qpos, a.data.qpos, atol=1e-9)


def _tester_line(kind: str, theme: str) -> Line:
    cfg = MachineConfig(kind=kind, cycle_s=2.0)
    return Line.build_line(LineConfig(machines=(MachineSpot("M1", -0.6, config=cfg),),
                                      theme=theme))


@pytest.mark.parametrize("kind", ["drawer", "pneumatic"])
def test_the_shield_box_look_of_v4_1_testers_is_visual_only(kind):
    a, b = _tester_line(kind, "factory"), _tester_line(kind, "electronics")
    for attr in ("nq", "nv", "nu", "nbody", "njnt", "neq"):
        assert getattr(a.model, attr) == getattr(b.model, attr), attr
    # every moving body keeps its mass and inertia (the tower's lamps, resized, are on the
    # machine's static body, where mass has no effect)
    moving = a.model.body_weldid != 0
    np.testing.assert_array_equal(b.model.body_weldid, a.model.body_weldid)
    np.testing.assert_allclose(b.model.body_mass[moving], a.model.body_mass[moving])
    np.testing.assert_allclose(b.model.body_inertia[moving], a.model.body_inertia[moving])
    assert _colliding(b.model) == _colliding(a.model)
    for line in (a, b):
        if kind == "pneumatic":
            line.machines["M1"].request_open()
        ctl = ScriptedTending(line)
        for _ in range(int(1.0 / ctl.dt)):
            ctl.step()
    np.testing.assert_allclose(b.data.qpos, a.data.qpos, atol=1e-9)


def test_a_testers_light_tower_shows_its_state():
    line = _tester_line("drawer", "electronics")
    m1, m = line.machines["M1"], line.model
    green, amber = m.geom("M1_lamp_green").id, m.geom("M1_lamp_amber").id
    assert m1.state == EMPTY
    lit = m.geom_rgba[green].copy()
    assert m.geom_pos[green][2] > m1.cfg.height + 0.05      # on a pole above the roof
    m1.state = RUNNING
    m1._show()
    assert not np.allclose(m.geom_rgba[green], lit)         # green off while testing
    assert m.geom_rgba[amber][0] > 0.9                      # amber on


def test_an_unknown_theme_is_refused():
    with pytest.raises(ValueError):
        build_line_spec(LineConfig(theme="spaceship"))


def test_without_dressing_the_theme_draws_nothing():
    cfg = LineConfig(theme="electronics")
    cfg = replace(cfg, cell=replace(cfg.cell, dressing=False))
    assert build_line_spec(cfg).compile().ngeom == build_line_spec(
        replace(cfg, theme="factory")).compile().ngeom
