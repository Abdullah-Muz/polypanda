"""v4.1: the electronics look (simulation/electronics.py) is visual only: the line's physics is the
same with either theme."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from polypanda.simulation.line import Line, LineConfig, build_line_spec


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


def test_an_unknown_theme_is_refused():
    with pytest.raises(ValueError):
        build_line_spec(LineConfig(theme="spaceship"))


def test_without_dressing_the_theme_draws_nothing():
    cfg = LineConfig(theme="electronics")
    cfg = replace(cfg, cell=replace(cfg.cell, dressing=False))
    assert build_line_spec(cfg).compile().ngeom == build_line_spec(
        replace(cfg, theme="factory")).compile().ngeom
