import mujoco
import numpy as np
import pytest

from polypanda.control.scripted_place import ScriptedPlace
from polypanda.environments.grasp import GraspEnv
from polypanda.evaluation.grasp import ScriptedDriver
from polypanda.evaluation.robustness import (
    CONDITIONS,
    OOD_BAND,
    Perturbation,
    ScenePhysics,
    ood_scenes,
    sample_perturbation,
)
from polypanda.evaluation.sets import generate_pickplace_eval_set
from polypanda.evaluation.transport import ScriptedCarryDriver
from polypanda.simulation.scenes import CUBE_X_RANGE, CUBE_Y_RANGE
from polypanda.supervisor.fsm import PickPlaceSupervisor

CUBE, TARGET = (0.45, -0.10, 0.3), (0.55, 0.12)


@pytest.fixture(scope="module")
def supervisor():
    return PickPlaceSupervisor(ScriptedDriver(), ScriptedCarryDriver(), ScriptedPlace())


def test_perturbations_touch_only_their_condition():
    rng = np.random.default_rng(0)
    nominal = Perturbation()
    for condition in CONDITIONS:
        p = sample_perturbation(rng, condition)
        changed = {k for k, v in vars(p).items() if v != getattr(nominal, k)}
        expected = {
            "mass": {"mass_scale"}, "friction": {"friction_scale"}, "size": {"size_scale"},
            "joints": {"arm_offset"}, "push": {"push_force_n", "push_angle", "push_step"},
            "ood": set(),
        }.get(condition)
        if expected is not None:
            assert changed == expected, condition


def test_nominal_perturbation_restores_the_model_exactly(supervisor):
    env = supervisor.envs[next(iter(supervisor.envs))]
    before = (env.model.geom_friction.copy(), env.model.geom_size.copy(),
              env.model.body_mass.copy(), env.model.body_inertia.copy(),
              env.model.dof_invweight0.copy())
    physics = ScenePhysics(env)
    physics.apply(Perturbation(mass_scale=2.0, friction_scale=0.5, size_scale=1.15))
    assert env.cube.half_size[0] == pytest.approx(0.023)
    physics.apply(Perturbation())
    after = (env.model.geom_friction, env.model.geom_size, env.model.body_mass,
             env.model.body_inertia, env.model.dof_invweight0)
    for a, b in zip(before, after, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-12)


def test_size_scaling_keeps_density_and_rests_on_the_table():
    env = GraspEnv()
    physics = ScenePhysics(env)
    m0 = float(env.model.body_mass[env.cube.body])
    physics.apply(Perturbation(size_scale=1.1))
    assert env.model.body_mass[env.cube.body] == pytest.approx(m0 * 1.1**3)
    env.reset(options={"cube": CUBE})
    for _ in range(10):
        mujoco.mj_step(env.model, env.data)
    assert env.data.xpos[env.cube.body][2] == pytest.approx(0.033, abs=5e-4)


def test_lower_friction_lets_a_heavy_cube_slip_out(supervisor):
    physics = [ScenePhysics(e) for e in supervisor.envs.values()]
    for ph in physics:
        ph.apply(Perturbation(mass_scale=2.0, friction_scale=0.5))
    try:
        r = supervisor.run(CUBE, TARGET)
    finally:
        for ph in physics:
            ph.apply(Perturbation())
    assert not r.success


def test_arm_offset_moves_the_start_pose():
    env = GraspEnv()
    env.reset(options={"cube": CUBE})
    home = env.data.qpos[env.arm.arm_qpos].copy()
    env.reset(options={"cube": CUBE, "arm_offset": [0.05] * 7})
    np.testing.assert_allclose(env.data.qpos[env.arm.arm_qpos] - home, 0.05, atol=1e-9)
    np.testing.assert_allclose(env.adapter.q_cmd, env.data.qpos[env.arm.arm_qpos])


def test_ood_scenes_lie_in_the_band_outside_the_region():
    eval_set = generate_pickplace_eval_set("t", 50, 0)
    poses = ood_scenes(eval_set, np.random.default_rng(1))
    x, y = poses[:, 0], poses[:, 1]
    inside = ((CUBE_X_RANGE[0] <= x) & (x <= CUBE_X_RANGE[1])
              & (CUBE_Y_RANGE[0] <= y) & (y <= CUBE_Y_RANGE[1]))
    assert not inside.any()
    assert (x >= CUBE_X_RANGE[0] - OOD_BAND).all() and (x <= CUBE_X_RANGE[1] + OOD_BAND).all()
    assert (np.abs(y) <= CUBE_Y_RANGE[1] + OOD_BAND).all()
