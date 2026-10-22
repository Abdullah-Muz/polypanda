import mujoco
import numpy as np
import pytest

from polypanda.control.classical_baseline import JacobianIKPolicy, RandomPolicy
from polypanda.environments.reach import ReachEnv
from polypanda.simulation.panda import tcp_position


@pytest.fixture(scope="module")
def ik():
    return JacobianIKPolicy()


def test_random_policy_actions():
    a, _ = RandomPolicy(seed=0).predict(np.zeros(27))
    b, _ = RandomPolicy(seed=0).predict(np.zeros(27))
    assert a.shape == (7,) and a.dtype == np.float32
    assert np.all(np.abs(a) <= 1) and np.array_equal(a, b)
    batch, _ = RandomPolicy(seed=0).predict(np.zeros((4, 27)))
    assert batch.shape == (4, 7)


def test_jacobian_matches_finite_differences(ik):
    model, data, idx = ik.model, mujoco.MjData(ik.model), ik.idx
    mujoco.mj_resetDataKeyframe(model, data, idx.home_key)
    q = data.qpos[idx.arm_qpos].copy() + np.array([0.1, -0.2, 0.3, 0.2, -0.1, 0.2, 0.3])
    J = ik.jacobian(q)
    eps = 1e-6
    for j in range(7):
        for sign, store in ((1, "plus"), (-1, "minus")):
            data.qpos[idx.arm_qpos] = q
            data.qpos[idx.arm_qpos[j]] += sign * eps
            mujoco.mj_kinematics(model, data)
            if store == "plus":
                plus = tcp_position(data, idx)
            else:
                minus = tcp_position(data, idx)
        np.testing.assert_allclose(J[:, j], (plus - minus) / (2 * eps), atol=1e-6)


def test_ik_does_not_touch_environment_state(ik):
    env = ReachEnv()
    obs, _ = env.reset(seed=0)
    before = env.data.qpos.copy()
    ik.predict(obs)
    np.testing.assert_array_equal(env.data.qpos, before)


def test_ik_batched_matches_single(ik):
    env = ReachEnv()
    obs = np.stack([env.reset(seed=s)[0] for s in range(3)])
    batch, _ = ik.predict(obs)
    for i in range(3):
        np.testing.assert_array_equal(batch[i], ik.predict(obs[i])[0])


def test_ik_actions_are_valid(ik):
    env = ReachEnv()
    obs, _ = env.reset(seed=0)
    a, _ = ik.predict(obs)
    assert a.shape == (7,) and a.dtype == np.float32
    assert np.abs(a).max() == pytest.approx(1.0)  # far from target: speed-limited, direction kept


@pytest.mark.parametrize("seed", range(1000, 1005))
def test_ik_reaches_tuning_targets(ik, seed):
    env = ReachEnv()
    obs, _ = env.reset(seed=seed)
    while True:
        obs, _, terminated, truncated, info = env.step(ik.predict(obs)[0])
        if terminated or truncated:
            break
    assert info["is_success"]
    assert info["distance"] < env.config.success.tolerance


def test_ik_integral_action_removes_gravity_sag(ik):
    """Holding past success, the integrated targets cancel the servos' steady-state sag."""
    env = ReachEnv()
    obs, _ = env.reset(seed=1000)
    for _ in range(120):  # 6 s, ignoring termination
        obs, *_ = env.step(ik.predict(obs)[0])
    assert np.linalg.norm(env.target - tcp_position(env.data, env.idx)) < 1e-4
