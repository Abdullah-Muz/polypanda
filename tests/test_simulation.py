import mujoco
import numpy as np

from polypanda.simulation.assets import load_panda_scene
from polypanda.simulation.panda import PandaIndices, reset_to_home

N_STEPS = 2000  # 4 s of simulated time at 0.002 s


def _rollout(model, data, idx, n=N_STEPS):
    reset_to_home(model, data, idx)
    for _ in range(n):
        mujoco.mj_step(model, data)
    return data.qpos.copy(), data.qvel.copy()


def test_home_hold_is_stable_and_finite():
    model, data = load_panda_scene()
    idx = PandaIndices.from_model(model)
    reset_to_home(model, data, idx)
    home_q = data.qpos[idx.arm_qpos].copy()
    qpos, qvel = _rollout(model, data, idx)
    assert np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel))
    assert np.abs(qpos[idx.arm_qpos] - home_q).max() < 1e-2


def test_reset_is_deterministic():
    model, data = load_panda_scene()
    idx = PandaIndices.from_model(model)
    first = _rollout(model, data, idx)
    second = _rollout(model, data, idx)  # same MjData, reset in between
    fresh = _rollout(model, mujoco.MjData(model), idx)
    for a, b, c in zip(first, second, fresh, strict=True):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, c)


def test_servo_tracks_a_target_step():
    model, data = load_panda_scene()
    idx = PandaIndices.from_model(model)
    reset_to_home(model, data, idx)
    target = data.ctrl[idx.arm_actuators].copy()
    target[0] += 0.2  # rotate the base joint by 0.2 rad
    data.ctrl[idx.arm_actuators] = target
    for _ in range(1500):  # 3 s
        mujoco.mj_step(model, data)
    assert abs(data.qpos[idx.arm_qpos[0]] - target[0]) < 0.02
