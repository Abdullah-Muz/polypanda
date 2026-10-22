import mujoco
import numpy as np
import pytest

from polypanda.control.action_adapter import JointDeltaActionAdapter
from polypanda.simulation.assets import load_panda_scene
from polypanda.simulation.panda import PandaIndices, reset_to_home

ZERO = np.zeros(7)


@pytest.fixture
def panda():
    model, data = load_panda_scene()
    idx = PandaIndices.from_model(model)
    reset_to_home(model, data, idx)
    adapter = JointDeltaActionAdapter(model, idx)
    adapter.reset_to_measured(data)
    return model, data, idx, adapter


def test_control_interval_is_20hz(panda):
    _, data, _, adapter = panda
    assert adapter.control_dt == pytest.approx(0.05)
    adapter.step(data, ZERO)
    assert data.time == pytest.approx(0.05)


def test_zero_action_holds_targets(panda):
    _, data, idx, adapter = panda
    start = adapter.q_cmd.copy()
    for _ in range(40):
        adapter.step(data, ZERO)
    np.testing.assert_array_equal(adapter.q_cmd, start)
    np.testing.assert_array_equal(data.ctrl[idx.arm_actuators], start)


def test_full_action_moves_each_target_by_max_delta(panda):
    _, data, _, adapter = panda
    start = adapter.q_cmd.copy()
    signs = np.array([1, -1, 1, 1, -1, -1, 1], dtype=float)
    adapter.apply(data, signs)
    np.testing.assert_allclose(adapter.q_cmd - start, 0.015 * signs)


def test_out_of_range_actions_are_clipped(panda):
    _, data, _, adapter = panda
    start = adapter.q_cmd.copy()
    adapter.apply(data, np.full(7, 5.0))
    np.testing.assert_allclose(adapter.q_cmd - start, np.full(7, 0.015))


def test_targets_saturate_at_ctrlrange(panda):
    _, data, _, adapter = panda
    push = np.zeros(7)
    push[0] = 1.0
    for _ in range(400):  # 400 * 0.015 = 6 rad, far beyond joint1's range
        adapter.apply(data, push)
    assert adapter.q_cmd[0] == adapter.high[0]
    assert adapter.at_limit[0] and not adapter.at_limit[1:].any()


@pytest.mark.parametrize("bad", [np.zeros(6), np.zeros(8), np.zeros((1, 7))])
def test_wrong_shape_rejected(panda, bad):
    _, data, _, adapter = panda
    with pytest.raises(ValueError, match="shape"):
        adapter.apply(data, bad)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_rejected_without_side_effects(panda, bad_value):
    _, data, idx, adapter = panda
    before_cmd, before_ctrl = adapter.q_cmd.copy(), data.ctrl.copy()
    action = np.zeros(7)
    action[3] = bad_value
    with pytest.raises(ValueError, match="non-finite"):
        adapter.apply(data, action)
    np.testing.assert_array_equal(adapter.q_cmd, before_cmd)
    np.testing.assert_array_equal(data.ctrl, before_ctrl)


def test_gripper_command_untouched(panda):
    _, data, idx, adapter = panda
    data.ctrl[idx.gripper_actuator] = 123.0
    rng = np.random.default_rng(0)
    for _ in range(20):
        adapter.step(data, rng.uniform(-1, 1, 7))
    assert data.ctrl[idx.gripper_actuator] == 123.0


def test_servo_follows_integrated_target(panda):
    _, data, idx, adapter = panda
    push = np.zeros(7)
    push[0] = 1.0
    start_q = data.qpos[idx.arm_qpos[0]]
    for _ in range(20):  # 20 * 0.015 = 0.3 rad commanded over 1 s
        adapter.step(data, push)
    for _ in range(20):  # 1 s to settle
        adapter.step(data, ZERO)
    assert data.qpos[idx.arm_qpos[0]] - start_q == pytest.approx(0.3, abs=0.01)


def test_reset_to_measured_matches_joints(panda):
    model, data, idx, adapter = panda
    for _ in range(40):  # let gravity sag open a gap between q_cmd and q
        adapter.step(data, ZERO)
    adapter.reset_to_measured(data)
    np.testing.assert_array_equal(adapter.q_cmd, data.qpos[idx.arm_qpos])
    np.testing.assert_array_equal(data.ctrl[idx.arm_actuators], adapter.q_cmd)


def test_adopt_ctrl_preserves_commands_across_handoff(panda):
    model, data, idx, adapter = panda
    adapter.step(data, np.full(7, 0.5))
    commanded = data.ctrl[idx.arm_actuators].copy()
    successor = JointDeltaActionAdapter(model, idx)
    successor.adopt_ctrl(data)
    np.testing.assert_array_equal(successor.q_cmd, commanded)
    successor.apply(data, ZERO)
    np.testing.assert_array_equal(data.ctrl[idx.arm_actuators], commanded)


def test_max_lead_bounds_target_ahead_of_joints(panda):
    model, data, idx, _ = panda
    adapter = JointDeltaActionAdapter(model, idx, max_lead=0.01)
    adapter.reset_to_measured(data)
    q = data.qpos[idx.arm_qpos].copy()
    for _ in range(5):  # without stepping physics the joints cannot follow
        adapter.apply(data, np.ones(7))
    assert np.all(adapter.q_cmd - q <= 0.01 + 1e-12)


def test_physics_stays_finite_under_random_actions(panda):
    _, data, _, adapter = panda
    rng = np.random.default_rng(1)
    for _ in range(400):  # 20 s of random flailing
        adapter.step(data, rng.uniform(-1, 1, 7))
        assert np.all(adapter.q_cmd >= adapter.low) and np.all(adapter.q_cmd <= adapter.high)
    assert np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
    assert data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number == 0
