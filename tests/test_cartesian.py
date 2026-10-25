"""Cartesian (hand-motion) actions: the policy moves the gripper, IK moves the joints (M8)."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from gymnasium.utils.env_checker import check_env

from polypanda.control.action_adapter import (
    CartesianActionAdapter,
    JointDeltaActionAdapter,
    action_schema,
    make_action_adapter,
)
from polypanda.control.ik import closing_yaw
from polypanda.environments.grasp import GraspConfig, GraspV2Env, gripper_tilt
from polypanda.environments.place import PlaceConfig, PlaceEnv
from polypanda.environments.reach import ActionConfig, CartesianConfig
from polypanda.environments.transport import TransportConfig, TransportEnv, TransportResidualEnv
from polypanda.policies.loader import IncompatiblePolicyError, check_compatible, load_run
from polypanda.simulation.assets import MENAGERIE_COMMIT, load_panda_scene
from polypanda.simulation.panda import PandaIndices, reset_to_home, tcp_position
from polypanda.simulation.scenes import cube_position
from polypanda.training.config import PPOParams, TrainConfig
from polypanda.training.runner import train

CART = ActionConfig(space="cartesian")
CART6 = ActionConfig(space="cartesian", cartesian=CartesianConfig(dof=6))


def _panda(config=CART):
    model, data = load_panda_scene()
    idx = PandaIndices.from_model(model)
    reset_to_home(model, data, idx)
    adapter = make_action_adapter(model, idx, config)
    adapter.reset_to_measured(data)
    return model, data, idx, adapter


def _R(data, idx):
    return data.site_xmat[idx.tcp_site].reshape(3, 3)


def _run(adapter, data, action, steps):
    for _ in range(steps):
        adapter.step(data, np.asarray(action, dtype=float))


def test_factory_and_schemas():
    model, data = load_panda_scene()
    idx = PandaIndices.from_model(model)
    assert isinstance(make_action_adapter(model, idx, ActionConfig()), JointDeltaActionAdapter)
    cart = make_action_adapter(model, idx, CART)
    assert isinstance(cart, CartesianActionAdapter) and cart.action_dim == 4
    assert make_action_adapter(model, idx, CART6).action_dim == 6
    assert action_schema(ActionConfig()) == "joint-increment-v1-7d"
    assert action_schema(CART) == "cartesian-v1-4d" == cart.schema
    assert action_schema(CART6) == "cartesian-v1-6d"
    with pytest.raises(ValueError, match="action space"):
        make_action_adapter(model, idx, ActionConfig(space="polar"))
    with pytest.raises(ValueError, match="shape"):
        cart.apply(data, np.zeros(7))


def test_zero_action_holds_the_hand_level():
    _, data, idx, adapter = _panda()
    start = tcp_position(data, idx)
    _run(adapter, data, np.zeros(4), 60)
    # Only the position servos' constant gravity sag (about 7 mm), no drift.
    assert np.linalg.norm(tcp_position(data, idx) - start) < 0.01
    assert gripper_tilt(_R(data, idx)) < 0.02


@pytest.mark.parametrize("axis,sign", [(0, 1.0), (1, 1.0), (2, -1.0)])
def test_each_axis_moves_the_hand_in_a_straight_line(axis, sign):
    _, data, idx, adapter = _panda()
    start = tcp_position(data, idx)
    action = np.zeros(4)
    action[axis] = sign
    _run(adapter, data, action, 10)    # 10 steps x 1 cm
    _run(adapter, data, np.zeros(4), 10)
    moved = tcp_position(data, idx) - start
    assert moved[axis] == pytest.approx(0.10 * sign, abs=0.012)
    across = [i for i in (0, 1) if i != axis]  # (height also carries the constant sag)
    assert np.abs(moved[across]).max() < 0.005
    assert gripper_tilt(_R(data, idx)) < 0.03


def test_yaw_turns_the_closing_direction_in_place():
    _, data, idx, adapter = _panda()
    start, yaw = tcp_position(data, idx), closing_yaw(_R(data, idx))
    _run(adapter, data, [0, 0, 0, 1.0], 10)  # 10 x 0.05 rad
    _run(adapter, data, np.zeros(4), 10)
    assert closing_yaw(_R(data, idx)) - yaw == pytest.approx(0.5, abs=0.02)
    assert np.linalg.norm((tcp_position(data, idx) - start)[:2]) < 0.005
    assert gripper_tilt(_R(data, idx)) < 0.03


def test_target_cannot_wind_up_or_leave_its_box():
    _, data, idx, adapter = _panda()
    _run(adapter, data, [0, 0, -1.0, 0], 120)  # 1.2 m of "down" commands into the table
    commanded, _ = adapter.ik.forward(adapter.q_cmd)
    assert adapter.target_pos[2] >= 0.01 - 1e-9
    assert np.abs(adapter.target_pos - commanded).max() <= 0.03 + 1e-6
    low = adapter.target_pos[2]
    _run(adapter, data, [0, 0, 1.0, 0], 3)  # reverses at once: nothing stored up
    assert adapter.target_pos[2] >= low + 0.025


def test_six_dof_tilt_is_limited():
    _, data, idx, adapter = _panda(CART6)
    _run(adapter, data, [0, 0, 0, 1.0, 0, 0], 40)  # 2 rad of roll commands
    assert gripper_tilt(adapter.target_R) <= 0.6 + 1e-6
    _run(adapter, data, np.zeros(6), 20)
    assert gripper_tilt(_R(data, idx)) == pytest.approx(0.6, abs=0.05)


def test_hand_off_from_joint_actions_does_not_jump():
    model, data, idx, joint = _panda(ActionConfig())
    rng = np.random.default_rng(0)
    for _ in range(15):
        joint.step(data, rng.uniform(-0.5, 0.5, 7))
    before = data.ctrl[idx.arm_actuators].copy()
    tilt = gripper_tilt(_R(data, idx))
    cart = make_action_adapter(model, idx, CART)
    cart.adopt_ctrl(data)
    np.testing.assert_array_equal(cart.q_cmd, before)
    cart.apply(data, np.zeros(4))  # levels a tilted hand by at most 0.05 rad per step
    assert np.abs(data.ctrl[idx.arm_actuators] - before).max() < 0.1
    assert tilt > 0.02  # the case is non-trivial
    for _ in range(int(tilt / 0.05) + 5):
        cart.step(data, np.zeros(4))
    assert gripper_tilt(_R(data, idx)) < 0.03


def test_tasks_take_cartesian_actions():
    grasp = GraspV2Env(replace(GraspConfig(), action=CART))
    assert grasp.action_space.shape == (4,)
    check_env(grasp, skip_render_check=True)
    place = PlaceEnv(replace(PlaceConfig(), action=CART6))
    assert place.action_space.shape == (6,)
    place.reset(seed=0)
    place.step(np.zeros(6, dtype=np.float32))
    with pytest.raises(ValueError, match="joint"):
        TransportResidualEnv(replace(TransportConfig(), action=CART))


def test_hand_motion_carries_a_cube():
    """A one-line rule (move the cube toward the goal) is enough: the interface makes carrying
    easy, unlike joint increments (plain PPO: 0 %)."""
    env = TransportEnv(replace(TransportConfig(), action=CART))
    wins = 0
    for i in (0, 50, 100, 150):
        env.reset(seed=i, options={"state_index": i})
        for _ in range(200):
            err = env.goal - cube_position(env.data, env.cube)
            action = np.append(np.clip(err / 0.02, -1, 1), 0.0).astype(np.float32)
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        wins += bool(info["is_success"])
    assert wins >= 3


@pytest.mark.parametrize("name,base,cls", [
    ("transport_cartesian.yaml", "transport.yaml", TransportConfig),
    ("grasp_cartesian_curriculum.yaml", "grasp_curriculum.yaml", GraspConfig),
])
def test_milestone8_configs_change_only_the_action_interface(name, base, cls):
    configs = Path(__file__).parents[1] / "configs"
    cfg, reference = cls.from_yaml(configs / name), cls.from_yaml(configs / base)
    assert action_schema(cfg.action) == "cartesian-v1-4d"
    assert cfg.action.cartesian == CartesianConfig()
    assert replace(cfg, action=reference.action) == reference


def test_runs_record_and_check_their_action_schema(tmp_path):
    cfg = TrainConfig(task="grasp_v2", n_envs=1, vec_env="dummy", total_timesteps=256,
                      eval_freq=256, eval_episodes=1, checkpoint_freq=256,
                      ppo=PPOParams(n_steps=128, batch_size=64, n_epochs=1))
    run_dir = train(cfg, replace(GraspConfig(), action=CART), tmp_path / "cart_smoke")
    meta = yaml.safe_load((run_dir / "metadata.yaml").read_text())
    assert meta["action_schema"] == "cartesian-v1-4d"
    policy = load_run(run_dir, "final")
    action, _ = policy.predict(np.zeros(41, dtype=np.float32))
    assert action.shape == (4,)
    with pytest.raises(IncompatiblePolicyError, match="action_schema"):
        check_compatible(meta)  # an environment taking joint increments refuses it
    joint_meta = {**meta, "action_schema": "joint-increment-v1-7d"}
    check_compatible(joint_meta)
    assert meta["robot_model_revision"] == MENAGERIE_COMMIT
