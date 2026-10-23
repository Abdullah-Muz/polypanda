import warnings
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

import polypanda.environments  # noqa: F401  (registers PolyPanda/Grasp-v0)
from polypanda.control.scripted_grasp import Phase, ScriptedGrasp
from polypanda.environments.grasp import (
    OBS_DIM,
    OBS_SLICES,
    GraspConfig,
    GraspEnv,
    alignment_error,
)
from polypanda.simulation.panda import tcp_position
from polypanda.simulation.scenes import cube_position

CONFIG = Path(__file__).parents[1] / "configs" / "grasp.yaml"
ZERO = np.zeros(7, dtype=np.float32)


def _run_scripted(env, placement, extra_steps=20):
    """Drive GraspEnv with the scripted baseline's arm actions (the env gates the gripper)."""
    obs, _ = env.reset(options={"cube": placement})
    ctrl = ScriptedGrasp(env.model, env.arm, env.cube, env.adapter.max_delta)
    ctrl.reset(env.data)
    infos = []
    while ctrl.phase != Phase.DONE and env.elapsed_steps < env.config.episode.horizon:
        action, _ = ctrl.act(env.data)
        obs, reward, terminated, truncated, info = env.step(action)
        infos.append((reward, info))
    for _ in range(extra_steps):
        obs, reward, terminated, truncated, info = env.step(ZERO)
        infos.append((reward, info))
    return obs, infos


def test_defaults_match_yaml():
    assert GraspConfig() == GraspConfig.from_yaml(CONFIG)


def test_spaces_and_check_env():
    env = GraspEnv()
    assert env.observation_space.shape == (OBS_DIM,) == (41,)
    obs, info = env.reset(seed=0)
    assert obs.dtype == np.float32 and env.observation_space.contains(obs)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(GraspEnv(), warn=True)


def test_observation_layout():
    env = GraspEnv()
    obs, _ = env.reset(options={"cube": (0.45, 0.1, 0.7)})
    tcp = tcp_position(env.data, env.arm)
    np.testing.assert_allclose(obs[OBS_SLICES["tcp"]], tcp, rtol=1e-6)
    np.testing.assert_allclose(obs[OBS_SLICES["cube_rel"]],
                               cube_position(env.data, env.cube) - tcp, rtol=1e-5, atol=1e-6)
    R = env.data.site_xmat[env.arm.tcp_site].reshape(3, 3)
    np.testing.assert_allclose(obs[OBS_SLICES["tcp_rot6"]], R[:, :2].T.flatten(), atol=1e-6)
    align = alignment_error(R, 0.7)
    assert obs[OBS_SLICES["align_sin4"]][0] == pytest.approx(np.sin(4 * align), abs=1e-6)
    assert obs[OBS_SLICES["finger_gap"]][0] == pytest.approx(0.08, abs=1e-3)
    assert obs[OBS_SLICES["gripper_closed"]][0] == 0.0
    assert obs[OBS_SLICES["cube_height"]][0] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("cube_yaw,closing,expected", [
    (0.0, 0.0, 0.0), (np.pi / 2, 0.0, 0.0), (0.1, 0.0, 0.1), (np.pi / 2 - 0.1, 0.0, -0.1),
])
def test_alignment_error_uses_square_symmetry(cube_yaw, closing, expected):
    R = np.eye(3)
    R[:, 1] = [np.cos(closing), np.sin(closing), 0.0]
    assert alignment_error(R, cube_yaw) == pytest.approx(expected, abs=1e-12)


def test_seeded_resets_are_deterministic_and_inside_region():
    a, b = GraspEnv(), GraspEnv()
    for seed in range(5):
        _, ia = a.reset(seed=seed)
        _, ib = b.reset(seed=seed)
        assert ia["cube"] == ib["cube"]
        x, y, _ = ia["cube"]
        assert 0.35 <= x <= 0.65 and -0.25 <= y <= 0.25


@pytest.mark.parametrize("placement", [(0.5, 0.0, 0.0), (0.42, 0.18, 2.2), (0.6, -0.2, -0.9)])
def test_scripted_actions_succeed_through_the_gate(placement):
    env = GraspEnv()
    _, infos = _run_scripted(env, placement)
    closed_at = next(i for i, (_, info) in enumerate(infos) if info["gripper_closed"])
    assert all(info["gripper_closed"] for _, info in infos[closed_at:])  # never reopened
    final = infos[-1][1]
    assert final["is_success"] and final["grasped"] and final["lift"] >= 0.05
    assert final["reward_terms"]["hold"] == env.config.reward.hold
    assert not any(info["floor_contact"] for _, info in infos)


def test_rewards_sum_and_lift_term_telescopes():
    env = GraspEnv()
    _, infos = _run_scripted(env, (0.5, 0.0, 0.3))
    lift_total = sum(info["reward_terms"]["lift"] for _, info in infos)
    final_lift = min(infos[-1][1]["lift"], env.config.reward.lift_cap)
    assert lift_total == pytest.approx(env.config.reward.lift * final_lift, abs=1e-9)
    for reward, info in infos:
        assert reward == pytest.approx(sum(info["reward_terms"].values()))


def test_gate_reopens_after_closing_on_nothing():
    env = GraspEnv()
    env.reset(options={"cube": (0.5, 0.0, 0.0)})
    ctrl = ScriptedGrasp(env.model, env.arm, env.cube, env.adapter.max_delta)
    ctrl.reset(env.data)
    while not env.closed:
        env.step(ctrl.act(env.data)[0])
    # Whisk the cube away just as the fingers start to close: they close on nothing.
    env.data.qpos[env.cube.qpos:env.cube.qpos + 3] = [0.3, 0.3, 0.03]
    mujoco.mj_forward(env.model, env.data)
    states = [env.step(ZERO)[4]["gripper_closed"] for _ in range(40)]
    assert states[0] and not states[-1]


def test_knocked_over_cube_terminates_with_penalty():
    env = GraspEnv()
    env.reset(options={"cube": (0.5, 0.0, 0.0)})
    quat = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]  # 90 deg about x: on its side
    env.data.qpos[env.cube.qpos:env.cube.qpos + 7] = [0.5, 0.0, 0.03, *quat]
    mujoco.mj_forward(env.model, env.data)
    _, reward, terminated, truncated, info = env.step(ZERO)
    assert terminated and not truncated and info["failure"] == "knocked_over"
    assert info["reward_terms"]["knocked_over"] == -env.config.reward.knocked_over


def test_random_policy_runs_full_horizon_without_success():
    env = GraspEnv()
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(env.config.episode.horizon):
        obs, reward, terminated, truncated, info = env.step(
            rng.uniform(-1, 1, 7).astype(np.float32))
        assert np.all(np.isfinite(obs)) and np.isfinite(reward)
        if terminated or truncated:
            break
    assert truncated or info["failure"] == "knocked_over"
    assert not info["is_success"]


def test_gym_make():
    env = gym.make("PolyPanda/Grasp-v0")
    assert env.reset(seed=0)[0].shape == (41,)
    env.close()
