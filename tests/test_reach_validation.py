"""Stress and edge-case validation of ReachEnv (Milestone 2.5)."""

import warnings

import mujoco
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from polypanda.environments.reach import ReachEnv

FAR_TARGET = np.array([0.3, 0.4, 0.15])


def test_sb3_check_env_without_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(ReachEnv(), warn=True)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_rollouts_stay_valid(seed):
    env = ReachEnv()
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    margin = 0.01  # rad; MuJoCo joint limits are soft constraints
    for _ in range(env.config.episode.horizon):
        obs, reward, terminated, truncated, info = env.step(
            rng.uniform(-1, 1, 7).astype(np.float32))
        assert obs.dtype == np.float32 and env.observation_space.contains(obs)
        assert np.all(np.isfinite(obs)) and np.isfinite(reward)
        assert info["failure"] is None
        assert np.all(env.adapter.q_cmd >= env.adapter.low)
        assert np.all(env.adapter.q_cmd <= env.adapter.high)
        q = env.data.qpos[env.idx.arm_qpos]
        assert np.all(q >= env._q_low - margin) and np.all(q <= env._q_high + margin)
        if terminated or truncated:
            break
    assert env.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number == 0


def test_penalties_trigger_under_deliberate_abuse():
    """Drive the arm into the floor, itself and joint 4's lower limit."""
    env = ReachEnv()
    env.reset(seed=0, options={"target": FAR_TARGET})
    abuse = np.array([0, 1, 0, -1, 0, -1, 0], dtype=np.float32)
    first_limit_step, contact_seen, max_penetration = None, False, 0.0
    for step in range(1, env.config.episode.horizon + 1):
        _, _, terminated, truncated, info = env.step(abuse)
        assert info["failure"] is None
        if info["contact"]:
            contact_seen = True
            assert info["reward_terms"]["contact"] == -env.config.reward.contact
            for i in range(env.data.ncon):
                max_penetration = max(max_penetration, -env.data.contact[i].dist)
        if info["joints_at_limit"] and first_limit_step is None:
            first_limit_step = step
            assert info["reward_terms"]["joint_limit"] < 0
        if terminated or truncated:
            break
    assert contact_seen
    # joint4 starts at -1.5708 with lower bound -3.0718: ceil(1.501 / 0.015) = 101 steps.
    assert first_limit_step == 101
    assert max_penetration < 0.01
    assert np.all(np.isfinite(env.data.qpos))


def test_reset_mid_episode_restarts_counters():
    env = ReachEnv()
    env.reset(seed=0, options={"target": FAR_TARGET})
    for _ in range(50):
        env.step(np.zeros(7, dtype=np.float32))
    env.reset(seed=1, options={"target": FAR_TARGET})
    horizon = env.config.episode.horizon
    for step in range(1, horizon + 1):
        _, _, terminated, truncated, _ = env.step(np.zeros(7, dtype=np.float32))
        assert not terminated
        assert truncated == (step == horizon)


def test_unseeded_resets_continue_seeded_stream():
    a, b = ReachEnv(), ReachEnv()
    targets = []
    for env in (a, b):
        env.reset(seed=9)
        targets.append([env.reset()[1]["target"] for _ in range(3)])
    np.testing.assert_array_equal(targets[0], targets[1])


@pytest.mark.parametrize("action", [np.zeros(6), np.zeros(8), np.full(7, np.nan)])
def test_invalid_actions_raise(action):
    env = ReachEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(action.astype(np.float32))


def test_many_resets_give_valid_targets():
    env = ReachEnv()
    env.reset(seed=0)
    low, high = np.array(env.config.workspace.low), np.array(env.config.workspace.high)
    for _ in range(200):
        _, info = env.reset()
        assert np.all(info["target"] >= low) and np.all(info["target"] <= high)
        assert info["distance"] >= env.config.episode.min_start_distance
