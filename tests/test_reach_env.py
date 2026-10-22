from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

import polypanda.environments  # noqa: F401  (registers PolyPanda/Reach-v0)
from polypanda.environments.reach import OBS_DIM, OBS_LAYOUT, ReachConfig, ReachEnv
from polypanda.simulation.panda import tcp_position

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "reach.yaml"
ZERO = np.zeros(7, dtype=np.float32)


def _env(**episode) -> ReachEnv:
    cfg = ReachConfig()
    if episode:
        cfg = replace(cfg, episode=replace(cfg.episode, **episode))
    return ReachEnv(cfg)


def _home_tcp() -> np.ndarray:
    env = _env()
    env.reset(seed=0)
    return tcp_position(env.data, env.idx)


def test_spaces():
    env = _env()
    assert env.action_space.shape == (7,) and env.action_space.dtype == np.float32
    assert env.observation_space.shape == (OBS_DIM,) == (27,)
    obs, info = env.reset(seed=0)
    assert obs.dtype == np.float32 and env.observation_space.contains(obs)
    assert info["distance"] >= env.config.episode.min_start_distance


def test_observation_layout():
    env = _env()
    obs, _ = env.reset(seed=1)
    obs, *_ = env.step(np.full(7, 0.5, dtype=np.float32))
    parts, start = {}, 0
    for name, n in OBS_LAYOUT:
        parts[name] = obs[start:start + n]
        start += n
    tcp = tcp_position(env.data, env.idx)
    np.testing.assert_allclose(parts["q"], env.data.qpos[env.idx.arm_qpos], rtol=1e-6)
    np.testing.assert_allclose(parts["dq"], env.data.qvel[env.idx.arm_qvel], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(parts["q_cmd"], env.adapter.q_cmd, rtol=1e-6)
    np.testing.assert_allclose(parts["tcp"], tcp, rtol=1e-6)
    np.testing.assert_allclose(parts["target_error"], env.target - tcp, rtol=1e-5, atol=1e-6)


def test_seeded_reset_is_deterministic():
    a, b, c = _env(), _env(), _env()
    obs_a, _ = a.reset(seed=42)
    obs_b, _ = b.reset(seed=42)
    obs_c, _ = c.reset(seed=43)
    np.testing.assert_array_equal(obs_a, obs_b)
    assert not np.array_equal(a.target, c.target)
    rng = np.random.default_rng(0)
    for _ in range(20):
        action = rng.uniform(-1, 1, 7).astype(np.float32)
        np.testing.assert_array_equal(a.step(action)[0], b.step(action)[0])


def test_target_option_overrides_sampling():
    env = _env()
    target = np.array([0.4, 0.2, 0.3])
    env.reset(seed=0, options={"target": target})
    np.testing.assert_array_equal(env.target, target)
    with pytest.raises(ValueError):
        env.reset(options={"target": [0.4, 0.2]})


def test_success_requires_dwell():
    env = _env()
    env.reset(seed=0, options={"target": _home_tcp()})  # already within tolerance
    dwell = env.config.success.dwell_steps
    for step in range(1, dwell + 1):
        _, reward, terminated, truncated, info = env.step(ZERO)
        assert info["distance"] < env.config.success.tolerance
        assert terminated == (step == dwell) and info["is_success"] == (step == dwell)
        assert not truncated
    assert info["reward_terms"]["success"] == env.config.reward.success_bonus


def test_timeout_truncates_not_terminates():
    env = _env(horizon=10)
    env.reset(seed=0, options={"target": np.array([0.3, 0.4, 0.15])})
    for step in range(1, 11):
        _, _, terminated, truncated, info = env.step(ZERO)
        assert not terminated and not info["is_success"]
        assert truncated == (step == 10)


def test_reward_is_sum_of_terms_and_progress_telescopes():
    env = _env()
    _, info = env.reset(seed=3)
    start_distance = info["distance"]
    rng = np.random.default_rng(3)
    progress_total = 0.0
    for _ in range(30):
        _, reward, terminated, truncated, info = env.step(rng.uniform(-1, 1, 7).astype(np.float32))
        assert reward == pytest.approx(sum(info["reward_terms"].values()))
        progress_total += info["reward_terms"]["progress"]
        if terminated or truncated:
            break
    expected = env.config.reward.progress * (start_distance - info["distance"])
    assert progress_total == pytest.approx(expected)


def test_action_penalty_scales_with_squared_action():
    env = _env()
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.ones(7, dtype=np.float32))
    assert info["reward_terms"]["action"] == pytest.approx(-env.config.reward.action * 7)
    _, _, _, _, info = env.step(np.full(7, 3.0, dtype=np.float32))  # clipped to 1
    assert info["reward_terms"]["action"] == pytest.approx(-env.config.reward.action * 7)


def test_start_joint_noise_perturbs_home():
    quiet, noisy = _env(), _env(start_joint_noise=0.1)
    obs_q, _ = quiet.reset(seed=5)
    obs_n, _ = noisy.reset(seed=5)
    dq = np.abs(obs_n[:7] - obs_q[:7])
    assert dq.max() > 0 and dq.max() <= 0.1 + 1e-6


def test_yaml_config_loads_and_rejects_unknown_keys(tmp_path):
    cfg = ReachConfig.from_yaml(CONFIG_PATH)
    assert cfg.success.tolerance == 0.02 and cfg.workspace.low == (0.25, -0.45, 0.10)
    bad = tmp_path / "bad.yaml"
    bad.write_text("reward:\n  progres: 1.0\n")
    with pytest.raises(ValueError, match="progres"):
        ReachConfig.from_yaml(bad)


def test_gym_make():
    env = gym.make("PolyPanda/Reach-v0")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (27,)
    env.close()
