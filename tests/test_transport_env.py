import warnings
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

import polypanda.environments  # noqa: F401  (registers PolyPanda/Transport-v0)
from polypanda.control.ik import PoseIK, closing_yaw, top_down_quat
from polypanda.environments.handoff import HandoffPool, capture_state, restore_state
from polypanda.environments.transport import (
    OBS_DIM,
    OBS_SLICES,
    TransportConfig,
    TransportEnv,
)
from polypanda.simulation.panda import tcp_position
from polypanda.simulation.scenes import cube_position, load_pick_scene

ROOT = Path(__file__).parents[1]
TRAIN = ROOT / "data" / "handoff" / "transport_train_v1.npz"
EVAL = ROOT / "eval_sets" / "transport_nominal_v1.npz"
ZERO = np.zeros(7, dtype=np.float32)


@pytest.fixture(scope="module")
def env():
    return TransportEnv()


def carry(env, steps=200, gain=0.2):
    """Closed-loop scripted carry: steer the hand so the cube's *current* position goes to goal."""
    ik = PoseIK()
    R = env.data.site_xmat[env.arm.tcp_site].reshape(3, 3)
    quat = top_down_quat(closing_yaw(R))
    infos = []
    for _ in range(steps):
        offset = cube_position(env.data, env.cube) - tcp_position(env.data, env.arm)
        q = env.data.qpos[env.arm.arm_qpos]
        a = ik.action(q, env.goal - offset, quat, gain, env.adapter.max_delta)
        _, reward, terminated, truncated, info = env.step(a)
        infos.append((reward, info))
        if terminated or truncated:
            break
    return infos


# --- handoff pools ------------------------------------------------------------------------

def test_pools_have_equal_shares_per_source():
    train, ev = HandoffPool.load(TRAIN), HandoffPool.load(EVAL)
    assert len(train) == 1200 and len(ev) == 200
    assert set(Counter(train.sources).values()) == {300}
    assert set(Counter(ev.sources).values()) == {50}
    assert train.goals is None and ev.goals.shape == (200, 3)


def test_restore_is_bit_exact():
    model, data = load_pick_scene()
    state = HandoffPool.load(TRAIN).states[17]
    restore_state(model, data, state)
    np.testing.assert_array_equal(capture_state(model, data), state)


def test_handoff_states_hold_a_lifted_cube(env):
    pool = HandoffPool.load(TRAIN)
    for i in range(0, len(pool), 150):   # every source, several states
        env.reset(options={"state_index": i, "goal": [0.5, 0.0, 0.2]})
        for _ in range(env.monitor.config.window):
            _, _, terminated, _, info = env.step(ZERO)
            assert not terminated
        assert info["grasped"], (i, pool.sources[i])
        assert cube_position(env.data, env.cube)[2] - env.cube.half_size[2] >= 0.04


def test_eval_goals_respect_goal_rules():
    model, data = load_pick_scene()
    ev, cfg = HandoffPool.load(EVAL), TransportConfig().goal
    from polypanda.simulation.scenes import CubeIndices
    cube = CubeIndices.from_model(model)
    for state, goal in zip(ev.states, ev.goals, strict=True):
        restore_state(model, data, state)
        assert np.linalg.norm(goal - cube_position(data, cube)) >= cfg.min_distance
        assert cfg.z_range[0] <= goal[2] <= cfg.z_range[1]


def test_tampered_pool_is_rejected(tmp_path):
    with np.load(TRAIN) as f:
        arrays = dict(f)
    arrays["states"] = arrays["states"].copy()
    arrays["states"][0, 3] += 1e-6
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, **arrays)
    with pytest.raises(ValueError, match="digest"):
        HandoffPool.load(bad)


# --- environment --------------------------------------------------------------------------

def test_defaults_match_yaml():
    assert TransportConfig() == TransportConfig.from_yaml(ROOT / "configs" / "transport.yaml")


def test_spaces_and_check_env(env):
    assert env.observation_space.shape == (OBS_DIM,) == (47,)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(TransportEnv(), warn=True)


def test_reset_continues_grasp_commands_and_observes_goal(env):
    obs, info = env.reset(options={"state_index": 5, "goal": [0.4, 0.1, 0.2]})
    np.testing.assert_array_equal(env.adapter.q_cmd, env.data.ctrl[env.arm.arm_qpos])
    cube = cube_position(env.data, env.cube)
    np.testing.assert_allclose(obs[OBS_SLICES["goal"]], [0.4, 0.1, 0.2], rtol=1e-6)
    np.testing.assert_allclose(obs[OBS_SLICES["goal_error"]], [0.4, 0.1, 0.2] - cube,
                               atol=1e-6)
    assert np.all(obs[OBS_SLICES["slip_velocity"]] == 0)
    assert info["start_source"] == env.pool.sources[5]


def test_seeded_resets_are_deterministic():
    a, b = TransportEnv(), TransportEnv()
    for s in range(3):
        ia, ib = a.reset(seed=s)[1], b.reset(seed=s)[1]
        assert ia["start_index"] == ib["start_index"]
        np.testing.assert_array_equal(ia["goal"], ib["goal"])


def test_scripted_closed_loop_carry_succeeds(env):
    ev = HandoffPool.load(EVAL)
    for i in (0, 60, 120, 180):  # one start from each grasp controller
        env.reset(options={"state": ev.states[i], "goal": ev.goals[i]})
        infos = carry(env)
        final = infos[-1][1]
        assert final["is_success"] and final["failure"] is None, (i, ev.sources[i])
        assert final["distance"] < 0.02


def test_rewards_sum_and_progress_telescopes(env):
    _, info0 = env.reset(options={"state_index": 3, "goal": [0.45, -0.1, 0.18]})
    infos = carry(env, steps=80)
    progress = sum(i["reward_terms"]["progress"] for _, i in infos)
    assert progress == pytest.approx(20.0 * (info0["distance"] - infos[-1][1]["distance"]))
    for reward, info in infos:
        assert reward == pytest.approx(sum(info["reward_terms"].values()))


def test_action_rate_penalty(env):
    env.reset(options={"state_index": 0, "goal": [0.5, 0.0, 0.2]})
    a1 = np.full(7, 0.5, dtype=np.float32)
    _, _, _, _, info = env.step(a1)
    assert info["reward_terms"]["action_rate"] == pytest.approx(-0.05 * 7 * 0.25)
    _, _, _, _, info = env.step(a1)
    assert info["reward_terms"]["action_rate"] == pytest.approx(0.0)


def test_opening_the_gripper_is_a_drop(env):
    env.reset(options={"state_index": 0, "goal": [0.5, 0.0, 0.2]})
    for _ in range(30):
        env.data.ctrl[env.arm.gripper_actuator] = 255.0  # sabotage: open the fingers
        _, reward, terminated, truncated, info = env.step(ZERO)
        if terminated:
            break
    assert terminated and info["failure"] == "dropped" and not truncated
    assert info["reward_terms"]["drop"] == -env.config.reward.drop
    assert env.elapsed_steps <= 15


def test_success_never_coexists_with_a_drop():
    """Regression: a tipped cube at the goal was once counted as delivered and dropped."""
    from dataclasses import replace
    cfg = TransportConfig()
    env = TransportEnv(replace(cfg, episode=replace(cfg.episode, max_cube_tilt=0.0)))
    ev = HandoffPool.load(EVAL)
    env.reset(options={"state": ev.states[0], "goal": ev.goals[0]})
    infos = carry(env)
    assert infos[-1][1]["failure"] == "dropped"  # any tilt now counts as fallen over
    assert not any(info["is_success"] or info["reward_terms"]["hold"] for _, info in infos)


def test_gym_make():
    e = gym.make("PolyPanda/Transport-v0")
    assert e.reset(seed=0)[0].shape == (47,)
    e.close()
