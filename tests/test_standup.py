"""The stand-up skill's environment (v3 step d2): starts, gate, tower, evaluation set."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from polypanda.environments.standup import (
    GAP_CAP,
    OBS_DIM,
    OBS_DIM_V2,
    OBS_DIM_V3,
    StandUpConfig,
    StandUpEnv,
    StandUpV2Env,
    StandUpV3Env,
)
from polypanda.evaluation.standup import (
    ScriptedStandUpDriver,
    StandUpEvalSet,
    generate_standup_set,
    run_standup_episode,
)
from polypanda.simulation.scenes import cube_tilt

ROOT = Path(__file__).parents[1]
CFG = StandUpConfig.from_yaml(ROOT / "configs" / "standup_v1.yaml")


@pytest.fixture(scope="module")
def env():
    return StandUpEnv(CFG.for_evaluation())


def test_a_carry_start_holds_the_cube_lying_across_its_width(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,) and np.all(np.isfinite(obs)) and info["start"] == "carry"
    assert cube_tilt(env.data, env.cube) > np.radians(80)            # lying
    assert env.data.eq_active[env.assist] and not env.released        # held (grasp assist)
    assert all(env._finger_contacts())


def test_helpful_starts_are_training_only_and_hold_the_cube_turned_up():
    cfg = replace(CFG, start=replace(CFG.start, helpful_prob=1.0))
    env = StandUpEnv(cfg)
    _, info = env.reset(seed=1)
    assert info["start"] == "helpful" and cube_tilt(env.data, env.cube) < np.radians(20)
    ev = StandUpEnv(cfg.for_evaluation())
    assert all(ev.reset(seed=s)[1]["start"] == "carry" for s in range(4))


def test_the_scripted_stand_up_releases_the_cube_upright_on_the_spot(env):
    s = generate_standup_set(env, "t", 3, 7)
    for i, ep in enumerate(s.episodes):
        result, _ = run_standup_episode(env, ScriptedStandUpDriver(), ep, i)
        assert result.success and result.released, result


def test_touching_the_tower_ends_the_episode(env):
    ep = env.draw_episode(np.random.default_rng(3))
    ep["tower"] = (ep["spot"][0], ep["spot"][1], 4)   # right under the hand, 24 cm tall
    env.reset(options={"episode": ep})
    down = np.zeros(6, dtype=np.float32)
    down[2] = -1.0
    for _ in range(20):
        _, _, terminated, _, info = env.step(down)
        if terminated:
            break
    assert terminated and info["failure"] == "tower_hit"


CFG2 = StandUpConfig.from_yaml(ROOT / "configs" / "standup_v2.yaml")


def test_v2_senses_the_towers_clearance_and_shapes_closing_in():
    env = StandUpV2Env(CFG2.for_evaluation())
    ep = env.draw_episode(np.random.default_rng(3))
    ep["tower"] = (ep["spot"][0] + 0.12, ep["spot"][1], 4)   # 12 cm beside the hand, 24 cm tall
    obs, _ = env.reset(options={"episode": ep})
    assert obs.shape == (OBS_DIM_V2,)
    gap0 = float(obs[OBS_DIM])
    assert 0.0 < gap0 < GAP_CAP
    toward = np.zeros(6, dtype=np.float32)
    toward[0] = 1.0                                       # move the hand toward the tower
    gaps, terms = [], []
    for _ in range(40):
        obs, _, terminated, _, info = env.step(toward)
        gaps.append(float(obs[OBS_DIM]))
        terms.append(info["reward_terms"]["clearance"])
        if terminated:
            break
    k = int(np.argmin(gaps))
    assert gaps[k] < CFG2.reward.clearance_margin and sum(terms[:k + 1]) < 0   # closer: cost
    phi = [min(0.0, g / CFG2.reward.clearance_margin - 1.0) for g in (gap0, gaps[-1])]
    assert sum(terms) == pytest.approx(CFG2.reward.clearance * (phi[1] - phi[0]))  # telescopes
    env.reset(options={"episode": {**ep, "tower": None}})
    assert float(env._observation()[OBS_DIM]) == pytest.approx(GAP_CAP)  # no tower: the cap


def test_demo_starts_hand_over_partway_through_the_scripted_stand_up_in_training_only():
    cfg = StandUpConfig.from_yaml(ROOT / "configs" / "standup_v2_demo.yaml")
    cfg = replace(cfg, start=replace(cfg.start, helpful_prob=0.0, demo_prob=1.0,
                                     demo_steps=(60, 60)))
    env = StandUpV2Env(cfg)
    obs, info = env.reset(seed=2)
    assert info["start"] == "demo" and env.elapsed_steps == 0 and obs.shape == (OBS_DIM_V2,)
    assert cube_tilt(env.data, env.cube) < np.radians(60)    # turned up by the scripted part
    _, _, _, _, step_info = env.step(np.zeros(6, dtype=np.float32))
    assert abs(step_info["reward_terms"]["progress"]) < 1.0  # no reward for the scripted part
    ev = StandUpV2Env(cfg.for_evaluation())
    assert all(ev.reset(seed=s)[1]["start"] == "carry" for s in range(3))


def test_v1_is_unchanged_by_v2(env):
    obs, _ = env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
    assert obs.shape == (OBS_DIM,) and "clearance" not in info["reward_terms"]


def test_round_5_releases_inside_the_success_tolerance_and_feels_the_tower_sooner():
    from polypanda.training.config import TrainConfig
    cfg = StandUpConfig.from_yaml(ROOT / "configs" / "standup_v2_r5.yaml")
    assert cfg.gate.xy_tol == pytest.approx(0.015)
    assert cfg.gate.xy_tol < cfg.check.tolerance                       # room to settle
    assert cfg.reward.clearance_margin == pytest.approx(0.15)
    train = TrainConfig.from_yaml(ROOT / "configs" / "standup_v2_r5_ppo.yaml")
    assert train.task == "standup_v2" and train.env_config == "configs/standup_v2_r5.yaml"
    assert not train.ppo.use_sde                    # round 4's exploration (gSDE set aside)
    gsde = TrainConfig.from_yaml(ROOT / "configs" / "standup_v2_r5_gsde_ppo.yaml")
    assert gsde.env_config == train.env_config
    assert gsde.ppo.use_sde and gsde.ppo.sde_sample_freq == 32


def test_eval_sets_round_trip_with_their_digest(env, tmp_path):
    s = generate_standup_set(env, "t", 4, 11)
    s.save(tmp_path / "t.json")
    back = StandUpEvalSet.load(tmp_path / "t.json")
    assert back.digest == s.digest and back.episodes == s.episodes
    assert generate_standup_set(env, "t", 4, 11).digest == s.digest     # reproducible


CFG3 = StandUpConfig.from_yaml(ROOT / "configs" / "standup_v3.yaml")


def _hand_dir(env):
    z = env.data.site_xmat[env.arm.tcp_site].reshape(3, 3)[:, 2]
    return float(np.arctan2(z[1], z[0])), abs(float(z[2]))


def _angle(a, b):
    return abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def test_v3_observes_a_reachable_commanded_turn_drawn_in_training():
    cfg = replace(CFG3, start=replace(CFG3.start, helpful_prob=0.0, demo_prob=0.0))
    env = StandUpV3Env(cfg)
    commands = []
    for s in range(4):
        obs, info = env.reset(seed=s)
        assert obs.shape == (OBS_DIM_V3,)
        np.testing.assert_allclose(obs[-2:], [np.cos(env.command), np.sin(env.command)],
                                   atol=1e-6)
        assert any(_angle(env.command, d) < 1e-9 for d in env.reachable_directions())
        commands.append(info["command"])
    assert len(set(commands)) > 1                              # drawn, not one fixed way
    ev = StandUpV3Env(cfg.for_evaluation())
    ep = ev.draw_episode(np.random.default_rng(0))
    ev.reset(options={"episode": ep, "command": 1.0})
    assert ev.command == pytest.approx(1.0)


def test_v3_turning_the_wrong_way_ends_the_episode():
    cfg = replace(CFG3, start=replace(CFG3.start, helpful_prob=1.0))
    env = StandUpV3Env(cfg)
    env.reset(seed=3)                         # helpful: already turned the commanded way
    direction, up = _hand_dir(env)
    assert up < cfg.command.level and _angle(direction, env.command) < 0.1
    _, _, terminated, _, info = env.step(np.zeros(6, dtype=np.float32))
    assert not terminated or info["failure"] != "wrong_turn"
    env.reset(seed=3)
    env.set_command(env.command + np.pi)      # the opposite way
    _, reward, terminated, _, info = env.step(np.zeros(6, dtype=np.float32))
    assert terminated and info["failure"] == "wrong_turn"
    assert info["reward_terms"]["wrong_turn"] == -cfg.command.wrong_turn


def test_v3_demo_starts_turn_the_commanded_way_and_the_shaping_telescopes():
    cfg = replace(CFG3, start=replace(CFG3.start, helpful_prob=0.0, demo_prob=1.0,
                                      demo_steps=(60, 60)))
    env = StandUpV3Env(cfg)
    _, info = env.reset(seed=5)
    assert info["start"] == "demo"
    direction, up = _hand_dir(env)
    assert up < cfg.command.level and _angle(direction, env.command) < cfg.command.tolerance
    ev = StandUpV3Env(cfg.for_evaluation())
    ev.reset(seed=5)
    phi0, total = ev._align_potential(), 0.0
    driver = ScriptedStandUpDriver()
    driver.reset(ev, direction=ev.command)
    for _ in range(40):
        _, _, terminated, _, info = ev.step(driver(ev, None))
        total += info["reward_terms"]["align"]
        assert not terminated
    assert total == pytest.approx(ev._align_potential() - phi0) and total > 1.0


def test_the_critic_picks_the_command():
    from polypanda.evaluation.standup import CommandedPolicyDriver

    class Critic:                            # rates the command by how close it is to 30 deg
        def predict_values(self, obs):
            import torch
            o = np.asarray(obs)
            return torch.as_tensor(np.cos(np.arctan2(o[:, -1], o[:, -2]) - np.radians(30)))

    class Policy:
        vec_normalize = None
        model = type("M", (), {"policy": Critic()})()

        def predict(self, obs, deterministic=True):
            return np.zeros(6, dtype=np.float32), None

    env = StandUpV3Env(CFG3.for_evaluation())
    env.reset(seed=0)
    reachable = env.reachable_directions()
    driver = CommandedPolicyDriver(Policy())
    driver.reset(env)
    best = min(reachable, key=lambda d: _angle(d, np.radians(30)))
    assert env.command == pytest.approx(best) and driver.choice == pytest.approx(np.degrees(best))
