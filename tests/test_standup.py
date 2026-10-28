"""The stand-up skill's environment (v3 step d2): starts, gate, tower, evaluation set."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from polypanda.environments.standup import (
    GAP_CAP,
    OBS_DIM,
    OBS_DIM_V2,
    StandUpConfig,
    StandUpEnv,
    StandUpV2Env,
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


def test_round_5_releases_inside_the_success_tolerance_and_explores_by_state():
    from polypanda.training.config import TrainConfig
    cfg = StandUpConfig.from_yaml(ROOT / "configs" / "standup_v2_r5.yaml")
    assert cfg.gate.xy_tol == pytest.approx(0.015)
    assert cfg.gate.xy_tol < cfg.check.tolerance                       # room to settle
    assert cfg.reward.clearance_margin == pytest.approx(0.15)
    train = TrainConfig.from_yaml(ROOT / "configs" / "standup_v2_r5_ppo.yaml")
    assert train.task == "standup_v2" and train.env_config == "configs/standup_v2_r5.yaml"
    assert train.ppo.use_sde and train.ppo.sde_sample_freq == 32


def test_eval_sets_round_trip_with_their_digest(env, tmp_path):
    s = generate_standup_set(env, "t", 4, 11)
    s.save(tmp_path / "t.json")
    back = StandUpEvalSet.load(tmp_path / "t.json")
    assert back.digest == s.digest and back.episodes == s.episodes
    assert generate_standup_set(env, "t", 4, 11).digest == s.digest     # reproducible
