"""The stand-up skill's environment (v3 step d2): starts, gate, tower, evaluation set."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from polypanda.environments.standup import OBS_DIM, StandUpConfig, StandUpEnv
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


def test_eval_sets_round_trip_with_their_digest(env, tmp_path):
    s = generate_standup_set(env, "t", 4, 11)
    s.save(tmp_path / "t.json")
    back = StandUpEvalSet.load(tmp_path / "t.json")
    assert back.digest == s.digest and back.episodes == s.episodes
    assert generate_standup_set(env, "t", 4, 11).digest == s.digest     # reproducible
