import warnings
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from polypanda.environments.grasp import (
    OBS_DIM,
    DistractorConfig,
    GraspConfig,
    GraspV2BlindClutterEnv,
    GraspV3Env,
)
from polypanda.evaluation.grasp import ScriptedDriver
from polypanda.evaluation.sets import GraspEvalSet, generate_grasp_eval_set
from polypanda.simulation.panda import reset_to_home
from polypanda.simulation.scenes import (
    DISTRACTOR_PARK,
    WIDE_X_RANGE,
    WIDE_Y_RANGE,
    CubeIndices,
    cube_position,
    load_pick_scene,
)

CONFIG = replace(GraspConfig(), distractors=DistractorConfig(max_count=2))
CUBE, OTHERS = (0.5, 0.0, 0.3), [(0.5, 0.15, 0.0), (0.38, -0.12, 0.7)]
ZERO = np.zeros(7, dtype=np.float32)


def test_distractors_rest_parked_out_of_reach_at_home():
    model, data = load_pick_scene(distractors=2)
    from polypanda.simulation.panda import PandaIndices
    reset_to_home(model, data, PandaIndices.from_model(model))
    others = CubeIndices.distractors(model)
    assert len(others) == 2
    for k, o in enumerate(others):
        np.testing.assert_allclose(cube_position(data, o), [*DISTRACTOR_PARK[k], 0.03], atol=1e-6)


def test_check_env():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(GraspV3Env(CONFIG), warn=True)


def test_random_resets_place_separated_distractors_and_observe_them():
    env = GraspV3Env(CONFIG)
    counts = set()
    for seed in range(12):
        obs, info = env.reset(seed=seed)
        assert obs.shape == (OBS_DIM + 8,)
        counts.add(len(info["distractors"]))
        cube = cube_position(env.data, env.cube)[:2]
        for k, other in enumerate(env.others):
            flag = obs[OBS_DIM + 4 * k]
            assert flag == float(env.present[k])
            if env.present[k]:
                xy = cube_position(env.data, other)[:2]
                assert np.linalg.norm(xy - cube) >= 0.10 - 1e-9
            else:
                np.testing.assert_array_equal(obs[OBS_DIM + 4 * k + 1:OBS_DIM + 4 * k + 4], 0)
    assert counts == {0, 1, 2}


def test_explicit_placements_control_the_distractors():
    env = GraspV3Env(CONFIG)
    _, info = env.reset(options={"cube": CUBE})
    assert info["distractors"] == [] and not any(env.present)
    _, info = env.reset(options={"cube": CUBE, "distractors": OTHERS})
    assert all(env.present)
    np.testing.assert_allclose(cube_position(env.data, env.others[1])[:2], OTHERS[1][:2])


def test_disturbing_and_knocking_a_distractor():
    env = GraspV3Env(CONFIG)
    env.reset(options={"cube": CUBE, "distractors": OTHERS})
    env.data.qvel[env.others[0].qvel] = 0.3  # shove it sideways
    _, _, terminated, _, info = env.step(ZERO)
    assert info["reward_terms"]["disturb"] < 0 and not terminated
    q = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]  # on its side
    env.data.qpos[env.others[1].qpos:env.others[1].qpos + 7] = [0.38, -0.12, 0.02, *q]
    mujoco.mj_forward(env.model, env.data)
    _, _, terminated, _, info = env.step(ZERO)
    assert terminated and info["failure"] == "knocked_distractor" and not info["is_success"]


def test_scripted_grasp_succeeds_among_distractors():
    env = GraspV3Env(CONFIG)
    driver = ScriptedDriver()
    obs, _ = env.reset(options={"cube": CUBE, "distractors": OTHERS})
    driver.reset(env)
    for _ in range(env.config.episode.horizon):
        obs, _, terminated, _, info = env.step(driver(env, obs))
        if terminated:
            break
    assert info["is_success"]
    for k, other in enumerate(env.others):
        np.testing.assert_allclose(cube_position(env.data, other)[:2], OTHERS[k][:2], atol=0.005)


def test_clutter_sets_round_trip(tmp_path):
    s = generate_grasp_eval_set("c", 20, 3, WIDE_X_RANGE, WIDE_Y_RANGE, distractors=2)
    assert s.distractors.shape == (20, 2, 3)
    for p, others in zip(s.placements, s.distractors, strict=True):
        pts = [p[:2], *others[:, :2]]
        d = [np.linalg.norm(a - b) for i, a in enumerate(pts) for b in pts[i + 1:]]
        assert min(d) >= 0.10
    s.save(tmp_path / "c.json")
    loaded = GraspEvalSet.load(tmp_path / "c.json")
    assert loaded.digest == s.digest
    np.testing.assert_array_equal(loaded.distractors, s.distractors)
    plain = generate_grasp_eval_set("c", 20, 3, WIDE_X_RANGE, WIDE_Y_RANGE)
    np.testing.assert_array_equal(plain.placements, s.placements)  # same target placements


def test_blind_clutter_env_keeps_the_v2_observation():
    env = GraspV2BlindClutterEnv(CONFIG)
    obs, _ = env.reset(options={"cube": CUBE, "distractors": OTHERS})
    assert obs.shape == (OBS_DIM,) and all(env.present)


@pytest.mark.parametrize("name", ["grasp_v3_clutter_curriculum.yaml"])
def test_v3_training_config_is_wide_plus_distractors(name):
    root = Path(__file__).parents[1] / "configs"
    v3 = GraspConfig.from_yaml(root / name)
    wide = GraspConfig.from_yaml(root / "grasp_wide_curriculum.yaml")
    assert v3.distractors.max_count == 2
    assert replace(v3, distractors=DistractorConfig()) == wide
