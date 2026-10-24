import warnings
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from polypanda.control.ik import closing_yaw
from polypanda.environments.grasp import (
    OBS_SCHEMA_V2,
    OBS_SLICES,
    GraspConfig,
    GraspEnv,
    GraspV2Env,
)
from polypanda.environments.tasks import task_spec
from polypanda.evaluation.grasp import ScriptedDriver

CONFIGS = Path(__file__).parents[1] / "configs"
ZERO = np.zeros(7, dtype=np.float32)


@pytest.fixture(scope="module")
def v2_config():
    return GraspConfig.from_yaml(CONFIGS / "grasp_v2.yaml")


def _home_closing_yaw() -> float:
    env = GraspEnv()
    env.reset(options={"cube": (0.5, 0.0, 0.0)})
    return closing_yaw(env._tcp_R())


def test_v2_configs_differ_from_v1_only_by_the_push_terms(v2_config):
    v1 = GraspConfig.from_yaml(CONFIGS / "grasp.yaml")
    stripped = replace(v2_config, reward=replace(v2_config.reward, push=0.0, pushed_away=0.0),
                       episode=replace(v2_config.episode, max_push=0.0))
    assert stripped == v1
    train = GraspConfig.from_yaml(CONFIGS / "grasp_v2_curriculum.yaml")
    assert train.curriculum.pregrasp_start_prob == 0.3
    assert train.for_evaluation() == v2_config
    spec = task_spec("grasp_v2")
    assert spec.env_cls is GraspV2Env and spec.obs_schema == OBS_SCHEMA_V2


def test_check_env(v2_config):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(GraspV2Env(v2_config), warn=True)


def test_target_yaw_is_committed_through_the_45_degree_tie(v2_config):
    env = GraspV2Env(v2_config)
    home = _home_closing_yaw()
    # cube faces at 44 deg from the wrist: the nearest pair is one way, the other is 46 deg away
    _, info = env.reset(options={"cube": (0.5, 0.0, home + np.radians(44))})
    target = info["target_closing_yaw"]
    errors = []
    for _ in range(12):  # turn the wrist (joint 7) steadily past the tie, toward the other pair
        env.data.qpos[env.arm.arm_qpos[6]] += np.radians(1.0)
        mujoco.mj_forward(env.model, env.data)
        m = env._measure()
        assert env._target_yaw == pytest.approx(target)  # never flips to the other pair
        errors.append(m["align"])
    steps = np.abs(np.diff(errors))
    assert steps.max() < np.radians(2)  # continuous: no jump at 45 deg (v1 jumps by 90 deg)
    obs = env._observation()
    e = env._measure()["align"]
    assert obs[OBS_SLICES["align_sin4"]][0] == pytest.approx(np.sin(2 * e), abs=1e-6)


def test_pushing_is_penalized_and_pushing_away_terminates(v2_config):
    env = GraspV2Env(v2_config)
    env.reset(options={"cube": (0.5, 0.0, 0.0)})
    env.data.qpos[env.cube.qpos] += 0.01  # cube shoved 1 cm while the fingers are open
    mujoco.mj_forward(env.model, env.data)
    # the push is attributed to the next step (moved relative to the pre-step position is 0),
    # so measure the displacement-based termination and the per-step term separately
    _, _, terminated, _, info = env.step(ZERO)
    assert not terminated and info["cube_displacement"] == pytest.approx(0.01, abs=2e-3)
    env.data.qpos[env.cube.qpos] += 0.05
    mujoco.mj_forward(env.model, env.data)
    _, _, terminated, _, info = env.step(ZERO)
    assert terminated and info["failure"] == "pushed_away"
    assert info["reward_terms"]["pushed_away"] == -5.0


def test_push_term_charges_motion_during_a_step(v2_config):
    env = GraspV2Env(v2_config)
    env.reset(options={"cube": (0.5, 0.0, 0.0)})
    env.data.qvel[env.cube.qvel] = 0.2  # sliding at 20 cm/s with the fingers open
    _, _, _, _, info = env.step(ZERO)
    assert info["reward_terms"]["push"] < -0.01


def test_scripted_grasp_still_succeeds_in_v2(v2_config):
    env = GraspV2Env(v2_config)
    driver = ScriptedDriver()
    for placement in [(0.45, 0.10, 0.3), (0.55, -0.15, 0.78)]:  # second: ~45 deg from the wrist
        obs, _ = env.reset(options={"cube": placement})
        driver.reset(env)
        for _ in range(env.config.episode.horizon):
            obs, _, terminated, _, info = env.step(driver(env, obs))
            if terminated:
                break
        assert info["is_success"], placement


def test_randomization_is_training_only():
    train = GraspConfig.from_yaml(CONFIGS / "grasp_v2_curriculum.yaml")
    assert train.randomization.enabled and not train.for_evaluation().randomization.enabled
    env = GraspV2Env(train)
    nominal = float(env.model.body_mass[env.cube.body])
    masses, frictions = set(), set()
    for seed in range(6):
        env.reset(seed=seed)
        masses.add(round(float(env.model.body_mass[env.cube.body]) / nominal, 6))
        frictions.add(round(float(env.model.geom_friction[env.cube.geom][0]), 6))
    assert len(masses) == 6 and min(masses) >= 0.5 and max(masses) <= 2.0
    assert len(frictions) == 6 and min(frictions) >= 0.5 and max(frictions) <= 1.5
    env.reset(options={"cube": (0.5, 0.0, 0.0)})  # explicit placement: nominal cube again
    assert env.model.body_mass[env.cube.body] == pytest.approx(nominal)
    assert env.model.geom_friction[env.cube.geom][0] == pytest.approx(1.0)


def test_v1_task_draws_no_randomization():
    a, b = GraspEnv(), GraspEnv()
    a.reset(seed=3)
    b.reset(seed=3)
    np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
    assert a._physics is None


def test_runs_are_evaluated_in_their_own_task_env():
    from polypanda.evaluation.pickplace import grasp_env_for
    root = Path(__file__).parents[1]
    run = root / "runs" / "grasp_ppo_curr_seed0"
    if not run.exists():
        pytest.skip("trained runs are not in git")
    env = grasp_env_for(f"run:{run}")
    assert type(env) is GraspEnv and env.config == GraspConfig()
    assert type(grasp_env_for("scripted")) is GraspEnv
