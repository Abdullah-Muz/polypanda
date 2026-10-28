"""The any-pose grasp's scripted comparison and the grasp robustness suite (v3 step d1)."""

from pathlib import Path

import numpy as np
import pytest

from polypanda.environments.grasp import GraspConfig, GraspV4Env
from polypanda.evaluation.grasp import ScriptedAnyPoseDriver, run_policy_episode
from polypanda.evaluation.robustness import run_grasp_condition
from polypanda.evaluation.sets import GraspEvalSet

ROOT = Path(__file__).parents[1]
CFG4 = GraspConfig.from_yaml(ROOT / "configs" / "grasp_anypose4_cartesian.yaml")


@pytest.mark.parametrize("lying", [True, False])
def test_the_scripted_any_pose_grasp_lifts_lying_and_upright_cubes(lying):
    env = GraspV4Env(CFG4.for_evaluation())
    result, _ = run_policy_episode(env, ScriptedAnyPoseDriver(), (0.55, 0.1, 0.6), 0, "far",
                                   lying=lying)
    assert result.success and result.gate_closures == 1


def test_the_grasp_suite_perturbs_each_episode_pairs_skills_and_restores_the_model():
    env = GraspV4Env(CFG4.for_evaluation())
    mass, size = float(env.model.body_mass[env.cube.body]), env.cube.half_size.copy()
    lying = GraspEvalSet.load(ROOT / "eval_sets" / "grasp_lying_v1.json").subset(2)
    summary, results = run_grasp_condition(ScriptedAnyPoseDriver(), lying, "combined", 0, env)
    assert summary["episodes"] == 2 and len(results) == 2
    assert all(r.mass_scale != 1 and r.size_scale != 1 and r.arm_offset_max > 0
               for r in results)
    assert float(env.model.body_mass[env.cube.body]) == pytest.approx(mass)
    assert np.allclose(env.cube.half_size, size)
    _, again = run_grasp_condition(ScriptedAnyPoseDriver(), lying, "combined", 0, env)
    assert [r.mass_scale for r in again] == [r.mass_scale for r in results]   # same draws
    with pytest.raises(ValueError):
        run_grasp_condition(ScriptedAnyPoseDriver(), lying, "push", 0, env)
