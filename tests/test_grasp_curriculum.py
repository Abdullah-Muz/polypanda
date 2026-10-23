"""Training-only pre-grasped starts: correct state, correct frequency, never in evaluation."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from polypanda.control.ik import PoseIK, top_down_quat
from polypanda.environments.grasp import OBS_SLICES, CurriculumConfig, GraspConfig, GraspEnv
from polypanda.simulation.panda import tcp_position
from polypanda.simulation.scenes import cube_position

CONFIGS = Path(__file__).parents[1] / "configs"


def _env(prob: float) -> GraspEnv:
    return GraspEnv(replace(GraspConfig(), curriculum=CurriculumConfig(prob)))


def test_curriculum_config_differs_from_task_only_in_curriculum():
    task = GraspConfig.from_yaml(CONFIGS / "grasp.yaml")
    train = GraspConfig.from_yaml(CONFIGS / "grasp_curriculum.yaml")
    assert train.curriculum.pregrasp_start_prob == 0.3
    assert replace(train, curriculum=task.curriculum) == task
    assert train.for_evaluation() == task


def test_pregrasp_start_holds_the_cube_on_the_table():
    env = _env(1.0)
    obs, info = env.reset(seed=0)
    assert info["start"] == "pregrasp" and env.closed
    state = env.monitor.update(env.data)
    # On the table the coupled fingers may squeeze one-sidedly (gravity sag offsets the hand by
    # a few mm while floor friction pins the cube); lifting makes it a two-finger grip, see
    # test_pregrasp_start_can_be_lifted_to_success. Here: closed around it with the full squeeze.
    assert state.left_contact or state.right_contact
    assert state.left_force + state.right_force == pytest.approx(2.0, abs=0.5)
    assert info["distance"] < 0.01                    # TCP at the cube centre
    assert abs(cube_position(env.data, env.cube)[2] - env.cube.half_size[2]) < 2e-3
    assert obs[OBS_SLICES["gripper_closed"]][0] == 1.0
    assert obs[OBS_SLICES["finger_gap"]][0] == pytest.approx(0.04, abs=2e-3)  # on the cube faces
    np.testing.assert_array_equal(env.adapter.q_cmd, env.data.ctrl[env.arm.arm_actuators])


def test_pregrasp_start_can_be_lifted_to_success():
    env = _env(1.0)
    env.reset(seed=1)
    ik = PoseIK()
    R = env.data.site_xmat[env.arm.tcp_site].reshape(3, 3)
    yaw = np.arctan2(R[1, 1], R[0, 1])
    target = tcp_position(env.data, env.arm) + [0.0, 0.0, 0.10]
    for _ in range(60):
        q = env.data.qpos[env.arm.arm_qpos]
        _, _, _, _, info = env.step(ik.action(q, target, top_down_quat(yaw), 0.2,
                                              env.adapter.max_delta))
    assert info["is_success"] and info["gripper_closed"]


def test_curriculum_frequency_and_never_with_explicit_placement():
    env = _env(0.3)
    starts = [env.reset(seed=s)[1]["start"] for s in range(60)]
    frac = starts.count("pregrasp") / len(starts)
    assert 0.1 < frac < 0.5
    for s in range(10):
        _, info = env.reset(seed=s, options={"cube": (0.5, 0.0, 0.3)})
        assert info["start"] == "home" and not env.closed


def test_disabled_curriculum_leaves_default_random_stream_unchanged():
    a, b = GraspEnv(), _env(0.0)
    for s in range(5):
        assert a.reset(seed=s)[1]["cube"] == b.reset(seed=s)[1]["cube"]
