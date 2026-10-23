import mujoco
import numpy as np
import pytest

from polypanda.control.ik import (
    PoseIK,
    closing_yaw,
    grasp_closing_yaw,
    orientation_error,
    top_down_quat,
)
from polypanda.simulation.panda import PandaIndices, reset_to_home, tcp_position
from polypanda.simulation.scenes import CubeIndices, load_pick_scene, set_cube_pose

# Cube placement region where top-down grasps were all feasible when mapped (300 samples).
CUBE_X, CUBE_Y = (0.35, 0.65), (-0.25, 0.25)
PREGRASP_HEIGHT = 0.12  # m above the cube centre


@pytest.fixture(scope="module")
def ik():
    return PoseIK()


def _quat_to_mat(quat):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    return R.reshape(3, 3)


@pytest.mark.parametrize("yaw", [0.0, 0.7, -2.0, np.pi])
def test_top_down_quat_points_down_and_closes_along_yaw(yaw):
    R = _quat_to_mat(top_down_quat(yaw))
    np.testing.assert_allclose(R[:, 2], [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(R[:, 1], [np.cos(yaw), np.sin(yaw), 0], atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)
    assert closing_yaw(R) == pytest.approx(np.arctan2(np.sin(yaw), np.cos(yaw)))


def test_home_closes_along_world_x(ik):
    model, data = load_pick_scene()
    arm = PandaIndices.from_model(model)
    reset_to_home(model, data, arm)
    _, R = ik.forward(data.qpos[arm.arm_qpos])
    assert closing_yaw(R) == pytest.approx(0.0, abs=1e-3)
    np.testing.assert_allclose(R[:, 2], [0, 0, -1], atol=1e-3)


@pytest.mark.parametrize("cube_yaw,current,expected", [
    (0.0, 0.0, 0.0), (1.6, 0.0, 1.6 - np.pi / 2), (-0.7, 0.0, -0.7), (3.0, 0.0, 3.0 - np.pi),
])
def test_grasp_yaw_uses_square_symmetry(cube_yaw, current, expected):
    result = grasp_closing_yaw(cube_yaw, current)
    assert result == pytest.approx(expected)
    assert abs(result - current) <= np.pi / 4 + 1e-12


def test_orientation_error_is_world_rotation_vector():
    R = _quat_to_mat(top_down_quat(0.0))
    assert np.allclose(orientation_error(R, top_down_quat(0.0)), 0)
    # Turning the closing yaw by +0.1 rad is a rotation of +0.1 about world z.
    np.testing.assert_allclose(orientation_error(R, top_down_quat(0.1)), [0, 0, 0.1], atol=1e-9)


def test_forward_matches_simulator(ik):
    model, data = load_pick_scene()
    arm = PandaIndices.from_model(model)
    reset_to_home(model, data, arm)
    q = data.qpos[arm.arm_qpos] + np.array([0.2, -0.1, 0.3, 0.2, -0.2, 0.1, 0.4])
    data.qpos[arm.arm_qpos] = q
    mujoco.mj_forward(model, data)
    pos, R = ik.forward(q)
    np.testing.assert_allclose(pos, tcp_position(data, arm), atol=1e-12)
    np.testing.assert_allclose(R, data.site_xmat[arm.tcp_site].reshape(3, 3), atol=1e-12)


def test_grasp_poses_reachable_and_collision_free_across_region(ik):
    model, data = load_pick_scene()
    arm, cube = PandaIndices.from_model(model), CubeIndices.from_model(model)
    reset_to_home(model, data, arm)
    q_home = data.qpos[arm.arm_qpos].copy()
    home_yaw = closing_yaw(ik.forward(q_home)[1])
    rng = np.random.default_rng(7)
    for _ in range(25):
        xy = (rng.uniform(*CUBE_X), rng.uniform(*CUBE_Y))
        yaw = rng.uniform(-np.pi, np.pi)
        quat = top_down_quat(grasp_closing_yaw(yaw, home_yaw))
        q = q_home
        for height in (PREGRASP_HEIGHT, 0.0):
            target = [*xy, cube.half_size[2] + height]
            q, converged = ik.solve(q, target, quat)
            assert converged, (xy, yaw, height)
            assert np.min(np.minimum(q - ik.q_low, ik.q_high - q)) > 0.02
            reset_to_home(model, data, arm)
            set_cube_pose(model, data, cube, xy, yaw)
            data.qpos[arm.arm_qpos] = q
            mujoco.mj_forward(model, data)
            touching = {
                frozenset((model.body(model.geom_bodyid[c.geom1]).name,
                           model.body(model.geom_bodyid[c.geom2]).name))
                for c in (data.contact[i] for i in range(data.ncon))
            }
            assert touching <= {frozenset({"world", "cube"})}, (xy, yaw, height, touching)


def test_action_is_normalized_and_direction_preserving(ik):
    model, data = load_pick_scene()
    arm = PandaIndices.from_model(model)
    reset_to_home(model, data, arm)
    q = data.qpos[arm.arm_qpos]
    far = ik.action(q, [0.4, 0.2, 0.2], top_down_quat(0.5), gain=1.0, max_delta=0.015)
    assert far.dtype == np.float32 and np.abs(far).max() == pytest.approx(1.0)
    raw = ik.step(q, [0.4, 0.2, 0.2], top_down_quat(0.5))
    np.testing.assert_allclose(far / np.linalg.norm(far), raw / np.linalg.norm(raw), atol=1e-6)
    near = ik.action(q, ik.forward(q)[0], top_down_quat(closing_yaw(ik.forward(q)[1])),
                     gain=1.0, max_delta=0.015)
    assert np.abs(near).max() < 0.05
