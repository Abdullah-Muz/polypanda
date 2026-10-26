"""Station frames: a scene rotated about the base axis, seen through StationView, is the original
scene to the policy - same observations, same response to the same actions."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from polypanda.environments.grasp import GraspConfig, GraspV2Env
from polypanda.environments.place import PlaceConfig, PlaceEnv
from polypanda.environments.reach import ActionConfig, CartesianConfig
from polypanda.supervisor.stations import StationView

CART4 = ActionConfig(space="cartesian")
CART6 = ActionConfig(space="cartesian", cartesian=CartesianConfig(dof=6))


def _rotate_scene(env, angle):
    """Turn the arm (joint 1) and the cube about the base axis by `angle`."""
    d = env.data
    q1 = env.arm.arm_qpos[0]
    d.qpos[q1] += angle
    d.ctrl[env.arm.arm_actuators[0]] += angle
    p = d.qpos[env.cube.qpos:env.cube.qpos + 3].copy()
    c, s = np.cos(angle), np.sin(angle)
    d.qpos[env.cube.qpos:env.cube.qpos + 2] = [c * p[0] - s * p[1], s * p[0] + c * p[1]]
    quat = d.qpos[env.cube.qpos + 3:env.cube.qpos + 7].copy()
    turn, out = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)]), np.zeros(4)
    mujoco.mju_mulQuat(out, turn, quat)
    d.qpos[env.cube.qpos + 3:env.cube.qpos + 7] = out
    v = d.qvel[env.cube.qvel:env.cube.qvel + 2].copy()  # world-frame linear velocity turns too
    d.qvel[env.cube.qvel:env.cube.qvel + 2] = [c * v[0] - s * v[1], s * v[0] + c * v[1]]
    mujoco.mj_forward(env.model, d)


@pytest.mark.parametrize("angle", [np.radians(90), np.radians(-140), np.radians(-40)])
def test_rotated_grasp_scene_looks_and_behaves_the_same(angle):
    config = replace(GraspConfig(), action=CART4)
    ref, rot = GraspV2Env(config), GraspV2Env(config)
    for env in (ref, rot):
        env.reset(options={"cube": (0.5, 0.05, 0.3)})
    _rotate_scene(rot, angle)
    view = StationView(rot, angle)
    obs_ref, _ = ref.reset(options={"continue": True})
    obs_rot, _ = view.reset()
    np.testing.assert_allclose(obs_rot, obs_ref, atol=1e-5)
    rng = np.random.default_rng(0)
    for _ in range(40):  # the same actions move both the same way (small: at -140 deg a large
        a = rng.uniform(-0.4, 0.4, 4).astype(np.float32)  # random walk reaches joint 1's limit)
        obs_ref, r_ref, *_ = ref.step(a)
        obs_rot, r_rot, *_ = view.step(a)
    np.testing.assert_allclose(obs_rot, obs_ref, atol=2e-3)
    assert r_rot == pytest.approx(r_ref, abs=1e-3)


def test_rotated_place_scene_with_tilting_actions():
    angle = np.radians(-110)
    config = replace(PlaceConfig(), action=CART6)
    ref, rot = PlaceEnv(config), PlaceEnv(config)
    for env in (ref, rot):
        env.reset(options={"state_index": 3})
    target = ref.target.copy()
    _rotate_scene(rot, angle)
    c, s = np.cos(angle), np.sin(angle)
    target_rot = [c * target[0] - s * target[1], s * target[0] + c * target[1], target[2]]
    view = StationView(rot, angle)
    obs_ref, _ = ref.reset(options={"continue": True, "target": target})
    obs_rot, _ = view.reset(target=target_rot)
    np.testing.assert_allclose(obs_rot, obs_ref, atol=1e-5)
    rng = np.random.default_rng(1)
    for _ in range(30):
        a = rng.uniform(-1, 1, 6).astype(np.float32)
        obs_ref, *_ = ref.step(a)
        obs_rot, *_ = view.step(a)
    up = view.slices["cube_up"]  # friction-cone directions are not exactly rotation-invariant,
    np.testing.assert_allclose(obs_rot[up], obs_ref[up], atol=1e-2)  # so the held cube's tilt
    rest = np.ones(len(obs_ref), bool)  # may drift by a few milliradians
    rest[up] = False
    np.testing.assert_allclose(obs_rot[rest], obs_ref[rest], atol=2e-3)


def test_station_box_bounds_rotate_with_the_frame():
    env = GraspV2Env(replace(GraspConfig(), action=CART4))
    env.reset(options={"cube": (0.5, 0.0, 0.0)})
    angle = np.radians(90)
    _rotate_scene(env, angle)
    view = StationView(env, angle)
    view.reset()
    y0 = env.data.site_xpos[env.arm.tcp_site][1]  # ~0.55: beyond the unrotated box (y <= 0.5)
    assert y0 > 0.5
    for _ in range(10):  # "forward" in the station frame is +y in the world
        view.step(np.array([1.0, 0, 0, 0], dtype=np.float32))
    assert env.adapter.target_pos[1] > y0 + 0.05  # not clipped at the world box's y = 0.5
