import mujoco
import numpy as np
import pytest

from polypanda.simulation.assets import load_panda_scene
from polypanda.simulation.panda import PandaIndices, reset_to_home
from polypanda.simulation.scenes import (
    CubeIndices,
    cube_position,
    cube_tilt,
    cube_yaw,
    load_pick_scene,
    set_cube_pose,
)


@pytest.fixture
def scene():
    model, data = load_pick_scene()
    arm, cube = PandaIndices.from_model(model), CubeIndices.from_model(model)
    reset_to_home(model, data, arm)
    return model, data, arm, cube


def _settle(model, data, seconds=2.0):
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def _contact_pairs(model, data):
    names = set()
    for i in range(data.ncon):
        c = data.contact[i]
        a = model.body(model.geom_bodyid[c.geom1]).name
        b = model.body(model.geom_bodyid[c.geom2]).name
        names.add(tuple(sorted((a, b))))
    return names


def test_cube_appended_without_disturbing_arm_indices(scene):
    model, _, arm, cube = scene
    base_model, _ = load_panda_scene()
    base = PandaIndices.from_model(base_model)
    assert model.nq == base_model.nq + 7 and model.nu == base_model.nu
    np.testing.assert_array_equal(arm.arm_qpos, base.arm_qpos)
    np.testing.assert_array_equal(arm.arm_actuators, base.arm_actuators)
    np.testing.assert_allclose(cube.half_size, [0.02, 0.02, 0.03])
    assert model.body_mass[cube.body] == pytest.approx(0.096)


def test_home_places_cube_upright_on_surface(scene):
    _, data, _, cube = scene
    np.testing.assert_allclose(cube_position(data, cube), [0.5, 0.0, 0.03], atol=1e-12)
    assert cube_tilt(data, cube) == pytest.approx(0.0, abs=1e-12)


def test_cube_rests_stably_and_only_touches_floor(scene):
    model, data, _, cube = scene
    start = cube_position(data, cube)
    _settle(model, data)
    assert np.linalg.norm(cube_position(data, cube) - start) < 5e-4
    assert cube_tilt(data, cube) < 1e-3
    assert np.abs(data.qvel[cube.qvel:cube.qvel + 6]).max() < 1e-3
    assert _contact_pairs(model, data) == {("cube", "world")}
    assert max(-data.contact[i].dist for i in range(data.ncon)) < 5e-4


@pytest.mark.parametrize("xy,yaw", [((0.45, 0.10), 0.7), ((0.6, -0.2), -1.2), ((0.35, 0.0), 0.0)])
def test_set_cube_pose_is_kept_after_settling(scene, xy, yaw):
    model, data, _, cube = scene
    set_cube_pose(model, data, cube, xy, yaw)
    np.testing.assert_allclose(cube_position(data, cube)[:2], xy, atol=1e-12)
    assert cube_yaw(data, cube) == pytest.approx(yaw, abs=1e-12)
    _settle(model, data)
    np.testing.assert_allclose(cube_position(data, cube)[:2], xy, atol=5e-4)
    assert cube_yaw(data, cube) == pytest.approx(yaw, abs=1e-3)


def test_reset_is_deterministic():
    model, data = load_pick_scene()
    arm, cube = PandaIndices.from_model(model), CubeIndices.from_model(model)
    finals = []
    for _ in range(2):
        reset_to_home(model, data, arm)
        set_cube_pose(model, data, cube, (0.45, 0.1), 0.3)
        _settle(model, data, 1.0)
        finals.append(data.qpos.copy())
    np.testing.assert_array_equal(finals[0], finals[1])
