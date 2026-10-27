"""Cube orientation either way up (v3): an upside-down cube is as upright as an upright one."""

import mujoco
import numpy as np

from polypanda.simulation.scenes import CubeIndices, cube_tilt, cube_up, load_pick_scene


def _set(model, data, cube, quat):
    data.qpos[cube.qpos + 3:cube.qpos + 7] = quat
    mujoco.mj_forward(model, data)


def test_tilt_and_up_axis_either_way_up():
    model, data = load_pick_scene()
    cube = CubeIndices.from_model(model, "cube")
    rng = np.random.default_rng(0)
    for _ in range(50):  # any tilt below 90 deg reads exactly as before the change
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        _set(model, data, cube, q)
        z = data.xmat[cube.body].reshape(3, 3)[:, 2]
        if z[2] >= 0:
            assert cube_tilt(data, cube) == float(np.arccos(np.clip(z[2], -1.0, 1.0)))
        if z[2] >= -0.5:                             # continuous through lying on a side
            assert np.array_equal(cube_up(data, cube), z)
    _set(model, data, cube, [0.0, 1.0, 0.0, 0.0])                     # upside down
    assert cube_tilt(data, cube) < 1e-9 and cube_up(data, cube)[2] > 0.999
    _set(model, data, cube, [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])  # on a side
    assert abs(cube_tilt(data, cube) - np.pi / 2) < 1e-6
