"""v4 sensing: the camera model and the classical cube-pose estimator on synthetic scenes."""

import mujoco
import numpy as np
import pytest

from polypanda.perception.camera import NOISE, OVERHEAD, STATION_B, add_camera, camera_pose
from polypanda.perception.estimator import (
    EstimatorConfig,
    estimate_cubes,
    identify,
    min_area_rect,
    world_rays,
    wrap_quarter,
)


def ray_cast(setup, boxes):
    """Exact z-depth of a table (z = 0) and upright boxes (centre, half sizes, yaw), plus the
    index of the box each pixel sees (-1: the table)."""
    K, R, t = setup.intrinsics, setup.rotation(), np.array(setup.pos)
    rays = world_rays(K, R)
    depth = np.where(rays[..., 2] < 0, -t[2] / rays[..., 2], np.inf)
    hit = -np.ones(depth.shape, dtype=int)
    for k, (c, h, yaw) in enumerate(boxes):
        cs, sn = np.cos(-yaw), np.sin(-yaw)
        Rz = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]])
        o, r = Rz @ (t - np.asarray(c)), rays @ Rz.T
        with np.errstate(divide="ignore", invalid="ignore"):
            a = (-np.asarray(h) - o) / r
            b = (np.asarray(h) - o) / r
        tmin = np.nanmax(np.minimum(a, b), axis=-1)
        tmax = np.nanmin(np.maximum(a, b), axis=-1)
        ok = (tmax >= tmin) & (tmin > 0) & (tmin < depth)
        depth[ok], hit[ok] = tmin[ok], k
    expected = np.where(rays[..., 2] < 0, -t[2] / rays[..., 2], np.inf)
    return depth.astype(np.float32), expected.astype(np.float32), hit


def run(setup, boxes, noise=None, seed=0):
    depth, expected, hit = ray_cast(setup, boxes)
    sigma = None
    if noise is not None:
        depth = noise.apply(depth, np.random.default_rng(seed))
        sigma = noise.sigma
    est = estimate_cubes(depth, expected, setup.intrinsics, setup.rotation(),
                         np.array(setup.pos), EstimatorConfig(), sigma)
    return est, hit


def nearest(est, xy):
    return min(est, key=lambda e: np.linalg.norm(e.xy - xy))


def test_min_area_rect_recovers_a_rotated_square():
    g = np.stack(np.meshgrid(np.linspace(-0.02, 0.02, 41), np.linspace(-0.02, 0.02, 41)), -1)
    yaw = np.radians(17.0)
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    pts = g.reshape(-1, 2) @ rot.T + [0.3, -0.1]
    centre, theta, sides = min_area_rect(pts, 0.5)
    assert np.linalg.norm(centre - [0.3, -0.1]) < 1e-4
    assert abs(np.degrees(wrap_quarter(theta - yaw))) < 0.1
    assert sides == pytest.approx((0.04, 0.04), abs=1e-4)


@pytest.mark.parametrize("xy,yaw", [((0.12, -0.31), 23.0), ((-0.55, 0.40), -38.0),
                                    ((0.70, 0.05), 4.0)])
def test_one_cube_at_a_known_pose(xy, yaw):
    half = (0.02, 0.02, 0.03)
    est, _ = run(OVERHEAD, [((*xy, 0.03), half, np.radians(yaw))])
    assert len(est) == 1
    e = est[0]
    assert np.linalg.norm(e.xy - xy) < 0.0015            # 1.5 mm at 3.3 mm per pixel
    assert abs(np.degrees(wrap_quarter(e.yaw - np.radians(yaw)))) < 2.0
    assert e.top_z == pytest.approx(0.06, abs=0.001)


def test_tower_top_and_close_neighbour_are_separate_clusters():
    half = (0.02, 0.02, 0.03)
    boxes = [((0.0, -0.5, 0.03), half, 0.1), ((0.002, -0.5, 0.09), half, 0.2),
             ((0.062, -0.5, 0.03), half, 0.0)]                # 6 cm away: a 2 cm gap
    est, hit = run(STATION_B, boxes)
    assert len(est) == 2
    top = nearest(est, np.array([0.002, -0.5]))
    assert top.top_z == pytest.approx(0.12, abs=0.001)
    assert np.linalg.norm(top.xy - [0.002, -0.5]) < 0.001
    assert abs(np.degrees(wrap_quarter(top.yaw - 0.2))) < 1.0
    # identity from the "segmentation" (box index per pixel): the stack is one target
    ident = identify(top, hit, {0: 0, 1: 0, 2: 1})
    assert ident.target == 0 and not ident.merged


def test_noisy_depth_still_locates_the_cube():
    half = (0.022, 0.022, 0.033)
    est, _ = run(OVERHEAD, [((0.3, 0.3, 0.033), half, 0.4)], NOISE["medium"], seed=3)
    assert len(est) == 1
    assert np.linalg.norm(est[0].xy - [0.3, 0.3]) < 0.004
    assert est[0].top_z == pytest.approx(0.066, abs=0.004)


def test_noise_model_drops_pixels_and_scales_with_distance():
    noise = NOISE["high"]
    depth = np.full((200, 200), 1.5, dtype=np.float32)
    out = noise.apply(depth, np.random.default_rng(0))
    valid = out > 0
    assert 0.03 < 1 - valid.mean() < 0.07
    assert np.std(out[valid] - 1.5) == pytest.approx(noise.sigma(1.5), rel=0.05)
    assert noise.sigma(1.5) > 4 * noise.sigma(0.7)
    assert NOISE["none"].is_clean


def test_mujoco_render_matches_the_pinhole_model():
    """The same estimate from MuJoCo's own depth render: checks the camera convention."""
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1])
    yaw = np.radians(31.0)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.03],
                            pos=[0.25, -0.4, 0.03], quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    add_camera(spec, OVERHEAD)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    t, R = camera_pose(model, data, OVERHEAD.name)
    assert np.allclose(R, OVERHEAD.rotation(), atol=1e-6) and np.allclose(t, OVERHEAD.pos)
    try:
        renderer = mujoco.Renderer(model, OVERHEAD.height, OVERHEAD.width)
    except Exception as exc:  # no OpenGL context on this machine
        pytest.skip(f"no offscreen renderer: {exc}")
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=OVERHEAD.name)
    depth = renderer.render().copy()
    _, expected, _ = ray_cast(OVERHEAD, [])
    renderer.close()
    est = estimate_cubes(depth, expected, OVERHEAD.intrinsics, R, t)
    assert len(est) == 1
    assert np.linalg.norm(est[0].xy - [0.25, -0.4]) < 0.0015
    assert abs(np.degrees(wrap_quarter(est[0].yaw - yaw))) < 2.0
    assert est[0].top_z == pytest.approx(0.06, abs=0.0015)
