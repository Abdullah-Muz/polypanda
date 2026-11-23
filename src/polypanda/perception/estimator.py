"""A classical cube-pose estimator for one depth image (no learning).

Steps, per image:

1. Back-project every valid pixel to a world point with the pinhole model (`backproject`).
2. Known-geometry subtraction: a pixel is foreground if it is clearly nearer than the depth the
   known scene would give there (`expected`: the cell's static geometry from its CAD model plus
   the robot at its measured joint angles, rendered without any cubes) and more than
   `min_height` above the table. This removes the table plane, the fixtures and the arm.
3. Cluster the foreground pixels: neighbouring pixels (8-connected) join one cluster when their
   depths differ by less than `link_tol`.
4. Per cluster: the top height h is the median height of its highest points (within
   `top_band`). Top-face pixels are those at most `top_cut` (or `cut_sigma` sigma) below h and
   not far above it. Each top-face pixel's ray is intersected with the horizontal plane z = h,
   so depth noise moves the top height (averaged over the face) but not the pixels' xy.
   The cube's yaw (mod 90 deg) is the orientation of the face's points by their fourth-order
   moment (`square_yaw`): the minimum-area rectangle, also computed and reported as
   `yaw_rect`, snaps to the pixel grid for cubes turned a few degrees from it. The cube's xy is
   the centre of the bounding rectangle at that yaw (between the 3 % and 97 % quantiles along
   each side, so a few stray points do not pull it), and its height the top face's height.

5. Occlusion check: if anything nearer to the camera than the top face (the arm, a taller
   cube) borders it within `occlusion_ring` pixels, part of the face may be hidden and the
   estimate is flagged `occluded`; a controller would withhold it and look again.

On a tower the cluster holds the whole stack, and the estimate is the top cube's pose.

When the sensor's noise model is known (`sigma`, metres as a function of depth), the margins of
steps 2-4 grow to `k_sigma` times it, as a datasheet-tuned pipeline would set them.

`identify` says which cube a cluster is from a segmentation render. That is evaluation only:
identity comes from the simulator's per-pixel geom ids, never the pose.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from polypanda.perception.camera import Intrinsics


@dataclass(frozen=True)
class EstimatorConfig:
    fg_margin: float = 0.010   # m: foreground is at least this much nearer than the known scene
    min_height: float = 0.015  # m above the table (removes the bench and anything flat on it)
    link_tol: float = 0.015    # m: neighbouring pixels join one cluster below this depth step
    top_band: float = 0.008    # m: points within this of the top height estimate it
    top_cut: float = 0.001     # m: top-face points lie at most this far below the top height ...
    cut_sigma: float = 0.5     # ... or this many sigma, if larger
    k_sigma: float = 3.0       # margins become at least k_sigma * sigma(depth) if sigma is known
    min_pixels: int = 20       # smaller clusters are discarded
    min_top_pixels: int = 8    # clusters with fewer top-face pixels are discarded
    yaw_step: float = 0.5      # deg, coarse rectangle search (refined to a tenth of it)
    occlusion_ring: int = 2    # px around the top face checked for anything nearer
    trim: float = 0.03         # quantile trimmed from each side of the rectangle for its centre


@dataclass
class CubeEstimate:
    xy: np.ndarray             # m, top-face centre (the rectangle at `yaw` around its points)
    xy_centroid: np.ndarray    # m, mean of the top-face points (an alternative centre)
    yaw: float                 # rad, in [-pi/4, pi/4): the face's orientation mod 90 deg
    yaw_rect: float            # rad, the minimum-area rectangle's orientation (an alternative)
    top_z: float               # m, height of the top face
    sides: tuple[float, float]  # m, rectangle side lengths (about a pixel short of the face)
    pixels: np.ndarray = field(repr=False)      # flat pixel indices of the whole cluster
    top_pixels: np.ndarray = field(repr=False)  # flat pixel indices of its top face
    occluded: bool = False     # something nearer borders the top face: the estimate is withheld


def world_rays(K: Intrinsics, R: np.ndarray) -> np.ndarray:
    """(H, W, 3) per-pixel rays in the world frame, scaled to unit depth along the optical axis."""
    return K.rays() @ R.T


def backproject(depth: np.ndarray, K: Intrinsics, R: np.ndarray, t: np.ndarray,
                rays: np.ndarray | None = None) -> np.ndarray:
    """(H, W, 3) world points; NaN where the depth is invalid (0 or not finite)."""
    rays = world_rays(K, R) if rays is None else rays
    d = np.where(np.isfinite(depth) & (depth > 0), depth, np.nan)
    return t + d[..., None] * rays


def min_area_rect(xy: np.ndarray, step_deg: float = 0.5
                  ) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Centre, orientation (rad, in [0, pi/2)) and side lengths of the smallest rectangle around
    2-D points, by a coarse search over orientations refined around the best one."""
    c = xy.mean(axis=0)
    p = xy - c

    def search(angles):
        cs, sn = np.cos(angles), np.sin(angles)
        u = p[:, :1] * cs + p[:, 1:] * sn
        v = -p[:, :1] * sn + p[:, 1:] * cs
        area = (u.max(0) - u.min(0)) * (v.max(0) - v.min(0))
        k = int(np.argmin(area))
        return angles[k], u[:, k], v[:, k]

    coarse = np.radians(np.arange(0.0, 90.0, step_deg))
    best, _, _ = search(coarse)
    fine = best + np.radians(np.linspace(-step_deg, step_deg, 21))
    theta, u, v = search(fine)
    uc, vc = (u.max() + u.min()) / 2, (v.max() + v.min()) / 2
    axis_u = np.array([np.cos(theta), np.sin(theta)])
    axis_v = np.array([-np.sin(theta), np.cos(theta)])
    centre = c + uc * axis_u + vc * axis_v
    return centre, float(theta % (np.pi / 2)), (float(np.ptp(u)), float(np.ptp(v)))


def square_yaw(xy: np.ndarray) -> float:
    """Orientation (rad, in [0, pi/2)) of a square filled with points, from its fourth-order
    moment: sum((u^4 + v^4)) over the points is smallest along the square's axes, and with
    z = x + iy about the centroid it is const + Re(exp(-4i theta) sum(z^4)) / 4."""
    z = (xy[:, 0] - xy[:, 0].mean()) + 1j * (xy[:, 1] - xy[:, 1].mean())
    return float(((np.angle(np.sum(z**4)) + np.pi) / 4) % (np.pi / 2))


def rect_at(xy: np.ndarray, theta: float, trim: float = 0.0
            ) -> tuple[np.ndarray, tuple[float, float]]:
    """Centre and side lengths of the bounding rectangle of 2-D points at orientation theta.
    With `trim` > 0 the centre is taken between the trim and 1 - trim quantiles along each axis
    instead of the extremes, so a few stray points on one side do not pull it."""
    axis_u = np.array([np.cos(theta), np.sin(theta)])
    axis_v = np.array([-np.sin(theta), np.cos(theta)])
    u, v = xy @ axis_u, xy @ axis_v
    (u0, u1), (v0, v1) = np.quantile(u, [trim, 1 - trim]), np.quantile(v, [trim, 1 - trim])
    centre = (u0 + u1) / 2 * axis_u + (v0 + v1) / 2 * axis_v
    return centre, (float(np.ptp(u)), float(np.ptp(v)))


def wrap_quarter(angle: float) -> float:
    """An angle (rad) folded into [-pi/4, pi/4): a square's yaw is defined mod 90 deg."""
    return float((angle + np.pi / 4) % (np.pi / 2) - np.pi / 4)


def _shifted(a: np.ndarray, dv: int, du: int) -> tuple[np.ndarray, np.ndarray]:
    """Views of `a` at pixels p and p + (dv, du), over the pixels where both exist."""
    h, w = a.shape[:2]
    ra, rb = slice(0, h - dv), slice(dv, h)
    ca, cb = (slice(0, w - du), slice(du, w)) if du >= 0 else (slice(-du, w), slice(0, w + du))
    return a[ra, ca], a[rb, cb]


def cluster_pixels(mask: np.ndarray, depth: np.ndarray, tol: np.ndarray | float
                   ) -> np.ndarray:
    """Connected components of `mask` (8-connected) whose neighbouring depths differ by less
    than `tol` (a scalar or a per-pixel array). Returns labels, -1 outside the mask."""
    labels = -np.ones(mask.shape, dtype=int)
    n = int(mask.sum())
    if n == 0:
        return labels
    index = -np.ones(mask.shape, dtype=int)
    index[mask] = np.arange(n)
    tol = np.broadcast_to(tol, mask.shape)
    rows, cols = [], []
    for dv, du in ((0, 1), (1, 0), (1, 1), (1, -1)):
        ma, mb = _shifted(mask, dv, du)
        da, db = _shifted(depth, dv, du)
        ta, _ = _shifted(tol, dv, du)
        ia, ib = _shifted(index, dv, du)
        ok = ma & mb & (np.abs(da - db) < ta)
        rows.append(ia[ok])
        cols.append(ib[ok])
    r, c = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    labels[mask] = comp
    return labels


def estimate_cubes(depth: np.ndarray, expected: np.ndarray, K: Intrinsics, R: np.ndarray,
                   t: np.ndarray, cfg: EstimatorConfig | None = None,
                   sigma: Callable[[np.ndarray], np.ndarray] | None = None,
                   rays: np.ndarray | None = None) -> list[CubeEstimate]:
    """Cube poses in one depth image. `expected`: the known scene's depth without cubes (same
    camera); `R`, `t`: the camera pose the estimator believes (camera-to-world)."""
    cfg = cfg or EstimatorConfig()
    rays = world_rays(K, R) if rays is None else rays
    depth = np.asarray(depth, dtype=float)
    valid = np.isfinite(depth) & (depth > 0)
    sig = np.zeros_like(depth) if sigma is None else np.asarray(sigma(np.where(valid, depth, 0)))
    pts = backproject(depth, K, R, t, rays)
    z = pts[..., 2]
    margin = np.maximum(cfg.fg_margin, cfg.k_sigma * sig)
    with np.errstate(invalid="ignore"):
        fg = valid & (depth < expected - margin) & (z > cfg.min_height)
    labels = cluster_pixels(fg, depth, cfg.link_tol + cfg.k_sigma * np.sqrt(2) * sig)
    flat_labels = labels.ravel()
    order = np.argsort(flat_labels, kind="stable")
    counts = np.bincount(flat_labels[flat_labels >= 0]) if fg.any() else np.zeros(0, int)
    starts = np.concatenate([[0], np.cumsum(counts)]) + int((flat_labels < 0).sum())
    zf, sf = z.ravel(), sig.ravel()
    flat_rays = rays.reshape(-1, 3)
    out = []
    for k, n in enumerate(counts):
        if n < cfg.min_pixels:
            continue
        pix = order[starts[k]:starts[k + 1]]
        zc = zf[pix]
        band = max(cfg.top_band, cfg.k_sigma * float(np.median(sf[pix])))
        high = np.percentile(zc, 98)
        h = float(np.median(zc[np.abs(zc - high) < band]))
        h = float(np.median(zc[np.abs(zc - h) < band]))
        # Top-face pixels: at most `cut` below h. A side-face pixel let in this way projects
        # outward by its depth below h times (horizontal / vertical distance to the camera), so
        # the cut is kept small; with noise it loses some top pixels at random instead.
        cut = max(cfg.top_cut, cfg.cut_sigma * float(np.median(sf[pix])))
        top = pix[(zc > h - cut) & (zc < h + band)]
        if len(top) < cfg.min_top_pixels:
            continue
        r = flat_rays[top]
        s = (h - t[2]) / r[:, 2]
        xy = t[:2] + s[:, None] * r[:, :2]
        theta = square_yaw(xy)
        centre, sides = rect_at(xy, theta, cfg.trim)
        _, theta_rect, _ = min_area_rect(xy, cfg.yaw_step)
        occluded = _occluded(top, depth, valid, max(cfg.fg_margin, cfg.k_sigma * float(
            np.median(sf[top]))), cfg.occlusion_ring)
        out.append(CubeEstimate(centre, xy.mean(axis=0), wrap_quarter(theta),
                                wrap_quarter(theta_rect), h, sides, pix, top, occluded))
    return out


def _occluded(top: np.ndarray, depth: np.ndarray, valid: np.ndarray, margin: float,
              ring: int) -> bool:
    """Something nearer to the camera than the top face (the arm, a taller cube) borders it
    within `ring` pixels, so part of the face may be hidden."""
    rows, cols = np.unravel_index(top, depth.shape)
    r0, r1 = max(rows.min() - ring, 0), min(rows.max() + ring + 1, depth.shape[0])
    c0, c1 = max(cols.min() - ring, 0), min(cols.max() + ring + 1, depth.shape[1])
    face = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    face[rows - r0, cols - c0] = True
    around = binary_dilation(face, iterations=ring) & ~face
    d, ok = depth[r0:r1, c0:c1], valid[r0:r1, c0:c1]
    return bool(np.any(around & ok & (d < depth.flat[top].min() - margin)))


@dataclass
class Identity:
    target: int | None      # the cube (stack) the cluster's top face mostly shows, if any
    counts: dict[int, int]  # cluster pixels per stack
    merged: bool            # the cluster holds more than one stack


def identify(est: CubeEstimate, seg_geom: np.ndarray, stack_of_geom: dict[int, int],
             min_pixels: int = 15) -> Identity:
    """Evaluation only: which stack a cluster is, from per-pixel geom ids of a segmentation
    render (`stack_of_geom` maps each cube geom to its stack; other geoms are ignored). The
    target is the stack with most top-face pixels; the cluster is merged if a second stack has
    at least `min_pixels` pixels in it."""
    flat = seg_geom.ravel()

    def tally(pix):
        c: dict[int, int] = {}
        for g, m in zip(*np.unique(flat[pix], return_counts=True), strict=True):
            s = stack_of_geom.get(int(g))
            if s is not None:
                c[s] = c.get(s, 0) + int(m)
        return c

    top, whole = tally(est.top_pixels), tally(est.pixels)
    target = max(top, key=top.get) if top else None
    merged = sum(m >= min_pixels for m in whole.values()) > 1
    return Identity(target, whole, merged)
