"""Pinhole depth cameras for the workcell, and a simple depth-noise model.

Camera convention (MuJoCo / OpenGL): the camera looks along its own -z axis, image right is its
+x axis and image up its +y axis. A rendered depth image holds, per pixel, the distance along the
optical axis (z-depth, metres), which is what `Intrinsics.rays` assumes.

Three setups are used in docs/pose_error.md:

- `OVERHEAD`: fixed on a frame 1.5 m above the bench, straight above the robot base, looking
  down. 640 x 576 pixels, 65 deg vertical field of view (70.6 deg horizontal): the resolution
  and field of view of a time-of-flight camera's narrow mode (Azure Kinect NFOV unbinned,
  640 x 576, 75 x 65 deg). It sees the whole bench within the arm's reach (+-0.95 m by
  +-1.12 m at table height), about 3.3 mm per pixel at the table.
- `STATION_B`: a camera near the machine at tower B (-90 deg), 0.6 m from the station, tilted
  21 deg from vertical so the recipe sign behind the station does not block it. 640 x 480,
  50 deg vertical, about 1.2 mm per pixel at the station.
- `HOVER_B`: straight down from 0.55 m above tower B, 640 x 480, 58 deg vertical (about 0.7 mm
  per pixel at a cube's top). It stands in for a wrist camera with the hand paused above the
  station before it moves in (look, then move); it is evaluated with the arm at home only.

`DepthNoise`: per-pixel Gaussian depth noise whose standard deviation grows with the square of
the distance, plus random pixel dropout and dropout at depth edges (stereo and time-of-flight
sensors both lose pixels along object boundaries). Dropped pixels read 0 (invalid), as real
sensors report them. The presets are in NOISE; their sources are in their notes.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics in pixels; pixel (row v, column u) has its centre at u + 0.5, v + 0.5."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_fovy(cls, width: int, height: int, fovy_deg: float) -> Intrinsics:
        """MuJoCo's camera model: square pixels, principal point at the image centre, `fovy` the
        full vertical field of view."""
        f = 0.5 * height / np.tan(np.radians(fovy_deg) / 2)
        return cls(f, f, width / 2, height / 2, width, height)

    def rays(self) -> np.ndarray:
        """(H, W, 3) ray directions in the camera frame with z component -1, so the point seen
        at depth d is d * ray. Row 0 is the top of the image."""
        uu, vv = np.meshgrid(np.arange(self.width) + 0.5, np.arange(self.height) + 0.5)
        return np.stack([(uu - self.cx) / self.fx, -(vv - self.cy) / self.fy,
                         -np.ones_like(uu)], axis=-1)

    def pixel_size(self, distance: float) -> float:
        """Metres per pixel on a surface facing the camera at `distance`."""
        return distance / self.fy


@dataclass(frozen=True)
class CameraSetup:
    name: str
    pos: tuple[float, float, float]      # m, world frame
    target: tuple[float, float, float]   # m, the point on the optical axis it is aimed at
    fovy: float                          # deg, full vertical field of view
    width: int
    height: int
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)  # world direction that points up in the image
    note: str = ""

    @property
    def intrinsics(self) -> Intrinsics:
        return Intrinsics.from_fovy(self.width, self.height, self.fovy)

    def rotation(self) -> np.ndarray:
        """Camera-to-world rotation; columns are the camera's x (image right), y (image up) and
        z (pointing back, away from the scene) axes in the world frame."""
        forward = np.subtract(self.target, self.pos).astype(float)
        z = -forward / np.linalg.norm(forward)
        x = np.cross(self.up, z)
        x /= np.linalg.norm(x)
        return np.column_stack([x, np.cross(z, x), z])

    def quat(self) -> np.ndarray:
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.rotation().flatten())
        return q


OVERHEAD = CameraSetup(
    "overhead", pos=(0.0, 0.0, 1.5), target=(0.0, 0.0, 0.0), fovy=65.0, width=640, height=576,
    up=(0.0, 1.0, 0.0),
    note="fixed, 1.5 m above the bench over the robot base, looking down; ToF narrow-mode "
         "resolution and field of view (640 x 576, 65 deg vertical)")
STATION_B = CameraSetup(
    "station_b", pos=(0.0, -0.72, 0.62), target=(0.0, -0.5, 0.06), fovy=50.0, width=640,
    height=480,
    note="fixed near the machine at tower B, 0.6 m from the station, 21 deg from vertical")
HOVER_B = CameraSetup(
    "hover_b", pos=(0.0, -0.5, 0.55), target=(0.0, -0.5, 0.0), fovy=58.0, width=640, height=480,
    up=(0.0, 1.0, 0.0),
    note="straight down from 0.55 m above tower B: a wrist camera with the hand paused above "
         "the station (arm at home in the evaluation)")
SETUPS = {s.name: s for s in (OVERHEAD, STATION_B, HOVER_B)}


def add_camera(spec: mujoco.MjSpec, setup: CameraSetup) -> None:
    """Add a fixed camera to the world body of an (uncompiled) spec."""
    spec.worldbody.add_camera(name=setup.name, pos=list(setup.pos), quat=list(setup.quat()),
                              fovy=setup.fovy)
    glob = spec.visual.global_
    glob.offwidth = max(glob.offwidth, setup.width)
    glob.offheight = max(glob.offheight, setup.height)


def camera_pose(model: mujoco.MjModel, data: mujoco.MjData, name: str
                ) -> tuple[np.ndarray, np.ndarray]:
    """World position and camera-to-world rotation of a compiled camera (after mj_forward)."""
    cam = model.camera(name).id
    return data.cam_xpos[cam].copy(), data.cam_xmat[cam].reshape(3, 3).copy()


@dataclass(frozen=True)
class DepthNoise:
    """sigma(z) = sigma0 + k2 * (z - z0)^2 (metres, z the depth along the optical axis)."""

    name: str
    sigma0: float = 0.0
    k2: float = 0.0
    z0: float = 0.0
    dropout: float = 0.0        # probability that any pixel reads invalid
    edge_dropout: float = 0.0   # extra probability for pixels on a depth edge
    edge_jump: float = 0.02     # m: a pixel is on an edge if a 4-neighbour differs by more
    note: str = ""

    def sigma(self, z: np.ndarray | float) -> np.ndarray | float:
        return self.sigma0 + self.k2 * (np.asarray(z) - self.z0) ** 2

    @property
    def is_clean(self) -> bool:
        return self.sigma0 == 0 and self.k2 == 0 and self.dropout == 0 and self.edge_dropout == 0

    def apply(self, depth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """A noisy copy of a clean depth image; invalid pixels read 0."""
        depth = np.asarray(depth, dtype=float)
        out = depth + rng.standard_normal(depth.shape) * self.sigma(depth)
        drop = rng.random(depth.shape) < self.dropout
        if self.edge_dropout > 0:
            edge = np.zeros(depth.shape, dtype=bool)
            dv = np.abs(np.diff(depth, axis=0)) > self.edge_jump
            du = np.abs(np.diff(depth, axis=1)) > self.edge_jump
            edge[1:] |= dv
            edge[:-1] |= dv
            edge[:, 1:] |= du
            edge[:, :-1] |= du
            drop |= edge & (rng.random(depth.shape) < self.edge_dropout)
        out[drop | (depth <= 0)] = 0.0
        return out.astype(np.float32)


# D435 depth RMS (Intel's formula, Grunnet-Jepsen et al., "Tuning depth cameras for best
# performance"): RMS = z^2 * subpixel / (focal_px * baseline), with subpixel 0.08,
# baseline 50 mm, focal 447 px (848 px wide, 87 deg horizontal field of view).
_D435_K2 = 0.08 / (0.5 * 848 / np.tan(np.radians(87.0 / 2)) * 0.050)

NOISE = {
    "none": DepthNoise("none", note="the renderer's exact depth"),
    "low": DepthNoise(
        "low", k2=0.001, dropout=0.005, edge_dropout=0.1,
        note="assumed good sensor: sigma = 1 mm at 1 m (2.3 mm at 1.5 m), 0.5 % dropout, "
             "10 % of edge pixels dropped"),
    "medium": DepthNoise(
        "medium", sigma0=0.0012, k2=0.0019, z0=0.4, dropout=0.02, edge_dropout=0.3,
        note="Kinect v1 axial noise model (Nguyen, Izadi and Lovell 2012): sigma = 1.2 mm + "
             "1.9 mm (z - 0.4 m)^2, 3.5 mm at 1.5 m; 2 % dropout, 30 % of edge pixels dropped"),
    "high": DepthNoise(
        "high", k2=float(_D435_K2), dropout=0.05, edge_dropout=0.5,
        note="RealSense D435 depth RMS formula (Intel): sigma = 3.6 mm z^2, 8 mm at 1.5 m; "
             "5 % dropout, 50 % of edge pixels dropped"),
}
