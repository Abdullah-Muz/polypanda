"""Task scenes built on the Panda model: currently a single cube for grasping.

The robot stands on the ground plane, which serves as the table surface (the usual table-mounted
arm setup, and the same layout as Menagerie's own single-cube pick scene, whose cube size,
friction and contact stiffness are used here). Menagerie files are never modified; everything is
added to an in-memory MjSpec.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from polypanda.simulation.assets import panda_spec

CUBE_BODY = "cube"
CUBE_JOINT = "cube_joint"
CUBE_GEOM = "cube"
CUBE_SITE = "cube_center"
DEFAULT_CUBE_XY = (0.5, 0.0)


@dataclass(frozen=True)
class CubeSpec:
    half_size: tuple[float, float, float] = (0.02, 0.02, 0.03)  # 4 x 4 x 6 cm, as in Menagerie
    friction: tuple[float, float, float] = (1.0, 0.03, 0.003)   # sliding, torsional, rolling
    solref: tuple[float, float] = (0.01, 1.0)                   # stiffer than default contact
    density: float = 1000.0                                     # kg/m^3 -> 0.096 kg


def add_cube(spec: mujoco.MjSpec, cube: CubeSpec | None = None,
             xy: tuple[float, float] = DEFAULT_CUBE_XY) -> None:
    cube = cube or CubeSpec()
    body = spec.worldbody.add_body(name=CUBE_BODY, pos=[*xy, cube.half_size[2]])
    body.add_freejoint(name=CUBE_JOINT)
    body.add_geom(name=CUBE_GEOM, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(cube.half_size),
                  friction=list(cube.friction), solref=list(cube.solref), density=cube.density,
                  rgba=[0.2, 0.7, 0.3, 1.0])
    body.add_site(name=CUBE_SITE, size=[0.005, 0.005, 0.005], group=4)


def load_pick_scene(cube: CubeSpec | None = None) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Panda (with TCP site) plus one free cube. The `home` keyframe gains the cube's rest pose."""
    spec = panda_spec()
    add_cube(spec, cube)
    model = spec.compile()
    return model, mujoco.MjData(model)


@dataclass(frozen=True)
class CubeIndices:
    body: int
    geom: int
    site: int
    qpos: int   # first of 7 free-joint qpos entries: x, y, z, qw, qx, qy, qz
    qvel: int   # first of 6 free-joint dofs
    half_size: np.ndarray

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> "CubeIndices":
        joint = model.joint(CUBE_JOINT)
        geom = model.geom(CUBE_GEOM)
        return cls(body=model.body(CUBE_BODY).id, geom=geom.id, site=model.site(CUBE_SITE).id,
                   qpos=int(joint.qposadr[0]), qvel=int(joint.dofadr[0]),
                   half_size=model.geom_size[geom.id].copy())


def set_cube_pose(model: mujoco.MjModel, data: mujoco.MjData, cube: CubeIndices,
                  xy: tuple[float, float], yaw: float = 0.0) -> None:
    """Place the cube upright on the surface at `xy` (m) rotated by `yaw` (rad), at rest."""
    quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    data.qpos[cube.qpos:cube.qpos + 7] = [*xy, cube.half_size[2], *quat]
    data.qvel[cube.qvel:cube.qvel + 6] = 0.0
    mujoco.mj_forward(model, data)


def cube_position(data: mujoco.MjData, cube: CubeIndices) -> np.ndarray:
    return data.xpos[cube.body].copy()


def cube_yaw(data: mujoco.MjData, cube: CubeIndices) -> float:
    """Rotation about world z of an upright cube (rad), from its body rotation matrix."""
    R = data.xmat[cube.body].reshape(3, 3)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def cube_tilt(data: mujoco.MjData, cube: CubeIndices) -> float:
    """Angle (rad) between the cube's z axis and world z; 0 when upright."""
    return float(np.arccos(np.clip(data.xmat[cube.body].reshape(3, 3)[2, 2], -1.0, 1.0)))
