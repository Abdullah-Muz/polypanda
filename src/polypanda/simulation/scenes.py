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
# Placement region where top-down grasps are reachable and collision-free (docs/grasp_task.md).
CUBE_X_RANGE = (0.35, 0.65)
CUBE_Y_RANGE = (-0.25, 0.25)
# The whole box where top-down grasps were verified feasible by IK (190 / 190 placements;
# docs/grasp_task.md): the wide region used from Milestone 7 on.
WIDE_X_RANGE = (0.30, 0.70)
WIDE_Y_RANGE = (-0.30, 0.30)


@dataclass(frozen=True)
class CubeSpec:
    half_size: tuple[float, float, float] = (0.02, 0.02, 0.03)  # 4 x 4 x 6 cm, as in Menagerie
    friction: tuple[float, float, float] = (1.0, 0.03, 0.003)   # sliding, torsional, rolling
    solref: tuple[float, float] = (0.01, 1.0)                   # stiffer than default contact
    density: float = 1000.0                                     # kg/m^3 -> 0.096 kg


DISTRACTOR = "distractor{}"
# Where unused distractor cubes wait: on the floor behind the robot, out of its reach.
DISTRACTOR_PARK = ((-0.8, -0.4), (-0.8, 0.4), (-0.8, 0.0), (-0.8, -0.8), (-0.8, 0.8))


def add_cube(spec: mujoco.MjSpec, cube: CubeSpec | None = None,
             xy: tuple[float, float] = DEFAULT_CUBE_XY, name: str = CUBE_BODY,
             rgba: tuple[float, ...] = (0.2, 0.7, 0.3, 1.0)) -> None:
    """A free cube: body and geom `name`, joint `<name>_joint`, centre site `<name>_center`."""
    cube = cube or CubeSpec()
    body = spec.worldbody.add_body(name=name, pos=[*xy, cube.half_size[2]])
    body.add_freejoint(name=f"{name}_joint")
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(cube.half_size),
                  friction=list(cube.friction), solref=list(cube.solref), density=cube.density,
                  rgba=list(rgba))
    body.add_site(name=f"{name}_center", size=[0.005, 0.005, 0.005], group=4)


def load_pick_scene(cube: CubeSpec | None = None, distractors: int = 0
                    ) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Panda (with TCP site) plus one free cube and, optionally, identical distractor cubes
    (blue) parked out of reach. The `home` keyframe gains every cube's rest pose."""
    spec = panda_spec()
    add_cube(spec, cube)
    for k in range(distractors):
        add_cube(spec, cube, DISTRACTOR_PARK[k], DISTRACTOR.format(k), (0.25, 0.4, 0.85, 1.0))
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
    def from_model(cls, model: mujoco.MjModel, name: str = CUBE_BODY) -> "CubeIndices":
        joint = model.joint(f"{name}_joint")
        geom = model.geom(name)
        return cls(body=model.body(name).id, geom=geom.id, site=model.site(f"{name}_center").id,
                   qpos=int(joint.qposadr[0]), qvel=int(joint.dofadr[0]),
                   half_size=model.geom_size[geom.id].copy())

    @classmethod
    def distractors(cls, model: mujoco.MjModel) -> list["CubeIndices"]:
        out, k = [], 0
        while mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, DISTRACTOR.format(k)) >= 0:
            out.append(cls.from_model(model, DISTRACTOR.format(k)))
            k += 1
        return out


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
    """Angle (rad) between the cube's z axis and vertical, either way up: 0 when upright or
    upside down (the cubes are symmetric; a knocked cube can land upside down), pi/2 when lying
    on a side. Up to v2 no cube was ever tilted beyond 90 deg, where both readings agree."""
    return float(np.arccos(min(1.0, abs(data.xmat[cube.body].reshape(3, 3)[2, 2]))))


def cube_up(data: mujoco.MjData, cube: CubeIndices) -> np.ndarray:
    """The cube's z axis, turned to point upward (see cube_tilt)."""
    up = data.xmat[cube.body].reshape(3, 3)[:, 2].copy()
    return up if up[2] >= 0 else -up
