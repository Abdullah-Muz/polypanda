"""v4: the robot on a linear rail (a "7th axis"), so one robot serves a line of stations.

The Panda's base body (`link0`) gets a slide joint along a horizontal world direction, driven by
a stiff position servo, as a carriage on a floor rail would be. Everything above the base is
unchanged. Each station of a line sits at the cell's standard pose in front of a rail stop
(`WorkcellConfig.radius` from the base axis, at table height), so a skill trained in front of the
fixed robot works at any stop: the controllers measure station angles about the base axis
(`ScriptedCell.angle_of`), solve IK in the base's frame (`ScriptedCell._solve`) and take world
points relative to the base in the station frame (`StationView(origin=...)`).

Two simplifications, disclosed: the base stays at table height (a real rail-mounted arm usually
stands on a carriage beside the benches), and the rail is driven by a position servo that the
supervisor commands; no policy moves it.

Without a rail (every v1-v3.2 configuration) the base stays welded to the world at the origin,
and nothing here is used.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

BASE_BODY = "link0"
RAIL_JOINT = "rail"
RAIL_ACTUATOR = "rail"


@dataclass(frozen=True)
class RailConfig:
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0)  # world direction of travel (horizontal)
    low: float = -1.0                 # m, travel range along `axis` from the base's start
    high: float = 1.0
    kp: float = 30000.0               # N/m, position servo on the carriage
    kv: float = 3000.0                # N s/m, servo damping
    armature: float = 5.0             # kg, carriage and drive inertia seen by the joint
    max_force: float = 2000.0         # N
    speed: float = 0.5                # m/s, average speed of a scripted move between stops
    visual: bool = True               # draw the rail and the carriage (never collide)


def add_rail(spec: mujoco.MjSpec, cfg: RailConfig | None = None) -> RailConfig:
    """Mount the robot's base on a rail: a slide joint on `link0`, its servo, and (visual only)
    the rail and a carriage plate. Keyframes get the rail's position (0) in front, since the
    new joint comes first in qpos. Returns the config used."""
    cfg = cfg or RailConfig()
    axis = np.asarray(cfg.axis, dtype=float)
    if abs(axis[2]) > 1e-9 or not np.isclose(np.linalg.norm(axis), 1.0):
        raise ValueError("the rail axis must be a horizontal unit vector")
    base = spec.body(BASE_BODY)
    base.add_joint(name=RAIL_JOINT, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=list(axis),
                   range=[cfg.low, cfg.high], limited=mujoco.mjtLimited.mjLIMITED_TRUE,
                   armature=cfg.armature)
    act = spec.add_actuator(name=RAIL_ACTUATOR, target=RAIL_JOINT,
                            trntype=mujoco.mjtTrn.mjTRN_JOINT)
    act.gaintype, act.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
    act.gainprm = [cfg.kp] + [0.0] * 9                 # force = kp (ctrl - q) - kv qdot
    act.biasprm = [0.0, -cfg.kp, -cfg.kv] + [0.0] * 7
    act.ctrlrange = [cfg.low, cfg.high]
    act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.forcerange = [-cfg.max_force, cfg.max_force]
    act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    # The base no longer moves with the world, so its contacts with the world (the bench) would
    # count and drag on it: exclude them, as they were while it was welded.
    spec.add_exclude(bodyname1="world", bodyname2=BASE_BODY)
    for key in spec.keys:   # the rail's coordinate comes first in qpos (ctrl is padded at the
        key.qpos = [0.0, *key.qpos]   # end on compile, where the rail's actuator is)
    if cfg.visual:
        _draw(spec, cfg, axis)
    return cfg


def _draw(spec: mujoco.MjSpec, cfg: RailConfig, axis: np.ndarray) -> None:
    yaw = float(np.arctan2(axis[1], axis[0]))
    quat = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    span = (cfg.low - 0.2, cfg.high + 0.2)
    centre, half = (span[0] + span[1]) / 2, (span[1] - span[0]) / 2
    rail = spec.worldbody.add_body(name="rail", pos=list(axis * centre), quat=quat)
    for side in (-1, 1):            # two rails set into the bench, under the carriage
        rail.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[half, 0.012, 0.0015],
                      pos=[0.0, side * 0.075, 0.0015], rgba=[0.55, 0.57, 0.6, 1.0],
                      contype=0, conaffinity=0)
    for end in span:                # end stops
        rail.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.1, 0.012],
                      pos=[end - centre, 0.0, 0.012], rgba=[0.95, 0.75, 0.1, 1.0],
                      contype=0, conaffinity=0)
    spec.body(BASE_BODY).add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.13, 0.11, 0.004],
                                  pos=[0.0, 0.0, 0.004], quat=quat, rgba=[0.15, 0.35, 0.75, 1.0],
                                  contype=0, conaffinity=0, mass=0.0)   # carriage plate


@dataclass(frozen=True)
class RailIndices:
    joint_qpos: int
    joint_dof: int
    actuator: int
    axis: np.ndarray

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> "RailIndices | None":
        """The rail's indices, or None when the model has no rail."""
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, RAIL_JOINT) < 0:
            return None
        j = model.joint(RAIL_JOINT)
        return cls(joint_qpos=int(j.qposadr[0]), joint_dof=int(j.dofadr[0]),
                   actuator=model.actuator(RAIL_ACTUATOR).id, axis=model.jnt_axis[j.id].copy())
