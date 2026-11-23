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
    """An industrial linear axis, drawn (none of it collides; the physics is the slide joint):
    a low aluminium bed with T-slots, two hardened guide rails, a toothed rack, a ribbed cable
    carrier and blue end caps with rubber bumpers on the bench; on the carriage (moving with the
    base), a blue mounting plate, end blocks with bolt heads, and a servo motor with its gearbox
    driving a pinion in the rack. The robot's base stays at bench height (every trained skill
    assumes it), so the bulk sits beside the robot, not under it."""
    yaw = float(np.arctan2(axis[1], axis[0]))
    quat = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    box, cyl = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER
    blue, black, steel = [0.13, 0.33, 0.72, 1.0], [0.08, 0.08, 0.09, 1.0], [0.72, 0.74, 0.78, 1.0]
    alu, dark = [0.62, 0.64, 0.67, 1.0], [0.25, 0.26, 0.28, 1.0]
    span = (cfg.low - 0.15, cfg.high + 0.15)
    centre, half = (span[0] + span[1]) / 2, (span[1] - span[0]) / 2
    rail = spec.worldbody.add_body(name="rail", pos=list(axis * centre), quat=quat)

    def fixed(size, pos, rgba, gtype=box, q=None):
        rail.add_geom(type=gtype, size=size, pos=pos, rgba=rgba, contype=0, conaffinity=0,
                      **({"quat": q} if q else {}))

    fixed([half, 0.13, 0.0015], [0.0, 0.0, 0.0015], alu)                       # bed
    for y in (-0.105, -0.035, 0.035, 0.105):                                   # T-slots
        fixed([half, 0.004, 0.0002], [0.0, y, 0.0031], dark)
    for side in (-1, 1):                                                       # guide rails
        fixed([half, 0.011, 0.0035], [0.0, side * 0.075, 0.0035], steel)
        fixed([half, 0.004, 0.0004], [0.0, side * 0.075, 0.0072], [0.85, 0.87, 0.9, 1.0])
    fixed([half, 0.008, 0.003], [0.0, -0.118, 0.003], dark)                    # rack
    n_teeth = int(2 * half / 0.012)
    for k in range(n_teeth):
        fixed([0.0025, 0.008, 0.0022], [-half + 0.006 + 0.012 * k, -0.118, 0.0082], black)
    # cable carrier, outside the motor's side
    fixed([half, 0.026, 0.017], [0.0, -0.245, 0.017], [0.12, 0.12, 0.13, 1.0])
    for k in range(int(2 * half / 0.03)):
        fixed([0.004, 0.027, 0.018], [-half + 0.015 + 0.03 * k, -0.245, 0.018], black)
    fixed([half, 0.0025, 0.0012], [0.0, -0.272, 0.034], [0.95, 0.75, 0.1, 1.0])
    for end, sign in ((-half, -1), (half, 1)):                                 # end caps
        fixed([0.02, 0.15, 0.028], [end + sign * 0.02, 0.0, 0.028], blue)
        fixed([0.006, 0.05, 0.016], [end - sign * 0.004, 0.0, 0.024], black)    # bumper
        for y in (-0.1, 0.1):
            fixed([0.006, 0.004], [end + sign * 0.02, y, 0.058], black, cyl)

    # on the carriage: geoms of the base body, given in the rail's frame
    base = spec.body(BASE_BODY)

    def moving(size, pos, rgba, gtype=box, q=None):
        base.add_geom(type=gtype, size=size, pos=list(rot @ np.asarray(pos, float)),
                      quat=q if q is not None else quat, rgba=rgba, contype=0, conaffinity=0,
                      mass=0.0)

    moving([0.13, 0.12, 0.006], [0.0, 0.0, 0.006], blue)                       # mounting plate
    for x in (-0.135, 0.135):                                                  # end blocks
        moving([0.015, 0.12, 0.016], [x, 0.0, 0.016], blue)
        for y in (-0.09, -0.03, 0.03, 0.09):
            moving([0.0045, 0.003], [x, y, 0.035], black, cyl)
    for x in (-0.09, 0.09):                                                    # bearing blocks
        for y in (-0.075, 0.075):
            moving([0.03, 0.02, 0.009], [x, y, 0.009], [0.2, 0.22, 0.25, 1.0])
    # servo motor and gearbox on the rack side, the motor lying along the rail
    turn_x = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])      # z onto x
    q_motor = np.zeros(4)
    mujoco.mju_mulQuat(q_motor, np.asarray(quat, float), turn_x)
    moving([0.04, 0.032, 0.032], [0.06, -0.165, 0.034], black)                 # gearbox
    moving([0.012, 0.01], [0.06, -0.165, 0.004], steel, cyl)                   # pinion housing
    moving([0.032, 0.07], [-0.05, -0.165, 0.036], [0.15, 0.36, 0.78, 1.0], cyl, q_motor)  # motor
    moving([0.0325, 0.012], [-0.128, -0.165, 0.036], black, cyl, q_motor)      # rear cap
    moving([0.0326, 0.004], [0.015, -0.165, 0.036], [0.85, 0.87, 0.9, 1.0], cyl, q_motor)
    moving([0.012, 0.01, 0.009], [-0.07, -0.165, 0.073], black)                # connector
    moving([0.006, 0.03], [-0.07, -0.205, 0.075], black, mujoco.mjtGeom.mjGEOM_CAPSULE,
           [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])                   # cable
    moving([0.02, 0.0005, 0.008], [-0.05, -0.198, 0.036], [0.95, 0.75, 0.1, 1.0])  # label


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
