"""Milestone 9 workcell: a work-order stacking cell around the Panda.

Stations sit on one circle around the robot base, so moving between any two of them is a pure
rotation of the base (joint 1), and every cycle swings the base widely:

    infeed pick point     +90 deg   (left)        an indexing conveyor brings one cube at a time
    towers A / B / C   -40 / -90 / -140 deg       each with a recipe sign behind it
    reject bin           +150 deg   (rear-left)   for cubes that cannot be used
    buffer slots 1-3  +10 / +35 / +60 deg (front-left)  cubes needed later (Layer 2)

Work orders: each tower has a recipe (colours from bottom to top), shown on its sign as a column
of coloured squares. Cubes come in four colours. Cubes wait hidden in a parts hopper next to the
belt and are fed onto the belt, a batch at a time, inside a covered tunnel; the belt indexes the
leading cube to the pick point.

Every work surface is at the height of the existing table (the ground plane, z = 0; the belt top
is 2 mm above it), so skills trained in front of the robot can transfer to any station by a
rotation about the base axis. Visual dressing (the factory hall of
polypanda.simulation.factory, stack light, signs, hopper, tunnel) has no collision.

The gripper squeezes harder than Menagerie's (~20 N instead of ~2 N; `finger_stiffness`): with
the original grip a cube slides out of the fingers during the long base swings.

Conveyor: a slatted carrier plate on a slide joint along the belt, driven by a velocity servo.
Cubes ride on it by friction; the carrier collides only with cubes (not the table, not the
robot). `Conveyor.index()` advances it until the leading cube reaches the pick point. Once the
belt is empty it rewinds (hidden in the tunnel, nothing on it) and loads the next batch.
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

from polypanda.simulation.assets import panda_spec
from polypanda.simulation.factory import add_control_cabinet, add_factory
from polypanda.simulation.panda import PandaIndices, reset_to_home
from polypanda.simulation.scenes import CubeIndices, CubeSpec, add_cube

CUBE_NAME = "cube{}"
COLOURS = {"red": (0.85, 0.12, 0.1, 1.0), "green": (0.1, 0.68, 0.25, 1.0),
           "blue": (0.12, 0.35, 0.9, 1.0), "yellow": (0.95, 0.8, 0.08, 1.0)}
TOWERS = "ABC"
BELT_CONTACT = 2  # contype/conaffinity bit shared only by the carrier and the cubes
_SIGN_OFF = (0.22, 0.23, 0.25, 1.0)


@dataclass(frozen=True)
class WorkcellConfig:
    radius: float = 0.5               # m, base axis to every station (centre of the trained region)
    infeed_angle: float = 90.0        # deg, about the base axis, from the robot's front (+x)
    tower_angles: tuple[float, float, float] = (-40.0, -90.0, -140.0)
    reject_angle: float = 150.0
    buffer_angles: tuple[float, ...] = (10.0, 35.0, 60.0)  # parking slots on the bench (Layer 2)
    n_cubes: int = 12                 # cube bodies in the model (3 towers x up to 4; Layer 2: 14)
    max_height: int = 6               # squares on each recipe sign
    batch: int = 3                    # cubes loaded onto the belt at a time
    cube_spacing: float = 0.12        # m between cubes queued on the belt
    first_cube: float = 0.50          # m upstream of the pick point when loaded
    belt_width: float = 0.12
    belt_lift: float = 0.002          # belt top above the table, so cubes ride on the belt only
    belt_speed: float = 0.15          # m/s top speed carrying cubes
    rewind_speed: float = 0.4         # m/s running back empty
    belt_accel_gain: float = 3.0      # 1/s: speed command = gain x remaining distance, capped
    belt_kv: float = 400.0            # velocity servo gain on the carrier
    sign_radius: float = 0.82         # m, signs stand behind their towers
    bin_size: float = 0.18            # m, outer side of the reject bin
    bin_height: float = 0.08
    dressing: bool = True             # the factory hall (visual only)
    # Finger servo stiffness (N/m). Menagerie's 100 N/m squeezes a 4 cm cube with ~2 N, too weak
    # to hold it through a 200 deg base swing (it slides out; measured). 1000 N/m gives ~20 N;
    # a real Franka Hand grasps with up to 70 N. User decision for Milestone 9 on.
    finger_stiffness: float = 1000.0
    # Cube contacts with torsional friction (condim 4; the cube's coefficient, 0.03 m as in
    # Menagerie, is otherwise unused): without it nothing resists the cube turning about the
    # finger axis, and over a long carry it slowly twists out of the grip (measured).
    torsional_friction: bool = True
    # Grasp assist (a disclosed simulation simplification): once a grasp is verified (both
    # fingers pressing on the cube), the cube is welded to the hand until just before the fingers
    # open. In this model a firmly held cube slides along the finger pads whenever the arm moves
    # (17 mm per lift, out of the hand within a few long swings), whatever the grip force, swing
    # speed, grip direction, friction model, contact stiffness, solver settings, finger damping
    # or pad size (all measured, Milestone 9). Grasping and releasing stay fully physical.
    grasp_assist: bool = True


def station_xy(angle_deg: float, radius: float) -> np.ndarray:
    a = np.radians(angle_deg)
    return np.array([radius * np.cos(a), radius * np.sin(a)])


def belt_direction(cfg: WorkcellConfig) -> np.ndarray:
    """Unit vector of belt travel (tangential to the station circle at the pick point)."""
    heading = np.radians(cfg.infeed_angle) + np.pi / 2
    return np.array([np.cos(heading), np.sin(heading)])


def queue_xy(cfg: WorkcellConfig) -> list[np.ndarray]:
    """Where a loaded batch sits on the belt (inside the tunnel), leading cube first."""
    pick, travel = station_xy(cfg.infeed_angle, cfg.radius), belt_direction(cfg)
    return [pick - (cfg.first_cube + k * cfg.cube_spacing) * travel for k in range(cfg.batch)]


def hopper_xy(cfg: WorkcellConfig) -> list[np.ndarray]:
    """Parking spots inside the parts hopper beside the belt's upstream end, one per cube."""
    pick, travel = station_xy(cfg.infeed_angle, cfg.radius), belt_direction(cfg)
    outward = np.array([-travel[1], travel[0]])
    if outward @ pick < 0:  # the side of the belt away from the robot
        outward = -outward
    return [pick - (0.5 + 0.075 * (k % 6)) * travel + (0.15 + 0.07 * (k // 6)) * outward
            for k in range(cfg.n_cubes)]


def _visual(body, **kwargs):
    """A geom that is drawn but never collides."""
    return body.add_geom(contype=0, conaffinity=0, **kwargs)


def _yaw_quat(angle: float) -> list[float]:
    return [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]


def _add_conveyor(spec: mujoco.MjSpec, cfg: WorkcellConfig) -> None:
    pick = station_xy(cfg.infeed_angle, cfg.radius)
    heading = np.radians(cfg.infeed_angle) + np.pi / 2       # travel direction: tangential
    total = cfg.first_cube + (cfg.batch - 1) * cfg.cube_spacing
    # Carrier frame: origin at the pick point, x along the travel direction.
    frame = spec.worldbody.add_body(name="conveyor", pos=[*pick, 0.0], quat=_yaw_quat(heading))
    upstream = total + cfg.cube_spacing + 0.1                 # carrier reaches behind the queue
    length = upstream + 0.08
    carrier = frame.add_body(name="belt_carrier", pos=[0.0, 0.0, 0.0])
    carrier.add_joint(name="belt", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0],
                      range=[-0.01, total + 0.02], damping=5.0, armature=0.1)
    half = [length / 2, cfg.belt_width / 2, 0.004]
    carrier.add_geom(name="belt", type=mujoco.mjtGeom.mjGEOM_BOX, size=half,
                     pos=[0.08 - length / 2, 0.0, cfg.belt_lift - half[2]],
                     contype=BELT_CONTACT, conaffinity=BELT_CONTACT, friction=[1.0, 0.03, 0.003],
                     rgba=[0.12, 0.12, 0.13, 1.0], mass=2.0)
    for k in range(int(length / 0.06)):  # light slats so the belt's motion is visible
        _visual(carrier, type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.006, cfg.belt_width / 2 - 0.004, 0.0004],
                pos=[0.08 - 0.03 - k * 0.06, 0.0, cfg.belt_lift + 0.0002],
                rgba=[0.35, 0.35, 0.37, 1.0])
    belt = spec.add_actuator(name="belt", target="belt", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    belt.gaintype, belt.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
    belt.gainprm = [cfg.belt_kv] + [0.0] * 9                  # velocity servo:
    belt.biasprm = [0.0, 0.0, -cfg.belt_kv] + [0.0] * 7       # force = kv (ctrl - qvel)
    belt.ctrlrange = [-0.5, 0.5]
    belt.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    # Static housing: frame, side rails (outside the fingers' reach), rollers, pick zone.
    span = (-(upstream + 0.02), total + 0.12)  # in the frame: upstream end, downstream end
    centre, half_len = (span[0] + span[1]) / 2, (span[1] - span[0]) / 2
    _visual(frame, type=mujoco.mjtGeom.mjGEOM_BOX, size=[half_len, cfg.belt_width / 2 + 0.03, 0.02],
            pos=[centre, 0.0, -0.0205], rgba=[0.25, 0.27, 0.3, 1.0])
    for side in (-1, 1):
        _visual(frame, type=mujoco.mjtGeom.mjGEOM_BOX, size=[half_len, 0.006, 0.012],
                pos=[centre, side * (cfg.belt_width / 2 + 0.035), 0.012],
                rgba=[0.95, 0.75, 0.1, 1.0])
    for end in span:
        _visual(frame, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.018, cfg.belt_width / 2 + 0.02],
                pos=[end, 0.0, -0.016], quat=[np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0],
                rgba=[0.6, 0.6, 0.62, 1.0])
    _visual(frame, type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, cfg.belt_width / 2, 0.0006],
            pos=[0.0, 0.0, cfg.belt_lift + 0.0006], rgba=[0.2, 0.8, 0.3, 0.35])  # pick zone
    # Tunnel over the upstream end, where batches are loaded out of sight; the cubes emerge
    # from its mouth one by one.
    mouth = cfg.first_cube - 0.08
    t_half = (-span[0] - mouth) / 2
    _visual(frame, name="tunnel", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[t_half, cfg.belt_width / 2 + 0.045, 0.06],
            pos=[-(mouth + t_half), 0.0, 0.06], rgba=[0.8, 0.81, 0.83, 1.0])
    _visual(frame, type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.004, cfg.belt_width / 2 + 0.05, 0.065],
            pos=[-mouth + 0.002, 0.0, 0.065], rgba=[0.95, 0.75, 0.1, 1.0])  # mouth frame


def _add_hopper(spec: mujoco.MjSpec, cfg: WorkcellConfig) -> None:
    """Parts hopper beside the tunnel: a closed cabinet the spare cubes wait in."""
    spots = np.array(hopper_xy(cfg))
    lo, hi = spots.min(axis=0) - 0.05, spots.max(axis=0) + 0.05
    centre, half = (lo + hi) / 2, (hi - lo) / 2
    _visual(spec.worldbody, name="hopper", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[half[0], half[1], 0.14], pos=[*centre, 0.14], rgba=[0.8, 0.81, 0.83, 1.0])
    _visual(spec.worldbody, type=mujoco.mjtGeom.mjGEOM_BOX, size=[half[0] + 0.01, half[1] + 0.01,
            0.008], pos=[*centre, 0.288], rgba=[0.95, 0.75, 0.1, 1.0])


def _add_tower(spec: mujoco.MjSpec, cfg: WorkcellConfig, t: int) -> None:
    angle = cfg.tower_angles[t]
    xy = station_xy(angle, cfg.radius)
    body = spec.worldbody.add_body(name=f"tower{TOWERS[t]}", pos=[*xy, 0.0],
                                   quat=_yaw_quat(np.radians(angle)))
    _visual(body, type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.06, 0.06, 0.0005], pos=[0, 0, 0.0005],
            rgba=[0.15, 0.15, 0.17, 1.0])
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):  # locating outline, 5 cm square
        size = [0.026, 0.002, 0.0008] if sx == 0 else [0.002, 0.026, 0.0008]
        _visual(body, type=mujoco.mjtGeom.mjGEOM_BOX, size=size,
                pos=[0.026 * sx, 0.026 * sy, 0.0011], rgba=[0.95, 0.75, 0.1, 1.0])
    body.add_site(name=f"tower{TOWERS[t]}_base", size=[0.005, 0.005, 0.005], group=4)
    # Recipe sign behind the tower, facing the robot: a column of squares, bottom to top.
    sign = spec.worldbody.add_body(name=f"sign{TOWERS[t]}",
                                   pos=[*station_xy(angle, cfg.sign_radius), 0.0],
                                   quat=_yaw_quat(np.radians(angle)))
    top = 0.12 + 0.072 * cfg.max_height
    _visual(sign, type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.012, 0.012, 0.06], pos=[0.0, 0, 0.06],
            rgba=[0.6, 0.6, 0.62, 1.0])                                               # post
    _visual(sign, type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.006, 0.055, (top - 0.1) / 2 + 0.02],
            pos=[0, 0, (top + 0.1) / 2], rgba=[0.12, 0.13, 0.15, 1.0])                # panel
    _visual(sign, type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.007, 0.058, 0.012],
            pos=[0, 0, top + 0.03], rgba=[0.95, 0.76, 0.08, 1.0])                     # header
    for k in range(cfg.max_height):
        _visual(sign, name=f"sign{TOWERS[t]}_{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.003, 0.032, 0.032], pos=[-0.008, 0, 0.15 + 0.072 * k], rgba=list(_SIGN_OFF))


def _add_buffer(spec: mujoco.MjSpec, cfg: WorkcellConfig) -> None:
    """Buffer slots: painted squares on the bench (visual only; cubes stand on the table)."""
    for s, angle in enumerate(cfg.buffer_angles):
        body = spec.worldbody.add_body(name=f"buffer{s}", pos=[*station_xy(angle, cfg.radius), 0],
                                       quat=_yaw_quat(np.radians(angle)))
        _visual(body, type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.045, 0.045, 0.0005],
                pos=[0, 0, 0.0005], rgba=[0.15, 0.15, 0.17, 1.0])
        for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            size = [0.026, 0.002, 0.0008] if sx == 0 else [0.002, 0.026, 0.0008]
            _visual(body, type=mujoco.mjtGeom.mjGEOM_BOX, size=size,
                    pos=[0.026 * sx, 0.026 * sy, 0.0011], rgba=[0.85, 0.87, 0.9, 1.0])


def _add_reject_bin(spec: mujoco.MjSpec, cfg: WorkcellConfig) -> None:
    xy = station_xy(cfg.reject_angle, cfg.radius)
    body = spec.worldbody.add_body(name="reject_bin", pos=[*xy, 0.0],
                                   quat=_yaw_quat(np.radians(cfg.reject_angle)))
    s, h, t = cfg.bin_size / 2, cfg.bin_height / 2, 0.006
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        size = [s, t, h] if sx == 0 else [t, s, h]
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=size,
                      pos=[(s - t) * sx, (s - t) * sy, h], rgba=[0.32, 0.33, 0.36, 1.0])
    _visual(body, type=mujoco.mjtGeom.mjGEOM_BOX, size=[s - t, s - t, 0.0005], pos=[0, 0, 0.0005],
            rgba=[0.15, 0.15, 0.17, 1.0])
    body.add_site(name="reject_bin", size=[0.005, 0.005, 0.005], pos=[0, 0, 0.05], group=4)


STACK_LIGHT = ("green", "amber", "red")
_LAMP_ON = {"green": (0.1, 0.95, 0.2, 1.0), "amber": (1.0, 0.65, 0.0, 1.0),
            "red": (1.0, 0.1, 0.08, 1.0)}
_LAMP_OFF = {k: tuple(0.25 * c for c in v[:3]) + (1.0,) for k, v in _LAMP_ON.items()}


def _add_stack_light(spec: mujoco.MjSpec, pos) -> None:
    body = spec.worldbody.add_body(name="stack_light", pos=list(pos))
    _visual(body, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.04, 0.01], pos=[0, 0, 0.01],
            rgba=[0.2, 0.2, 0.22, 1.0])
    _visual(body, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.012, 0.25], pos=[0, 0, 0.26],
            rgba=[0.6, 0.6, 0.62, 1.0])
    for k, name in enumerate(STACK_LIGHT):
        _visual(body, name=f"lamp_{name}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[0.03, 0.03], pos=[0, 0, 0.55 + 0.062 * k], rgba=list(_LAMP_OFF[name]))
    _visual(body, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.032, 0.006],
            pos=[0, 0, 0.55 + 0.062 * 3 - 0.025], rgba=[0.15, 0.15, 0.17, 1.0])


MENAGERIE_FINGER_STIFFNESS = 100.0  # N/m: actuator8 biasprm[1] = -100; gainprm[0] = 0.04 x 100/255


def set_finger_stiffness(spec: mujoco.MjSpec, stiffness: float) -> None:
    """Scale the gripper's position servo: grip force on a cube of width w is about
    stiffness x w / 2, while ctrl 255 still opens the fingers fully (0.04 m each)."""
    act = spec.actuator("actuator8")
    scale = stiffness / MENAGERIE_FINGER_STIFFNESS
    gain, bias = list(act.gainprm), list(act.biasprm)
    gain[0] *= scale
    bias[1] *= scale
    act.gainprm, act.biasprm = gain, bias


def build_workcell_spec(cfg: WorkcellConfig | None = None, cube: CubeSpec | None = None
                        ) -> mujoco.MjSpec:
    cfg = cfg or WorkcellConfig()
    spec = panda_spec()
    set_finger_stiffness(spec, cfg.finger_stiffness)
    # Restyle the Menagerie floor as the bench top (its collision plane stays infinite).
    floor = spec.geom("floor")
    floor.size = [0.925, 0.95, 0.05]
    floor.pos = [0.075, 0.0, 0.0]
    spec.add_texture(name="bench", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, mark=mujoco.mjtMark.mjMARK_EDGE,
                     rgb1=[0.62, 0.64, 0.67], rgb2=[0.58, 0.6, 0.63], markrgb=[0.5, 0.52, 0.55],
                     width=300, height=300)
    spec.add_material(name="bench", textures=["", "bench"], texuniform=True, texrepeat=[8, 8],
                      reflectance=0.05)
    floor.material = "bench"
    if cfg.dressing:
        add_factory(spec)
        light_pos = add_control_cabinet(spec.worldbody, (2.25, -1.1))
    else:
        light_pos = (0.93, 0.0, 0.0)
    _add_conveyor(spec, cfg)
    _add_hopper(spec, cfg)
    for t in range(len(cfg.tower_angles)):
        _add_tower(spec, cfg, t)
    _add_reject_bin(spec, cfg)
    _add_stack_light(spec, light_pos)
    cube = cube or CubeSpec()
    for k, xy in enumerate(hopper_xy(cfg)):
        add_cube(spec, cube, xy=tuple(xy), name=CUBE_NAME.format(k), rgba=(0.5, 0.5, 0.5, 1.0))
        spec.geom(CUBE_NAME.format(k)).conaffinity = 1 | BELT_CONTACT  # rides on the belt
        if cfg.torsional_friction:
            spec.geom(CUBE_NAME.format(k)).condim = 4
        if cfg.grasp_assist:  # inactive until Workcell.attach()
            spec.add_equality(name=f"assist{k}", type=mujoco.mjtEq.mjEQ_WELD,
                              objtype=mujoco.mjtObj.mjOBJ_BODY, name1="hand",
                              name2=CUBE_NAME.format(k), active=False)
    _add_buffer(spec, cfg)  # added last: every earlier body and geom keeps its index (Layer 1)
    spec.visual.global_.azimuth = 200
    spec.visual.global_.elevation = -32
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080  # renders, video
    return spec


@dataclass(frozen=True)
class WorkOrder:
    """Recipes for the three towers (colours bottom to top) and the cubes' arrival order, as
    (colour, tower index) pairs. Layer 1: every arriving cube can go straight onto its tower.
    Layer 2 (supervisor.planner.mixed_order): random order with surplus cubes, tower None (the
    planner routes each cube)."""

    recipes: tuple[tuple[str, ...], ...]
    arrivals: tuple[tuple[str, int | None], ...]

    @classmethod
    def random(cls, rng: np.random.Generator, heights: tuple[int, int] = (1, 4),
               colours: tuple[str, ...] = tuple(COLOURS), towers: int = 3) -> "WorkOrder":
        recipes = tuple(tuple(rng.choice(colours, size=rng.integers(heights[0], heights[1] + 1)))
                        for _ in range(towers))
        recipes = tuple(tuple(str(c) for c in r) for r in recipes)
        remaining = [list(r) for r in recipes]
        arrivals = []
        while any(remaining):  # always-placeable order: the next cube some tower needs
            t = int(rng.choice([i for i, r in enumerate(remaining) if r]))
            arrivals.append((remaining[t].pop(0), t))
        return cls(recipes, tuple(arrivals))


@dataclass
class Workcell:
    """The compiled cell plus handles for its moving parts."""

    model: mujoco.MjModel
    data: mujoco.MjData
    config: WorkcellConfig
    cubes: list[CubeIndices] = field(default_factory=list)
    order: WorkOrder | None = None

    @classmethod
    def build(cls, config: WorkcellConfig | None = None, cube: CubeSpec | None = None
              ) -> "Workcell":
        config = config or WorkcellConfig()
        model = build_workcell_spec(config, cube).compile()
        cell = cls(model, mujoco.MjData(model), config)
        cell.cubes = [CubeIndices.from_model(model, CUBE_NAME.format(k))
                      for k in range(config.n_cubes)]
        cell.conveyor = Conveyor(cell)
        return cell

    @property
    def pick_xy(self) -> np.ndarray:
        return station_xy(self.config.infeed_angle, self.config.radius)

    def tower_xy(self, t: int) -> np.ndarray:
        return station_xy(self.config.tower_angles[t], self.config.radius)

    @property
    def reject_xy(self) -> np.ndarray:
        return station_xy(self.config.reject_angle, self.config.radius)

    def buffer_xy(self, slot: int) -> np.ndarray:
        return station_xy(self.config.buffer_angles[slot], self.config.radius)

    def reset(self, order: WorkOrder, rng: np.random.Generator | None = None,
              lateral: float = 0.0, yaw: float = 0.0) -> None:
        """Robot at home, belt empty at its start, every cube parked in the hopper; the cubes the
        order needs get its colours (cube k is the k-th arrival) and the signs show the recipes.
        Cubes are loaded onto the belt with random lateral offsets up to `lateral` (m) and yaw up
        to `yaw` (rad) either way."""
        cfg = self.config
        if len(order.arrivals) > cfg.n_cubes:
            raise ValueError(f"order needs {len(order.arrivals)} cubes, the cell has {cfg.n_cubes}")
        reset_to_home(self.model, self.data, PandaIndices.from_model(self.model))
        self.detach()
        self.data.qpos[self.model.joint("belt").qposadr[0]] = 0.0
        self.data.qvel[self.model.joint("belt").dofadr[0]] = 0.0
        self.data.ctrl[self.model.actuator("belt").id] = 0.0
        heading = float(np.arctan2(*belt_direction(cfg)[::-1]))
        for cube, spot in zip(self.cubes, hopper_xy(cfg), strict=True):
            self.data.qpos[cube.qpos:cube.qpos + 7] = [*spot, cube.half_size[2],
                                                       *_yaw_quat(heading)]
            self.data.qvel[cube.qvel:cube.qvel + 6] = 0.0
        for k, cube in enumerate(self.cubes):
            colour = order.arrivals[k][0] if k < len(order.arrivals) else None
            self.model.geom_rgba[cube.geom] = COLOURS[colour] if colour else (0.5, 0.5, 0.5, 1.0)
        for t, recipe in enumerate(order.recipes):
            for k in range(cfg.max_height):
                g = self.model.geom(f"sign{TOWERS[t]}_{k}").id
                self.model.geom_rgba[g] = COLOURS[recipe[k]] if k < len(recipe) else _SIGN_OFF
        self.order = order
        self.conveyor.reset(rng, lateral, yaw)
        mujoco.mj_forward(self.model, self.data)

    # --- grasp assist ---------------------------------------------------------------------
    def grasp_verified(self, cube: CubeIndices, min_force: float = 2.0) -> bool:
        """Both fingers press on the cube (normal force each above `min_force` N)."""
        m, d = self.model, self.data
        force = {m.body("left_finger").id: 0.0, m.body("right_finger").id: 0.0}
        f6 = np.zeros(6)
        for i in range(d.ncon):
            con = d.contact[i]
            if cube.geom in (con.geom1, con.geom2):
                other = m.geom_bodyid[con.geom2 if con.geom1 == cube.geom else con.geom1]
                if other in force:
                    mujoco.mj_contactForce(m, d, i, f6)
                    force[other] += f6[0]
        return all(f > min_force for f in force.values())

    def attach(self, cube: CubeIndices) -> bool:
        """Grasp assist: weld the cube to the hand where it is now, if the grasp is verified.
        Returns whether it was attached (False when assist is off or the grasp is not held)."""
        if not self.config.grasp_assist or not self.grasp_verified(cube):
            return False
        m, d = self.model, self.data
        eq = m.equality(f"assist{self.cubes.index(cube)}").id
        hand = m.body("hand").id
        R_hand = d.xmat[hand].reshape(3, 3)
        rel_pos = R_hand.T @ (d.xpos[cube.body] - d.xpos[hand])
        rel_quat, inv = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(inv, d.xquat[hand])
        mujoco.mju_mulQuat(rel_quat, inv, d.xquat[cube.body])
        m.eq_data[eq, :3] = 0.0           # anchor at the cube's centre
        m.eq_data[eq, 3:6] = rel_pos       # cube pose relative to the hand
        m.eq_data[eq, 6:10] = rel_quat
        m.eq_data[eq, 10] = 1.0            # torque scale
        d.eq_active[eq] = 1
        return True

    def detach(self, cube: CubeIndices | None = None) -> None:
        """Release the grasp assist (one cube, or all)."""
        if not self.config.grasp_assist:
            return
        for k, c in enumerate(self.cubes):
            if cube is None or c is cube:
                self.data.eq_active[self.model.equality(f"assist{k}").id] = 0

    def attached(self, cube: CubeIndices) -> bool:
        return bool(self.config.grasp_assist and self.data.eq_active[
            self.model.equality(f"assist{self.cubes.index(cube)}").id])

    def set_stack_light(self, *on: str) -> None:
        for name in STACK_LIGHT:
            self.model.geom_rgba[self.model.geom(f"lamp_{name}").id] = (
                _LAMP_ON[name] if name in on else _LAMP_OFF[name])

    def cube_colour(self, cube: CubeIndices) -> str | None:
        k = self.cubes.index(cube)
        return self.order.arrivals[k][0] if self.order and k < len(self.order.arrivals) else None


class Conveyor:
    """Belt indexing and batch feeding.

    `index()` asks for the next cube at the pick point: if cubes are on the belt, the belt runs
    until the leading one is there; if the belt is empty and cubes are still due, it rewinds
    (empty) and the next batch is loaded inside the tunnel, then indexed. `update()` runs the
    belt and does the loading; call it once per control step.
    """

    def __init__(self, cell: Workcell) -> None:
        self.cell = cell
        self.travel = belt_direction(cell.config)
        self.joint = cell.model.joint("belt")
        self.actuator = cell.model.actuator("belt").id
        self.target = 0.0
        self.next_arrival = 0
        self.loading = False
        self._rng, self._lateral, self._yaw = None, 0.0, 0.0

    def reset(self, rng=None, lateral: float = 0.0, yaw: float = 0.0) -> None:
        self.target, self.next_arrival, self.loading = 0.0, 0, False
        self._rng, self._lateral, self._yaw = rng, lateral, yaw
        self._load_batch()

    @property
    def due(self) -> int:
        """Cubes of the order not yet loaded onto the belt."""
        order = self.cell.order
        return (len(order.arrivals) if order else 0) - self.next_arrival

    def _load_batch(self) -> None:
        cell, cfg = self.cell, self.cell.config
        normal = np.array([-self.travel[1], self.travel[0]])
        heading = float(np.arctan2(self.travel[1], self.travel[0]))
        for start in queue_xy(cfg):
            if self.due <= 0:
                break
            cube = cell.cubes[self.next_arrival]
            rng = self._rng
            offset = rng.uniform(-self._lateral, self._lateral) if rng is not None else 0.0
            angle = heading + (rng.uniform(-self._yaw, self._yaw) if rng is not None else 0.0)
            cell.data.qpos[cube.qpos:cube.qpos + 7] = [
                *(start + offset * normal), cube.half_size[2] + cfg.belt_lift, *_yaw_quat(angle)]
            cell.data.qvel[cube.qvel:cube.qvel + 6] = 0.0
            self.next_arrival += 1
        mujoco.mj_forward(cell.model, cell.data)

    @property
    def position(self) -> float:
        return float(self.cell.data.qpos[self.joint.qposadr[0]])

    def upstream_distance(self, cube: CubeIndices) -> float:
        """How far before the pick point (along the belt) a cube is; < 0 once past it."""
        return float((self.cell.pick_xy - self.cell.data.xpos[cube.body][:2]) @ self.travel)

    def on_belt(self, cube: CubeIndices) -> bool:
        cfg, p = self.cell.config, self.cell.data.xpos[cube.body]
        lateral = abs(float((p[:2] - self.cell.pick_xy) @ np.array([-self.travel[1],
                                                                     self.travel[0]])))
        resting = abs(p[2] - cube.half_size[2] - cfg.belt_lift) < 0.004
        return resting and lateral < cfg.belt_width / 2 and self.upstream_distance(cube) > -0.05

    def lead(self) -> CubeIndices | None:
        queued = [c for c in self.cell.cubes if self.on_belt(c)]
        return min(queued, key=self.upstream_distance) if queued else None

    def index(self) -> bool:
        """Bring the next cube to the pick point (rewinding and loading a batch if the belt is
        empty). Returns False when the order has no cubes left."""
        lead = self.lead()
        if lead is not None:
            self.target = self.position + max(0.0, self.upstream_distance(lead))
            return True
        if self.due <= 0:
            return False
        self.target, self.loading = 0.0, True
        return True

    def update(self) -> None:
        cfg = self.cell.config
        if self.loading and abs(self.position) < 1e-3 and not self.moving:
            self.loading = False
            self._load_batch()
            self.index()
        remaining = self.target - self.position
        cap = cfg.belt_speed if remaining >= 0 else cfg.rewind_speed
        speed = float(np.clip(cfg.belt_accel_gain * remaining, -cap, cap))
        self.cell.data.ctrl[self.actuator] = speed if abs(remaining) > 1e-4 else 0.0

    @property
    def moving(self) -> bool:
        return abs(self.target - self.position) > 1e-3 or abs(
            self.cell.data.qvel[self.joint.dofadr[0]]) > 1e-3

    @property
    def ready(self) -> bool:
        """A cube is waiting at the pick point and the belt has stopped."""
        lead = self.lead()
        return (not self.loading and not self.moving and lead is not None
                and abs(self.upstream_distance(lead)) < 0.005)
