"""Milestone 9 workcell: a stacking cell around the Panda.

Stations sit on a circle around the robot base, so every cycle swings the base widely:

    infeed pick point   +90 deg  (left)        an indexing conveyor brings one cube at a time
    tower fixture      -110 deg  (right-rear)  where the 3-cube tower is built
    reject bin         +150 deg  (rear-left)   for cubes that cannot be stacked

Every work surface is at the height of the existing table (the ground plane, z = 0; the belt top
is 2 mm above it), so skills trained in front of the robot can transfer to any station by a
rotation about the base axis. Visual dressing (bench, legs, factory floor, fence, floor
markings, stack light) has no collision.

The gripper squeezes harder than Menagerie's (~20 N instead of ~2 N; `finger_stiffness`): with
the original grip a cube slides out of the fingers during the long base swings.

Conveyor: a slatted carrier plate on a slide joint along the belt, driven by a velocity servo.
Cubes ride on it by friction. `Conveyor.index()` advances it until the leading cube on the belt
reaches the pick point, decelerating smoothly; the robot picks there and the belt indexes the
next cube. The carrier collides only with cubes (not the table, not the robot).
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

from polypanda.simulation.assets import panda_spec
from polypanda.simulation.panda import PandaIndices, reset_to_home
from polypanda.simulation.scenes import CubeIndices, CubeSpec, add_cube

CUBE_NAME = "cube{}"
CUBE_COLOURS = ((0.85, 0.35, 0.15, 1.0), (0.15, 0.55, 0.85, 1.0), (0.95, 0.75, 0.1, 1.0))
BELT_CONTACT = 2  # contype/conaffinity bit shared only by the carrier and the cubes


@dataclass(frozen=True)
class WorkcellConfig:
    radius: float = 0.5               # m, base axis to every station (centre of the trained region)
    infeed_angle: float = 90.0        # deg, about the base axis, from the robot's front (+x)
    tower_angle: float = -110.0
    reject_angle: float = 150.0
    n_cubes: int = 3
    cube_spacing: float = 0.12        # m between cubes queued on the belt
    first_cube: float = 0.50          # m upstream of the pick point at the start
    belt_width: float = 0.12
    belt_lift: float = 0.002          # belt top above the table, so cubes ride on the belt only
    belt_speed: float = 0.15          # m/s top speed
    belt_accel_gain: float = 3.0      # 1/s: speed command = gain x remaining distance, capped
    belt_kv: float = 400.0            # velocity servo gain on the carrier
    bin_size: float = 0.18            # m, outer side of the reject bin
    bin_height: float = 0.08
    dressing: bool = True             # bench, legs, floor, fence, markings (visual only)
    # Finger servo stiffness (N/m). Menagerie's 100 N/m squeezes a 4 cm cube with ~2 N, too weak
    # to hold it through a 200 deg base swing (it slides out; measured). 1000 N/m gives ~20 N;
    # a real Franka Hand grasps with up to 70 N. User decision for Milestone 9 on.
    finger_stiffness: float = 1000.0


def station_xy(angle_deg: float, radius: float) -> np.ndarray:
    a = np.radians(angle_deg)
    return np.array([radius * np.cos(a), radius * np.sin(a)])


def belt_direction(cfg: WorkcellConfig) -> np.ndarray:
    """Unit vector of belt travel (tangential to the station circle at the pick point)."""
    heading = np.radians(cfg.infeed_angle) + np.pi / 2
    return np.array([np.cos(heading), np.sin(heading)])


def queue_xy(cfg: WorkcellConfig) -> list[np.ndarray]:
    """Start positions of the cubes queued upstream on the belt, leading cube first."""
    pick, travel = station_xy(cfg.infeed_angle, cfg.radius), belt_direction(cfg)
    return [pick - (cfg.first_cube + k * cfg.cube_spacing) * travel for k in range(cfg.n_cubes)]


def _visual(body, **kwargs):
    """A geom that is drawn but never collides."""
    return body.add_geom(contype=0, conaffinity=0, **kwargs)


def _yaw_quat(angle: float) -> list[float]:
    return [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]


def _add_conveyor(spec: mujoco.MjSpec, cfg: WorkcellConfig) -> None:
    pick = station_xy(cfg.infeed_angle, cfg.radius)
    heading = np.radians(cfg.infeed_angle) + np.pi / 2       # travel direction: tangential
    total = cfg.first_cube + (cfg.n_cubes - 1) * cfg.cube_spacing
    # Carrier frame: origin at the pick point, x along the travel direction.
    frame = spec.worldbody.add_body(name="conveyor", pos=[*pick, 0.0], quat=_yaw_quat(heading))
    upstream = total + cfg.cube_spacing + 0.1                 # carrier reaches behind the queue
    length = upstream + 0.08
    carrier = frame.add_body(name="belt_carrier", pos=[0.0, 0.0, 0.0])
    carrier.add_joint(name="belt", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0],
                      range=[0.0, total + 0.02], damping=5.0, armature=0.1)
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
    # Static housing: frame, side rails (outside the fingers' reach), rollers, end stop marker.
    span = (-(total + 0.25), total + 0.12)  # in the frame: upstream end, downstream end
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


def _add_tower_fixture(spec: mujoco.MjSpec, cfg: WorkcellConfig) -> None:
    xy = station_xy(cfg.tower_angle, cfg.radius)
    body = spec.worldbody.add_body(name="tower_fixture", pos=[*xy, 0.0],
                                   quat=_yaw_quat(np.radians(cfg.tower_angle)))
    _visual(body, type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.06, 0.06, 0.0005], pos=[0, 0, 0.0005],
            rgba=[0.15, 0.15, 0.17, 1.0])
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):  # locating outline, 5 cm square
        size = [0.026, 0.002, 0.0008] if sx == 0 else [0.002, 0.026, 0.0008]
        _visual(body, type=mujoco.mjtGeom.mjGEOM_BOX, size=size,
                pos=[0.026 * sx, 0.026 * sy, 0.0011], rgba=[0.95, 0.75, 0.1, 1.0])
    body.add_site(name="tower_base", size=[0.005, 0.005, 0.005], group=4)


def _add_reject_bin(spec: mujoco.MjSpec, cfg: WorkcellConfig) -> None:
    xy = station_xy(cfg.reject_angle, cfg.radius)
    body = spec.worldbody.add_body(name="reject_bin", pos=[*xy, 0.0],
                                   quat=_yaw_quat(np.radians(cfg.reject_angle)))
    s, h, t = cfg.bin_size / 2, cfg.bin_height / 2, 0.006
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        size = [s, t, h] if sx == 0 else [t, s, h]
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=size,
                      pos=[(s - t) * sx, (s - t) * sy, h], rgba=[0.75, 0.15, 0.12, 1.0])
    _visual(body, type=mujoco.mjtGeom.mjGEOM_BOX, size=[s - t, s - t, 0.0005], pos=[0, 0, 0.0005],
            rgba=[0.3, 0.08, 0.07, 1.0])
    body.add_site(name="reject_bin", size=[0.005, 0.005, 0.005], pos=[0, 0, 0.05], group=4)


STACK_LIGHT = ("green", "amber", "red")
_LAMP_ON = {"green": (0.1, 0.95, 0.2, 1.0), "amber": (1.0, 0.65, 0.0, 1.0),
            "red": (1.0, 0.1, 0.08, 1.0)}
_LAMP_OFF = {k: tuple(0.25 * c for c in v[:3]) + (1.0,) for k, v in _LAMP_ON.items()}


def _add_stack_light(spec: mujoco.MjSpec, xy) -> None:
    body = spec.worldbody.add_body(name="stack_light", pos=[*xy, 0.0])
    _visual(body, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.04, 0.01], pos=[0, 0, 0.01],
            rgba=[0.2, 0.2, 0.22, 1.0])
    _visual(body, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.012, 0.25], pos=[0, 0, 0.26],
            rgba=[0.6, 0.6, 0.62, 1.0])
    for k, name in enumerate(STACK_LIGHT):
        _visual(body, name=f"lamp_{name}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[0.03, 0.03], pos=[0, 0, 0.55 + 0.062 * k], rgba=list(_LAMP_OFF[name]))
    _visual(body, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.032, 0.006],
            pos=[0, 0, 0.55 + 0.062 * 3 - 0.025], rgba=[0.15, 0.15, 0.17, 1.0])


def _add_dressing(spec: mujoco.MjSpec) -> None:
    """Bench, legs, factory floor, fence, hazard markings: drawn only, never collide."""
    world = spec.worldbody
    bench_x, bench_y, top = (-0.85, 1.0), (-0.85, 0.85), -0.0006
    cx, cy = sum(bench_x) / 2, sum(bench_y) / 2
    hx, hy = (bench_x[1] - bench_x[0]) / 2, (bench_y[1] - bench_y[0]) / 2
    floor_z = -0.8
    _visual(world, type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, 0.03], pos=[cx, cy, top - 0.03],
            rgba=[0.55, 0.57, 0.6, 1.0])                                  # bench top slab
    for sx in (-1, 1):
        for sy in (-1, 1):
            _visual(world, type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.03, (-floor_z - 0.06) / 2],
                    pos=[cx + sx * (hx - 0.06), cy + sy * (hy - 0.06), (floor_z - 0.06) / 2],
                    rgba=[0.3, 0.32, 0.36, 1.0])
    _visual(world, name="factory_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4, 4, 0.05],
            pos=[0, 0, floor_z], material="factory_floor")
    # Hazard border on the floor around the bench, and the fence beyond it.
    r = 1.45
    for k, (pos, size) in enumerate((([cx, r, 0], [r + 0.05, 0.04, 0]),
                                     ([cx, -r, 0], [r + 0.05, 0.04, 0]),
                                     ([cx + r, 0, 0], [0.04, r, 0]),
                                     ([cx - r, 0, 0], [0.04, r, 0]))):
        _visual(world, name=f"hazard_{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[size[0], size[1], 0.001], pos=[pos[0], pos[1], floor_z + 0.001],
                material="hazard")
    fence = 1.75
    for k, (x, y, horizontal) in enumerate(((cx, fence, True), (cx, -fence, True),
                                            (cx - fence, 0, False))):
        half = [fence, 0.01, 0.6] if horizontal else [0.01, fence, 0.6]
        _visual(world, name=f"fence_{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=half,
                pos=[x, y, floor_z + 0.65], material="fence_mesh")
        for t in np.linspace(-fence, fence, 5):
            px, py = (x + t, y) if horizontal else (x, y + t)
            _visual(world, type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025, 0.025, 0.65],
                    pos=[px, py, floor_z + 0.65], rgba=[0.95, 0.75, 0.1, 1.0])


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
    floor.size = [0.925, 0.85, 0.05]
    floor.pos = [0.075, 0.0, 0.0]
    spec.add_texture(name="bench", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, mark=mujoco.mjtMark.mjMARK_EDGE,
                     rgb1=[0.62, 0.64, 0.67], rgb2=[0.58, 0.6, 0.63], markrgb=[0.5, 0.52, 0.55],
                     width=300, height=300)
    spec.add_material(name="bench", textures=["", "bench"], texuniform=True, texrepeat=[8, 8],
                      reflectance=0.05)
    floor.material = "bench"
    if cfg.dressing:
        spec.add_texture(name="factory_floor", type=mujoco.mjtTexture.mjTEXTURE_2D,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                         rgb1=[0.42, 0.43, 0.45], rgb2=[0.38, 0.39, 0.41], width=512, height=512)
        spec.add_material(name="factory_floor", textures=["", "factory_floor"], texuniform=True,
                          texrepeat=[6, 6])
        spec.add_texture(name="hazard", type=mujoco.mjtTexture.mjTEXTURE_2D,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[0.95, 0.75, 0.1],
                         rgb2=[0.1, 0.1, 0.1], width=64, height=64)
        spec.add_material(name="hazard", textures=["", "hazard"], texuniform=True,
                          texrepeat=[40, 40])
        spec.add_material(name="fence_mesh", rgba=[0.75, 0.78, 0.8, 0.18])
        _add_dressing(spec)
    _add_conveyor(spec, cfg)
    _add_tower_fixture(spec, cfg)
    _add_reject_bin(spec, cfg)
    _add_stack_light(spec, (0.85, -0.7))
    cube = cube or CubeSpec()
    for k, xy in enumerate(queue_xy(cfg)):
        add_cube(spec, cube, xy=tuple(xy), name=CUBE_NAME.format(k), rgba=CUBE_COLOURS[k % 3])
        spec.geom(CUBE_NAME.format(k)).conaffinity = 1 | BELT_CONTACT  # rides on the belt
    spec.visual.global_.azimuth = 200
    spec.visual.global_.elevation = -32
    return spec


@dataclass
class Workcell:
    """The compiled cell plus handles for its moving parts."""

    model: mujoco.MjModel
    data: mujoco.MjData
    config: WorkcellConfig
    cubes: list[CubeIndices] = field(default_factory=list)

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

    @property
    def tower_xy(self) -> np.ndarray:
        return station_xy(self.config.tower_angle, self.config.radius)

    @property
    def reject_xy(self) -> np.ndarray:
        return station_xy(self.config.reject_angle, self.config.radius)

    def reset(self, rng: np.random.Generator | None = None, lateral: float = 0.0,
              yaw: float = 0.0) -> None:
        """Robot at home, belt at its start, cubes queued upstream on the belt, optionally with
        random lateral offsets up to `lateral` (m) and yaw up to `yaw` (rad) either way."""
        reset_to_home(self.model, self.data, PandaIndices.from_model(self.model))
        cfg = self.config
        self.data.qpos[self.model.joint("belt").qposadr[0]] = 0.0
        self.data.qvel[self.model.joint("belt").dofadr[0]] = 0.0
        self.data.ctrl[self.model.actuator("belt").id] = 0.0
        travel = belt_direction(cfg)
        normal = np.array([-travel[1], travel[0]])
        heading = float(np.arctan2(travel[1], travel[0]))
        for cube, start in zip(self.cubes, queue_xy(cfg), strict=True):
            offset = rng.uniform(-lateral, lateral) if rng is not None and lateral else 0.0
            xy = start + offset * normal
            angle = heading + (rng.uniform(-yaw, yaw) if rng is not None and yaw else 0.0)
            self.data.qpos[cube.qpos:cube.qpos + 7] = [
                *xy, cube.half_size[2] + cfg.belt_lift, *_yaw_quat(angle)]
            self.data.qvel[cube.qvel:cube.qvel + 6] = 0.0
        self.conveyor.reset()
        mujoco.mj_forward(self.model, self.data)

    def set_stack_light(self, *on: str) -> None:
        for name in STACK_LIGHT:
            self.model.geom_rgba[self.model.geom(f"lamp_{name}").id] = (
                _LAMP_ON[name] if name in on else _LAMP_OFF[name])


class Conveyor:
    """Indexing control of the belt: run until the leading cube reaches the pick point."""

    def __init__(self, cell: Workcell) -> None:
        self.cell = cell
        self.travel = belt_direction(cell.config)
        self.joint = cell.model.joint("belt")
        self.actuator = cell.model.actuator("belt").id
        self.target = 0.0

    def reset(self) -> None:
        self.target = 0.0

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

    def index(self) -> bool:
        """Set the belt to bring the leading cube on the belt to the pick point. Returns False
        if no cube is left on the belt."""
        queued = [c for c in self.cell.cubes if self.on_belt(c)]
        if not queued:
            return False
        lead = min(queued, key=self.upstream_distance)
        self.target = self.position + max(0.0, self.upstream_distance(lead))
        return True

    def update(self) -> None:
        """Speed command toward the current target (call once per control step)."""
        cfg = self.cell.config
        remaining = self.target - self.position
        speed = float(np.clip(cfg.belt_accel_gain * remaining, -cfg.belt_speed, cfg.belt_speed))
        self.cell.data.ctrl[self.actuator] = speed if abs(remaining) > 1e-4 else 0.0

    @property
    def moving(self) -> bool:
        return abs(self.target - self.position) > 1e-3 or abs(
            self.cell.data.qvel[self.joint.dofadr[0]]) > 1e-3
