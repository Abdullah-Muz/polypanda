"""DoorEnv (v4.0): open or close the sliding lid of a benchtop machine the robot has never seen.

The learned door skill of docs/v4_plan.md (sections 3.2, 3.3, 3.5). Each episode is a new machine
(domain randomization, drawn without recompiling: simulation/machine_variation.py) and a
command, `open` or `close`, given per episode as the stand-up is given its turn.

**Scene.** A minimal training scene for speed: the Panda on the bench (no rail, conveyor or hall)
and one machine at station angle 0, its pocket at the cell's 0.5 m radius (scene.station_angle
puts it elsewhere, for the station-frame test). In the cell the skill runs at the machine's angle
through supervisor.stations.StationView (reset option "continue").

**Episodes.** `open`: the lid starts closed and unlatched, as the machine leaves it at the end of
a cycle. `close`: the lid starts open, between the opening that uncovers the pocket ("needed": the
opening along the slide plus the lid's overlap) and its travel less 5 mm. The arm starts in the
cell's carry posture above the machine (TCP 0.32 m high over the pocket, pointing down), jittered
by up to 2 cm and 10 deg, fingers open.

**What is randomized** (DoorVariationConfig, per episode): which way the lid slides (right, left,
out) and how far (sideways 0.12-0.14 m, outward 0.265-0.275 m), its Coulomb friction and damping
(log-uniform), a closing spring on half the machines (with a soft-close damper and a hold-open
detent that `open` must reach), where the handle sits along its bar and along the slide
and how high, the latch's closing speed (log-uniform), where the machine stands (+-1.5 cm) and
how it is turned (+-10 deg).

**Observation** (schema door-v1-49d, OBS_LAYOUT), only what a real robot could know:
- its arm: joints q, velocities dq and commanded targets q_cmd (as the grasp);
- the TCP's position and rotation (6-D: the first two columns);
- the gripper: finger gap, whether it is commanded closed, and each finger touching the handle
  bar (the grasp's contact flags);
- `handle_rel`: the handle bar's centre relative to the TCP, from a **camera estimate**: the true
  centre plus an error drawn once per episode (sensing.handle_bias, 1 mm per axis, the station
  camera of docs/pose_error.md) plus a small per-step noise. The camera only sees the handle while
  the hand is not over it: within sensing.occlusion_radius (horizontally) the last estimate is
  held, as the estimator withholds occluded estimates;
- `bar_align`: the camera's estimate of which way the bar runs, as the error of the fingers'
  closing direction to the bar's normal (sin and cos of twice the error, the fingers being
  symmetric). A camera sees the bar's direction, and the fingers must close across it; it gives
  the slide's axis but not its sense (left or right), travel, stiffness or latch.
  sensing.observe_bar_yaw = false zeroes these slots, for a stricter variant (open question);
- `touch_travel`: how far the TCP has moved since a finger came into contact with the bar (zero
  once it has been off the bar for sensing.touch_release steps): the robot's own hand motion,
  which is how it knows the lid's motion while it pushes or holds the handle, whether it hooks
  the bar (fingers open, straddling it) or grips it;
- `force`: the external force on the arm, estimated as a real Panda does from its joint torque
  sensors; here the sum of the contact forces on the robot's bodies plus noise (1 N per axis);
- `door_io`: the two signals the machine exposes: door-closed switch (lid within 2 mm of shut)
  and latched;
- `door_command`: one-hot (open, close).
DoorV2Env (schema door-v2-50d) appends `door_open_switch`: the lid within 2 mm of its open stop.

Not observed: the lid's slide direction, travel, friction, damping, spring and latch speed, nor
the lid's displacement. A robot only knows the lid's displacement through its own hand while it
moves the handle (touch_travel, and the held handle estimate), and the camera's view of the
handle when the hand is clear of it. Finding those properties by feel is the point of the skill.

**Action** (5-D): the cartesian-v1-4d hand motion of CartesianActionAdapter (x, y, z, yaw; the
hand points down), plus a gripper command with hysteresis: above gripper.close_above it closes,
below gripper.open_below it opens, in between it keeps its state. The policy commands the gripper
itself, rather than a grasp-style gate, so it may learn either way to move the lid: **hook** the
bar (fingers open, straddling it, the trailing finger pushing; what the scripted door and the
expert do, since gripping the bar is fragile in this simulator: fingertip pads 1-2 mm off centre
jam on its edge) or grip it, and letting go stays its own decision (a gate would need privileged
knowledge to decide when the lid has gone far enough). The dead band stops exploration noise
from chattering the fingers. The fingers move at most gripper.speed (0.1 m/s each, a real hand's
order): commanded straight shut, the cell's stiff finger servo slams them through the bar within
one control step. The action schema is ACTION_SCHEMA.

**Reward** (DoorRewardConfig; terms in info["reward_terms"]):
- `approach`: per metre the TCP nears the true handle, until the lid's goal is reached;
- `touch`: a potential of `touch` while a finger touches the bar, until the goal is reached;
- `progress`: per unit of the lid's progress toward the commanded end (open: opening / needed,
  capped at 1; close: the share of the starting opening closed); falls give it back;
- `retreat`: once the goal is reached, per metre the hand, off the bar, rises above it (capped);
- `success`: a bonus, once, at success (the episode ends);
- penalties: `force` per newton of the estimated force above force_limit; `collision` per step
  the arm touches the machine's body (walls, roof, pocket) or the bench; `slam` per m/s the lid
  hits a stop faster than allowed (outward 0.15, closing 0.3 m/s: a latch needs at most 0.1),
  plus `slam_hit` once per such hit (both paid from `slam_margin` below the allowed speeds, a
  training margin; the slams counted in stats are those over the allowed speeds);
  `stop_press` per step the arm pushes harder than stop_force while the lid sits at a stop;
  `time` per step; `action` x |hand action|^2.
Terms that belong to one phase (approach, touch before the goal; retreat after it) are paid only
between two steps in the same phase, so reaching the goal is never penalized.

**Success.** `open`: the lid opened at least "needed"; `close`: latched. Both: the hand clear of
the handle (no finger touching the bar) with the TCP at least success.clear_height above the
bar, all held for success.hold_steps (0.5 s). Nothing requires the fingers to have closed (a
hooking policy never closes them). The episode then terminates. Otherwise it is truncated at
`horizon` steps (15 s at 20 Hz) and info["failure"] says why: not_opened, not_closed,
closed_not_latched (shut too slowly to latch), not_released (the goal reached but the hand not
clear above the bar), never_touched, or non_finite.

**Drawers (v4.1, docs/v4_1_plan.md sections 1 and 4.1).** `variation.kinds` lists the tester
kinds drawn: ("lid",), the default, is v4.0's task exactly (the same scene, draws, episodes and
observations, so v4.0's policies run unchanged; tests/test_lid_identity.py); with "drawer" in it,
the scene also holds a drawer tester ("D"; the lid machine stays "M") at the same station, the
other one parked out of the way, and each episode draws its kind (uniformly; no draw with one
kind). A drawer opens toward the robot (slide
"toward"); its handle is a bar on the drawer's front plate, hooked from above with open fingers:
the finger behind the bar pulls it open and pushes the plate shut, so for a drawer "touching the
handle" (the contact flags, touch_travel, the touch reward, success's "clear of the handle") means
the bar or the plate. The observation is unchanged: the kind is not an input, but it shows in the
camera's handle estimate (a drawer's bar is low and in front, about 0.15 m above the bench, a
lid's on top at 0.2 m or more). Drawers vary in travel, friction, damping, the latch speed, and
where the bar sits (along it, up and down, and how far it stands off the plate); by default
without a closing spring.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from polypanda.config import load_yaml
from polypanda.control.action_adapter import action_limit, bound_action, make_action_adapter
from polypanda.control.ik import PoseIK, closing_yaw, top_down_quat
from polypanda.environments.grasp import GRIPPER_OPEN
from polypanda.environments.reach import ActionConfig, CartesianConfig
from polypanda.simulation.assets import panda_spec
from polypanda.simulation.machine import (
    ALL_SLIDES,
    DRAWER_SLIDE,
    SLIDES,
    Machine,
    MachineConfig,
    add_machine,
)
from polypanda.simulation.machine_variation import MachineLayout, needed_opening, variant
from polypanda.simulation.panda import PandaIndices, reset_to_home, tcp_position
from polypanda.simulation.workcell import set_finger_stiffness

OBS_SCHEMA = "door-v1-49d"
OBS_LAYOUT = (
    ("q", 7), ("dq", 7), ("q_cmd", 7),
    ("tcp", 3), ("tcp_rot6", 6), ("finger_gap", 1), ("gripper_closed", 1), ("contacts", 2),
    ("handle_rel", 3), ("bar_align", 2), ("touch_travel", 3), ("force", 3),
    ("door_io", 2), ("door_command", 2),
)
OBS_DIM = sum(n for _, n in OBS_LAYOUT)
OBS_SLICES = {
    name: slice(start, start + n)
    for (name, n), start in zip(
        OBS_LAYOUT, np.cumsum([0] + [n for _, n in OBS_LAYOUT[:-1]]), strict=True)
}
# v2 (DoorV2Env, task door_v2): v1 plus the machine's door-open switch, appended at the end so a
# v1 network widens to it with a zero weight (training.runner, init_pad_obs)
OBS_SCHEMA_V2 = "door-v2-50d"
OBS_LAYOUT_V2 = OBS_LAYOUT + (("door_open_switch", 1),)
OBS_DIM_V2 = sum(n for _, n in OBS_LAYOUT_V2)
OBS_SLICES_V2 = {**OBS_SLICES, "door_open_switch": slice(OBS_DIM, OBS_DIM_V2)}
ACTION_SCHEMA = "cartesian-v1-4d+gripper"   # the hand motion of cartesian-v1-4d, then a gripper
COMMANDS = ("open", "close")
MACHINE = "M"
DOOR_KINDS = ("lid", "drawer")       # the testers the robot opens (a pneumatic one opens itself)
MACHINES = {"lid": MACHINE, "drawer": "D"}
PARKED = (0.0, 5.0)                  # m: where the machine of the other kind waits (out of reach)
FINGER_OPEN = 0.04       # m per finger at gripper ctrl 255 (the Menagerie hand)


@dataclass(frozen=True)
class DoorSceneConfig:
    station_angle: float = 0.0       # deg about the base axis (0: straight in front)
    radius: float = 0.5              # m, base axis to the machine's pocket (the cell's)
    finger_stiffness: float = 1000.0  # N/m, the cell's gripper (simulation.workcell)


@dataclass(frozen=True)
class DoorVariationConfig:
    """Machine variants, drawn per episode (training and validation alike)."""
    slides: tuple[str, ...] = SLIDES                    # drawn uniformly
    travel_side: tuple[float, float] = (0.12, 0.14)     # m; needed sideways: 0.115
    travel_out: tuple[float, float] = (0.265, 0.275)    # m; needed outward: 0.255 (reach: the
    #                                                     arm's limit is about 0.8 m out)
    friction: tuple[float, float] = (0.2, 10.0)         # N, log-uniform (nominal 0.4)
    damping: tuple[float, float] = (1.0, 10.0)          # N s/m, log-uniform (nominal 4)
    spring_prob: float = 0.5                            # share of machines with a closing spring
    spring: tuple[float, float] = (10.0, 40.0)          # N/m pulling it shut, uniform; a springy
    #                                                     lid stays open only in its detent
    soft_close_speed: float = 0.2                       # m/s: a springy lid's damper keeps it
    #                                                     from shutting itself faster than this
    handle_offset_side: float = 0.03   # m either way along the bar, sideways lids (bar along the
    handle_offset_out: float = 0.01    # 27 cm lid); outward lids (bar across the 13 cm lid)
    handle_shift_side: float = 0.02    # m either way along the slide
    handle_shift_out: float = 0.01     # (outward: kept small, the arm's reach)
    handle_height: tuple[float, float] = (0.025, 0.05)  # m, bar centre above the lid's top
    latch_speed: tuple[float, float] = (0.01, 0.1)      # m/s closing speed to latch, log-uniform
    position: float = 0.015            # m either way, each axis of the station frame
    turn: float = 10.0                 # deg either way
    # v4.1: the testers drawn (uniformly; no draw with one kind). ("lid",) is v4.0's task, draw
    # for draw; "drawer" adds drawer testers (simulation/machine.py), varied as below
    kinds: tuple[str, ...] = ("lid",)
    travel_drawer: tuple[float, float] = (0.12, 0.14)   # m; needed: 0.112
    drawer_spring_prob: float = 0.0    # share of self-closing drawers (springs as on lids)
    handle_offset_drawer: float = 0.03   # m either way along the bar (an 18 cm plate)
    handle_shift_drawer: float = 0.005   # m up or down on the plate (the hand clears the box's
    #                                      top by 24 mm less this when it hooks the bar)
    handle_height_drawer: tuple[float, float] = (0.035, 0.05)  # m, bar centre off the plate: a
    #                                      finger needs ~2 cm between bar and plate

    def __post_init__(self) -> None:
        if not self.kinds or set(self.kinds) - set(DOOR_KINDS):
            raise ValueError(f"kinds must be some of {DOOR_KINDS}, not {self.kinds!r}")


@dataclass(frozen=True)
class DoorStartConfig:
    open_prob: float = 0.5           # share of `open` commands
    carry_height: float = 0.32       # m, TCP (the cell's carry posture)
    jitter_xy: float = 0.02          # m either way
    jitter_z: float = 0.02           # m either way
    jitter_yaw: float = 0.17         # rad either way (10 deg)
    close_margin: float = 0.005      # m: close episodes start at most travel - this open


@dataclass(frozen=True)
class DoorSensingConfig:
    handle_bias: float = 0.001       # m per axis, drawn once per episode (docs/pose_error.md)
    handle_noise: float = 0.0003     # m per axis per step
    occlusion_radius: float = 0.06   # m: the hand closer than this (horizontally) hides it
    bar_yaw_bias: float = 0.0087     # rad (0.5 deg), once per episode
    bar_yaw_noise: float = 0.0017    # rad per step
    observe_bar_yaw: bool = True     # false: the bar_align slots read zero
    force_noise: float = 1.0         # N per axis per step
    force_clip: float = 100.0        # N per axis
    closed_switch: float = 0.002     # m: the door-closed switch closes within this of shut
    open_switch: float = 0.002       # m: the door-open switch closes within this of the open stop
    #                                  (observed by DoorV2Env only; the feel script reads it)
    touch_release: int = 5           # steps off the bar before touch_travel starts again at 0


@dataclass(frozen=True)
class DoorGripperConfig:
    close_above: float = 0.25        # gripper action above this closes the fingers
    open_below: float = -0.25        # below this opens them; in between they keep their state
    # m/s per finger: the finger servo's target moves at most this fast (updated every
    # `chunk` physics steps). A step command from open to shut, with the cell's 1000 N/m finger
    # servo, slams the fingers through the 16 mm handle bar within one control step (measured);
    # a real Franka Hand closes at up to about 0.05-0.1 m/s per finger.
    speed: float = 0.1
    chunk: int = 5


@dataclass(frozen=True)
class DoorSuccessConfig:
    clear_height: float = 0.03       # m, TCP above the bar's centre, off the bar
    hold_steps: int = 10             # consecutive steps (0.5 s)


@dataclass(frozen=True)
class DoorFocusConfig:
    """Training only (DoorConfig.for_evaluation drops it): a share of episodes drawn from chosen
    cells, "command/slide/spring" or "command/slide/plain" (e.g. "open/out/spring"; a drawer's
    slide is "toward"), the rest as usual. With prob 0 the episode stream is the plain one, draw
    for draw."""
    prob: float = 0.0
    cells: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for c in self.cells:
            parts = c.split("/")
            if (len(parts) != 3 or parts[0] not in COMMANDS or parts[1] not in ALL_SLIDES
                    or parts[2] not in ("spring", "plain")):
                raise ValueError(f"focus cell {c!r}: command/slide/spring|plain")
        if self.prob > 0 and not self.cells:
            raise ValueError("focus prob > 0 needs cells")


@dataclass(frozen=True)
class DoorRewardConfig:
    approach: float = 10.0           # per m the TCP nears the handle (before the goal)
    touch: float = 0.5               # potential while a finger touches the bar (before it)
    progress: float = 10.0           # per unit of progress (0 -> 1) toward the commanded end
    retreat: float = 20.0            # per m the hand, off the bar, rises above it (after it)
    retreat_cap: float = 0.05        # m
    success: float = 10.0            # once, at success
    force: float = 0.01              # per N of estimated force above force_limit, per step
    force_limit: float = 40.0        # N
    collision: float = 0.1           # per step touching the machine's body or the bench
    slam: float = 20.0               # per m/s over the allowed speed, when the lid hits a stop
    slam_open_speed: float = 0.15    # m/s allowed at the open stop
    slam_close_speed: float = 0.3    # m/s allowed at the closed stop
    slam_hit: float = 0.0            # once per slam, however slight (dev round 1's opens hit the
    #                                  open stop at 0.152-0.224 m/s: 20 per m/s cost them < 1.5)
    slam_margin: float = 0.0         # m/s: slam and slam_hit are paid from this much below the
    #                                  allowed speeds (a training margin, dev round 3: round 2's
    #                                  slams were 0.151-0.205 m/s); stats["slams"] still counts
    #                                  hits over the allowed speeds themselves
    stop_press: float = 0.05         # per step pushing harder than stop_force at a stop
    stop_force: float = 25.0         # N
    time: float = 0.005              # per step
    action: float = 0.001            # x |hand action|^2
    non_finite: float = 10.0         # on numerical blow-up (terminates)


def _cartesian4() -> ActionConfig:
    return ActionConfig(space="cartesian", cartesian=CartesianConfig(dof=4))


@dataclass(frozen=True)
class DoorConfig:
    action: ActionConfig = field(default_factory=_cartesian4)
    scene: DoorSceneConfig = field(default_factory=DoorSceneConfig)
    variation: DoorVariationConfig = field(default_factory=DoorVariationConfig)
    start: DoorStartConfig = field(default_factory=DoorStartConfig)
    sensing: DoorSensingConfig = field(default_factory=DoorSensingConfig)
    gripper: DoorGripperConfig = field(default_factory=DoorGripperConfig)
    success: DoorSuccessConfig = field(default_factory=DoorSuccessConfig)
    reward: DoorRewardConfig = field(default_factory=DoorRewardConfig)
    focus: DoorFocusConfig = field(default_factory=DoorFocusConfig)   # training only
    horizon: int = 300               # policy steps (15 s at 20 Hz)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DoorConfig:
        return load_yaml(cls, path)

    def for_evaluation(self) -> DoorConfig:
        """The task as evaluated: without the training-only focus on chosen cells."""
        return replace(self, focus=DoorFocusConfig())


def build_door_scene(cfg: DoorSceneConfig, kinds: tuple[str, ...] = ("lid",)
                     ) -> tuple[mujoco.MjModel, mujoco.MjData, dict[str, MachineConfig]]:
    """The Panda on the bench and one nominal machine of each kind (named MACHINES[kind]),
    re-laid out per episode. With the lid alone (the default) this is v4.0's scene."""
    spec = panda_spec()
    set_finger_stiffness(spec, cfg.finger_stiffness)
    a = np.radians(cfg.station_angle)
    compiled = {}
    for kind in DOOR_KINDS:            # (a fixed order: the lid machine first, as in v4.0)
        if kind in kinds:
            compiled[kind] = add_machine(spec, MACHINES[kind],
                                         (cfg.radius * np.cos(a), cfg.radius * np.sin(a)),
                                         cfg.station_angle, MachineConfig(kind=kind))
    model = spec.compile()
    return model, mujoco.MjData(model), compiled


def _rz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _wrap(angle: float, period: float) -> float:
    return float((angle + period / 2) % period - period / 2)


class DoorEnv(gym.Env):
    metadata = {"render_modes": []}
    ACTION_SCHEMA = ACTION_SCHEMA     # (recorded in a trained run's metadata)

    def __init__(self, config: DoorConfig | None = None, scene=None,
                 machine: Machine | None = None) -> None:
        """`scene` (model, data) with `machine` (its runtime): run on a machine in an existing
        scene (the line), started with reset option "continue"; by default the env builds its own
        training scene and draws a machine variant every episode."""
        self.config = config or DoorConfig()
        a = self.config.action
        if a.space != "cartesian" or a.cartesian.dof != 4:
            raise ValueError("the door skill acts with cartesian-v1-4d hand motions")
        if scene is None:
            kinds = self.config.variation.kinds
            self.model, self.data, compiled = build_door_scene(self.config.scene, kinds)
            self.machines = {k: Machine(self.model, self.data, MACHINES[k], c, parts=[])
                             for k, c in compiled.items()}
            self.layouts = {k: MachineLayout(self.model, MACHINES[k], c)
                            for k, c in compiled.items()}
            self.frame = np.radians(self.config.scene.station_angle)
        else:
            if machine is None:
                raise ValueError("a shared scene needs its machine's runtime")
            if machine.cfg.kind not in DOOR_KINDS:
                raise ValueError(f"a {machine.cfg.kind!r} machine opens itself")
            self.model, self.data = scene
            self.machines, self.layouts = {machine.cfg.kind: machine}, None
            self.frame = 0.0    # until set_station() (StationView does it)
        self.base = np.zeros(3)  # where the robot's base axis stands (a rail, v4: set_station)
        m = self.model
        self.arm = PandaIndices.from_model(m)
        self.adapter = make_action_adapter(m, self.arm, self.config.action)
        self.hand = m.body("hand").id
        self.arm_bodies = {b for b in range(m.nbody) if b and m.body_rootid[b]
                           == m.body_rootid[self.hand]}
        self._fingers = (m.body("left_finger").id, m.body("right_finger").id)
        self._finger_qpos = [m.joint(f"finger_joint{i}").qposadr[0] for i in (1, 2)]
        self._activate(next(iter(self.machines)))
        f = self.config.focus
        if self.layouts is not None and any(
                ("drawer" if c.split("/")[1] == DRAWER_SLIDE else "lid") not in self.layouts
                for c in f.cells):
            raise ValueError("a focus cell's kind is not among variation.kinds")
        limit = action_limit(self.config.action)
        self.action_space = spaces.Box(-limit, limit, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self._ik = PoseIK()
        self._carry = self._carry_posture()
        self.command = "open"
        self.closed = False
        self.episode: dict | None = None
        self._steps = self._hold = 0
        self._prev: dict = {}

    @property
    def elapsed_steps(self) -> int:
        return self._steps

    def _activate(self, kind: str) -> None:
        """Work on the machine of `kind` (lid or drawer): its runtime, layout, handle, the geoms
        that count as touching the handle (the bar; a drawer's front plate too, which the finger
        behind the bar pushes it shut by) and the bodies that count as collisions (its body,
        walls, roof and pocket, and the bench; not the moving lid or drawer)."""
        self.kind = kind
        self.machine = self.machines[kind]
        self.layout = self.layouts[kind] if self.layouts is not None else None
        m, name = self.model, self.machine.name
        self.bar_geom = m.geom(f"{name}_handle").id
        self.touch_geoms = {self.bar_geom}
        if kind == "drawer":
            self.touch_geoms.add(m.geom(f"{name}_drawer_face").id)
        self.lid_body = self.machine.door_body
        self.machine_bodies = {self.machine.body} | {
            m.body(f"{name}_pocket{k}").id for k in range(4)} | {0}

    # --- episodes ------------------------------------------------------------------------
    def _carry_posture(self) -> np.ndarray:
        """The cell's carry posture over a station straight ahead (ScriptedCell)."""
        home = self._ik.data.qpos[self._ik.idx.arm_qpos].copy()
        _, R = self._ik.forward(home)
        q, ok = self._ik.solve(home, [self.config.scene.radius, 0.0,
                                      self.config.start.carry_height],
                               top_down_quat(closing_yaw(R)))
        if not ok:
            raise RuntimeError("carry posture unreachable")
        return q

    def draw_episode(self, rng: np.random.Generator, command: str | None = None) -> dict:
        """One episode: the command, the machine variant and where it stands (in the station
        frame), the lid's start, the arm's jitter and the camera's errors."""
        v, s, n = self.config.variation, self.config.start, self.config.sensing

        def log_uniform(lo_hi):
            lo, hi = np.log(lo_hi)
            return float(np.exp(rng.uniform(lo, hi))) if hi > lo else float(lo_hi[0])

        def sym(x):
            return float(rng.uniform(-x, x))

        f = self.config.focus
        focus = (f.cells[int(rng.integers(len(f.cells)))].split("/")
                 if f.prob > 0 and rng.uniform() < f.prob else None)   # (no draw at prob 0)
        draw_command = COMMANDS[0] if rng.uniform() < s.open_prob else COMMANDS[1]
        command = command or (focus[0] if focus else draw_command)
        kind = v.kinds[0]
        if len(v.kinds) > 1:                                  # (no draw with one kind)
            kind = str(v.kinds[int(rng.integers(len(v.kinds)))])
        if focus:
            kind = "drawer" if focus[1] == DRAWER_SLIDE else "lid"
        if kind == "drawer":
            travel = float(rng.uniform(*v.travel_drawer))
            friction, damping = log_uniform(v.friction), log_uniform(v.damping)
            spring = (float(rng.uniform(*v.spring)) if rng.uniform() < v.drawer_spring_prob
                      else 0.0)
            if focus:
                spring = ((spring or float(rng.uniform(*v.spring))) if focus[2] == "spring"
                          else 0.0)
            ep = {
                "kind": kind, "command": command, "slide": DRAWER_SLIDE, "travel": travel,
                "friction": friction, "damping": damping, "spring": spring,
                "handle_offset": sym(v.handle_offset_drawer),
                "handle_shift": sym(v.handle_shift_drawer),
                "handle_height": float(rng.uniform(*v.handle_height_drawer)),
                "latch_speed": log_uniform(v.latch_speed),
                "offset": (sym(v.position), sym(v.position)), "turn": sym(v.turn),
            }
            needed = needed_opening(self.layouts["drawer"].compiled)
        else:
            slide = str(v.slides[int(rng.integers(len(v.slides)))])
            slide = focus[1] if focus else slide
            out = slide == "out"
            travel = float(rng.uniform(*(v.travel_out if out else v.travel_side)))
            friction, damping = log_uniform(v.friction), log_uniform(v.damping)
            spring = float(rng.uniform(*v.spring)) if rng.uniform() < v.spring_prob else 0.0
            if focus:
                spring = ((spring or float(rng.uniform(*v.spring))) if focus[2] == "spring"
                          else 0.0)
            ep = {
                "command": command, "slide": slide, "travel": travel,
                "friction": friction, "damping": damping, "spring": spring,
                "handle_offset": sym(v.handle_offset_out if out else v.handle_offset_side),
                "handle_shift": sym(v.handle_shift_out if out else v.handle_shift_side),
                "handle_height": float(rng.uniform(*v.handle_height)),
                "latch_speed": log_uniform(v.latch_speed),
                "offset": (sym(v.position), sym(v.position)), "turn": sym(v.turn),
            }
            if len(v.kinds) > 1:      # (v4.0's lid-only episodes carry no kind)
                ep = {"kind": kind, **ep}
            needed = needed_opening(replace(MachineConfig(), slide=slide))
        lid_hi = max(needed, travel - s.close_margin)
        lid = float(rng.uniform(needed, lid_hi))
        if ep["spring"] > 0:
            lid = travel - 0.002          # a springy lid is left open in its detent
            # a soft-close damper, as springy guards have: shutting itself from fully open, the
            # lid moves no faster than soft_close_speed
            ep["damping"] = max(ep["damping"], ep["spring"] * travel / v.soft_close_speed)
        ep["lid"] = 0.0 if command == "open" else lid
        ep["jitter"] = (sym(s.jitter_xy), sym(s.jitter_xy), sym(s.jitter_z), sym(s.jitter_yaw))
        ep["handle_bias"] = tuple(float(x) for x in rng.normal(0.0, n.handle_bias, 3))
        ep["bar_yaw_bias"] = float(rng.normal(0.0, n.bar_yaw_bias))
        return ep

    def machine_config(self, ep: dict) -> MachineConfig:
        """The episode's machine variant (of its kind: "kind", a lid without it)."""
        layout = self.layouts[ep.get("kind", "lid")]
        return variant(layout.compiled, slide=ep["slide"], travel=ep["travel"],
                       lid_friction=ep["friction"], lid_damping=ep["damping"],
                       lid_spring=ep["spring"], handle_offset=ep["handle_offset"],
                       handle_shift=ep["handle_shift"], handle_height=ep["handle_height"],
                       latch_speed=ep["latch_speed"])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if options.get("continue"):
            return self._continue(options)
        if self.layout is None:
            raise ValueError("a shared scene starts with the reset option 'continue'")
        ep = dict(options["episode"]) if "episode" in options else self.draw_episode(
            self.np_random, options.get("command"))
        self._start(ep)
        self._begin(ep["command"], ep["handle_bias"], ep["bar_yaw_bias"])
        self.episode = ep
        return self._observation(), self._reset_info("draw" if "episode" not in options
                                                     else "explicit")

    def _start(self, ep: dict) -> None:
        m, d = self.model, self.data
        kind = ep.get("kind", "lid")
        if kind not in self.layouts:
            raise ValueError(f"a {kind!r} episode, but variation.kinds is {tuple(self.layouts)}")
        self._activate(kind)
        cfg = self.machine_config(ep)
        reset_to_home(m, d, self.arm)
        self.layout.apply(m, cfg, self.machine)
        sc = self.config.scene
        R = _rz(self.frame)
        xy = (R @ np.array([sc.radius + ep["offset"][0], ep["offset"][1], 0.0]))[:2]
        self.layout.place(m, xy, self.frame + np.radians(ep["turn"]))
        for other, layout in self.layouts.items():      # the other kind waits out of reach
            if other != kind:
                layout.place(m, PARKED, 0.0)
        d.qpos[self.machine.lid_qpos] = ep["lid"]
        d.qvel[self.machine.lid_dof] = 0.0
        # the arm: the carry posture, jittered in the station frame, then turned to the station
        jx, jy, jz, jyaw = ep["jitter"]
        _, R_carry = self._ik.forward(self._carry)
        q, ok = self._ik.solve(self._carry, [sc.radius + jx, jy,
                                             self.config.start.carry_height + jz],
                               top_down_quat(closing_yaw(R_carry) + jyaw))
        if not ok:
            q = self._carry.copy()
        q = q.copy()
        q[0] += self.frame
        d.qpos[self.arm.arm_qpos] = q
        for adr in self._finger_qpos:
            d.qpos[adr] = 0.04                     # fingers open
        d.qvel[:] = 0.0
        d.ctrl[self.arm.gripper_actuator] = GRIPPER_OPEN
        mujoco.mj_forward(m, d)
        self.machine.reset()

    def _continue(self, options: dict):
        """Start from the scene as it is (a skill inside a larger task): option "command"
        ("open" or "close"). The camera's error is drawn now, unless options "handle_bias" (m,
        3 values in the station frame) and "bar_yaw_bias" (rad) give it (they are drawn anyway, so
        the random stream does not depend on them)."""
        command = options.get("command", "open")
        if command not in COMMANDS:
            raise ValueError(f"command must be one of {COMMANDS}, not {command!r}")
        n = self.config.sensing
        bias = tuple(float(x) for x in self.np_random.normal(0.0, n.handle_bias, 3))
        yaw_bias = float(self.np_random.normal(0.0, n.bar_yaw_bias))
        if options.get("handle_bias") is not None:
            bias = tuple(float(x) for x in options["handle_bias"])
        if options.get("bar_yaw_bias") is not None:
            yaw_bias = float(options["bar_yaw_bias"])
        mujoco.mj_forward(self.model, self.data)
        self._begin(command, bias, yaw_bias)
        self.episode = None
        return self._observation(), self._reset_info("continue")

    def _begin(self, command: str, bias, yaw_bias: float) -> None:
        """Start the episode's bookkeeping from the scene as it is now."""
        if command not in COMMANDS:
            raise ValueError(f"command must be one of {COMMANDS}, not {command!r}")
        self.command = command
        if self.layout is None:
            self.adapter.adopt_ctrl(self.data)      # in a cell: keep the servo commands
        else:
            self.adapter.reset_to_measured(self.data)
        ctrl = float(self.data.ctrl[self.arm.gripper_actuator])
        self.closed = ctrl < GRIPPER_OPEN / 2
        self._finger_target = FINGER_OPEN * ctrl / GRIPPER_OPEN
        self._bias = self._frame_vec(np.asarray(bias, dtype=float))
        self._yaw_bias = yaw_bias
        self._steps = self._hold = 0
        self.needed = needed_opening(self.machine.cfg)
        self.lid_start = max(self.machine.lid, 0.01)
        self.stats = {"collision_steps": 0, "slams": 0, "peak_force": 0.0, "touched": False,
                      "goal_reached": False}
        self._contacts = self._finger_contacts()
        self._touch_at = tcp_position(self.data, self.arm) if any(self._contacts) else None
        self._off_bar = 0
        self._force = self._external_force()[0]
        self._seen_handle = None
        self._seen_yaw = None
        self._look()
        self._prev = self._potentials()

    def _reset_info(self, start: str) -> dict:
        return {"episode": self.episode, "start": start, "command": self.command,
                "needed": self.needed, "lid": self.machine.lid}

    def set_station(self, angle: float, base=(0.0, 0.0)) -> None:
        """The station this env's machine stands at: its angle (rad) about the robot's base axis
        and where that axis stands on the table (a robot on a rail). StationView sets it, so that
        in a shared scene (the line) the scripts' beliefs (BaselineDoor._station, the nominal
        handle) and the camera's errors are drawn in the right frame, and the hand target is
        reported in the world (CartesianActionAdapter.set_base)."""
        self.frame = float(angle)
        self.base = np.array([float(base[0]), float(base[1]), 0.0])
        self.adapter.set_frame(self.frame)
        self.adapter.set_base(base)

    # --- measurements --------------------------------------------------------------------
    def _frame_vec(self, v: np.ndarray) -> np.ndarray:
        """A vector drawn in the station frame, in the world (so a turned scene draws the same
        errors, turned)."""
        return _rz(self.frame) @ v

    def handle_xyz(self) -> np.ndarray:
        return self.machine.handle_xyz()

    def bar_yaw(self) -> float:
        """The bar's direction in the world (rad, mod pi)."""
        R = self.data.xmat[self.lid_body].reshape(3, 3)
        bar = R @ self.machine.cfg.bar_axis
        return float(np.arctan2(bar[1], bar[0]))

    def slide_axis(self) -> np.ndarray:
        """The lid's opening direction in the world (privileged: experts and tests only)."""
        return self.data.xmat[self.machine.body].reshape(3, 3) @ self.machine.cfg.slide_axis

    def door_open_switch(self) -> bool:
        """The machine's door-open switch: the lid within sensing.open_switch of its open stop (a
        signal the machine exposes, as the door-closed switch and latched)."""
        return self.machine.lid >= self.machine.cfg.travel - self.config.sensing.open_switch

    def _finger_gap(self) -> float:
        return float(sum(self.data.qpos[a] for a in self._finger_qpos))

    def _tcp_R(self) -> np.ndarray:
        return self.data.site_xmat[self.arm.tcp_site].reshape(3, 3)

    def _finger_contacts(self) -> tuple[bool, bool]:
        found = dict.fromkeys(self._fingers, False)
        d, m = self.data, self.model
        touch = self.touch_geoms
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 in touch:
                other = m.geom_bodyid[c.geom2]
            elif c.geom2 in touch:
                other = m.geom_bodyid[c.geom1]
            else:
                continue
            if other in found:
                found[other] = True
        return found[self._fingers[0]], found[self._fingers[1]]

    def _external_force(self) -> tuple[np.ndarray, bool]:
        """(the summed contact force on the robot from everything else (world, N), whether the
        robot touches the machine's body or the bench)."""
        m, d = self.model, self.data
        total, hit, f6 = np.zeros(3), False, np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            r1, r2 = b1 in self.arm_bodies, b2 in self.arm_bodies
            if r1 == r2:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            on2 = c.frame.reshape(3, 3).T @ f6[:3]          # the force on geom2, world frame
            total += on2 if r2 else -on2
            if (b2 if r1 else b1) in self.machine_bodies:
                hit = True
        return total, hit

    def _look(self) -> None:
        """The camera: re-measure the handle unless the hand hides it."""
        n = self.config.sensing
        true = self.handle_xyz()
        tcp = tcp_position(self.data, self.arm)
        if self._seen_handle is None or np.linalg.norm((true - tcp)[:2]) > n.occlusion_radius:
            noise = self._frame_vec(self.np_random.normal(0.0, n.handle_noise, 3))
            self._seen_handle = true + self._bias + noise
            self._seen_yaw = (self.bar_yaw() + self._yaw_bias
                              + float(self.np_random.normal(0.0, n.bar_yaw_noise)))

    def _bar_align(self) -> float:
        """The fingers' closing direction's error to the (seen) bar's normal, mod pi."""
        return _wrap(self._seen_yaw + np.pi / 2 - closing_yaw(self._tcp_R()), np.pi)

    def _clear_of_bar(self) -> bool:
        """No finger touches the handle bar (open or closed: a hook never closes them)."""
        return not any(self._contacts)

    def _track_touch(self) -> None:
        """Where the hand was when a finger came onto the bar (touch_travel's origin); forgotten
        after sensing.touch_release steps off it."""
        if any(self._contacts):
            self._off_bar = 0
            if self._touch_at is None:
                self._touch_at = tcp_position(self.data, self.arm)
        else:
            self._off_bar += 1
            if self._off_bar >= self.config.sensing.touch_release:
                self._touch_at = None

    def goal_reached(self) -> bool:
        """open: the pocket uncovered, and a springy lid in its detent, so it stays open; close:
        latched."""
        if self.command == "open":
            m = self.machine
            return m.lid >= self.needed and (m.cfg.lid_spring <= 0 or m.held_open)
        return self.machine.latched

    def _open_target(self) -> float:
        """The opening the open command must reach: the needed one, or a springy lid's detent."""
        c = self.machine.cfg
        return max(self.needed, c.travel - c.detent_depth) if c.lid_spring > 0 else self.needed

    def _progress(self) -> float:
        q = self.machine.lid
        if self.command == "open":
            return float(np.clip(q / self._open_target(), 0.0, 1.0))
        return 1.0 if self.machine.latched else float(np.clip(1.0 - q / self.lid_start, 0.0, 1.0))

    def _clearance(self) -> float:
        """TCP height above the bar's centre."""
        return float(tcp_position(self.data, self.arm)[2] - self.handle_xyz()[2])

    def _potentials(self) -> dict:
        tcp = tcp_position(self.data, self.arm)
        goal = self.goal_reached()
        return {
            "goal": goal,
            "distance": float(np.linalg.norm(self.handle_xyz() - tcp)),
            "touch": float(any(self._contacts)),
            "progress": self._progress(),
            "retreat": (min(max(self._clearance(), 0.0), self.config.reward.retreat_cap)
                        if self._clear_of_bar() else 0.0),
            "lid": self.machine.lid,
        }

    # --- step ----------------------------------------------------------------------------
    def _gripper(self, g: float) -> None:
        c = self.config.gripper
        if not self.closed and g > c.close_above:
            self.closed = True
        elif self.closed and g < c.open_below:
            self.closed = False

    def _physics(self, hand: np.ndarray) -> None:
        """One control step: the hand's joint targets, then the physics in chunks, the finger
        target moving toward open or shut at most gripper.speed."""
        c, m, d = self.config.gripper, self.model, self.data
        self.adapter.apply(d, hand)
        goal = 0.0 if self.closed else FINGER_OPEN
        left = self.adapter.n_substeps
        while left > 0:
            n = min(c.chunk, left)
            limit = c.speed * n * m.opt.timestep
            self._finger_target += float(np.clip(goal - self._finger_target, -limit, limit))
            d.ctrl[self.arm.gripper_actuator] = GRIPPER_OPEN * self._finger_target / FINGER_OPEN
            mujoco.mj_step(m, d, nstep=n)
            left -= n

    def step(self, action):
        cfg, r = self.config, self.config.reward
        a, bound_terms = bound_action(action, cfg.action)
        self._gripper(float(a[4]))
        self._physics(a[:4])
        self.machine.update()
        self._steps += 1
        if not (np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))):
            return (np.zeros(self.observation_space.shape, dtype=np.float32), -r.non_finite,
                    True, False,
                    {"is_success": False, "failure": "non_finite", "reward_terms": {},
                     "command": self.command})
        self._contacts = self._finger_contacts()
        self._track_touch()
        force, hit = self._external_force()
        self._force = force
        self._look()
        now, prev = self._potentials(), self._prev
        same_phase = now["goal"] == prev["goal"]
        q, q_prev = now["lid"], prev["lid"]
        dt = self.adapter.control_dt
        travel = self.machine.cfg.travel
        over = -np.inf                     # m/s the lid hits a stop over the allowed speed
        if q >= travel - 0.001 > q_prev:
            over = (q - q_prev) / dt - r.slam_open_speed
        elif q <= 0.001 < q_prev:
            over = (q_prev - q) / dt - r.slam_close_speed
        slam = max(0.0, over + r.slam_margin)   # penalized from slam_margin below the allowed
        f = float(np.linalg.norm(force))
        at_stop = q >= travel - 0.001 or q <= 0.001
        terms = {
            "approach": (r.approach * (prev["distance"] - now["distance"])
                         if same_phase and not now["goal"] else 0.0),
            "touch": (r.touch * (now["touch"] - prev["touch"])
                     if same_phase and not now["goal"] else 0.0),
            "progress": r.progress * (now["progress"] - prev["progress"]),
            "retreat": (r.retreat * (now["retreat"] - prev["retreat"])
                        if same_phase and now["goal"] else 0.0),
            "force": -r.force * max(0.0, f - r.force_limit),
            "collision": -r.collision if hit else 0.0,
            "slam": -r.slam * slam - (r.slam_hit if slam > 0 else 0.0),
            "stop_press": -r.stop_press if at_stop and f > r.stop_force else 0.0,
            "time": -r.time,
            "action": -r.action * float(a[:4] @ a[:4]),
            **bound_terms,
        }
        s = cfg.success
        holding = now["goal"] and self._clear_of_bar() and self._clearance() >= s.clear_height
        self._hold = self._hold + 1 if holding else 0
        success = self._hold >= s.hold_steps
        if success:
            terms["success"] = r.success
        st = self.stats
        st["collision_steps"] += hit
        st["slams"] += over > 0                 # counted at the allowed speed itself
        st["peak_force"] = max(st["peak_force"], f)
        st["touched"] |= any(self._contacts)
        st["goal_reached"] |= now["goal"]
        self._prev = now
        terminated = success
        truncated = not terminated and self._steps >= cfg.horizon
        info = {
            "is_success": success,
            "failure": None if success else self._failure(),
            "command": self.command,
            "lid": q,
            "needed": self.needed,
            "latched": self.machine.latched,
            "goal_reached": now["goal"],
            "touching": any(self._contacts),
            "clear_of_bar": self._clear_of_bar(),
            "clearance": self._clearance(),
            "force": f,
            "machine_contact": hit,
            "distance": now["distance"],
            "stats": dict(st),
            "reward_terms": terms,
        }
        return self._observation(), float(sum(terms.values())), terminated, truncated, info

    def _failure(self) -> str:
        """Why the episode has not succeeded (yet): read at its end."""
        if self.goal_reached():
            return "not_released"
        if not self.stats["touched"]:
            return "never_touched"
        if self.command == "open":
            return "not_held_open" if self.machine.lid >= self.needed else "not_opened"
        if self.machine.lid <= self.machine.cfg.latch_catch:
            return "closed_not_latched"
        return "not_closed"

    def _observation(self) -> np.ndarray:
        d, n = self.data, self.config.sensing
        R = self._tcp_R()
        tcp = tcp_position(d, self.arm)
        e = self._bar_align()
        align = [np.sin(2 * e), np.cos(2 * e)] if n.observe_bar_yaw else [0.0, 0.0]
        travel = tcp - self._touch_at if self._touch_at is not None else np.zeros(3)
        force = np.clip(self._force + self._frame_vec(
            self.np_random.normal(0.0, n.force_noise, 3)), -n.force_clip, n.force_clip)
        return np.concatenate([
            d.qpos[self.arm.arm_qpos],
            d.qvel[self.arm.arm_qvel],
            self.adapter.q_cmd,
            tcp,
            R[:, :2].T.flatten(),
            [self._finger_gap(), float(self.closed)],
            [float(self._contacts[0]), float(self._contacts[1])],
            self._seen_handle - tcp,
            align,
            travel,
            force,
            [float(self.machine.lid <= n.closed_switch), float(self.machine.latched)],
            [float(self.command == "open"), float(self.command == "close")],
        ]).astype(np.float32)


class DoorV2Env(DoorEnv):
    """Door task v2 (schema door-v2-50d, task door_v2): v1 plus the machine's **door-open
    switch** (the lid within sensing.open_switch, 2 mm, of its open stop), as the last input.

    Why (dev round 3, 2026-11-24): v1 observes the door-closed switch and latched but nothing at
    the open end. Round 3's policies let go of the bar while the lid still moved and let it coast
    ("fling"); when a springy lid stopped short of its hold-open detent, its spring pulled it shut
    again, and the policy, which cannot see the lid's displacement once its hand is off the bar,
    never went back (seed 2's final checkpoint: 6 of its 8 failures). Real machines expose a
    door-open switch next to the door-closed one; the switch on means the lid is open enough for
    any variant and, on a springy lid, in its detent."""

    def __init__(self, config: DoorConfig | None = None, scene=None,
                 machine: Machine | None = None) -> None:
        super().__init__(config, scene, machine)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM_V2,),
                                            dtype=np.float32)

    def _observation(self) -> np.ndarray:
        return np.append(super()._observation(),
                         np.float32(self.door_open_switch())).astype(np.float32)
