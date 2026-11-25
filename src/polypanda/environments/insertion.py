"""InsertionEnv (v4.1): seat a held keyed module in a tight test nest (docs/v4_1_targets.md, Part I;
docs/v4_1_plan.md section 4.2). The learned precision insertion's environment; docs/insertion_env.md
has the design, the reward and how episodes are recorded.

**Scene.** The test cell's robot on its rail with the cell's three testers at their stops (A, a
lid, at -0.9 m; B, a drawer, at -0.45 m; C, a pneumatic lid, at 0 m; simulation/test_cell.py's
`planned_testers`: tight, keyed nests with stiff contacts, `machine.tight_nest`), the infeed
conveyor at stop 0 (a module is picked from it to draw its place in the hand), and one keyed module
with the cell's stiff contacts; no stations, dressing or hall (a bare scene, for speed). The
fingers never collide with each other, as in the test cell.

**An episode** (`draw_episode`, replayed exactly with reset option `episode`):
- the tester (A, B or C: its rail stop and its nest's height, a drawer's on its tray 1 cm up),
  drawn uniformly; where it stands (+-1.5 cm on each axis of its station frame) and how it is
  turned (+-10 deg), as v4.0 drew its machines;
- its nest: clearance per side uniform 0.25-0.5 mm, chamfer uniform 0.5-1.0 mm
  (simulation/machine_variation.NestLayout re-cuts the compiled nest, no recompile);
- the camera's error on the nest: position normal sigma 0.8 mm per horizontal axis, clipped at
  3 mm (radially), yaw normal sigma 0.5 deg clipped at 2 deg;
- the module's place in the hand **as a real pick leaves it**: the module arrives on the infeed
  conveyor with the cell's placement noise (+-3 mm across the belt, +-5 deg; the test cell's
  run_shift) and the cell's controller picks it for the tester (ScriptedTending.pick_for, the
  keyed grasp). The grasp (the grasp assist's weld, i.e. the module's pose in the hand, and the
  fingers) is stored in the episode (`grasp`), so an episode replays without the pick. Without
  `grasp`, reset simulates the pick first (`simulate_pick`); training can draw grasps from a pool
  of such picks instead (`InsertionConfig.pick_pool`, scripts/make_insertion_sets.py --picks).

**The start** (the same for the policy and both scripts): the robot at the tester's rail stop in
the cell's carry posture facing it, the module held, the tester open (a lid or pneumatic lid slid
open, a drawer pulled out and held at its stop by its index pin); the tester measured by the
camera (the episode's error) and the insertion aimed as the cell's insertion slot aims it
(control/test_cell.aim: over the believed nest at the carry height, the wrist turned so the
module's key meets the believed slot). Time and force are counted from there.

**Observation** (OBS_LAYOUT, schema insertion-v1-42d), in the **believed nest frame** (origin
the camera's estimate of the nest's centre on its floor, axes along the believed nest):
- the arm: joints q, velocities dq, commanded targets q_cmd;
- the hand: TCP position and rotation (6-D: the first two columns), the commanded hand target
  and its yaw relative to the aimed yaw;
- the wrist force-torque sensor: force (N) and torque (N m, about the module's origin) on the
  held module from everything but the fingers (control/scripted_insertion.part_wrench, what the
  search script reads), with small sensor noise;
- whether the module has been released, and the elapsed share of the time budget.
Not observed: the nest's true pose, the module's place in the hand, the clearance, the chamfer.

**Action** (5-D, ACTION_SCHEMA): a hand motion in the believed nest frame, dx, dy, dz (up to
`max_step`, 10 mm per step) and a turn of the wrist dyaw (up to `max_rot`, 0.05 rad), integrated
into a persistent hand target (the target stays within `max_lead` of the measured hand and within
`max_turn` of the aimed yaw), solved to the arm's joint targets by the cell's own IK
(ScriptedCell._solve); and **release** (above `release_above`: the grasp assist lets go and the
fingers open; it cannot be undone). The cell's 20 Hz: one action per 50 ms (25 x 2 ms). This is
the form the scripts' own commands take (a hand target and yaw per control step, the fingers
opened once), so their runs convert to (observation, action) demonstrations
(evaluation/insertion.record_script).

**Success** (docs/v4_1_targets.md): the module seated (its bottom within 1 mm of the nest's
floor, tilted under 2 deg, inside the clearance plus 0.2 mm, its key in the slot), released, the
hand back at the carry height (within 1 cm) and clear of the module, and **the contact force on
the nest never above 50 N** (the largest normal force of a contact between the module and the
tester or the nest's floor, checked at every 2 ms physics step), all within the 30 s budget (600
steps).

**Reward** (InsertionRewardConfig; terms in info["reward_terms"]): see docs/insertion_env.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from polypanda.config import load_yaml
from polypanda.control.ik import closing_yaw, top_down_quat
from polypanda.control.scripted_tending import Belief, ScriptedTending
from polypanda.environments.grasp import GRIPPER_CLOSED, GRIPPER_OPEN
from polypanda.simulation.line import Line, LineConfig, build_line_spec
from polypanda.simulation.machine import Machine
from polypanda.simulation.machine_variation import MachineLayout, NestLayout
from polypanda.simulation.rail import RailConfig, RailIndices
from polypanda.simulation.scenes import CubeIndices
from polypanda.simulation.test_cell import PLANNED_KINDS, planned_testers
from polypanda.simulation.workcell import (
    CUBE_NAME,
    Conveyor,
    WorkcellConfig,
    WorkOrder,
    _yaw_quat,
    belt_direction,
    queue_xy,
)

OBS_LAYOUT = (
    ("q", 7), ("dq", 7), ("q_cmd", 7),
    ("tcp", 3), ("tcp_rot6", 6), ("target", 3), ("target_yaw", 1),
    ("force", 3), ("torque", 3), ("released", 1), ("time", 1),
)
OBS_DIM = sum(n for _, n in OBS_LAYOUT)
OBS_SCHEMA = f"insertion-v1-{OBS_DIM}d"
OBS_SLICES = {
    name: slice(start, start + n)
    for (name, n), start in zip(
        OBS_LAYOUT, np.cumsum([0] + [n for _, n in OBS_LAYOUT[:-1]]), strict=True)
}
ACTION_SCHEMA = "insertion-v1-5d"     # dx, dy, dz, dyaw in the believed nest frame; release
TESTERS = tuple(PLANNED_KINDS)        # ("TA", "TB", "TC"): lid, drawer, pneumatic
SUBSTEPS = 25                         # physics steps per control step (the cell's 20 Hz)
PART = CUBE_NAME.format(0)
FAILURES = ("damage", "not_seated", "not_released", "not_clear", "non_finite")


# --- configuration ------------------------------------------------------------------------------
@dataclass(frozen=True)
class InsertionVariationConfig:
    """The episode distribution of docs/v4_1_targets.md (training and the sets alike)."""
    testers: tuple[str, ...] = TESTERS          # drawn uniformly
    clearance: tuple[float, float] = (0.00025, 0.0005)   # m per side, uniform
    chamfer: tuple[float, float] = (0.0005, 0.001)       # m, uniform
    position: float = 0.015         # m either way, each axis of the station frame (v4.0's)
    turn: float = 10.0              # deg either way (v4.0's)
    camera_xy: float = 0.0008       # m, sigma per horizontal axis
    camera_xy_clip: float = 0.003   # m, radially
    camera_yaw: float = 0.5         # deg, sigma
    camera_yaw_clip: float = 2.0    # deg
    pick_lateral: float = 0.003     # m either way across the belt (the test cell's run_shift)
    pick_yaw: float = 5.0           # deg either way


@dataclass(frozen=True)
class InsertionActionConfig:
    max_step: float = 0.01          # m per step per axis (0.2 m/s; the cell's vertical moves peak
    #                                 at 7.9 mm per step)
    max_rot: float = 0.05           # rad per step (the search's wiggle needs 0.046)
    max_lead: float = 0.03          # m: the hand target within this of the measured hand, per axis
    max_turn: float = 0.35          # rad: the hand's yaw within this of the aimed yaw
    release_above: float = 0.5      # release action above this lets go


@dataclass(frozen=True)
class InsertionSensingConfig:
    force_noise: float = 0.1        # N per axis per step (the wrist force-torque sensor)
    torque_noise: float = 0.002     # N m per axis per step


@dataclass(frozen=True)
class InsertionSuccessConfig:
    floor_gap: float = 0.001        # m: the module's bottom within this of the nest's floor
    tilt: float = 2.0               # deg
    slack: float = 0.0002           # m beyond the clearance (stiff contacts' depth; the probe's)
    force_limit: float = 50.0       # N: the damage limit (docs/v4_1_targets.md)
    carry_margin: float = 0.01      # m: the hand back within this of the carry height


@dataclass(frozen=True)
class InsertionRewardConfig:
    insert_z: float = 10.0          # per m the held module's bottom nears the nest's floor
    insert_xy: float = 100.0        # per m its centre nears the nest's centre (capped at 1 cm)
    seat: float = 3.0               # once, the first time the module is seated
    release_seated: float = 1.0     # once, released while seated
    release_unseated: float = 2.0   # once, released while not seated (a penalty)
    retreat: float = 5.0            # per m the hand rises after the release (to the carry height)
    success: float = 10.0           # once, at success (the episode ends)
    force: float = 0.05             # per N of nest force above force_soft, per step
    force_soft: float = 20.0        # N
    damage: float = 10.0            # once, the nest force above the damage limit (ends it)
    failure: float = 5.0            # once, released and the hand up but not seated (ends it)
    collision: float = 0.05         # per step the arm (not the module) touches the tester
    time: float = 0.01              # per step
    action: float = 0.001           # x |hand action|^2
    non_finite: float = 10.0


@dataclass(frozen=True)
class InsertionStartConfig:
    carry_height: float = 0.32      # m, TCP (the cell's carry posture)
    settle_steps: int = 10          # control steps held still after the module is put in the hand


@dataclass(frozen=True)
class InsertionConfig:
    variation: InsertionVariationConfig = field(default_factory=InsertionVariationConfig)
    action: InsertionActionConfig = field(default_factory=InsertionActionConfig)
    sensing: InsertionSensingConfig = field(default_factory=InsertionSensingConfig)
    success: InsertionSuccessConfig = field(default_factory=InsertionSuccessConfig)
    reward: InsertionRewardConfig = field(default_factory=InsertionRewardConfig)
    start: InsertionStartConfig = field(default_factory=InsertionStartConfig)
    horizon: int = 600              # policy steps (30 s at 20 Hz: the search's own budget)
    # training only: draw the module's grasp from this pool of real picks (a JSON file written by
    # scripts/make_insertion_sets.py --picks, from a stream apart from the evaluation sets');
    # None: simulate the episode's pick at reset
    pick_pool: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> InsertionConfig:
        return load_yaml(cls, path)


STRETCH = {"camera_xy": 0.0015, "camera_xy_clip": 0.004, "camera_yaw": 1.0,
           "camera_yaw_clip": 3.0}   # insertion_stretch_v1 (docs/v4_1_targets.md)


# --- the scene ----------------------------------------------------------------------------------
def scene_config() -> LineConfig:
    """The bare test cell: robot, rail, conveyor, the three planned testers, one keyed module."""
    return LineConfig(cell=WorkcellConfig(tower_angles=(), n_cubes=1, dressing=False),
                      rail=RailConfig(low=-0.95, high=0.1, visual=False),
                      machines=planned_testers(), keyed_parts=True, part_contact="stiff")


def build_scene() -> Line:
    cfg = scene_config()
    spec = build_line_spec(cfg)
    # as in the test cell (simulation/test_cell.build_test_cell_spec): the fingers never collide
    spec.add_exclude(bodyname1="left_finger", bodyname2="right_finger")
    model = spec.compile()
    line = Line(model, mujoco.MjData(model), cfg.cell)
    line.line = cfg
    line.cubes = [CubeIndices.from_model(model, PART)]
    line.conveyor = Conveyor(line)
    line.rail = RailIndices.from_model(model)
    line.spots = {s.name: s for s in cfg.machines}
    line.machines = {s.name: Machine(model, line.data, s.name, s.config, line.cubes)
                     for s in cfg.machines}
    line.reset_line(_order())
    return line


def _order() -> WorkOrder:
    return WorkOrder(recipes=(), arrivals=(("red", None),))


class InsertionController(ScriptedTending):
    """The cell's scripted controller (ScriptedTending: the pick, the keyed turn, the scripts'
    moves), with hooks for the env: `physics_hook` after every 2 ms physics step (the nest-force
    watch), `pre_hook` before and `control_hook` after every 50 ms control step, and the last
    hand command it solved (`last_command`: world position, quaternion). The physics are
    ScriptedCell.step's, one physics step at a time (mj_step 25 times is mj_step(nstep=25))."""

    def __init__(self, line: Line) -> None:
        super().__init__(line)
        self.on_step = None                 # the testers' cycles play no part here
        self.physics_hook = self.pre_hook = self.control_hook = None
        self.last_command: tuple[np.ndarray, np.ndarray] | None = None

    def _solve(self, pos, quat, seed) -> np.ndarray:
        self.last_command = (np.array(pos, dtype=float), np.array(quat, dtype=float))
        return super()._solve(pos, quat, seed)

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            if self.pre_hook is not None:
                self.pre_hook()
            self.d.ctrl[self.arm.arm_actuators] = self.q_cmd
            self.d.ctrl[self.arm.gripper_actuator] = self.grip
            self.cell.conveyor.update()
            for _ in range(SUBSTEPS):
                mujoco.mj_step(self.m, self.d)
                if self.physics_hook is not None:
                    self.physics_hook()
            if self.control_hook is not None:
                self.control_hook()


def _rz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _quat_mat(quat) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, dtype=float))
    return R.reshape(3, 3)


def load_pick_pool(path: str | Path) -> dict[str, list[dict]]:
    """A pool of real picks (scripts/make_insertion_sets.py --picks): {tester: [pick, ...]}."""
    data = json.loads(Path(path).read_text())
    pool: dict[str, list[dict]] = {}
    for entry in data["picks"]:
        pool.setdefault(entry["tester"], []).append(entry)
    return pool


# --- the environment ----------------------------------------------------------------------------
class InsertionEnv(gym.Env):
    metadata = {"render_modes": []}
    ACTION_SCHEMA = ACTION_SCHEMA

    def __init__(self, config: InsertionConfig | None = None) -> None:
        self.config = config or InsertionConfig()
        for t in self.config.variation.testers:
            if t not in TESTERS:
                raise ValueError(f"testers must be some of {TESTERS}, not {t!r}")
        self.line = build_scene()
        self.model, self.data = self.line.model, self.line.data
        m = self.model
        self.part = self.line.cubes[0]
        self.ctl = InsertionController(self.line)
        self.arm = self.ctl.arm
        self.dt = SUBSTEPS * m.opt.timestep
        self.spots = self.line.spots
        self.layouts = {n: MachineLayout(m, n, s.config) for n, s in self.spots.items()}
        self.nests = {n: NestLayout(m, n, s.config) for n, s in self.spots.items()}
        # what touches what: the module's collision geoms; each tester's (its body and every body
        # on it: lid or drawer, nest walls) and, for a nest standing on the bench, the bench
        collide = (m.geom_contype > 0) | (m.geom_conaffinity > 0)
        self._is_part = np.zeros(m.ngeom, bool)
        self._is_part[(m.geom_bodyid == self.part.body) & collide] = True
        floor = m.geom("floor").id
        self._nest_geoms, self._tester_bodies = {}, {}
        for name, spot in self.spots.items():
            root = m.body(name).id
            bodies = {b for b in range(m.nbody) if m.body_rootid[b] == root}
            mask = np.isin(m.geom_bodyid, list(bodies)) & collide
            if spot.config.nest_floor == 0.0:
                mask[floor] = True
            self._nest_geoms[name] = mask
            self._tester_bodies[name] = bodies | {0}
        hand_root = m.body_rootid[m.body("hand").id]
        self._arm_bodies = {b for b in range(m.nbody) if b and m.body_rootid[b] == hand_root}
        self._fingers = {m.body("left_finger").id, m.body("right_finger").id}
        self._finger_qpos = [int(m.joint(f"finger_joint{k}").qposadr[0]) for k in (1, 2)]
        self._weld = m.equality("assist0").id
        self.pool = (load_pick_pool(self.config.pick_pool) if self.config.pick_pool else None)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.episode: dict | None = None
        self.tester = TESTERS[0]
        self.recorder = None
        self._f6 = np.zeros(6)
        self._pyramidal = m.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL

    # --- episodes -----------------------------------------------------------------------------
    def draw_episode(self, rng: np.random.Generator, tester: str | None = None) -> dict:
        """One episode (JSON-able): the tester and where it stands, its nest, the camera's
        error and the infeed's placement of the module (its grasp: from the pick pool if the
        config has one, else simulated at reset, or by `complete_episode`)."""
        v = self.config.variation

        def sym(x):
            return float(rng.uniform(-x, x))

        drawn = str(v.testers[int(rng.integers(len(v.testers)))])
        tester = tester or drawn
        ep = {
            "tester": tester, "kind": PLANNED_KINDS[tester],
            "clearance": float(rng.uniform(*v.clearance)),
            "chamfer": float(rng.uniform(*v.chamfer)),
            "offset": [sym(v.position), sym(v.position)], "turn": sym(v.turn),
        }
        e = rng.normal(0.0, v.camera_xy, 2)
        r = float(np.linalg.norm(e))
        if r > v.camera_xy_clip:
            e *= v.camera_xy_clip / r
        yaw = float(np.clip(rng.normal(0.0, v.camera_yaw), -v.camera_yaw_clip, v.camera_yaw_clip))
        ep["camera_xy"] = [float(e[0]), float(e[1])]       # m, in the station frame
        ep["camera_yaw"] = float(np.radians(yaw))           # rad
        ep["pick"] = {"lateral": sym(v.pick_lateral), "yaw": float(np.radians(sym(v.pick_yaw)))}
        if self.pool is not None:
            entries = self.pool[tester]
            entry = entries[int(rng.integers(len(entries)))]
            ep["pick"], ep["grasp"] = dict(entry["pick"]), dict(entry["grasp"])
        return ep

    def complete_episode(self, ep: dict) -> dict:
        """The episode with its grasp (simulating its pick if it has none)."""
        if "grasp" in ep:
            return ep
        return {**ep, "grasp": self.simulate_pick(ep)}

    def tester_config(self, ep: dict):
        """The episode's tester as built: the compiled tester with the episode's nest."""
        base = self.spots[ep["tester"]].config
        return replace(base, clearance=ep["clearance"],
                       pocket_straight=base.pocket_depth - ep["chamfer"])

    def _layout(self, ep: dict) -> None:
        """Every tester where the cell has it, with its compiled nest, except the episode's:
        moved and turned as drawn, with the drawn nest."""
        m = self.model
        line_cfg = self.line.line
        for name, spot in self.spots.items():
            xy = line_cfg.station_xy(spot.stop, spot.angle)
            yaw = np.radians(spot.angle)
            cfg = spot.config
            if name == ep["tester"]:
                xy = xy + (_rz(yaw) @ [*ep["offset"], 0.0])[:2]
                yaw += np.radians(ep["turn"])
                cfg = self.tester_config(ep)
            self.layouts[name].place(m, xy, yaw)
            self.nests[name].apply(m, cfg, self.line.machines[name])

    def _station(self, tester: str) -> tuple[float, float]:
        spot = self.spots[tester]
        return float(spot.stop), float(np.radians(spot.angle))

    def camera_belief(self, ep: dict) -> Belief:
        """What the camera measures of the episode's tester (its true nest, with the episode's
        error), and the tester as the cell has it programmed (its nominal nest)."""
        name = ep["tester"]
        m = self.line.machines[name]
        R = self.data.xmat[m.body].reshape(3, 3)
        _, angle = self._station(name)
        err = _rz(angle) @ [*ep["camera_xy"], 0.0]
        return Belief(origin=m.pocket_xyz() + err,
                      yaw=float(np.arctan2(R[1, 0], R[0, 0])) + ep["camera_yaw"],
                      config=self.spots[name].config)

    def simulate_pick(self, ep: dict) -> dict:
        """The cell's pick of the episode's module for its tester: the module arrives on the
        infeed conveyor with the episode's placement, the robot (at rail stop 0, facing the
        infeed) picks it with ScriptedTending.pick_for. Returns the grasp: the grasp assist's
        weld (the module's pose in the hand frame) and the fingers' positions."""
        line, d, ctl = self.line, self.data, self.ctl
        self._detach_hooks()
        line.reset_line(_order())
        self._layout(ep)
        mujoco.mj_forward(self.model, d)
        cfg = line.config
        travel = belt_direction(cfg)
        normal = np.array([-travel[1], travel[0]])
        heading = float(np.arctan2(travel[1], travel[0]))
        start = queue_xy(cfg)[0]
        p = self.part
        d.qpos[p.qpos:p.qpos + 7] = [*(start + ep["pick"]["lateral"] * normal),
                                     p.half_size[2] + cfg.belt_lift,
                                     *_yaw_quat(heading + ep["pick"]["yaw"])]
        d.qvel[p.qvel:p.qvel + 6] = 0.0
        mujoco.mj_forward(self.model, d)
        ctl.q_cmd, ctl.grip = ctl.home.copy(), GRIPPER_OPEN
        ctl.beliefs[ep["tester"]] = self.camera_belief(ep)
        line.conveyor.index()
        ctl.to_carry()
        ctl.swing_to(np.radians(cfg.infeed_angle))
        for _ in range(int(30.0 / ctl.dt)):
            if line.conveyor.ready:
                break
            ctl.step()
        if not line.conveyor.ready:
            raise RuntimeError("the conveyor did not bring the module")
        ctl.pick_for(p, ep["tester"])
        if not line.attached(p):
            raise RuntimeError("the pick did not hold the module")
        eq = self.model.eq_data[self._weld]
        return {"rel_pos": [float(x) for x in eq[3:6]],
                "rel_quat": [float(x) for x in eq[6:10]],
                "fingers": [float(d.qpos[a]) for a in self._finger_qpos]}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if "episode" in options:
            ep, how = dict(options["episode"]), "explicit"
        else:
            ep, how = self.draw_episode(self.np_random, options.get("tester")), "draw"
        if ep["tester"] not in self.spots:
            raise ValueError(f"unknown tester {ep['tester']!r}")
        ep = self.complete_episode(ep)
        self._start(ep)
        self.episode = ep
        return self.observation(), {"episode": ep, "start": how, "tester": self.tester}

    def _start(self, ep: dict) -> None:
        """The robot at the tester's stop in the carry posture facing it, the module in the hand
        as its pick left it, the tester open; then measured by the camera and aimed (`aim`)."""
        from polypanda.control.test_cell import aim
        m, d, line, ctl = self.model, self.data, self.line, self.ctl
        self._detach_hooks()
        self.tester = name = ep["tester"]
        line.reset_line(_order())
        self._layout(ep)
        stop, angle = self._station(name)
        d.qpos[line.rail.joint_qpos] = stop
        d.ctrl[line.rail.actuator] = stop
        q = ctl.carry.copy()
        q[0] += angle
        d.qpos[self.arm.arm_qpos] = q
        g = ep["grasp"]
        for adr, x in zip(self._finger_qpos, g["fingers"], strict=True):
            d.qpos[adr] = x
        for other, mach in line.machines.items():       # the episode's tester open, others shut
            open_ = mach.cfg.travel if other == name else 0.0
            d.qpos[mach.door_qpos] = open_
        d.qvel[:] = 0.0
        mujoco.mj_kinematics(m, d)
        hand = m.body("hand").id
        rel_pos, rel_quat = np.asarray(g["rel_pos"]), np.asarray(g["rel_quat"])
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, d.xquat[hand], rel_quat)
        p = self.part
        d.qpos[p.qpos:p.qpos + 7] = [*(d.xpos[hand] + d.xmat[hand].reshape(3, 3) @ rel_pos),
                                     *quat]
        m.eq_data[self._weld, :3] = 0.0
        m.eq_data[self._weld, 3:6] = rel_pos
        m.eq_data[self._weld, 6:10] = rel_quat
        m.eq_data[self._weld, 10] = 1.0
        d.eq_active[self._weld] = 1
        mujoco.mj_forward(m, d)
        for mach in line.machines.values():
            mach.reset()
        if line.spots[name].kind == "drawer":
            line.machines[name].index_open()
        ctl.q_cmd, ctl.grip = q.copy(), GRIPPER_CLOSED
        d.ctrl[self.arm.arm_actuators] = q
        d.ctrl[self.arm.gripper_actuator] = GRIPPER_CLOSED
        ctl.ik_failures = 0
        ctl.step(self.config.start.settle_steps)
        self.belief = self.camera_belief(ep)
        ctl.beliefs = {name: self.belief}
        ctl.last_command = None
        self.aim_xy, self.aim_yaw = aim(ctl, p, name)
        if ctl.last_command is None:
            raise RuntimeError("aim did not command the hand")
        self.target = ctl.last_command[0].copy()
        self.target_yaw = float(self.aim_yaw)
        # the believed nest frame: the camera's nest centre on the nest's (programmed) floor
        self.nest_origin = np.array([*self.belief.origin[:2], self.belief.config.nest_floor])
        self.nest_R = _rz(self.belief.yaw)
        self.t0 = float(d.time)
        self.steps = 0
        self.released = False
        self.ever_seated = False
        self.stats = {"peak_force": 0.0, "peak_force_ctl": 0.0, "peak_force_total": 0.0,
                      "collision_steps": 0, "lead_clips": 0, "turn_clips": 0}
        self._step_peak = 0.0
        self._last_force = 0.0
        self._damaged = False
        self._attach_hooks()
        self._prev = self._potentials()

    def _attach_hooks(self) -> None:
        self.ctl.physics_hook = self._watch
        self.ctl.control_hook = self._after_control_step
        self.ctl.pre_hook = self._before_control_step

    def _detach_hooks(self) -> None:
        self.ctl.physics_hook = self.ctl.control_hook = self.ctl.pre_hook = None

    # --- measurements -------------------------------------------------------------------------
    def _nest_contacts(self) -> np.ndarray:
        """The indices of the contacts between the module and the tester or the nest's floor."""
        d = self.data
        if d.ncon == 0:
            return np.zeros(0, dtype=int)
        g = d.contact.geom
        nest = self._nest_geoms[self.tester]
        return np.flatnonzero((self._is_part[g[:, 0]] & nest[g[:, 1]])
                              | (self._is_part[g[:, 1]] & nest[g[:, 0]]))

    def nest_force(self) -> float:
        """The **nest force** now: the largest normal force of a contact between the module and
        the tester (its nest's walls and chamfers, a drawer's tray, the tester's body) or, for a
        nest standing on the bench, the bench under it (N). The measure the cell and the probe
        report (control/test_cell._contact_force, scripts/probe_nest_insertion.ContactMonitor),
        with the nest's floor included on every tester. (With pyramidal friction cones, the
        model's, a contact's normal force is the sum of its pyramid's edge forces, as
        mj_contactForce decodes it; tests check they agree.)"""
        hit = self._nest_contacts()
        if not len(hit):
            return 0.0
        d = self.data
        if self._pyramidal:
            con = d.contact
            adr, dim = con.efc_address[hit], con.dim[hit]
            adr, dim = adr[adr >= 0], dim[adr >= 0]
            if not len(adr):
                return 0.0
            edges = np.where(dim > 1, 2 * (dim - 1), 1)
            if np.all(edges == edges[0]):
                rows = adr[:, None] + np.arange(edges[0])
                return float(np.abs(d.efc_force[rows].sum(axis=1)).max())
        peak = 0.0
        for i in hit:
            mujoco.mj_contactForce(self.model, d, int(i), self._f6)
            peak = max(peak, abs(float(self._f6[0])))
        return peak

    def nest_force_total(self) -> float:
        """The magnitude of the summed contact force between the module and the nest (N)."""
        d, total = self.data, np.zeros(3)
        for i in self._nest_contacts():
            mujoco.mj_contactForce(self.model, d, int(i), self._f6)
            c = d.contact[int(i)]
            on2 = c.frame.reshape(3, 3).T @ self._f6[:3]
            total += on2 if self._is_part[c.geom2] else -on2
        return float(np.linalg.norm(total))

    def _watch(self) -> None:
        """After every physics step: the nest force."""
        f = self.nest_force()
        self._last_force = f
        if f > self._step_peak:
            self._step_peak = f
            if f > self.stats["peak_force"]:
                self.stats["peak_force"] = f
                if f > self.config.success.force_limit:
                    self._damaged = True

    def _before_control_step(self) -> None:
        if self.recorder is not None:
            self.recorder.before_step(self)

    def _after_control_step(self) -> None:
        self.steps += 1
        st = self.stats
        st["peak_force_ctl"] = max(st["peak_force_ctl"], self._last_force)
        if self._last_force > 0.0:
            st["peak_force_total"] = max(st["peak_force_total"], self.nest_force_total())
        st["collision_steps"] += self.arm_touches_tester()
        if not self.line.attached(self.part):
            self.released = True
        if self.recorder is not None:
            self.recorder.after_step(self)

    def arm_touches_tester(self) -> bool:
        """The arm (not the module) touches the tester or the bench."""
        m, d = self.model, self.data
        bodies = self._tester_bodies[self.tester]
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (b1 in self._arm_bodies and b2 in bodies) or (b2 in self._arm_bodies
                                                             and b1 in bodies):
                return True
        return False

    def fingers_touch_part(self) -> bool:
        m, d = self.model, self.data
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (b1 == self.part.body and b2 in self._fingers) or (
                    b2 == self.part.body and b1 in self._fingers):
                return True
        return False

    def nest_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """The true nest: its centre on its floor (world) and its rotation."""
        site = self.line.machines[self.tester].pocket_site
        return (self.data.site_xpos[site].copy(),
                self.data.site_xmat[site].reshape(3, 3).copy())

    def seat(self) -> dict:
        """The module in the true nest's frame: centre offset (m), gap under it (m), tilt (deg),
        key error (deg), and whether it is seated."""
        d, p, s = self.data, self.part, self.config.success
        origin, R = self.nest_frame()
        rel = R.T @ (d.xpos[p.body] - origin)
        Rp = d.xmat[p.body].reshape(3, 3)
        tilt = float(np.degrees(np.arccos(np.clip(Rp[2, 2], -1.0, 1.0))))
        gap = float(rel[2] - p.half_size[2])
        mach = self.line.machines[self.tester]
        offset = float(np.abs(rel[:2]).max())
        key = mach.key_error(p)
        seated = (gap < s.floor_gap and tilt < s.tilt and offset <= mach.cfg.clearance + s.slack
                  and (key is None or abs(key) <= mach.cfg.key_tolerance))
        return {"offset": offset, "xy": rel[:2], "gap": gap, "tilt": tilt,
                "key_error": key, "seated": bool(seated)}

    def hand_up(self) -> bool:
        tcp = self.data.site_xpos[self.arm.tcp_site]
        return bool(tcp[2] >= self.config.start.carry_height - self.config.success.carry_margin)

    def outcome(self) -> dict:
        """The insertion judged now: success, or why not (FAILURES)."""
        s = self.seat()
        finite = bool(np.all(np.isfinite(self.data.qpos)))
        released = self.released and not self.line.attached(self.part)
        clear = released and self.hand_up() and not self.fingers_touch_part()
        damaged = self.stats["peak_force"] > self.config.success.force_limit
        success = finite and s["seated"] and clear and not damaged
        failure = (None if success else "non_finite" if not finite else "damage" if damaged
                   else "not_released" if not released else "not_seated" if not s["seated"]
                   else "not_clear")
        seat = {"offset_mm": round(s["offset"] * 1000, 4), "gap_mm": round(s["gap"] * 1000, 4),
                "tilt_deg": round(s["tilt"], 4),
                "key_error_deg": None if s["key_error"] is None else round(s["key_error"], 3)}
        return {"success": bool(success), "failure": failure, "seated": s["seated"],
                "released": bool(released), "hand_up": self.hand_up(),
                "time_s": round(float(self.data.time - self.t0), 3), "steps": self.steps,
                "seat": seat,
                **{k: (round(float(v), 3) if isinstance(v, float) else int(v))
                   for k, v in self.stats.items()}}

    # --- observation ---------------------------------------------------------------------------
    def to_nest(self, v: np.ndarray) -> np.ndarray:
        """A world vector in the believed nest frame."""
        return self.nest_R.T @ v

    def wrench(self) -> np.ndarray:
        """The wrist force-torque sensor: control/scripted_insertion.part_wrench (the force and
        the torque about the module's origin on the module, from everything but the robot,
        world frame), computed the same way with fewer array allocations (tests check they
        agree)."""
        m, d, p = self.model, self.data, self.part.body
        n = d.ncon
        out = np.zeros(6)
        if n == 0:
            return out
        b1, b2 = m.geom_bodyid[d.contact.geom1], m.geom_bodyid[d.contact.geom2]
        hit = np.flatnonzero((b1 == p) | (b2 == p))
        if not len(hit):
            return out
        ox, oy, oz = d.xpos[p]
        for i in hit:
            other = b2[i] if b1[i] == p else b1[i]
            if other in self._arm_bodies:
                continue                                     # the fingers holding it
            mujoco.mj_contactForce(m, d, int(i), self._f6)
            c = d.contact[int(i)]
            f = c.frame.reshape(3, 3).T @ self._f6[:3]
            if b2[i] != p:
                f = -f
            rx, ry, rz = c.pos[0] - ox, c.pos[1] - oy, c.pos[2] - oz
            out[0] += f[0]
            out[1] += f[1]
            out[2] += f[2]
            out[3] += ry * f[2] - rz * f[1]
            out[4] += rz * f[0] - rx * f[2]
            out[5] += rx * f[1] - ry * f[0]
        return out

    def observation(self) -> np.ndarray:
        d, n = self.data, self.config.sensing
        tcp = d.site_xpos[self.arm.tcp_site]
        R = self.nest_R.T @ d.site_xmat[self.arm.tcp_site].reshape(3, 3)
        # (a wrist sensor feels the module only while the hand holds it)
        w = np.zeros(6) if self.released else self.wrench()
        rng = self.np_random
        force = self.to_nest(w[:3]) + rng.normal(0.0, n.force_noise, 3)
        torque = self.to_nest(w[3:]) + rng.normal(0.0, n.torque_noise, 3)
        return np.concatenate([
            d.qpos[self.arm.arm_qpos], d.qvel[self.arm.arm_qvel], self.ctl.q_cmd,
            self.to_nest(tcp - self.nest_origin), R[:, :2].T.flatten(),
            self.to_nest(self.target - self.nest_origin), [self.target_yaw - self.aim_yaw],
            force, torque, [float(self.released)],
            [self.steps / self.config.horizon],
        ]).astype(np.float32)

    # --- a script's commands as actions (evaluation/insertion.record_script) --------------------
    def action_from_command(self) -> np.ndarray:
        """The action that takes the hand target from where it is to the controller's last
        command (the position and yaw a script solved for this control step), and releases if
        the script has just let go."""
        c = self.config.action
        pos, yaw = self.target, self.target_yaw
        if self.ctl.last_command is not None:
            pos = self.ctl.last_command[0]
            seen = closing_yaw(_quat_mat(self.ctl.last_command[1]))
            yaw = self.target_yaw + _wrap(seen - self.target_yaw)
        a = np.zeros(5)
        a[:3] = self.to_nest(pos - self.target) / c.max_step
        a[3] = (yaw - self.target_yaw) / c.max_rot
        a[4] = 1.0 if (not self.released and not self.line.attached(self.part)) else -1.0
        self._command = (np.array(pos, dtype=float), float(yaw))
        return a

    def adopt_command(self) -> None:
        """The hand target becomes the script's command (after action_from_command)."""
        self.target, self.target_yaw = self._command[0].copy(), self._command[1]

    # --- step ----------------------------------------------------------------------------------
    def command(self, action) -> None:
        """Apply one action to the hand target and the grip (no physics)."""
        c, ctl = self.config.action, self.ctl
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if a[4] > c.release_above and not self.released:
            self.line.detach(self.part)
            ctl.grip = GRIPPER_OPEN
        target = self.target + self.nest_R @ (c.max_step * a[:3])
        tcp = self.data.site_xpos[self.arm.tcp_site]
        bounded = np.clip(target, tcp - c.max_lead, tcp + c.max_lead)
        self.stats["lead_clips"] += bool(np.any(bounded != target))
        yaw = self.target_yaw + c.max_rot * a[3]
        turned = float(np.clip(yaw, self.aim_yaw - c.max_turn, self.aim_yaw + c.max_turn))
        self.stats["turn_clips"] += turned != yaw
        self.target, self.target_yaw = bounded, turned
        ctl.q_cmd = ctl._solve(self.target, top_down_quat(self.target_yaw), ctl.q_cmd)

    def step(self, action):
        r = self.config.reward
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        released_before = self.released
        self._step_peak = 0.0
        self.command(a)
        self.ctl.step()
        if not (np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))):
            info = {"is_success": False, "failure": "non_finite", "reward_terms": {},
                    "stats": dict(self.stats), "outcome": None}
            return (np.zeros(self.observation_space.shape, np.float32), -r.non_finite, True,
                    False, info)
        now, prev = self._potentials(), self._prev
        held = not now["released"] and not prev["released"]
        after = now["released"] and prev["released"]
        terms = {
            "insert": (r.insert_z * (prev["gap"] - now["gap"])
                       + r.insert_xy * (prev["xy"] - now["xy"])) if held else 0.0,
            "seat": r.seat if now["seated"] and not self.ever_seated else 0.0,
            "release": ((r.release_seated if now["seated"] else -r.release_unseated)
                        if now["released"] and not released_before else 0.0),
            "retreat": r.retreat * (now["height"] - prev["height"]) if after else 0.0,
            "force": -r.force * max(0.0, self._step_peak - r.force_soft),
            "collision": -r.collision if self.arm_touches_tester() else 0.0,
            "time": -r.time,
            "action": -r.action * float(a[:4] @ a[:4]),
        }
        self.ever_seated |= now["seated"]
        self._prev = now
        out = None
        terminated = False
        if self._damaged:
            terms["damage"] = -r.damage
            terminated = True
        elif now["released"] and self.hand_up():
            out = self.outcome()
            if out["success"]:
                terms["success"] = r.success
                terminated = True
            elif out["failure"] == "not_seated":
                terms["failure"] = -r.failure
                terminated = True
        truncated = not terminated and self.steps >= self.config.horizon
        if out is None and (terminated or truncated):
            out = self.outcome()
        info = {
            "is_success": bool(out is not None and out["success"]),
            "failure": None if out is None else out["failure"],
            "outcome": out, "seated": now["seated"], "released": now["released"],
            "nest_force": self._step_peak, "stats": dict(self.stats), "reward_terms": terms,
        }
        return self.observation(), float(sum(terms.values())), terminated, truncated, info

    def _potentials(self) -> dict:
        d, p = self.data, self.part
        origin, R = self.nest_frame()
        rel = R.T @ (d.xpos[p.body] - origin)
        s = self.seat()
        return {"released": self.released, "seated": s["seated"],
                "gap": float(np.clip(rel[2] - p.half_size[2], 0.0, 0.3)),
                "xy": float(min(np.linalg.norm(rel[:2]), 0.01)),
                "height": float(min(d.site_xpos[self.arm.tcp_site][2],
                                    self.config.start.carry_height))}
