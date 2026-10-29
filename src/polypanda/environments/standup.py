"""StandUpEnv (v3 step d2): stand a lying cube, held in the hand, upright on the table.

A cube knocked off a tower mostly lies on a side; the any-pose grasp (d1) picks it up gripped
across its 4 cm width, and the cell carries it, hand pointing down, over the spot where it is to
stand. From there this skill turns the hand sideways so the cube's long axis turns vertical,
lowers it, lets go just above the table, and backs the hand away without knocking it over.

Physics as in the workcell, where the skill runs (`gripper`): the cell's grasp assist holds the
cube welded to the hand until the fingers open (the disclosed simplification of
simulation.workcell); the release and the cube settling are physical. A gate opens the fingers
once the cube is upright, low and over the spot, mirroring the grasp's closing gate. The cell's
staging spots lie about 23 cm from a tower, so most episodes put a tower of 1-4 cubes (a fixed
box, observed) at about that distance in a random direction; touching it ends the episode.

Success: the placed check (resting, upright within 10 deg, within check.tolerance of the spot,
still, clear of the gripper) holding at the end of the episode, as for the place skill. The
skill is trained in front of the robot and used at any base angle (supervisor.stations).
"""

from dataclasses import dataclass, field, replace
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from polypanda.config import load_yaml
from polypanda.control.action_adapter import action_limit, bound_action, make_action_adapter
from polypanda.control.ik import PoseIK, top_down_quat
from polypanda.environments.grasp import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    GripperConfig,
    RandomizationConfig,
)
from polypanda.environments.place import PlacedCheck, PlacedCheckConfig
from polypanda.environments.reach import ActionConfig
from polypanda.simulation.assets import panda_spec
from polypanda.simulation.panda import PandaIndices, reset_to_home, tcp_position
from polypanda.simulation.randomization import CubePhysics
from polypanda.simulation.scenes import (
    CubeIndices,
    add_cube,
    cube_position,
    cube_tilt,
    cube_up,
)
from polypanda.simulation.workcell import set_finger_stiffness

OBS_SCHEMA = "standup-v1-52d"
OBS_LAYOUT = (
    ("q", 7), ("dq", 7), ("q_cmd", 7),
    ("tcp", 3), ("tcp_rot6", 6), ("finger_gap", 1), ("released", 1),
    ("cube_rel", 3), ("cube_up", 3), ("cube_height", 1), ("contacts", 2),
    ("target", 3), ("target_error", 3), ("tower_rel", 3), ("tower", 2),
)
OBS_DIM = sum(n for _, n in OBS_LAYOUT)

CUBE_HEIGHT = 0.06       # m, one tower cube
TOWER_HALF = 0.12        # m, half height of the tower box (4 cubes); it sinks into the table
TOWER_PARK = (-0.8, 0.8)  # out of reach, behind the robot
OBSTACLE_RANGE = 0.5      # m from the spot: the bodies the arm keeps clear of in a shared scene


@dataclass(frozen=True)
class StandUpStartConfig:
    radius: tuple[float, float] = (0.32, 0.62)   # m, the spot's distance from the base axis
    angle: tuple[float, float] = (-0.44, 0.44)   # rad (25 deg) about the base axis
    carry_height: float = 0.32                   # m, TCP height (the cell's carry posture)
    grip_offset: float = 0.01                    # m along the cube's long axis, either way
    grip_z: tuple[float, float] = (-0.005, 0.01)  # m, cube centre minus TCP (d1's gate window)
    # Training only: this fraction of episodes start with the hand already turned and low over
    # the spot (reset to helpful states, as the grasp's curriculum starts)
    helpful_prob: float = 0.0
    # Training only (d2 round 4): this fraction of the other episodes start from a state along
    # the scripted stand-up (control.scripted_standup): it acts for demo_steps (uniform, policy
    # steps) from the carry posture, then the policy takes over. Demonstration states as starts
    # only: the policy still learns from reward alone (no imitation loss).
    demo_prob: float = 0.0
    demo_steps: tuple[int, int] = (0, 120)


@dataclass(frozen=True)
class TowerConfig:
    prob: float = 0.8                            # episodes with a tower
    distance: tuple[float, float] = (0.20, 0.26)  # m, spot to tower centre
    min_radius: float = 0.30                     # m, tower centre from the base axis
    max_cubes: int = 4                           # tower height: 1..max_cubes cubes


@dataclass(frozen=True)
class StandUpGateConfig:
    # rad (15 deg): the cube's long axis from vertical. The grasp assist's soft weld lets the
    # cube lag a turning hand (it creeps back over seconds); released 1-2 cm up within 15 deg,
    # it rocks back upright (success still needs it within check.max_tilt, 10 deg)
    max_tilt: float = 0.26
    max_drop: float = 0.025      # m: the cube's bottom at most this far above the table
    xy_tol: float = 0.03         # m: the cube's centre to the spot
    dwell_steps: int = 2


@dataclass(frozen=True)
class StandUpRewardConfig:
    progress: float = 20.0       # per m the cube centre nears the release point (held)
    upright: float = 2.0         # per rad the cube's tilt from vertical shrinks (held)
    release_height: float = 0.012  # m, the release point's cube bottom above the table
    retreat: float = 5.0         # per m the hand moves away from the released cube (capped)
    retreat_cap: float = 0.10
    stood: float = 0.5           # per step while the placed check holds
    # The per-step action costs stay below the terminal penalties over a whole episode, even
    # for the untrained (std 1) policy: with action_rate 0.05 a full episode cost -60..-90 and
    # the first runs learned to end episodes early on the tower (-5; stopped at 0.2M steps)
    action: float = 0.001        # x |a|^2 (random 6-D: about 0.005 per step)
    action_rate: float = 0.002   # x |a - a_prev|^2 (random: about 0.02 per step)
    floor: float = 0.1           # per step with the arm or fingers on the table
    toppled: float = 10.0        # the released cube fell over; terminates
    tower: float = 10.0          # the arm or cube touched the tower; terminates
    # v2 (StandUpV2Env): potential-based clearance shaping, clearance x the change of
    # -max(0, 1 - gap / clearance_margin), gap = the arm's and cube's distance to the tower:
    # closing in costs up to `clearance`, backing off earns it back (it cannot accumulate)
    clearance: float = 0.0
    clearance_margin: float = 0.05  # m
    # v3 (StandUpV3Env): grip strain, per step per rad the grasp assist's weld is stretched
    # beyond strain_from (the cube pressed or dragged against the table: failures peaked at
    # 20-40 deg, successes at 9-17 deg); and per step the gripper touches the released cube
    # (from 3 steps after the release: knocked over or pushed off the spot backing out)
    strain: float = 0.0
    strain_from: float = 0.35    # rad (20 deg)
    touch: float = 0.0
    # v3: per step the cube is still held. Holding still cost nothing, and policies froze in
    # front of a tower or with the cube pressed down off the spot until the episode ran out
    # (never_released 35-89 per 600). Over a whole episode it stays below the tower penalty.
    hold: float = 0.0
    # v3 with the shield (round 10): per step x the fraction of the action the shield cut
    # (|a - a_used| / |a|), so the policy learns paths the shield need not correct instead of
    # pushing against it (round 9's policies were never told and froze in front of the tower)
    shield: float = 0.0


@dataclass(frozen=True)
class TurnCommandConfig:
    """v3 (StandUpV3Env): the direction to turn the hand to is an input of the skill.

    Each round-2..5 policy turned the hand one fixed way (the one reachable from the carry
    posture in 198 of 200 starts) and hit towers behind its wrist; the scripted stand-up, which
    picks among `directions` per start, failed the same way when held to one direction (77.5 %
    against 93 %). In training the command is drawn among the reachable directions, so the
    policy learns to turn every way it can; in use, its critic picks the command
    (evaluation.standup.CommandedPolicyDriver)."""
    directions: int = 16          # candidate directions of the turned hand's z axis (world)
    level: float = 0.5            # |hand z . up| below this: the hand is turned
    tolerance: float = 0.785      # rad: a turned hand's z within this of the command
    align: float = 2.0            # potential-based shaping toward the commanded turn
    wrong_turn: float = 10.0      # turned the wrong way (before the release); terminates
    # The wrong-turn rule is a training device (it makes the policy follow the command), not
    # part of the task: off in evaluation, like the helpful and demonstration starts. Checked
    # at 60 deg of turn it also cut off turns still swinging round to the command (round 6 at
    # 1M steps, 40 fresh starts: 72 % with it, 81 % without).
    enforce: bool = True
    # How training draws the command (d2 round 7). "uniform": among the reachable directions
    # (round 6). "chooser": with probability 1 - explore, uniformly among the `top_k`
    # reachable directions a turn chooser (training.turn_chooser, file `chooser`) rates best,
    # else uniformly. Round 6's policies, trained on all 16 directions alike, carried out a
    # random reachable one 57-72 % of the time: even a perfect choice stayed at 85-92 %.
    sampler: str = "uniform"
    chooser: str | None = None
    top_k: int = 3
    explore: float = 0.2


@dataclass(frozen=True)
class TowerShieldConfig:
    """v3 (d2 round 9): a safety shield between the policy and the arm (StandUpV3Env).

    Rounds 7-8 (85.7 / 83.5 % held-out) failed mostly on tower hits (54 / 57 of 600), largely
    the wrist swinging into the tower while turning; even the best turn direction for a start
    failed 10-15 % of the time. Each step the shield checks the clearance from the tower of the
    arm (and the cube it holds) at the joint targets the action would command, by kinematics; if
    it would
    come within `min_gap` and closer than at the current targets, the step is retried at the
    `scales` of the action, the last of which (0) holds the targets. The policy still decides
    every motion; the shield only refuses ones that would touch the tower (a "safety shield",
    as in provably safe RL). Part of the skill: on in training and in evaluation.

    Round 9 (86.2 % held-out): tower hits 54 -> 3 of 600, but never_released 19 -> 71, the
    robot stopped 3-6 cm short with the cube still tilted; 28 of the 71 starts round 7 had
    stood. Scaling the whole action down stops it dead in front of the tower. With `slide`
    (round 10) a refused step is first retried with the part of the hand's motion that heads
    into the tower removed (along the closest points' normal), at full, half and no turning,
    so the hand keeps moving round and up it; only then the `scales`.

    Round 10 (87.3 %, 0 tower hits): 39 of its 64 freezes were deadlocks, the arm or cube 1.4 cm
    from the tower, where every move that is not a hold came closer, so every step was held from
    about step 73 on, often with the cube upright 3-9 cm off the spot. With `floor` set, a step
    the shield would hold is retried once more, the same candidates accepted down to `floor`."""
    enabled: bool = False
    min_gap: float = 0.015        # m, at the commanded joints (the arm lags them by ~1 cm)
    scales: tuple[float, ...] = (1.0, 0.5, 0.25, 0.0)
    slide: bool = False
    floor: float | None = None    # m, the hard minimum when min_gap would hold the arm


@dataclass(frozen=True)
class ReplanConfig:
    """v3 (d2 round 11): re-plan when the shield holds the arm (StandUpV3Env).

    Round 10's deadlocks (39 of 64 freezes) were the commanded turn's wrist or cube against the
    tower, held from about step 73 on. After `stall` steps the shield held every move, the hand
    lifts and backs away from the tower (`back` steps: up at full speed, away at half; part of
    one environment step, so the policy is not credited with moves it did not choose), and the
    command becomes the one a turn chooser rates best from where the hand now is, among those
    not yet tried, at most `max_replans` times an episode. The chooser is the training command
    chooser (TurnCommandConfig.chooser), or in use the one CommandedPolicyDriver chooses with.
    Practice starts, round 10's policies without retraining (200 x 3, seed 779): 90.2 %
    against 88.0 % (only re-planning 14, only not 1), never a tower hit. Part of the skill: on
    in training and in evaluation."""
    enabled: bool = False
    stall: int = 20               # steps (1 s) with every move held
    back: int = 12                # steps of backing away
    max_replans: int = 4


@dataclass(frozen=True)
class StandUpConfig:
    action: ActionConfig = field(default_factory=ActionConfig)
    start: StandUpStartConfig = field(default_factory=StandUpStartConfig)
    tower: TowerConfig = field(default_factory=TowerConfig)
    gate: StandUpGateConfig = field(default_factory=StandUpGateConfig)
    check: PlacedCheckConfig = field(default_factory=lambda: PlacedCheckConfig(tolerance=0.03))
    reward: StandUpRewardConfig = field(default_factory=StandUpRewardConfig)
    command: TurnCommandConfig = field(default_factory=TurnCommandConfig)
    shield: TowerShieldConfig = field(default_factory=TowerShieldConfig)
    replan: ReplanConfig = field(default_factory=ReplanConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    horizon: int = 300           # policy steps (15 s); no termination on success
    topple_tilt: float = 0.785   # rad (45 deg) after release = toppled

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StandUpConfig":
        return load_yaml(cls, path)

    def for_evaluation(self) -> "StandUpConfig":
        """Every episode starts from the carry posture with the nominal cube; no training
        devices (helpful and demonstration starts, the wrong-turn rule)."""
        return replace(self, start=replace(self.start, helpful_prob=0.0, demo_prob=0.0),
                       randomization=RandomizationConfig(),
                       command=replace(self.command, enforce=False, explore=0.0))


def load_standup_scene(gripper: GripperConfig) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Panda, one cube (weld `assist` to the hand, inactive), and a tower box (parked)."""
    spec = panda_spec()
    add_cube(spec)
    if gripper.finger_stiffness is not None:
        set_finger_stiffness(spec, gripper.finger_stiffness)
    if gripper.torsional_friction:
        spec.geom("cube").condim = 4
    spec.add_equality(name="assist", type=mujoco.mjtEq.mjEQ_WELD,
                      objtype=mujoco.mjtObj.mjOBJ_BODY, name1="hand", name2="cube", active=False)
    tower = spec.worldbody.add_body(name="tower", pos=[*TOWER_PARK, -TOWER_HALF])
    tower.add_geom(name="tower", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[0.02, 0.02, TOWER_HALF], rgba=[0.55, 0.55, 0.6, 1.0])
    model = spec.compile()
    return model, mujoco.MjData(model)


def _hand_R(direction: float) -> np.ndarray:
    """A hand turned sideways: TCP z horizontal at `direction` (rad), x pointing down."""
    z = np.array([np.cos(direction), np.sin(direction), 0.0])
    y = np.cross([0.0, 0.0, 1.0], z)
    return np.column_stack([np.cross(y, z), y, z])


def _quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


class StandUpEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: StandUpConfig | None = None, scene=None, cube: str = "cube"
                 ) -> None:
        """`scene` (model, data): run in an existing scene, e.g. the workcell, on the cube body
        named `cube` (its grasp-assist weld `assist<k>` for `cube<k>`), started with the reset
        option "continue"; by default the env builds its own scene with one tower box."""
        self.config = config or StandUpConfig()
        self.model, self.data = scene or load_standup_scene(self.config.gripper)
        m = self.model
        self.arm = PandaIndices.from_model(m)
        self.cube = CubeIndices.from_model(m, cube)
        self.adapter = make_action_adapter(m, self.arm, self.config.action)
        self.check = PlacedCheck(m, self.arm, self.cube, self.config.check)
        self.assist = m.equality(f"assist{cube[len('cube'):]}").id
        self.hand = m.body("hand").id
        own = scene is None
        self.tower_body = m.body("tower").id if own else None
        self.tower_geom = m.geom("tower").id if own else None
        # the robot's own bodies (in a cell, not the other cubes, the conveyor or the bin)
        self.arm_bodies = {b for b in range(m.nbody) if b and m.body_rootid[b]
                           == m.body_rootid[self.hand]}
        # what the arm and held cube must keep clear of: the tower box, or in a cell every
        # other body near the spot (set at each start)
        self._obstacle_geoms: list[int] = []
        self._obstacle_bodies: set[int] = set()
        self._finger_qpos = [m.joint(f"finger_joint{i}").qposadr[0] for i in (1, 2)]
        self._fingers = (m.body("left_finger").id, m.body("right_finger").id)
        limit = action_limit(self.config.action)
        self.action_space = spaces.Box(-limit, limit, shape=(self.adapter.action_dim,),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self._ik = PoseIK()
        self._physics = None
        self._demo_driver = None
        self.spot = np.zeros(2)
        self.tower = None            # (x, y, height) or None
        self.released = False
        self._steps = self._dwell = 0
        self._prev = {}
        self._prev_action = np.zeros(self.adapter.action_dim)
        self._contacts = (False, False)

    @property
    def elapsed_steps(self) -> int:
        return self._steps

    # --- episodes ------------------------------------------------------------------------
    def draw_episode(self, rng: np.random.Generator, helpful: bool = False) -> dict:
        """A start: the spot, the hand's closing yaw, the grip on the cube, and the tower."""
        s, t = self.config.start, self.config.tower
        r, a = rng.uniform(*s.radius), rng.uniform(*s.angle)
        spot = (r * np.cos(a), r * np.sin(a))
        ep = {"spot": spot, "yaw": float(rng.uniform(-np.pi, np.pi)),
              "offset": float(rng.uniform(-s.grip_offset, s.grip_offset)),
              "grip_z": float(rng.uniform(*s.grip_z)), "tower": None, "helpful": helpful}
        if rng.uniform() < t.prob:
            for _ in range(100):  # gives up quietly (no tower) if none fits
                d, b = rng.uniform(*t.distance), rng.uniform(-np.pi, np.pi)
                xy = np.array(spot) + d * np.array([np.cos(b), np.sin(b)])
                if np.linalg.norm(xy) >= t.min_radius:
                    cubes = int(rng.integers(1, t.max_cubes + 1))
                    ep["tower"] = (float(xy[0]), float(xy[1]), cubes)
                    break
        if helpful:
            ep["turn"] = float(rng.uniform(-np.pi, np.pi))   # the turned hand's z direction
            ep["height"] = float(rng.uniform(0.03, 0.12))   # cube bottom above the table
            ep["shift"] = tuple(rng.uniform(-0.02, 0.02, 2).tolist())
        return ep

    def _start(self, ep: dict) -> bool:
        """Set the scene to episode `ep`. False if the start pose is out of reach."""
        m, d, c = self.model, self.data, self.cube
        reset_to_home(m, d, self.arm)
        tower = ep.get("tower")
        if tower:
            x, y, n = tower
            m.body_pos[self.tower_body] = [x, y, n * CUBE_HEIGHT - TOWER_HALF]
            self.tower = np.array([x, y, n * CUBE_HEIGHT])
        else:
            m.body_pos[self.tower_body] = [*TOWER_PARK, -TOWER_HALF]
            self.tower = None
        self._obstacle_geoms = [self.tower_geom] if tower else []
        self._obstacle_bodies = {self.tower_body} if tower else set()
        self.spot = np.asarray(ep["spot"], dtype=float)
        # The grip d1 leaves: hand pointing down, fingers across the cube's width, the cube's
        # long axis horizontal along the hand's x axis
        R0 = np.zeros(9)
        mujoco.mju_quat2Mat(R0, top_down_quat(ep["yaw"]))
        R0 = R0.reshape(3, 3)
        long_axis = R0[:, 0]
        R_cube = np.column_stack([[0.0, 0.0, 1.0], np.cross(long_axis, [0.0, 0.0, 1.0]),
                                  long_axis])
        p_rel = np.array([ep["offset"], 0.0, 0.0]) + R0.T @ [0.0, 0.0, ep["grip_z"]]
        R_rel = R0.T @ R_cube
        if ep.get("helpful"):
            R = _hand_R(ep["turn"])
            centre = np.array([*(self.spot + ep["shift"]), c.half_size[2] + ep["height"]])
            tcp = centre - R @ p_rel
        else:
            R, tcp = R0, np.array([*self.spot, self.config.start.carry_height])
        q, ok = self._ik.solve(d.qpos[self.arm.arm_qpos], tcp, _quat(R))
        if not ok:
            return False
        d.qpos[self.arm.arm_qpos] = q
        d.ctrl[self.arm.arm_actuators] = q
        for adr in self._finger_qpos:
            d.qpos[adr] = c.half_size[0]                 # finger pads on the cube's faces
        d.ctrl[self.arm.gripper_actuator] = GRIPPER_CLOSED
        mujoco.mj_forward(m, d)
        site = d.site_xpos[self.arm.tcp_site].copy()
        R_site = d.site_xmat[self.arm.tcp_site].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (R_site @ R_rel).flatten())
        d.qpos[c.qpos:c.qpos + 7] = [*(site + R_site @ p_rel), *quat]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        self._attach()
        mujoco.mj_step(m, d, nstep=20)                   # fingers settle on the faces
        return True

    def _attach(self) -> None:
        """Grasp assist, as Workcell.attach: weld the cube to the hand where it is now."""
        m, d = self.model, self.data
        R_hand = d.xmat[self.hand].reshape(3, 3)
        inv, rel_quat = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(inv, d.xquat[self.hand])
        mujoco.mju_mulQuat(rel_quat, inv, d.xquat[self.cube.body])
        m.eq_data[self.assist, :3] = 0.0
        m.eq_data[self.assist, 3:6] = R_hand.T @ (d.xpos[self.cube.body] - d.xpos[self.hand])
        m.eq_data[self.assist, 6:10] = rel_quat
        m.eq_data[self.assist, 10] = 1.0
        d.eq_active[self.assist] = 1

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if options.get("continue"):                      # a skill inside a larger task
            return self._continue(options)
        if "episode" in options:                         # explicit (evaluation): nominal cube
            ep = dict(options["episode"])                # (or its "physics": the robustness
            physics = ep.pop("physics", None)            # suite's mass, friction, size scales)
            if physics is not None or self._physics is not None:
                self._randomize(*(physics or (1.0, 1.0, 1.0)))
            if not self._start(ep):
                raise ValueError(f"start out of reach: {ep}")
            self._begin()
            self._on_start(ep, options)
            start = "helpful" if ep.get("helpful") else "carry"
        else:
            s, rnd = self.config.start, self.config.randomization
            if rnd.enabled:
                lo, hi = np.log(rnd.mass_scale)
                self._randomize(float(np.exp(self.np_random.uniform(lo, hi))),
                                float(self.np_random.uniform(*rnd.friction_scale)))
            helpful = self.np_random.uniform() < s.helpful_prob
            demo = 0  # (no draws when demo starts are off: earlier runs replay unchanged)
            if not helpful and s.demo_prob > 0 and self.np_random.uniform() < s.demo_prob:
                demo = int(self.np_random.integers(s.demo_steps[0], s.demo_steps[1] + 1))
            for _ in range(50):
                ep = self.draw_episode(self.np_random, helpful)
                if not self._start(ep):
                    continue
                self._begin()
                if not self._on_start(ep, None):
                    continue
                if demo and not self._demo_prefix(demo):
                    continue           # the scripted stand-up failed on this one: draw again
                break
            else:
                raise RuntimeError("no reachable start in 50 draws")
            start = "helpful" if helpful else "demo" if demo else "carry"
        return self._observation(), {"episode": ep, "start": start,
                                     "distance": self._prev["distance"]}

    def _continue(self, options: dict):
        """Start from the scene as it is (the cube already held by the grasp assist, the hand
        over the spot), as in the workcell: options "spot" (x, y), "tower" (x, y, top height)
        for the observation or None, and optionally "command". The obstacles are every body
        that is not the robot, the cube or the world within `OBSTACLE_RANGE` of the spot."""
        m, d = self.model, self.data
        if not d.eq_active[self.assist]:
            raise ValueError("continue: the cube is not held by the grasp assist")
        self.spot = np.asarray(options["spot"], dtype=float)
        tower = options.get("tower")
        self.tower = None if tower is None else np.asarray(tower, dtype=float)
        mujoco.mj_forward(m, d)
        here = np.array([*self.spot, 0.0])
        self._obstacle_geoms = [
            g for g in range(m.ngeom)
            if (m.geom_contype[g] or m.geom_conaffinity[g])
            and m.geom_bodyid[g] not in self.arm_bodies
            and m.geom_bodyid[g] not in (0, self.cube.body)
            and np.linalg.norm((d.geom_xpos[g] - here)[:2]) - m.geom_rbound[g] < OBSTACLE_RANGE]
        self._obstacle_bodies = {int(m.geom_bodyid[g]) for g in self._obstacle_geoms}
        self._begin()
        self._on_start({}, options)
        return self._observation(), {"episode": None, "start": "continue",
                                     "distance": self._prev["distance"]}

    def _begin(self) -> None:
        """Start the episode's bookkeeping from the scene as it is now."""
        self.adapter.adopt_ctrl(self.data)
        self.check.reset()
        self.released = False
        self._steps = self._dwell = 0
        self._prev = self._progress_state()
        self._prev_action = np.zeros(self.adapter.action_dim)
        self._contacts = self._finger_contacts()

    def _demo_prefix(self, steps: int) -> bool:
        """Let the scripted stand-up act for `steps` steps, then hand over: the episode (its
        horizon, rewards and checks) starts from where it got to. False if it failed first."""
        from polypanda.control.scripted_standup import ScriptedStandUpDriver  # (imports this)
        if self._demo_driver is None:
            self._demo_driver = ScriptedStandUpDriver()
        self._demo_driver.reset(self, direction=self._demo_direction())
        for _ in range(steps):
            _, _, terminated, _, _ = self.step(self._demo_driver(self, None))
            if terminated:
                return False
        released, dwell = self.released, self._dwell
        self.check.reset()
        self._steps = 0
        self._prev = {**self._progress_state(), "released": released}
        self._prev_action = np.zeros(self.adapter.action_dim)
        self.released, self._dwell = released, dwell
        return True

    def _actuate(self, a: np.ndarray) -> None:
        """Command the arm with action `a` and advance the physics one control step."""
        self.adapter.step(self.data, a)

    def _on_start(self, ep: dict, options: dict | None) -> bool:
        """Hook once a start is set (options: the evaluation's, None in training). False: the
        start does not suit, draw another (training only)."""
        return True

    def _demo_direction(self) -> float | None:
        """The direction the demonstration turns the hand to (None: the scripted plan's own)."""
        return None

    def _randomize(self, mass: float, friction: float, size: float = 1.0) -> None:
        if self._physics is None:
            self._physics = CubePhysics(self.model, self.cube)
        self._physics.apply(mass_scale=mass, friction_scale=friction, size_scale=size)

    # --- measurements --------------------------------------------------------------------
    def _release_point(self) -> np.ndarray:
        r = self.config.reward
        return np.array([*self.spot, self.cube.half_size[2] + r.release_height])

    def _progress_state(self) -> dict:
        cube = cube_position(self.data, self.cube)
        away = np.linalg.norm(tcp_position(self.data, self.arm) - cube)
        return {"distance": float(np.linalg.norm(self._release_point() - cube)),
                "tilt": cube_tilt(self.data, self.cube),
                "away": float(min(away, self.config.reward.retreat_cap))}

    def _finger_contacts(self) -> tuple[bool, bool]:
        found = dict.fromkeys(self._fingers, False)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            b = (self.model.geom_bodyid[con.geom1], self.model.geom_bodyid[con.geom2])
            if self.cube.body in b:
                other = b[1] if b[0] == self.cube.body else b[0]
                if other in found:
                    found[other] = True
        return found[self._fingers[0]], found[self._fingers[1]]

    def _contact_flags(self) -> tuple[bool, bool]:
        """(arm or fingers on the table, the arm or the cube on the tower or another obstacle)."""
        floor = tower = False
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            b = {self.model.geom_bodyid[con.geom1], self.model.geom_bodyid[con.geom2]}
            if b & self._obstacle_bodies and (b & self.arm_bodies or self.cube.body in b):
                tower = True
            elif 0 in b and b & self.arm_bodies:
                floor = True
        return floor, tower

    def step(self, action):
        cfg, r = self.config, self.config.reward
        a, bound_terms = bound_action(action, cfg.action)
        self._actuate(a)
        self._steps += 1
        if not (np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))):
            return (np.zeros(OBS_DIM, dtype=np.float32), -r.toppled, True, False,
                    {"is_success": False, "failure": "non_finite", "reward_terms": {}})
        cube = cube_position(self.data, self.cube)
        tilt = cube_tilt(self.data, self.cube)
        xy_err = float(np.linalg.norm(cube[:2] - self.spot))
        if not self.released:
            g = cfg.gate
            bottom = cube[2] - self.cube.half_size[2]
            ready = tilt < g.max_tilt and -0.003 < bottom < g.max_drop and xy_err < g.xy_tol
            self._dwell = self._dwell + 1 if ready else 0
            if self._dwell >= g.dwell_steps:
                self.released = True
                self.data.eq_active[self.assist] = 0
                self.data.ctrl[self.arm.gripper_actuator] = GRIPPER_OPEN
        self._contacts = self._finger_contacts()
        floor, tower = self._contact_flags()
        toppled = self.released and tilt > cfg.topple_tilt
        stood = self.check.update(self.data, self.spot, float(self.cube.half_size[2]))
        now = self._progress_state()
        held = not self.released
        terms = {
            "progress": r.progress * (self._prev["distance"] - now["distance"]) if held else 0.0,
            "upright": r.upright * (self._prev["tilt"] - now["tilt"]) if held else 0.0,
            "retreat": 0.0 if held else r.retreat * (now["away"] - self._prev["away"]),
            "stood": r.stood if stood else 0.0,
            "action": -r.action * float(a @ a),
            "action_rate": -r.action_rate * float(np.sum((a - self._prev_action) ** 2)),
            "floor": -r.floor if floor else 0.0,
            "toppled": -r.toppled if toppled else 0.0,
            "tower": -r.tower if tower else 0.0,
            **bound_terms,
        }
        if self.released and not self._prev.get("released"):
            terms["retreat"] = 0.0       # (the hand's distance before release is not progress)
        self._prev, self._prev_action = {**now, "released": self.released}, a
        terminated = bool(toppled or tower)
        truncated = not terminated and self._steps >= cfg.horizon
        info = {
            "is_success": stood,
            "failure": "toppled" if toppled else "tower_hit" if tower else None,
            "released": self.released,
            "distance": now["distance"],
            "xy_error": xy_err,
            "cube_tilt": tilt,
            "floor_contact": floor,
            "gripper_contact": self.check.gripper_contact(self.data),
            "reward_terms": terms,
        }
        return self._observation(), float(sum(terms.values())), terminated, truncated, info

    def _observation(self) -> np.ndarray:
        d = self.data
        R = d.site_xmat[self.arm.tcp_site].reshape(3, 3)
        cube = cube_position(d, self.cube)
        tcp = tcp_position(d, self.arm)
        target = np.array([*self.spot, self.cube.half_size[2]])
        if self.tower is None:
            tower_rel, tower = np.zeros(3), [0.0, 0.0]
        else:
            tower_rel, tower = self.tower - tcp, [1.0, float(self.tower[2])]
        return np.concatenate([
            d.qpos[self.arm.arm_qpos],
            d.qvel[self.arm.arm_qvel],
            self.adapter.q_cmd,
            tcp,
            R[:, :2].T.flatten(),
            [sum(d.qpos[adr] for adr in self._finger_qpos), float(self.released)],
            cube - tcp,
            cube_up(d, self.cube),
            [cube[2] - self.cube.half_size[2]],
            [float(self._contacts[0]), float(self._contacts[1])],
            target,
            target - cube,
            tower_rel,
            tower,
        ]).astype(np.float32)


OBS_SCHEMA_V2 = "standup-v2-56d"   # v1 + the tower clearance (distance, and its direction)
OBS_LAYOUT_V2 = OBS_LAYOUT + (("tower_gap", 1), ("tower_away", 3))
OBS_DIM_V2 = sum(n for _, n in OBS_LAYOUT_V2)
GAP_CAP = 0.2   # m: clearances beyond this read as this (and with no tower)


class StandUpV2Env(StandUpEnv):
    """Stand-up v2 (d2 round 3): v1 plus a sense of the tower before touching it.

    v1 policies stood every cube up with no tower near (12/12) but hit the tower in a quarter
    of the episodes with one, mostly mid-turn: they saw only where the tower's top is, and
    the tower penalty came with the contact. v2 observes the clearance, the smallest distance
    from the arm or the cube to the tower (capped at GAP_CAP), and the direction away from it
    (tower point to arm point), as a proximity sensor would, and adds reward.clearance shaping.
    """

    def __init__(self, config: StandUpConfig | None = None, scene=None, cube: str = "cube"
                 ) -> None:
        super().__init__(config, scene, cube)
        m = self.model
        self._gap_geoms = [g for g in range(m.ngeom)
                           if (m.geom_bodyid[g] in self.arm_bodies
                               and (m.geom_contype[g] or m.geom_conaffinity[g]))
                           or g == self.cube.geom]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM_V2,),
                                            dtype=np.float32)
        self._fromto = np.zeros(6)
        self._clear = 0.0

    def _tower_gap(self) -> tuple[float, np.ndarray]:
        """The arm's and cube's clearance from the tower (in a shared scene, the nearest
        obstacle) (m) and the vector from the tower's nearest point to theirs."""
        best, away = GAP_CAP, np.zeros(3)
        for o in self._obstacle_geoms:
            for g in self._gap_geoms:
                dist = mujoco.mj_geomDistance(self.model, self.data, g, o, GAP_CAP, self._fromto)
                if dist < best:
                    best, away = dist, self._fromto[:3] - self._fromto[3:]
        return max(float(best), 0.0), away

    def _clearance_potential(self, gap: float) -> float:
        r = self.config.reward
        return -max(0.0, 1.0 - gap / r.clearance_margin)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._clear = self._clearance_potential(self._tower_gap()[0])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if info.get("failure") == "non_finite":
            return np.zeros(OBS_DIM_V2, dtype=np.float32), reward, terminated, truncated, info
        clear = self._clearance_potential(self._tower_gap()[0])
        term = self.config.reward.clearance * (clear - self._clear)
        self._clear = clear
        info["reward_terms"]["clearance"] = term
        return obs, reward + term, terminated, truncated, info

    def _observation(self) -> np.ndarray:
        gap, away = self._tower_gap()
        return np.concatenate([super()._observation(), [gap], away]).astype(np.float32)


OBS_SCHEMA_V3 = "standup-v3-58d"   # v2 + the commanded turn direction
OBS_LAYOUT_V3 = OBS_LAYOUT_V2 + (("command", 2),)
OBS_DIM_V3 = sum(n for _, n in OBS_LAYOUT_V3)


class StandUpV3Env(StandUpV2Env):
    """Stand-up v3 (d2 round 6): the direction to turn the hand to is commanded.

    Rounds 2-5 learned one turn direction each and hit towers behind the wrist (root-cause
    analysis, 2026-10-28). Here the policy observes a command, the world direction its turned
    hand's z axis is to point (cos, sin), and learns to turn every reachable way:
    - training draws the command among the `command.directions` the arm can reach from the
      start (the scripted planner's IK; no tower check), so it also learns which ones fail;
    - turning more than `command.tolerance` away from it ends the episode (wrong_turn);
    - potential-based shaping toward the commanded turn (`command.align`);
    - demonstration starts turn the commanded way.
    In use, evaluation.standup.CommandedPolicyDriver sets the command its critic rates best.
    Also the robot-safety terms `reward.strain`, `reward.touch` and `reward.hold`.
    """

    def __init__(self, config: StandUpConfig | None = None, scene=None, cube: str = "cube"
                 ) -> None:
        super().__init__(config, scene, cube)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM_V3,),
                                            dtype=np.float32)
        self.command = 0.0            # rad, world direction of the turned hand's z axis
        self._align = 0.0
        self._release_step = None
        self._planner = None
        self._chooser = None

    # --- commands ------------------------------------------------------------------------
    def _plan_driver(self):
        if self._planner is None:
            from polypanda.control.scripted_standup import ScriptedStandUpDriver
            self._planner = ScriptedStandUpDriver()
            self._planner.directions = self.config.command.directions
        return self._planner

    def reachable_directions(self, rng: np.random.Generator | None = None,
                             first: bool = False) -> list[float]:
        """The candidate directions whose turn and descent the arm can reach from here (all,
        or with `first` the first found, in the order `rng` shuffles them)."""
        planner = self._plan_driver()
        cands = list(planner.candidates())
        if rng is not None:
            rng.shuffle(cands)
        found = []
        for d in cands:
            if planner.plan(self, float(d))[0] is not None:
                found.append(float(d))
                if first:
                    break
        return found

    def set_command(self, direction: float) -> None:
        self.command = float(direction)
        self._align = self._align_potential()

    def _command_vec(self) -> np.ndarray:
        return np.array([np.cos(self.command), np.sin(self.command)])

    def _on_start(self, ep: dict, options: dict | None) -> bool:
        if options is not None and "command" in options:
            self.set_command(options["command"])
        elif ep.get("helpful"):
            self.set_command(ep["turn"])       # already turned that way
        elif options is not None:              # evaluation without one: the first reachable
            found = self.reachable_directions(first=True)
            a = float(np.arctan2(self.spot[1], self.spot[0]))
            self.set_command(found[0] if found else a + np.pi / 2)
        else:
            d = self._draw_command()
            if d is None:
                return False
            self.set_command(d)
        return True

    def _draw_command(self) -> float | None:
        """A training command (TurnCommandConfig.sampler); None if none is reachable."""
        c = self.config.command
        if c.sampler == "chooser" and self.np_random.uniform() >= c.explore:
            if self._chooser is None:
                from polypanda.training.turn_chooser import TurnChooser
                self._chooser = TurnChooser.load(c.chooser)
            planner = self._plan_driver()
            cands = [float(d) for d in planner.candidates()]
            obs = []
            for d in cands:
                self.set_command(d)
                obs.append(self.view_observation())
            top = []
            for i in np.argsort(-self._chooser.scores(np.stack(obs))):
                if planner.plan(self, cands[i])[0] is not None:
                    top.append(cands[i])
                    if len(top) == c.top_k:
                        break
            if top:
                return top[int(self.np_random.integers(len(top)))]
        elif c.sampler not in ("uniform", "chooser"):
            raise ValueError(f"unknown command sampler {c.sampler!r}")
        found = self.reachable_directions(self.np_random, first=True)
        return found[0] if found else None

    def _demo_direction(self) -> float | None:
        return self.command

    # --- re-planning (ReplanConfig) --------------------------------------------------------
    def _replan_command(self) -> float | None:
        """The untried command the turn chooser rates best from the current state (the first
        untried candidate without a chooser); None when every candidate was tried."""
        cands = [float(d) for d in self._plan_driver().candidates()
                 if all(abs(float(d) - t) > 1e-6 for t in self._tried)]
        if not cands:
            return None
        if self._chooser is None and self.config.command.chooser:
            from polypanda.training.turn_chooser import TurnChooser
            self._chooser = TurnChooser.load(self.config.command.chooser)
        if self._chooser is None:
            return cands[0]
        current, obs = self.command, []
        for d in cands:
            self.set_command(d)
            obs.append(self.view_observation())
        self.set_command(current)
        return cands[int(np.argmax(self._chooser.scores(np.stack(obs))))]

    def _replan(self) -> None:
        """Back away from the tower, then command the chooser's best untried direction."""
        rp = self.config.replan
        self._arm_gap_at(self.data.qpos[self.arm.arm_qpos])
        n = self._shield_normal
        h = np.zeros(3) if n is None else np.array([n[0], n[1], 0.0])
        h = self.adapter._frame_rotations()[0] @ h          # actions are in the station frame
        h /= max(float(np.linalg.norm(h)), 1e-9)
        away = np.array([0.5 * h[0], 0.5 * h[1], 1.0, 0.0, 0.0, 0.0])
        for _ in range(min(rp.back, self.config.horizon - self._steps)):
            self._actuate(away)
            self._steps += 1
        d = self._replan_command()
        if d is not None:
            self.set_command(d)
        # the shaping restarts from here: the backing away was not the policy's move
        self._prev = {**self._progress_state(), "released": self.released}
        self._prev_action = away
        self._clear = self._clearance_potential(self._tower_gap()[0])
        self._held, self._replans = 0, self._replans + 1

    # --- measurements --------------------------------------------------------------------
    def _hand_z(self) -> np.ndarray:
        return self.data.site_xmat[self.arm.tcp_site].reshape(3, 3)[:, 2]

    def _align_potential(self) -> float:
        """align x (hand z . command): 0 pointing down, 1 turned the commanded way."""
        return self.config.command.align * float(self._hand_z()[:2] @ self._command_vec())

    def _command_error(self) -> float:
        h = self._hand_z()[:2]
        n = np.linalg.norm(h)
        return float(np.arccos(np.clip(h @ self._command_vec() / n, -1, 1))) if n > 0 else 0.0

    def _weld_lag(self) -> float:
        """Rad between the cube's orientation and where the grasp assist's weld holds it."""
        target, weld = np.zeros(4), self.model.eq_data[self.assist, 6:10]
        mujoco.mju_mulQuat(target, self.data.xquat[self.hand], weld)
        c = abs(float(target @ self.data.xquat[self.cube.body]))
        return 2.0 * float(np.arccos(min(1.0, c)))

    # --- the safety shield ---------------------------------------------------------------
    def _arm_gap_at(self, q: np.ndarray) -> float:
        """The clearance from the tower (m) of the arm with its joints at q, and of the cube
        while held, where the grasp assist's weld carries it (kinematics only). Also sets
        `_shield_normal`, the unit world vector from the tower's closest point to the arm's
        (None if nothing is within GAP_CAP)."""
        m, c = self.model, self.cube
        if not hasattr(self, "_shield_data"):
            self._shield_data = mujoco.MjData(m)
        s = self._shield_data
        s.qpos[:] = self.data.qpos
        s.qpos[self.arm.arm_qpos] = q
        mujoco.mj_kinematics(m, s)
        geoms = self._gap_geoms
        if self.data.eq_active[self.assist]:      # held: the cube goes where the hand takes it
            R = s.xmat[self.hand].reshape(3, 3)
            quat = np.zeros(4)
            mujoco.mju_mulQuat(quat, s.xquat[self.hand], m.eq_data[self.assist, 6:10])
            s.qpos[c.qpos:c.qpos + 3] = s.xpos[self.hand] + R @ m.eq_data[self.assist, 3:6]
            s.qpos[c.qpos + 3:c.qpos + 7] = quat
            mujoco.mj_kinematics(m, s)
        else:
            geoms = [g for g in geoms if g != c.geom]
        gap, self._shield_normal = GAP_CAP, None
        for o in self._obstacle_geoms:
            for g in geoms:
                d = mujoco.mj_geomDistance(m, s, g, o, GAP_CAP, self._fromto)
                if d < gap:
                    v = (self._fromto[:3] - self._fromto[3:]) * (1.0 if d >= 0 else -1.0)
                    n = float(np.linalg.norm(v))
                    gap, self._shield_normal = d, (v / n if n > 1e-9 else None)
        return float(gap)

    def _shield_candidates(self, a: np.ndarray) -> list[np.ndarray]:
        """The actions the shield tries in turn (TowerShieldConfig)."""
        sh, out = self.config.shield, [a]
        n = self._shield_normal
        if sh.slide and n is not None:
            n = self.adapter._frame_rotations()[0] @ n     # actions are in the station frame
            t = np.asarray(a[:3], dtype=np.float64).copy()
            t -= min(0.0, float(t @ n)) * n                # drop the part heading in
            for turn in (1.0, 0.5, 0.0):
                b = np.asarray(a, dtype=np.float64).copy()
                b[:3], b[3:] = t, b[3:] * turn
                out.append(b)
        return out + [a * s for s in sh.scales if s != 1.0]

    def _actuate(self, a: np.ndarray) -> None:
        sh = self.config.shield
        if not sh.enabled or not self._obstacle_geoms:
            super()._actuate(a)
            self._shield_scale = 1.0
            return
        ad = self.adapter
        saved = (ad.q_cmd.copy(), ad.target_pos.copy(), ad.target_R.copy(),
                 ad._velocity_step.copy())
        ref = self._arm_gap_at(saved[0])
        a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)

        def attempt(b: np.ndarray) -> float:
            ad.q_cmd, ad.target_pos, ad.target_R, ad._velocity_step = (
                saved[0].copy(), saved[1].copy(), saved[2].copy(), saved[3].copy())
            return self._arm_gap_at(ad.apply(self.data, b))

        cands = self._shield_candidates(a)
        moves = [b for b in cands if np.any(b)]
        hold = np.zeros_like(a)
        b = next((b for b in moves if (g := attempt(b)) >= sh.min_gap or g >= ref - 1e-4), None)
        if b is None and sh.floor is not None:
            b = next((b for b in moves if attempt(b) >= sh.floor), None)
        if b is None:
            b = hold
            attempt(b)
        size, cut = float(np.linalg.norm(a)), float(np.linalg.norm(a - b))
        self._shield_scale = 1.0 - min(1.0, cut / size) if size > 0 else 1.0  # 1: untouched
        mujoco.mj_step(self.model, self.data, nstep=ad.n_substeps)

    # --- episode -------------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # a demonstration prefix steps inside reset: no re-planning during it
        self._tried, self._held, self._replans, self._resetting = [], 0, 0, True
        obs, info = super().reset(seed=seed, options=options)
        self._align = self._align_potential()
        self._release_step = 0 if self.released else None
        self._tried, self._held, self._replans, self._resetting = [], 0, 0, False
        info["command"] = self.command
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if info.get("failure") == "non_finite":
            return np.zeros(OBS_DIM_V3, dtype=np.float32), reward, terminated, truncated, info
        c, r = self.config.command, self.config.reward
        terms = info["reward_terms"]
        new = {"align": 0.0, "strain": 0.0, "touch": 0.0, "hold": 0.0, "wrong_turn": 0.0,
               "shield": -r.shield * (1.0 - getattr(self, "_shield_scale", 1.0))}
        if not self.released:
            phi = self._align_potential()
            new["align"], self._align = phi - self._align, phi
            new["strain"] = -r.strain * max(0.0, self._weld_lag() - r.strain_from)
            new["hold"] = -r.hold
            # (after a re-plan the hand is still turned the old way: no wrong-turn rule)
            if (c.enforce and not terminated and not self._replans
                    and abs(self._hand_z()[2]) < c.level and self._command_error() > c.tolerance):
                new["wrong_turn"] = -c.wrong_turn
                terminated, truncated = True, False
                info["failure"] = "wrong_turn"
        else:
            if self._release_step is None:
                self._release_step = self._steps
            if self._steps - self._release_step >= 3 and info["gripper_contact"]:
                new["touch"] = -r.touch
        terms.update(new)
        info["shield_scale"] = getattr(self, "_shield_scale", 1.0)
        rp = self.config.replan
        if (rp.enabled and self._obstacle_geoms and not self._resetting
                and not (self.released or terminated or truncated)):
            self._held = self._held + 1 if info["shield_scale"] < 0.05 else 0
            if self._held >= rp.stall and self._replans < rp.max_replans:
                self._tried.append(self.command)
                self._replan()
                obs = self._observation()
                truncated = self._steps >= self.config.horizon
        info["replans"] = self._replans
        info["command"] = self.command
        return obs, reward + sum(new.values()), terminated, truncated, info

    def _observation(self) -> np.ndarray:
        return np.concatenate([super()._observation(), self._command_vec()]).astype(np.float32)

    # the observation as the policy sees it: at a station of the workcell, StationView sets
    # this to its station-frame transform (so a turn chooser scores what the policy sees)
    obs_transform = None

    def view_observation(self) -> np.ndarray:
        o = self._observation()
        return o if self.obs_transform is None else self.obs_transform(o)
