"""GraspEnv: pick up a cube with seven joint-increment actions and a gated scripted gripper.

The policy moves the arm for the whole task (approach, align, descend, lift). The fingers close
by a fixed gate when the TCP is centred on the cube, aligned with a pair of faces and close to
vertical; they reopen if they close on nothing, allowing a retry. Episodes always run the full
horizon (lesson from reach v1). Success: the verified grasp predicate holds with the cube at
least `success.lift` above the table. Reward terms are returned in info["reward_terms"].
"""

from dataclasses import dataclass, field, replace
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from polypanda.config import load_yaml
from polypanda.control.action_adapter import (
    action_limit,
    bound_action,
    make_action_adapter,
)
from polypanda.control.ik import PoseIK, closing_yaw, grasp_closing_yaw, top_down_quat
from polypanda.environments.reach import ActionConfig
from polypanda.simulation.grasp import GraspCheckConfig, GraspMonitor
from polypanda.simulation.panda import PandaIndices, reset_to_home, tcp_position
from polypanda.simulation.randomization import CubePhysics
from polypanda.simulation.scenes import (
    CUBE_X_RANGE,
    CUBE_Y_RANGE,
    CubeIndices,
    cube_position,
    cube_tilt,
    cube_yaw,
    load_pick_scene,
    set_cube_pose,
)

GRIPPER_OPEN, GRIPPER_CLOSED = 255.0, 0.0

OBS_SCHEMA = "grasp-v1-41d"
OBS_SCHEMA_V2 = "grasp-v2-41d"  # same layout; the alignment slots hold sin/cos(2 x error)
OBS_LAYOUT = (
    ("q", 7), ("dq", 7), ("q_cmd", 7),
    ("tcp", 3), ("tcp_rot6", 6), ("finger_gap", 1), ("gripper_closed", 1),
    ("cube_rel", 3), ("align_sin4", 1), ("align_cos4", 1), ("cube_height", 1),
    ("cube_tilt", 1), ("contacts", 2),
)
OBS_DIM = sum(n for _, n in OBS_LAYOUT)
OBS_SLICES = {
    name: slice(start, start + n)
    for (name, n), start in zip(
        OBS_LAYOUT, np.cumsum([0] + [n for _, n in OBS_LAYOUT[:-1]]), strict=True)
}


@dataclass(frozen=True)
class PlacementConfig:
    x_range: tuple[float, float] = CUBE_X_RANGE
    y_range: tuple[float, float] = CUBE_Y_RANGE


@dataclass(frozen=True)
class GateConfig:
    xy_tol: float = 0.01      # m, TCP to cube centre in the horizontal plane
    z_tol: float = 0.01       # m, TCP height to cube centre height
    align_tol: float = 0.15   # rad (~9 deg) to the nearest cube face pair
    tilt_tol: float = 0.15    # rad, gripper z axis from straight down
    dwell_steps: int = 2      # consecutive steps meeting all conditions before closing
    reopen_gap: float = 0.02  # m: fingers closer than this have closed on nothing
    reopen_steps: int = 10    # consecutive steps below reopen_gap before reopening (0.5 s)


@dataclass(frozen=True)
class GraspEpisodeConfig:
    horizon: int = 300            # policy steps (15 s); no termination on success
    max_cube_tilt: float = 1.047  # rad (60 deg): cube knocked over, episode terminates
    max_push: float = 0.0         # v2: m the cube may be pushed before the grasp (0 = no limit)


@dataclass(frozen=True)
class GraspSuccessConfig:
    lift: float = 0.05  # m above resting height, with the verified grasp predicate


@dataclass(frozen=True)
class GraspRewardConfig:
    approach: float = 20.0     # per metre of TCP-to-cube distance reduced
    align: float = 1.0         # per radian of face-alignment error reduced
    lift: float = 20.0         # per metre of cube height gained (capped; falls give it back)
    lift_cap: float = 0.10     # m
    hold: float = 0.5          # per step while grasped and lifted >= success.lift
    action: float = 0.005      # times ||a||^2
    gripper_tilt: float = 0.1  # per radian of gripper tilt, per step
    floor: float = 0.1         # per step with the arm or fingers touching the floor
    knocked_over: float = 5.0  # once, when the cube tips past max_cube_tilt
    push: float = 0.0          # v2: per metre the cube moves horizontally with the fingers open
    pushed_away: float = 0.0   # v2: once, when the cube is pushed beyond episode.max_push


@dataclass(frozen=True)
class CurriculumConfig:
    # Training-only "reset to helpful states": this fraction of randomly placed episodes starts
    # with the gripper already closed around the cube at its grasp pose, so the policy
    # experiences lift and hold rewards it would otherwise rarely discover. Never applied to
    # resets with an explicit placement (evaluation) or to validation (for_evaluation()).
    pregrasp_start_prob: float = 0.0
    # v4 (GraspV4Env) only, training only: this fraction of randomly placed episodes starting
    # from home instead start with the open hand hovering 2-5 cm above the cube, up to 1 cm off
    # its centre in x and y and 25 deg off the grip direction, so the policy often meets the
    # last steps (turn, descend, close) it otherwise rarely reaches for a lying cube.
    preclose_start_prob: float = 0.0


@dataclass(frozen=True)
class RandomizationConfig:
    # Training-only domain randomization of the cube, drawn per randomly placed episode: mass
    # log-uniform and contact friction uniform in these scale ranges (simulation.randomization).
    # Like the curriculum, never applied to explicit placements or to validation.
    mass_scale: tuple[float, float] = (1.0, 1.0)
    friction_scale: tuple[float, float] = (1.0, 1.0)

    @property
    def enabled(self) -> bool:
        return self.mass_scale != (1.0, 1.0) or self.friction_scale != (1.0, 1.0)


@dataclass(frozen=True)
class DistractorConfig:
    # v3 (GraspV3Env) only: other cubes on the table. Randomly placed episodes (training and
    # validation) draw 0..max_count of them uniformly; explicit placements pass their own.
    max_count: int = 0
    min_separation: float = 0.10   # m, centre to centre, from the target cube and each other
    disturb: float = 20.0          # penalty per metre a distractor moves (same scale as push)
    knocked_over: float = 5.0      # once, when a distractor tips past episode.max_cube_tilt


@dataclass(frozen=True)
class RestingConfig:
    # v4 (GraspV4Env) only: how randomly placed cubes rest. This fraction lie on a side (long
    # axis horizontal, any direction), as knocked cubes mostly do (v3 recovery); the rest stand.
    lying_prob: float = 0.0
    max_lying_turn: float = 0.52  # rad: a lying cube whose long axis tilts further out of
    #                               horizontal is knocked
    # The gate's height window for a lying cube: the hand may be this far below the cube's
    # centre (the table stops it lower; 4 cm tall, the cube is gripped fine 1.5 cm low) and
    # gate.z_tol above. The default is symmetric, as for an upright cube.
    lying_gate_z_below: float | None = None


@dataclass(frozen=True)
class GraspConfig:
    action: ActionConfig = field(default_factory=ActionConfig)
    placement: PlacementConfig = field(default_factory=PlacementConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    episode: GraspEpisodeConfig = field(default_factory=GraspEpisodeConfig)
    success: GraspSuccessConfig = field(default_factory=GraspSuccessConfig)
    reward: GraspRewardConfig = field(default_factory=GraspRewardConfig)
    check: GraspCheckConfig = field(default_factory=GraspCheckConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    distractors: DistractorConfig = field(default_factory=DistractorConfig)
    resting: RestingConfig = field(default_factory=RestingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GraspConfig":
        return load_yaml(cls, path)

    def for_evaluation(self) -> "GraspConfig":
        """The task as evaluated: every episode starts from home with the nominal cube on the
        table."""
        return replace(self, curriculum=CurriculumConfig(), randomization=RandomizationConfig())


def alignment_error(R_tcp: np.ndarray, cube_yaw_value: float) -> float:
    """Closing-axis error to the nearest cube face pair, wrapped to [-pi/4, pi/4]."""
    delta = cube_yaw_value - closing_yaw(R_tcp)
    return float((delta + np.pi / 4) % (np.pi / 2) - np.pi / 4)


def gripper_tilt(R_tcp: np.ndarray) -> float:
    """Angle (rad) between the TCP z axis and straight down."""
    return float(np.arccos(np.clip(-R_tcp[2, 2], -1.0, 1.0)))


class GraspEnv(gym.Env):
    metadata = {"render_modes": []}
    distractor_slots = 0  # extra cubes in the scene (GraspV3Env)

    def __init__(self, config: GraspConfig | None = None, scene=None, cube: str = "cube") -> None:
        """`scene` (model, data): run in an existing scene, e.g. the Milestone 9 workcell, on the
        cube body named `cube`; by default the env builds its own pick scene."""
        self.config = config or GraspConfig()
        self.model, self.data = scene or load_pick_scene(distractors=self.distractor_slots)
        self.arm = PandaIndices.from_model(self.model)
        self.cube = CubeIndices.from_model(self.model, cube)
        self.adapter = make_action_adapter(self.model, self.arm, self.config.action)
        self.monitor = GraspMonitor(self.model, self.arm, self.cube, self.config.check)
        self._finger_qpos = [self.model.joint(f"finger_joint{i}").qposadr[0] for i in (1, 2)]
        limit = action_limit(self.config.action)
        self.action_space = spaces.Box(-limit, limit, shape=(self.adapter.action_dim,),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.closed = False
        self._dwell = self._empty = self._steps = 0
        self._prev = {"distance": 0.0, "align": 0.0, "lift": 0.0}
        self._state = None
        self._ik = None  # built on first curriculum reset
        self._physics = None  # built on first randomized reset
        self.cube_scales = {"mass": 1.0, "friction": 1.0}

    @property
    def elapsed_steps(self) -> int:
        return self._steps

    def _randomize(self, mass: float, friction: float) -> None:
        if self._physics is None:
            self._physics = CubePhysics(self.model, self.cube)
        self._physics.apply(mass_scale=mass, friction_scale=friction)
        self.cube_scales = {"mass": mass, "friction": friction}

    # --- measurements ------------------------------------------------------------------
    def _tcp_R(self) -> np.ndarray:
        return self.data.site_xmat[self.arm.tcp_site].reshape(3, 3)

    def _finger_gap(self) -> float:
        return float(sum(self.data.qpos[a] for a in self._finger_qpos))

    def _measure(self) -> dict:
        R, tcp = self._tcp_R(), tcp_position(self.data, self.arm)
        centre = cube_position(self.data, self.cube)
        return {
            "rel": centre - tcp,
            "distance": float(np.linalg.norm(centre - tcp)),
            "align": alignment_error(R, cube_yaw(self.data, self.cube)),
            "lift": float(centre[2] - self.cube.half_size[2]),
            "tilt": gripper_tilt(R),
            "cube_tilt": cube_tilt(self.data, self.cube),
        }

    def _floor_contact(self) -> bool:
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            bodies = (self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2])
            if 0 in bodies and self.cube.body not in bodies:
                return True
        return False

    # --- gym API -----------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if options.get("continue"):  # start from the scene as it is (a skill inside a larger task)
            return self._continue()
        cfg = self.config.placement
        reset_to_home(self.model, self.data, self.arm)
        if "arm_offset" in options:  # robustness tests: start away from the exact home pose
            q = self.data.qpos[self.arm.arm_qpos] + np.asarray(options["arm_offset"])
            self.data.qpos[self.arm.arm_qpos] = np.clip(q, self.adapter.low, self.adapter.high)
            mujoco.mj_forward(self.model, self.data)
        pregrasp = False
        if "cube" in options:
            x, y, yaw = (float(v) for v in options["cube"])
        else:
            x = self.np_random.uniform(*cfg.x_range)
            y = self.np_random.uniform(*cfg.y_range)
            yaw = self.np_random.uniform(-np.pi, np.pi)
            prob = self.config.curriculum.pregrasp_start_prob
            pregrasp = prob > 0 and self.np_random.uniform() < prob  # no draw when disabled
            rnd = self.config.randomization
            if rnd.enabled:  # training-only domain randomization; no draws when disabled
                lo, hi = np.log(rnd.mass_scale)
                self._randomize(mass=float(np.exp(self.np_random.uniform(lo, hi))),
                                friction=float(self.np_random.uniform(*rnd.friction_scale)))
        if "cube" in options and self._physics is not None:
            self._randomize(mass=1.0, friction=1.0)  # explicit placements use the nominal cube
        self._place_cube(x, y, yaw, options)
        if pregrasp:
            pregrasp = self._pregrasp_start()
        if pregrasp:
            self.adapter.adopt_ctrl(self.data)  # keep the grasp-pose targets; no sag-induced jump
        else:
            self.adapter.reset_to_measured(self.data)
            self.data.ctrl[self.arm.gripper_actuator] = GRIPPER_OPEN
        self.monitor.reset()
        self._state = self.monitor.update(self.data)
        self.closed = pregrasp
        self._dwell = self._empty = self._steps = 0
        m = self._measure()
        self._prev = {"distance": m["distance"], "align": abs(m["align"]),
                      "lift": min(m["lift"], self.config.reward.lift_cap)}
        info = {"cube": (x, y, yaw), "distance": m["distance"],
                "start": "pregrasp" if pregrasp else "home"}
        return self._observation(m), info

    def _place_cube(self, x: float, y: float, yaw: float, options: dict) -> None:
        """Set the cube down at the start of an episode (upright; GraspV4Env may lay it down)."""
        set_cube_pose(self.model, self.data, self.cube, (x, y), yaw)

    def _knocked(self, m: dict) -> bool:
        """Whether the cube has been knocked over (the episode then ends)."""
        return m["cube_tilt"] > self.config.episode.max_cube_tilt

    def _height_ok(self, rel_z: float) -> bool:
        """The gate's height condition (rel_z: cube centre minus TCP)."""
        return abs(rel_z) < self.config.gate.z_tol

    def _continue(self):
        """Begin an episode from the current physics state: the arm keeps its commands (no
        jump), the fingers open, the cube stays where it is."""
        self.adapter.adopt_ctrl(self.data)
        self.data.ctrl[self.arm.gripper_actuator] = GRIPPER_OPEN
        self.monitor.reset()
        self._state = self.monitor.update(self.data)
        self.closed = False
        self._dwell = self._empty = self._steps = 0
        m = self._measure()
        self._prev = {"distance": m["distance"], "align": abs(m["align"]),
                      "lift": min(m["lift"], self.config.reward.lift_cap)}
        c = cube_position(self.data, self.cube)
        return self._observation(m), {"cube": (float(c[0]), float(c[1]),
                                               cube_yaw(self.data, self.cube)),
                                      "distance": m["distance"], "start": "continue"}

    def _pregrasp_start(self, settle_steps: int = 100) -> bool:
        """Training curriculum: arm at the IK grasp pose, fingers closed on the cube.

        Fingers are placed on the cube faces and squeeze for 0.2 s of physics so contacts are
        established. Returns False (normal start) if IK does not converge.
        """
        if self._ik is None:
            self._ik = PoseIK()
        quat = top_down_quat(grasp_closing_yaw(cube_yaw(self.data, self.cube),
                                               closing_yaw(self._tcp_R())))
        q, ok = self._ik.solve(self.data.qpos[self.arm.arm_qpos],
                               cube_position(self.data, self.cube), quat)
        if not ok:
            return False
        self.data.qpos[self.arm.arm_qpos] = q
        self.data.ctrl[self.arm.arm_actuators] = q
        for adr in self._finger_qpos:
            self.data.qpos[adr] = self.cube.half_size[0]  # finger pads on the cube faces
        self.data.ctrl[self.arm.gripper_actuator] = GRIPPER_CLOSED
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=settle_steps)
        return True

    def _update_gate(self, m: dict) -> None:
        g = self.config.gate
        if not self.closed:
            ready = (np.hypot(*m["rel"][:2]) < g.xy_tol and self._height_ok(m["rel"][2])
                     and abs(m["align"]) < g.align_tol and m["tilt"] < g.tilt_tol)
            self._dwell = self._dwell + 1 if ready else 0
            if self._dwell >= g.dwell_steps:
                self.closed, self._empty = True, 0
        else:
            self._empty = self._empty + 1 if self._finger_gap() < g.reopen_gap else 0
            if self._empty >= g.reopen_steps:
                self.closed, self._dwell = False, 0
        self.data.ctrl[self.arm.gripper_actuator] = GRIPPER_CLOSED if self.closed else GRIPPER_OPEN

    def step(self, action):
        cfg, r = self.config, self.config.reward
        a, bound_terms = bound_action(action, self.config.action)
        self.adapter.step(self.data, a)
        self._steps += 1
        if not (np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))):
            obs = np.zeros(OBS_DIM, dtype=np.float32)
            return obs, -r.knocked_over, True, False, {"is_success": False,
                                                       "failure": "non_finite",
                                                       "reward_terms": {}}

        m = self._measure()
        self._update_gate(m)
        self._state = state = self.monitor.update(self.data)
        lifted = min(m["lift"], r.lift_cap)
        success = bool(state.grasped and m["lift"] >= cfg.success.lift)
        floor = self._floor_contact()
        knocked = self._knocked(m)

        terms = {
            "approach": r.approach * (self._prev["distance"] - m["distance"]),
            "align": r.align * (self._prev["align"] - abs(m["align"])),
            "lift": r.lift * (lifted - self._prev["lift"]),
            "hold": r.hold if success else 0.0,
            "action": -r.action * float(a @ a),
            "gripper_tilt": -r.gripper_tilt * m["tilt"],
            "floor": -r.floor if floor else 0.0,
            "knocked_over": -r.knocked_over if knocked else 0.0,
            **bound_terms,
        }
        self._prev = {"distance": m["distance"], "align": abs(m["align"]), "lift": lifted}
        terminated = knocked
        truncated = not terminated and self._steps >= cfg.episode.horizon
        info = {
            "is_success": success,
            "failure": "knocked_over" if knocked else None,
            "grasped": state.grasped,
            "gripper_closed": self.closed,
            "distance": m["distance"],
            "align_error": m["align"],
            "lift": m["lift"],
            "contacts": (state.left_contact, state.right_contact),
            "floor_contact": floor,
            "reward_terms": terms,
        }
        return self._observation(m), float(sum(terms.values())), terminated, truncated, info

    def _observation(self, m: dict | None = None) -> np.ndarray:
        m = m or self._measure()
        R, s = self._tcp_R(), self._state
        return np.concatenate([
            self.data.qpos[self.arm.arm_qpos],
            self.data.qvel[self.arm.arm_qvel],
            self.adapter.q_cmd,
            tcp_position(self.data, self.arm),
            R[:, :2].T.flatten(),  # first two columns: a continuous 6-D rotation encoding
            [self._finger_gap(), float(self.closed)],
            m["rel"],
            [np.sin(4 * m["align"]), np.cos(4 * m["align"]), m["lift"], m["cube_tilt"]],
            [float(s.left_contact), float(s.right_contact)],
        ]).astype(np.float32)


class GraspV2Env(GraspEnv):
    """Grasp task v2: a committed grasp yaw, and pushing the cube is penalized.

    v1 policies were weakest on cubes about 45 deg from the wrist, where two face pairs are equally
    good and the v1 alignment error (to the nearest pair, observed as sin/cos of 4x the error)
    offers no tie-break. v2 commits: at reset the target closing yaw is the face pair nearest the
    wrist (ties broken by rounding, as in the scripted grasp); afterwards it follows the cube's yaw
    continuously (nearest to the previous target), so the target never flips. The alignment error
    is to that target, wrapped to +-90 deg (the fingers are symmetric), and is observed as sin/cos
    of 2x the error. The gate also requires alignment with the committed pair.

    v1 policies also often pushed the cube out of reach. v2 charges reward.push per metre the cube
    moves horizontally while the fingers are open, and ends the episode (reward.pushed_away) once
    it is more than episode.max_push from where it started, before the fingers closed.
    """

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._target_yaw = None
        obs, info = super().reset(seed=seed, options=options)
        self._start_xy = cube_position(self.data, self.cube)[:2].copy()
        info["target_closing_yaw"] = self._target_yaw
        return obs, info

    def _measure(self) -> dict:
        m = super()._measure()
        current = closing_yaw(self._tcp_R())
        reference = current if self._target_yaw is None else self._target_yaw
        self._target_yaw = grasp_closing_yaw(cube_yaw(self.data, self.cube), reference)
        error = self._target_yaw - current
        m["align"] = float((error + np.pi / 2) % np.pi - np.pi / 2)
        return m

    def _observation(self, m: dict | None = None) -> np.ndarray:
        m = m or self._measure()
        obs = super()._observation(m)
        obs[OBS_SLICES["align_sin4"]] = np.sin(2 * m["align"])
        obs[OBS_SLICES["align_cos4"]] = np.cos(2 * m["align"])
        return obs

    def step(self, action):
        cfg, r = self.config, self.config.reward
        open_before = not self.closed
        xy_before = cube_position(self.data, self.cube)[:2].copy()
        obs, reward, terminated, truncated, info = super().step(action)
        if info.get("failure") == "non_finite":
            return obs, reward, terminated, truncated, info
        xy = cube_position(self.data, self.cube)[:2]
        moved = float(np.linalg.norm(xy - xy_before)) if open_before else 0.0
        displaced = float(np.linalg.norm(xy - self._start_xy))
        pushed_away = bool(open_before and not self.closed and cfg.episode.max_push > 0
                           and displaced > cfg.episode.max_push and not terminated)
        terms = {"push": -r.push * moved,
                 "pushed_away": -r.pushed_away if pushed_away else 0.0}
        info["reward_terms"].update(terms)
        info["cube_displacement"] = displaced
        if pushed_away:
            info["failure"] = "pushed_away"
            terminated, truncated = True, False
        return obs, reward + sum(terms.values()), terminated, truncated, info


OBS_SCHEMA_V3 = "grasp-v3-49d"  # v2 layout + per distractor slot: present flag, cube - TCP (3)


class GraspV3Env(GraspV2Env):
    """Grasp task v3: v2 (committed grasp yaw) on a table with up to two other cubes.

    The distractors are the same size as the target and must be left alone: they are observed
    (presence flag and position relative to the gripper, per slot), moving one costs
    distractors.disturb per metre, and tipping one over ends the episode (distractors.knocked_over,
    failure `knocked_distractor`). Unused slots park their cube out of reach and observe as
    absent. Built for multi-cube tasks (stacking), where the other cubes are part of the scene.
    """

    distractor_slots = 2

    def __init__(self, config: GraspConfig | None = None) -> None:
        super().__init__(config)
        self.others = CubeIndices.distractors(self.model)
        dim = OBS_DIM + 4 * len(self.others)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)
        self.present = [False] * len(self.others)
        self._other_xy = [np.zeros(2) for _ in self.others]

    def _place_others(self, poses) -> None:
        for k, other in enumerate(self.others):
            if k < len(poses):
                x, y, yaw = (float(v) for v in poses[k])
                set_cube_pose(self.model, self.data, other, (x, y), yaw)
                self.present[k] = True
            else:
                self.present[k] = False  # stays parked (home keyframe)
        mujoco.mj_forward(self.model, self.data)
        self._other_xy = [cube_position(self.data, o)[:2].copy() for o in self.others]

    def _draw_others(self) -> list[tuple[float, float, float]]:
        cfg, d = self.config.placement, self.config.distractors
        count = int(self.np_random.integers(0, d.max_count + 1)) if d.max_count else 0
        taken = [cube_position(self.data, self.cube)[:2]]
        poses = []
        for _ in range(count):
            for _ in range(100):  # rejection sampling; gives up quietly if the table is full
                xy = np.array([self.np_random.uniform(*cfg.x_range),
                               self.np_random.uniform(*cfg.y_range)])
                if all(np.linalg.norm(xy - t) >= d.min_separation for t in taken):
                    taken.append(xy)
                    poses.append((*xy, float(self.np_random.uniform(-np.pi, np.pi))))
                    break
        return poses

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        options = options or {}
        obs, info = super().reset(seed=seed, options=options)
        poses = options.get("distractors", []) if "cube" in options else self._draw_others()
        self._place_others(poses)
        info["distractors"] = [tuple(p) for p in poses]
        return self._observation(), info

    def _floor_contact(self) -> bool:
        cubes = {self.cube.body, *(o.body for o in self.others)}
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            bodies = {self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]}
            if 0 in bodies and not bodies & cubes:
                return True
        return False

    def _observation(self, m: dict | None = None) -> np.ndarray:
        obs = super()._observation(m)
        tcp = tcp_position(self.data, self.arm)
        extra = []
        for present, other in zip(self.present, self.others, strict=True):
            rel = cube_position(self.data, other) - tcp if present else np.zeros(3)
            extra += [float(present), *rel]
        return np.concatenate([obs, np.asarray(extra, dtype=np.float32)])

    def step(self, action):
        d, ep = self.config.distractors, self.config.episode
        obs, reward, terminated, truncated, info = super().step(action)
        if info.get("failure") == "non_finite":
            return np.zeros(self.observation_space.shape, dtype=np.float32), reward, \
                terminated, truncated, info
        moved, knocked = 0.0, False
        for k, other in enumerate(self.others):
            if not self.present[k]:
                continue
            xy = cube_position(self.data, other)[:2]
            moved += float(np.linalg.norm(xy - self._other_xy[k]))
            self._other_xy[k] = xy.copy()
            knocked |= cube_tilt(self.data, other) > ep.max_cube_tilt
        terms = {"disturb": -d.disturb * moved,
                 "knocked_distractor": -d.knocked_over if knocked and not terminated else 0.0}
        info["reward_terms"].update(terms)
        if knocked and not terminated:
            info["failure"] = "knocked_distractor"
            info["is_success"] = False
            terminated, truncated = True, False
        return obs, reward + sum(terms.values()), terminated, truncated, info


class GraspV2BlindClutterEnv(GraspV3Env):
    """Evaluation only: v2's observation (no distractor slots) in a scene WITH distractor cubes,
    so v2 policies, which cannot see other cubes, can be measured on clutter sets."""

    def __init__(self, config: GraspConfig | None = None) -> None:
        super().__init__(config)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)

    def _observation(self, m: dict | None = None) -> np.ndarray:
        return GraspV2Env._observation(self, m)


OBS_SCHEMA_V4 = "grasp-v4-41d"  # v2 layout; lift from the cube's resting height; lying cubes


class GraspV4Env(GraspV2Env):
    """Grasp task v4 (v3 recovery): v2 with the cube resting either upright or lying on a side.

    Knocked cubes mostly come to rest lying on a side (scripts/disturbance_ceiling.py), and a cube
    lying on a side can only be stood up again if it is gripped across its 4 cm width, the
    fingers closing across its long axis. So for a lying cube the committed closing direction is
    across the long axis (symmetric every pi, not pi/2), and the gate requires that alignment.
    `resting.lying_prob` of the randomly placed episodes start lying (long axis horizontal, any
    direction); explicit placements pass options["lying"]. Whether the cube lies is read from its
    pose at the start of the episode, so the env also works on a cube already in a scene
    (options["continue"], the work-order cell).

    Lift and success are measured from the cube's own resting height (2 cm lying, 3 cm upright),
    so "lifted 5 cm" means the same either way, and the observation's cube height slot holds it.
    A lying cube is knocked when its long axis tilts further than resting.max_lying_turn out of
    horizontal (it may turn or roll on the table); an upright cube, as in v2, when it tilts past
    episode.max_cube_tilt. The
    observation layout is v2's (the cube tilt slot reads pi/2 for a lying cube), so a v4 policy
    can start from a v2 policy's weights.
    """

    lays_cubes_down = True  # (explicit placements accept options["lying"])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._lying, self._lying_target = False, None
        self._rest_z = self.cube.half_size[2]
        obs, info = super().reset(seed=seed, options=options)
        options = options or {}
        p = self.config.curriculum.preclose_start_prob
        if (p > 0 and "cube" not in options and not options.get("continue")
                and info["start"] == "home" and self.np_random.uniform() < p
                and self._preclose_start()):
            info["start"] = "preclose"
            m = self._measure()
            self._prev = {"distance": m["distance"], "align": abs(m["align"]),
                          "lift": min(m["lift"], self.config.reward.lift_cap)}
            info["distance"] = m["distance"]
            obs = self._observation(m)
        info["lying"] = self._lying
        return obs, info

    def _preclose_start(self) -> bool:
        """Training curriculum: the open hand hovering 2-5 cm above the cube, up to 1 cm off its
        centre in x and in y and 25 deg off the grip direction. Returns False (home start) if
        IK fails."""
        if self._ik is None:
            self._ik = PoseIK()
        m = self._measure()
        rng = self.np_random
        yaw = closing_yaw(self._tcp_R()) + m["align"] + rng.uniform(-0.44, 0.44)
        target = cube_position(self.data, self.cube) + [rng.uniform(-0.01, 0.01),
                                                         rng.uniform(-0.01, 0.01),
                                                         rng.uniform(0.02, 0.05)]
        q, ok = self._ik.solve(self.data.qpos[self.arm.arm_qpos], target, top_down_quat(yaw))
        if not ok:
            return False
        self.data.qpos[self.arm.arm_qpos] = q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.adapter.reset_to_measured(self.data)
        self.data.ctrl[self.arm.gripper_actuator] = GRIPPER_OPEN
        self.monitor.reset()
        self._state = self.monitor.update(self.data)
        self._target_yaw = self._lying_target = None   # (committed again from here)
        return True

    def _height_ok(self, rel_z: float) -> bool:
        below = self.config.resting.lying_gate_z_below
        if not self._lying or below is None:
            return super()._height_ok(rel_z)
        return -self.config.gate.z_tol < rel_z < below

    def _place_cube(self, x: float, y: float, yaw: float, options: dict) -> None:
        if "cube" in options:
            lying = bool(options.get("lying", False))
        else:
            p = self.config.resting.lying_prob
            lying = p > 0 and self.np_random.uniform() < p  # no draw when disabled
        if not lying:
            super()._place_cube(x, y, yaw, options)
        else:  # long axis along `yaw`: the cube's z axis turned down to horizontal
            q_yaw = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
            q_down = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])  # 90 deg about y
            quat = np.zeros(4)
            mujoco.mju_mulQuat(quat, q_yaw, q_down)
            self.data.qpos[self.cube.qpos:self.cube.qpos + 7] = [x, y, self.cube.half_size[0],
                                                                 *quat]
            self.data.qvel[self.cube.qvel:self.cube.qvel + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)
        self._start_rest()

    def _start_rest(self) -> None:
        """Record how the cube rests now: lying or upright, and its resting height."""
        self._lying = cube_tilt(self.data, self.cube) > np.pi / 4
        self._rest_z = self.cube.half_size[0] if self._lying else self.cube.half_size[2]

    def _continue(self):
        self._target_yaw = self._lying_target = None
        self._start_rest()
        return super()._continue()

    def _long_axis_yaw(self) -> float:
        axis = self.data.xmat[self.cube.body].reshape(3, 3)[:, 2]
        return float(np.arctan2(axis[1], axis[0]))

    def _measure(self) -> dict:
        m = super()._measure()  # v2: the committed face pair (upright)
        m["lift"] = float(cube_position(self.data, self.cube)[2] - self._rest_z)
        if self._lying:
            current = closing_yaw(self._tcp_R())
            across = self._long_axis_yaw() + np.pi / 2
            reference = current if self._lying_target is None else self._lying_target
            self._lying_target = grasp_closing_yaw(across, reference, symmetry=np.pi)
            error = self._lying_target - current
            m["align"] = float((error + np.pi / 2) % np.pi - np.pi / 2)
            self._target_yaw = self._lying_target
        return m

    def _knocked(self, m: dict) -> bool:
        if not self._lying:
            return super()._knocked(m)
        axis = self.data.xmat[self.cube.body].reshape(3, 3)[:, 2]   # (turning on the table is
        out_of_lying = np.arcsin(min(1.0, abs(float(axis[2]))))      # fine, as for an upright
        return out_of_lying > self.config.resting.max_lying_turn    # cube turning on its base)

    def _pregrasp_start(self, settle_steps: int = 100) -> bool:
        if not self._lying:
            return super()._pregrasp_start(settle_steps)
        if self._ik is None:
            self._ik = PoseIK()
        yaw = grasp_closing_yaw(self._long_axis_yaw() + np.pi / 2, closing_yaw(self._tcp_R()),
                                symmetry=np.pi)
        q, ok = self._ik.solve(self.data.qpos[self.arm.arm_qpos],
                               cube_position(self.data, self.cube), top_down_quat(yaw))
        if not ok:
            return False
        self.data.qpos[self.arm.arm_qpos] = q
        self.data.ctrl[self.arm.arm_actuators] = q
        for adr in self._finger_qpos:
            self.data.qpos[adr] = self.cube.half_size[0]  # across the 4 cm width
        self.data.ctrl[self.arm.gripper_actuator] = GRIPPER_CLOSED
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=settle_steps)
        return True
