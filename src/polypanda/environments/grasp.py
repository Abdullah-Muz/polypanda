"""GraspEnv: pick up a cube with seven joint-increment actions and a gated scripted gripper.

The policy moves the arm for the whole task (approach, align, descend, lift). The fingers close
by a fixed gate when the TCP is centred on the cube, aligned with a pair of faces and close to
vertical; they reopen if they close on nothing, allowing a retry. Episodes always run the full
horizon (lesson from reach v1). Success: the verified grasp predicate holds with the cube at
least `success.lift` above the table. Reward terms are returned in info["reward_terms"].
"""

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from polypanda.config import load_yaml
from polypanda.control.action_adapter import ACTION_DIM, JointDeltaActionAdapter
from polypanda.control.ik import PoseIK, closing_yaw, grasp_closing_yaw, top_down_quat
from polypanda.environments.reach import ActionConfig
from polypanda.simulation.grasp import GraspCheckConfig, GraspMonitor
from polypanda.simulation.panda import PandaIndices, reset_to_home, tcp_position
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


@dataclass(frozen=True)
class CurriculumConfig:
    # Training-only "reset to helpful states": this fraction of randomly placed episodes starts
    # with the gripper already closed around the cube at its grasp pose, so the policy
    # experiences lift and hold rewards it would otherwise rarely discover. Never applied to
    # resets with an explicit placement (evaluation) or to validation (for_evaluation()).
    pregrasp_start_prob: float = 0.0


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

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GraspConfig":
        return load_yaml(cls, path)

    def for_evaluation(self) -> "GraspConfig":
        """The task as evaluated: every episode starts from home with the cube on the table."""
        return replace(self, curriculum=CurriculumConfig())


def alignment_error(R_tcp: np.ndarray, cube_yaw_value: float) -> float:
    """Closing-axis error to the nearest cube face pair, wrapped to [-pi/4, pi/4]."""
    delta = cube_yaw_value - closing_yaw(R_tcp)
    return float((delta + np.pi / 4) % (np.pi / 2) - np.pi / 4)


def gripper_tilt(R_tcp: np.ndarray) -> float:
    """Angle (rad) between the TCP z axis and straight down."""
    return float(np.arccos(np.clip(-R_tcp[2, 2], -1.0, 1.0)))


class GraspEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: GraspConfig | None = None) -> None:
        self.config = config or GraspConfig()
        self.model, self.data = load_pick_scene()
        self.arm = PandaIndices.from_model(self.model)
        self.cube = CubeIndices.from_model(self.model)
        self.adapter = JointDeltaActionAdapter(self.model, self.arm, **asdict(self.config.action))
        self.monitor = GraspMonitor(self.model, self.arm, self.cube, self.config.check)
        self._finger_qpos = [self.model.joint(f"finger_joint{i}").qposadr[0] for i in (1, 2)]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.closed = False
        self._dwell = self._empty = self._steps = 0
        self._prev = {"distance": 0.0, "align": 0.0, "lift": 0.0}
        self._state = None
        self._ik = None  # built on first curriculum reset

    @property
    def elapsed_steps(self) -> int:
        return self._steps

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
        cfg = self.config.placement
        reset_to_home(self.model, self.data, self.arm)
        pregrasp = False
        if "cube" in options:
            x, y, yaw = (float(v) for v in options["cube"])
        else:
            x = self.np_random.uniform(*cfg.x_range)
            y = self.np_random.uniform(*cfg.y_range)
            yaw = self.np_random.uniform(-np.pi, np.pi)
            prob = self.config.curriculum.pregrasp_start_prob
            pregrasp = prob > 0 and self.np_random.uniform() < prob  # no draw when disabled
        set_cube_pose(self.model, self.data, self.cube, (x, y), yaw)
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
            ready = (np.hypot(*m["rel"][:2]) < g.xy_tol and abs(m["rel"][2]) < g.z_tol
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
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
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
        knocked = m["cube_tilt"] > cfg.episode.max_cube_tilt

        terms = {
            "approach": r.approach * (self._prev["distance"] - m["distance"]),
            "align": r.align * (self._prev["align"] - abs(m["align"])),
            "lift": r.lift * (lifted - self._prev["lift"]),
            "hold": r.hold if success else 0.0,
            "action": -r.action * float(a @ a),
            "gripper_tilt": -r.gripper_tilt * m["tilt"],
            "floor": -r.floor if floor else 0.0,
            "knocked_over": -r.knocked_over if knocked else 0.0,
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
