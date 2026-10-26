"""TransportEnv: carry a grasped cube to a 3-D goal and hold it there.

Episodes start from grasp-to-transport handoff states (polypanda.environments.handoff) produced
by both scripted and learned grasp controllers, so transport sees the start distribution it will
receive in pick-and-place. The gripper stays closed; releasing belongs to the place skill.
"""

from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from polypanda.config import load_yaml
from polypanda.control.action_adapter import (
    ACTION_DIM,
    action_limit,
    bound_action,
    make_action_adapter,
)
from polypanda.control.ik import PoseIK
from polypanda.control.scripted_carry import carry_action, level_quat
from polypanda.environments.grasp import GRIPPER_CLOSED, gripper_tilt
from polypanda.environments.handoff import HandoffPool, restore_state
from polypanda.environments.reach import ActionConfig
from polypanda.simulation.grasp import GraspCheckConfig, GraspMonitor
from polypanda.simulation.panda import PandaIndices, tcp_position
from polypanda.simulation.scenes import CubeIndices, cube_position, cube_tilt, load_pick_scene

REPO_ROOT = Path(__file__).resolve().parents[3]

OBS_SCHEMA = "transport-v1-47d"
OBS_LAYOUT = (
    ("q", 7), ("dq", 7), ("q_cmd", 7),
    ("tcp", 3), ("tcp_rot6", 6), ("finger_gap", 1),
    ("cube_rel", 3), ("slip_velocity", 3), ("cube_height", 1), ("cube_tilt", 1),
    ("contacts", 2), ("goal", 3), ("goal_error", 3),
)
OBS_DIM = sum(n for _, n in OBS_LAYOUT)
OBS_SLICES = {
    name: slice(start, start + n)
    for (name, n), start in zip(
        OBS_LAYOUT, np.cumsum([0] + [n for _, n in OBS_LAYOUT[:-1]]), strict=True)
}


@dataclass(frozen=True)
class GoalConfig:
    x_range: tuple[float, float] = (0.35, 0.65)
    y_range: tuple[float, float] = (-0.25, 0.25)
    z_range: tuple[float, float] = (0.10, 0.25)  # cube-centre height (m)
    min_distance: float = 0.15                    # from the cube at handoff (m)
    max_tries: int = 1000


def sample_goal(rng: np.random.Generator, cube: np.ndarray, cfg: GoalConfig) -> np.ndarray:
    """Uniform goal in the box, at least `min_distance` from the cube's current position."""
    for _ in range(cfg.max_tries):
        goal = np.array([rng.uniform(*cfg.x_range), rng.uniform(*cfg.y_range),
                         rng.uniform(*cfg.z_range)])
        if np.linalg.norm(goal - cube) >= cfg.min_distance:
            return goal
    raise RuntimeError("no goal found; check GoalConfig")


@dataclass(frozen=True)
class TransportEpisodeConfig:
    horizon: int = 200          # policy steps (10 s); no termination on success
    drop_steps: int = 5         # consecutive steps without both finger contacts = dropped
    max_cube_tilt: float = 1.4  # rad (~80 deg) in hand: treated as dropped


@dataclass(frozen=True)
class TransportSuccessConfig:
    tolerance: float = 0.02     # m, cube centre to goal
    dwell_steps: int = 5


@dataclass(frozen=True)
class TransportRewardConfig:
    progress: float = 20.0      # per metre of cube-to-goal distance reduced (telescoping)
    hold: float = 0.5           # per step, 1 -> 0 inside tolerance, only with a verified grasp
    action: float = 0.005       # x ||a||^2
    action_rate: float = 0.05   # x ||a_t - a_{t-1}||^2: smooth motion, less slip
    gripper_tilt: float = 0.1   # per rad
    cube_tilt: float = 0.1      # per rad of cube tilt in the world
    floor: float = 0.1          # per step with the arm or fingers touching the floor
    drop: float = 5.0           # once, lost grip (contacts lost or cube tipped); terminates
    touchdown: float = 5.0      # once, cube lowered onto the table while held; terminates
    # v2: the v1 pilot sank the cube onto the table while carrying it sideways, warned only by
    # the terminal touchdown. Continuous signal below a safe carrying height (cube lift, m):
    safe_lift: float = 0.06
    low_height: float = 10.0    # per metre below safe_lift, per step


@dataclass(frozen=True)
class ResidualConfig:
    """Used by TransportResidualEnv only."""
    scale: float = 0.3       # policy output range added to the base action (normalized units)
    base_gain: float = 0.2   # scripted-carry IK gain, as for the baseline
    penalty: float = 0.01    # x ||policy output||^2 per step: deviate only when it helps


@dataclass(frozen=True)
class TransportConfig:
    action: ActionConfig = field(default_factory=ActionConfig)
    goal: GoalConfig = field(default_factory=GoalConfig)
    episode: TransportEpisodeConfig = field(default_factory=TransportEpisodeConfig)
    success: TransportSuccessConfig = field(default_factory=TransportSuccessConfig)
    reward: TransportRewardConfig = field(default_factory=TransportRewardConfig)
    check: GraspCheckConfig = field(default_factory=GraspCheckConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    start_pool: str = "data/handoff/transport_train_v1.npz"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TransportConfig":
        return load_yaml(cls, path)


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() or p.exists() else REPO_ROOT / p


class TransportEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: TransportConfig | None = None, scene=None,
                 cube: str = "cube") -> None:
        """`scene` (model, data): run in an existing scene (e.g. the workcell) on the cube body
        named `cube`, starting from its current state (reset option "continue"); by default the
        env builds its own scene and starts from the handoff pool."""
        self.config = config or TransportConfig()
        self.model, self.data = scene or load_pick_scene()
        self.arm = PandaIndices.from_model(self.model)
        self.cube = CubeIndices.from_model(self.model, cube)
        self.adapter = make_action_adapter(self.model, self.arm, self.config.action)
        self.monitor = GraspMonitor(self.model, self.arm, self.cube, self.config.check)
        self.pool = (HandoffPool.load(_resolve(self.config.start_pool), self.model)
                     if scene is None else None)
        self._finger_qpos = [self.model.joint(f"finger_joint{i}").qposadr[0] for i in (1, 2)]
        limit = action_limit(self.config.action)
        self.action_space = spaces.Box(-limit, limit, shape=(self.adapter.action_dim,),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.goal = np.zeros(3)
        self.start_index = 0
        self._steps = self._dwell = self._no_contact = 0
        self._distance = 0.0
        self._p_rel = np.zeros(3)
        self._slip = np.zeros(3)
        self._prev_action = np.zeros(self.adapter.action_dim)
        self._state = None
        self._bound_terms = {}

    @property
    def elapsed_steps(self) -> int:
        return self._steps

    def _hand_frame_cube(self) -> np.ndarray:
        """Cube centre in the TCP frame (m)."""
        R = self.data.site_xmat[self.arm.tcp_site].reshape(3, 3)
        return R.T @ (cube_position(self.data, self.cube) - tcp_position(self.data, self.arm))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if options.get("continue"):  # from the scene as it is (a skill inside a larger task)
            self.start_index = -1
        elif "state" in options:
            state, self.start_index = np.asarray(options["state"], dtype=float), -1
            restore_state(self.model, self.data, state)
        else:
            self.start_index = int(options.get("state_index",
                                               self.np_random.integers(len(self.pool))))
            restore_state(self.model, self.data, self.pool.states[self.start_index])
        self.data.ctrl[self.arm.gripper_actuator] = GRIPPER_CLOSED
        self.adapter.adopt_ctrl(self.data)  # continue from the grasp skill's commands: no jump
        cube = cube_position(self.data, self.cube)
        if "goal" in options:
            self.goal = np.asarray(options["goal"], dtype=float).copy()
        else:
            self.goal = sample_goal(self.np_random, cube, self.config.goal)
        self.monitor.reset()
        self._state = self.monitor.update(self.data)
        self._steps = self._dwell = self._no_contact = 0
        self._distance = float(np.linalg.norm(self.goal - cube))
        self._p_rel = self._hand_frame_cube()
        self._slip = np.zeros(3)
        self._prev_action = np.zeros(self.adapter.action_dim)
        self._after_reset()
        source = self.pool.sources[self.start_index] if self.start_index >= 0 else "given"
        return self._observation(), {"goal": self.goal.copy(), "distance": self._distance,
                                     "start_index": self.start_index, "start_source": source}

    # Extension points (TransportResidualEnv); defaults give plain transport.
    def _after_reset(self) -> None:
        pass

    def _applied_action(self, action) -> np.ndarray:
        a, self._bound_terms = bound_action(action, self.config.action)
        return a

    def _extra_terms(self) -> dict:
        return self._bound_terms

    def step(self, action):
        cfg, r = self.config, self.config.reward
        a = self._applied_action(action)
        self.adapter.step(self.data, a)
        self._steps += 1
        if not (np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))):
            return (np.zeros(self.observation_space.shape, dtype=np.float32), -r.drop, True,
                    False, {"is_success": False, "failure": "non_finite", "reward_terms": {}})

        cube = cube_position(self.data, self.cube)
        distance = float(np.linalg.norm(self.goal - cube))
        p_rel = self._hand_frame_cube()
        self._slip = (p_rel - self._p_rel) / self.adapter.control_dt
        self._p_rel = p_rel
        self._state = state = self.monitor.update(self.data)
        self._no_contact = 0 if state.both_contact else self._no_contact + 1
        tilt = cube_tilt(self.data, self.cube)
        lift = cube[2] - self.cube.half_size[2]
        on_table = lift < 0.01
        # Touchdown = set down while both fingers still hold it; reaching the table unheld is a
        # fall, i.e. lost grip (even before drop_steps without contact have accumulated).
        touched_down = bool(on_table and state.both_contact)
        lost_grip = bool(self._no_contact >= cfg.episode.drop_steps
                         or tilt > cfg.episode.max_cube_tilt
                         or (on_table and not state.both_contact))
        if lost_grip:
            touched_down = False
        dropped = lost_grip or touched_down  # any terminal failure
        # A cube that has fallen over in the fingers is not delivered, even if the grasp
        # predicate still holds (found in the baseline evaluation: 80.1 deg tilt at the goal).
        at_goal = distance < cfg.success.tolerance and state.grasped and not dropped
        self._dwell = self._dwell + 1 if at_goal else 0
        success = self._dwell >= cfg.success.dwell_steps
        R = self.data.site_xmat[self.arm.tcp_site].reshape(3, 3)
        floor = any(
            0 in (self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2])
            and self.cube.body not in (self.model.geom_bodyid[c.geom1],
                                       self.model.geom_bodyid[c.geom2])
            for c in (self.data.contact[i] for i in range(self.data.ncon)))

        terms = {
            "progress": r.progress * (self._distance - distance),
            "hold": r.hold * (1.0 - distance / cfg.success.tolerance) if at_goal else 0.0,
            "action": -r.action * float(a @ a),
            "action_rate": -r.action_rate * float(np.sum((a - self._prev_action) ** 2)),
            "gripper_tilt": -r.gripper_tilt * gripper_tilt(R),
            "cube_tilt": -r.cube_tilt * tilt,
            "floor": -r.floor if floor else 0.0,
            "low_height": -r.low_height * max(0.0, r.safe_lift - lift),
            "drop": -r.drop if lost_grip else 0.0,
            "touchdown": -r.touchdown if touched_down else 0.0,
            **self._extra_terms(),
        }
        self._distance = distance
        self._prev_action = a
        terminated = dropped
        truncated = not terminated and self._steps >= cfg.episode.horizon
        failure = "lost_grip" if lost_grip else "lowered_onto_table" if touched_down else None
        info = {
            "is_success": success,
            "failure": failure,
            "lift": float(lift),
            "distance": distance,
            "grasped": state.grasped,
            "contacts": (state.left_contact, state.right_contact),
            "slip_speed": float(np.linalg.norm(self._slip)),
            "cube_tilt": tilt,
            "floor_contact": floor,
            "reward_terms": terms,
        }
        return self._observation(), float(sum(terms.values())), terminated, truncated, info

    def _observation(self) -> np.ndarray:
        R = self.data.site_xmat[self.arm.tcp_site].reshape(3, 3)
        cube = cube_position(self.data, self.cube)
        tcp = tcp_position(self.data, self.arm)
        s = self._state
        return np.concatenate([
            self.data.qpos[self.arm.arm_qpos],
            self.data.qvel[self.arm.arm_qvel],
            self.adapter.q_cmd,
            tcp,
            R[:, :2].T.flatten(),
            [sum(self.data.qpos[adr] for adr in self._finger_qpos)],
            cube - tcp,
            self._slip,
            [cube[2] - self.cube.half_size[2], cube_tilt(self.data, self.cube)],
            [float(s.left_contact), float(s.right_contact)],
            self.goal,
            self.goal - cube,
        ]).astype(np.float32)


RESIDUAL_OBS_SCHEMA = "transport-residual-v1-54d"


class TransportResidualEnv(TransportEnv):
    """Residual RL on the scripted carry.

    Applied action = clip(base + scale * policy_output, -1, 1), where base is the scripted
    closed-loop carry (polypanda.control.scripted_carry). A zero policy output reproduces the
    scripted baseline exactly. The observation appends the base action the policy's output will
    be added to (schema transport-residual-v1-54d), and a small penalty on the policy output
    keeps corrections to where they help. The base controller uses privileged information (true
    cube pose, exact kinematics), as documented for the baseline.
    """

    def __init__(self, config: TransportConfig | None = None) -> None:
        super().__init__(config)
        if self.config.action.space != "joint":
            raise ValueError("residual transport adds to the scripted carry's joint actions; "
                             "it needs action.space 'joint'")
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM + ACTION_DIM,),
                                            dtype=np.float32)
        self._ik = PoseIK()
        self._quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._base = np.zeros(ACTION_DIM)
        self._residual = np.zeros(ACTION_DIM)

    def _after_reset(self) -> None:
        self._quat = level_quat(self.data, self.arm)

    def _base_action(self) -> np.ndarray:
        rc = self.config.residual
        return carry_action(self._ik, self.data, self.arm, self.cube, self.goal, self._quat,
                            rc.base_gain, self.adapter.max_delta).astype(np.float64)

    def _applied_action(self, action) -> np.ndarray:
        self._residual = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        return np.clip(self._base + self.config.residual.scale * self._residual, -1.0, 1.0)

    def _extra_terms(self) -> dict:
        return {"residual": -self.config.residual.penalty * float(self._residual @ self._residual)}

    def _observation(self) -> np.ndarray:
        self._base = self._base_action()  # the base the next policy output is added to
        return np.concatenate([super()._observation(), self._base.astype(np.float32)])
