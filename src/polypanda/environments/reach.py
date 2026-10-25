"""ReachEnv: move the Panda's TCP to a 3-D target with seven joint-increment actions.

Observation (float32, world frame, schema "reach-v1-27d"); see OBS_LAYOUT for the order.
Reward terms are returned individually in info["reward_terms"]; see docs/reach_task.md.
"""

from dataclasses import dataclass, field
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
from polypanda.environments.targets import ReachTargetSampler, WorkspaceConfig
from polypanda.simulation.assets import load_panda_scene
from polypanda.simulation.panda import ARM_JOINTS, PandaIndices, reset_to_home, tcp_position

OBS_SCHEMA = "reach-v1-27d"
OBS_LAYOUT = (("q", 7), ("dq", 7), ("q_cmd", 7), ("tcp", 3), ("target_error", 3))
OBS_DIM = sum(n for _, n in OBS_LAYOUT)
OBS_SLICES = {
    name: slice(start, start + n)
    for (name, n), start in zip(
        OBS_LAYOUT, np.cumsum([0] + [n for _, n in OBS_LAYOUT[:-1]]), strict=True)
}


@dataclass(frozen=True)
class CartesianConfig:
    """Hand-motion actions (Milestone 8), used when ActionConfig.space is "cartesian"."""
    dof: int = 4                # 4: x, y, z, yaw (gripper down); 3: wrist held; 6: full rotation
    max_step: float = 0.01      # m per policy step per axis (0.2 m/s at 20 Hz)
    max_rot: float = 0.05       # rad per policy step per axis
    max_tilt: float = 0.6       # rad: 6-D targets stay within this of pointing straight down
    max_lead: float = 0.03      # m: target at most this far ahead of the commanded hand, per axis
    box_low: tuple[float, float, float] = (0.15, -0.5, 0.01)   # TCP target bounds (m)
    box_high: tuple[float, float, float] = (0.85, 0.5, 0.6)
    ik_iters: int = 2           # damped least-squares iterations per policy step
    posture_gain: float = 0.02  # null-space pull toward the home arm posture, per step
    max_accel: float | None = None  # m/s^2 limit on the hand target's change of velocity


@dataclass(frozen=True)
class ActionConfig:
    max_delta: float = 0.015
    n_substeps: int = 25
    max_lead: float | None = None
    space: str = "joint"        # "joint": 7 joint increments; "cartesian": hand motion through IK
    cartesian: CartesianConfig = field(default_factory=CartesianConfig)
    bound_penalty: float = 0.0  # >0: outputs beyond +-1 cost this x squared excess (bound_action)


@dataclass(frozen=True)
class EpisodeConfig:
    horizon: int = 200                # policy steps (10 s at 20 Hz), then truncated
    min_start_distance: float = 0.10  # m between the start TCP and a sampled target
    start_joint_noise: float = 0.0    # rad, uniform perturbation of the home pose


@dataclass(frozen=True)
class SuccessConfig:
    tolerance: float = 0.02  # m
    dwell_steps: int = 5     # consecutive policy steps within tolerance
    # Reach v1 terminated here; the policy then never saw post-success states and learned to
    # pass through targets without holding them. v2 runs the full horizon.
    terminate: bool = False


@dataclass(frozen=True)
class RewardConfig:
    progress: float = 20.0      # per metre of distance reduced
    hold: float = 0.5           # per step, scaled 1 -> 0 from target centre to tolerance edge
    success_bonus: float = 0.0  # paid at most once per episode (v1 used 5.0 with termination)
    action: float = 0.005       # times ||a||^2
    contact: float = 0.1        # per step with any contact
    joint_limit: float = 0.01   # per joint target on a ctrlrange bound
    non_finite: float = 10.0    # on numerical blow-up (episode terminates)


@dataclass(frozen=True)
class ReachConfig:
    action: ActionConfig = field(default_factory=ActionConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    success: SuccessConfig = field(default_factory=SuccessConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ReachConfig":
        return load_yaml(cls, path)


class ReachEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: ReachConfig | None = None) -> None:
        self.config = config or ReachConfig()
        self.model, self.data = load_panda_scene()
        self.idx = PandaIndices.from_model(self.model)
        self.adapter = make_action_adapter(self.model, self.idx, self.config.action)
        self.sampler = ReachTargetSampler(self.model, self.idx, self.config.workspace)
        joint_ids = [self.model.joint(name).id for name in ARM_JOINTS]
        self._q_low, self._q_high = self.model.jnt_range[joint_ids].T

        limit = action_limit(self.config.action)
        self.action_space = spaces.Box(-limit, limit, shape=(self.adapter.action_dim,),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)

        self.target = np.zeros(3)
        self._distance = 0.0
        self._dwell = 0
        self._steps = 0
        self._bonus_paid = False

    @property
    def elapsed_steps(self) -> int:
        """Policy steps taken since the last reset."""
        return self._steps

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        reset_to_home(self.model, self.data, self.idx)
        noise = self.config.episode.start_joint_noise
        if noise > 0:
            q = self.data.qpos[self.idx.arm_qpos] + self.np_random.uniform(-noise, noise, 7)
            self.data.qpos[self.idx.arm_qpos] = np.clip(q, self._q_low, self._q_high)
            mujoco.mj_forward(self.model, self.data)
        self.adapter.reset_to_measured(self.data)

        tcp = tcp_position(self.data, self.idx)
        if "target" in options:
            target = np.asarray(options["target"], dtype=float)
            if target.shape != (3,) or not np.all(np.isfinite(target)):
                raise ValueError(f"options['target'] must be 3 finite values, got {target}")
            self.target = target.copy()
        else:
            self.target = self.sampler.sample(
                self.np_random, avoid=tcp, min_distance=self.config.episode.min_start_distance,
            ).position
        self._distance = float(np.linalg.norm(self.target - tcp))
        self._dwell = 0
        self._steps = 0
        self._bonus_paid = False
        return self._observation(), {"distance": self._distance, "target": self.target.copy()}

    def step(self, action):
        cfg = self.config
        a, bound_terms = bound_action(action, self.config.action)
        self.adapter.step(self.data, a)
        self._steps += 1

        if not (np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))):
            info = {"distance": float("nan"), "is_success": False, "failure": "non_finite",
                    "reward_terms": {"non_finite": -cfg.reward.non_finite}}
            obs = np.nan_to_num(self._observation(), nan=0.0, posinf=0.0, neginf=0.0)
            return obs, -cfg.reward.non_finite, True, False, info

        distance = float(np.linalg.norm(self.target - tcp_position(self.data, self.idx)))
        self._dwell = self._dwell + 1 if distance < cfg.success.tolerance else 0
        success = self._dwell >= cfg.success.dwell_steps
        contact = self.data.ncon > 0
        joints_at_limit = int(self.adapter.at_limit.sum())

        first_success = success and not self._bonus_paid
        self._bonus_paid |= success
        terms = {
            "progress": cfg.reward.progress * (self._distance - distance),
            "hold": cfg.reward.hold * max(0.0, 1.0 - distance / cfg.success.tolerance),
            "success": cfg.reward.success_bonus if first_success else 0.0,
            "action": -cfg.reward.action * float(a @ a),
            "contact": -cfg.reward.contact if contact else 0.0,
            "joint_limit": -cfg.reward.joint_limit * joints_at_limit,
            **bound_terms,
        }
        self._distance = distance
        terminated = success and cfg.success.terminate
        truncated = not terminated and self._steps >= cfg.episode.horizon
        info = {
            "distance": distance,
            "is_success": success,
            "failure": None,
            "contact": contact,
            "joints_at_limit": joints_at_limit,
            "reward_terms": terms,
        }
        return self._observation(), float(sum(terms.values())), terminated, truncated, info

    def _observation(self) -> np.ndarray:
        tcp = tcp_position(self.data, self.idx)
        return np.concatenate([
            self.data.qpos[self.idx.arm_qpos],
            self.data.qvel[self.idx.arm_qvel],
            self.adapter.q_cmd,
            tcp,
            self.target - tcp,
        ]).astype(np.float32)
