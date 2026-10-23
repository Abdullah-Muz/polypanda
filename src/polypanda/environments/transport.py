"""TransportEnv: carry a grasped cube to a 3-D goal and hold it there.

Episodes start from grasp-to-transport handoff states (polypanda.environments.handoff) produced
by both scripted and learned grasp controllers, so transport sees the start distribution it will
receive in pick-and-place. The gripper stays closed; releasing belongs to the place skill.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from polypanda.config import load_yaml
from polypanda.environments.reach import ActionConfig
from polypanda.simulation.grasp import GraspCheckConfig


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
    drop: float = 5.0           # once; the episode terminates


@dataclass(frozen=True)
class TransportConfig:
    action: ActionConfig = field(default_factory=ActionConfig)
    goal: GoalConfig = field(default_factory=GoalConfig)
    episode: TransportEpisodeConfig = field(default_factory=TransportEpisodeConfig)
    success: TransportSuccessConfig = field(default_factory=TransportSuccessConfig)
    reward: TransportRewardConfig = field(default_factory=TransportRewardConfig)
    check: GraspCheckConfig = field(default_factory=GraspCheckConfig)
    start_pool: str = "data/handoff/transport_train_v1.npz"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TransportConfig":
        return load_yaml(cls, path)
