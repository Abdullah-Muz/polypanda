"""Training run configuration (configs/reach_*.yaml)."""

from dataclasses import dataclass, field
from pathlib import Path

from polypanda.config import load_yaml


@dataclass(frozen=True)
class PPOParams:
    learning_rate: float = 3e-4
    n_steps: int = 512           # per env per rollout; rollout = n_steps * n_envs transitions
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: tuple[int, ...] = (128, 128)


@dataclass(frozen=True)
class TrainConfig:
    algorithm: str = "ppo"
    env_config: str = "configs/reach.yaml"
    seed: int = 0
    n_envs: int = 8
    vec_env: str = "subproc"         # "subproc" or "dummy"
    total_timesteps: int = 1_000_000
    normalize_obs: bool = True
    normalize_reward: bool = True
    eval_freq: int = 20_000          # env transitions between validation evaluations
    eval_episodes: int = 50
    eval_seed: int = 100_000         # validation target stream; never the held-out test set
    checkpoint_freq: int = 100_000   # env transitions between checkpoints
    ppo: PPOParams = field(default_factory=PPOParams)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        return load_yaml(cls, path)
