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
    log_std_init: float = 0.0    # initial log of the action noise std (0 -> std 1, SB3 default)


@dataclass(frozen=True)
class SACParams:
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    learning_starts: int = 10_000    # random-action transitions before the first update
    batch_size: int = 256
    tau: float = 0.005               # target-network Polyak averaging rate
    gamma: float = 0.99
    train_freq: int = 1              # env.step calls between update phases
    gradient_steps: int = -1         # -1: one gradient step per collected transition (UTD 1)
    ent_coef: str | float = "auto"   # entropy temperature, tuned automatically
    net_arch: tuple[int, ...] = (128, 128)


@dataclass(frozen=True)
class TrainConfig:
    algorithm: str = "ppo"
    task: str = "reach"              # key of polypanda.environments.tasks.TASKS
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
    torch_threads: int | None = None  # trainer's PyTorch threads; None = PyTorch default
    # Warm start (v3): begin from another run's network weights and observation/return
    # normalization ("{seed}" is replaced by this run's seed), e.g. runs/grasp_cart_seed{seed}.
    init_from: str | None = None
    init_checkpoint: str = "best"    # "best" or "final"
    # Warm start: reset the policy's action noise (log std, every action) to this value. A
    # trained policy's noise is small (std 0.1-0.5), too small to explore a new behaviour.
    init_log_std: float | None = None
    ppo: PPOParams = field(default_factory=PPOParams)
    sac: SACParams = field(default_factory=SACParams)

    @property
    def algo_params(self) -> PPOParams | SACParams:
        try:
            return {"ppo": self.ppo, "sac": self.sac}[self.algorithm]
        except KeyError:
            raise ValueError(f"unsupported algorithm {self.algorithm!r}") from None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        return load_yaml(cls, path)
