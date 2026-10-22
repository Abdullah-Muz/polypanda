"""Train an SB3 policy on ReachEnv and write a self-describing run directory.

Run directory layout:
  config.yaml, env_config.yaml        resolved configuration snapshots
  metadata.yaml                       policy registry entry (schemas, revisions, artifacts)
  best_model.zip, best_vecnormalize.pkl    best on the validation stream (not the test set)
  final_model.zip, final_vecnormalize.pkl
  checkpoints/, eval/evaluations.npz, tb/  (TensorBoard)
"""

import dataclasses
import hashlib
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from polypanda.control.action_adapter import ACTION_SCHEMA
from polypanda.environments.reach import OBS_SCHEMA, ReachConfig, ReachEnv
from polypanda.simulation.assets import MENAGERIE_COMMIT
from polypanda.training.callbacks import RewardTermsCallback, SaveVecNormalizeCallback
from polypanda.training.config import TrainConfig

REPO_ROOT = Path(__file__).resolve().parents[3]


def plain(obj):
    """Dataclass/tuple structure -> YAML-safe dicts and lists."""
    if dataclasses.is_dataclass(obj):
        obj = dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {k: plain(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [plain(v) for v in obj]
    return obj


def _git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT,
                              check=False).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


def _lock_hash() -> str | None:
    lock = REPO_ROOT / "uv.lock"
    return hashlib.sha256(lock.read_bytes()).hexdigest()[:16] if lock.exists() else None


def _vec_env(env_config: ReachConfig, n_envs: int, seed: int, kind: str):
    cls = SubprocVecEnv if kind == "subproc" and n_envs > 1 else DummyVecEnv
    return make_vec_env(ReachEnv, n_envs=n_envs, seed=seed, vec_env_cls=cls,
                        env_kwargs={"config": env_config})


def build_model(cfg: TrainConfig, env, tensorboard_log: str | None):
    if cfg.algorithm != "ppo":
        raise ValueError(f"unsupported algorithm {cfg.algorithm!r}")
    params = dataclasses.asdict(cfg.ppo)
    arch = list(params.pop("net_arch"))
    return PPO("MlpPolicy", env, seed=cfg.seed, device="cpu", verbose=0,
               tensorboard_log=tensorboard_log,
               policy_kwargs={"net_arch": {"pi": arch, "vf": arch}}, **params)


def train(cfg: TrainConfig, env_config: ReachConfig, run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(plain(cfg), sort_keys=False))
    (run_dir / "env_config.yaml").write_text(yaml.safe_dump(plain(env_config), sort_keys=False))

    env = VecNormalize(_vec_env(env_config, cfg.n_envs, cfg.seed, cfg.vec_env),
                       norm_obs=cfg.normalize_obs, norm_reward=cfg.normalize_reward,
                       gamma=cfg.ppo.gamma)
    eval_env = VecNormalize(_vec_env(env_config, 1, cfg.eval_seed, "dummy"),
                            norm_obs=cfg.normalize_obs, norm_reward=False, training=False)
    model = build_model(cfg, env, str(run_dir / "tb"))
    git = _git_state()  # recorded before training, so later edits cannot mislabel the run

    per_env = lambda steps: max(1, steps // cfg.n_envs)  # noqa: E731  callbacks count env.step calls
    callbacks = CallbackList([
        RewardTermsCallback(),
        EvalCallback(eval_env, n_eval_episodes=cfg.eval_episodes,
                     eval_freq=per_env(cfg.eval_freq), deterministic=True,
                     best_model_save_path=str(run_dir), log_path=str(run_dir / "eval"),
                     callback_on_new_best=SaveVecNormalizeCallback(
                         run_dir / "best_vecnormalize.pkl")),
        CheckpointCallback(save_freq=per_env(cfg.checkpoint_freq),
                           save_path=str(run_dir / "checkpoints"), name_prefix="model",
                           save_vecnormalize=True),
    ])

    start = time.perf_counter()
    model.learn(total_timesteps=cfg.total_timesteps, callback=callbacks,
                tb_log_name=cfg.algorithm)
    wall = time.perf_counter() - start
    model.save(run_dir / "final_model.zip")
    env.save(str(run_dir / "final_vecnormalize.pkl"))
    env.close()
    eval_env.close()

    metadata = {
        "policy_id": run_dir.name,
        "skill": "reach",
        "algorithm": cfg.algorithm.upper(),
        "seed": cfg.seed,
        "observation_schema": OBS_SCHEMA,
        "action_schema": ACTION_SCHEMA,
        "robot_model_revision": MENAGERIE_COMMIT,
        "training_config": "config.yaml",
        "env_config": "env_config.yaml",
        "checkpoints": {"best": "best_model.zip", "final": "final_model.zip"},
        "normalization": {"best": "best_vecnormalize.pkl", "final": "final_vecnormalize.pkl"},
        "validation_metrics": "eval/evaluations.npz",
        "total_timesteps": int(model.num_timesteps),
        "wall_clock_s": round(wall, 1),
        "steps_per_s": round(model.num_timesteps / wall, 1),
        "package_lock_hash": _lock_hash(),
        "git": git,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (run_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
    return run_dir
