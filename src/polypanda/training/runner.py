"""Train an SB3 policy on a registered task and write a self-describing run directory.

Run directory layout:
  config.yaml, env_config.yaml        resolved configuration snapshots
  metadata.yaml                       policy registry entry (schemas, revisions, artifacts)
  best_model.zip, best_vecnormalize.pkl    best on the validation stream (not the test set)
  final_model.zip, final_vecnormalize.pkl
  checkpoints/, eval/evaluations.npz, tb/  (TensorBoard)
"""

import dataclasses
import hashlib
import pickle
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import yaml
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from polypanda.control.action_adapter import action_schema
from polypanda.environments.tasks import task_spec
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


def _vec_env(env_cls: type, env_config, n_envs: int, seed: int, kind: str):
    cls = SubprocVecEnv if kind == "subproc" and n_envs > 1 else DummyVecEnv
    return make_vec_env(env_cls, n_envs=n_envs, seed=seed, vec_env_cls=cls,
                        env_kwargs={"config": env_config})


def build_model(cfg: TrainConfig, env, tensorboard_log: str | None):
    params = dataclasses.asdict(cfg.algo_params)
    arch = list(params.pop("net_arch"))
    common = {"seed": cfg.seed, "device": "cpu", "verbose": 0,
              "tensorboard_log": tensorboard_log}
    if cfg.algorithm == "ppo":
        policy_kwargs = {"net_arch": {"pi": arch, "vf": arch},
                         "log_std_init": params.pop("log_std_init")}
        return PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, **common, **params)
    return SAC("MlpPolicy", env, policy_kwargs={"net_arch": {"pi": arch, "qf": arch}},
               **common, **params)


def _warm_start(cfg: TrainConfig, model, env: VecNormalize) -> dict:
    """Start from another run: its network weights and its observation and return
    normalization (the weights expect inputs scaled as in their own training). The source must
    have the same observation and action sizes."""
    source = Path(cfg.init_from.format(seed=cfg.seed))
    which = cfg.init_checkpoint
    loaded = type(model).load(source / f"{which}_model.zip", device="cpu")
    if (loaded.observation_space.shape != model.observation_space.shape
            or loaded.action_space.shape != model.action_space.shape):
        raise ValueError(f"{source}: observation/action sizes differ from this task's")
    model.policy.load_state_dict(loaded.policy.state_dict())
    with open(source / f"{which}_vecnormalize.pkl", "rb") as f:
        stats = pickle.load(f)
    env.obs_rms, env.ret_rms = stats.obs_rms, stats.ret_rms
    meta = yaml.safe_load((source / "metadata.yaml").read_text())
    return {"run": str(source), "checkpoint": which, "policy_id": meta["policy_id"],
            "git": meta.get("git", {}).get("commit")}


def train(cfg: TrainConfig, env_config, run_dir: Path) -> Path:
    gamma = cfg.algo_params.gamma  # also rejects unsupported algorithms before any setup
    task = task_spec(cfg.task)
    if not isinstance(env_config, task.config_cls):
        raise TypeError(f"task {cfg.task!r} needs {task.config_cls.__name__}, "
                        f"got {type(env_config).__name__}")
    if cfg.torch_threads is not None:
        torch.set_num_threads(cfg.torch_threads)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(plain(cfg), sort_keys=False))
    (run_dir / "env_config.yaml").write_text(yaml.safe_dump(plain(env_config), sort_keys=False))

    env = VecNormalize(_vec_env(task.env_cls, env_config, cfg.n_envs, cfg.seed, cfg.vec_env),
                       norm_obs=cfg.normalize_obs, norm_reward=cfg.normalize_reward,
                       gamma=gamma)
    # Validation always measures the task as evaluated (e.g. no training curriculum starts).
    eval_config = env_config.for_evaluation() if hasattr(env_config, "for_evaluation") \
        else env_config
    eval_env = VecNormalize(_vec_env(task.env_cls, eval_config, 1, cfg.eval_seed, "dummy"),
                            norm_obs=cfg.normalize_obs, norm_reward=False, training=False)
    model = build_model(cfg, env, str(run_dir / "tb"))
    init = _warm_start(cfg, model, env) if cfg.init_from else None
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
        "skill": cfg.task,
        "algorithm": cfg.algorithm.upper(),
        "seed": cfg.seed,
        "observation_schema": task.obs_schema,
        "action_schema": action_schema(env_config.action),
        "robot_model_revision": MENAGERIE_COMMIT,
        "training_config": "config.yaml",
        "env_config": "env_config.yaml",
        "checkpoints": {"best": "best_model.zip", "final": "final_model.zip"},
        "normalization": {"best": "best_vecnormalize.pkl", "final": "final_vecnormalize.pkl"},
        "validation_metrics": "eval/evaluations.npz",
        "total_timesteps": int(model.num_timesteps),
        "wall_clock_s": round(wall, 1),
        "steps_per_s": round(model.num_timesteps / wall, 1),
        "torch_threads": torch.get_num_threads(),
        "package_lock_hash": _lock_hash(),
        "git": git,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **({"init_from": init} if init else {}),
    }
    (run_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
    return run_dir
