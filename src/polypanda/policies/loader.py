"""Load a trained policy from a run directory, refusing incompatible ones."""

from pathlib import Path

import yaml
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from polypanda.control.action_adapter import ACTION_SCHEMA, action_schema
from polypanda.environments.tasks import TASKS
from polypanda.simulation.assets import MENAGERIE_COMMIT

ALGORITHMS = {"PPO": PPO, "SAC": SAC, "TD3": TD3}


class IncompatiblePolicyError(RuntimeError):
    pass


class LoadedPolicy:
    """SB3 model plus its frozen observation normalization; SB3-style predict()."""

    def __init__(self, model, vec_normalize: VecNormalize | None, metadata: dict, which: str):
        self.model = model
        self.vec_normalize = vec_normalize
        self.metadata = metadata
        self.which = which

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        if self.vec_normalize is not None:
            observation = self.vec_normalize.normalize_obs(observation)
        return self.model.predict(observation, state=state, episode_start=episode_start,
                                  deterministic=deterministic)


def run_env_config(run_dir: str | Path, skill: str):
    """The task configuration a run was trained with (its env_config.yaml snapshot)."""
    return TASKS[skill].config_cls.from_yaml(Path(run_dir) / "env_config.yaml")


def make_run_env(run_dir: str | Path, skill: str):
    """The run's task environment, configured as the task is evaluated (no training-only
    curriculum or randomization), with the action interface the policy was trained on."""
    config = run_env_config(run_dir, skill)
    if hasattr(config, "for_evaluation"):
        config = config.for_evaluation()
    return TASKS[skill].env_cls(config)


def check_compatible(metadata: dict, expected_action_schema: str = ACTION_SCHEMA) -> None:
    """expected_action_schema: what the environment the policy will drive takes (by default
    joint increments; a run's own environment takes action_schema(its config.action))."""
    skill = metadata.get("skill")
    if skill not in TASKS:
        raise IncompatiblePolicyError(f"skill: unknown skill {skill!r}")
    expected = {"observation_schema": TASKS[skill].obs_schema,
                "action_schema": expected_action_schema,
                "robot_model_revision": MENAGERIE_COMMIT}
    problems = [f"{k}: policy has {metadata.get(k)!r}, environment expects {v!r}"
                for k, v in expected.items() if metadata.get(k) != v]
    if problems:
        raise IncompatiblePolicyError("; ".join(problems))


def _in_progress_metadata(run_dir: Path) -> dict:
    """Provisional metadata for a run still training (metadata.yaml is written at the end)."""
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    skill = cfg.get("task", "reach")
    return {"policy_id": run_dir.name, "skill": skill, "algorithm": cfg["algorithm"].upper(),
            "seed": cfg["seed"], "observation_schema": TASKS[skill].obs_schema,
            "action_schema": action_schema(run_env_config(run_dir, skill).action),
            "robot_model_revision": MENAGERIE_COMMIT,
            "checkpoints": {"best": "best_model.zip"},
            "normalization": {"best": "best_vecnormalize.pkl"},
            "total_timesteps": 0, "in_progress": True}


def load_run(run_dir: str | Path, which: str = "best") -> LoadedPolicy:
    """Load a finished run, or the current best checkpoint of a run still training."""
    run_dir = Path(run_dir)
    meta_file = run_dir / "metadata.yaml"
    if meta_file.exists():
        metadata = yaml.safe_load(meta_file.read_text())
    elif which == "best" and (run_dir / "best_model.zip").exists():
        metadata = _in_progress_metadata(run_dir)
    else:
        raise FileNotFoundError(f"{run_dir}: no metadata.yaml and no best checkpoint yet")
    skill = metadata.get("skill")
    config = (run_env_config(run_dir, skill)
              if skill in TASKS and (run_dir / "env_config.yaml").exists() else None)
    check_compatible(metadata, action_schema(config.action) if config else ACTION_SCHEMA)
    model = ALGORITHMS[metadata["algorithm"]].load(run_dir / metadata["checkpoints"][which],
                                                   device="cpu")
    vec_normalize = None
    stats = run_dir / metadata["normalization"][which]
    if stats.exists():
        env_cls = TASKS[skill].env_cls
        vec_normalize = VecNormalize.load(str(stats), DummyVecEnv([lambda: env_cls(config)]))
        # (config None: the task's defaults; only the observation space is checked here)
        vec_normalize.training = False
        vec_normalize.norm_reward = False
    return LoadedPolicy(model, vec_normalize, metadata, which)
