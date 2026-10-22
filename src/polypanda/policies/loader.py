"""Load a trained policy from a run directory, refusing incompatible ones."""

from pathlib import Path

import yaml
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from polypanda.control.action_adapter import ACTION_SCHEMA
from polypanda.environments.reach import OBS_SCHEMA, ReachEnv
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


def check_compatible(metadata: dict) -> None:
    expected = {"observation_schema": OBS_SCHEMA, "action_schema": ACTION_SCHEMA,
                "robot_model_revision": MENAGERIE_COMMIT}
    problems = [f"{k}: policy has {metadata.get(k)!r}, environment expects {v!r}"
                for k, v in expected.items() if metadata.get(k) != v]
    if problems:
        raise IncompatiblePolicyError("; ".join(problems))


def load_run(run_dir: str | Path, which: str = "best") -> LoadedPolicy:
    run_dir = Path(run_dir)
    metadata = yaml.safe_load((run_dir / "metadata.yaml").read_text())
    check_compatible(metadata)
    model = ALGORITHMS[metadata["algorithm"]].load(run_dir / metadata["checkpoints"][which],
                                                   device="cpu")
    vec_normalize = None
    stats = run_dir / metadata["normalization"][which]
    if stats.exists():
        vec_normalize = VecNormalize.load(str(stats), DummyVecEnv([ReachEnv]))
        vec_normalize.training = False
        vec_normalize.norm_reward = False
    return LoadedPolicy(model, vec_normalize, metadata, which)
