"""Load a trained run's policy as a plain function of the raw observation (frozen,
deterministic), with the run's own observation normalization."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def load_run_policy(spec: str):
    """`spec` is a run directory, optionally with "@best" (default) or "@final"."""
    from stable_baselines3 import PPO

    path, _, which = spec.partition("@")
    which = which or "best"
    run = Path(path)
    model = PPO.load(run / f"{which}_model.zip", device="cpu")
    with open(run / f"{which}_vecnormalize.pkl", "rb") as f:
        venv = pickle.load(f)
    mean, var = venv.obs_rms.mean, venv.obs_rms.var
    eps, clip = venv.epsilon, venv.clip_obs

    def act(obs: np.ndarray) -> np.ndarray:
        x = np.clip((obs - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)
        return model.predict(x, deterministic=True)[0]

    return act


def run_env_cls(spec: str) -> type:
    """The environment class a run was trained on (its config.yaml task), e.g. DoorV2Env for a
    door_v2 run, so it is evaluated on the observation it knows."""
    import yaml

    from polypanda.environments.tasks import task_spec

    run = Path(spec.partition("@")[0])
    return task_spec(yaml.safe_load((run / "config.yaml").read_text())["task"]).env_cls
