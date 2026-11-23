"""v4.0 demonstrations first: recording the feel script and cloning it (training/bc.py), and
loading a run's policy (evaluation/policies.py)."""

import numpy as np

from polypanda.environments.door import OBS_DIM, DoorConfig
from polypanda.evaluation.policies import load_run_policy
from polypanda.training.bc import (
    CloneConfig,
    RecordConfig,
    _record_chunk,
    clone,
    discounted_returns,
    save_vecnormalize,
)
from polypanda.training.config import TrainConfig


def test_discounted_returns_restart_at_each_episode_end():
    rew = np.array([1.0, 1.0, 1.0, 2.0, 2.0])
    done = np.array([False, False, True, False, True])
    ret = discounted_returns(rew, done, 0.5)
    np.testing.assert_allclose(ret, [1.75, 1.5, 1.0, 3.0, 2.0])


def test_recording_keeps_successful_episodes_with_the_scripts_clean_actions():
    seeds = [RecordConfig.seed * 1000 + k for k in range(3)]
    out = _record_chunk((seeds, 1.0, 0.2))          # all three with noise
    assert out["total"] == 3 and out["kept"] == len(out["obs"]) >= 2
    for o, a, d in zip(out["obs"], out["act"], out["done"], strict=True):
        assert o.shape[1] == OBS_DIM and a.shape == (len(o), 5)
        assert d[-1] and not d[:-1].any()
        assert np.all(np.abs(a) <= 1.0 + 1e-6)
        assert np.all(a[:, 4] == -1.0)                 # the script keeps the fingers open


def test_a_clone_saves_and_loads_as_a_run(tmp_path):
    out = _record_chunk(([RecordConfig.seed * 1000 + 7], 0.0, 0.0))
    demos = {k: np.concatenate(out[k]) for k in ("obs", "act", "rew", "episode", "done")}
    train_cfg = TrainConfig.from_yaml("configs/door_v1_ppo.yaml")
    env_config = DoorConfig.from_yaml(train_cfg.env_config)
    model, venv, log = clone(demos, train_cfg, env_config,
                             CloneConfig(epochs=2, value_epochs=1, val_share=0.0))
    assert len(log["train_mse"]) == 2 and log["train_mse"][-1] < log["train_mse"][0] + 1.0
    assert float(model.policy.log_std.mean()) == CloneConfig.log_std
    model.save(tmp_path / "best_model.zip")
    save_vecnormalize(venv, tmp_path / "best_vecnormalize.pkl")
    act = load_run_policy(str(tmp_path))
    a = act(demos["obs"][0])
    assert a.shape == (5,) and np.all(np.isfinite(a))
