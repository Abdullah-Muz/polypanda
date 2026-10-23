"""End-to-end smoke test of the training pipeline on a tiny budget (no learning expected)."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from polypanda.environments.grasp import GraspConfig
from polypanda.environments.reach import ReachConfig
from polypanda.evaluation.grasp import PolicyDriver, evaluate_grasp_policy
from polypanda.evaluation.reach import evaluate
from polypanda.evaluation.sets import GraspEvalSet, ReachEvalSet
from polypanda.policies.loader import IncompatiblePolicyError, load_run
from polypanda.training.config import PPOParams, SACParams, TrainConfig
from polypanda.training.runner import train

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    cfg = TrainConfig(n_envs=2, vec_env="dummy", total_timesteps=1024, eval_freq=512,
                      eval_episodes=2, checkpoint_freq=512,
                      ppo=PPOParams(n_steps=256, batch_size=64, n_epochs=2))
    return train(cfg, ReachConfig(), tmp_path_factory.mktemp("runs") / "smoke")


def test_yaml_configs_load():
    ppo = TrainConfig.from_yaml(ROOT / "configs" / "reach_ppo.yaml")
    assert ppo.algorithm == "ppo" and ppo.n_envs == 8 and ppo.algo_params.net_arch == (128, 128)
    sac = TrainConfig.from_yaml(ROOT / "configs" / "reach_sac.yaml")
    assert sac.algorithm == "sac" and not sac.normalize_reward and sac.torch_threads == 1
    assert sac.algo_params.ent_coef == "auto" and sac.algo_params.net_arch == (128, 128)


def test_grasp_task_end_to_end(tmp_path):
    cfg = TrainConfig(task="grasp", env_config="configs/grasp.yaml", n_envs=1, vec_env="dummy",
                      total_timesteps=512, eval_freq=256, eval_episodes=1, checkpoint_freq=512,
                      ppo=PPOParams(n_steps=256, batch_size=64, n_epochs=1))
    run_dir = train(cfg, GraspConfig(), tmp_path / "grasp_smoke")
    meta = yaml.safe_load((run_dir / "metadata.yaml").read_text())
    assert meta["skill"] == "grasp" and meta["observation_schema"] == "grasp-v1-41d"
    policy = load_run(run_dir, "final")
    eval_set = GraspEvalSet.load(ROOT / "eval_sets" / "grasp_nominal_v1.json").subset(1)
    summary, episodes = evaluate_grasp_policy(PolicyDriver(policy), eval_set)
    assert summary["episodes"] == 1 and episodes[0].failure is not None  # untrained: fails


def test_task_and_env_config_must_match(tmp_path):
    cfg = TrainConfig(task="grasp", n_envs=1, vec_env="dummy", total_timesteps=1)
    with pytest.raises(TypeError, match="GraspConfig"):
        train(cfg, ReachConfig(), tmp_path / "mismatch")
    assert not (tmp_path / "mismatch").exists()


def test_grasp_ppo_config_loads():
    cfg = TrainConfig.from_yaml(ROOT / "configs" / "grasp_ppo.yaml")
    assert cfg.task == "grasp" and cfg.algorithm == "ppo" and cfg.n_envs == 8


def test_sac_end_to_end(tmp_path):
    cfg = TrainConfig(algorithm="sac", n_envs=1, vec_env="dummy", total_timesteps=400,
                      normalize_reward=False, eval_freq=200, eval_episodes=1,
                      checkpoint_freq=200,
                      sac=SACParams(buffer_size=1000, learning_starts=100, batch_size=32))
    run_dir = train(cfg, ReachConfig(), tmp_path / "sac_smoke")
    meta = yaml.safe_load((run_dir / "metadata.yaml").read_text())
    assert meta["algorithm"] == "SAC" and meta["total_timesteps"] >= 400
    policy = load_run(run_dir, "final")
    eval_set = ReachEvalSet.load(ROOT / "eval_sets" / "reach_nominal_v1.json").subset(1)
    summary, _ = evaluate(policy, eval_set)
    assert summary["episodes"] == 1


def test_run_directory_contents(run_dir):
    for name in ("config.yaml", "env_config.yaml", "metadata.yaml", "best_model.zip",
                 "best_vecnormalize.pkl", "final_model.zip", "final_vecnormalize.pkl",
                 "eval/evaluations.npz"):
        assert (run_dir / name).exists(), name
    assert any((run_dir / "checkpoints").iterdir())
    assert any((run_dir / "tb").rglob("events.out.tfevents.*"))
    meta = yaml.safe_load((run_dir / "metadata.yaml").read_text())
    assert meta["observation_schema"] == "reach-v1-27d"
    assert meta["action_schema"] == "joint-increment-v1-7d"
    assert meta["total_timesteps"] >= 1024 and meta["package_lock_hash"]
    snapshot = TrainConfig.from_yaml(run_dir / "config.yaml")
    assert snapshot.total_timesteps == 1024


def test_reward_terms_logged_to_tensorboard(run_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = next((run_dir / "tb").rglob("events.out.tfevents.*"))
    acc = EventAccumulator(str(events))
    acc.Reload()
    tags = set(acc.Tags()["scalars"])
    assert {"reward_terms/progress", "reward_terms/action", "rollout/mean_distance_m",
            "eval/success_rate"} <= tags


@pytest.mark.parametrize("which", ["best", "final"])
def test_loaded_policy_evaluates(run_dir, which):
    policy = load_run(run_dir, which)
    assert policy.vec_normalize is not None and not policy.vec_normalize.training
    eval_set = ReachEvalSet.load(ROOT / "eval_sets" / "reach_nominal_v1.json").subset(2)
    summary, episodes = evaluate(policy, eval_set)
    assert summary["episodes"] == 2


def test_normalization_is_applied(run_dir):
    policy = load_run(run_dir)
    obs = np.ones(27, dtype=np.float32)
    normalized = policy.vec_normalize.normalize_obs(obs)
    assert not np.allclose(normalized, obs)


def test_incompatible_policy_rejected(run_dir, tmp_path):
    bad = tmp_path / "bad_run"
    bad.mkdir()
    meta = yaml.safe_load((run_dir / "metadata.yaml").read_text())
    for name in ("best_model.zip", "best_vecnormalize.pkl"):
        (bad / name).write_bytes((run_dir / name).read_bytes())
    (bad / "metadata.yaml").write_text(
        yaml.safe_dump({**meta, "observation_schema": "reach-v0-24d"}))
    with pytest.raises(IncompatiblePolicyError, match="observation_schema"):
        load_run(bad)


def test_unsupported_algorithm_rejected(tmp_path):
    with pytest.raises(ValueError, match="unsupported"):
        train(replace(TrainConfig(), algorithm="a2c", vec_env="dummy", n_envs=1,
                      total_timesteps=1), ReachConfig(), tmp_path / "x")
