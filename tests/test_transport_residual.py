from pathlib import Path

import numpy as np
import pytest

from polypanda.environments.handoff import HandoffPool
from polypanda.environments.transport import (
    OBS_DIM,
    TransportEnv,
    TransportResidualEnv,
)
from polypanda.evaluation.transport import ScriptedCarryDriver, evaluate_transport
from polypanda.training.config import TrainConfig

ROOT = Path(__file__).parents[1]
EVAL = ROOT / "eval_sets" / "transport_nominal_v1.npz"


class ZeroResidual:
    def reset(self, env):
        pass

    def __call__(self, env, obs):
        return np.zeros(7, dtype=np.float32)


def _cases(idx):
    pool = HandoffPool.load(EVAL)
    return HandoffPool(pool.states[idx], tuple(pool.sources[i] for i in idx),
                       pool.placements[idx], pool.goals[idx], pool.meta)


def test_zero_residual_reproduces_scripted_carry_exactly():
    cases = _cases([0, 55, 110, 165])
    _, scripted = evaluate_transport(ScriptedCarryDriver(), cases)
    _, residual = evaluate_transport(ZeroResidual(), cases, env_cls=TransportResidualEnv)
    for s, r in zip(scripted, residual, strict=True):
        assert s.success == r.success
        assert s.final_distance_mm == pytest.approx(r.final_distance_mm, abs=1e-9)
        assert s.first_success_s == r.first_success_s


def test_observation_appends_the_base_action():
    env = TransportResidualEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM + 7,) and env.observation_space.contains(obs)
    np.testing.assert_allclose(obs[OBS_DIM:], env._base, atol=1e-6)
    plain = TransportEnv()
    plain_obs, _ = plain.reset(seed=0)
    np.testing.assert_allclose(obs[:OBS_DIM], plain_obs, atol=1e-6)


def test_residual_is_scaled_clipped_and_penalised():
    env = TransportResidualEnv()
    env.reset(seed=1)
    base = env._base.copy()
    applied = env._applied_action(np.full(7, 1.0))
    np.testing.assert_allclose(applied, np.clip(base + 0.3, -1, 1))
    np.testing.assert_allclose(env._applied_action(np.full(7, 5.0)), applied)  # input clipped
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.full(7, 0.5, dtype=np.float32))
    assert info["reward_terms"]["residual"] == pytest.approx(-0.01 * 7 * 0.25)


def test_residual_task_registered_and_config_loads():
    from polypanda.environments.tasks import task_spec
    cfg = TrainConfig.from_yaml(ROOT / "configs" / "transport_residual_ppo.yaml")
    spec = task_spec(cfg.task)
    assert spec.env_cls is TransportResidualEnv
    assert spec.obs_schema == "transport-residual-v1-54d"
    assert cfg.ppo.log_std_init == pytest.approx(-0.7)


def test_log_std_init_reaches_the_policy(tmp_path):
    from dataclasses import replace

    from stable_baselines3.common.vec_env import DummyVecEnv

    from polypanda.training.config import PPOParams
    from polypanda.training.runner import build_model
    cfg = TrainConfig(task="transport_residual", n_envs=1, vec_env="dummy",
                      ppo=replace(PPOParams(), log_std_init=-0.7))
    model = build_model(cfg, DummyVecEnv([TransportResidualEnv]), None)
    np.testing.assert_allclose(model.policy.log_std.detach().numpy(), -0.7)
