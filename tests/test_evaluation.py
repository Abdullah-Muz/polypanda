from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from polypanda.control.classical_baseline import JacobianIKPolicy, RandomPolicy
from polypanda.environments.reach import ReachConfig, ReachEnv
from polypanda.evaluation.reach import evaluate, wilson_interval
from polypanda.evaluation.sets import ReachEvalSet, generate_reach_eval_set
from polypanda.simulation.assets import load_panda_scene
from polypanda.simulation.panda import PandaIndices, reset_to_home, tcp_position

SET_PATH = Path(__file__).parents[1] / "eval_sets" / "reach_nominal_v1.json"


@pytest.fixture(scope="module")
def eval_set():
    return ReachEvalSet.load(SET_PATH)


def test_committed_set_regenerates_exactly(eval_set):
    regenerated = generate_reach_eval_set(eval_set.name, len(eval_set), eval_set.seed)
    np.testing.assert_array_equal(regenerated.targets, eval_set.targets)
    assert regenerated.digest == eval_set.digest
    assert regenerated.regions == eval_set.regions


def test_set_contents_are_valid(eval_set):
    assert len(eval_set) == 200
    low, high = np.array(eval_set.workspace["low"]), np.array(eval_set.workspace["high"])
    assert np.all(eval_set.targets >= low) and np.all(eval_set.targets <= high)
    model, data = load_panda_scene()
    idx = PandaIndices.from_model(model)
    reset_to_home(model, data, idx)
    home = tcp_position(data, idx)
    assert np.linalg.norm(eval_set.targets - home, axis=1).min() >= eval_set.min_start_distance
    for p, q in zip(eval_set.targets[:10], eval_set.witness_q[:10], strict=True):
        data.qpos[idx.arm_qpos] = q
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(tcp_position(data, idx), p, atol=1e-12)


def test_eval_stream_differs_from_env_seed(eval_set):
    env = ReachEnv()
    _, info = env.reset(seed=eval_set.seed)
    assert not np.allclose(info["target"], eval_set.targets[0])


def test_tampered_file_is_rejected(tmp_path, eval_set):
    original = SET_PATH.read_text()
    text = original.replace(repr(float(eval_set.targets[0][0])), "0.5", 1)
    assert text != original  # the edit must actually happen
    bad = tmp_path / "bad.json"
    bad.write_text(text)
    with pytest.raises(ValueError, match="digest"):
        ReachEvalSet.load(bad)


def test_wilson_interval():
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)


def test_ik_evaluation_on_subset(eval_set):
    summary, episodes = evaluate(JacobianIKPolicy(), eval_set.subset(5))
    assert summary["success_rate"] == 1.0 and summary["failures"] == {}
    assert summary["held_rate_given_success"] == 1.0
    for e in episodes:
        assert e.success and e.held and e.failure is None
        assert e.steps_to_success >= 5  # dwell
        assert e.final_distance < 1e-3


def test_random_evaluation_records_timeouts(eval_set):
    summary, episodes = evaluate(RandomPolicy(seed=0), eval_set.subset(3))
    assert summary["success_rate"] == 0.0
    assert summary["failures"] == {"timeout": 3}
    assert all(e.steps_to_success is None and e.held is None for e in episodes)


def test_evaluation_does_not_terminate_on_success():
    cfg = ReachConfig()
    env = ReachEnv(replace(cfg, success=replace(cfg.success, terminate=False)))
    obs, _ = env.reset(seed=0)
    env.reset(options={"target": tcp_position(env.data, env.idx)})
    for _ in range(20):
        _, _, terminated, _, info = env.step(np.zeros(7, dtype=np.float32))
        assert not terminated
    assert info["is_success"]


def test_sb3_model_plumbing(tmp_path, eval_set):
    from stable_baselines3 import PPO

    model = PPO("MlpPolicy", ReachEnv(), n_steps=64, batch_size=32, device="cpu", seed=0)
    model.save(tmp_path / "untrained")
    loaded = PPO.load(tmp_path / "untrained", device="cpu")
    summary, episodes = evaluate(loaded, eval_set.subset(2))
    assert summary["episodes"] == 2 and len(episodes) == 2
