import mujoco
import numpy as np
import pytest

from polypanda.environments.targets import ReachTargetSampler, WorkspaceConfig
from polypanda.simulation.assets import load_panda_scene
from polypanda.simulation.panda import PandaIndices, reset_to_home, tcp_position


@pytest.fixture(scope="module")
def panda():
    model, _ = load_panda_scene()
    return model, PandaIndices.from_model(model)


def _sample(panda, seed, n=50, **kwargs):
    model, idx = panda
    sampler = ReachTargetSampler(model, idx)
    rng = np.random.default_rng(seed)
    return sampler, [sampler.sample(rng, **kwargs) for _ in range(n)]


def test_same_seed_same_targets(panda):
    _, a = _sample(panda, seed=7)
    _, b = _sample(panda, seed=7)
    _, c = _sample(panda, seed=8)
    np.testing.assert_array_equal([t.position for t in a], [t.position for t in b])
    assert not np.allclose([t.position for t in a], [t.position for t in c])


def test_targets_inside_workspace_and_witness_within_limits(panda):
    sampler, targets = _sample(panda, seed=1, n=200)
    for t in targets:
        assert np.all(t.position >= sampler.low) and np.all(t.position <= sampler.high)
        assert np.all(t.witness_q >= sampler.q_low) and np.all(t.witness_q <= sampler.q_high)


def test_witness_reaches_target_without_contact(panda):
    model, idx = panda
    _, targets = _sample(panda, seed=2)
    data = mujoco.MjData(model)
    for t in targets:
        reset_to_home(model, data, idx)
        data.qpos[idx.arm_qpos] = t.witness_q
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(tcp_position(data, idx), t.position, atol=1e-12)
        assert data.ncon == 0


def test_min_distance_from_avoid_point(panda):
    avoid = np.array([0.5545, 0.0, 0.5216])  # TCP at home
    _, targets = _sample(panda, seed=3, avoid=avoid, min_distance=0.15)
    assert min(np.linalg.norm(t.position - avoid) for t in targets) >= 0.15


def test_sampling_does_not_touch_simulation_state(panda):
    model, idx = panda
    data = mujoco.MjData(model)
    reset_to_home(model, data, idx)
    before = data.qpos.copy()
    sampler = ReachTargetSampler(model, idx)
    sampler.sample(np.random.default_rng(0))
    np.testing.assert_array_equal(data.qpos, before)


def test_impossible_workspace_raises(panda):
    model, idx = panda
    sampler = ReachTargetSampler(
        model, idx, WorkspaceConfig(low=(0, 0, 5.0), high=(1, 1, 6.0), max_tries=200))
    with pytest.raises(RuntimeError, match="no valid target"):
        sampler.sample(np.random.default_rng(0))


def test_acceptance_rate_tracked(panda):
    sampler, targets = _sample(panda, seed=4, n=100)
    assert sampler.accepted == 100
    assert 0.0 < sampler.acceptance_rate < 1.0


def test_invalid_config_rejected(panda):
    model, idx = panda
    with pytest.raises(ValueError):
        ReachTargetSampler(model, idx, WorkspaceConfig(low=(1, 0, 0), high=(0, 1, 1)))
    with pytest.raises(ValueError):
        ReachTargetSampler(model, idx, WorkspaceConfig(joint_margin=5.0))
