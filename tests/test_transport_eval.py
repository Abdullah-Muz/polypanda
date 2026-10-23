from pathlib import Path

import numpy as np

from polypanda.environments.handoff import HandoffPool
from polypanda.evaluation.transport import ScriptedCarryDriver, evaluate_transport

EVAL = Path(__file__).parents[1] / "eval_sets" / "transport_nominal_v1.npz"


class RandomDriver:
    def __init__(self):
        self.rng = np.random.default_rng(0)

    def reset(self, env):
        pass

    def __call__(self, env, obs):
        return self.rng.uniform(-1, 1, 7).astype(np.float32)


class StillDriver:
    def reset(self, env):
        pass

    def __call__(self, env, obs):
        return np.zeros(7, dtype=np.float32)


def _one_per_source():
    pool = HandoffPool.load(EVAL)
    idx = [0, 50, 100, 150]
    return HandoffPool(pool.states[idx], tuple(pool.sources[i] for i in idx),
                       pool.placements[idx], pool.goals[idx], pool.meta)


def test_scripted_carry_succeeds_from_every_grasp_source():
    summary, episodes = evaluate_transport(ScriptedCarryDriver(), _one_per_source())
    assert summary["success_rate"] == 1.0 and summary["drop_rate"] == 0.0
    assert len(summary["by_source"]) == 4
    for e in episodes:
        assert e.success and e.final_distance_mm < 20.0 and e.failure is None


def test_failure_classification():
    _, random_eps = evaluate_transport(RandomDriver(), _one_per_source())
    assert all(e.failure == "dropped" and e.drop_s is not None for e in random_eps)
    _, still_eps = evaluate_transport(StillDriver(), _one_per_source())
    assert all(e.failure == "never_reached" and not e.dropped for e in still_eps)
