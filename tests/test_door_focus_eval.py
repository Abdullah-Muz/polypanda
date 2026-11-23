"""v4.0 door, dev round 2: the training-only focus on chosen cells, the once-per-slam penalty
(environments/door.py) and validation on fixed episodes (training/fixed_eval.py)."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from polypanda.environments.door import DoorConfig, DoorEnv, DoorFocusConfig, DoorRewardConfig
from polypanda.training.config import TrainConfig
from polypanda.training.fixed_eval import FixedEpisodes, summarize_cells

FOCUS = DoorFocusConfig(prob=1.0, cells=("open/out/spring",))


def test_focus_draws_only_its_cells_and_evaluation_drops_it():
    env = DoorEnv(DoorConfig(focus=FOCUS))
    rng = np.random.default_rng(3)
    for _ in range(30):
        ep = env.draw_episode(rng)
        assert (ep["command"], ep["slide"]) == ("open", "out") and ep["spring"] > 0
        assert ep["lid"] == 0.0
        assert ep["damping"] >= ep["spring"] * ep["travel"] / 0.2 - 1e-9
    assert DoorConfig(focus=FOCUS).for_evaluation().focus == DoorFocusConfig()
    plain = DoorEnv(DoorConfig(focus=replace(FOCUS, cells=("close/left/plain",))))
    ep = plain.draw_episode(np.random.default_rng(4))
    assert (ep["command"], ep["slide"], ep["spring"]) == ("close", "left", 0.0)


def test_focus_off_keeps_the_episode_stream():
    a = DoorEnv(DoorConfig())
    b = DoorEnv(DoorConfig(focus=replace(FOCUS, prob=0.0)))
    ra, rb = np.random.default_rng(9), np.random.default_rng(9)
    for _ in range(20):
        assert a.draw_episode(ra) == b.draw_episode(rb)


@pytest.mark.parametrize("cell", ["open/up/spring", "open/out", "push/out/plain"])
def test_focus_cells_are_checked(cell):
    with pytest.raises(ValueError):
        DoorFocusConfig(prob=0.5, cells=(cell,))


def test_a_slam_costs_slam_hit_once_on_top():
    env = DoorEnv(DoorConfig(reward=DoorRewardConfig(slam_hit=2.0)))
    ep = env.draw_episode(np.random.default_rng(0), "open")
    ep |= {"slide": "out", "travel": 0.27, "spring": 0.0, "friction": 0.2, "damping": 1.0}
    env.reset(seed=0, options={"episode": ep})
    m = env.machine
    env.data.qpos[m.lid_qpos] = 0.27 - 0.02
    env.data.qvel[m.lid_dof] = 0.8
    mujoco.mj_forward(env.model, env.data)
    env._prev = env._potentials()
    q_prev = m.lid
    a = np.zeros(5, dtype=np.float32)
    a[4] = -1.0
    _, _, _, _, info = env.step(a)
    r, dt = env.config.reward, env.adapter.control_dt
    excess = (info["lid"] - q_prev) / dt - r.slam_open_speed
    assert excess > 0 and info["stats"]["slams"] == 1
    assert info["reward_terms"]["slam"] == pytest.approx(-r.slam * excess - 2.0)


def test_fixed_episodes_replay_the_same_machines_and_log_each():
    env = FixedEpisodes(DoorEnv(replace(DoorConfig(), horizon=3)), n=4, seed=123)
    a = np.zeros(5, dtype=np.float32)

    def play(k):
        firsts = []
        for _ in range(k):
            obs, _ = env.reset()
            firsts.append(obs.copy())
            done = False
            while not done:
                _, _, term, trunc, _ = env.step(a)
                done = term or trunc
        return firsts
    first = play(4)
    rows = env.get_rows()
    assert [r["index"] for r in rows] == [0, 1, 2, 3]
    assert [r["command"] for r in rows] == [e["command"] for e in env.episodes]
    env.rewind()
    again = play(4)
    for x, y in zip(first, again, strict=True):
        np.testing.assert_array_equal(x, y)
    s = summarize_cells(env.get_rows())
    assert s["n"] == 4 and s["success"] == 0 and sum(
        int(v.split("/")[1]) for v in s["cells"].values()) == 4


def test_train_config_reads_eval_fixed_episodes():
    cfg = TrainConfig.from_yaml("configs/door_v1_bc_ppo_r2.yaml")
    assert cfg.eval_fixed_episodes and cfg.init_from == "runs/door_v1_bc_ppo_seed{seed}"
    env_cfg = DoorConfig.from_yaml(cfg.env_config)
    assert env_cfg.focus.cells == ("open/out/spring",) and env_cfg.reward.slam_hit > 0
