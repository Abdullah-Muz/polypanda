"""The any-pose grasp (v4, v3 recovery): cubes resting upright or lying on a side."""

from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from polypanda.environments.grasp import OBS_SLICES, GraspConfig, GraspV2Env, GraspV4Env

ROOT = Path(__file__).parents[1]
CFG = GraspConfig.from_yaml(ROOT / "configs" / "grasp_anypose_cartesian.yaml")


def test_explicit_placements_rest_lying_or_upright():
    env = GraspV4Env(CFG.for_evaluation())
    obs, info = env.reset(seed=0, options={"cube": (0.5, 0.1, 0.7), "lying": True})
    assert info["lying"] and abs(obs[OBS_SLICES["cube_tilt"]][0] - np.pi / 2) < 1e-3
    assert abs(obs[OBS_SLICES["cube_height"]][0]) < 1e-3       # lift from its own resting height
    axis = env.data.xmat[env.cube.body].reshape(3, 3)[:, 2]
    assert abs(np.arctan2(axis[1], axis[0]) - 0.7) < 1e-6     # long axis along the given yaw
    obs, info = env.reset(seed=0, options={"cube": (0.5, 0.1, 0.7)})
    assert not info["lying"] and obs[OBS_SLICES["cube_tilt"]][0] < 1e-3


def test_random_placements_lie_at_the_configured_rate():
    env = GraspV4Env(CFG.for_evaluation())
    lying = [env.reset(seed=s)[1]["lying"] for s in range(80)]
    assert 0.4 < np.mean(lying) < 0.8


@pytest.mark.parametrize("lying", [True, False])
def test_a_pregrasp_start_lifts_to_success(lying):
    cfg = replace(CFG, curriculum=replace(CFG.curriculum, pregrasp_start_prob=1.0),
                  randomization=replace(CFG.randomization, mass_scale=(1, 1),
                                        friction_scale=(1, 1)))
    env = GraspV4Env(replace(cfg, resting=replace(cfg.resting,
                                                  lying_prob=1.0 if lying else 0.0)))
    _, info = env.reset(seed=3)
    assert info["start"] == "pregrasp" and info["lying"] == lying and env.closed
    up = np.zeros(env.action_space.shape, dtype=np.float32)
    up[2] = 0.3  # 6 cm/s: in the weak training grip a full-speed lift pivots the cube (v2 too)
    success = False
    for _ in range(40):
        _, _, terminated, _, info = env.step(up)
        assert not terminated
        success |= info["is_success"]
    assert success and info["lift"] > 0.05


def test_a_lying_cube_may_turn_on_the_table_but_not_tip_up():
    env = GraspV4Env(CFG.for_evaluation())
    env.reset(seed=0, options={"cube": (0.5, 0.0, 0.0), "lying": True})
    m = env._measure()
    assert not env._knocked(m)
    c = env.cube
    q = env.data.qpos[c.qpos + 3:c.qpos + 7].copy()
    turn = np.array([np.cos(0.4), 0.0, 0.0, np.sin(0.4)])     # 46 deg about vertical
    mujoco.mju_mulQuat(env.data.qpos[c.qpos + 3:c.qpos + 7], turn, q)
    mujoco.mj_forward(env.model, env.data)
    assert not env._knocked(env._measure())
    tip = np.array([np.cos(0.35), 0.0, np.sin(0.35), 0.0])    # 40 deg about y: long axis up
    mujoco.mju_mulQuat(env.data.qpos[c.qpos + 3:c.qpos + 7], tip, q)
    mujoco.mj_forward(env.model, env.data)
    assert env._knocked(env._measure())


def test_lying_eval_sets_round_trip_and_keep_old_digests(tmp_path):
    from polypanda.evaluation.sets import GraspEvalSet, generate_grasp_eval_set
    nominal = GraspEvalSet.load(ROOT / "eval_sets" / "grasp_nominal_v1.json")  # (digest checked)
    assert not nominal.lying
    lying = generate_grasp_eval_set("t", 20, 1, x_range=(0.3, 0.78), y_range=(-0.2, 0.2),
                                    lying=True)
    upright = generate_grasp_eval_set("t", 20, 1, x_range=(0.3, 0.78), y_range=(-0.2, 0.2))
    assert lying.digest != upright.digest
    lying.save(tmp_path / "s.json")
    back = GraspEvalSet.load(tmp_path / "s.json")
    assert back.lying and back.digest == lying.digest and back.subset(5).lying


CFG2 = GraspConfig.from_yaml(ROOT / "configs" / "grasp_anypose2_cartesian.yaml")


def test_preclose_starts_hover_open_above_the_cube_in_training_only():
    cfg = replace(CFG2, curriculum=replace(CFG2.curriculum, pregrasp_start_prob=0.0,
                                           preclose_start_prob=1.0))
    env = GraspV4Env(cfg)
    starts = 0
    for s in range(12):
        _, info = env.reset(seed=s)
        if info["start"] != "preclose":
            continue                                  # (IK can fail: then a home start)
        starts += 1
        m = env._measure()
        assert not env.closed and np.abs(m["rel"][:2]).max() < 0.012  # up to 1 cm in x and y
        assert -0.055 < m["rel"][2] < -0.015          # hand 2-5 cm above the cube's centre
        assert abs(m["align"]) < 0.45                 # within 25 deg (+ a little IK slack)
    assert starts >= 9
    ev = GraspV4Env(cfg.for_evaluation())
    assert all(ev.reset(seed=s)[1]["start"] == "home" for s in range(8))


def test_the_lying_height_window_reaches_lower():
    env = GraspV4Env(CFG2.for_evaluation())
    env.reset(seed=0, options={"cube": (0.5, 0.0, 0.0), "lying": True})
    assert env._height_ok(0.013) and env._height_ok(-0.009) and not env._height_ok(-0.012)
    assert not env._height_ok(0.016)
    env.reset(seed=0, options={"cube": (0.5, 0.0, 0.0)})            # upright: symmetric
    assert not env._height_ok(0.013) and env._height_ok(0.009)
    round1 = GraspV4Env(CFG.for_evaluation())                        # no window: symmetric
    round1.reset(seed=0, options={"cube": (0.5, 0.0, 0.0), "lying": True})
    assert not round1._height_ok(0.013)


def test_v2_is_unchanged_by_the_v4_hooks():
    """v2 placements stand upright and v2 never lays a cube down (resting is v4 only)."""
    cfg = replace(CFG, resting=replace(CFG.resting, lying_prob=1.0))
    env = GraspV2Env(cfg.for_evaluation())
    for s in range(5):
        env.reset(seed=s)
        assert env.data.xmat[env.cube.body][8] > 0.999
