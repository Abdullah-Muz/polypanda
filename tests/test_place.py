import warnings
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from polypanda.control.scripted_place import ScriptedPlace, upright_quat
from polypanda.environments.handoff import HandoffPool
from polypanda.environments.place import OBS_DIM, PlaceEnv
from polypanda.evaluation.place import evaluate_place

ROOT = Path(__file__).parents[1]
TRAIN = ROOT / "data" / "handoff" / "place_train_v1.npz"
EVAL = ROOT / "eval_sets" / "place_nominal_v1.npz"
ZERO = np.zeros(7, dtype=np.float32)


@pytest.fixture(scope="module")
def env():
    return PlaceEnv()


def _cases(idx):
    pool = HandoffPool.load(EVAL)
    return HandoffPool(pool.states[idx], tuple(pool.sources[i] for i in idx),
                       pool.placements[idx], pool.goals[idx], pool.meta)


def test_pools_cover_all_grasp_sources():
    train, ev = HandoffPool.load(TRAIN), HandoffPool.load(EVAL)
    assert len(train) == 1190 and len(ev) == 198
    assert len(Counter(train.sources)) == len(Counter(ev.sources)) == 4
    np.testing.assert_allclose(ev.goals[:, 2], 0.12)  # pre-place height


def test_spaces_and_check_env(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,) == (48,) and env.observation_space.contains(obs)
    assert info["target"][2] == pytest.approx(env.cube.half_size[2])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(PlaceEnv(), warn=True)


def test_gate_stays_closed_while_high(env):
    env.reset(options={"state_index": 0})
    for _ in range(10):
        _, _, _, _, info = env.step(ZERO)
        assert not info["released"] and info["lift"] > 0.05


def test_scripted_place_succeeds_from_every_grasp_source():
    summary, eps = evaluate_place(ScriptedPlace(), _cases([0, 50, 100, 150]))
    assert summary["success_rate"] == 1.0
    for e in eps:
        assert e.released and e.final_xy_error_mm < 20 and e.final_tilt_deg < 10


def test_placed_check_rejects_a_cube_still_in_the_fingers(env):
    env.reset(options={"state_index": 3})
    for _ in range(40):  # hold still: the cube stays pinched in the air
        _, _, _, _, info = env.step(ZERO)
    assert not info["is_success"] and not info["released"]


def test_placed_check_rejects_a_cube_on_its_side():
    env = PlaceEnv()
    env.reset(options={"state_index": 0})
    tx, ty = env.target[:2]
    quat = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]  # 90 deg about x
    env.data.qpos[env.cube.qpos:env.cube.qpos + 7] = [tx, ty, 0.02, *quat]
    env.data.qvel[:] = 0.0
    env.data.ctrl[env.arm.gripper_actuator] = 255.0
    env.released = True
    mujoco.mj_forward(env.model, env.data)
    for _ in range(15):
        _, _, terminated, _, info = env.step(ZERO)
        if terminated:
            break
    assert not info["is_success"] and info["failure"] == "toppled"


def test_upright_quat_is_identity_correction_for_an_upright_cube(env):
    env.reset(options={"state_index": 0})
    R_tcp = env.data.site_xmat[env.arm.tcp_site].reshape(3, 3)
    R_cube = env.data.xmat[env.cube.body].reshape(3, 3)
    q = upright_quat(env.data, env.arm, env.cube)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    R_cube_new = R.reshape(3, 3) @ (R_tcp.T @ R_cube)  # cube orientation under the new grip
    np.testing.assert_allclose(R_cube_new[:, 2], [0, 0, 1], atol=1e-9)  # stands upright


def test_observation_includes_the_cube_up_axis(env):
    from polypanda.environments.place import OBS_LAYOUT
    obs, _ = env.reset(options={"state_index": 0})
    start = sum(n for name, n in OBS_LAYOUT[:[k for k, _ in OBS_LAYOUT].index("cube_up")])
    up = obs[start:start + 3]
    np.testing.assert_allclose(up, env.data.xmat[env.cube.body].reshape(3, 3)[:, 2], atol=1e-6)
    tilt = obs[start - 1]
    assert np.degrees(np.arccos(np.clip(up[2], -1, 1))) == pytest.approx(np.degrees(tilt), abs=0.1)


def test_learned_place_task_and_config():
    from dataclasses import replace

    from polypanda.environments.place import PlaceConfig
    from polypanda.environments.tasks import task_spec
    spec = task_spec("place")
    assert spec.env_cls is PlaceEnv and spec.obs_schema == "place-v2-48d"
    cfg = PlaceConfig.from_yaml(ROOT / "configs" / "place_cartesian.yaml")
    assert cfg.action.space == "cartesian" and cfg.action.cartesian.dof == 6
    assert cfg.reward.cube_tilt == 0.3 and cfg.action.bound_penalty == 0.1
    # everything else is the scripted place's task
    default = PlaceConfig()
    assert replace(cfg, action=default.action,
                   reward=replace(cfg.reward, cube_tilt=0.0)) == default
    env = PlaceEnv(cfg)
    assert env.action_space.shape == (6,)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(env, warn=False)
    env.reset(options={"state_index": 0})
    _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
    assert info["reward_terms"]["cube_tilt"] == pytest.approx(-0.3 * info["cube_tilt"])


@pytest.fixture(scope="module")
def stack_env():
    from dataclasses import replace

    from polypanda.environments.place import PlaceConfig
    cfg = PlaceConfig.from_yaml(ROOT / "configs" / "place_stack_cartesian.yaml")
    base = PlaceConfig.from_yaml(ROOT / "configs" / "place_cartesian.yaml")
    assert cfg.stack.max_level == 5 and replace(cfg, stack=base.stack) == base  # only towers added
    return PlaceEnv(cfg)


@pytest.mark.parametrize("level", [0, 1, 3, 5])
def test_stacking_starts_over_a_tower(stack_env, level):
    env, h = stack_env, 0.06
    _, info = env.reset(seed=level, options={"state_index": 5, "level": level})
    assert info["level"] == level
    assert env.target[2] == pytest.approx(h * (level + 0.5))
    assert env.data.xpos[env.cube.body][2] - env.target[2] == pytest.approx(0.09, abs=0.003)
    for k, t in enumerate(env.tower[:level]):  # a straight-ish tower at the target
        p = env.data.xpos[t.body]
        assert np.abs(p[:2] - env.target[:2]).max() <= 0.004 + 1e-9
        assert p[2] == pytest.approx(h * (k + 0.5), abs=1e-3)
    for _ in range(10):  # holding still: nothing falls, the tower stands
        _, _, terminated, _, info = env.step(np.zeros(6, dtype=np.float32))
        assert not terminated and not info["released"]
    assert env._tower_moved() < 0.002


def test_knocking_the_tower_fails(stack_env):
    env = stack_env
    env.reset(seed=0, options={"state_index": 5, "level": 2})
    top = env.tower[1]
    env.data.qpos[top.qpos] += 0.03  # the top tower cube pushed off
    mujoco.mj_forward(env.model, env.data)
    _, _, terminated, _, info = env.step(np.zeros(6, dtype=np.float32))
    assert terminated and info["failure"] == "tower_knocked" and not info["is_success"]
    assert info["reward_terms"]["tower"] == -5.0
    assert info["reward_terms"]["tower_disturb"] == 0.0  # v1: no charge below the limit


def test_v3_aims_at_the_top_cube_and_rewards_centring():
    from polypanda.environments.place import PlaceConfig
    cfg = PlaceConfig.from_yaml(ROOT / "configs" / "place_stack_v3_cartesian.yaml")
    assert cfg.stack.aim_at_top and cfg.check.tolerance == 0.02 and cfg.gate.xy_tol == 0.02
    assert cfg.reward.centring == 0.0 and cfg.stack.disturb_penalty == 300.0
    assert cfg.reward.precise == 0.5 and cfg.reward.precise_tol == 0.01
    from dataclasses import replace
    cfg = replace(cfg, reward=replace(cfg.reward, centring=20.0))  # (the term itself)
    env = PlaceEnv(cfg)
    env.reset(seed=3, options={"state_index": 5, "level": 2})
    top = env.data.xpos[env.tower[1].body]
    np.testing.assert_allclose(env.target[:2], top[:2])          # the cube below, not the pool goal
    assert env.target[2] == pytest.approx(0.06 * 2.5)
    assert np.linalg.norm(env.target[:2] - env.pool.goals[5][:2]) > 1e-4  # (the tower is jittered)
    _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
    assert info["reward_terms"]["centring"] == 0.0                 # charged only after release
    env.released = True
    _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
    assert info["reward_terms"]["centring"] == pytest.approx(-20.0 * info["xy_error"])
    assert info["reward_terms"]["precise"] == 0.0  # not placed (still held)
    env.reset(seed=3, options={"state_index": 5, "level": 0})   # on the table: the pool goal
    np.testing.assert_allclose(env.target[:2], env.pool.goals[5][:2])


def test_v2_charges_every_millimetre_a_tower_cube_moves():
    from dataclasses import replace

    from polypanda.environments.place import PlaceConfig
    cfg = PlaceConfig.from_yaml(ROOT / "configs" / "place_stack_v2_cartesian.yaml")
    v1 = PlaceConfig.from_yaml(ROOT / "configs" / "place_stack_cartesian.yaml")
    assert cfg.stack.disturb_penalty == 1000.0 and cfg.stack.disturb_tol == 0.005
    assert replace(cfg, stack=replace(cfg.stack, disturb_penalty=0.0, disturb_tol=0.01)) == v1
    env = PlaceEnv(cfg)
    env.reset(seed=0, options={"state_index": 5, "level": 2})
    _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
    before = info["tower_moved"]
    env.data.qpos[env.tower[1].qpos] += 0.003  # a 3 mm nudge: within the limit, but charged
    mujoco.mj_forward(env.model, env.data)
    _, _, terminated, _, info = env.step(np.zeros(6, dtype=np.float32))
    assert not terminated and info["tower_moved"] == pytest.approx(0.003, abs=5e-4)
    assert info["reward_terms"]["tower_disturb"] == pytest.approx(
        -1000.0 * (info["tower_moved"] - before))
