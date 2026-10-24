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
    assert obs.shape == (OBS_DIM,) == (45,) and env.observation_space.contains(obs)
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
