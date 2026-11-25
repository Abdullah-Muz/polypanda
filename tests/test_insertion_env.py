"""v4.1: the learned insertion's env (environments/insertion.py), its runtime nests
(simulation/machine_variation.NestLayout) and the scripts and demonstrations on its episodes
(evaluation/insertion.py)."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from polypanda.control.scripted_cell import ScriptedCell
from polypanda.control.scripted_insertion import part_wrench
from polypanda.environments.insertion import (
    ACTION_SCHEMA,
    OBS_DIM,
    OBS_LAYOUT,
    OBS_SCHEMA,
    OBS_SLICES,
    STRETCH,
    TESTERS,
    InsertionConfig,
    InsertionController,
    InsertionEnv,
    build_scene,
)
from polypanda.environments.tasks import task_spec
from polypanda.evaluation.insertion import record_script, replay, run_script, summarize
from polypanda.simulation.assets import panda_spec
from polypanda.simulation.machine import MachineConfig, add_machine, tight_nest
from polypanda.simulation.machine_variation import NestLayout, nest_geoms, refresh_bvh


@pytest.fixture(scope="module")
def env():
    return InsertionEnv()


@pytest.fixture(scope="module")
def episodes(env):
    """One episode per tester with its simulated pick, the camera's error zeroed out."""
    rng = np.random.default_rng(7)
    out = {}
    for t in TESTERS:
        ep = env.draw_episode(rng, t)
        ep |= {"camera_xy": [0.0, 0.0], "camera_yaw": 0.0, "clearance": 0.0004,
               "chamfer": 0.001}
        out[t] = env.complete_episode(ep)
    return out


# --- spaces, schema, configs ------------------------------------------------------------------
def test_spaces_schema_and_registry(env, episodes):
    assert OBS_SCHEMA == f"insertion-v1-{OBS_DIM}d" and OBS_DIM == 42
    assert len({name for name, _ in OBS_LAYOUT}) == len(OBS_LAYOUT)
    assert env.observation_space.shape == (OBS_DIM,)
    assert env.action_space.shape == (5,) and env.ACTION_SCHEMA == ACTION_SCHEMA
    obs, info = env.reset(seed=0, options={"episode": episodes["TA"]})
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    spec = task_spec("insertion")
    assert spec.env_cls is InsertionEnv and spec.obs_schema == OBS_SCHEMA


def test_the_config_loads_from_yaml():
    cfg = InsertionConfig.from_yaml("configs/insertion_v1.yaml")
    assert cfg == replace(InsertionConfig(), pick_pool=cfg.pick_pool)
    assert cfg.pick_pool == "data/handoff/insertion_picks_train_v1.json"


# --- the runtime nest -------------------------------------------------------------------------
def _compiled(cfg: MachineConfig) -> mujoco.MjModel:
    spec = panda_spec()
    add_machine(spec, "M", (0.5, 0.0), 0.0, cfg)
    return spec.compile()


@pytest.mark.parametrize("kind, key_wall", [("lid", 0), ("drawer", 1)])
def test_a_runtime_nest_matches_the_compiled_one(kind, key_wall):
    base = tight_nest(MachineConfig(kind=kind, key_wall=key_wall))
    cfg = tight_nest(MachineConfig(kind=kind, key_wall=key_wall), clearance=0.00047,
                     chamfer=0.0006)
    ref, model = _compiled(cfg), _compiled(base)
    nest = NestLayout(model, "M", base)
    nest.apply(model, cfg)
    geoms, _ = nest_geoms("M", cfg)
    assert len(geoms) >= 12
    for name in geoms:
        g = ref.geom(name).id
        for f in ("geom_pos", "geom_size", "geom_rbound", "geom_aabb"):
            np.testing.assert_allclose(getattr(model, f)[g], getattr(ref, f)[g], atol=1e-12,
                                       err_msg=f"{name} {f}")
        q, q_ref = model.geom_quat[g], ref.geom_quat[g]
        assert min(np.abs(q - q_ref).max(), np.abs(q + q_ref).max()) < 1e-12, name
    with pytest.raises(ValueError):
        nest.apply(model, replace(cfg, pocket_depth=0.02))


def test_the_nest_bvh_is_rebuilt_as_the_compiler_builds_it():
    cfg = tight_nest()
    model = _compiled(cfg)
    nest = NestLayout(model, "M", cfg)
    before = model.bvh_aabb.copy()
    for b in nest.bodies:
        refresh_bvh(model, b)
    np.testing.assert_allclose(model.bvh_aabb, before, atol=1e-12)


# --- the controller and the measurements ----------------------------------------------------
def test_the_controller_steps_the_physics_as_the_cell_does():
    """InsertionController.step (one physics step at a time, with hooks) gives bit-identical
    states to ScriptedCell.step (mj_step with nstep=25)."""
    a, b = build_scene(), build_scene()
    ca, cb = InsertionController(a), ScriptedCell(b)
    calls = []
    ca.physics_hook = lambda: calls.append(1)
    for c in (ca, cb):
        c.q_cmd = c.carry.copy()
        c.q_cmd[0] -= 0.4
    ca.step(15)
    cb.step(15)
    assert len(calls) == 15 * 25
    np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
    np.testing.assert_array_equal(a.data.qvel, b.data.qvel)


def test_the_nest_force_and_wrench_agree_with_mujocos_contact_forces(env, episodes):
    env.reset(seed=1, options={"episode": episodes["TC"]})
    run_script(env, "straight_down")                 # ends with the module seated and loose
    for _ in range(3):
        env.ctl.step()
        f6, peak = np.zeros(6), 0.0
        for i in env._nest_contacts():
            mujoco.mj_contactForce(env.model, env.data, int(i), f6)
            peak = max(peak, abs(f6[0]))
        assert peak > 0.1                    # its weight on the nest's floor (on a few contacts)
        assert env.nest_force() == pytest.approx(peak, rel=1e-9, abs=1e-12)
        np.testing.assert_allclose(env.wrench(), part_wrench(env.model, env.data, env.part),
                                   rtol=1e-9, atol=1e-12)


# --- episodes ---------------------------------------------------------------------------------
def test_episodes_are_drawn_from_the_targets_distribution(env):
    rng = np.random.default_rng(3)
    eps = [env.draw_episode(rng) for _ in range(400)]
    assert {e["tester"] for e in eps} == set(TESTERS)
    c = np.array([e["clearance"] for e in eps])
    ch = np.array([e["chamfer"] for e in eps])
    assert 0.00025 <= c.min() and c.max() <= 0.0005
    assert 0.0005 <= ch.min() and ch.max() <= 0.001
    r = np.array([np.hypot(*e["camera_xy"]) for e in eps])
    assert r.max() <= 0.003 + 1e-12 and r.mean() == pytest.approx(0.001, abs=0.00015)
    yaw = np.degrees([e["camera_yaw"] for e in eps])
    assert np.abs(yaw).max() <= 2.0 and np.std(yaw) == pytest.approx(0.5, abs=0.07)
    assert all("grasp" not in e for e in eps)        # (simulated at reset, or by the set maker)
    stretch = InsertionEnv.__new__(InsertionEnv)
    stretch.config = replace(InsertionConfig(), variation=replace(InsertionConfig().variation,
                                                                   **STRETCH))
    stretch.pool = None
    rs = [stretch.draw_episode(rng) for _ in range(400)]
    r = np.array([np.hypot(*e["camera_xy"]) for e in rs])
    assert r.max() <= 0.004 + 1e-12 and r.mean() == pytest.approx(0.0019, abs=0.0003)


def test_reset_replays_an_episode_exactly(env, episodes):
    ep = episodes["TB"]
    obs1, _ = env.reset(seed=5, options={"episode": ep})
    q1 = env.data.qpos.copy()
    obs2, _ = env.reset(seed=5, options={"episode": ep})
    np.testing.assert_array_equal(env.data.qpos, q1)
    np.testing.assert_array_equal(obs1, obs2)


@pytest.mark.parametrize("tester", TESTERS)
def test_the_start_is_the_cells_aimed_insertion(env, episodes, tester):
    """The module held, the tester open (a drawer at its stop, pinned), the hand at the carry
    height over the believed nest, the module's key turned to the believed slot."""
    env.reset(seed=0, options={"episode": episodes[tester]})
    d, mach = env.data, env.line.machines[tester]
    assert env.line.attached(env.part) and not env.released
    assert mach.lid >= mach.cfg.travel - 0.002
    assert mach.indexed == (tester == "TB")
    tcp = d.site_xpos[env.arm.tcp_site]
    # (the aim's turn keeps the measured height, which the held module's weight sags a little)
    assert tcp[2] == pytest.approx(0.32, abs=0.02)
    # (the insertion starts as the turn's last step ends, as in the cell: the arm still settling)
    np.testing.assert_allclose(d.xpos[env.part.body][:2], env.belief.origin[:2], atol=0.005)
    assert abs(mach.key_error(env.part)) < 5.0
    assert env.steps == 0 and env.stats["peak_force"] == 0.0
    assert env.ctl.regrasps == 0
    obs = env.observation()
    np.testing.assert_allclose(obs[OBS_SLICES["tcp"]][:2], 0.0, atol=0.005)
    np.testing.assert_allclose(obs[OBS_SLICES["target"]][:2], 0.0, atol=1e-6)
    assert obs[OBS_SLICES["released"]][0] == 0.0 and obs[OBS_SLICES["time"]][0] == 0.0


# --- the scripts, success and the demonstrations ---------------------------------------------
@pytest.mark.parametrize("tester", TESTERS)
def test_the_search_seats_and_its_recorded_actions_replay(env, episodes, tester):
    ep = episodes[tester]
    env.reset(seed=2, options={"episode": ep})
    rec = record_script(env, "search")
    out = rec["outcome"]
    assert out["success"], out
    assert out["seated"] and out["released"] and out["hand_up"]
    assert out["peak_force"] < 50.0 and rec["exceeded"] == 0
    assert len(rec["obs"]) == len(rec["actions"]) + 1 == out["steps"] + 1
    assert (rec["actions"][:, 4] > 0).sum() == 1     # one release
    rp = replay(env, ep, rec["actions"], seed=2)
    assert rp["outcome"]["success"]
    assert rp["outcome"]["peak_force"] == pytest.approx(out["peak_force"], abs=1e-6)
    assert len(rp["actions"]) <= len(rec["actions"])


def test_straight_down_seats_with_an_exact_camera(env, episodes):
    env.reset(seed=3, options={"episode": episodes["TA"]})
    out = run_script(env, "straight_down")
    assert out["seated"] and out["released"] and out["hand_up"]
    assert out["time_s"] < 10.0


def test_success_needs_the_release_and_the_hand_back_up(env, episodes):
    """Replaying the search's actions without its release never succeeds; with it, success comes
    only once the hand is back at the carry height."""
    ep = episodes["TA"]
    env.reset(seed=4, options={"episode": ep})
    actions = record_script(env, "search")["actions"]
    held = actions.copy()
    held[:, 4] = -1.0
    out = replay(env, ep, held, seed=4)["outcome"]
    assert not out["success"] and out["failure"] == "not_released"
    env.reset(seed=4, options={"episode": ep})
    k = int(np.flatnonzero(actions[:, 4] > 0)[0])
    for a in actions[: k + 1]:
        obs, r, term, trunc, info = env.step(a)
        assert not term
    assert info["seated"] and info["released"] and info["reward_terms"]["release"] > 0
    assert obs[OBS_SLICES["released"]][0] == 1.0


def test_pressing_a_jammed_module_is_damage(env, episodes):
    """Driving the module straight down onto the nest's rim, 3 mm off: the nest force passes the
    50 N damage limit and the episode ends as a failure."""
    ep = episodes["TC"] | {"camera_xy": [0.003, 0.0]}
    env.reset(seed=5, options={"episode": ep})
    for _ in range(200):
        obs, r, term, trunc, info = env.step(np.array([0.0, 0.0, -0.5, 0.0, -1.0]))
        if term or trunc:
            break
    assert term and info["failure"] == "damage" and not info["is_success"]
    assert info["stats"]["peak_force"] > 50.0 and info["reward_terms"]["damage"] < 0


def test_the_time_limit_truncates(env, episodes):
    env.reset(seed=6, options={"episode": episodes["TA"]})
    zero = np.array([0.0, 0.0, 0.0, 0.0, -1.0])
    n = 0
    while True:
        obs, r, term, trunc, info = env.step(zero)
        n += 1
        if term or trunc:
            break
    assert trunc and not term and n == env.config.horizon
    assert info["failure"] == "not_released"
    assert obs[OBS_SLICES["time"]][0] == pytest.approx(1.0)


def test_summaries_count_successes_and_budget():
    rows = [{"success": True, "failure": None, "time_s": 12.0, "peak_force": 8.0,
             "peak_force_ctl": 6.0, "tester": "TA", "kind": "lid", "camera_error_mm": 0.7,
             "camera_yaw_deg": 0.2, "clearance_mm": 0.3, "chamfer_mm": 0.6},
            {"success": True, "failure": None, "time_s": 31.0, "peak_force": 9.0,
             "peak_force_ctl": 6.0, "tester": "TB", "kind": "drawer", "camera_error_mm": 1.7,
             "camera_yaw_deg": -0.6, "clearance_mm": 0.45, "chamfer_mm": 0.9},
            {"success": False, "failure": "damage", "time_s": 5.0, "peak_force": 80.0,
             "peak_force_ctl": 40.0, "tester": "TC", "kind": "pneumatic",
             "camera_error_mm": 2.5, "camera_yaw_deg": 1.5, "clearance_mm": 0.25,
             "chamfer_mm": 0.5}]
    s = summarize(rows)
    assert s["success"] == 2 and s["success_within_budget"] == 1
    assert s["over_damage_limit"] == 1 and s["over_damage_limit_ctl"] == 0
    assert s["failures"] == {"damage": 1}
    assert s["time_to_success_s"]["median"] == pytest.approx(21.5)


# --- the training pick pool ------------------------------------------------------------------
def test_training_draws_grasps_from_the_pick_pool():
    cfg = InsertionConfig.from_yaml("configs/insertion_v1.yaml")
    env = InsertionEnv.__new__(InsertionEnv)
    env.config = cfg
    from polypanda.environments.insertion import load_pick_pool
    env.pool = load_pick_pool(cfg.pick_pool)
    assert set(env.pool) == set(TESTERS) and sum(map(len, env.pool.values())) == 1000
    ep = env.draw_episode(np.random.default_rng(0))
    assert ep["grasp"] in [e["grasp"] for e in env.pool[ep["tester"]]]
