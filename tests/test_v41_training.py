"""v4.1 round 1's training plumbing: validation cells per task (training/fixed_eval.py), the
door's recording and a warm-started clone for any task (training/bc.py), and the round's configs
(configs/door_v3*.yaml, configs/insertion_bc*_r1.yaml)."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from polypanda.config import from_dict
from polypanda.environments.door import OBS_DIM_V2, DoorConfig, DoorEnv, DoorV2Env
from polypanda.environments.insertion import OBS_SLICES, InsertionConfig, InsertionEnv
from polypanda.evaluation.policies import load_run_policy
from polypanda.training.bc import (
    BCRunConfig,
    CloneConfig,
    RecordConfig,
    _record_chunk,
    clone,
    latch_labels,
    save_vecnormalize,
)
from polypanda.training.config import TrainConfig
from polypanda.training.fixed_eval import (
    DOOR,
    INSERTION,
    FixedEpisodes,
    cells_for,
    summarize_cells,
    summarize_insertion,
)

ROOT = Path(__file__).resolve().parents[1]
V40_ROW = ["index", "command", "slide", "spring", "success", "failure", "steps", "peak_force",
           "slams"]
V40_SUMMARY = ["n", "success", "cells", "worst_cell_rate", "steps_to_success_median",
               "peak_force_median", "slam_episodes"]


def play(env, k, action):
    for _ in range(k):
        env.reset()
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(action)
            done = term or trunc


# --- validation cells per task ------------------------------------------------------------------
def test_v4_0_door_rows_and_summary_keep_their_keys_in_order():
    env = FixedEpisodes(DoorV2Env(replace(DoorConfig(), horizon=2)), n=3, seed=5)
    assert env.task_cells is DOOR
    play(env, 3, np.zeros(5, dtype=np.float32))
    rows = env.get_rows()
    assert [list(r) for r in rows] == [V40_ROW] * 3
    s = env.summary()
    assert list(s) == V40_SUMMARY and s == summarize_cells(rows)
    assert env.summary(2) == summarize_cells(rows[:2])


def test_door_rows_with_kinds_add_the_targets_kind_x_command_cells():
    cfg = DoorConfig.from_yaml(ROOT / "configs" / "door_v3.yaml").for_evaluation()
    env = FixedEpisodes(DoorV2Env(replace(cfg, horizon=2)), n=6, seed=11)
    play(env, 6, np.zeros(5, dtype=np.float32))
    rows = env.get_rows()
    assert all(list(r) == V40_ROW[:4] + ["kind"] + V40_ROW[4:] for r in rows)
    s = env.summary()
    assert list(s)[:len(V40_SUMMARY)] == V40_SUMMARY
    kinds = {f"{e['kind']}/{e['command']}" for e in env.episodes}
    assert set(s["kind_command"]) == kinds
    assert sum(int(v.split("/")[1]) for v in s["kind_command"].values()) == 6
    assert s["worst_kind_command_rate"] == 0.0


def test_insertion_rows_are_per_tester_with_clearance_and_camera_bins():
    cfg = replace(InsertionConfig.from_yaml(ROOT / "configs" / "insertion_v1.yaml"), horizon=2)
    env = FixedEpisodes(InsertionEnv(cfg), n=3, seed=100000)
    assert env.task_cells is INSERTION and cells_for(env.unwrapped) is INSERTION
    assert all("grasp" in e for e in env.episodes)      # (grasps from the training pick pool)
    a = np.zeros(5, dtype=np.float32)
    a[4] = -1.0
    play(env, 3, a)
    rows = env.get_rows()
    assert [r["index"] for r in rows] == [0, 1, 2]
    assert [r["tester"] for r in rows] == [e["tester"] for e in env.episodes]
    assert all(not r["success"] and r["steps"] == 2 for r in rows)
    s = env.summary()
    assert s == summarize_insertion(rows) and s["n"] == 3 and s["success"] == 0
    assert sum(int(v.split("/")[1]) for v in s["cells"].values()) == 3
    assert sum(int(v.split("/")[1]) for v in s["by_clearance_mm"].values()) == 3
    assert sum(int(v.split("/")[1]) for v in s["by_camera_error_mm"].values()) == 3
    assert s["damage_episodes"] == 0 and s["worst_cell_rate"] == 0.0
    assert json.loads(json.dumps(s)) == s


def test_insertion_summary_counts_damage_and_bins():
    rows = [{"tester": "TA", "kind": "lid", "clearance_mm": 0.26, "camera_error_mm": 0.2,
             "success": True, "failure": None, "steps": 100, "peak_force": 12.0},
            {"tester": "TB", "kind": "drawer", "clearance_mm": 0.49, "camera_error_mm": 2.5,
             "success": False, "failure": "damage", "steps": 40, "peak_force": 61.0}]
    s = summarize_insertion(rows)
    assert s["cells"] == {"TA (lid)": "1/1", "TB (drawer)": "0/1"}
    assert s["damage_episodes"] == 1 and s["failures"] == {"damage": 1}
    assert s["by_clearance_mm"] == {"0.25-0.3125": "1/1", "0.4375-0.5001": "0/1"}
    assert s["by_camera_error_mm"] == {"0-0.5": "1/1", "2-3.01": "0/1"}
    assert s["steps_to_success_median"] == 100.0


def test_the_fixed_episode_callback_writes_insertion_cells(tmp_path):
    from polypanda.training.runner import train
    cfg = replace(TrainConfig.from_yaml(ROOT / "configs" / "insertion_ppo_r1.yaml"),
                  n_envs=1, vec_env="dummy", total_timesteps=8, eval_freq=4, eval_episodes=2,
                  checkpoint_freq=1000, init_from=None,
                  ppo=replace(TrainConfig().ppo, n_steps=8, batch_size=8, n_epochs=1))
    env_cfg = replace(InsertionConfig.from_yaml(ROOT / cfg.env_config), horizon=2)
    run = train(cfg, env_cfg, tmp_path / "run")
    lines = [json.loads(x) for x in (run / "eval" / "cells.jsonl").read_text().splitlines()]
    assert len(lines) == 2 and all(x["n"] == 2 and "by_clearance_mm" in x for x in lines)


# --- configs ------------------------------------------------------------------------------------
def test_optional_dataclass_fields_load_from_a_mapping():
    bc = from_dict(BCRunConfig, {"training_config": "a.yaml", "demos": "d.npz",
                                 "record": {"episodes": 7, "task": "door_v2"}})
    assert bc.record == RecordConfig(episodes=7, task="door_v2")
    assert from_dict(BCRunConfig, {"training_config": "a", "demos": "d"}).record is None


def test_door_v3_is_round_5s_env_with_drawers_and_wide_starts():
    r5 = DoorConfig.from_yaml(ROOT / "configs" / "door_v2_r5.yaml")
    v3 = DoorConfig.from_yaml(ROOT / "configs" / "door_v3.yaml")
    want = replace(r5, variation=replace(r5.variation, kinds=("lid", "drawer")),
                   start=replace(r5.start, wide=True))
    assert v3 == want
    dev = json.loads((ROOT / "eval_sets" / "door_dev_v2.json").read_text())
    assert json.loads(json.dumps(v3.start.__dict__)) == dev["start"]
    assert json.loads(json.dumps(v3.for_evaluation().variation.__dict__)) == dev["variation"]


def test_round_1_configs_chain_clone_to_ppo():
    door_bc = BCRunConfig.from_yaml(ROOT / "configs" / "door_v3_bc_r1_mix.yaml")
    door = TrainConfig.from_yaml(ROOT / door_bc.training_config)
    assert door.task == "door_v2" and door.env_config == "configs/door_v3.yaml"
    assert door.init_from == "runs/door_v3_bc_r1_mix_seed{seed}"
    assert door.init_checkpoint == "final"
    assert door_bc.record.teacher == "runs/door_v2_bc_ppo_r5_seed{seed}@final"
    first = BCRunConfig.from_yaml(ROOT / "configs" / "door_v3_bc_r1.yaml")
    assert first.record == replace(door_bc.record, teacher=None) and first.clone == door_bc.clone
    r5 = TrainConfig.from_yaml(ROOT / "configs" / "door_v2_bc_ppo_r5.yaml")
    assert door.ppo == r5.ppo and door.total_timesteps == r5.total_timesteps     # round 5's PPO
    assert door_bc.clone.init_from == "runs/door_v2_bc_ppo_r5_seed{seed}"
    assert door_bc.clone.init_checkpoint == "final" and door_bc.clone.log_std is None
    assert door_bc.record.task == "door_v2" and door_bc.record.script == "feel_v2"
    assert door_bc.record.env_config == door.env_config
    assert door.n_envs == 8 and door.eval_episodes == 200 and door.eval_freq == 100_000


def test_the_insertion_is_pure_ppo():
    """The user's decision of 2026-11-25: PPO from scratch, no demonstrations or clone."""
    ins = TrainConfig.from_yaml(ROOT / "configs" / "insertion_ppo_r1.yaml")
    assert ins.task == "insertion" and ins.init_from is None
    assert ins.env_config == "configs/insertion_v1.yaml" and ins.eval_fixed_episodes
    assert ins.n_envs == 8 and ins.eval_freq == 100_000 and ins.eval_seed == 100_000
    assert ins.ppo.log_std_init == (-2.3, -2.3, -2.3, -2.3, -1.2)
    assert not (ROOT / "configs" / "insertion_bc_r1.yaml").exists()


def test_round_1_demonstrations_draw_from_training_streams():
    """The door demonstrations' episode seeds (seed * 1000 + k) are no door set's seed, nor a
    training env's (config seed + rank) or validation's (eval_seed)."""
    set_seeds = {json.loads(p.read_text())["seed"]
                 for p in (ROOT / "eval_sets").glob("door_*.json")}
    assert {20261130, 20261201} <= set_seeds
    rec = BCRunConfig.from_yaml(ROOT / "configs" / "door_v3_bc_r1_mix.yaml").record
    seeds = set(range(rec.seed * 1000, rec.seed * 1000 + rec.episodes))
    assert not seeds & set_seeds
    ppo = yaml.safe_load((ROOT / "configs" / "door_v3_bc_ppo_r1.yaml").read_text())
    assert not seeds & set(range(ppo["eval_seed"], ppo["eval_seed"] + 1))
    assert not seeds & set(range(0, 64))
    assert rec.seed != RecordConfig.seed          # not v4.0's demonstrations


# --- recording and cloning ----------------------------------------------------------------------
def test_v4_0s_recording_defaults_are_unchanged():
    rec = RecordConfig()
    assert (rec.task, rec.env_config, rec.script, rec.seed, rec.noise, rec.noise_share) == (
        "door", None, "feel", 20261130, 0.2, 0.5)


def test_door_v3_recording_records_door_v2_observations_with_kinds():
    seeds = [20261133 * 1000 + k for k in range(2)]
    out = _record_chunk((seeds, 0.0, 0.0, "door_v2", "configs/door_v3.yaml", "feel_v2"))
    assert out["total"] == 2 and len(out["rows"]) == 2
    assert {r["kind"] for r in out["rows"]} <= {"lid", "drawer"}
    for o, a in zip(out["obs"], out["act"], strict=True):
        assert o.shape[1] == OBS_DIM_V2 and a.shape == (len(o), 5)
    assert out["kept"] == sum(r["success"] for r in out["rows"])


def test_a_teacher_is_tried_first_and_the_script_records_where_it_fails(tmp_path):
    """A teacher that never touches the door (a fresh clone of random data) fails every
    episode: the script's run is recorded instead, as without a teacher."""
    train_cfg = TrainConfig.from_yaml(ROOT / "configs" / "door_v3_bc_ppo_r1.yaml")
    env_cfg = DoorConfig.from_yaml(ROOT / train_cfg.env_config)
    model, venv, _ = clone(_demos("door_v2", seed=4), train_cfg, env_cfg,
                           CloneConfig(epochs=1, value_epochs=1, val_share=0.0))
    model.save(tmp_path / "final_model.zip")
    save_vecnormalize(venv, tmp_path / "final_vecnormalize.pkl")
    seeds = [20261133 * 1000 + 5]
    args = (seeds, 0.0, 0.0, "door_v2", "configs/door_v3.yaml", "feel_v2")
    plain = _record_chunk(args)
    taught = _record_chunk(args + (f"{tmp_path}@final",))
    assert taught["rows"][0]["source"] == "script" and plain["rows"][0]["source"] == "script"
    for k in ("obs", "act", "rew"):
        for a, b in zip(plain[k], taught[k], strict=True):
            np.testing.assert_array_equal(a, b)


def _demos(task, n_steps=64, seed=0):
    """Small synthetic demonstrations of a task's shapes (two episodes)."""
    rng = np.random.default_rng(seed)
    dim = {"door_v2": OBS_DIM_V2, "insertion": 42}[task]
    obs = rng.normal(size=(n_steps, dim)).astype(np.float32)
    act = np.clip(rng.normal(0.0, 0.3, size=(n_steps, 5)), -1, 1).astype(np.float32)
    done = np.zeros(n_steps, bool)
    done[n_steps // 2 - 1] = done[-1] = True
    episode = np.repeat([0, 1], n_steps // 2)
    return {"obs": obs, "act": act, "rew": rng.normal(size=n_steps), "done": done,
            "episode": episode, "kept": 2, "total": 2}


def test_latched_release_labels_change_only_the_released_steps():
    obs = np.zeros((5, 42), np.float32)
    obs[3:, OBS_SLICES["released"]] = 1.0
    act = -np.ones((5, 5), np.float32)
    act[2, 4] = 1.0                                  # the release step
    out = latch_labels(act, obs, 4, OBS_SLICES["released"].start)
    np.testing.assert_array_equal(out[:, 4], [-1, -1, 1, 1, 1])
    np.testing.assert_array_equal(out[:, :4], act[:, :4])
    assert act[3, 4] == -1.0                          # (a copy)


def test_an_insertion_clone_is_a_run_with_its_own_action_noise(tmp_path):
    """(Cloning works for any task; the v4.1 insertion itself is pure PPO.)"""
    train_cfg = TrainConfig.from_yaml(ROOT / "configs" / "insertion_ppo_r1.yaml")
    env_cfg = InsertionConfig.from_yaml(ROOT / train_cfg.env_config)
    cfg = CloneConfig(epochs=2, value_epochs=1, val_share=0.0,
                      log_std=(-3.0, -3.0, -3.0, -3.0, -1.0), latch_action=4,
                      latch_obs=OBS_SLICES["released"].start)
    model, venv, log = clone(_demos("insertion"), train_cfg, env_cfg, cfg)
    np.testing.assert_allclose(model.policy.log_std.detach().numpy(), [-3, -3, -3, -3, -1])
    assert "init" not in log
    assert model.observation_space.shape == (42,)
    model.save(tmp_path / "final_model.zip")
    save_vecnormalize(venv, tmp_path / "final_vecnormalize.pkl")
    a = load_run_policy(f"{tmp_path}@final")(np.zeros(42, np.float32))
    assert a.shape == (5,) and np.all(np.isfinite(a))


def test_a_warm_clone_starts_as_its_source(tmp_path):
    """With no training, the warm-started clone acts as the source run does (its inputs
    re-expressed in the demonstrations' normalization), keeps its noise and return scale."""
    train_cfg = TrainConfig.from_yaml(ROOT / "configs" / "door_v3_bc_ppo_r1.yaml")
    env_cfg = DoorConfig.from_yaml(ROOT / train_cfg.env_config)
    src = tmp_path / "src_seed3"
    src.mkdir()
    demos = _demos("door_v2", seed=1)
    model, venv, _ = clone(demos, train_cfg, env_cfg,
                           CloneConfig(epochs=3, value_epochs=1, val_share=0.0, log_std=-1.7,
                                       seed=3))
    model.save(src / "final_model.zip")
    save_vecnormalize(venv, src / "final_vecnormalize.pkl")
    (src / "metadata.yaml").write_text(yaml.safe_dump({"policy_id": "src_seed3"}))
    other = _demos("door_v2", seed=2)
    other["obs"] = other["obs"] * 2.0 + 0.5            # other statistics
    warm, wvenv, log = clone(other, train_cfg, env_cfg,
                             CloneConfig(epochs=0, value_epochs=0, val_share=0.5, log_std=None,
                                         seed=3, init_from=str(tmp_path / "src_seed{seed}")))
    assert log["init"]["policy_id"] == "src_seed3" and "start_val_mse" in log["init"]
    assert np.allclose(warm.policy.log_std.detach().numpy(), -1.7)
    assert wvenv.ret_rms.var == pytest.approx(venv.ret_rms.var)
    warm.save(tmp_path / "final_model.zip")
    save_vecnormalize(wvenv, tmp_path / "final_vecnormalize.pkl")
    a_src = load_run_policy(f"{src}@final")
    a_warm = load_run_policy(f"{tmp_path}@final")
    for o in demos["obs"][:8]:
        np.testing.assert_allclose(a_warm(o), a_src(o), atol=1e-4)


def test_ppo_takes_a_log_std_per_action():
    from polypanda.training.runner import build_model
    env_cfg = replace(InsertionConfig.from_yaml(ROOT / "configs" / "insertion_v1.yaml"), horizon=2)
    from stable_baselines3.common.vec_env import DummyVecEnv
    venv = DummyVecEnv([lambda: InsertionEnv(env_cfg)])
    cfg = TrainConfig(task="insertion", ppo=replace(TrainConfig().ppo,
                                                    log_std_init=(-2.3, -2.3, -2.3, -2.3, -1.2)))
    model = build_model(cfg, venv, None)
    np.testing.assert_allclose(model.policy.log_std.detach().numpy(), [-2.3] * 4 + [-1.2],
                               rtol=1e-6)
    with pytest.raises(ValueError):
        build_model(replace(cfg, ppo=replace(cfg.ppo, log_std_init=(-1.0, -1.0))), venv, None)
    assert build_model(TrainConfig(task="insertion"), venv, None).policy.log_std.mean() == 0.0


def test_door_env_classes_for_recording():
    from polypanda.training.bc import _record_env
    assert type(_record_env("door", None)) is DoorEnv
    env = _record_env("door_v2", "configs/door_v3.yaml")
    assert type(env) is DoorV2Env and env.config.focus.prob == 0.0
    assert env.config.variation.kinds == ("lid", "drawer") and env.config.start.wide
