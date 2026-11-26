"""v4.1 insertion round 2 (docs/v4_1_insertion_r1_rca.md): the env's round-2 options (start-height
curriculum, release interlock, compliant action layer, keypoint progress reward), the curriculum's
callback, the round's configs, and that the defaults are round 1's env exactly."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from polypanda.environments.insertion import InsertionConfig, InsertionEnv
from polypanda.training.callbacks import StartCurriculumCallback
from polypanda.training.config import TrainConfig

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "configs" / "insertion_v1.yaml"
V2 = ROOT / "configs" / "insertion_v2.yaml"
R1_STATS = {"peak_force", "peak_force_ctl", "peak_force_total", "collision_steps", "lead_clips",
            "turn_clips"}
# Round 1's env at d657eb7 (before round 2's code), configs/insertion_v1.yaml: seeded random
# actions drifting down (scratch golden_r1.py): steps, return, the sum of |obs|, the failure
GOLDEN = [(4242, 0.3, 60, 0.8701978835840971, 1103.019271850586, None),
          (4243, 0.6, 60, -16.741801185824492, 271.1213607788086, "damage"),
          (4244, 0.15, 80, -11.773493063923905, 1649.3820457458496, "damage")]
GOLDEN_STEPS = {4242: 60, 4243: 14, 4244: 69}


@pytest.fixture(scope="module")
def v1():
    return InsertionEnv(InsertionConfig.from_yaml(V1))


@pytest.fixture(scope="module")
def v2():
    cfg = InsertionConfig.from_yaml(V2)
    return InsertionEnv(replace(cfg, curriculum=replace(cfg.curriculum, enabled=False)))


def _episode(env, tester: str, **over) -> dict:
    ep = env.draw_episode(np.random.default_rng(5), tester)
    return ep | {"camera_xy": [0.0, 0.0], "camera_yaw": 0.0, "clearance": 0.0003,
                 "chamfer": 0.0007} | over


# --- round 1, exactly -----------------------------------------------------------------------------
def test_the_defaults_are_round_1s_env_exactly(v1):
    for k, (seed, std, steps, ret, obs_sum, failure) in enumerate(GOLDEN):
        rng = np.random.default_rng(100 + k)
        obs, info = v1.reset(seed=seed)
        assert "start_height" not in info
        total, s, n = 0.0, float(np.abs(obs).sum()), 0
        while n < steps:
            a = np.clip(rng.normal(0.0, std, 5), -1, 1)
            a[2] -= 0.4
            if k == 0:
                a[4] = -1.0
            obs, r, term, trunc, info = v1.step(a)
            total += r
            s += float(np.abs(obs).sum())
            n += 1
            if term or trunc:
                break
        assert n == GOLDEN_STEPS[seed] and info["failure"] == failure
        assert total == pytest.approx(ret, abs=1e-9) and s == pytest.approx(obs_sum, abs=1e-6)
        assert set(info["stats"]) == R1_STATS and "phi" not in v1._prev


def test_round_2_keeps_the_targets_task():
    cfg, ref = InsertionConfig.from_yaml(V2), InsertionConfig()
    for name in ("variation", "sensing", "success", "start", "horizon"):
        assert getattr(cfg, name) == getattr(ref, name)
    assert cfg.compliance.enabled and cfg.curriculum.enabled and cfg.action.release_window
    assert cfg.reward.progress == "keypoints_best" and cfg.reward.release_unseated == 0.0
    assert cfg.reward.collision_force
    ev = cfg.for_evaluation()
    assert not ev.curriculum.enabled and ev.compliance == cfg.compliance
    assert ev.action == cfg.action and ev.reward == cfg.reward
    assert InsertionConfig.from_yaml(V1) == replace(ref, pick_pool=ev.pick_pool)


def test_a_policy_is_evaluated_with_its_own_action_layer(tmp_path):
    from polypanda.evaluation.insertion import policy_env_config
    from polypanda.training.runner import plain
    cfg = InsertionConfig.from_yaml(V2)
    (tmp_path / "env_config.yaml").write_text(yaml.safe_dump(plain(cfg)))
    assert policy_env_config(tmp_path) == cfg.for_evaluation()
    bad = replace(cfg, success=replace(cfg.success, force_limit=80.0))
    (tmp_path / "env_config.yaml").write_text(yaml.safe_dump(plain(bad)))
    with pytest.raises(ValueError, match="success"):
        policy_env_config(tmp_path)


# --- the start-height curriculum ------------------------------------------------------------------
@pytest.mark.parametrize("tester", ["TA", "TB", "TC"])
def test_curriculum_starts_inside_the_nest_are_engaged_and_held(v2, tester):
    ep = _episode(v2, tester, camera_xy=[0.0018, -0.001], camera_yaw=np.radians(1.0))
    for h in (-0.008, -0.003, -0.0005):
        obs, info = v2.reset(seed=1, options={"episode": ep, "start_height": h})
        s = v2.seat()
        assert info["start_height"] == h and np.all(np.isfinite(obs))
        assert v2.line.attached(v2.part) and v2.ctl.ik_failures == 0
        assert abs(v2.part_height_over_rim() - h) < 0.0006
        assert v2.nest_force() < 1.0                            # resting, not jammed
        assert s["offset"] <= ep["clearance"] + 0.0002          # on the true nest's axis
        assert abs(s["key_error"]) < 0.5 and s["tilt"] < 1.5    # key in the slot, upright


def test_curriculum_starts_above_the_rim_keep_the_aim(v2):
    ep = _episode(v2, "TC", camera_xy=[0.002, 0.001])
    v2.reset(seed=1, options={"episode": ep})
    for _ in range(20):                         # the carry start, settled: where it aims
        v2.step([0, 0, 0, 0, -1.0])
    aimed = v2.data.xpos[v2.part.body][:2].copy()
    for h in (0.002, 0.03):
        v2.reset(seed=1, options={"episode": ep, "start_height": h})
        assert abs(v2.part_height_over_rim() - h) < 0.0006 and v2.nest_force() == 0.0
        # the module over the believed nest as the aim holds it (the camera's error kept)
        assert np.abs(v2.data.xpos[v2.part.body][:2] - aimed).max() < 0.0005
        assert v2.seat()["offset"] > 0.0005                         # not the true nest's axis


def test_the_curriculum_draws_between_its_low_bound_and_the_carry_start():
    env = InsertionEnv(InsertionConfig.from_yaml(V2))
    env.set_start_low(0.05)
    heights = []
    for seed in range(4):
        _, info = env.reset(seed=seed)
        heights.append(info["start_height"])
        assert 0.05 - 0.0006 <= env.part_height_over_rim()
    assert all(h is not None and 0.05 <= h for h in heights)
    env.set_start_low(0.5)                     # above the carry start: the carry start itself
    _, info = env.reset(seed=0)
    assert info["start_height"] is None
    ev = InsertionEnv(env.config.for_evaluation())
    _, info = ev.reset(seed=0)
    assert "start_height" not in info


class _Envs:
    def __init__(self):
        self.calls = []

    def env_method(self, name, *args):
        self.calls.append((name, *args))


def test_the_curriculum_callback_moves_the_low_bound_on_the_rolling_success(tmp_path):
    cur = replace(InsertionConfig.from_yaml(V2).curriculum, window=4)
    envs, logged = _Envs(), {}
    model = SimpleNamespace(get_env=lambda: envs, num_timesteps=0,
                            logger=SimpleNamespace(record=logged.__setitem__))
    cb = StartCurriculumCallback(cur, high=0.02, log_path=tmp_path / "c.jsonl")
    cb.init_callback(model)
    cb.on_training_start({}, {})
    assert envs.calls == [("set_start_low", -0.003)]

    def episodes(successes):
        for ok in successes:
            cb.locals = {"dones": [True, False], "infos": [{"is_success": ok}, {}]}
            cb.on_step()
    episodes([True] * 4)                              # 100 % > 80 %: up 5 mm
    assert cb.low == pytest.approx(0.002) and envs.calls[-1] == ("set_start_low", cb.low)
    assert logged["curriculum/low_mm"] == pytest.approx(2.0)
    episodes([True, True, True, False])               # 75 %: stays
    assert cb.low == pytest.approx(0.002)
    episodes([False] * 4)                             # 0 % < 10 %: down 3 mm
    assert cb.low == pytest.approx(-0.001)
    episodes([False] * 12)                            # never below low_min
    assert cb.low == pytest.approx(cur.low_min)
    episodes([True] * 40)                             # never above high
    assert cb.low == pytest.approx(0.02)
    assert len((tmp_path / "c.jsonl").read_text().splitlines()) == 16   # one per window


# --- the release interlock ------------------------------------------------------------------------
def test_a_release_takes_effect_only_near_the_floor(v2):
    ep = _episode(v2, "TA")
    v2.reset(seed=1, options={"episode": ep})
    for _ in range(3):
        _, r, term, _, info = v2.step([0, 0, 0, 0, 1.0])
        assert not info["released"] and info["reward_terms"]["release"] == 0.0 and not term
    assert v2.line.attached(v2.part) and info["stats"]["release_blocked"] == 3
    # the module's bottom 2 mm over the floor (the pocket is 15 mm deep): it lets go
    v2.reset(seed=1, options={"episode": ep, "start_height": -0.013})
    _, _, _, _, info = v2.step([0, 0, 0, 0, 1.0])
    assert not v2.line.attached(v2.part) and info["stats"]["release_blocked"] == 0


# --- the compliant action layer -------------------------------------------------------------------
@pytest.mark.parametrize("tester", ["TA", "TB"])
def test_pressing_down_on_the_rim_is_compliant(v1, v2, tester):
    """Aimed 2.5 mm off (on the rim, past the chamfer), straight down at full speed: round 1's
    stiff arm breaks the 50 N limit; round 2's layer holds it near its 10 N threshold."""
    ep = _episode(v1, tester, camera_xy=[0.0025, 0.0], chamfer=0.0005, clearance=0.00025)
    down = [0, 0, -1.0, 0, -1.0]
    v1.reset(seed=1, options={"episode": ep})
    for _ in range(80):
        _, _, term, _, info = v1.step(down)
        if term:
            break
    assert info["failure"] == "damage"
    v2.reset(seed=1, options={"episode": ep})
    forces = []
    for _ in range(80):
        _, _, term, _, info = v2.step(down)
        forces.append(info["nest_force"])
        assert not term
    assert max(forces) < 25.0 and max(forces[-20:]) < 15.0 and min(forces[-20:]) > 0.5
    assert info["stats"]["comply_steps"] > 20 and v2.line.attached(v2.part)


def test_the_rest_target_holds_the_hand_still(v2):
    v2.reset(seed=1, options={"episode": _episode(v2, "TB")})
    for _ in range(20):
        v2.step([0, 0, 0, 0, -1.0])
    # the target that holds the hand where it is, in free air, is the target itself
    assert np.abs(v2.rest_target() - v2.target).max() < 0.0003
    assert abs(v2._rest_yaw - v2.target_yaw) < 0.002                   # and its yaw


def test_twisting_against_the_tester_is_compliant(v2):
    """Started engaged (the key in the slot), the wrist turned hard both ways: the layer holds
    the twist to a few N m instead of the stiff wrist's tens."""
    v2.reset(seed=1, options={"episode": _episode(v2, "TA"), "start_height": -0.005})
    torques = []
    for k in range(60):
        v2.step([0, 0, 0, 1.0 if (k // 15) % 2 == 0 else -1.0, -1.0])
        torques.append(abs(v2.external_force()[1]))
    assert max(torques) < 5.0 and v2.nest_force() < 30.0 and v2.stats["peak_force"] < 50.0


# --- the keypoint progress reward -----------------------------------------------------------------
def test_the_keypoint_potential_is_gentle_far_and_steep_near(v2):
    phi = [v2.keypoint_potential(d) for d in (0.0, 0.001, 0.002, 0.015, 0.1, 0.25)]
    assert all(a > b for a, b in zip(phi, phi[1:], strict=False))
    assert phi[0] - phi[2] == pytest.approx(5.1, abs=0.3)     # the last 2 mm: ~+5
    assert phi[2] - phi[5] == pytest.approx(5.8, abs=0.3)     # the approach: ~+5
    v2.reset(seed=1, options={"episode": _episode(v2, "TC"), "start_height": -0.013})
    assert v2.keypoint_distance() == pytest.approx(0.002, abs=0.0006)   # 2 mm over the floor


def test_the_held_progress_term_pays_new_bests_only(v2):
    v2.reset(seed=1, options={"episode": _episode(v2, "TA"), "start_height": 0.03})
    start = v2._prev["phi"]
    _, _, _, _, info = v2.step([0, 0, -1.0, 0, -1.0])
    t = info["reward_terms"]
    assert t["insert"] == pytest.approx(v2._prev["phi"] - start) and t["insert"] > 0.0
    assert t["time"] == -0.005 and t["force"] == 0.0
    best, paid = v2._prev["phi"], t["insert"]
    for dz in [1.0] * 4 + [-1.0] * 10:          # away (unpaid, uncharged) and back down past it
        _, _, _, _, info = v2.step([0, 0, dz, 0, -1.0])
        assert info["reward_terms"]["insert"] >= 0.0
        paid += info["reward_terms"]["insert"]
        best = max(best, v2._prev["phi"])
    assert paid == pytest.approx(best - start)  # in all: the best potential reached


def test_the_arm_is_charged_for_pushing_on_the_tester_not_touching(v2, monkeypatch):
    v2.reset(seed=1, options={"episode": _episode(v2, "TB")})
    monkeypatch.setattr(v2, "arm_force", lambda: 18.0)
    assert v2._collision_term() == 0.0                       # under the soft limit, 20 N
    monkeypatch.setattr(v2, "arm_force", lambda: 30.0)
    assert v2._collision_term() == pytest.approx(-0.02 * 10.0)
    assert v2.stats["peak_arm_force"] == 30.0


# --- the round's PPO config -----------------------------------------------------------------------
def test_round_2_is_pure_ppo_with_a_kl_brake():
    from stable_baselines3.common.vec_env import DummyVecEnv

    from polypanda.training.runner import build_model
    cfg = TrainConfig.from_yaml(ROOT / "configs" / "insertion_ppo_r2.yaml")
    assert cfg.task == "insertion" and cfg.init_from is None
    assert cfg.env_config == "configs/insertion_v2.yaml" and cfg.eval_fixed_episodes
    p = cfg.ppo
    assert (p.learning_rate, p.n_epochs, p.target_kl, p.ent_coef, p.gamma) == (
        1e-4, 5, 0.02, 0.001, 0.995)
    env_cfg = replace(InsertionConfig.from_yaml(V1), horizon=2)
    venv = DummyVecEnv([lambda: InsertionEnv(env_cfg)])
    model = build_model(cfg, venv, None)
    assert model.target_kl == 0.02
    np.testing.assert_allclose(model.policy.log_std.detach().numpy(), [-1.6] * 4 + [-1.0],
                               rtol=1e-6)
    assert TrainConfig().ppo.target_kl is None
