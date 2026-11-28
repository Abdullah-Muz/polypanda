"""v4.1: the door's hand-over starts drawn wide (DoorStartConfig.wide) and the v2 door sets
(docs/v4_1_targets.md, Part II; docs/door_env.md). Off by default: tests/test_lid_identity.py
holds v4.0's start, draw for draw."""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from polypanda.environments.door import DoorConfig, DoorEnv, DoorStartConfig
from polypanda.simulation.panda import tcp_position

ROOT = Path(__file__).resolve().parents[1]
DEV_V2 = ROOT / "eval_sets" / "door_dev_v2.json"


def load_sets():
    spec = importlib.util.spec_from_file_location("make_door_sets",
                                                  ROOT / "scripts" / "make_door_sets.py")
    sets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sets)
    return sets


def wide_env() -> DoorEnv:
    return DoorEnv(load_sets().set_config(kinds=("lid", "drawer"), wide=True))


def test_the_start_is_v4_0s_unless_asked():
    assert DoorStartConfig().wide is False
    assert DoorConfig().start == DoorStartConfig()


def test_a_given_kind_does_not_change_the_stream():
    env = DoorEnv(replace(DoorConfig(), variation=replace(DoorConfig().variation,
                                                           kinds=("lid", "drawer"))))
    drawn = env.draw_episode(np.random.default_rng(4))
    same = env.draw_episode(np.random.default_rng(4), kind=drawn["kind"])
    assert same == drawn
    other = env.draw_episode(np.random.default_rng(4),
                             kind="lid" if drawn["kind"] == "drawer" else "drawer")
    assert other["kind"] != drawn["kind"] and other["command"] == drawn["command"]
    with pytest.raises(ValueError):
        DoorEnv().draw_episode(np.random.default_rng(0), kind="drawer")


def test_wide_starts_are_reachable_clear_and_span_the_heights():
    env = wide_env()
    s = env.config.start
    rng = np.random.default_rng(11)
    heights, radial = [], []
    for k in range(24):
        ep = env.draw_episode(rng, ("open", "close")[k % 2], ("lid", "drawer")[(k // 2) % 2])
        env.reset(seed=k, options={"episode": ep})
        tcp = tcp_position(env.data, env.arm)
        jx, jy, jz, _ = ep["jitter"]
        want = np.array([env.config.scene.radius + jx, jy, s.carry_height + jz])
        np.testing.assert_allclose(tcp, want, atol=2e-4)       # the IK reached it
        h = tcp[2] - env.handle_xyz()[2]
        assert s.wide_height[0] - 2e-4 <= h <= s.wide_height[1] + 2e-4
        assert env.arm_clearance() >= s.wide_clearance - 1e-6
        assert s.wide_radial[0] <= jx <= s.wide_radial[1] and abs(jy) <= s.wide_side
        heights.append(h)
        radial.append(jx)
    assert min(heights) < 0.12 and max(heights) > 0.25     # wide, not v4.0's 8 cm
    assert min(radial) < 0.0 < max(radial)


def test_a_wide_episode_replays_exactly():
    env = wide_env()
    ep = env.draw_episode(np.random.default_rng(2), "close", "drawer")
    a, _ = env.reset(seed=1, options={"episode": ep})
    env.step(np.zeros(5, dtype=np.float32))
    b, _ = env.reset(seed=1, options={"episode": json.loads(json.dumps(ep))})
    np.testing.assert_array_equal(a, b)


def test_a_set_with_both_kinds_takes_them_in_turn():
    eps, variation = load_sets().make_set(8, 3, False, ("lid", "drawer"), wide=True)
    assert [(e["kind"], e["command"]) for e in eps[:4]] == [
        ("lid", "open"), ("lid", "close"), ("drawer", "open"), ("drawer", "close")]
    assert variation["kinds"] == ["lid", "drawer"] or variation["kinds"] == ("lid", "drawer")


@pytest.mark.skipif(not DEV_V2.exists(), reason="door_dev_v2 not generated yet")
def test_door_dev_v2_is_as_targeted():
    """docs/v4_1_targets.md, Part II: 400 episodes, 100 per kind x command, wide starts,
    seed 20261130, and the stored digest."""
    import hashlib
    stored = json.loads(DEV_V2.read_text())
    assert stored["seed"] == 20261130 and stored["n"] == 400 and stored["start"]["wide"]
    cells = {}
    for e in stored["episodes"]:
        key = (e["kind"], e["command"])
        cells[key] = cells.get(key, 0) + 1
    assert cells == {(k, c): 100 for k in ("lid", "drawer") for c in ("open", "close")}
    body = json.dumps(stored["episodes"], sort_keys=True)
    assert hashlib.sha256(body.encode()).hexdigest()[:12] == stored["digest"]


@pytest.mark.skipif(not DEV_V2.exists(), reason="door_dev_v2 not generated yet")
@pytest.mark.skipif(os.environ.get("CI") == "true",
                    reason="exact floating point of the project's WSL machine")
def test_door_dev_v2_is_drawn_again_exactly():
    """make_door_sets.py draws door_dev_v2 again to the byte. The draw runs the door env, so the
    bytes are this machine's floating point (as tests/test_lid_identity.py's redraw of
    door_dev_v1, which differs on some of GitHub's runners): checked on the project's WSL only."""
    stored = json.loads(DEV_V2.read_text())
    episodes, variation = load_sets().make_set(400, 20261130, False, ("lid", "drawer"),
                                               wide=True)
    assert json.dumps(episodes, sort_keys=True) == json.dumps(stored["episodes"], sort_keys=True)
    assert json.loads(json.dumps(variation)) == stored["variation"]


def test_training_draws_from_other_streams_than_the_door_sets():
    """Training envs are seeded with their config's seed (+ the env's rank) and validate on
    eval_seed; the sets are drawn from np.random.default_rng(<the set's seed>). No door config
    may use a set's seed."""
    set_seeds = {json.loads(p.read_text())["seed"]
                 for p in (ROOT / "eval_sets").glob("door_*.json")}
    assert {20261124, 20261125, 20261126, 4101} <= set_seeds
    for path in (ROOT / "configs").glob("door_*.yaml"):
        cfg = yaml.safe_load(path.read_text()) or {}
        for key in ("seed", "eval_seed"):
            if key in cfg:
                assert not set_seeds & set(range(cfg[key], cfg[key] + 64)), (path, key)
