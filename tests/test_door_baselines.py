"""v4.0: the door skill's scripted baselines (control/door_baselines.py) and their evaluation
(scripts/evaluate_door.py)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from polypanda.control.door_baselines import MODES, BaselineDoor
from polypanda.control.scripted_door import run_expert
from polypanda.environments.door import DoorEnv


@pytest.fixture(scope="module")
def env():
    return DoorEnv()


def nominal_episode(env, command, slide="right", **changes):
    """The nominal machine at its nominal place, with a few things changed."""
    ep = env.draw_episode(np.random.default_rng(0), command)
    ep |= {"slide": slide, "travel": 0.12 if slide != "out" else 0.27, "friction": 0.4,
           "damping": 4.0, "spring": 0.0, "handle_offset": 0.0, "handle_shift": 0.0,
           "handle_height": 0.035, "latch_speed": 0.02, "offset": (0.0, 0.0), "turn": 0.0,
           "jitter": (0.0, 0.0, 0.0, 0.0), "handle_bias": (0.0, 0.0, 0.0),
           "bar_yaw_bias": 0.0}
    if command == "close":
        ep["lid"] = 0.115
    ep |= changes
    if ep["spring"] > 0:
        ep["damping"] = max(ep["damping"], ep["spring"] * ep["travel"] / 0.2)
        if command == "close":
            ep["lid"] = ep["travel"] - 0.002
    return ep


def run(env, mode, ep):
    script = BaselineDoor(env, mode)
    info = run_expert(env, options={"episode": ep}, expert=script)
    return info, script


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("command", ["open", "close"])
def test_every_baseline_works_the_nominal_machine(env, mode, command):
    info, _ = run(env, mode, nominal_episode(env, command))
    assert info["is_success"], info["failure"]


def test_the_camera_script_cannot_tell_a_left_slide_but_the_feel_script_tries_the_other_way(env):
    ep = nominal_episode(env, "open", "left")
    info, _ = run(env, "camera", ep)
    assert not info["is_success"] and info["failure"] == "not_opened"
    info, script = run(env, "feel", ep)
    assert info["is_success"], info["failure"]
    assert script.flipped


def test_the_feel_script_closes_a_wrongly_believed_lid_after_trying_the_other_way(env):
    info, script = run(env, "feel", nominal_episode(env, "close", "left"))
    assert info["is_success"], info["failure"]
    assert script.flipped


def test_the_feel_script_retries_a_latch_that_did_not_catch(env):
    ep = nominal_episode(env, "close", latch_speed=0.09)
    info, _ = run(env, "camera", ep)
    assert info["failure"] == "closed_not_latched"
    info, script = run(env, "feel", ep)
    assert info["is_success"], info["failure"]
    assert script.tries >= 1


def test_the_feel_script_seats_a_springy_lid_in_its_detent(env):
    ep = nominal_episode(env, "open", travel=0.135, spring=30.0)
    info, _ = run(env, "camera", ep)                  # opens to the believed 0.115: short of
    assert info["failure"] in ("not_held_open", "not_opened")  # the detent, so it creeps shut
    info, _ = run(env, "feel", ep)
    assert info["is_success"], info["failure"]
    assert env.machine.held_open


def test_the_nominal_script_misses_a_moved_machine_that_the_camera_script_finds(env):
    ep = nominal_episode(env, "open", offset=(0.0, -0.03))
    assert not run(env, "nominal", ep)[0]["is_success"]
    assert run(env, "camera", ep)[0]["is_success"]


def _evaluate_door():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_door.py"
    spec = importlib.util.spec_from_file_location("evaluate_door", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_evaluation_summarizes_per_cell():
    ev = _evaluate_door()
    rows = [{"command": "open", "slide": "left", "spring": False, "success": True,
             "failure": None, "steps": 100, "peak_force": 10.0, "slams": 0},
            {"command": "open", "slide": "left", "spring": False, "success": False,
             "failure": "not_opened", "steps": 300, "peak_force": 30.0, "slams": 1}]
    s = ev.summarize(rows)
    assert s["success"] == 1 and s["cells"]["open/left/plain"]["rate"] == 0.5
    assert s["steps_to_success_median"] == 100 and s["slam_episodes"] == 1
    lo, hi = ev.wilson(1, 2)
    assert 0.0 < lo < 0.5 < hi < 1.0
