"""v4.0 T5: the door actors inside the line's tending cycle (evaluation/line_door.py)."""

import json
from pathlib import Path

import numpy as np
import pytest

from polypanda.evaluation.line_door import machine_from_episode, run_cycle
from polypanda.evaluation.tending import NOMINAL_SPOT
from polypanda.simulation.line import Line, LineConfig

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def episodes():
    return json.loads((ROOT / "eval_sets/door_dev_v1.json").read_text())["episodes"]


def test_a_door_episode_becomes_the_lines_m1_where_the_door_env_puts_it(episodes):
    ep = dict(episodes[0], offset=(0.01, 0.0), turn=5.0)
    spot = machine_from_episode(ep)
    assert spot.config.slide == ep["slide"] and spot.config.travel == ep["travel"]
    assert spot.config.lid_spring == ep["spring"]
    # +x in the station frame (away from the robot) is -y in the world at station angle -90
    np.testing.assert_allclose(spot.offset, (0.0, -0.01), atol=1e-12)
    line = Line.build_line(LineConfig(machines=(spot,)))
    nominal = Line.build_line(LineConfig(machines=(NOMINAL_SPOT,)))
    moved = line.machines["M1"].pocket_xyz() - nominal.machines["M1"].pocket_xyz()
    np.testing.assert_allclose(moved[:2], (0.0, -0.01), atol=1e-6)


@pytest.mark.parametrize("actor", ["expert", "feel_v2"])
@pytest.mark.parametrize("handover", ["carry", "door_start"])
def test_a_door_script_runs_the_whole_cycle_and_the_machine_starts(episodes, actor, handover):
    i = next(k for k, e in enumerate(episodes) if e["slide"] == "right" and e["spring"] == 0)
    row = run_cycle(actor, i, episodes[i], handover)
    assert row["success"], row
    assert row["open"]["success"] and row["close"]["success"] and row["seated"]
    assert row["handover"] == handover


def test_the_door_start_hand_over_lowers_the_hand_to_the_trained_start_height(episodes):
    from polypanda.control.scripted_tending import ScriptedTending
    from polypanda.evaluation.line_door import door_start_z, lower_to_door_start
    line = Line.build_line(LineConfig(machines=(machine_from_episode(episodes[0]),)))
    ctl = ScriptedTending(line)
    ctl.go_to("M1")
    high = line.data.site_xpos[ctl.arm.tcp_site][2]
    lower_to_door_start(ctl)
    tcp = line.data.site_xpos[ctl.arm.tcp_site]
    assert high > door_start_z() + 0.1                  # the carry posture is well above it
    assert abs(tcp[2] - door_start_z()) < 0.01
    np.testing.assert_allclose(tcp[:2], ctl.beliefs["M1"].origin[:2], atol=0.01)
