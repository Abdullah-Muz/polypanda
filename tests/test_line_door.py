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
def test_a_door_script_runs_the_whole_cycle_and_the_machine_starts(episodes, actor):
    i = next(k for k, e in enumerate(episodes) if e["slide"] == "right" and e["spring"] == 0)
    row = run_cycle(actor, i, episodes[i])
    assert row["success"], row
    assert row["open"]["success"] and row["close"]["success"] and row["seated"]
