"""Work-order evaluation in the Milestone 9 cell (scripted skills; trained runs are not in the
repository)."""

import numpy as np
import pytest

from polypanda.evaluation.cell import CellEvalConfig, evaluate_cell, skill_from_spec
from polypanda.simulation.workcell import WorkOrder


def test_scripted_cell_orders_succeed_and_are_reproducible():
    config = CellEvalConfig(n=2, heights=(1, 1))
    summary, episodes = evaluate_cell(None, None, config)
    assert [e.success for e in episodes] == [True, True] and summary["success_rate"] == 1.0
    assert summary["towers"] == 6 and summary["cubes"] == 6
    assert summary["learned_pick"] is None and summary["learned_place"] is None
    assert all(e.failure is None and e.towers_ok == "111" and e.sim_s > 0 for e in episodes)
    # order i comes from its own stream (seed, i): the same in every configuration and run
    for e in episodes:
        order = WorkOrder.random(np.random.default_rng([config.seed, e.index]), heights=(1, 1))
        assert e.heights == "-".join(str(len(r)) for r in order.recipes)


def test_mixed_orders_report_buffer_use():
    config = CellEvalConfig(n=1, heights=(1, 1), mixed=True)
    assert CellEvalConfig(mixed=True).n_cubes == 14 and CellEvalConfig().n_cubes == 12
    summary, episodes = evaluate_cell(None, None, config)
    e = episodes[0]
    assert e.success and e.needed_rejected == 0
    assert e.cubes == 3 + e.rejected  # one cube per tower; any extra was surplus
    assert summary["buffer"]["rejected"] == e.rejected


def test_skill_specs():
    assert skill_from_spec("scripted", "pick")[0] is None
    with pytest.raises(ValueError):
        skill_from_spec("runs/grasp", "pick")
