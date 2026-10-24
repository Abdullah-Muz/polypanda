from dataclasses import replace
from pathlib import Path

import numpy as np

from polypanda.environments.grasp import GraspConfig
from polypanda.evaluation.sets import (
    GraspEvalSet,
    PickPlaceEvalSet,
    generate_grasp_eval_set,
    generate_pickplace_eval_set,
)
from polypanda.simulation.scenes import CUBE_X_RANGE, CUBE_Y_RANGE, WIDE_X_RANGE, WIDE_Y_RANGE
from polypanda.supervisor.fsm import SupervisorConfig

ROOT = Path(__file__).parents[1]


def test_nominal_sets_still_regenerate_exactly():
    grasp = GraspEvalSet.load(ROOT / "eval_sets" / "grasp_nominal_v1.json")
    assert generate_grasp_eval_set(grasp.name, len(grasp), grasp.seed).digest == grasp.digest
    pp = PickPlaceEvalSet.load(ROOT / "eval_sets" / "pickplace_nominal_v1.json")
    assert generate_pickplace_eval_set(pp.name, len(pp), pp.seed).digest == pp.digest


def test_wide_sets_cover_the_wide_region():
    g = generate_grasp_eval_set("w", 400, 1, WIDE_X_RANGE, WIDE_Y_RANGE)
    x, y = g.placements[:, 0], g.placements[:, 1]
    assert x.min() < CUBE_X_RANGE[0] and x.max() > CUBE_X_RANGE[1]
    assert (x >= WIDE_X_RANGE[0]).all() and (x <= WIDE_X_RANGE[1]).all()
    assert (np.abs(y) <= WIDE_Y_RANGE[1]).all() and np.abs(y).max() > CUBE_Y_RANGE[1]
    assert g.x_range == WIDE_X_RANGE
    p = generate_pickplace_eval_set("w", 50, 1, x_range=WIDE_X_RANGE, y_range=WIDE_Y_RANGE)
    assert p.x_range == WIDE_X_RANGE and (p.targets[:, 0] >= WIDE_X_RANGE[0]).all()


def test_wide_training_config_changes_only_the_region():
    nominal = GraspConfig.from_yaml(ROOT / "configs" / "grasp_curriculum.yaml")
    wide = GraspConfig.from_yaml(ROOT / "configs" / "grasp_wide_curriculum.yaml")
    assert wide.placement.x_range == WIDE_X_RANGE and wide.placement.y_range == WIDE_Y_RANGE
    assert replace(wide, placement=nominal.placement) == nominal


def test_default_retry_region_is_the_nominal_region():
    assert SupervisorConfig().retry_region == (CUBE_X_RANGE, CUBE_Y_RANGE)
