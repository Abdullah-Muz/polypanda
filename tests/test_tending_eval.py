"""v4.0 Phase 1: machine variants and the scripts' beliefs (evaluation/tending.py)."""

import numpy as np
import pytest

from polypanda.evaluation import tending as T


def test_level_zero_is_the_nominal_machine_for_every_property():
    nominal = T.MachineConfig()
    for prop in T.LEVELS:
        v = T.make_variant(prop, 0, np.random.default_rng(0))
        assert v.offset == (0.0, 0.0) and v.turn == 0.0
        if prop == "slide":
            assert v.config.slide == nominal.slide and v.config.travel == nominal.travel
        else:
            assert v.config == nominal


def test_variants_are_drawn_from_the_seed_alone():
    a = T.make_variant("position", 3, np.random.default_rng(7))
    b = T.make_variant("position", 3, np.random.default_rng(7))
    assert a == b
    assert np.hypot(*a.offset) == pytest.approx(T.LEVELS["position"][3])


def test_the_handle_never_drops_below_20_mm():
    for seed in range(20):
        v = T.make_variant("handle_height", 5, np.random.default_rng(seed))
        assert v.config.handle_height >= 0.02


def test_the_nominal_belief_ignores_where_the_machine_really_stands():
    v = T.Variant("position", 4, 0.05, offset=(0.05, 0.0))
    line = T.build(v)
    belief = T.nominal_belief(line)
    np.testing.assert_allclose(belief.origin[:2], [-0.6, -0.5], atol=1e-9)
    np.testing.assert_allclose(line.machines["M1"].pocket_xyz()[:2], [-0.55, -0.5], atol=1e-9)


def test_the_camera_belief_is_the_truth_within_its_error():
    v = T.Variant("turn", 3, 10.0, turn=10.0)
    line = T.build(v)
    belief = T.camera_belief(line, np.random.default_rng(1))
    m1 = line.machines["M1"]
    assert np.abs(belief.origin - m1.pocket_xyz()).max() < 0.005       # 5 sigma
    assert np.abs(belief.handle - m1.handle_xyz()).max() < 0.005
    assert np.degrees(belief.yaw) == pytest.approx(-80.0, abs=2.5)
    assert belief.config.slide == "right"                              # what it cannot see


def test_both_scripts_load_the_nominal_machine():
    v = T.make_variant("latch", 0, np.random.default_rng(3))
    for k, script in enumerate(("nominal", "camera")):
        row = T.run_trial(v, script, 100 + k)
        assert row["success"], row
        assert row["machine_contacts"] == 0 and row["ik_failures"] == 0
