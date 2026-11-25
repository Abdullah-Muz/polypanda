"""v4.1: the learned insertion's committed evaluation sets (scripts/make_insertion_sets.py,
docs/v4_1_targets.md): read only, nothing runs on them."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from polypanda.evaluation.insertion import EVAL_SPAWN_KEY

SETS = {"insertion_dev_v1": (400, 20261127, False), "insertion_final_v1": (600, 20261128, False),
        "insertion_stretch_v1": (400, 20261129, True)}


@pytest.mark.parametrize("name", sorted(SETS))
def test_the_committed_sets_are_the_targets_sets(name):
    """docs/v4_1_targets.md's three sets: sizes, seeds, the stretch distribution, the evaluation
    stream, a grasp in every episode and an unchanged digest (read only: nothing runs on them)."""
    n, seed, stretch = SETS[name]
    data = json.loads(Path(f"eval_sets/{name}.json").read_text())
    eps = data["episodes"]
    assert data["name"] == name and data["n"] == n == len(eps) and data["seed"] == seed
    assert data["stretch"] == stretch and data["spawn_key"] == EVAL_SPAWN_KEY
    body = json.dumps(eps, sort_keys=True).encode()
    assert hashlib.sha256(body).hexdigest()[:12] == data["digest"]
    assert all("grasp" in e for e in eps)
    clip = 0.004 if stretch else 0.003
    assert max(np.hypot(*e["camera_xy"]) for e in eps) <= clip + 1e-12
    assert data["variation"]["camera_xy"] == (0.0015 if stretch else 0.0008)
