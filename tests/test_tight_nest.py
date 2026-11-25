"""v4.1: tight test nests (machine.tight_nest) and stiff part contacts (LineConfig.part_contact)."""

from dataclasses import replace

import numpy as np
import pytest

from polypanda.simulation.contacts import CONTACTS, contact
from polypanda.simulation.line import LineConfig, MachineSpot, build_line_spec
from polypanda.simulation.machine import MachineConfig, tight_nest


def test_a_tight_nest_is_keyed_stiff_and_chamfered():
    cfg = tight_nest(clearance=0.0003, chamfer=0.001)
    assert cfg.keyed and cfg.contact == "stiff" and cfg.clearance == 0.0003
    assert cfg.pocket_depth - cfg.pocket_straight == pytest.approx(0.001)
    with pytest.raises(ValueError):
        tight_nest(clearance=0.0001)            # below what stiff contacts resolve
    with pytest.raises(ValueError):
        MachineConfig(contact="rubber")


def test_the_nest_and_the_parts_get_the_stiff_contacts():
    cfg = LineConfig(machines=(MachineSpot("M1", -0.6, config=tight_nest()),),
                     keyed_parts=True, part_contact="stiff")
    m = build_line_spec(cfg).compile()
    stiff = CONTACTS["stiff"]
    for g in range(m.ngeom):
        name = m.geom(g).name
        if name.startswith("M1_pocket") or name.startswith("cube"):
            np.testing.assert_allclose(m.geom_solref[g], stiff["solref"], err_msg=name)
            np.testing.assert_allclose(m.geom_solimp[g], stiff["solimp"], err_msg=name)


def test_the_default_line_is_unchanged():
    """v4.0's line (no tight nest, default part contacts) compiles to the same contact
    parameters as before v4.1: pocket solref 0.01 and the geoms' default solimp."""
    m = build_line_spec(LineConfig()).compile()
    pocket = [g for g in range(m.ngeom) if m.geom(g).name.startswith("M1_pocket")]
    assert pocket
    for g in pocket:
        np.testing.assert_allclose(m.geom_solref[g], [0.01, 1.0])
        np.testing.assert_allclose(m.geom_solimp[g], [0.9, 0.95, 0.001, 0.5, 2.0])


def test_contact_returns_copies():
    c = contact("stiff")
    c["solref"][0] = 99.0
    assert CONTACTS["stiff"]["solref"][0] == 0.004
    with pytest.raises(ValueError):
        contact("soft-ish")
    assert replace(MachineConfig(), contact="stiff").contact == "stiff"
