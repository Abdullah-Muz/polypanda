"""Contact settings for tight fits (v4.1, docs/v4_1_plan.md section 3).

The cell's default contacts let parts and fixtures interpenetrate by 0.3-1.0 mm (95th percentile,
results/v4/nest_insertion_probe.json), more than a test nest's clearance. "stiff" is the stiffest
stable setting at the 2 ms timestep (a time constant of two timesteps) with a narrower, harder
impedance zone; it keeps the depth at or below about 0.2 mm, so nests of 0.25 mm clearance and
wider are resolved. Applied to both sides of a contact (a part and its nest), as the probe did.
"""

CONTACTS = {
    "default": {"solref": [0.01, 1.0], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0]},
    "stiff": {"solref": [0.004, 1.0], "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0]},
}


def contact(name: str) -> dict:
    """The solref and solimp of a named setting (copies, safe to hand to MjSpec)."""
    try:
        c = CONTACTS[name]
    except KeyError:
        raise ValueError(f"unknown contact setting {name!r}; known: {sorted(CONTACTS)}") from None
    return {"solref": list(c["solref"]), "solimp": list(c["solimp"])}
