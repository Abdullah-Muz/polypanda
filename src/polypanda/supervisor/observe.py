"""The work-order cell as it physically is (v3, closed loop): which cubes stand on each tower and in
each buffer slot, and which have fallen loose, read from the simulation rather than from the plan.

A cube stands on a tower at level k when it is upright (tilt under 5 deg), its centre is at that
level's height (within 5 mm) and within 2 cm of the fixture, and every level below is occupied:
the same tolerances as the final tower check (ScriptedCell.check). A cube is in a buffer slot when
it stands upright on the table within 2 cm of the slot. A cube is loose when it has arrived (been
loaded onto the belt) but is on no tower or slot, not on the belt, not in the reject bin and not
held; loose cubes carry their rest state (simulation.disturbances.rest_state).
"""

from dataclasses import dataclass, field

import numpy as np

from polypanda.simulation.disturbances import rest_state
from polypanda.simulation.workcell import Workcell

LEVEL_TOL, XY_TOL, TILT_TOL = 0.005, 0.02, 5.0


@dataclass
class CellView:
    towers: list[list[int]]                  # per tower, cube indices standing, bottom first
    parked: list[int | None]                 # per buffer slot, the cube standing there
    loose: dict[int, dict] = field(default_factory=dict)  # cube index -> rest state

    def height(self, t: int) -> int:
        return len(self.towers[t])


def _tilt(cell: Workcell, k: int) -> float:
    return float(np.degrees(np.arccos(np.clip(cell.data.xmat[cell.cubes[k].body][8], -1, 1))))


def observe(cell: Workcell, held: int | None = None) -> CellView:
    """Read the cell. `held`: a cube in the gripper, if any (never loose)."""
    d, cfg = cell.data, cell.config
    arrived = range(cell.conveyor.next_arrival)
    candidates = [k for k in arrived if k != held and not cell.attached(cell.cubes[k])
                  and not cell.conveyor.on_belt(cell.cubes[k])]
    used: set[int] = set()
    towers = []
    for t in range(len(cfg.tower_angles)):
        fixture, stack = cell.tower_xy(t), []
        while True:
            z = cell.cubes[0].half_size[2] * (2 * len(stack) + 1)
            here = [k for k in candidates if k not in used
                    and abs(d.xpos[cell.cubes[k].body][2] - z) < LEVEL_TOL
                    and np.linalg.norm(d.xpos[cell.cubes[k].body][:2] - fixture) < XY_TOL
                    and _tilt(cell, k) < TILT_TOL]
            if not here:
                break
            stack.append(here[0])
            used.add(here[0])
        towers.append(stack)
    parked: list[int | None] = []
    for s in range(len(cfg.buffer_angles)):
        slot = cell.buffer_xy(s)
        here = [k for k in candidates if k not in used
                and np.linalg.norm(d.xpos[cell.cubes[k].body][:2] - slot) < XY_TOL
                and abs(d.xpos[cell.cubes[k].body][2] - cell.cubes[k].half_size[2]) < LEVEL_TOL
                and _tilt(cell, k) < TILT_TOL]
        parked.append(here[0] if here else None)
        used.update(here[:1])
    loose = {}
    for k in candidates:
        if k in used:
            continue
        p = d.xpos[cell.cubes[k].body]
        if np.abs(p[:2] - cell.reject_xy).max() < cfg.bin_size / 2:
            continue  # rejected
        loose[k] = rest_state(cell, cell.cubes[k])
    return CellView(towers, parked, loose)
