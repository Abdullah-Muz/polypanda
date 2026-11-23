"""v4.0: the tending line, the cell on a rail with machines at their own stops (docs/v4_plan.md).

The v4.0 line keeps the cell's infeed conveyor, buffer slots, reject bin and tower A (the outfeed
stack) exactly where they are around rail stop 0, and adds machines at further stops along the
rail, each at the cell's standard pose in front of its stop (`radius` from the base axis, at
table height). The robot drives to a machine's stop for machine work; every skill then runs in
its station frame as at a fixed station (simulation/rail.py).

Default: one machine, M1, at stop -0.6 m, on the side away from the conveyor (station angle -90).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from polypanda.simulation.line_dressing import dress_line
from polypanda.simulation.machine import RIB_GEOM, Machine, MachineConfig, Rib, add_machine
from polypanda.simulation.rail import RailConfig, RailIndices, add_rail
from polypanda.simulation.scenes import CubeIndices
from polypanda.simulation.workcell import (
    CUBE_NAME,
    Conveyor,
    Workcell,
    WorkcellConfig,
    WorkOrder,
    build_workcell_spec,
    station_xy,
)


@dataclass(frozen=True)
class MachineSpot:
    name: str
    stop: float                       # m along the rail
    angle: float = -90.0              # deg about the base axis at that stop
    config: MachineConfig = field(default_factory=MachineConfig)
    # where the machine actually stands, relative to its programmed place (a variant): an
    # offset on the bench (m, world x and y) and a turn (deg) about its pocket's centre
    offset: tuple[float, float] = (0.0, 0.0)
    turn: float = 0.0


def _line_cell() -> WorkcellConfig:
    return WorkcellConfig(tower_angles=(-40.0,))     # tower A only: the outfeed stack


@dataclass(frozen=True)
class LineConfig:
    cell: WorkcellConfig = field(default_factory=_line_cell)
    # travel just past the stops in use (0 and M1's -0.6), so the drawn rail with its end caps
    # stays on the bench (x from -0.85) and short of the buffer slots (x from 0.44)
    rail: RailConfig = field(default_factory=lambda: RailConfig(low=-0.65, high=0.15))
    machines: tuple[MachineSpot, ...] = (MachineSpot("M1", -0.6),)
    # keyed parts (off by default; docs/keyed_part.md): every part gets a rib on its local +x
    # face, a second geom on its body (the part's box stays its main geom), so it fits a keyed
    # pocket (MachineConfig.keyed) only one way round
    keyed_parts: bool = False
    rib: Rib = Rib()

    def station_xy(self, stop: float, angle_deg: float) -> np.ndarray:
        """Where a station at `angle_deg` in front of rail stop `stop` sits on the bench."""
        return np.asarray(self.rail.axis[:2]) * stop + station_xy(angle_deg, self.cell.radius)


def build_line_spec(cfg: LineConfig | None = None) -> mujoco.MjSpec:
    cfg = cfg or LineConfig()
    spec = build_workcell_spec(cfg.cell)
    if cfg.cell.dressing:
        dress_line(spec, cfg.cell)   # the line's industrial look (visual only)
    add_rail(spec, cfg.rail)
    for spot in cfg.machines:
        xy = cfg.station_xy(spot.stop, spot.angle) + np.asarray(spot.offset)
        add_machine(spec, spot.name, xy, spot.angle + spot.turn, spot.config)
    if cfg.keyed_parts:
        for k in range(cfg.cell.n_cubes):
            add_rib(spec, CUBE_NAME.format(k), cfg.rib)
    return spec


RIB_RGBA = (0.3, 0.32, 0.35, 1.0)     # dark, so the part's orientation reads at a glance


def add_rib(spec: mujoco.MjSpec, part: str, rib: Rib) -> None:
    """A keyed part's rib: a box on the part's local +x face, centred across it, standing on
    its bottom, with the part's own contact properties and density. 10 x 8 x 12 mm of the
    part's density weighs 1.0 g against the part's 96 g; it moves the centre of mass 0.24 mm
    out towards the rib and 0.24 mm down (negligible)."""
    box = spec.geom(part)
    half = np.asarray(box.size)
    spec.body(part).add_geom(
        name=RIB_GEOM.format(part), type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[rib.proud / 2, rib.width / 2, rib.height / 2],
        pos=[half[0] + rib.proud / 2, 0.0, -half[2] + rib.height / 2],
        friction=list(box.friction), solref=list(box.solref), condim=box.condim,
        contype=box.contype, conaffinity=box.conaffinity, density=box.density,
        rgba=list(RIB_RGBA))


class Line(Workcell):
    """The compiled line: the cell, its rail and its machines."""

    line: LineConfig
    machines: dict[str, Machine]
    spots: dict[str, MachineSpot]

    @classmethod
    def build_line(cls, cfg: LineConfig | None = None) -> Line:
        cfg = cfg or LineConfig()
        model = build_line_spec(cfg).compile()
        line = cls(model, mujoco.MjData(model), cfg.cell)
        line.line = cfg
        line.cubes = [CubeIndices.from_model(model, CUBE_NAME.format(k))
                      for k in range(cfg.cell.n_cubes)]
        line.conveyor = Conveyor(line)
        line.rail = RailIndices.from_model(model)
        line.spots = {s.name: s for s in cfg.machines}
        line.machines = {s.name: Machine(model, line.data, s.name, s.config, line.cubes)
                         for s in cfg.machines}
        line.reset_line()
        return line

    def reset_line(self, order: WorkOrder | None = None) -> None:
        """Robot home at rail stop 0, belt and cubes as Workcell.reset, machines empty with
        their lids closed and unlatched."""
        self.reset(order or WorkOrder(recipes=((),), arrivals=()))
        self.data.ctrl[self.rail.actuator] = 0.0
        for m in self.machines.values():
            self.data.qpos[m.lid_qpos] = 0.0
            self.data.qvel[m.lid_dof] = 0.0
            m.reset()
        mujoco.mj_forward(self.model, self.data)

    def update_machines(self) -> None:
        for m in self.machines.values():
            m.update()
