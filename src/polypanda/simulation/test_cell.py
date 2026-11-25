"""v4.1: the electronics test cell around the testers (docs/v4_1_plan.md, section 1).

The v4.0 line grown into a test cell: three testers on one side of the rail, the infeed conveyor,
a barcode scanner over its pick point, a vision station, a PASS tray and the FAIL bin on the
other. Every station stands at the cell's standard pose in front of its rail stop (0.5 m from the
base axis, on the bench at z = 0), so every skill, scripted or learned, works there as at a fixed
station (simulation/rail.py). World frame, the robot's base riding the rail along x:

    station              rail stop   angle    centre (x, y)       what it is
    infeed conveyor          0.0      +90     ( 0.0,    0.5)      the cell's belt, unchanged
      barcode scanner        0.0      +90     over the pick point  a camera (scan = label read)
    tester A  "TA"          -0.9      -90     (-0.9,   -0.5)      lid tester (each with a tight,
    tester B  "TB"          -0.45     -90     (-0.45,  -0.5)      drawer (*)  keyed, stiff nest)
    tester C  "TC"           0.0      -90     ( 0.0,   -0.5)      pneumatic (*)
                                              (*) a lid tester stands in until track B merges
    vision station          -0.45    +150     (-0.883,  0.25)     a camera; the robot presents the
                                                                  held module to it (no set-down)
    PASS tray               -0.9     +150     (-1.333,  0.25)     2 x 4 pockets
    FAIL bin                -0.9     -150     (-1.333, -0.25)     the cell's reject bin, moved, red,
                                                                  in a red-taped quality zone
    REWORK tray             -0.45    +150     (-0.883,  0.25)     2 x 2 pockets on the bench, under
                                                                  the presented module: modules
                                                                  the robot could not get tested

Why this layout: the conveyor's belt and frame run along y = 0.5 from x = -0.88 (its downstream
end, past the pick point) to +0.98, with the hopper beside its upstream end: all of it is the
v4.0 cell's, unchanged, so the +y side at radius 0.5 is taken from x = -0.88 on. The bench ends
at x = -0.85. The testers are 0.32 m wide, so they stand 0.45 m apart (13 cm gaps; a lid sliding
12 cm to the robot's right overhangs its tester by 2.5 cm into a gap). The output side (vision,
tray, bin) goes to -x, at +-150 deg (between the rail and the conveyor, and past the conveyor's
end), onto a 0.7 m **bench extension** (x from -1.55 to -0.85, drawn; the bench's collision
plane is infinite already), and the rail's travel is extended to -0.95..0.1 so every stop is in
range. Nothing stands at 180 deg: joint 1 reaches +-166 deg, and the cell's carry posture turns
the base only. The PASS tray is on the same side as the vision station (a short swing for the
~92 % that pass); the FAIL bin across (a long swing, for the few that fail). The REWORK tray
stands on the bench at the vision station, under where the robot presents a module (the camera's
view passes over it; the -y side has no room for it between the bin, tester A and the rail's
cable carrier). Tower A (v4.0's outfeed stack) and the buffer slots are not used; the tower is
left out and the slots' painted squares are hidden.

**The vision station** is a camera on a post that looks at a module **held in the carry posture**
over the station's point (`present_z` its centre's height there), at the module's inspected face,
from the outer side, `vision_elevation` above the horizontal: the robot brings the module, turns
its wrist so that face looks at the camera, stops briefly and goes on (control/test_cell.py,
`present`). Nothing stands on the bench there (v4.1's first build set the module down on a plate
under the camera and picked it up again; the plate is gone).

**The REWORK tray** takes the modules the robot could not get tested (one that would not seat in a
nest after its retries, one knocked out of its seat before its test started, one whose test was
aborted): they are not judged good or bad; an operator takes them to the rework bench (in the
hall's red quality zone, out of the robot's reach) for a retest. Drawn as the PASS tray is, in
orange (5S: on hold).

**Testers.** `TesterSpot` is the line's MachineSpot with the `kind` it is built as and the kind the
plan gives it (`planned`): A a lid, B a drawer, C a pneumatic tester (`PLANNED_KINDS`). The drawer
and the pneumatic tester are built on another branch (`MachineConfig.kind`, track B); until that
merges, `built_kinds()` is ("lid",) and B and C are **lid testers standing in** for them (v4.0's
machine with `machine.tight_nest` as its nest), marked `planned="drawer"` / `"pneumatic"` and
recorded as stand-ins in every result. Once MachineConfig has a `kind`, `planned_cell()` builds
the plan's three kinds (the controller already opens each kind its own way:
control/test_cell.py). The cycle (the test) defaults to 45 s, the middle of the plan's 30-60 s.

**Modules.** The cell's parts, keyed (`LineConfig.keyed_parts`) and stiff (`part_contact`), drawn
as sensor modules: a dark housing (the part's own box), a green board edge around it, a white
label with a barcode on its **front** (the side face at the part's local -y), gold contact pads
underneath. Every one of these is a visual geom on the part's body (contype 0, conaffinity 0,
mass 0): the part's mass, inertia and collision geometry are exactly the plain part's
(tests/test_test_cell.py checks it). Why the front and not the top: the cell grips a part at its
centre, so a held part's top is 4 mm under the hand's palm, which no camera sees past; the
front is a face the fingers leave free when a module is taken out of a keyed nest (they close on
the rib's face and the face opposite), and it faces the scanner on the conveyor (the belt carries
the parts with their local +x along its travel, so their front faces away from the robot).

**Hidden truth.** Each module of a shift carries a truth drawn from its own seeded stream
(`draw_truths`): about 5 % fail the electrical test (an open circuit, a short, or an RF
measurement out of spec: what the tester finds), about 3 % have a visible defect, drawn as
something a camera can see: a missing label (the label and barcode are hidden) or a red mark on
the housing's front, under the label (a burn or a damaged-seal mark), of a drawn **severity**: its
size, from a faint 2.4 x 1.6 mm spot to a 12 x 8 mm mark, so the small ones are hard to see.
Independently, about 4 % of modules carry harmless brownish **flux residue** on the front, beside
where a mark would be (cosmetic: the module
is good if nothing else is wrong), which a colour rule can mistake for a mark. The truth is never
read by the controller except through the tester's result (electrical) and the two cameras
(visible), whose images carry exposure and sensor noise (perception/inspection.py).

**Bodies are recycled.** The model has `n_bodies` part bodies; a shift may have more modules.
The conveyor (`ModuleFeeder`) loads the next module of the shift onto any free body waiting in
the hopper, and gives it that module's look. When the PASS or REWORK tray is full or the FAIL bin
holds a few modules, an operator takes them away (`release`: the bodies go back into the hopper,
out of sight). This is a simulation shortcut for the operator's tray exchange; it costs the robot
no time, and it is counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import mujoco
import numpy as np

from polypanda.perception.camera import CameraSetup, add_camera
from polypanda.simulation.line import Line, LineConfig, MachineSpot, build_line_spec
from polypanda.simulation.machine import MachineConfig, tight_nest
from polypanda.simulation.rail import RailConfig, RailIndices
from polypanda.simulation.scenes import CubeIndices
from polypanda.simulation.workcell import (
    CUBE_NAME,
    Conveyor,
    WorkcellConfig,
    WorkOrder,
    _yaw_quat,
    hopper_xy,
    queue_xy,
)

KINDS = ("lid", "drawer", "pneumatic")
PLANNED_KINDS = {"TA": "lid", "TB": "drawer", "TC": "pneumatic"}   # docs/v4_1_plan.md, section 1
HIDDEN_GROUP = 3                  # a geom group the renderers and the viewer do not draw
DEFECTS = ("missing_label", "red_mark")                       # visible
ELECTRICAL_DEFECTS = ("open_circuit", "short", "rf_out_of_spec")
MIN_SEVERITY = 0.2                # the smallest red mark, as a share of the largest's size

HOUSING_RGBA = (0.13, 0.14, 0.15, 1.0)
LABEL_RGBA = (0.95, 0.95, 0.93, 1.0)
BAR_RGBA = (0.04, 0.04, 0.05, 1.0)
MARK_RGBA = (0.92, 0.06, 0.05, 1.0)
RESIDUE_RGBA = (0.62, 0.21, 0.13, 1.0)  # brownish-red flux residue (harmless)
EDGE_RGBA = (0.05, 0.42, 0.2, 1.0)     # the PCB's green
PAD_RGBA = (0.86, 0.69, 0.27, 1.0)     # gold
FAIL_RED = (0.78, 0.1, 0.08, 1.0)
REWORK_ORANGE = (0.95, 0.5, 0.08, 1.0)
PRESENT_Z = 0.315                 # m: a held module's centre in the carry posture (the TCP at
                                  # 0.32 m, less the arm's sag; measured)
FRONT_UP = 0.010                  # m: the label's layout sits this far above the front's centre
FRONT_AIM = 0.0085                # m above the front's centre: the middle of that layout (from
                                  # the label's top to a mark's bottom), where cameras aim


def built_kinds() -> tuple[str, ...]:
    """The tester kinds this build can make: all three once MachineConfig has a `kind` (track B,
    simulation/machine.py), else only the lid."""
    return KINDS if "kind" in MachineConfig.__dataclass_fields__ else ("lid",)


@dataclass(frozen=True)
class TesterSpot(MachineSpot):
    """A tester at a rail stop: the line's MachineSpot plus the kind it is built as (`kind`) and
    the kind the plan gives it (`planned`; a lid tester with planned "drawer" is a stand-in)."""
    kind: str = "lid"
    planned: str | None = None    # None: the same as `kind`

    def __post_init__(self):
        for k in (self.kind, self.planned_kind):
            if k not in KINDS:
                raise ValueError(f"kind must be one of {KINDS}, not {k!r}")
        built = getattr(self.config, "kind", "lid")
        if built != self.kind:
            raise ValueError(f"tester {self.name}: kind {self.kind!r} but its machine is {built!r}")

    @property
    def planned_kind(self) -> str:
        return self.planned or self.kind

    @property
    def stand_in(self) -> bool:
        return self.planned_kind != self.kind


def tester(name: str, stop: float, kind: str = "lid", planned: str | None = None,
           cycle_s: float = 45.0, **machine) -> TesterSpot:
    """A tester of `kind` (with a tight, keyed, stiff nest) at rail stop `stop`, facing the
    robot from -90 deg. `machine`: further MachineConfig fields."""
    if kind not in built_kinds():
        raise NotImplementedError(f"tester {name}: kind {kind!r} is not built on this branch "
                                  "(MachineConfig.kind comes with the tester kinds, track B)")
    if kind != "lid":
        machine["kind"] = kind
    cfg = tight_nest(MachineConfig(cycle_s=cycle_s, **machine))
    return TesterSpot(name, stop, -90.0, config=cfg, kind=kind, planned=planned)


STOPS = {"TA": -0.9, "TB": -0.45, "TC": 0.0}


def stand_in_testers(cycle_s: float = 45.0) -> tuple[TesterSpot, ...]:
    """The plan's three testers, all built as lid testers (B and C standing in for the drawer
    and the pneumatic tester): the default, on every build, so that merging the tester kinds
    changes no result by itself."""
    return tuple(tester(n, STOPS[n], "lid", planned=k, cycle_s=cycle_s)
                 for n, k in PLANNED_KINDS.items())


def planned_testers(cycle_s: float = 45.0) -> tuple[TesterSpot, ...]:
    """The plan's three testers as their own kinds (needs MachineConfig.kind, track B)."""
    return tuple(tester(n, STOPS[n], k, cycle_s=cycle_s) for n, k in PLANNED_KINDS.items())


@dataclass(frozen=True)
class Station:
    """A station in front of a rail stop: its stop (m) and its angle about the base axis (deg)."""
    stop: float
    angle: float


@dataclass(frozen=True)
class TestCellConfig:
    testers: tuple[TesterSpot, ...] = field(default_factory=stand_in_testers)
    infeed: Station = Station(0.0, 90.0)          # fixed by the cell's conveyor (do not move)
    vision: Station = Station(-0.45, 150.0)
    tray: Station = Station(-0.9, 150.0)
    fail_bin: Station = Station(-0.9, -150.0)
    rework: Station = Station(-0.45, 150.0)       # on the bench at the vision station
    # the PASS tray: rows along the radius (the fingers close radially, so radial neighbours
    # need room for the open fingers), columns across it
    tray_rows: int = 2
    tray_cols: int = 4
    tray_pitch: tuple[float, float] = (0.10, 0.075)     # m: radial, tangential
    rework_rows: int = 2           # the REWORK tray, laid out as the PASS tray
    rework_cols: int = 2
    rail: RailConfig = field(default_factory=lambda: RailConfig(low=-0.95, high=0.1))
    bench_extension: tuple[float, float] = (-1.55, -0.85)   # m of x, drawn
    n_bodies: int = 20             # part bodies in the model (recycled through a shift)
    theme: str = "electronics"
    dressing: bool = True
    # cameras: square images, small and fast (software rendering), 0.28 m from the module's
    # front (aimed at its label-and-mark layout, FRONT_AIM), looking at it straight from the
    # outer side: the scanner 45 deg down at a module on
    # the belt, the vision camera `vision_elevation` down at a held one (the hand above it, 4 mm
    # over its top, overhangs the front by 9 mm: from 25 deg it hides the top, not the front)
    camera_px: int = 96
    camera_fovy: float = 18.0
    camera_distance: float = 0.28
    vision_elevation: float = 25.0     # deg
    present_z: float = PRESENT_Z

    def __post_init__(self):
        for t in self.testers:
            if t.kind not in built_kinds():
                raise NotImplementedError(
                    f"tester {t.name}: kind {t.kind!r} is not built on this branch yet")

    def line_config(self) -> LineConfig:
        cell = WorkcellConfig(tower_angles=(), n_cubes=self.n_bodies, dressing=self.dressing)
        return LineConfig(cell=cell, rail=self.rail, machines=self.testers, keyed_parts=True,
                          part_contact="stiff", theme=self.theme)

    def xy(self, station: Station) -> np.ndarray:
        return self.line_config().station_xy(station.stop, station.angle)

    def tray_pockets(self, which: str = "pass") -> list[np.ndarray]:
        """The PASS (or "rework") tray's pockets (x, y), row by row, the inner row first."""
        station, rows, cols = ((self.tray, self.tray_rows, self.tray_cols) if which == "pass"
                               else (self.rework, self.rework_rows, self.rework_cols))
        c = self.xy(station)
        a = np.radians(station.angle)
        u, t = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        return [c + (r - (rows - 1) / 2) * self.tray_pitch[0] * u
                + (k - (cols - 1) / 2) * self.tray_pitch[1] * t
                for r in range(rows) for k in range(cols)]

    def camera(self, name: str) -> CameraSetup:
        """The scanner (over the conveyor's pick point: a module's front on the belt) or the
        vision station's camera (a held module's front, presented over the station's point)."""
        station = {"scanner": self.infeed, "vision": self.vision}[name]
        a = np.radians(station.angle)
        out = np.array([np.cos(a), np.sin(a), 0.0])
        if name == "scanner":                 # the module's centre on the belt (2 mm lift)
            centre, elevation = 0.032, 45.0
        else:
            centre, elevation = self.present_z, self.vision_elevation
        target = np.array([*self.xy(station), centre + FRONT_AIM]) + 0.02 * out   # on the front
        e = np.radians(elevation)
        pos = target + self.camera_distance * (np.cos(e) * out + np.array([0.0, 0.0, np.sin(e)]))
        return CameraSetup(name, tuple(pos), tuple(target), self.camera_fovy, self.camera_px,
                           self.camera_px)


# --- hidden truth ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModuleTruth:
    """What a module really is. `electrical_pass`: what a correctly seated test reports, and if
    it fails, why (`electrical_defect`, one of ELECTRICAL_DEFECTS); `defect`: None or one of
    DEFECTS (visible), with its `severity` (a red mark's size, MIN_SEVERITY..1 of the largest);
    `residue`: harmless flux residue's size (0: none, else 0.3..1 of an 8 x 5.6 mm patch)."""
    serial: str
    electrical_pass: bool
    defect: str | None = None
    electrical_defect: str | None = None
    severity: float = 1.0
    residue: float = 0.0

    def __post_init__(self):
        if self.defect is not None and self.defect not in DEFECTS:
            raise ValueError(f"defect must be None or one of {DEFECTS}")
        if not self.electrical_pass and self.electrical_defect is None:
            object.__setattr__(self, "electrical_defect", ELECTRICAL_DEFECTS[0])

    @property
    def good(self) -> bool:
        """Good: passes the electrical test and has no visible defect (residue is harmless)."""
        return self.electrical_pass and self.defect is None

    @property
    def defects(self) -> list[str]:
        return [d for d in (self.electrical_defect, self.defect) if d is not None]


def draw_truths(n: int, seed: int, fail_rate: float = 0.05, defect_rate: float = 0.03,
                residue_rate: float = 0.04) -> list[ModuleTruth]:
    """The hidden truth of a shift's `n` modules, from their own stream (seeded by `seed` alone,
    so nothing the robot does changes it). Seven draws per module, always, so module k's truth
    does not depend on the rates or the other modules."""
    rng = np.random.default_rng([seed, 4101])
    out = []
    for k in range(n):
        e, ekind, v, vkind, sev, r, rsize = rng.random(7)
        fails = bool(e < fail_rate)
        defect = DEFECTS[int(vkind * len(DEFECTS))] if v < defect_rate else None
        out.append(ModuleTruth(
            f"M{seed:04d}-{k:04d}", not fails, defect,
            ELECTRICAL_DEFECTS[int(ekind * len(ELECTRICAL_DEFECTS))] if fails else None,
            round(float(MIN_SEVERITY + (1 - MIN_SEVERITY) * sev), 3) if defect else 1.0,
            round(float(0.3 + 0.7 * rsize), 3) if r < residue_rate else 0.0))
    return out


# --- building -------------------------------------------------------------------------------
def _vis(body, size, pos, rgba, name=None, gtype=mujoco.mjtGeom.mjGEOM_BOX, **kw):
    if name is not None:
        kw["name"] = name
    return body.add_geom(type=gtype, size=list(size), pos=list(pos), rgba=list(rgba),
                         contype=0, conaffinity=0, mass=0.0, **kw)


BARS = (1, 2, 1, 1, 3, 1, 2, 1, 1, 2, 1)            # barcode bar widths, in units of 1 mm
MARK_HALF = (0.006, 0.004)        # m: the largest red mark's half sizes (severity 1)
RESIDUE_HALF = (0.004, 0.0028)    # m: the largest residue patch's half sizes


def add_module_look(spec: mujoco.MjSpec, part: str) -> None:
    """Draw part `part` (a 40 x 40 x 60 mm box) as a sensor module: visual geoms on its body
    only. Its own box becomes the dark housing (Workcell/TestCell colour it at reset). The label
    and anything visible that is wrong with the module are on its front (local -y): seen from
    outside the front, its x runs from left to right and z up, in the layout v4.1's first build
    had on the top (label above, a mark or residue below), raised by FRONT_UP to clear the board
    edge. Box sizes are (across, depth, up)."""
    body = spec.body(part)
    hx, hy, hz = spec.geom(part).size
    e = 0.0002                                    # proud of a face, so it draws over it
    z_edge = -hz + 0.022                          # the board edge, above the key rib
    for k, (sx, sy) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
        size = [e, hy + e, 0.0009] if sx else [hx + e, e, 0.0009]
        _vis(body, size, [sx * (hx + e), sy * (hy + e), z_edge], EDGE_RGBA, f"{part}_edge{k}")

    def front(u, w, half_u, half_w, proud, rgba, name, **kw):
        # (the front faces -y: seen from there, +x is to the left, so u runs along -x)
        _vis(body, [half_u, proud, half_w], [-u, -hy - proud, FRONT_UP + w], rgba, name, **kw)

    # the label with its barcode; the red mark under it (hidden unless defective)
    label_c = 0.005
    front(0.0, label_c, 0.014, 0.008, 0.00015, LABEL_RGBA, f"{part}_label")
    unit = 0.001
    x = -sum(BARS) * unit / 2
    for k, w in enumerate(BARS):
        if k % 2 == 0:                            # bars and gaps alternate
            front(x + w * unit / 2, label_c, w * unit / 2, 0.0055, 0.0004, BAR_RGBA,
                  f"{part}_bar{k}")
        x += w * unit
    front(0.0, -0.012, *MARK_HALF, 0.00015, MARK_RGBA, f"{part}_mark", group=HIDDEN_GROUP)
    front(0.0115, -0.012, *RESIDUE_HALF, 0.00012, RESIDUE_RGBA, f"{part}_residue",
          group=HIDDEN_GROUP)
    for i in range(3):                            # contact pads underneath
        for j in range(2):
            _vis(body, [0.003, 0.002, 0.0001], [(i - 1) * 0.011, (j - 0.5) * 0.012, -hz - 0.0001],
                 PAD_RGBA, f"{part}_pad{2 * i + j}")


def _text(spec, name, lines, fg, bg, size=(256, 96)):
    from polypanda.simulation.line_dressing import _text_texture
    return _text_texture(spec, name, lines, size=size, fg=fg, bg=bg)


def _add_bench_extension(spec: mujoco.MjSpec, cfg: TestCellConfig) -> None:
    from polypanda.simulation.factory import BENCH_Y, FLOOR_Z
    x0, x1 = cfg.bench_extension
    cx, hx = (x0 + x1) / 2, (x1 - x0) / 2
    cy, hy = sum(BENCH_Y) / 2, (BENCH_Y[1] - BENCH_Y[0]) / 2
    world = spec.worldbody
    top = {"material": "worktop"} if spec.material("worktop") is not None else {}
    _vis(world, [hx, hy, 0.025], [cx, cy, -0.0006 - 0.025], [0.6, 0.62, 0.65, 1.0],
         "bench_extension", **top)
    for sx in (-1, 1):
        for sy in (-1, 1):
            _vis(world, [0.03, 0.03, (-0.075 - FLOOR_Z - 0.03) / 2],
                 [cx + sx * (hx - 0.05), cy + sy * (hy - 0.05), (-0.075 + FLOOR_Z + 0.03) / 2],
                 [0.75, 0.77, 0.8, 1.0])


def _add_tray(spec: mujoco.MjSpec, cfg: TestCellConfig, which: str = "pass") -> None:
    """The PASS (or REWORK) tray: an ESD tray with a grid of pockets, drawn (no collision: the
    modules stand on the bench through it, its floor 0.5 mm high). Its dividers are 8 mm tall,
    below the open fingers (19 mm and more above the bench when a module is set down)."""
    station, rows, cols = ((cfg.tray, cfg.tray_rows, cfg.tray_cols) if which == "pass"
                           else (cfg.rework, cfg.rework_rows, cfg.rework_cols))
    c, a = cfg.xy(station), np.radians(station.angle)
    body = spec.worldbody.add_body(name=f"{which}_tray", pos=[*c, 0.0], quat=_yaw_quat(a))
    pr, pc = cfg.tray_pitch
    hr, hc = rows * pr / 2 + 0.01, cols * pc / 2 + 0.01
    _vis(body, [hr, hc, 0.00025], [0, 0, 0.00025], [0.16, 0.2, 0.26, 1.0], f"{which}_tray_floor")
    for r in range(rows + 1):
        _vis(body, [0.003, hc, 0.004], [-hr + 0.01 + r * pr, 0, 0.004], [0.2, 0.26, 0.34, 1.0])
    for k in range(cols + 1):
        _vis(body, [hr, 0.003, 0.004], [0, -hc + 0.01 + k * pc, 0.004], [0.2, 0.26, 0.34, 1.0])
    lip, text, bg = (((0.12, 0.62, 0.25, 1.0), "PASS", (25, 140, 60)) if which == "pass"
                     else (REWORK_ORANGE, "REWORK", (235, 125, 20)))
    _vis(body, [0.004, hc, 0.006], [hr + 0.004, 0, 0.006], lip)             # coloured lip
    if cfg.dressing:
        mat = _text(spec, f"{which}_sign", [text], (255, 255, 255), bg)
        _vis(body, [0.02, 0.07, 0.025], [hr + 0.05, 0, 0.12], [1, 1, 1, 1], material=mat)
        _vis(body, [0.006, 0.006, 0.05], [hr + 0.05, 0, 0.05], [0.75, 0.77, 0.8, 1.0])


def _restyle_fail_bin(spec: mujoco.MjSpec, cfg: TestCellConfig) -> None:
    """Move the cell's reject bin to the FAIL station and paint it red (its walls keep their
    size, contacts and mass)."""
    body = spec.body("reject_bin")
    a = np.radians(cfg.fail_bin.angle)
    body.pos = [*cfg.xy(cfg.fail_bin), 0.0]
    body.quat = _yaw_quat(a)
    for g in body.geoms:
        g.rgba = list(FAIL_RED) if g.contype or g.conaffinity else [0.35, 0.05, 0.04, 1.0]
    if cfg.dressing:
        mat = _text(spec, "fail_sign", ["FAIL"], (255, 255, 255), (190, 25, 20))
        s = cfg.line_config().cell.bin_size / 2
        _vis(body, [0.02, 0.07, 0.025], [s + 0.05, 0, 0.14], [1, 1, 1, 1], material=mat)
        _vis(body, [0.006, 0.006, 0.06], [s + 0.05, 0, 0.06], [0.75, 0.77, 0.8, 1.0])
        # the red quality zone on the bench (5S red: nonconforming parts), taped around the bin
        # and its sign; the hall's rework bench stands in its own red zone outside the guarding
        h, w = s + 0.065, 0.006
        for k in range(4):
            along = k % 2 == 0
            sign = 1 if k < 2 else -1
            size = [w, h + w, 0.0003] if along else [h + w, w, 0.0003]
            pos = [sign * h, 0, 0.0003] if along else [0, sign * h, 0.0003]
            _vis(body, size, pos, FAIL_RED, f"quality_tape{k}")


def _add_station_hardware(spec: mujoco.MjSpec, cfg: TestCellConfig) -> None:
    """The scanner and the vision camera: drawn housings on posts (no collision), the cameras
    themselves with their lights, and the vision camera's ring light."""
    for name in ("scanner", "vision"):
        cam = cfg.camera(name)
        add_camera(spec, cam)
        pos = np.asarray(cam.pos)
        world = spec.worldbody
        R = cam.rotation()
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.flatten())
        # (both behind the camera, which looks along its -z: it must not see its own housing)
        _vis(world, [0.018, 0.014, 0.025], pos + R[:, 2] * 0.037, [0.1, 0.1, 0.11, 1.0],
             f"{name}_housing", quat=list(q))
        _vis(world, [0.012, 0.004], pos + R[:, 2] * 0.008, [0.2, 0.5, 0.95, 1.0] if name ==
             "scanner" else [0.95, 0.95, 1.0, 1.0], f"{name}_lens",
             gtype=mujoco.mjtGeom.mjGEOM_CYLINDER, quat=list(q))
        back = R[:2, 2] / np.linalg.norm(R[:2, 2])         # behind the camera, on the bench
        _vis(world, [0.008, 0.008, (pos[2] + 0.02) / 2], [*(pos[:2] + 0.045 * back),
             (pos[2] - 0.02) / 2], [0.75, 0.77, 0.8, 1.0], f"{name}_post")
        if name == "vision":                               # a ring light round the lens
            _vis(world, [0.03, 0.003], pos + R[:, 2] * 0.010, [0.85, 0.9, 1.0, 1.0],
                 "vision_ring_light", gtype=mujoco.mjtGeom.mjGEOM_CYLINDER, quat=list(q))
        # the camera's own light, a narrow spot from the lens at the module's front (a front
        # faces sideways, which the hall's lights from above leave dim, and differently in
        # each look of the hall), so that what the rules see depends on the module only (the
        # scanner's brighter: it lights the front at 45 deg)
        look = np.asarray(cam.target) - pos
        light = [1.0] * 3 if name == "scanner" else [0.7] * 3
        world.add_light(name=f"{name}_light", pos=list(pos),
                        dir=list(look / np.linalg.norm(look)), diffuse=light,
                        specular=[0.05] * 3, cutoff=12.0, exponent=1.0, castshadow=False)


def build_test_cell_spec(cfg: TestCellConfig | None = None) -> mujoco.MjSpec:
    cfg = cfg or TestCellConfig()
    spec = build_line_spec(cfg.line_config())
    for k in range(cfg.n_bodies):
        add_module_look(spec, CUBE_NAME.format(k))
    for s in range(3):                     # the cell's buffer slots are not used here
        b = spec.body(f"buffer{s}")
        if b is not None:
            for g in b.geoms:
                g.group = HIDDEN_GROUP
    _restyle_fail_bin(spec, cfg)
    _add_tray(spec, cfg, "pass")
    _add_tray(spec, cfg, "rework")
    _add_station_hardware(spec, cfg)
    if cfg.dressing:
        _add_bench_extension(spec, cfg)
    return spec


# --- the runtime ----------------------------------------------------------------------------
class ModuleFeeder(Conveyor):
    """The cell's conveyor fed with a shift's modules: each batch takes free bodies from the
    hopper and gives them the next modules of the shift (TestCell.take_body)."""

    @property
    def due(self) -> int:
        return self.cell.modules_due

    def _load_batch(self) -> None:
        cell, cfg = self.cell, self.cell.config
        normal = np.array([-self.travel[1], self.travel[0]])
        heading = float(np.arctan2(self.travel[1], self.travel[0]))
        for start in queue_xy(cfg):
            if self.due <= 0:
                break
            k = cell.take_body()
            if k is None:
                break
            cube, rng = cell.cubes[k], self._rng
            offset = rng.uniform(-self._lateral, self._lateral) if rng is not None else 0.0
            angle = heading + (rng.uniform(-self._yaw, self._yaw) if rng is not None else 0.0)
            cell.data.qpos[cube.qpos:cube.qpos + 7] = [
                *(start + offset * normal), cube.half_size[2] + cfg.belt_lift, *_yaw_quat(angle)]
            cell.data.qvel[cube.qvel:cube.qvel + 6] = 0.0
            self.next_arrival += 1
        mujoco.mj_forward(cell.model, cell.data)


@dataclass
class _Look:
    housing: int
    label: list[int]
    mark: int
    residue: int


class TestCell(Line):
    """The compiled test cell: the line (rail, testers, conveyor) plus its stations, the
    modules' looks and hidden truths, and the two cameras."""

    cell_config: TestCellConfig

    @classmethod
    def build(cls, cfg: TestCellConfig | None = None) -> TestCell:   # type: ignore[override]
        cfg = cfg or TestCellConfig()
        model = build_test_cell_spec(cfg).compile()
        # the near clipping plane is a share of the model's extent, which the factory hall makes
        # some 40 m (near plane 0.4 m): the inspection cameras look from 0.28 m. 5 cm instead.
        model.vis.map.znear = min(model.vis.map.znear, 0.05 / model.stat.extent)
        line_cfg = cfg.line_config()
        cell = cls(model, mujoco.MjData(model), line_cfg.cell)
        cell.cell_config = cfg
        cell.line = line_cfg
        cell.cubes = [CubeIndices.from_model(model, CUBE_NAME.format(k))
                      for k in range(cfg.n_bodies)]
        cell.rail = RailIndices.from_model(model)
        cell.spots = {s.name: s for s in cfg.testers}
        from polypanda.simulation.machine import Machine
        cell.machines = {s.name: Machine(model, cell.data, s.name, s.config, cell.cubes)
                         for s in cfg.testers}
        for m in cell.machines.values():
            m.finished_rgba = HOUSING_RGBA       # a tested module keeps its housing's colour
        cell.looks = []
        for k in range(cfg.n_bodies):
            name = CUBE_NAME.format(k)
            label = [model.geom(f"{name}_label").id] + [
                model.geom(f"{name}_bar{i}").id for i in range(0, len(BARS), 2)]
            cell.looks.append(_Look(model.geom(name).id, label, model.geom(f"{name}_mark").id,
                                    model.geom(f"{name}_residue").id))
        cell._label_group = int(model.geom_group[cell.looks[0].label[0]])
        cell.pockets = cfg.tray_pockets("pass")
        cell.rework_pockets = cfg.tray_pockets("rework")
        cell._renderer = None
        cell.truths: list[ModuleTruth] = []
        cell.body_module: list[int | None] = [None] * cfg.n_bodies
        cell.loaded = 0
        cell.conveyor = ModuleFeeder(cell)
        cell.reset_shift([])
        return cell

    # --- stations ---------------------------------------------------------------------------
    @property
    def reject_xy(self) -> np.ndarray:            # the FAIL bin (Workcell.in_bin uses it)
        return self.cell_config.xy(self.cell_config.fail_bin)

    @property
    def vision_xy(self) -> np.ndarray:
        """Where a module is presented to the vision camera (held over this point)."""
        return self.cell_config.xy(self.cell_config.vision)

    def tray(self, which: str = "pass") -> list[np.ndarray]:
        return self.pockets if which == "pass" else self.rework_pockets

    def pocket_of(self, part: CubeIndices, tol: float = 0.012, which: str = "pass"
                  ) -> int | None:
        """The PASS (or "rework") tray pocket a part stands in (upright, on the bench, centred
        within `tol`), or None."""
        p = self.data.xpos[part.body]
        up = self.data.xmat[part.body].reshape(3, 3)[2, 2]
        if up < 0.99 or abs(p[2] - part.half_size[2]) > 0.003:
            return None
        for i, xy in enumerate(self.tray(which)):
            if np.linalg.norm(p[:2] - xy) < tol:
                return i
        return None

    def free_pocket(self, which: str = "pass") -> int | None:
        taken = {self.pocket_of(c, which=which) for c in self.cubes}
        return next((i for i in range(len(self.tray(which))) if i not in taken), None)

    # --- the shift's modules ----------------------------------------------------------------
    @property
    def modules_due(self) -> int:
        """Modules of the shift not yet loaded onto the belt."""
        return len(self.truths) - self.loaded

    def module_of(self, part: CubeIndices) -> int | None:
        """The shift's module index a body currently carries (None: a free body)."""
        return self.body_module[self.cubes.index(part)]

    def free_bodies(self) -> list[int]:
        return [k for k, m in enumerate(self.body_module) if m is None]

    def take_body(self) -> int | None:
        """A free body for the next module of the shift (its look set), or None."""
        free = self.free_bodies()
        if not free or self.modules_due <= 0:
            return None
        k = free[0]
        self.body_module[k] = self.loaded
        self.show(k, self.truths[self.loaded])
        self.loaded += 1
        return k

    def release(self, k: int) -> None:
        """An operator takes body k's module away (a full tray, the fail bin emptied, a jam
        cleared): the body goes back into the hopper, free for a later module."""
        cube = self.cubes[k]
        if self.attached(cube):
            self.detach(cube)
        heading = float(np.arctan2(*self.conveyor.travel[::-1]))
        spot = hopper_xy(self.config)[k]
        self.data.qpos[cube.qpos:cube.qpos + 7] = [*spot, cube.half_size[2], *_yaw_quat(heading)]
        self.data.qvel[cube.qvel:cube.qvel + 6] = 0.0
        self.body_module[k] = None
        self.show(k, None)
        mujoco.mj_forward(self.model, self.data)

    def show(self, k: int, truth: ModuleTruth | None) -> None:
        """Give body k a module's look: the housing, the label unless missing, the red mark at
        its severity's size if marked, the residue at its size if any. (Sizes of visual geoms
        only: nothing collides with them.)"""
        look, m = self.looks[k], self.model
        m.geom_rgba[look.housing] = HOUSING_RGBA
        missing = truth is not None and truth.defect == "missing_label"
        for g in look.label:
            m.geom_group[g] = HIDDEN_GROUP if missing else self._label_group
        marked = truth is not None and truth.defect == "red_mark"
        m.geom_group[look.mark] = self._label_group if marked else HIDDEN_GROUP
        m.geom_size[look.mark, ::2] = np.asarray(MARK_HALF) * (truth.severity if marked else 1.0)
        residue = truth.residue if truth is not None else 0.0
        m.geom_group[look.residue] = self._label_group if residue > 0 else HIDDEN_GROUP
        m.geom_size[look.residue, ::2] = np.asarray(RESIDUE_HALF) * (residue or 1.0)

    def reset_shift(self, truths: list[ModuleTruth], rng: np.random.Generator | None = None,
                    lateral: float = 0.0, yaw: float = 0.0) -> None:
        """Start a shift: the robot home at rail stop 0, every body in the hopper, testers empty
        with their lids closed and unlatched, and the first batch of `truths` on the belt."""
        self.truths, self.loaded = list(truths), 0
        self.body_module = [None] * len(self.cubes)
        for k in range(len(self.cubes)):
            self.show(k, None)
        self.reset(WorkOrder(recipes=(), arrivals=()), rng, lateral=lateral, yaw=yaw)
        for k in range(len(self.cubes)):         # (Workcell.reset greys every part)
            self.model.geom_rgba[self.looks[k].housing] = HOUSING_RGBA
        self.data.ctrl[self.rail.actuator] = 0.0
        for m in self.machines.values():
            self.data.qpos[m.lid_qpos] = 0.0
            self.data.qvel[m.lid_dof] = 0.0
            m.reset()
        mujoco.mj_forward(self.model, self.data)

    def tester_result(self, name: str) -> str | None:
        """Tester `name`'s result signal: "pass" or "fail" once its test has run its whole cycle
        (the module's hidden electrical truth: the test of a seated module is taken to be
        right), "fault" if the test was aborted (Machine: lid opened or module unseated while
        running), None while there is no result."""
        from polypanda.simulation.machine import DONE, FAULT
        m = self.machines[name]
        if m.state == FAULT:
            return "fault"
        if m.state != DONE or m.part is None:
            return None
        k = self.module_of(m.part)
        if k is None:
            return None
        return "pass" if self.truths[k].electrical_pass else "fail"

    # --- cameras ----------------------------------------------------------------------------
    def camera_image(self, name: str) -> np.ndarray:
        """An RGB image from the scanner or the vision camera (no shadows or reflections: an
        inspection camera has its own ring light, and it renders 3x faster in software)."""
        if self._renderer is None:
            px = self.cell_config.camera_px
            self._renderer = mujoco.Renderer(self.model, px, px)
        r = self._renderer
        r.update_scene(self.data, camera=name)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        return r.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def cell_config(**changes) -> TestCellConfig:
    """TestCellConfig with fields changed; `cycle_s` sets every tester's test time."""
    cycle = changes.pop("cycle_s", None)
    cfg = TestCellConfig(**changes)
    if cycle is not None:
        cfg = replace(cfg, testers=tuple(replace(t, config=replace(t.config, cycle_s=cycle))
                                         for t in cfg.testers))
    return cfg


# (classes whose names start with "Test" are not pytest test classes)
TestCell.__test__ = False
TestCellConfig.__test__ = False
TesterSpot.__test__ = False
