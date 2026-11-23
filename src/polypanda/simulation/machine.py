"""v4: a benchtop machine to tend (docs/v4_plan.md, sections 3.2 and 3.5).

A box on the bench with a fixture pocket on its floor, a roof with an opening over the pocket and
a lid that slides on top of the roof to close it. The robot opens the lid by its handle, sets a
part into the pocket, closes the lid until it latches, and the machine runs a cycle; when the
cycle ends the lid unlocks and the part is taken out again.

Frame: the machine's body sits at its station point (the pocket's centre, on the bench) and is
turned by the station angle, so its local +x points away from the robot (depth), +y to the
robot's left as it faces the machine, +z up.

Physical (collides): the four walls, the roof around the opening, the lid, the handle and its
posts, and the pocket. Visual only: the window, the control panel, the spindle head, the status
lamp, the hazard trim.

The lid is a slide joint, damped, with Coulomb friction. It **latches** when it comes within
`latch_catch` of closed moving at least `latch_speed` (a firm last push): an equality constraint
then holds it shut. The machine starts only with a part seated in the pocket and the lid latched
(the interlock), keeps the lid locked while it runs, and unlocks it at the end of the cycle.
Opening the lid or losing the part mid-cycle is a fault.

Everything a machine variant can change is a field of MachineConfig: where the handle sits and how
high, which way the lid slides, how far, how stiffly, the latch, and the colour. Each variant is a
separate compiled model (sizes and joint axes are compile-time); variants are drawn per episode.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from polypanda.simulation.scenes import CubeIndices

EMPTY, LOADED, RUNNING, DONE, FAULT = "empty", "loaded", "running", "done", "fault"
LAMP_FOR_STATE = {EMPTY: "green", LOADED: "green", RUNNING: "amber", DONE: "blue", FAULT: "red"}
LAMPS = ("blue", "green", "amber", "red")          # bottom to top
_LAMP_ON = {"blue": (0.18, 0.49, 0.88, 1.0), "green": (0.12, 0.75, 0.23, 1.0),
            "amber": (1.0, 0.62, 0.0, 1.0), "red": (0.9, 0.15, 0.12, 1.0)}
_LAMP_OFF = {k: (0.22 * v[0], 0.22 * v[1], 0.22 * v[2], 1.0) for k, v in _LAMP_ON.items()}
SLIDES = ("right", "left", "out")   # lid opens to the robot's right, its left, or away from it
_CUBE_HALF = 0.02                   # the part's half width (the cell's 4 x 4 x 6 cm blocks)
RIB_GEOM = "{}_rib"                 # a keyed part's rib: a second geom on the part's body
_KEY_BOSS = 0.003                   # m of wall kept behind a key slot


@dataclass(frozen=True)
class Rib:
    """The key of a keyed part (docs/keyed_part.md): a small tab on the part's local +x face,
    centred across it and standing on the part's bottom, so the part fits a keyed pocket only
    one way round. It is low, under the pocket's 15 mm rim, where the fingers (which close on
    the part's upper half, 19 mm and more above its bottom) never reach. It stands 8 mm proud:
    across the rib the part is then 48 mm, 4 mm more than the 44 mm pocket, so a part the wrong
    way round cannot be pushed in by squeezing the soft contacts (at 5 mm proud, 1 mm of
    interference, a wrong part pressed in by the arm crept down to the floor in seconds)."""
    width: float = 0.010              # m, across the face
    proud: float = 0.008              # m, out of the face
    height: float = 0.012             # m, up from the part's bottom


def rib_geom(model: mujoco.MjModel, part: CubeIndices) -> int:
    """The id of the part's rib geom, or -1 for a plain part."""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                             RIB_GEOM.format(model.body(part.body).name))


@dataclass(frozen=True)
class MachineConfig:
    # --- body (m) ---
    width: float = 0.32               # along y, across the robot's view
    depth: float = 0.28               # along x, away from the robot
    height: float = 0.17              # top of the roof
    wall: float = 0.01
    roof: float = 0.008
    opening: tuple[float, float] = (0.24, 0.10)   # (depth, width) of the hole over the pocket
    colour: tuple[float, float, float] = (0.72, 0.74, 0.77)
    # --- lid ---
    slide: str = "right"              # one of SLIDES
    travel: float = 0.12              # m, closed to fully open
    lid_overlap: float = 0.015        # m the lid overlaps the opening on each side
    lid_thickness: float = 0.008
    lid_mass: float = 0.35            # kg
    lid_damping: float = 4.0          # N s/m
    lid_friction: float = 0.4         # N, Coulomb friction of the slide
    lid_spring: float = 0.0           # N/m, pulls the lid closed (0: none)
    # --- handle: a bar on two posts, across the slide direction ---
    handle_offset: float = 0.0        # m, along the bar's own axis from the lid's centre line
    handle_shift: float = 0.0         # m, along the slide axis from the lid's centre
    handle_height: float = 0.035      # m, bar centre above the lid's top
    handle_length: float = 0.12
    # the bar's section: a flat-faced rectangular pull bar, 16 mm across (along the slide)
    # and 24 mm tall, so a finger meets its face even a few mm high or low
    handle_radius: float = 0.008      # half the bar's width along the slide (16 mm: a bar
                                      # narrower than ~16 mm jams the Panda's fingers, measured)
    handle_half_height: float = 0.012
    # --- latch ---
    latch_catch: float = 0.002        # m from closed within which it can latch
    latch_speed: float = 0.02         # m/s closing speed needed to latch (a firm last push)
    fault_open: float = 0.005         # m: the lid opened this far while running is a fault
    # --- the fixture pocket (as probed in Phase 0) ---
    clearance: float = 0.002          # m per side
    pocket_depth: float = 0.015
    pocket_straight: float = 0.010
    lead_in_deg: float = 45.0
    pocket_wall: float = 0.010
    # --- the key (off by default): a slot in one pocket wall that takes a keyed part's rib, so
    # the part is seated only one way round (docs/keyed_part.md) ---
    keyed: bool = False
    key_wall: int = 0                 # which wall has the slot: k x 90 deg about +z from the
                                      # machine's +x (0: the back wall, away from the robot)
    rib: Rib = Rib()                  # the rib the slot is cut for
    key_clearance: float = 0.002      # m: slot width = rib + 2 x this; slot depth = rib + this
    key_tolerance: float = 10.0       # deg: a seated keyed part's rib points at the slot within
    # --- the cycle ---
    cycle_s: float = 40.0             # s of simulated time

    def __post_init__(self):
        if self.slide not in SLIDES:
            raise ValueError(f"slide must be one of {SLIDES}, not {self.slide!r}")
        if self.keyed and self.rib.width / 2 + self.key_clearance >= _CUBE_HALF:
            raise ValueError("the key slot is wider than the part")

    @property
    def key_yaw(self) -> float:
        """The direction the slot opens to from the pocket's centre (rad, machine frame)."""
        return self.key_wall % 4 * np.pi / 2

    # derived geometry, in the machine's frame
    @property
    def slide_axis(self) -> np.ndarray:
        """Unit vector the lid moves along to open."""
        return {"right": np.array([0.0, -1.0, 0.0]), "left": np.array([0.0, 1.0, 0.0]),
                "out": np.array([1.0, 0.0, 0.0])}[self.slide]

    @property
    def lid_half(self) -> np.ndarray:
        return np.array([self.opening[0] / 2 + self.lid_overlap,
                         self.opening[1] / 2 + self.lid_overlap, self.lid_thickness / 2])

    @property
    def lid_top(self) -> float:
        return self.height + self.lid_thickness

    @property
    def bar_axis(self) -> np.ndarray:
        """The handle bar runs across the slide direction, so the fingers close along it and
        pull or push the lid with their faces."""
        a = self.slide_axis
        return np.array([-a[1], a[0], 0.0]) if self.slide == "out" else np.array([1.0, 0.0, 0.0])

    def handle_centre(self, lid_q: float = 0.0) -> np.ndarray:
        """The bar's centre in the machine's frame, with the lid open by `lid_q`."""
        a = self.slide_axis
        return (self.bar_axis * self.handle_offset + a * (self.handle_shift + lid_q)
                + np.array([0.0, 0.0, self.lid_top + self.handle_height]))


def add_machine(spec: mujoco.MjSpec, name: str, xy, angle_deg: float,
                cfg: MachineConfig | None = None) -> MachineConfig:
    """Add a machine named `name` with its pocket's centre at `xy` on the bench, facing a robot
    whose base axis lies in direction `angle_deg` + 180 from it (its +x points away from the
    robot). Returns the config used."""
    cfg = cfg or MachineConfig()
    a = np.radians(angle_deg)
    body = spec.worldbody.add_body(name=name, pos=[float(xy[0]), float(xy[1]), 0.0],
                                   quat=[np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])
    box = mujoco.mjtGeom.mjGEOM_BOX
    rgba = [*cfg.colour, 1.0]
    hx, hy, top = cfg.depth / 2, cfg.width / 2, cfg.height
    wall_h = top - cfg.roof
    solid = {"solref": [0.01, 1.0]}
    # walls
    body.add_geom(name=f"{name}_wall_front", type=box, size=[cfg.wall / 2, hy, wall_h / 2],
                  pos=[-hx + cfg.wall / 2, 0, wall_h / 2], rgba=rgba, **solid)
    body.add_geom(name=f"{name}_wall_back", type=box, size=[cfg.wall / 2, hy, wall_h / 2],
                  pos=[hx - cfg.wall / 2, 0, wall_h / 2], rgba=rgba, **solid)
    for side, sign in (("left", 1), ("right", -1)):
        body.add_geom(name=f"{name}_wall_{side}", type=box,
                      size=[hx - cfg.wall, cfg.wall / 2, wall_h / 2],
                      pos=[0, sign * (hy - cfg.wall / 2), wall_h / 2], rgba=rgba, **solid)
    # roof around the opening
    ox, oy = cfg.opening[0] / 2, cfg.opening[1] / 2
    roof_rgba = [0.5 * c for c in cfg.colour] + [1.0]
    rz = top - cfg.roof / 2
    for tag, size, pos in (("front", [(hx - ox) / 2, hy, cfg.roof / 2], [-(hx + ox) / 2, 0, rz]),
                           ("back", [(hx - ox) / 2, hy, cfg.roof / 2], [(hx + ox) / 2, 0, rz]),
                           ("left", [ox, (hy - oy) / 2, cfg.roof / 2], [0, (hy + oy) / 2, rz]),
                           ("right", [ox, (hy - oy) / 2, cfg.roof / 2], [0, -(hy + oy) / 2, rz])):
        body.add_geom(name=f"{name}_roof_{tag}", type=box, size=size, pos=pos, rgba=roof_rgba,
                      **solid)
    _add_pocket(body, name, cfg)
    _add_lid(spec, body, name, cfg)
    _add_dressing(body, name, cfg)
    body.add_site(name=f"{name}_pocket", pos=[0, 0, 0], size=[0.004] * 3, group=4)
    return cfg


def _add_pocket(body, name: str, cfg: MachineConfig) -> None:
    """The fixture: four walls around a square opening, a straight part under a sloped lead-in
    (the geometry of scripts/probe_fixture_insertion.py)."""
    a = _CUBE_HALF + cfg.clearance
    theta = np.radians(cfg.lead_in_deg)
    rise = cfg.pocket_depth - cfg.pocket_straight
    run = rise / np.tan(theta)
    length, half_thick = rise / np.sin(theta), 0.003
    along = np.array([np.cos(theta), 0.0, np.sin(theta)])
    normal = np.array([-np.sin(theta), 0.0, np.cos(theta)])
    lead_pos = np.array([a, 0.0, cfg.pocket_straight]) + along * length / 2 - normal * half_thick
    lead_quat = [np.cos(theta / 2), 0.0, -np.sin(theta / 2), 0.0]
    extra = {"solref": [0.01, 1.0], "rgba": [0.36, 0.4, 0.44, 1.0]}
    box = mujoco.mjtGeom.mjGEOM_BOX
    w = cfg.pocket_wall
    for k in range(4):
        q = [np.cos(k * np.pi / 4), 0.0, 0.0, np.sin(k * np.pi / 4)]
        side = body.add_body(name=f"{name}_pocket{k}", quat=q)
        if cfg.keyed and k == cfg.key_wall % 4:
            _add_keyed_wall(side, f"{name}_pocket", k, cfg, a, lead_pos, lead_quat, length,
                            half_thick, run, rise, extra)
            continue
        side.add_geom(name=f"{name}_pocket_wall{k}", type=box,
                      size=[w / 2, a + w, cfg.pocket_straight / 2],
                      pos=[a + w / 2, 0.0, cfg.pocket_straight / 2], **extra)
        side.add_geom(name=f"{name}_pocket_lead{k}", type=box, size=[length / 2, a + w, half_thick],
                      pos=lead_pos.tolist(), quat=lead_quat, **extra)
        side.add_geom(name=f"{name}_pocket_back{k}", type=box,
                      size=[(w - run) / 2, a + w, rise / 2],
                      pos=[a + run + (w - run) / 2, 0.0, cfg.pocket_straight + rise / 2], **extra)


def _add_keyed_wall(side, prefix: str, k: int, cfg: MachineConfig, a: float, lead_pos,
                    lead_quat, length: float, half_thick: float, run: float, rise: float,
                    extra: dict) -> None:
    """A pocket wall with the key slot: the wall's three pieces (straight part, lead-in, block
    behind the lead-in) split either side of a slot through the wall's full height, and a block
    closing the slot behind (at least `_KEY_BOSS` thick, standing out behind the wall where the
    slot reaches further than the wall is thick). The slot is the rib's width plus
    `key_clearance` each side, and reaches `rib.proud + key_clearance` from the pocket's inner
    face, so a rib on a part pushed against this wall (by the pocket's own clearance) still has
    `key_clearance` to spare."""
    box = mujoco.mjtGeom.mjGEOM_BOX
    w, s = cfg.pocket_wall, cfg.rib.width / 2 + cfg.key_clearance     # s: half the slot's width
    half_y = (a + w - s) / 2                                           # each side piece
    back = a + cfg.rib.proud + cfg.key_clearance                       # the slot's floor
    outer = max(a + w, back + _KEY_BOSS)
    for tag, sign in (("a", 1.0), ("b", -1.0)):
        y = sign * (s + half_y)
        side.add_geom(name=f"{prefix}_wall{k}{tag}", type=box,
                      size=[w / 2, half_y, cfg.pocket_straight / 2],
                      pos=[a + w / 2, y, cfg.pocket_straight / 2], **extra)
        side.add_geom(name=f"{prefix}_lead{k}{tag}", type=box,
                      size=[length / 2, half_y, half_thick],
                      pos=[lead_pos[0], y, lead_pos[2]], quat=lead_quat, **extra)
        side.add_geom(name=f"{prefix}_back{k}{tag}", type=box,
                      size=[(w - run) / 2, half_y, rise / 2],
                      pos=[a + run + (w - run) / 2, y, cfg.pocket_straight + rise / 2], **extra)
    side.add_geom(name=f"{prefix}_slot{k}", type=box,
                  size=[(outer - back) / 2, s, cfg.pocket_depth / 2],
                  pos=[(outer + back) / 2, 0.0, cfg.pocket_depth / 2], **extra)


def _add_lid(spec: mujoco.MjSpec, body, name: str, cfg: MachineConfig) -> None:
    half = cfg.lid_half
    lid = body.add_body(name=f"{name}_lid", pos=[0.0, 0.0, cfg.height + half[2]])
    lid.add_joint(name=f"{name}_lid", type=mujoco.mjtJoint.mjJNT_SLIDE,
                  axis=cfg.slide_axis.tolist(), range=[0.0, cfg.travel],
                  limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=cfg.lid_damping,
                  frictionloss=cfg.lid_friction, stiffness=cfg.lid_spring, springref=0.0,
                  armature=0.02)
    lid.add_geom(name=f"{name}_lid", type=mujoco.mjtGeom.mjGEOM_BOX, size=half.tolist(),
                 mass=cfg.lid_mass, rgba=[0.55, 0.6, 0.66, 1.0], solref=[0.01, 1.0])
    # the handle: a bar on two posts, on the lid's top
    c = cfg.handle_centre() - np.array([0.0, 0.0, cfg.height + half[2]])  # in the lid's frame
    bar = cfg.bar_axis
    yaw = float(np.arctan2(bar[1], bar[0]))
    # the bar's length runs along its z: turn z onto the bar's direction (horizontal)
    q_bar = _quat_z_to(bar)
    # (after q_bar, the box's local x is vertical and its local y runs along the closing
    # direction)
    lid.add_geom(name=f"{name}_handle", type=mujoco.mjtGeom.mjGEOM_BOX,
                 size=[cfg.handle_half_height, cfg.handle_radius, cfg.handle_length / 2],
                 pos=c.tolist(), quat=q_bar,
                 rgba=[0.12, 0.12, 0.13, 1.0], mass=0.05, friction=[1.0, 0.03, 0.003])
    post_h = cfg.handle_height
    for s in (-1, 1):
        p = c + bar * s * (cfg.handle_length / 2 - 0.01)
        p[2] = half[2] + post_h / 2
        lid.add_geom(name=f"{name}_handle_post{(s + 1) // 2}", type=mujoco.mjtGeom.mjGEOM_BOX,
                     size=[0.006, 0.006, post_h / 2], pos=p.tolist(),
                     quat=[np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)],
                     rgba=[0.12, 0.12, 0.13, 1.0], mass=0.01)
    lid.add_site(name=f"{name}_handle", pos=c.tolist(), size=[0.004] * 3, group=4)
    # the latch: holds the lid joint at its closed position while active
    spec.add_equality(name=f"{name}_latch", type=mujoco.mjtEq.mjEQ_JOINT,
                      objtype=mujoco.mjtObj.mjOBJ_JOINT, name1=f"{name}_lid", active=False,
                      data=[0.0] * 11)


def _quat_z_to(v: np.ndarray) -> list[float]:
    """A quaternion turning +z onto the horizontal unit vector v."""
    yaw = float(np.arctan2(v[1], v[0]))
    # +z -> +x by turning about +y by +90 deg, then about +z by yaw
    qy = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])
    qz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, qz, qy)
    return out.tolist()


def _visual(body, **kw):
    return body.add_geom(contype=0, conaffinity=0, **kw)


def _add_dressing(body, name: str, cfg: MachineConfig) -> None:
    box, cyl = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER
    hx, hy, top = cfg.depth / 2, cfg.width / 2, cfg.height
    _visual(body, name=f"{name}_window", type=box, size=[0.001, 0.11, 0.045],
            pos=[-hx - 0.001, 0.0, 0.09], rgba=[0.6, 0.85, 0.95, 0.45])
    _visual(body, name=f"{name}_panel", type=box, size=[0.05, 0.006, 0.045],
            pos=[-hx / 3, hy + 0.006, 0.1], rgba=[0.22, 0.24, 0.27, 1.0])
    _visual(body, name=f"{name}_screen", type=box, size=[0.032, 0.001, 0.022],
            pos=[-hx / 3, hy + 0.0125, 0.115], rgba=[0.43, 0.83, 1.0, 1.0])
    ox, oy = cfg.opening[0] / 2, cfg.opening[1] / 2
    for tag, size, pos in (("f", [0.004, oy + 0.008, 0.0006], [-ox - 0.004, 0, top + 0.0006]),
                           ("b", [0.004, oy + 0.008, 0.0006], [ox + 0.004, 0, top + 0.0006]),
                           ("l", [ox, 0.004, 0.0006], [0, oy + 0.004, top + 0.0006]),
                           ("r", [ox, 0.004, 0.0006], [0, -oy - 0.004, top + 0.0006])):
        _visual(body, name=f"{name}_trim_{tag}", type=box, size=size, pos=pos,
                rgba=[0.95, 0.76, 0.0, 1.0])
    # the status lamp at the back corner on the robot's left
    lx, ly = hx - 0.03, hy - 0.03
    _visual(body, name=f"{name}_lamp_base", type=cyl, size=[0.012, 0.004],
            pos=[lx, ly, top + 0.004], rgba=[0.12, 0.12, 0.13, 1.0])
    for k, lamp in enumerate(LAMPS):
        _visual(body, name=f"{name}_lamp_{lamp}", type=cyl, size=[0.01, 0.0085],
                pos=[lx, ly, top + 0.0175 + 0.018 * k], rgba=list(_LAMP_OFF[lamp]))
    # the spindle head: parked clear of the opening; it lowers over the part while running
    _visual(body, name=f"{name}_spindle", type=cyl, size=[0.016, 0.035],
            pos=list(_spindle_pos(cfg, 0.0)), rgba=[0.84, 0.86, 0.88, 1.0])
    _visual(body, name=f"{name}_tool", type=cyl, size=[0.004, 0.012],
            pos=list(_spindle_pos(cfg, 0.0) - [0, 0, 0.047]), rgba=[0.3, 0.3, 0.32, 1.0])


def _spindle_pos(cfg: MachineConfig, working: float) -> np.ndarray:
    """The spindle's centre: parked under the roof beside the opening (working = 0), or over the
    part with its tool tip 5 mm above a 6 cm part (working = 1)."""
    z = 0.06 + 0.005 + 0.059                  # tool tip = centre - 0.059 (spindle and tool)
    parked = np.array([0.0, cfg.opening[1] / 2 + 0.035, z])
    over = np.array([0.0, 0.0, z])
    return parked + (over - parked) * working


class Machine:
    """The runtime side of a machine: its lid, latch, interlock, cycle and lamp. Call `update`
    once per control step (e.g. from the cell controller's on_step)."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, name: str,
                 cfg: MachineConfig, parts: list[CubeIndices]) -> None:
        self.m, self.d, self.name, self.cfg, self.parts = model, data, name, cfg, parts
        j = model.joint(f"{name}_lid")
        self.lid_qpos, self.lid_dof = int(j.qposadr[0]), int(j.dofadr[0])
        self.latch_eq = model.equality(f"{name}_latch").id
        self.body = model.body(name).id
        self.pocket_site = model.site(f"{name}_pocket").id
        self.handle_site = model.site(f"{name}_handle").id
        self.lamps = {k: model.geom(f"{name}_lamp_{k}").id for k in LAMPS}
        self.spindle = model.geom(f"{name}_spindle").id
        self.tool = model.geom(f"{name}_tool").id
        self.finished_rgba = (0.75, 0.55, 0.15, 1.0)    # machined: a bronze-like finish
        self.reset()

    # --- state ------------------------------------------------------------------------------
    def reset(self) -> None:
        self.state = EMPTY
        self.part: CubeIndices | None = None
        self.started = None
        self.finished_parts: list[CubeIndices] = []
        self._peak_closing = 0.0      # fastest closing speed in the last centimetre (m/s)
        self._armed = self.lid > 0.005
        self._prev_q, self._prev_t = self.lid, float(self.d.time)
        self.d.eq_active[self.latch_eq] = 0
        self._show()

    @property
    def lid(self) -> float:
        """How far the lid is open (m; 0 = closed)."""
        return float(self.d.qpos[self.lid_qpos])

    @property
    def latched(self) -> bool:
        return bool(self.d.eq_active[self.latch_eq])

    def unlatch(self) -> None:
        """Release the lid. It stays closed until something pulls it open, and the latch
        catches again only when it is pushed shut after being opened."""
        self.d.eq_active[self.latch_eq] = 0
        self._armed, self._peak_closing = False, 0.0

    def handle_xyz(self) -> np.ndarray:
        return self.d.site_xpos[self.handle_site].copy()

    def pocket_xyz(self) -> np.ndarray:
        return self.d.site_xpos[self.pocket_site].copy()

    def in_pocket(self, part: CubeIndices) -> tuple[bool, bool]:
        """(the part is over the pocket and low, it is seated): seated means on the pocket's
        floor (gap under 1.5 mm), upright (tilt under 3 deg) and inside the opening; in a keyed
        pocket, a keyed part must also be the right way round (its rib in the slot: see
        `key_error`)."""
        d = self.d
        R_m = d.xmat[self.body].reshape(3, 3)
        rel = R_m.T @ (d.xpos[part.body] - d.xpos[self.body])
        R = d.xmat[part.body].reshape(3, 3)
        tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))
        gap = float(rel[2] - part.half_size[2])
        inside = bool(np.abs(rel[:2]).max() <= self.cfg.clearance + 0.0005)
        near = bool(np.abs(rel[:2]).max() < 0.04 and rel[2] < 0.06)
        seated = inside and gap < 0.0015 and tilt < 3.0
        if seated and self.cfg.keyed:
            err = self.key_error(part)
            seated = err is None or abs(err) <= self.cfg.key_tolerance
        return near, bool(seated)

    def key_error(self, part: CubeIndices) -> float | None:
        """How far a keyed part's rib points away from the slot (deg, about the vertical,
        wrapped to -180..180; 0 = the right way round, +-90 or 180 = a wrong one). None for an
        unkeyed pocket or a plain part (without a rib, every way round is right)."""
        if not self.cfg.keyed or rib_geom(self.m, part) < 0:
            return None
        R_m = self.d.xmat[self.body].reshape(3, 3)
        rib = R_m.T @ self.d.xmat[part.body].reshape(3, 3)[:, 0]   # the rib's face normal
        err = np.degrees(np.arctan2(rib[1], rib[0]) - self.cfg.key_yaw)
        return float((err + 180.0) % 360.0 - 180.0)

    def seated_part(self) -> CubeIndices | None:
        for p in self.parts:
            if self.in_pocket(p)[1]:
                return p
        return None

    # --- per control step -------------------------------------------------------------------
    def update(self) -> None:
        q, t = self.lid, float(self.d.time)
        # closing speed over the last control step, from the lid's travel (a firm push can shut
        # it within one step, between two samples of its velocity)
        dt = t - self._prev_t
        closing = (self._prev_q - q) / dt if dt > 0 else 0.0
        self._prev_q, self._prev_t = q, t
        if q > 0.005:
            self._armed = True        # the latch catches only a lid pushed shut after opening
        self._peak_closing = max(self._peak_closing, closing) if q <= 0.01 else 0.0
        if (self._armed and not self.latched and q <= self.cfg.latch_catch
                and self._peak_closing >= self.cfg.latch_speed):
            self.d.eq_active[self.latch_eq] = 1
            self.m.eq_data[self.latch_eq, 0] = 0.0        # the latch pulls it fully shut
            self._armed, self._peak_closing = False, 0.0
        seated = self.seated_part()
        if self.state == EMPTY:
            if seated is not None:
                self.state, self.part = LOADED, seated
        elif self.state == LOADED:
            if seated is None:
                self.state, self.part = EMPTY, None
            elif self.latched:
                self.state, self.started = RUNNING, float(self.d.time)
        elif self.state == RUNNING:
            if seated is None or q > self.cfg.fault_open:
                self.state = FAULT
            elif self.d.time - self.started >= self.cfg.cycle_s:
                self.state = DONE
                self.m.geom_rgba[self.part.geom] = self.finished_rgba
                self.finished_parts.append(self.part)
                self.unlatch()
        elif self.state == DONE:
            if self.part is not None and not self.in_pocket(self.part)[0]:
                self.state, self.part = EMPTY, None
        self._show()

    def _show(self) -> None:
        on = LAMP_FOR_STATE[self.state]
        for k, g in self.lamps.items():
            self.m.geom_rgba[g] = _LAMP_ON[k] if k == on else _LAMP_OFF[k]
        working = 1.0 if self.state == RUNNING else 0.0
        p = _spindle_pos(self.cfg, working)
        self.m.geom_pos[self.spindle] = p
        self.m.geom_pos[self.tool] = p - [0.0, 0.0, 0.047]
