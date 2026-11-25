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

**Tester kinds (v4.1, docs/v4_1_plan.md section 1).** `MachineConfig.kind`:
- "lid" (the default): the machine above, exactly as in v4.0;
- "drawer": a box with an opening in its front (the side facing the robot) and a drawer on a
  slide joint that opens **toward** the robot (the machine's -x), carrying the nest. Opened, it
  brings the nest out of the box to where the robot loads it from above. The drawer's front plate
  carries a horizontal bar on two posts, standing off the plate so a finger fits between them: the
  robot hooks it from above with open fingers, the finger behind the bar pulling it open and
  pushing the plate shut. The box is the machine body; the drawer, its plate, bar and nest are the
  drawer body. The machine's origin is the nest's centre with the drawer at its open stop (at
  `nest_inset` = the travel, by default), so a drawer is placed like a lid machine: by where the
  robot loads it. Its nest stands on the drawer's tray (`nest_floor` above the bench), and the
  robot reaches into it with its fingers closing across the drawer (`hand_axis`). An **index
  pin** (`Machine.index_open`, signalled by the cell) holds an open drawer at its stop while the
  robot loads it; the door env never engages it (there the drawer slides freely);
- "pneumatic": the lid machine without a handle, its lid driven by a pneumatic cylinder (an
  integrated-velocity servo, so the lid moves at a steady `pneumatic_speed` with at most
  `pneumatic_force`) on the cell's signal: `Machine.request_open()` and `request_close()`. It opens
  itself at the end of a cycle, as shield boxes do. The robot only loads and unloads it.
Latch, interlock, cycle, lamps and keyed nests work the same on every kind: the runtime reads the
kind's slide joint ("door": `Machine.door_qpos`; v4.0's `lid_qpos` and `Machine.lid` name it for
every kind).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from polypanda.simulation.contacts import contact
from polypanda.simulation.scenes import CubeIndices

EMPTY, LOADED, RUNNING, DONE, FAULT = "empty", "loaded", "running", "done", "fault"
LAMP_FOR_STATE = {EMPTY: "green", LOADED: "green", RUNNING: "amber", DONE: "blue", FAULT: "red"}
LAMPS = ("blue", "green", "amber", "red")          # bottom to top
_LAMP_ON = {"blue": (0.18, 0.49, 0.88, 1.0), "green": (0.12, 0.75, 0.23, 1.0),
            "amber": (1.0, 0.62, 0.0, 1.0), "red": (0.9, 0.15, 0.12, 1.0)}
_LAMP_OFF = {k: (0.22 * v[0], 0.22 * v[1], 0.22 * v[2], 1.0) for k, v in _LAMP_ON.items()}
SLIDES = ("right", "left", "out")   # lid opens to the robot's right, its left, or away from it
DRAWER_SLIDE = "toward"             # a drawer opens toward the robot (the machine's -x)
ALL_SLIDES = SLIDES + (DRAWER_SLIDE,)
KINDS = ("lid", "drawer", "pneumatic")
# the moving part's body and slide joint (the same name), per kind
DOOR_NAME = {"lid": "{}_lid", "drawer": "{}_drawer", "pneumatic": "{}_lid"}
_CUBE_HALF = 0.02                   # the part's half width (the cell's 4 x 4 x 6 cm blocks)
_PART_HEIGHT = 0.06
RIB_GEOM = "{}_rib"                 # a keyed part's rib: a second geom on the part's body
_KEY_BOSS = 0.003                   # m of wall kept behind a key slot
DRAWER_PLATE = 0.012                # m, the drawer's front plate (thickness)
DRAWER_RIM = 0.006                  # m, the tray's rims (thickness)
DRAWER_BAR_DROP = 0.008             # m, the drawer bar's nominal centre below the box's top: the
#                                     hooking hand's underside (32 mm above its TCP) then clears
#                                     the roof by 2.4 cm (pushed shut, the hand reaches over it)
POST_HALF = 0.006                   # m, the handle posts' square half section
POST_INSET = 0.01                   # m, the posts stand this far in from the bar's ends
PNEUMATIC_KP = 2000.0               # N/m, the cylinder's servo stiffness
PNEUMATIC_KV = 40.0                 # N s/m, its damping (about critical for the 0.35 kg lid)


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
    detent_depth: float = 0.008       # m: a springy lid is held open in the last this of its
                                      # travel (a hold-open detent, as springy guards have)
    detent_force: float = 1.0         # N the detent holds a springy lid against its stop with
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
    # the pocket's contact setting (simulation/contacts.py): "stiff" for tight nests (v4.1)
    contact: str = "default"
    # --- the cycle ---
    cycle_s: float = 40.0             # s of simulated time
    # --- v4.1: the tester's kind (one of KINDS; see the module's docstring) ---
    kind: str = "lid"
    # the drawer (kind "drawer"). The shared fields keep their meaning where they can: width,
    # depth, height, wall and roof size the box; travel, lid_* and the latch the drawer's slide;
    # the handle fields place the bar on the drawer's front plate: handle_offset along the bar,
    # handle_shift across it on the plate (up), handle_height how far it stands off the plate
    # (the bar's centre; the room for a finger is that less the bar's half width)
    nest_inset: float | None = None   # m the nest sits behind the origin (+x) with the drawer shut
    #                                   (None: the travel, so at its open stop the nest is at the
    #                                   origin; resolved when the config is made, then fixed)
    drawer_width: float = 0.16        # m, the tray across (y)
    drawer_floor: float = 0.01        # m, the tray's top (the nest's floor) above the bench
    drawer_front: float = 0.05        # m from the nest's centre to the front plate's inner face
    #                                   (and to the back rim)
    drawer_rim: float = 0.02          # m, the tray's side and back rims above its floor
    drawer_room: float = 0.03         # m between the nest's back edge and the box's front at the
    #                                   needed opening (needed_opening)
    # the pneumatic lid (kind "pneumatic")
    pneumatic_speed: float = 0.08     # m/s: the nominal 0.12 m of travel in 1.5 s
    pneumatic_force: float = 40.0     # N, the cylinder's force limit (it stalls against a hand)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, not {self.kind!r}")
        if self.kind == "drawer":
            # a drawer always opens toward the robot; the shared default ("right") means that
            if self.slide == "right":
                object.__setattr__(self, "slide", DRAWER_SLIDE)
            if self.slide != DRAWER_SLIDE:
                raise ValueError(f"a drawer slides {DRAWER_SLIDE!r}, not {self.slide!r}")
            if self.nest_inset is None:
                object.__setattr__(self, "nest_inset", float(self.travel))
            self._check_drawer()
        elif self.slide not in SLIDES:
            raise ValueError(f"slide must be one of {SLIDES}, not {self.slide!r}")
        if self.keyed and self.rib.width / 2 + self.key_clearance >= _CUBE_HALF:
            raise ValueError("the key slot is wider than the part")
        contact(self.contact)             # refuses an unknown setting
        if not 0.0 < self.pocket_straight <= self.pocket_depth:
            raise ValueError("the pocket's straight part must be within its depth")

    def _check_drawer(self) -> None:
        """A drawer's parts must fit: the nest in the tray, the tray in the box behind its front,
        a loaded part under the roof, the plate inside the box's width."""
        if self.nest_half + 0.004 > min(self.drawer_front, self.drawer_width / 2 - DRAWER_RIM):
            raise ValueError("the nest does not fit in the drawer's tray")
        if 2 * self.drawer_front + DRAWER_RIM > self.depth - self.wall - 0.01:
            raise ValueError("the drawer is longer than the box")
        if self.drawer_floor + _PART_HEIGHT + 0.005 > self.height - self.roof:
            raise ValueError("a part in the nest does not fit under the roof")
        if self.drawer_width / 2 + 0.012 > self.width / 2 - self.wall:
            raise ValueError("the drawer is wider than the box's front opening allows")

    @property
    def key_yaw(self) -> float:
        """The direction the slot opens to from the pocket's centre (rad, machine frame)."""
        return self.key_wall % 4 * np.pi / 2

    @property
    def nest_half(self) -> float:
        """Half the pocket's outer size in plan (m): its walls, or the block behind a key slot
        where that stands out further."""
        a = _CUBE_HALF + self.clearance
        outer = a + self.pocket_wall
        if self.keyed:
            outer = max(outer, a + self.rib.proud + self.key_clearance + _KEY_BOSS)
        return float(outer)

    # derived geometry, in the machine's frame
    @property
    def slide_axis(self) -> np.ndarray:
        """Unit vector the lid (or drawer) moves along to open."""
        return {"right": np.array([0.0, -1.0, 0.0]), "left": np.array([0.0, 1.0, 0.0]),
                "out": np.array([1.0, 0.0, 0.0]),
                DRAWER_SLIDE: np.array([-1.0, 0.0, 0.0])}[self.slide]

    @property
    def door_origin(self) -> np.ndarray:
        """Where the moving part's body sits, shut (machine frame): the lid's centre, or the
        drawer's nest centre on its floor."""
        if self.kind == "drawer":
            return np.array([self.nest_inset, 0.0, self.drawer_floor])
        return np.array([0.0, 0.0, self.height + self.lid_thickness / 2])

    @property
    def nest_floor(self) -> float:
        """The nest's floor above the bench (m): the drawer's tray top for a drawer, the bench
        itself for a lid or pneumatic tester (whose pocket stands on the bench)."""
        return float(self.drawer_floor) if self.kind == "drawer" else 0.0

    @property
    def hand_axis(self) -> float:
        """The direction (rad, machine frame, mod pi) the robot's fingers must close along when
        it reaches into the nest: the machine's x for a lid (the hand, about 200 x 65 mm, passes
        the lid's 240 x 100 mm opening only lengthwise), its y for a drawer (65 mm thick across
        its fingers, the hand fits between the drawer's front plate and the box; lengthwise it
        would reach over both: needed_opening)."""
        return np.pi / 2 if self.kind == "drawer" else 0.0

    @property
    def drawer_face(self) -> float:
        """A drawer's front plate's outer face, shut (m along the machine's x)."""
        return float(self.nest_inset - self.drawer_front - DRAWER_PLATE)

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
        pull or push the lid with their faces. (A drawer's runs across its front, along +y.)"""
        if self.slide == DRAWER_SLIDE:
            return np.array([0.0, 1.0, 0.0])
        a = self.slide_axis
        return np.array([-a[1], a[0], 0.0]) if self.slide == "out" else np.array([1.0, 0.0, 0.0])

    def handle_centre(self, lid_q: float = 0.0) -> np.ndarray:
        """The bar's centre in the machine's frame, with the lid (or drawer) open by `lid_q`."""
        a = self.slide_axis
        if self.kind == "drawer":
            return (self.bar_axis * self.handle_offset + a * lid_q
                    + np.array([self.drawer_face - self.handle_height, 0.0,
                                self.height - DRAWER_BAR_DROP + self.handle_shift]))
        return (self.bar_axis * self.handle_offset + a * (self.handle_shift + lid_q)
                + np.array([0.0, 0.0, self.lid_top + self.handle_height]))


def tight_nest(cfg: MachineConfig | None = None, clearance: float = 0.0003,
               chamfer: float = 0.001) -> MachineConfig:
    """A v4.1 test nest (docs/v4_1_plan.md, sections 1 and 3): the pocket `clearance` per side
    (at least 0.25 mm, what stiff contacts resolve), a 45-degree `chamfer` on its rim instead of
    Phase 0's 5 mm lead-in, keyed, with stiff contacts. The part needs `part_contact="stiff"` on
    the line too."""
    from dataclasses import replace
    cfg = cfg or MachineConfig()
    if clearance < 0.00025:
        raise ValueError("stiff contacts resolve clearances of 0.25 mm and wider (Phase 0)")
    return replace(cfg, clearance=clearance, pocket_straight=cfg.pocket_depth - chamfer,
                   keyed=True, contact="stiff")


def add_machine(spec: mujoco.MjSpec, name: str, xy, angle_deg: float,
                cfg: MachineConfig | None = None) -> MachineConfig:
    """Add a machine named `name` with its pocket's centre at `xy` on the bench, facing a robot
    whose base axis lies in direction `angle_deg` + 180 from it (its +x points away from the
    robot). Returns the config used. (For a drawer, the pocket's centre with the drawer at its
    open stop: see the module's docstring.)"""
    cfg = cfg or MachineConfig()
    a = np.radians(angle_deg)
    body = spec.worldbody.add_body(name=name, pos=[float(xy[0]), float(xy[1]), 0.0],
                                   quat=[np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])
    if cfg.kind == "drawer":
        _add_drawer_box(body, name, cfg)
        drawer = _add_drawer(spec, body, name, cfg)
        _add_pocket(drawer, name, cfg)
        _add_drawer_dressing(body, name, cfg)
        drawer.add_site(name=f"{name}_pocket", pos=[0, 0, 0], size=[0.004] * 3, group=4)
        return cfg
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
    extra = {**contact(cfg.contact), "rgba": [0.36, 0.4, 0.44, 1.0]}
    if cfg.contact == "default":
        extra.pop("solimp")             # as before v4.1: the geoms' own default solimp
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
    if cfg.kind == "pneumatic":       # no handle: a cylinder moves it
        spec.add_equality(name=f"{name}_latch", type=mujoco.mjtEq.mjEQ_JOINT,
                          objtype=mujoco.mjtObj.mjOBJ_JOINT, name1=f"{name}_lid", active=False,
                          data=[0.0] * 11)
        _add_cylinder(spec, name, cfg)
        return
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


def _add_cylinder(spec: mujoco.MjSpec, name: str, cfg: MachineConfig) -> None:
    """A pneumatic lid's cylinder: an integrated-velocity servo on the lid's joint. The control
    is the speed the cylinder's flow valves set (Machine.request_open, request_close: +-
    pneumatic_speed), its integral the target position, clamped to the travel, so the lid runs
    to a stop at a steady speed, as a flow-controlled cylinder does, and pushes with at most
    pneumatic_force (it stalls against a hand in the way rather than crushing it)."""
    act = spec.add_actuator(name=f"{name}_cylinder", target=f"{name}_lid",
                            trntype=mujoco.mjtTrn.mjTRN_JOINT)
    act.set_to_intvelocity(kp=PNEUMATIC_KP, kv=PNEUMATIC_KV)
    act.actlimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.actrange = [0.0, cfg.travel]
    act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.ctrlrange = [-1.0, 1.0]
    act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.forcerange = [-cfg.pneumatic_force, cfg.pneumatic_force]


def _add_drawer_box(body, name: str, cfg: MachineConfig) -> None:
    """A drawer tester's box: side and back walls, a roof, and a front frame (two jambs) either
    side of the drawer's opening. Its front face is where the drawer's plate rests, shut (the
    plate and the box never collide: the drawer is the box's child body)."""
    box = mujoco.mjtGeom.mjGEOM_BOX
    rgba = [*cfg.colour, 1.0]
    solid = {"solref": [0.01, 1.0]}
    x0 = cfg.nest_inset - cfg.drawer_front           # the box's front face
    hx, hy, top, w = cfg.depth / 2, cfg.width / 2, cfg.height, cfg.wall
    xc, wall_h = x0 + hx, top - cfg.roof
    for side, sign in (("left", 1), ("right", -1)):
        body.add_geom(name=f"{name}_wall_{side}", type=box, size=[hx, w / 2, wall_h / 2],
                      pos=[xc, sign * (hy - w / 2), wall_h / 2], rgba=rgba, **solid)
    body.add_geom(name=f"{name}_wall_back", type=box, size=[w / 2, hy - w, wall_h / 2],
                  pos=[x0 + cfg.depth - w / 2, 0.0, wall_h / 2], rgba=rgba, **solid)
    oy = cfg.drawer_width / 2 + 0.004                # the front opening's half width
    jamb = (hy - w - oy) / 2
    for side, sign in (("left", 1), ("right", -1)):
        body.add_geom(name=f"{name}_wall_front_{side}", type=box,
                      size=[w / 2, jamb, wall_h / 2],
                      pos=[x0 + w / 2, sign * (oy + jamb), wall_h / 2], rgba=rgba, **solid)
    body.add_geom(name=f"{name}_roof", type=box, size=[hx, hy, cfg.roof / 2],
                  pos=[xc, 0.0, top - cfg.roof / 2], rgba=[0.5 * c for c in cfg.colour] + [1.0],
                  **solid)


def drawer_handle_layout(cfg: MachineConfig):
    """A drawer's bar, posts and handle site in the drawer body's frame: (the bar's centre (also
    the site), its quaternion, the two posts' centres, the posts' half size). The bar runs along
    the plate (+y), 16 mm along the slide and 24 mm tall like the lid's; the posts reach from the
    plate's face to the bar's centre. One function for the compiler (add_machine) and runtime
    variants (machine_variation.MachineLayout.apply), so the two agree exactly."""
    c = cfg.handle_centre() - cfg.door_origin
    bar = cfg.bar_axis
    posts = []
    for s in (-1, 1):
        p = c + bar * s * (cfg.handle_length / 2 - POST_INSET)
        p[0] += cfg.handle_height / 2
        posts.append(p)
    size = np.array([cfg.handle_height / 2, POST_HALF, POST_HALF])
    return c, np.array(_quat_z_to(bar)), posts, size


def _add_drawer(spec: mujoco.MjSpec, body, name: str, cfg: MachineConfig):
    """The drawer: a tray (floor, side and back rims) with the nest on its floor, the front
    plate that closes the box's front (from just above the bench to the roof's top), and the
    handle bar on the plate; on a slide joint toward the robot, damped, with Coulomb friction, and
    latched like a lid. Its body's origin is the nest's centre on the tray's floor."""
    box = mujoco.mjtGeom.mjGEOM_BOX
    drawer = body.add_body(name=f"{name}_drawer", pos=cfg.door_origin.tolist())
    drawer.add_joint(name=f"{name}_drawer", type=mujoco.mjtJoint.mjJNT_SLIDE,
                     axis=cfg.slide_axis.tolist(), range=[0.0, cfg.travel],
                     limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=cfg.lid_damping,
                     frictionloss=cfg.lid_friction, stiffness=cfg.lid_spring, springref=0.0,
                     armature=0.02)
    f, w, r = cfg.drawer_front, cfg.drawer_width / 2, DRAWER_RIM
    tray = {"rgba": [0.55, 0.6, 0.66, 1.0], "solref": [0.01, 1.0]}
    drawer.add_geom(name=f"{name}_drawer_floor", type=box, size=[f + r / 2, w, 0.004],
                    pos=[r / 2, 0.0, -0.004], mass=0.6 * cfg.lid_mass, **tray)
    lo, hi = 0.002 - cfg.drawer_floor, cfg.height - cfg.drawer_floor
    drawer.add_geom(name=f"{name}_drawer_face", type=box,
                    size=[DRAWER_PLATE / 2, w + 0.012, (hi - lo) / 2],
                    pos=[-f - DRAWER_PLATE / 2, 0.0, (hi + lo) / 2], mass=0.4 * cfg.lid_mass,
                    **tray)
    for side, sign in (("left", 1), ("right", -1)):
        drawer.add_geom(name=f"{name}_drawer_rim_{side}", type=box,
                        size=[f + r / 2, r / 2, cfg.drawer_rim / 2],
                        pos=[r / 2, sign * (w - r / 2), cfg.drawer_rim / 2], mass=0.01, **tray)
    drawer.add_geom(name=f"{name}_drawer_rim_back", type=box,
                    size=[r / 2, w - r, cfg.drawer_rim / 2],
                    pos=[f + r / 2, 0.0, cfg.drawer_rim / 2], mass=0.01, **tray)
    # the handle: a bar on two posts standing off the plate
    c, q_bar, posts, size = drawer_handle_layout(cfg)
    dark = [0.12, 0.12, 0.13, 1.0]
    drawer.add_geom(name=f"{name}_handle", type=box,
                    size=[cfg.handle_half_height, cfg.handle_radius, cfg.handle_length / 2],
                    pos=c.tolist(), quat=q_bar.tolist(), rgba=dark, mass=0.05,
                    friction=[1.0, 0.03, 0.003])
    for k, p in enumerate(posts):
        drawer.add_geom(name=f"{name}_handle_post{k}", type=box, size=size.tolist(),
                        pos=p.tolist(), quat=[1.0, 0.0, 0.0, 0.0], rgba=dark, mass=0.01)
    drawer.add_site(name=f"{name}_handle", pos=c.tolist(), size=[0.004] * 3, group=4)
    spec.add_equality(name=f"{name}_latch", type=mujoco.mjtEq.mjEQ_JOINT,
                      objtype=mujoco.mjtObj.mjOBJ_JOINT, name1=f"{name}_drawer", active=False,
                      data=[0.0] * 11)
    # the index pin: holds the drawer at its open stop while the robot loads it (off unless the
    # cell engages it: Machine.index_open), as stiff as the nest's own contacts (a millimetre or
    # two of give under 40 N; the latch's default setting gives centimetres)
    spec.add_equality(name=f"{name}_index", type=mujoco.mjtEq.mjEQ_JOINT,
                      objtype=mujoco.mjtObj.mjOBJ_JOINT, name1=f"{name}_drawer", active=False,
                      data=[cfg.travel] + [0.0] * 10, **contact("stiff"))
    return drawer


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


def _add_drawer_dressing(body, name: str, cfg: MachineConfig) -> None:
    """A drawer tester's visual parts: the control panel and screen on its side, a hazard trim
    along the roof's front edge, the status lamp at the back corner, and the test head (named
    spindle and tool, as the lid machine's, so the runtime drives it the same way)."""
    box, cyl = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER
    x0 = cfg.nest_inset - cfg.drawer_front
    hx, hy, top = cfg.depth / 2, cfg.width / 2, cfg.height
    xp = x0 + hx - hx / 3
    _visual(body, name=f"{name}_panel", type=box, size=[0.05, 0.006, 0.045],
            pos=[xp, hy + 0.006, 0.1], rgba=[0.22, 0.24, 0.27, 1.0])
    _visual(body, name=f"{name}_screen", type=box, size=[0.032, 0.001, 0.022],
            pos=[xp, hy + 0.0125, 0.115], rgba=[0.43, 0.83, 1.0, 1.0])
    _visual(body, name=f"{name}_trim_f", type=box, size=[0.004, hy - 0.01, 0.0006],
            pos=[x0 + 0.006, 0.0, top + 0.0006], rgba=[0.95, 0.76, 0.0, 1.0])
    lx, ly = x0 + cfg.depth - 0.03, hy - 0.03
    _visual(body, name=f"{name}_lamp_base", type=cyl, size=[0.012, 0.004],
            pos=[lx, ly, top + 0.004], rgba=[0.12, 0.12, 0.13, 1.0])
    for k, lamp in enumerate(LAMPS):
        _visual(body, name=f"{name}_lamp_{lamp}", type=cyl, size=[0.01, 0.0085],
                pos=[lx, ly, top + 0.0175 + 0.018 * k], rgba=list(_LAMP_OFF[lamp]))
    half, drop = _spindle_size(cfg)
    _visual(body, name=f"{name}_spindle", type=cyl, size=[0.016, half],
            pos=list(_spindle_pos(cfg, 0.0)), rgba=[0.84, 0.86, 0.88, 1.0])
    _visual(body, name=f"{name}_tool", type=cyl, size=[0.004, 0.012],
            pos=list(_spindle_pos(cfg, 0.0) - [0, 0, drop]), rgba=[0.3, 0.3, 0.32, 1.0])


def _spindle_size(cfg: MachineConfig) -> tuple[float, float]:
    """(the spindle's half height, its centre to the tool's centre): a shorter head in a drawer
    tester, whose nest stands on the tray's floor, so it stays under the roof."""
    return (0.028, 0.040) if cfg.kind == "drawer" else (0.035, 0.047)


def _spindle_pos(cfg: MachineConfig, working: float) -> np.ndarray:
    """The spindle's centre: parked under the roof beside the opening (working = 0), or over the
    part with its tool tip 5 mm above a 6 cm part (working = 1). In a drawer tester: beside and
    over the shut drawer's nest."""
    if cfg.kind == "drawer":
        z = cfg.drawer_floor + _PART_HEIGHT + 0.005 + _spindle_size(cfg)[1] + 0.012
        parked = np.array([cfg.nest_inset, cfg.drawer_width / 2 + 0.035, z])
        over = np.array([cfg.nest_inset, 0.0, z])
        return parked + (over - parked) * working
    z = 0.06 + 0.005 + 0.059                  # tool tip = centre - 0.059 (spindle and tool)
    parked = np.array([0.0, cfg.opening[1] / 2 + 0.035, z])
    over = np.array([0.0, 0.0, z])
    return parked + (over - parked) * working


class Machine:
    """The runtime side of a machine: its lid (or drawer), latch, interlock, cycle and lamp. Call
    `update` once per control step (e.g. from the cell controller's on_step).

    The moving part is the "door" whatever the kind: `door_qpos`, `door_dof`, `door_body`.
    v4.0's names `lid_qpos`, `lid_dof` and `lid` (the opening) stay, for every kind, so code
    written for lids drives a drawer's slide too. The pocket's frame is `nest_body`: the machine's
    body for a lid, the drawer's for a drawer (its nest moves with it)."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, name: str,
                 cfg: MachineConfig, parts: list[CubeIndices]) -> None:
        self.m, self.d, self.name, self.cfg, self.parts = model, data, name, cfg, parts
        door = DOOR_NAME[cfg.kind].format(name)
        j = model.joint(door)
        self.door_qpos, self.door_dof = int(j.qposadr[0]), int(j.dofadr[0])
        self.lid_qpos, self.lid_dof = self.door_qpos, self.door_dof
        self.door_body = model.body(door).id
        self.latch_eq = model.equality(f"{name}_latch").id
        self.index_eq = model.equality(f"{name}_index").id if cfg.kind == "drawer" else -1
        self.body = model.body(name).id
        self.nest_body = self.door_body if cfg.kind == "drawer" else self.body
        self.pocket_site = model.site(f"{name}_pocket").id
        self.handle_site = (model.site(f"{name}_handle").id if cfg.kind != "pneumatic" else -1)
        self.cylinder = (model.actuator(f"{name}_cylinder").id if cfg.kind == "pneumatic"
                         else -1)
        self.lamps = {k: model.geom(f"{name}_lamp_{k}").id for k in LAMPS}
        self.spindle = model.geom(f"{name}_spindle").id
        self.tool = model.geom(f"{name}_tool").id
        self._tool_drop = _spindle_size(cfg)[1]
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
        if self.index_eq >= 0:
            self.d.eq_active[self.index_eq] = 0
        if self.cylinder >= 0:        # the cylinder holds the lid where it is, until signalled
            self.d.ctrl[self.cylinder] = 0.0
            self.d.act[self.m.actuator_actadr[self.cylinder]] = float(
                np.clip(self.lid, 0.0, self.cfg.travel))
        self._hold_open()
        self._show()

    @property
    def lid(self) -> float:
        """How far the lid (or drawer) is open (m; 0 = closed)."""
        return float(self.d.qpos[self.lid_qpos])

    @property
    def opening(self) -> float:
        """How far the door (lid or drawer) is open (m; 0 = closed): `lid` by its general name."""
        return self.lid

    # --- the pneumatic lid's signals (kind "pneumatic") ----------------------------------------
    def request_open(self) -> bool:
        """Open a pneumatic lid: its cylinder runs it to the open stop at pneumatic_speed. Refused
        (False) while a cycle runs: the interlock keeps the lid locked. A latched lid is released
        first."""
        self._need_cylinder()
        if self.state == RUNNING:
            return False
        if self.latched:
            self.unlatch()
        self.d.ctrl[self.cylinder] = self.cfg.pneumatic_speed
        return True

    def request_close(self) -> bool:
        """Close a pneumatic lid: its cylinder runs it shut at pneumatic_speed, fast enough for
        the latch to catch (as a robot's firm last push), and the cycle starts if a part is
        seated."""
        self._need_cylinder()
        self.d.ctrl[self.cylinder] = -self.cfg.pneumatic_speed
        return True

    @property
    def cylinder_target(self) -> float | None:
        """Where a pneumatic lid's cylinder is driving it (m), or None without a cylinder."""
        if self.cylinder < 0:
            return None
        return float(self.d.act[self.m.actuator_actadr[self.cylinder]])

    def _need_cylinder(self) -> None:
        if self.cylinder < 0:
            raise ValueError(f"{self.name} is a {self.cfg.kind!r} machine: it has no cylinder; "
                             "the robot opens it")

    @property
    def latched(self) -> bool:
        return bool(self.d.eq_active[self.latch_eq])

    # --- the drawer's index pin (kind "drawer") -----------------------------------------------
    def index_open(self, catch: float = 0.01) -> bool:
        """Engage a drawer's index pin: it pulls the drawer onto its open stop and holds it there,
        so the nest stays put while the robot loads or unloads it (a free slide gives way to any
        push on the nest's chamfer: a part pressed into an open drawer shoves it shut). The cell
        signals it once the drawer is open, as it signals a pneumatic tester. Catches only a
        drawer within `catch` (m) of its stop; returns whether it is engaged. (Not used by the
        door env: there a drawer slides freely.)"""
        if self.index_eq < 0:
            raise ValueError(f"{self.name} is a {self.cfg.kind!r} machine: it has no index pin")
        if self.lid < self.cfg.travel - catch:
            return False
        self.m.eq_data[self.index_eq, 0] = self.cfg.travel
        self.d.eq_active[self.index_eq] = 1
        return True

    def release_index(self) -> None:
        """Withdraw a drawer's index pin (before it is pushed shut). No-op on other kinds."""
        if self.index_eq >= 0:
            self.d.eq_active[self.index_eq] = 0

    @property
    def indexed(self) -> bool:
        """A drawer held at its open stop by its index pin."""
        return self.index_eq >= 0 and bool(self.d.eq_active[self.index_eq])

    @property
    def held_open(self) -> bool:
        """A springy lid sits in its hold-open detent. (A lid without a spring stays wherever it
        is left, so it needs none.)"""
        return self.cfg.lid_spring > 0 and self.lid >= self.cfg.travel - self.cfg.detent_depth

    def _hold_open(self) -> None:
        """The detent of a springy lid: in the last `detent_depth` of its travel it cancels the
        spring and holds the lid against its stop with `detent_force`. Pushed out of it, the
        spring pulls the lid shut again. Applied as a joint force, renewed every control step."""
        self.d.qfrc_applied[self.lid_dof] = (self.cfg.lid_spring * self.lid + self.cfg.detent_force
                                             if self.held_open else 0.0)

    def unlatch(self) -> None:
        """Release the lid. It stays closed until something pulls it open, and the latch
        catches again only when it is pushed shut after being opened."""
        self.d.eq_active[self.latch_eq] = 0
        self._armed, self._peak_closing = False, 0.0

    def handle_xyz(self) -> np.ndarray:
        if self.handle_site < 0:
            raise ValueError(f"{self.name} is a pneumatic machine: it has no handle")
        return self.d.site_xpos[self.handle_site].copy()

    def pocket_xyz(self) -> np.ndarray:
        return self.d.site_xpos[self.pocket_site].copy()

    def in_pocket(self, part: CubeIndices) -> tuple[bool, bool]:
        """(the part is over the pocket and low, it is seated): seated means on the pocket's
        floor (gap under 1.5 mm), upright (tilt under 3 deg) and inside the opening; in a keyed
        pocket, a keyed part must also be the right way round (its rib in the slot: see
        `key_error`)."""
        d = self.d
        R_m = d.xmat[self.nest_body].reshape(3, 3)
        rel = R_m.T @ (d.xpos[part.body] - d.xpos[self.nest_body])
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
        R_m = self.d.xmat[self.nest_body].reshape(3, 3)
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
        self._hold_open()
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
                if self.cylinder >= 0:
                    self.request_open()   # a pneumatic tester opens itself when its test ends
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
        self.m.geom_pos[self.tool] = p - [0.0, 0.0, self._tool_drop]
