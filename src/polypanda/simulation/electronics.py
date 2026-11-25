"""v4.1: an electronics-factory look for the tending line (visual only; docs/v4_1_plan.md, sec. 5).

Applied by `restyle_electronics` to a line spec after the factory hall, the line dressing, the
rail and the machines are added. Nothing here collides or has mass, and nothing is added to a
moving body, so the physics of the line is unchanged. It:

- re-textures the hall: a light ESD vinyl floor with fine conductive flecks, a pale blue cell
  zone, green ESD-area boundary tape, white walls and steel, a blue ESD mat on the workbench,
  clear polycarbonate guarding on light aluminium;
- brightens and cools the light (ceiling LED panels, a cooler key light, more ambient);
- restyles the line's machines as test stations (brushed metal, a blue gasket line);
- adds an SMT line behind the cell (magazine loader, printer, paste inspection, two
  pick-and-place machines, a reflow oven, optical inspection, an OK/NG unloader, each with a
  light tower, boards on the conveyor), a smart reel tower and dry cabinets, a burn-in rack with
  unit LEDs, a red quality zone with a rework bench and fail bin, ESD signs, overhead ionizers
  over the bench, and a test dashboard (andon) screen.
"""

from __future__ import annotations

import mujoco
import numpy as np

from polypanda.simulation.factory import BENCH_X, BENCH_Y, FLOOR_Z, GUARD
from polypanda.simulation.line_dressing import _text_texture

_BOX, _CYL = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER
BEIGE = [0.86, 0.85, 0.8, 1.0]
DARK = [0.2, 0.21, 0.23, 1.0]
ALU = [0.8, 0.82, 0.85, 1.0]
WHITE = [0.93, 0.94, 0.95, 1.0]
ESD_BLUE = [0.16, 0.42, 0.7, 1.0]
GLASS = [0.55, 0.68, 0.8, 0.35]
GREEN, AMBER, RED = [0.1, 0.85, 0.3, 1.0], [1.0, 0.65, 0.05, 1.0], [0.9, 0.1, 0.08, 1.0]
PCB = [0.05, 0.42, 0.2, 1.0]


def _g(body, size, pos, rgba, gtype=_BOX, quat=None, material=None, name=None):
    kw = {}
    if quat is not None:
        kw["quat"] = quat
    if material is not None:
        kw["material"] = material
    if name is not None:
        kw["name"] = name
    return body.add_geom(type=gtype, size=list(size), pos=list(pos), rgba=list(rgba),
                         contype=0, conaffinity=0, mass=0.0, **kw)


def _floor_box(world, x, y, sx, sy, h, rgba, **kw):
    """A box standing on the hall floor: centre (x, y), half sizes sx, sy, height h."""
    return _g(world, (sx, sy, h / 2), (x, y, FLOOR_Z + h / 2), rgba, **kw)


def _texture(spec, name: str, image: np.ndarray) -> str:
    h, w, c = image.shape
    tex = spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, width=w, height=h,
                           nchannel=c)
    tex.data = np.ascontiguousarray(image, dtype=np.uint8).tobytes()
    return name


def _retexture(spec, material: str, texture: str, rgba=(1, 1, 1, 1), repeat=None, **kw) -> None:
    m = spec.material(material)
    if m is None:
        return
    m.textures = ["", texture] + [""] * 8
    m.rgba = list(rgba)
    if repeat is not None:
        m.texrepeat = list(repeat)
    for k, v in kw.items():
        setattr(m, k, v)


def _recolour(spec, material: str, rgba, **kw) -> None:
    m = spec.material(material)
    if m is None:
        return
    m.textures = [""] * 10
    m.rgba = list(rgba)
    for k, v in kw.items():
        setattr(m, k, v)


def _hall(spec) -> None:
    rng = np.random.default_rng(20261124)
    n = 512
    # ESD vinyl: light grey, faint cloudiness, fine dark conductive flecks
    base = 0.8 + 0.015 * rng.standard_normal((n, n)).cumsum(0).cumsum(1) / n ** 1.2
    base = np.clip(base, 0.74, 0.86)
    flecks = rng.random((n, n)) < 0.012
    floor = np.where(flecks, 0.45, base)
    _texture(spec, "esd_floor", (np.stack([floor, floor * 1.0, floor * 1.02], -1) * 255
                                 ).clip(0, 255).astype(np.uint8))
    zone = np.full((n, n), 0.78)
    zone = np.where(rng.random((n, n)) < 0.01, 0.5, zone)
    _texture(spec, "esd_zone", (np.stack([zone * 0.9, zone * 0.97, zone * 1.06], -1) * 255
                                ).clip(0, 255).astype(np.uint8))
    # green ESD boundary tape with a light centre stripe
    m = 64
    tape = np.zeros((m, m, 3), np.uint8)
    tape[:] = [40, 150, 70]
    tape[m // 2 - 4:m // 2 + 4] = [235, 240, 235]
    _texture(spec, "esd_tape", tape)
    # a muted blue-grey ESD mat with a faint grain
    mat = 0.5 + 0.015 * rng.standard_normal((n, n))
    _texture(spec, "esd_mat", (np.stack([mat * 0.62, mat * 0.86, mat * 1.08], -1) * 255
                               ).clip(0, 255).astype(np.uint8))
    wall = np.full((n, n), 0.95) - 0.03 * ((np.arange(n) % 128) < 2)[None, :]
    _texture(spec, "white_wall", (np.stack([wall, wall, wall * 1.01], -1) * 255
                                  ).clip(0, 255).astype(np.uint8))
    _retexture(spec, "epoxy", "esd_floor", repeat=(10, 10), reflectance=0.08)
    _retexture(spec, "epoxy_dark", "esd_zone", repeat=(6, 6), reflectance=0.06)
    _retexture(spec, "hazard_stripes", "esd_tape", repeat=(30, 1))
    _retexture(spec, "worktop", "esd_mat", repeat=(4, 4), reflectance=0.0, shininess=0.1,
               specular=0.05)
    _retexture(spec, "wall", "white_wall", repeat=(0.15, 0.15))
    _recolour(spec, "guard_black", ALU, shininess=0.7, specular=0.6)
    _recolour(spec, "guard_mesh", [0.82, 0.9, 0.97, 0.14], shininess=0.9, specular=0.8)
    _recolour(spec, "steel_grey", [0.9, 0.91, 0.93, 1.0], shininess=0.4)
    _recolour(spec, "cabinet", WHITE, shininess=0.5)
    _recolour(spec, "light_strip", [1.0, 1.0, 1.0, 1.0], emission=1.0)
    _recolour(spec, "window", [0.75, 0.82, 0.9, 1.0], emission=0.35)
    for tex in spec.textures:  # the sky above the open hall: a bright, cool ceiling
        if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            tex.rgb1, tex.rgb2 = [0.9, 0.92, 0.95], [0.7, 0.73, 0.78]
    world = spec.worldbody
    # a clean hall: hide the automotive hall's steel I-beam columns and their striped bases
    # (tall geoms with those materials; floor tape and roof trusses stay); group 4 is not drawn
    for g in world.geoms:
        if g.material in ("steel_grey", "hazard_stripes") and g.size[2] >= 0.5:
            g.group = 4
    top = FLOOR_Z + 4.2                      # a lower, cleanroom-like ceiling of LED panels
    for x in np.linspace(-7.5, 7.5, 7):
        for y in np.linspace(-7.5, 7.5, 7):
            _g(world, (0.6, 0.3, 0.01), (x, y, top), [1, 1, 1, 1],
               material="light_strip")


def _lights(spec) -> None:
    for light in spec.lights:
        if light.name == "key":
            light.diffuse = [0.6, 0.63, 0.68]
            light.specular = [0.15, 0.15, 0.16]
        elif light.name == "fill":
            light.diffuse = [0.38, 0.4, 0.43]
    spec.worldbody.add_light(name="cool_fill", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             pos=[0, 0, 5], dir=[-0.3, -0.4, -1.0], diffuse=[0.25, 0.27, 0.3],
                             specular=[0.02, 0.02, 0.02], castshadow=False)
    spec.visual.headlight.ambient = [0.33, 0.34, 0.36]
    spec.visual.headlight.diffuse = [0.18, 0.18, 0.19]
    spec.visual.rgba.haze = [0.9, 0.92, 0.95, 1.0]


def _testers(spec, names) -> None:
    """The line's machines restyled as test stations: brushed-metal walls, a dark roof with a
    blue gasket line, and a test-status screen."""
    for name in names:
        for g in spec.geoms:
            if not g.name or not g.name.startswith(f"{name}_"):
                continue
            if "roof" in g.name:
                g.rgba = [0.42, 0.45, 0.5, 1.0]
            elif any(k in g.name for k in ("wall", "side", "front", "back", "base")):
                g.rgba = [0.78, 0.8, 0.83, 1.0]
        lid = spec.body(f"{name}_lid")
        if lid is not None:
            for g in lid.geoms:
                if g.name and "handle" in g.name:
                    continue
                g.rgba = [0.72, 0.75, 0.79, 1.0]


def _light_tower(body, x, y, z, states=(1, 0, 0)) -> None:
    """A stack light: green, amber, red from the bottom (lit = 1)."""
    _g(body, (0.012, 0.06), (x, y, z + 0.06), DARK, _CYL)
    for k, (colour, lit) in enumerate(zip((GREEN, AMBER, RED), states, strict=True)):
        c = colour if lit else [0.25 * v for v in colour[:3]] + [1.0]
        _g(body, (0.028, 0.025), (x, y, z + 0.15 + 0.052 * k), c, _CYL,
           material="light_strip" if lit else None)


def _smt_line(spec) -> None:
    """An SMT line behind the cell, along x at y = Y0, boards flowing toward +x."""
    world = spec.worldbody
    y0, conveyor_z = 3.3, FLOOR_Z + 0.95
    machines = [  # (name, length along x, depth, height, colour, light states)
        ("LOADER", 0.7, 0.9, 1.35, BEIGE, (1, 0, 0)),
        ("PRINTER", 1.25, 1.25, 1.45, BEIGE, (1, 0, 0)),
        ("SPI", 1.0, 1.15, 1.45, BEIGE, (1, 0, 0)),
        ("PNP 1", 1.5, 1.6, 1.55, BEIGE, (1, 0, 0)),
        ("PNP 2", 1.5, 1.6, 1.55, BEIGE, (0, 1, 0)),
        ("REFLOW", 3.2, 1.25, 1.35, [0.74, 0.76, 0.79, 1.0], (1, 0, 0)),
        ("AOI", 1.0, 1.15, 1.5, BEIGE, (1, 0, 0)),
        ("UNLOADER", 0.7, 0.9, 1.35, BEIGE, (1, 0, 0)),
    ]
    gap = 0.35
    total = sum(m[1] for m in machines) + gap * (len(machines) - 1)
    x = -total / 2 + 0.3
    for name, length, depth, height, colour, states in machines:
        cx = x + length / 2
        _floor_box(world, cx, y0, length / 2, depth / 2, 0.25, DARK)            # plinth
        _g(world, (length / 2, depth / 2, (height - 0.25) / 2),
           (cx, y0, FLOOR_Z + 0.25 + (height - 0.25) / 2), colour)
        _g(world, (length / 2 + 0.002, depth / 2 - 0.05, 0.015),
           (cx, y0, FLOOR_Z + 0.9), ESD_BLUE)                                     # accent band
        if name.startswith("PNP"):   # glass top with a gantry head inside
            _g(world, (length / 2 - 0.08, depth / 2 - 0.1, 0.16),
               (cx, y0, FLOOR_Z + height + 0.16), GLASS)
            _g(world, (length / 2 - 0.12, 0.04, 0.03),
               (cx, y0, FLOOR_Z + height + 0.2), DARK)
            _g(world, (0.07, 0.08, 0.07), (cx - 0.2, y0, FLOOR_Z + height + 0.14), ALU)
        elif name == "REFLOW":       # long hood, a red heat-zone display
            _g(world, (length / 2 - 0.05, depth / 2 - 0.05, 0.12),
               (cx, y0, FLOOR_Z + height + 0.12), [0.62, 0.64, 0.67, 1.0])
            for k in range(8):
                _g(world, (0.12, 0.003, 0.02),
                   (cx - length / 2 + 0.3 + k * 0.36, y0 - depth / 2 - 0.002, FLOOR_Z + 1.15),
                   [0.95, 0.25 + 0.04 * k, 0.1, 1.0], material="light_strip")
        else:                        # a dark viewing window on the front
            _g(world, (length / 2 - 0.12, 0.003, 0.18),
               (cx, y0 - depth / 2 - 0.002, FLOOR_Z + 1.15), [0.1, 0.13, 0.17, 0.85])
        label = _text_texture(spec, f"smt_{name.replace(' ', '')}", [name], size=(256, 64),
                              fg=(30, 40, 50), bg=(235, 236, 230))
        _g(world, (0.0006, 0.16, 0.04), (cx, y0 - depth / 2 - 0.004, FLOOR_Z + 1.38),
           [1, 1, 1, 1], material=label, quat=[np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
        _light_tower(world, cx + length / 2 - 0.12, y0 + depth / 2 - 0.12,
                     FLOOR_Z + height + (0.32 if name.startswith("PNP") else 0.0), states)
        if x > -total / 2 + 0.3:     # the board conveyor bridging the gap before this machine
            _g(world, (gap / 2 + 0.05, 0.1, 0.02), (x - gap / 2, y0, conveyor_z), ALU)
            _g(world, (0.06, 0.05, 0.003), (x - gap / 2, y0, conveyor_z + 0.023), PCB)
        x += length + gap


def _storage(spec) -> None:
    world = spec.worldbody
    # a smart reel tower: tall, white, a dark glass front showing reels, a status beacon
    tx, ty = 3.5, -0.9
    _floor_box(world, tx, ty, 0.5, 0.5, 2.2, WHITE)
    _g(world, (0.003, 0.38, 0.8), (tx - 0.503, ty, FLOOR_Z + 1.25), [0.1, 0.12, 0.15, 0.9])
    for row in range(6):
        for col in range(5):
            _g(world, (0.09, 0.012), (tx - 0.49, ty - 0.3 + col * 0.15,
                                      FLOOR_Z + 0.6 + row * 0.25),
               [0.15, 0.16, 0.18, 1.0], _CYL, quat=[np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0])
    _light_tower(world, tx + 0.35, ty + 0.35, FLOOR_Z + 2.2)
    # two dry cabinets: glass doors, a lit (green) interior, shelves of reels
    for k in range(2):
        cx, cy = 3.55, -2.0 - k * 0.75
        _floor_box(world, cx, cy, 0.3, 0.33, 1.85, [0.88, 0.89, 0.9, 1.0])
        _g(world, (0.003, 0.3, 0.8), (cx - 0.303, cy, FLOOR_Z + 0.95), [0.75, 1.0, 0.8, 0.55],
           material="light_strip")
        for s in range(5):
            _g(world, (0.004, 0.29, 0.005), (cx - 0.3, cy, FLOOR_Z + 0.3 + s * 0.32), ALU)
            for r in range(4):
                _g(world, (0.07, 0.01), (cx - 0.25, cy - 0.21 + r * 0.14,
                                         FLOOR_Z + 0.38 + s * 0.32),
                   [0.12, 0.13, 0.15, 1.0], _CYL, quat=[np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0])


def _burn_in(spec) -> None:
    """A burn-in rack: an aluminium frame, shelves of powered units with status LEDs."""
    world = spec.worldbody
    bx, by = -3.4, 1.0
    _floor_box(world, bx, by, 0.04, 0.6, 2.0, ALU)
    _floor_box(world, bx + 0.6, by, 0.04, 0.6, 2.0, ALU)
    rng = np.random.default_rng(7)
    for s in range(6):
        z = FLOOR_Z + 0.25 + s * 0.3
        _g(world, (0.34, 0.6, 0.008), (bx + 0.3, by, z), ALU)
        for u in range(8):
            uy = by - 0.52 + u * 0.15
            _g(world, (0.12, 0.06, 0.04), (bx + 0.25, uy, z + 0.05), [0.12, 0.12, 0.13, 1.0])
            led = GREEN if rng.random() > 0.12 else AMBER
            _g(world, (0.004, 0.008, 0.008), (bx + 0.37, uy, z + 0.06), led,
               material="light_strip")


def _quality_zone(spec) -> None:
    """A red-taped quality zone outside the guarding: a rework bench with a fume-extraction
    arm and a red fail bin."""
    world = spec.worldbody
    qx, qy, sx, sy = 3.3, 1.6, 0.9, 0.7
    z = FLOOR_Z + 0.0014
    tape = (((qx, qy + sy, z), (sx, 0.04, 0.0006)), ((qx, qy - sy, z), (sx, 0.04, 0.0006)),
            ((qx + sx, qy, z), (0.04, sy, 0.0006)), ((qx - sx, qy, z), (0.04, sy, 0.0006)))
    for pos, size in tape:
        _g(world, size, pos, RED)
    _floor_box(world, qx + 0.15, qy + 0.25, 0.6, 0.3, 0.9, [0.85, 0.86, 0.88, 1.0])     # bench
    _g(world, (0.58, 0.28, 0.004), (qx + 0.15, qy + 0.25, FLOOR_Z + 0.904), ESD_BLUE)
    _g(world, (0.015, 0.4), (qx + 0.6, qy + 0.45, FLOOR_Z + 1.3), ALU, _CYL)     # fume arm post
    _g(world, (0.35, 0.018, 0.018), (qx + 0.35, qy + 0.45, FLOOR_Z + 1.68), ALU)
    _g(world, (0.05, 0.06), (qx + 0.05, qy + 0.45, FLOOR_Z + 1.6), DARK, _CYL)
    _floor_box(world, qx - 0.45, qy - 0.3, 0.22, 0.22, 0.6, RED)                       # fail bin
    sign = _text_texture(spec, "quality_sign", ["QUALITY", "NCR / REWORK"], size=(320, 128),
                         fg=(255, 255, 255), bg=(190, 25, 20))
    _g(world, (0.003, 0.3, 0.12), (qx - sx + 0.05, qy, FLOOR_Z + 1.6), [1, 1, 1, 1],
       material=sign)
    _g(world, (0.01, 0.01, 0.75), (qx - sx + 0.07, qy, FLOOR_Z + 0.75), ALU)


def _esd_and_ionizers(spec) -> None:
    world = spec.worldbody
    cx, cy = sum(BENCH_X) / 2, sum(BENCH_Y) / 2
    sign = _text_texture(spec, "esd_sign", ["ATTENTION", "ESD PROTECTED AREA"],
                         size=(384, 160), fg=(20, 20, 20), bg=(250, 205, 30))
    for x, y in ((cx + GUARD + 0.02, cy + 0.6), (cx + GUARD + 0.02, cy - 0.6)):
        _g(world, (0.003, 0.22, 0.1), (x, y, FLOOR_Z + 1.25), [1, 1, 1, 1], material=sign)
    # overhead ionizers with task lights, hung from the ceiling on thin rods, over the testers'
    # side of the bench (out of the robot's way and of the usual camera views)
    ceiling, unit_z = FLOOR_Z + 4.2, FLOOR_Z + 2.4
    for x, y in ((BENCH_X[0] + 0.25, cy - 0.55), (BENCH_X[0] + 0.25, cy + 0.55)):
        _g(world, (0.006, (ceiling - unit_z) / 2), (x, y, (ceiling + unit_z) / 2), ALU, _CYL)
        _g(world, (0.09, 0.2, 0.035), (x, y, unit_z), [0.55, 0.57, 0.6, 1.0])
        for f in (-1, 1):
            _g(world, (0.045, 0.003), (x, y + f * 0.1, unit_z - 0.036), [0.15, 0.15, 0.16, 1.0],
               _CYL)
        _g(world, (0.07, 0.18, 0.003), (x, y, unit_z - 0.038), [1, 1, 1, 1],
           material="light_strip")


def _andon(spec, lines) -> None:
    """A large test-dashboard screen on a stand outside the guarding, facing the cell."""
    world = spec.worldbody
    x, y = sum(BENCH_X) / 2 + GUARD + 0.6, -1.2
    _floor_box(world, x + 0.1, y, 0.03, 0.03, 1.6, ALU)
    _floor_box(world, x + 0.1, y, 0.2, 0.25, 0.02, ALU)
    mat = _text_texture(spec, "andon", lines, size=(768, 448), fg=(220, 240, 235),
                        bg=(12, 26, 38))
    _g(world, (0.02, 0.48, 0.29), (x + 0.07, y, FLOOR_Z + 1.85), DARK)
    _g(world, (0.0008, 0.45, 0.265), (x + 0.049, y, FLOOR_Z + 1.85), [1, 1, 1, 1], material=mat)


ANDON = ["TEST CELL 3   -   SHIFT B", "T1  TESTING  0:42      T2  READY", "T3  PASS",
         "UNITS 128     YIELD 96.1 %", "TAKT 58 s     UPTIME 92 %"]


def restyle_electronics(spec: mujoco.MjSpec, machines=("M1",), andon=ANDON) -> None:
    """Restyle a built line spec as an electronics test cell (visual only)."""
    _hall(spec)
    _lights(spec)
    _testers(spec, machines)
    _smt_line(spec)
    _storage(spec)
    _burn_in(spec)
    _quality_zone(spec)
    _esd_and_ionizers(spec)
    _andon(spec, andon)
