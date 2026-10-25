"""Factory-hall dressing for the Milestone 9 workcell: drawn only, never collides.

A modern, clean automotive plant with classic safety markings:
- polished light-grey epoxy floor with a darker cell zone, yellow walkway lines and hazard edges;
- a workstation (aluminium-profile frame, anthracite worktop) carrying the robot and stations;
- black machine-guard mesh around the cell with a striped entry and light-curtain posts;
- a control cabinet with an operator screen and the stack light on top;
- a hall: steel I-beam columns with hazard bases, roof trusses with light strips, and walls
  with a window band.

Textures are generated here (numpy), so nothing is downloaded and nothing is branded. Every
geom has contype = conaffinity = 0: the physics of the cell is identical with or without it.
"""

import mujoco
import numpy as np

FLOOR_Z = -0.8           # m: the worktop (the robot's table plane, z = 0) is 0.8 m above the floor
BENCH_X = (-0.85, 1.0)
BENCH_Y = (-0.95, 0.95)
GUARD = 1.7              # m, half side of the guarded cell (centred on the bench)
HALL = 10.0              # m, half side of the hall


def _texture(spec: mujoco.MjSpec, name: str, image: np.ndarray) -> None:
    """Add a 2-D texture from an (h, w, 3 or 4) uint8 array."""
    h, w, c = image.shape
    tex = spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, width=w, height=h,
                           nchannel=c)
    tex.data = np.ascontiguousarray(image, dtype=np.uint8).tobytes()


def _noise(rng, size: int, scale: int) -> np.ndarray:
    """Smooth value noise in [0, 1] (coarse random grid, bilinear upsampling)."""
    coarse = rng.random((size // scale + 2, size // scale + 2))
    y = np.linspace(0, size / scale, size, endpoint=False)
    y0 = y.astype(int)
    f = y - y0
    rows = coarse[y0] * (1 - f)[:, None] + coarse[y0 + 1] * f[:, None]
    return rows[:, y0] * (1 - f)[None, :] + rows[:, y0 + 1] * f[None, :]


def _textures(spec: mujoco.MjSpec) -> None:
    rng = np.random.default_rng(42)
    n = 512
    # Epoxy floor: mid grey with a soft, large-scale cloudiness (no speckle: it reads as dirt).
    base = 0.47 + 0.03 * _noise(rng, n, 128) + 0.012 * _noise(rng, n, 32)
    floor = np.stack([base, base * 1.01, base * 1.04], axis=-1)
    _texture(spec, "epoxy", np.clip(floor * 255, 0, 255).astype(np.uint8))
    zone = 0.36 + 0.025 * _noise(rng, n, 128) + 0.01 * _noise(rng, n, 32)
    _texture(spec, "epoxy_dark", np.clip(np.stack([zone, zone * 1.01, zone * 1.05], -1) * 255,
                                         0, 255).astype(np.uint8))
    # Anthracite worktop.
    top = 0.2 + 0.02 * _noise(rng, n, 32) + 0.01 * rng.random((n, n))
    _texture(spec, "worktop", np.clip(np.stack([top, top, top * 1.05], -1) * 255, 0, 255
                                      ).astype(np.uint8))
    # Hazard stripes: yellow and black at 45 degrees.
    m = 128
    i, j = np.mgrid[0:m, 0:m]
    stripe = ((i + j) // (m // 4)) % 2 == 0
    hazard = np.where(stripe[..., None], [240, 190, 20], [25, 25, 25]).astype(np.uint8)
    _texture(spec, "hazard_stripes", hazard)
    # Wall panels: light grey with vertical seams.
    wall = np.full((n, n), 0.82) - 0.06 * ((np.arange(n) % 128) < 3)[None, :]
    wall = wall + 0.015 * _noise(rng, n, 64)
    _texture(spec, "wall", np.clip(np.stack([wall, wall, wall * 1.02], -1) * 255, 0, 255
                                   ).astype(np.uint8))


def _materials(spec: mujoco.MjSpec) -> None:
    def mat(name, texture=None, rgba=(1, 1, 1, 1), repeat=(1, 1), **kw):
        m = spec.add_material(name=name, rgba=list(rgba), texrepeat=list(repeat), **kw)
        if texture:
            m.textures = ["", texture] + [""] * 8  # slot 1: the RGB texture role
            m.texuniform = True
        return m

    # Reflectance stays 0 everywhere: MuJoCo's planar reflections streak on curved geoms.
    mat("epoxy", "epoxy", repeat=(0.08, 0.08), shininess=0.7, specular=0.35)
    mat("epoxy_dark", "epoxy_dark", repeat=(0.15, 0.15), shininess=0.6, specular=0.3)
    mat("worktop", "worktop", repeat=(2, 2), shininess=0.3, specular=0.2)
    mat("hazard_stripes", "hazard_stripes", repeat=(6, 6))
    mat("guard_mesh", rgba=(0.1, 0.1, 0.12, 0.3), shininess=0.2)   # see-through guard panel
    mat("wall", "wall", repeat=(0.15, 0.15))
    mat("sign_panel", rgba=(0.12, 0.13, 0.15, 1), shininess=0.3)
    mat("aluminium", rgba=(0.78, 0.8, 0.83, 1), shininess=0.8, specular=0.7)
    mat("steel_grey", rgba=(0.6, 0.62, 0.65, 1), shininess=0.4, specular=0.3)
    mat("guard_black", rgba=(0.12, 0.12, 0.13, 1), shininess=0.3)
    mat("safety_yellow", rgba=(0.95, 0.76, 0.08, 1), shininess=0.4)
    mat("cabinet", rgba=(0.82, 0.83, 0.85, 1), shininess=0.5, specular=0.3)
    mat("screen", rgba=(0.1, 0.35, 0.6, 1), emission=0.7)
    mat("light_strip", rgba=(1.0, 0.98, 0.92, 1), emission=1.0)
    mat("window", rgba=(0.62, 0.7, 0.78, 1), emission=0.25)
    mat("rubber", rgba=(0.06, 0.06, 0.07, 1))
    mat("estop_red", rgba=(0.75, 0.05, 0.05, 1), shininess=0.6)


def _box(body, pos, size, material, name=None, quat=None):
    kw = {"name": name} if name else {}
    if quat is not None:
        kw["quat"] = quat
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(pos), size=list(size),
                  material=material, contype=0, conaffinity=0, group=0, **kw)


def _cyl(body, pos, radius, half, material, axis=None):
    kw = {}
    if axis == "x":
        kw["quat"] = [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]
    elif axis == "y":
        kw["quat"] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=list(pos), size=[radius, half],
                  material=material, contype=0, conaffinity=0, **kw)


def _floor(world) -> None:
    cx, cy = sum(BENCH_X) / 2, sum(BENCH_Y) / 2
    world.add_geom(name="factory_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[HALL, HALL, 0.05],
                   pos=[0, 0, FLOOR_Z], material="epoxy", contype=0, conaffinity=0)
    _box(world, (cx, cy, FLOOR_Z + 0.0005), (GUARD, GUARD, 0.0005), "epoxy_dark", "cell_zone")
    z = FLOOR_Z + 0.0012
    # Hazard edge just inside the guarding, and two yellow walkway lines around the cell.
    edge = GUARD - 0.06
    for k, (pos, size) in enumerate(((((cx, cy + edge, z)), (edge, 0.05, 0.0006)),
                                     (((cx, cy - edge, z)), (edge, 0.05, 0.0006)),
                                     (((cx + edge, cy, z)), (0.05, edge, 0.0006)),
                                     (((cx - edge, cy, z)), (0.05, edge, 0.0006)))):
        _box(world, pos, size, "hazard_stripes", f"hazard_{k}")
    for r in (GUARD + 0.9, GUARD + 2.3):
        for pos, size in (((cx, cy + r, z), (r, 0.05, 0.0006)),
                          ((cx, cy - r, z), (r, 0.05, 0.0006)),
                          ((cx + r, cy, z), (0.05, r, 0.0006)),
                          ((cx - r, cy, z), (0.05, r, 0.0006))):
            _box(world, pos, size, "safety_yellow")


def _workstation(world) -> None:
    cx, cy = sum(BENCH_X) / 2, sum(BENCH_Y) / 2
    hx, hy = (BENCH_X[1] - BENCH_X[0]) / 2, (BENCH_Y[1] - BENCH_Y[0]) / 2
    _box(world, (cx, cy, -0.0006 - 0.025), (hx, hy, 0.025), "worktop", "worktop")
    _box(world, (cx, cy, -0.06), (hx - 0.02, hy - 0.02, 0.012), "aluminium")       # frame rail
    leg_top, leg_bottom = -0.075, FLOOR_Z + 0.03  # under the worktop and frame rail
    for sx in (-1, 1):
        for sy in (-1, 1):
            _box(world, (cx + sx * (hx - 0.05), cy + sy * (hy - 0.05), (leg_top + leg_bottom) / 2),
                 (0.03, 0.03, (leg_top - leg_bottom) / 2), "aluminium")
            _box(world, (cx + sx * (hx - 0.05), cy + sy * (hy - 0.05), FLOOR_Z + 0.015),
                 (0.05, 0.05, 0.015), "guard_black")                                 # feet
    _box(world, (cx, cy, FLOOR_Z + 0.2), (hx - 0.06, hy - 0.06, 0.012), "aluminium")  # shelf


def _guarding(world) -> None:
    """Mesh panels on black posts around the cell; the front (+x) has an open entry with
    light-curtain posts and a striped kick plate."""
    cx, cy = sum(BENCH_X) / 2, sum(BENCH_Y) / 2
    h = 1.3
    entry = 0.3  # half width of the entry opening on the +x side (between the middle posts)
    posts = np.linspace(-GUARD, GUARD, 6)
    for side in ("+y", "-y", "-x", "+x"):
        for p in posts:
            if side in ("+y", "-y"):
                x, y = cx + p, cy + (GUARD if side == "+y" else -GUARD)
            else:
                x, y = cx + (GUARD if side == "+x" else -GUARD), cy + p
            if side == "+x" and abs(p) < entry:
                continue
            _box(world, (x, y, FLOOR_Z + h / 2 + 0.05), (0.025, 0.025, h / 2 + 0.05), "guard_black")
        for a, b in zip(posts[:-1], posts[1:], strict=True):
            mid, half = (a + b) / 2, (b - a) / 2 - 0.03
            if side == "+x" and abs(mid) < entry:
                continue
            panel_z = FLOOR_Z + 0.1 + h / 2
            if side in ("+y", "-y"):
                y = cy + (GUARD if side == "+y" else -GUARD)
                _box(world, (cx + mid, y, panel_z), (half, 0.004, h / 2), "guard_mesh")
                _box(world, (cx + mid, y, FLOOR_Z + 0.06), (half, 0.006, 0.05), "hazard_stripes")
                for rail_z in (FLOOR_Z + 0.1 + h, FLOOR_Z + 0.1 + h / 2):  # top and mid rails
                    _box(world, (cx + mid, y, rail_z), (half + 0.03, 0.015, 0.015), "guard_black")
            else:
                x = cx + (GUARD if side == "+x" else -GUARD)
                _box(world, (x, cy + mid, panel_z), (0.004, half, h / 2), "guard_mesh")
                _box(world, (x, cy + mid, FLOOR_Z + 0.06), (0.006, half, 0.05), "hazard_stripes")
                for rail_z in (FLOOR_Z + 0.1 + h, FLOOR_Z + 0.1 + h / 2):
                    _box(world, (x, cy + mid, rail_z), (0.015, half + 0.03, 0.015), "guard_black")
    for s in (-1, 1):  # light-curtain posts at the entry
        x, y = cx + GUARD, cy + s * (entry - 0.04)
        _box(world, (x, y, FLOOR_Z + 0.75), (0.03, 0.03, 0.75), "safety_yellow")
        _box(world, (x - 0.031, y, FLOOR_Z + 0.75), (0.002, 0.012, 0.6), "rubber")
    _box(world, (cx + GUARD, cy, FLOOR_Z + 0.0015), (0.08, entry, 0.0008), "hazard_stripes")


def add_control_cabinet(world, xy) -> tuple[float, float, float]:
    """Grey cabinet with an operator screen; returns the top centre (for the stack light)."""
    x, y = xy
    h = 1.75
    _box(world, (x, y, FLOOR_Z + h / 2), (0.3, 0.25, h / 2), "cabinet", "control_cabinet")
    _box(world, (x - 0.302, y, FLOOR_Z + 1.25), (0.002, 0.16, 0.11), "screen", "hmi_screen")
    _box(world, (x - 0.303, y, FLOOR_Z + 1.25), (0.001, 0.18, 0.13), "guard_black")
    _cyl(world, (x - 0.305, y + 0.12, FLOOR_Z + 1.02), 0.025, 0.01, "estop_red", axis="x")
    _box(world, (x - 0.302, y, FLOOR_Z + 0.45), (0.002, 0.2, 0.3), "steel_grey")       # door
    return (x, y, FLOOR_Z + h)


def _hall(world) -> None:
    top = FLOOR_Z + 7.0
    for x in (-HALL, HALL):
        _box(world, (x, 0, FLOOR_Z + 3.5), (0.1, HALL, 3.5), "wall")
        _box(world, (x * 0.99, 0, FLOOR_Z + 5.0), (0.02, HALL * 0.9, 0.6), "window")
    for y in (-HALL, HALL):
        _box(world, (0, y, FLOOR_Z + 3.5), (HALL, 0.1, 3.5), "wall")
        _box(world, (0, y * 0.99, FLOOR_Z + 5.0), (HALL * 0.9, 0.02, 0.6), "window")
    for cx in (-7.5, 7.5):  # steel I-beam columns with hazard-wrapped bases
        for cy in (-6.0, 0.0, 6.0):
            _box(world, (cx, cy, FLOOR_Z + 3.5), (0.15, 0.012, 3.5), "steel_grey")
            for s in (-1, 1):
                _box(world, (cx, cy + s * 0.15, FLOOR_Z + 3.5), (0.15, 0.012, 3.5), "steel_grey")
            _box(world, (cx, cy, FLOOR_Z + 0.6), (0.17, 0.17, 0.6), "hazard_stripes")
    for y in np.linspace(-HALL + 2, HALL - 2, 5):  # roof trusses with light strips
        _box(world, (0, y, top), (HALL, 0.08, 0.25), "steel_grey")
        for x in np.linspace(-6, 6, 4):
            _box(world, (x, y + 0.6, top - 0.5), (1.2, 0.08, 0.02), "light_strip")
            _box(world, (x, y + 0.6, top - 0.47), (1.25, 0.1, 0.02), "steel_grey")


def add_factory(spec: mujoco.MjSpec) -> None:
    """Add the factory hall around the cell (textures, materials, geoms, lights). The table
    plane (Menagerie's floor geom) becomes the worktop surface."""
    _textures(spec)
    _materials(spec)
    for tex in spec.textures:  # the sky above the open hall: a soft grey "ceiling"
        if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            tex.rgb1, tex.rgb2 = [0.62, 0.65, 0.7], [0.3, 0.32, 0.35]
    table = spec.geom("floor")
    table.material = "worktop"
    world = spec.worldbody
    _floor(world)
    _workstation(world)
    _guarding(world)
    _hall(world)
    # Lighting: soft overhead key light with shadows, plus fills from the roof.
    spot = mujoco.mjtLightType.mjLIGHT_SPOT
    directional = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    world.add_light(name="key", type=spot, pos=[1.5, -1.5, 3.5], dir=[-0.4, 0.4, -1.0],
                    cutoff=55, exponent=2, diffuse=[0.55, 0.55, 0.53], specular=[0.2, 0.2, 0.2],
                    castshadow=True)
    world.add_light(name="fill", type=directional, pos=[0, 0, 5], dir=[0.3, 0.2, -1.0],
                    diffuse=[0.22, 0.22, 0.23], specular=[0.03, 0.03, 0.03], castshadow=False)
    for light in spec.lights:  # Menagerie's own overhead light: softer, no shadow
        if light.name not in ("key", "fill"):
            light.diffuse = [0.25, 0.25, 0.25]
            light.castshadow = False
    spec.visual.headlight.diffuse = [0.2, 0.2, 0.2]
    spec.visual.headlight.ambient = [0.22, 0.22, 0.24]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.quality.shadowsize = 4096
    spec.visual.rgba.haze = [0.75, 0.78, 0.82, 1.0]
