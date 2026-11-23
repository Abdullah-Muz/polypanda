"""v4 line: an industrial look for the conveyor and the recipe board (visual only).

Applied to the v4 tending line only (simulation/line.py); the fixed v1-v3.2 cell keeps its
original look, so its renders and videos stay as they were. Nothing here collides or has mass,
and nothing is added to a moving body, so the physics is unchanged:

- the **conveyor** gets aluminium side frames, a drive motor with its gearbox and an end guard
  at the downstream end, a light barrier with its beam across the belt at the pick point, an
  emergency-stop button, and a status beacon on the feeder tunnel;
- each **recipe board** becomes a monitor on a pole: a black housing, a lit screen, a header
  ("TOWER A - RECIPE"), level numbers, and the recipe as tiles on the screen. The tiles are the
  board's existing squares (`sign{T}_{k}`), restyled, so `Workcell.reset` still lights them in
  the recipe's colours.
"""

from __future__ import annotations

import mujoco
import numpy as np

from polypanda.simulation.workcell import TOWERS, WorkcellConfig

_BOX, _CYL = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER
_BLUE, _BLACK = [0.13, 0.33, 0.72, 1.0], [0.08, 0.08, 0.09, 1.0]
_ALU, _GREY = [0.62, 0.64, 0.67, 1.0], [0.3, 0.31, 0.33, 1.0]


def _vis(body, size, pos, rgba, gtype=_BOX, quat=None, **kw):
    return body.add_geom(type=gtype, size=size, pos=pos, rgba=rgba, contype=0, conaffinity=0,
                         mass=0.0, **({"quat": quat} if quat is not None else {}), **kw)


def dress_line(spec: mujoco.MjSpec, cfg: WorkcellConfig) -> None:
    dress_conveyor(spec, cfg)
    for t in range(len(cfg.tower_angles)):
        recipe_screen(spec, cfg, t)


def dress_conveyor(spec: mujoco.MjSpec, cfg: WorkcellConfig) -> None:
    """Details on the conveyor's static frame (its x runs along the travel, the pick point at
    its origin; -y is the side away from the robot)."""
    frame = spec.body("conveyor")
    total = cfg.first_cube + (cfg.batch - 1) * cfg.cube_spacing
    upstream = total + cfg.cube_spacing + 0.1
    span = (-(upstream + 0.02), total + 0.12)
    centre, half = (span[0] + span[1]) / 2, (span[1] - span[0]) / 2
    side = cfg.belt_width / 2 + 0.045
    for s in (-1, 1):                                      # aluminium side frames
        _vis(frame, [half, 0.004, 0.022], [centre, s * side, 0.018], _ALU)
        _vis(frame, [half, 0.0045, 0.0025], [centre, s * side, 0.041], _BLACK)
    end = span[1]
    along_y = [np.cos(-np.pi / 4), np.sin(-np.pi / 4), 0.0, 0.0]     # a cylinder's z onto +y
    out = -1.0                                              # the side away from the robot
    _vis(frame, [0.028, 0.11, 0.03], [end - 0.02, 0.0, 0.026], _GREY)              # end guard
    _vis(frame, [0.035, 0.03, 0.03], [end - 0.04, out * (side + 0.035), 0.034], _BLACK)
    _vis(frame, [0.03, 0.06], [end - 0.04, out * (side + 0.125), 0.036], _BLUE, _CYL, along_y)
    _vis(frame, [0.0305, 0.012], [end - 0.04, out * (side + 0.19), 0.036], _BLACK, _CYL, along_y)
    _vis(frame, [0.031, 0.004], [end - 0.04, out * (side + 0.07), 0.036],
         [0.85, 0.87, 0.9, 1.0], _CYL, along_y)
    _vis(frame, [0.012, 0.012, 0.009], [end - 0.06, out * (side + 0.13), 0.074], _BLACK)
    # light barrier across the belt at the pick point
    for s in (-1, 1):
        _vis(frame, [0.009, 0.009, 0.045], [0.0, s * (side + 0.016), 0.045], _BLACK)
        _vis(frame, [0.004, 0.002], [0.0, s * (side + 0.006), 0.03], [0.9, 0.1, 0.08, 1.0], _CYL,
             along_y)
    _vis(frame, [0.0007, side, 0.0007], [0.0, 0.0, 0.03], [1.0, 0.08, 0.05, 0.45])  # its beam
    # emergency stop on the far side, near the downstream end
    _vis(frame, [0.022, 0.022, 0.035], [end - 0.16, out * (side + 0.06), 0.035],
         [0.95, 0.78, 0.05, 1.0])
    _vis(frame, [0.016, 0.006], [end - 0.16, out * (side + 0.06), 0.076],
         [0.85, 0.05, 0.05, 1.0], _CYL)
    # a status beacon on the feeder tunnel, near its mouth
    mouth = cfg.first_cube - 0.08
    _vis(frame, [0.011, 0.012], [-(mouth + 0.05), 0.0, 0.132], [0.15, 0.85, 0.25, 1.0], _CYL)
    _vis(frame, [0.013, 0.004], [-(mouth + 0.05), 0.0, 0.124], _BLACK, _CYL)


def _text_texture(spec: mujoco.MjSpec, name: str, lines: list[str], size=(512, 96),
                  fg=(225, 235, 245), bg=(10, 22, 40)) -> str:
    """A 2D texture with text on it (PIL), loaded from memory; returns its material's name."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(size[1] / (len(lines) + 0.6)))
    except OSError:
        font = ImageFont.load_default()
    step = size[1] / len(lines)
    for k, text in enumerate(lines):
        box = draw.textbbox((0, 0), text, font=font)
        draw.text(((size[0] - (box[2] - box[0])) / 2 - box[0],
                   step * k + (step - (box[3] - box[1])) / 2 - box[1]), text, font=font, fill=fg)
    # mirrored: MuJoCo maps a 2D texture onto a box's -x face reversed
    data = np.ascontiguousarray(np.asarray(img, dtype=np.uint8)[:, ::-1])
    tex = spec.add_texture(name=f"{name}_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                           width=size[0], height=size[1], nchannel=3)
    tex.data = bytes(data.tobytes())
    mat = spec.add_material(name=f"{name}_mat", emission=0.6)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{name}_tex"
    return f"{name}_mat"


def recipe_screen(spec: mujoco.MjSpec, cfg: WorkcellConfig, t: int) -> None:
    """Restyle tower t's recipe board as a monitor on a pole (the board's frame: -x faces the
    robot, z up). The six recipe squares become tiles on the screen, bottom to top."""
    body = spec.body(f"sign{TOWERS[t]}")
    tiles = [g for g in body.geoms if g.name.startswith(f"sign{TOWERS[t]}_")]
    plain = [g for g in body.geoms if not g.name]          # post, panel, header
    for g in plain:                                        # hide the old board
        g.rgba = [0.0, 0.0, 0.0, 0.0]
    z0, z1 = 0.11, 0.63                                    # the monitor's housing
    zc, hz = (z0 + z1) / 2, (z1 - z0) / 2
    _vis(body, [0.009, 0.009, z0 / 2], [0.0, 0.0, z0 / 2], _GREY)                  # pole
    _vis(body, [0.05, 0.05, 0.004], [0.0, 0.0, 0.004], _GREY)                      # foot
    _vis(body, [0.014, 0.082, hz], [0.0, 0.0, zc], _BLACK)                         # housing
    _vis(body, [0.0008, 0.074, hz - 0.008], [-0.0148, 0.0, zc], [0.02, 0.06, 0.13, 1.0])  # glass
    header = _text_texture(spec, f"header{TOWERS[t]}", [f"TOWER {TOWERS[t]}", "RECIPE"],
                           size=(384, 160))
    _vis(body, [0.0006, 0.07, 0.03], [-0.016, 0.0, z1 - 0.04], [1, 1, 1, 1], material=header)
    levels = _text_texture(spec, f"levels{TOWERS[t]}",
                           [str(k) for k in range(cfg.max_height, 0, -1)], size=(64, 384))
    top_tile, bottom_tile = 0.15 + 0.072 * (cfg.max_height - 1), 0.15
    _vis(body, [0.0006, 0.012, (top_tile - bottom_tile) / 2 + 0.03],
         [-0.016, 0.054, (top_tile + bottom_tile) / 2], [1, 1, 1, 1], material=levels)
    # the recipe, as tiles on the screen; they glow (the material's emission) in whatever colour
    # Workcell.reset gives them (a geom's own colour overrides its material's)
    glow = spec.add_material(name=f"tiles{TOWERS[t]}_mat", emission=0.7)
    for g in tiles:
        g.size = [0.0008, 0.045, 0.03]
        g.pos = [-0.0162, -0.012, g.pos[2]]
        g.material = glow.name
