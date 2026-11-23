"""Render the keyed part and the keyed pocket close up (docs/keyed_part.md).

Three panels of M1's pocket on the v4 line with keyed parts and a keyed machine, seen steeply
from behind the machine (the slot's side), its roof and lid hidden and its walls drawn faintly
(a cutaway, for the render only): the empty pocket with its slot in the back wall; a part the
right way round, seated with its rib (dark) in the slot; a part turned 90 deg, resting on its
rib on the lead-in. The parts are dropped in and settle physically, as in tests/test_keyed_part.py.

Usage:
  uv run --group media python scripts/render_keyed_part.py --out media/v4_keyed_part.png
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from polypanda.simulation.line import Line, LineConfig, MachineSpot
from polypanda.simulation.machine import MachineConfig
from polypanda.simulation.workcell import COLOURS, WorkcellConfig, _yaw_quat

PANELS = (("The keyed pocket: a slot in the back wall", None),
          ("Right way round: seated, rib in the slot", 0.0),
          ("Turned 90 deg: it rests on its rib", 90.0))


def build() -> Line:
    cfg = MachineConfig(keyed=True, cycle_s=1e6)
    return Line.build_line(LineConfig(cell=WorkcellConfig(tower_angles=(-40.0,), dressing=False),
                                      machines=(MachineSpot("M1", -0.6, config=cfg),),
                                      keyed_parts=True))


def cutaway(line: Line) -> None:
    """Draw the machine's walls, roof, lid and handle see-through; hide the spindle."""
    m = line.model
    for g in range(m.ngeom):
        name = m.geom(g).name
        if name.startswith("M1_") and not name.startswith("M1_pocket"):
            m.geom_rgba[g, 3] = 0.1 if "wall" in name else 0.0


def stage(line: Line, wrong: float | None) -> None:
    line.reset_line()
    m, d = line.model, line.data
    m1 = line.machines["M1"]
    d.qpos[m1.lid_qpos] = m1.cfg.travel
    part = line.cubes[0]
    m.geom_rgba[part.geom] = COLOURS["blue"]
    if wrong is not None:
        p = m1.pocket_xyz()
        R = d.xmat[m1.body].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0]) + np.radians(wrong)
        d.qpos[part.qpos:part.qpos + 7] = [p[0], p[1], part.half_size[2] + 0.02, *_yaw_quat(yaw)]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    for _ in range(80):                     # 2 s for the part to drop in and settle
        line.update_machines()
        mujoco.mj_step(m, d, nstep=25)
    mujoco.mj_forward(m, d)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("media/v4_keyed_part.png"))
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=760)
    args = parser.parse_args()

    line = build()
    cutaway(line)
    renderer = mujoco.Renderer(line.model, args.height, args.width)
    camera = mujoco.MjvCamera()
    pocket = line.machines["M1"].pocket_xyz()
    # azimuth and elevation give the direction the camera looks along: from behind the machine
    # (-y of the pocket, world: the slot's side) towards the robot (+y), steeply down so the
    # rib shows in the slot in front of the part, a little from the side
    camera.azimuth, camera.elevation, camera.distance = 115.0, -62.0, 0.25
    camera.lookat[:] = [pocket[0], pocket[1], 0.012]
    tiles = []
    font = ImageFont.load_default(size=30) if hasattr(ImageFont, "load_default") else None
    for caption, wrong in PANELS:
        stage(line, wrong)
        renderer.update_scene(line.data, camera)
        img = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, args.width, 50], fill=(20, 22, 26))
        draw.text((16, 9), caption, fill=(240, 240, 240), font=font)
        tiles.append(img)
    sheet = Image.new("RGB", (args.width * len(tiles) + 8 * (len(tiles) - 1), args.height),
                      (20, 22, 26))
    for k, img in enumerate(tiles):
        sheet.paste(img, (k * (args.width + 8), 0))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
