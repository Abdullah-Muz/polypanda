"""Render stills of the v4.1 electronics test cell (simulation/test_cell.py).

"overview" and "closeup": a staged moment of a shift: tester A testing (lid shut, amber), tester
B open with a module in its nest, tester C idle, modules waiting on the conveyor, three in the
PASS tray, one in the REWORK tray, the robot at B's stop; the whole cell, and a close-up of B's
nest and the modules (label, barcode, board edge on their fronts). Staged, not simulated (the
parts are set where a shift would leave them).

"inspect": the robot showing a held module to the vision camera, simulated (the module picked
up and presented by the cell controller, TestCellController.present), with the camera's own
image in a corner (3x, the rule's window outlined) and what the rule read from it.

Usage:
  uv run python scripts/render_test_cell.py --out media/v41_test_cell
  uv run python scripts/render_test_cell.py --testers planned --views overview closeup
Writes <out>_<view>.png for each view.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from polypanda.control.test_cell import TestCellController
from polypanda.perception.inspection import NOISE, RULE, inspect
from polypanda.simulation.machine import RUNNING
from polypanda.simulation.test_cell import ModuleTruth, TestCell, cell_config, planned_testers
from polypanda.simulation.workcell import _yaw_quat

VIEWS = {  # lookat (x, y, z), azimuth, elevation, distance
    "overview": ((-0.6, 0.0, 0.05), 90.0, -52.0, 2.05),
    "closeup": ((-0.45, -0.5, 0.12), 100.0, -58.0, 0.6),
    "inspect": ((-1.0, 0.31, 0.35), -70.0, -12.0, 0.48),
}


def truths() -> list[ModuleTruth]:
    out = [ModuleTruth(f"M0001-{k:04d}", True) for k in range(12)]
    out[3] = ModuleTruth("M0001-0003", True, "red_mark", severity=0.9)
    out[5] = ModuleTruth("M0001-0005", True, residue=1.0)
    return out


def stage(cell: TestCell) -> None:
    m, d = cell.model, cell.data
    cell.reset_shift(truths())
    ctl = TestCellController(cell)
    ctl.q_cmd = ctl.home.copy()
    ctl.to_carry()
    ctl.go_to("TB")
    while cell.take_body() is not None:       # every body gets a module
        pass

    def put(k: int, xy, yaw: float, z: float = 0.03) -> None:
        cube = cell.cubes[cell.body_module.index(k)]
        d.qpos[cube.qpos:cube.qpos + 7] = [xy[0], xy[1], z, *_yaw_quat(yaw)]
        d.qvel[cube.qvel:cube.qvel + 6] = 0.0

    def in_nest(k: int, name: str) -> None:
        mach = cell.machines[name]
        R = d.xmat[mach.body].reshape(3, 3)
        put(k, mach.pocket_xyz()[:2], float(np.arctan2(R[1, 0], R[0, 0])) + mach.cfg.key_yaw,
            mach.pocket_xyz()[2] + 0.03)                    # (a drawer's nest is on its tray)

    tb = cell.machines["TB"]
    # (a drawer stands open at its stop, held there by its index pin)
    d.qpos[tb.lid_qpos] = tb.cfg.travel - (0.0 if tb.cfg.kind == "drawer" else 0.005)
    mujoco.mj_forward(m, d)
    in_nest(4, "TA")
    in_nest(5, "TB")
    for i, k in enumerate((6, 7, 8)):
        put(k, cell.pockets[i], np.radians(150.0))
    put(3, cell.rework_pockets[0], np.radians(-150.0))
    belt_z = cell.cubes[0].half_size[2] + cell.config.belt_lift
    for i, k in enumerate((0, 1, 2)):         # waiting on the belt, the first at the pick point
        put(k, (cell.pick_xy[0] + 0.1 * i, cell.pick_xy[1]), np.pi, belt_z)
    for k in (9, 10, 11):                     # out of sight, in the hopper
        put(k, (3.0 + k, 3.0), 0.0)
    mujoco.mj_forward(m, d)
    ta = cell.machines["TA"]
    ta.state, ta.part = RUNNING, None
    ta._show()


def stage_inspect(cell: TestCell) -> np.ndarray:
    """The robot presents module 3 (a red mark) to the vision camera, after taking it out of
    tester A as the cell would (held across its rib's face); returns the camera's image."""
    m, d = cell.model, cell.data
    cell.reset_shift(truths()[:4])
    ctl = TestCellController(cell, nest_belief="true")
    ctl.q_cmd = ctl.home.copy()
    ctl.to_carry()
    ctl.go_to("TA")
    while cell.take_body() is not None:
        pass
    cube = cell.cubes[cell.body_module.index(3)]
    ta = cell.machines["TA"]
    R = d.xmat[ta.body].reshape(3, 3)
    d.qpos[cube.qpos:cube.qpos + 7] = [*ta.pocket_xyz()[:2], 0.03,
                                       *_yaw_quat(float(np.arctan2(R[1, 0], R[0, 0])))]
    d.qvel[cube.qvel:cube.qvel + 6] = 0.0
    for k, x in zip((0, 1, 2), (0.0, 0.1, 0.2), strict=True):   # the others on the belt
        c = cell.cubes[cell.body_module.index(k)]
        d.qpos[c.qpos:c.qpos + 7] = [cell.pick_xy[0] + x, cell.pick_xy[1],
                                     c.half_size[2] + cell.config.belt_lift, *_yaw_quat(np.pi)]
    d.qpos[ta.lid_qpos] = ta.cfg.travel - 0.005           # staged: A open, its test done
    mujoco.mj_forward(m, d)
    ctl.unload(cube)
    ctl.present(cube)
    return cell.camera_image("vision")


def render(cell: TestCell, view: str, width: int, height: int) -> np.ndarray:
    m = cell.model
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, width)
    m.vis.global_.offheight = max(m.vis.global_.offheight, height)
    lookat, az, el, dist = VIEWS[view]
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.azimuth, cam.elevation, cam.distance = az, el, dist
    with mujoco.Renderer(m, height, width) as r:
        r.update_scene(cell.data, cam)
        return r.render().copy()


def with_inset(still: np.ndarray, image: np.ndarray, scale: int = 3) -> Image.Image:
    """The still with the vision camera's image in its top right corner, the rule's window
    outlined and the rule's reading under it."""
    h, w = image.shape[:2]
    big = Image.fromarray(image).resize((w * scale, h * scale), Image.NEAREST)
    draw = ImageDraw.Draw(big)
    mw, mh = (int(n * (1 - s) / 2) for n, s in ((w, RULE.window[0]), (h, RULE.window[1])))
    draw.rectangle([mw * scale, mh * scale, (w - mw) * scale - 1, (h - mh) * scale - 1],
                   outline=(0, 230, 255), width=2)
    ins = inspect(NOISE.apply(image, np.random.default_rng(0)))
    out = Image.fromarray(still)
    pad = 12
    x0 = out.width - big.width - pad
    out.paste(big, (x0, pad))
    text = (f"vision camera: label {'seen' if ins.label else 'NOT seen'}, "
            f"red mark {'SEEN' if ins.mark else 'not seen'}")
    d = ImageDraw.Draw(out)
    d.rectangle([x0, pad + big.height, x0 + big.width, pad + big.height + 22], fill=(0, 0, 0))
    d.text((x0 + 6, pad + big.height + 5), text, fill=(255, 255, 255))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, default=Path("media/v41_test_cell"))
    p.add_argument("--width", type=int, default=1600)
    p.add_argument("--height", type=int, default=900)
    p.add_argument("--views", nargs="+", default=list(VIEWS), choices=list(VIEWS))
    p.add_argument("--testers", choices=("stand_in", "planned"), default="stand_in",
                   help="stand_in: three lid testers; planned: lid, drawer, pneumatic")
    args = p.parse_args()
    cell = TestCell.build(cell_config(testers=planned_testers()) if args.testers == "planned"
                          else None)
    cell.model.mat_reflectance[:] = 0.0   # floor reflections render with streaks off-screen
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staged = [v for v in args.views if v != "inspect"]
    if staged:
        stage(cell)
        for view in staged:
            path = args.out.parent / f"{args.out.name}_{view}.png"
            Image.fromarray(render(cell, view, args.width, args.height)).save(path)
            print(f"wrote {path}")
    if "inspect" in args.views:
        image = stage_inspect(cell)
        path = args.out.parent / f"{args.out.name}_inspect.png"
        with_inset(render(cell, "inspect", args.width, args.height), image).save(path)
        print(f"wrote {path}")
    cell.close()


if __name__ == "__main__":
    main()
