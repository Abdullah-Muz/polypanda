"""Render stills of the v4.1 electronics test cell (simulation/test_cell.py).

A staged moment of a shift: tester A testing (lid shut, amber), tester B open with a module in
its nest, tester C idle, modules waiting on the conveyor, three in the PASS tray, one on the
vision station's plate, the robot at B's stop. Two views: the whole cell, and a close-up of B's
nest and the modules (label, barcode, board edge; one with a red mark, one with flux residue).
Staged, not simulated (the parts are set where a shift would leave them).

Usage:
  uv run python scripts/render_test_cell.py --out media/v41_test_cell
Writes <out>_overview.png and <out>_closeup.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from polypanda.control.test_cell import TestCellController
from polypanda.simulation.machine import RUNNING
from polypanda.simulation.test_cell import ModuleTruth, TestCell
from polypanda.simulation.workcell import _yaw_quat

VIEWS = {  # lookat (x, y, z), azimuth, elevation, distance
    "overview": ((-0.6, 0.0, 0.05), 90.0, -52.0, 2.05),
    "closeup": ((-0.45, -0.5, 0.12), 100.0, -58.0, 0.6),
}


def stage(cell: TestCell) -> None:
    m, d = cell.model, cell.data
    truths = [ModuleTruth(f"M0001-{k:04d}", True) for k in range(12)]
    truths[3] = ModuleTruth("M0001-0003", True, "red_mark", severity=0.9)
    truths[5] = ModuleTruth("M0001-0005", True, residue=1.0)
    cell.reset_shift(truths)
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
        put(k, mach.pocket_xyz()[:2], float(np.arctan2(R[1, 0], R[0, 0])) + mach.cfg.key_yaw)

    tb = cell.machines["TB"]
    d.qpos[tb.lid_qpos] = tb.cfg.travel - 0.005
    in_nest(4, "TA")
    in_nest(5, "TB")
    for i, k in enumerate((6, 7, 8)):
        put(k, cell.pockets[i], np.radians(150.0))
    put(3, cell.vision_xy, np.radians(170.0))
    belt_z = cell.cubes[0].half_size[2] + cell.config.belt_lift
    for i, k in enumerate((0, 1, 2)):         # waiting on the belt, the first at the pick point
        put(k, (cell.pick_xy[0] + 0.1 * i, cell.pick_xy[1]), 0.0, belt_z)
    for k in (9, 10, 11):                     # out of sight, in the hopper
        put(k, (3.0 + k, 3.0), 0.0)
    mujoco.mj_forward(m, d)
    ta = cell.machines["TA"]
    ta.state, ta.part = RUNNING, None
    ta._show()


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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, default=Path("media/v41_test_cell"))
    p.add_argument("--width", type=int, default=1600)
    p.add_argument("--height", type=int, default=900)
    p.add_argument("--views", nargs="+", default=list(VIEWS), choices=list(VIEWS))
    args = p.parse_args()
    cell = TestCell.build()
    cell.model.mat_reflectance[:] = 0.0   # floor reflections render with streaks off-screen
    stage(cell)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for view in args.views:
        path = args.out.parent / f"{args.out.name}_{view}.png"
        Image.fromarray(render(cell, view, args.width, args.height)).save(path)
        print(f"wrote {path}")
    cell.close()


if __name__ == "__main__":
    main()
