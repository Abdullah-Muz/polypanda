"""Look at the v4.1 electronics test cell in MuJoCo's viewer, running a shift live.

Usage:
  uv run python scripts/view_test_cell.py                      # real testers, search insertion
  uv run python scripts/view_test_cell.py --static             # just look around the cell
  uv run python scripts/view_test_cell.py --insertion straight_down --speed 2 --n 10
The robot on its rail picks each module off the conveyor, scans it, opens a tester (A lid and
B drawer with the door script, C pneumatic on its own), seats the module in the nest, closes it,
and when a test ends unloads it, shows it to the vision camera on the fly and sorts it: PASS
tray, FAIL bin (red quality zone) or REWORK tray (not tested). Close the window to stop. Mouse:
left-drag rotates, right-drag moves, scroll zooms; double-click a body to centre on it.
"""

import argparse
import time

import mujoco
import mujoco.viewer

from polypanda.control.test_cell import TestCellController
from polypanda.simulation.test_cell import (
    TestCell,
    cell_config,
    draw_truths,
    planned_testers,
    stand_in_testers,
)
from polypanda.visualization.viewer import fast_rendering


class Closed(Exception):
    pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--testers", choices=("planned", "stand_in"), default="planned",
                   help="planned: A lid, B drawer, C pneumatic; stand_in: three lids")
    p.add_argument("--insertion", choices=("search", "straight_down"), default="search")
    p.add_argument("--door", default="feel_v2", help="a door script or run:<path>[@final]")
    p.add_argument("--n", type=int, default=12, help="modules in the shift")
    p.add_argument("--seed", type=int, default=7, help="the shift's modules and noise")
    p.add_argument("--cycle-s", type=float, default=45.0, help="each test's duration (s)")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    p.add_argument("--static", action="store_true", help="only show the cell")
    p.add_argument("--no-dressing", action="store_true", help="the bare cell (faster)")
    p.add_argument("--shadows", action="store_true", help="render shadows and reflections")
    args = p.parse_args()

    testers = (planned_testers if args.testers == "planned" else stand_in_testers)(args.cycle_s)
    cell = TestCell.build(cell_config(cycle_s=args.cycle_s, dressing=not args.no_dressing,
                                      testers=testers))
    if not args.shadows:
        fast_rendering(cell.model)
    mujoco.mj_forward(cell.model, cell.data)

    with mujoco.viewer.launch_passive(cell.model, cell.data) as viewer:
        cam = viewer.cam            # from the operator's side, over the rail toward the testers
        cam.azimuth, cam.elevation, cam.distance = 90.0, -32.0, 3.0
        cam.lookat[:] = [-0.45, 0.0, 0.25]
        if args.static:
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)
            return
        dt = 25 * cell.model.opt.timestep
        clock = {"t": time.perf_counter()}
        ctl = TestCellController(cell, door=args.door, insertion=args.insertion,
                                 seed=args.seed, log=lambda s: print(f"  {s}"))
        tick = ctl._tick

        def tick_and_show(exclude=None):       # every control step, door moves included
            tick(exclude)
            if not viewer.is_running():
                raise Closed
            viewer.sync()
            clock["t"] += dt / args.speed
            time.sleep(max(0.0, clock["t"] - time.perf_counter()))

        ctl._tick = tick_and_show
        ctl.on_step = tick_and_show
        truths = draw_truths(args.n, args.seed, 0.05, 0.03, 0.04)
        print(f"shift of {args.n} modules (seed {args.seed}), testers {args.testers}, "
              f"door {args.door}, insertion {args.insertion}")
        try:
            ctl.run_shift(truths, seed=args.seed)
        except Closed:
            return
        print("shift done; close the window to exit")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


if __name__ == "__main__":
    main()
