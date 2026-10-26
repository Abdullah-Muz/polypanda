"""Look at the Milestone 9 work-order cell, running work orders.

Usage: uv run python scripts/view_workcell.py [--speed 1.0] [--static] [--seed 0] [--heights 1 4]
                                              [--pick run:runs/<grasp run>] [--place run:...]
Each order gives the three towers a random recipe (four colours, bottom to top), shown on the
signs behind them. Cubes come out of the hopper through the tunnel in an order that can always
go straight onto a tower; the conveyor indexes each one to the pick point. The cell controller
picks it, carries it in a fixed posture high enough to clear the towers, swings the base to the
tower that needs it and stacks it. By default every step is scripted
(polypanda.control.scripted_cell, the baseline); --pick and --place run trained policies instead
(polypanda.supervisor.cell; a run still training uses its current best checkpoint). The stack
light runs a lamp test, then shows green while an order runs. --static only shows the cell.
Close the window to stop.
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from polypanda.evaluation.cell import skill_from_spec
from polypanda.simulation.workcell import TOWERS, Workcell, WorkOrder
from polypanda.supervisor.cell import LearnedCell
from polypanda.visualization.viewer import fast_rendering


class Closed(Exception):
    pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    parser.add_argument("--static", action="store_true", help="only show the cell")
    parser.add_argument("--shadows", action="store_true", help="render shadows and reflections")
    parser.add_argument("--seed", type=int, default=0, help="random work orders from this seed")
    parser.add_argument("--heights", type=int, nargs=2, default=(1, 4), help="tower height range")
    parser.add_argument("--jitter", action="store_true",
                        help="cubes arrive offset and rotated (+-1.5 cm, +-15 deg)")
    parser.add_argument("--pick", default="scripted", help="'scripted' or run:<grasp run dir>")
    parser.add_argument("--place", default="scripted", help="'scripted' or run:<place run dir>")
    args = parser.parse_args()
    pick, pick_desc, *_ = skill_from_spec(args.pick, "pick")
    place, place_desc, *_ = skill_from_spec(args.place, "place")
    print(f"pick: {pick_desc}; place: {place_desc}; moves between stations: scripted")
    cell = Workcell.build()
    if not args.shadows:
        fast_rendering(cell.model)
    rng = np.random.default_rng(args.seed)
    cell.reset(WorkOrder.random(rng, heights=tuple(args.heights)))
    with mujoco.viewer.launch_passive(cell.model, cell.data) as viewer:
        cam = viewer.cam  # over the robot's shoulder toward the towers and their signs
        cam.azimuth, cam.elevation, cam.distance, cam.lookat[:] = 200, -28, 2.9, [0, -0.05, 0.15]
        if args.static:
            cell.set_stack_light("green")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)
            return
        dt = 25 * cell.model.opt.timestep
        clock = {"t": time.perf_counter()}

        def show():
            if not viewer.is_running():
                raise Closed
            viewer.sync()
            clock["t"] += dt / args.speed
            time.sleep(max(0.0, clock["t"] - time.perf_counter()))

        controller = LearnedCell(cell, grasp=pick, place=place, on_step=show)
        try:
            n = 1
            while True:
                order = WorkOrder.random(rng, heights=tuple(args.heights))
                print(f"order {n}: " + "; ".join(f"{TOWERS[t]} " + "-".join(r)
                                                 for t, r in enumerate(order.recipes)))
                for lamps in (("green", "amber", "red"), ("red",), ("amber",), ("green",)):
                    cell.set_stack_light(*lamps)  # lamp test
                    controller.step(8)
                clock["t"] = time.perf_counter()
                picks, places = len(controller.picks), len(controller.places)
                towers = controller.run_order(
                    order, rng, lateral=0.015 if args.jitter else 0.0,
                    yaw=np.radians(15) if args.jitter else 0.0, log=lambda s: print("  " + s))
                ok = controller.check(towers)
                cell.set_stack_light("green" if all(ok) else "red")
                print("  towers " + ", ".join(f"{TOWERS[t]} {'OK' if g else 'WRONG'}"
                                              for t, g in enumerate(ok)))
                for name, rows in (("learned picks", controller.picks[picks:]),
                                   ("learned places", controller.places[places:])):
                    if rows:
                        fails = [r["failure"] or "timeout" for r in rows if not r["success"]]
                        print(f"  {name}: {len(rows) - len(fails)}/{len(rows)} succeeded"
                              + (f" (failures: {', '.join(fails)})" if fails else ""))
                controller.step(int(3.0 / dt))
                n += 1
        except Closed:
            pass


if __name__ == "__main__":
    main()
