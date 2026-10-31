"""Look at the Milestone 9 work-order cell, running work orders.

Usage: uv run python scripts/view_workcell.py [--speed 1.0] [--static] [--seed 0] [--heights 1 4]
                                              [--pick run:runs/<grasp run>] [--place run:...]
Each order gives the three towers a random recipe (four colours, bottom to top), shown on the
signs behind them. Cubes come out of the hopper through the tunnel in an order that can always
go straight onto a tower; the conveyor indexes each one to the pick point. The cell controller
picks it, carries it in a fixed posture high enough to clear the towers, swings the base to the
tower that needs it and stacks it. By default every step is scripted
(polypanda.control.scripted_cell, the baseline); --pick and --place run trained policies instead
(polypanda.supervisor.cell; a run still training uses its current best checkpoint). --mixed
runs Layer 2 orders: cubes arrive in random order with surplus ones, and the planner sends each
to a tower, a buffer slot on the bench (front left) or the reject bin. The stack
light runs a lamp test, then shows green while an order runs. --static only shows the cell.
Close the window to stop.

v3, knock it yourself: --recover (Layer 2 with the closed loop) and press K in the viewer. The
next time a cube is placed on a tower at least 2 cubes tall, that tower is knocked; the cell
notices and rebuilds it (with --standup run:DIR, lying cubes are stood up by the learned stand-up;
--pick run:runs/grasp_anypose5_seed1 grasps them). --auto-knock knocks one random tower in every
order, as the disturbance suites do. --spares N with --recovery-planner rescue-all | best-rule |
run:DIR calls spare cubes up the belt. (K also toggles the viewer's skybox; the hall hides it.)
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from polypanda.evaluation.cell import CellEvalConfig, run_cell_order, skill_from_spec
from polypanda.simulation.disturbances import Knock, make_suite
from polypanda.simulation.workcell import TOWERS, Workcell, WorkcellConfig, WorkOrder
from polypanda.supervisor.cell import LearnedCell
from polypanda.supervisor.planner import mixed_order
from polypanda.supervisor.symbolic import BEST_RULE, Calibration, load_planner
from polypanda.visualization.viewer import fast_rendering

KEY_K = 75  # GLFW key code


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
    parser.add_argument("--mixed", action="store_true",
                        help="Layer 2: random arrival order with surplus cubes; the planner "
                             "routes each cube to a tower, a buffer slot or the reject bin")
    parser.add_argument("--slots", type=int, default=3, help="with --mixed: buffer slots")
    parser.add_argument("--batch-choice", action="store_true",
                        help="with --mixed: take any cube on the belt, not only the front one")
    parser.add_argument("--pick", default="scripted", help="'scripted' or run:<grasp run dir>")
    parser.add_argument("--place", default="scripted", help="'scripted' or run:<place run dir>")
    parser.add_argument("--aim", choices=("fixture", "trained"), default="fixture",
                        help="learned place target: the fixture centre, or as trained (v3: the "
                             "top tower cube)")
    parser.add_argument("--recover", action="store_true",
                        help="v3 (implies --mixed): read the towers after a knock and rebuild; "
                             "press K to knock")
    parser.add_argument("--standup", default="scripted",
                        help="v3: 'scripted' or run:<stand-up run dir> for lying cubes")
    parser.add_argument("--auto-knock", action="store_true",
                        help="v3, with --recover: one random knock per order")
    parser.add_argument("--spares", type=int, default=0, help="v3, with --recover: spare cubes")
    parser.add_argument("--recovery-planner", default="rescue-all",
                        help="v3, with --spares: rescue-all, best-rule or run:<planner run dir>")
    parser.add_argument("--replay", type=int, metavar="INDEX",
                        help="replay evaluation order INDEX exactly as evaluate_cell.py ran it "
                             "(same order, knock, jitter and policies), again and again")
    parser.add_argument("--orders-seed", type=int, default=CellEvalConfig.seed,
                        help="with --replay: evaluate_cell.py's --seed")
    parser.add_argument("--disturbances", type=Path,
                        help="with --replay: the disturbance suite evaluate_cell.py ran")
    args = parser.parse_args()
    args.mixed = args.mixed or args.recover
    pick, pick_desc, *_ = skill_from_spec(args.pick, "pick")
    place, place_desc, *_ = skill_from_spec(args.place, "place")
    standup, standup_desc, *_ = skill_from_spec(args.standup, "standup")
    print(f"pick: {pick_desc}; place: {place_desc}; moves between stations: scripted")
    if args.recover:
        print(f"stand-up: {standup_desc}; press K in the viewer to knock the next tower")
    replace, spares = None, {}
    if args.spares:
        if args.recovery_planner == "best-rule":
            replace = BEST_RULE
        elif args.recovery_planner.startswith("run:"):
            replace = load_planner(Path(args.recovery_planner[4:]))
        spares = {"spares": args.spares, "replace": replace,
                  "times": Calibration.from_calibration_run().times}
    cell = Workcell.build(WorkcellConfig(n_cubes=(14 if args.mixed else 12) + args.spares))
    if not args.shadows:
        fast_rendering(cell.model)
    rng = np.random.default_rng(args.seed)
    knocks = np.random.default_rng([args.seed, 1])
    cell.reset(WorkOrder.random(rng, heights=tuple(args.heights)))
    controller: LearnedCell | None = None

    def key(code: int) -> None:
        if code == KEY_K and args.recover and controller is not None:
            controller.knock_request = Knock.sample(knocks)
            print("  knock armed: the next tower a cube is placed on (2 or more high) gets it")

    with mujoco.viewer.launch_passive(cell.model, cell.data, key_callback=key) as viewer:
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

        controller = LearnedCell(cell, grasp=pick, place=place, on_step=show, aim=args.aim,
                                 standup=standup)
        controller.recovery_planner = replace
        try:
            if args.replay is not None:
                config = CellEvalConfig(seed=args.orders_seed, mixed=True, recover=args.recover,
                                        disturbances=(str(args.disturbances)
                                                      if args.disturbances else None),
                                        spares=args.spares)
                while True:
                    for lamps in (("green", "amber", "red"), ("red",), ("amber",), ("green",)):
                        cell.set_stack_light(*lamps)
                        controller.step(8)
                    clock["t"] = time.perf_counter()
                    e = run_cell_order(controller, config, args.replay)
                    cell.set_stack_light("green" if e.success else "red")
                    print(f"order {args.replay}: {'complete' if e.success else e.failure}; "
                          f"knock {e.disturbance or 'none'}, {e.fallen} cubes fell; "
                          f"{e.sim_s:.0f} s simulated")
                    controller.step(int(3.0 / dt))
            n = 1
            while True:
                if args.mixed:
                    order, _ = mixed_order(rng, heights=tuple(args.heights))
                else:
                    order = WorkOrder.random(rng, heights=tuple(args.heights))
                print(f"order {n}: " + "; ".join(f"{TOWERS[t]} " + "-".join(r)
                                                 for t, r in enumerate(order.recipes)))
                if args.mixed:
                    print("  arriving: " + ", ".join(c for c, _ in order.arrivals))
                for lamps in (("green", "amber", "red"), ("red",), ("amber",), ("green",)):
                    cell.set_stack_light(*lamps)  # lamp test
                    controller.step(8)
                clock["t"] = time.perf_counter()
                picks, places = len(controller.picks), len(controller.places)
                jitter = {"lateral": 0.015 if args.jitter else 0.0,
                          "yaw": np.radians(15) if args.jitter else 0.0}
                if args.mixed:
                    knock = (make_suite([order.recipes], knocks, no_knock=0.0)[0]
                             if args.auto_knock else None)
                    if knock is not None:
                        print(f"  a knock will hit tower {TOWERS[knock.tower]} at height "
                              f"{knock.height}")
                    towers = controller.run_mixed_order(order, rng, slots=args.slots,
                                                        batch_choice=args.batch_choice,
                                                        log=lambda s: print("  " + s),
                                                        disturbance=knock, recover=args.recover,
                                                        **spares, **jitter)
                else:
                    towers = controller.run_order(order, rng, log=lambda s: print("  " + s),
                                                  **jitter)
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
