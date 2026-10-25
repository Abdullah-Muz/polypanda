"""Watch full pick-and-place: the supervisor chains grasp, transport and place on held-out scenes.

Usage: uv run python scripts/demo_pickplace.py [--grasp scripted] [--episodes 6]
       uv run python scripts/demo_pickplace.py --grasp run:runs/grasp_ppo_curr_seed1
       uv run python scripts/demo_pickplace.py --grasp run:runs/grasp_cart_seed1 \
           --transport run:runs/transport_cart_seed0 --place run:runs/place_cart_seed0
       uv run python scripts/demo_pickplace.py --scenes 62 178     # specific scenes (retries)
The green disc on the table is the target spot (2 cm placement tolerance). The terminal prints
each stage as the supervisor switches skills, including recoveries and re-grasps. Each skill
runs in its own environment; the viewer mirrors whichever one is active. Close the window to stop.
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from polypanda.environments.handoff import capture_state, restore_state
from polypanda.evaluation.pickplace import grasp_driver_from_spec, grasp_env_for, skill_from_spec
from polypanda.evaluation.sets import PickPlaceEvalSet
from polypanda.simulation.scenes import load_pick_scene
from polypanda.supervisor.fsm import PickPlaceSupervisor, Stage
from polypanda.visualization.viewer import fast_rendering


class Closed(Exception):
    pass


def draw_target(viewer, xy, radius: float) -> None:
    scn = viewer.user_scn
    mujoco.mjv_initGeom(scn.geoms[0], type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                        size=[radius, 0.0005, 0], pos=[xy[0], xy[1], 0.0005],
                        mat=np.eye(3).flatten(), rgba=[0.1, 0.9, 0.2, 0.5])
    scn.ngeom = 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grasp", default="scripted", help="scripted | run:DIR[@best|final]")
    parser.add_argument("--transport", default="scripted", help="scripted | run:DIR[@...]")
    parser.add_argument("--place", default="scripted", help="scripted | run:DIR[@...]")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--scenes", type=int, nargs="*", help="scene indices to show")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    parser.add_argument("--shadows", action="store_true", help="render shadows and reflections")
    parser.add_argument("--set", type=Path, default=Path("eval_sets/pickplace_nominal_v1.json"))
    args = parser.parse_args()

    grasp = grasp_driver_from_spec(args.grasp)[0]
    transport, transport_env = skill_from_spec(args.transport, "transport")[:2]
    place, place_env = skill_from_spec(args.place, "place")[:2]
    supervisor = PickPlaceSupervisor(grasp, transport, place, grasp_env=grasp_env_for(args.grasp),
                                     transport_env=transport_env, place_env=place_env)
    eval_set = PickPlaceEvalSet.load(args.set)
    model, data = load_pick_scene()  # display copy; mirrors the active skill's environment
    if not args.shadows:
        fast_rendering(model)
    frame = supervisor.dt / args.speed

    with mujoco.viewer.launch_passive(model, data) as viewer:
        current = {"stage": None, "t": 0}

        def show(stage, env):
            if not viewer.is_running():
                raise Closed
            start = time.perf_counter()
            if stage != current["stage"]:
                print(f"  {current['t'] * supervisor.dt:5.1f} s  {stage.value.upper()}")
                current["stage"] = stage
            current["t"] += 1
            restore_state(model, data, capture_state(env.model, env.data))
            viewer.sync()
            time.sleep(max(0.0, frame - (time.perf_counter() - start)))

        supervisor.on_step = show
        scenes = args.scenes if args.scenes else range(args.episodes)
        try:
            for i in scenes:
                pose, target = eval_set.placements[i], eval_set.targets[i]
                print(f"scene {i}: cube at ({pose[0]:.2f}, {pose[1]:.2f}), "
                      f"target ({target[0]:.2f}, {target[1]:.2f})")
                draw_target(viewer, target, supervisor.envs[Stage.PLACE].config.check.tolerance)
                current.update(stage=None, t=0)
                r = supervisor.run(pose, target)
                if r.success:
                    retries = r.grasp_attempts - 1
                    retry = f" after {retries} re-grasp(s)" if retries else ""
                    print(f"  PLACED in {r.total_s:.1f} s{retry}, "
                          f"{r.final_xy_error_mm:.1f} mm from the target")
                else:
                    print(f"  FAILED at {r.failure_stage}: {r.failure_reason} "
                          f"({r.total_s:.1f} s)")
                time.sleep(1.0 / args.speed)
        except Closed:
            pass


if __name__ == "__main__":
    main()
