"""Watch the scripted grasp baseline pick up cubes from the held-out placement set.

Usage: uv run python scripts/demo_grasp.py [--episodes 5] [--speed 1.0]
Close the window to stop.
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from polypanda.control.action_adapter import JointDeltaActionAdapter
from polypanda.control.scripted_grasp import GRIPPER_OPEN, Phase, ScriptedGrasp
from polypanda.evaluation.sets import GraspEvalSet
from polypanda.simulation.grasp import GraspMonitor
from polypanda.simulation.panda import PandaIndices, reset_to_home
from polypanda.simulation.scenes import CubeIndices, load_pick_scene, set_cube_pose


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    parser.add_argument("--set", type=Path, default=Path("eval_sets/grasp_nominal_v1.json"))
    args = parser.parse_args()

    model, data = load_pick_scene()
    arm, cube = PandaIndices.from_model(model), CubeIndices.from_model(model)
    adapter = JointDeltaActionAdapter(model, arm)
    monitor = GraspMonitor(model, arm, cube)
    controller = ScriptedGrasp(model, arm, cube, adapter.max_delta)
    eval_set = GraspEvalSet.load(args.set)
    frame = adapter.control_dt / args.speed

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.sitegroup[4] = 1  # show the TCP site
        for i, (x, y, yaw) in enumerate(eval_set.placements[:args.episodes]):
            reset_to_home(model, data, arm)
            set_cube_pose(model, data, cube, (x, y), yaw)
            adapter.reset_to_measured(data)
            data.ctrl[arm.gripper_actuator] = GRIPPER_OPEN
            monitor.reset()
            controller.reset(data)
            viewer.sync()
            time.sleep(0.5 / args.speed)
            phase, state = controller.phase, None
            while controller.phase != Phase.DONE:
                if not viewer.is_running():
                    return
                start = time.perf_counter()
                action, gripper = controller.act(data)
                data.ctrl[arm.gripper_actuator] = gripper
                adapter.step(data, action)
                state = monitor.update(data)
                viewer.sync()
                if controller.phase != phase:
                    phase = controller.phase
                    print(f"  {phase.name.lower()}")
                time.sleep(max(0.0, frame - (time.perf_counter() - start)))
            outcome = "GRASPED" if state.grasped else f"failed ({controller.timed_out})"
            print(f"cube {i + 1}/{args.episodes} at ({x:.2f}, {y:+.2f}), yaw {yaw:+.2f}: "
                  f"{outcome}, lifted {state.lift * 100:.1f} cm")
            time.sleep(1.0 / args.speed)


if __name__ == "__main__":
    main()
