"""Open the interactive viewer on PolyPanda's Panda model (Menagerie scene + TCP site), at home.

Usage: uv run python scripts/view_panda.py [--cube]
The TCP is drawn as a small red sphere between the fingertips. --cube loads the pick scene.
"""

import argparse
import time

import mujoco
import mujoco.viewer

from polypanda.simulation.assets import load_panda_scene
from polypanda.simulation.panda import PandaIndices, reset_to_home
from polypanda.simulation.scenes import load_pick_scene
from polypanda.visualization.viewer import fast_rendering


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cube", action="store_true", help="add the graspable cube")
    parser.add_argument("--shadows", action="store_true", help="render shadows and reflections")
    args = parser.parse_args()
    model, data = load_pick_scene() if args.cube else load_panda_scene()
    if not args.shadows:
        fast_rendering(model)
    idx = PandaIndices.from_model(model)
    reset_to_home(model, data, idx)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.sitegroup[4] = 1  # the TCP site lives in group 4, hidden by default
        start = time.perf_counter()
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            lag = data.time - (time.perf_counter() - start)  # keep simulated time ~ wall time
            if lag > 0:
                time.sleep(lag)


if __name__ == "__main__":
    main()
