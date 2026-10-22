"""Open the interactive viewer on PolyPanda's Panda model (Menagerie scene + TCP site), at home.

Usage: uv run python scripts/view_panda.py
The TCP is drawn as a small red sphere between the fingertips.
"""

import time

import mujoco
import mujoco.viewer

from polypanda.simulation.assets import load_panda_scene
from polypanda.simulation.panda import PandaIndices, reset_to_home


def main() -> None:
    model, data = load_panda_scene()
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
