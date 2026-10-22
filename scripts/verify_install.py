"""Milestone 1 install check: library versions, Panda model contract, headless stepping, throughput.

Usage: uv run python scripts/verify_install.py [--steps N]
Exits non-zero if any check fails.
"""

import argparse
import platform
import resource
import sys
import time

import gymnasium
import mujoco
import numpy as np
import stable_baselines3
import torch

from polypanda.simulation.assets import (
    MENAGERIE_COMMIT,
    load_panda_scene,
    menagerie_revision,
    panda_scene_path,
)
from polypanda.simulation.panda import ARM_ACTUATORS, PandaIndices, reset_to_home

failures: list[str] = []


def check(ok: bool, label: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        failures.append(label)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50_000)
    args = parser.parse_args()

    print("== Environment")
    print(f"  Python {platform.python_version()} on {platform.platform()}")
    print(f"  MuJoCo {mujoco.__version__} | Gymnasium {gymnasium.__version__} | "
          f"PyTorch {torch.__version__} | SB3 {stable_baselines3.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()} (expected False, CPU-only build)")
    print(f"  torch threads: {torch.get_num_threads()}")

    print("== Menagerie model")
    revision = menagerie_revision()
    print(f"  scene: {panda_scene_path()}")
    print(f"  revision: {revision}")
    check(revision == MENAGERIE_COMMIT, "Menagerie checkout matches pinned commit")

    model, data = load_panda_scene()
    idx = PandaIndices.from_model(model)
    print(f"  timestep {model.opt.timestep} s | nq {model.nq} | nv {model.nv} | nu {model.nu}")
    print(f"  {'actuator':<10} {'id':>2} {'kp':>6} {'kd':>6} {'ctrlrange':>18} {'forcerange':>14}")
    for name, a in zip(ARM_ACTUATORS, idx.arm_actuators, strict=True):
        lo, hi = model.actuator_ctrlrange[a]
        flo, fhi = model.actuator_forcerange[a]
        print(f"  {name:<10} {a:>2} {model.actuator_gainprm[a, 0]:>6.0f} "
              f"{-model.actuator_biasprm[a, 2]:>6.0f} {f'[{lo:.4f}, {hi:.4f}]':>18} "
              f"{f'[{flo:.0f}, {fhi:.0f}]':>14}")
    g = idx.gripper_actuator
    print(f"  gripper actuator id {g}, ctrlrange {model.actuator_ctrlrange[g]}")
    check(model.nu == 8, "8 actuators (7 arm + 1 gripper)")
    check(np.isclose(model.opt.timestep, 0.002), "physics timestep is 0.002 s (500 Hz)")

    print("== Home keyframe")
    reset_to_home(model, data, idx)
    home_q = data.qpos[idx.arm_qpos].copy()
    home_ctrl = data.ctrl[idx.arm_actuators].copy()
    print(f"  arm qpos: {np.round(home_q, 4)}")
    check(np.allclose(home_q, home_ctrl), "home arm targets equal home joint positions")
    lo, hi = model.actuator_ctrlrange[idx.arm_actuators].T
    check(bool(np.all((home_ctrl >= lo) & (home_ctrl <= hi))), "home targets within ctrlrange")

    print(f"== Headless stepping ({args.steps} steps, holding home targets)")
    start = time.perf_counter()
    for _ in range(args.steps):
        mujoco.mj_step(model, data)
    elapsed = time.perf_counter() - start
    drift = np.abs(data.qpos[idx.arm_qpos] - home_q).max()
    finite = all(np.all(np.isfinite(x)) for x in (data.qpos, data.qvel, data.qacc))
    print(f"  simulated {data.time:.1f} s in {elapsed:.2f} s wall")
    rate = args.steps / elapsed
    print(f"  {rate:,.0f} physics steps/s | real-time factor {data.time / elapsed:.1f}x"
          f" | ~{rate / 25:,.0f} policy steps/s at 25 substeps")
    print(f"  max arm drift from home: {drift:.2e} rad | max |qvel|: {np.abs(data.qvel).max():.2e}")
    check(finite, "state stays finite")
    check(drift < 1e-2, "arm holds home pose within 0.01 rad")
    bad_qacc = data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number
    check(bad_qacc == 0, "no bad-acceleration warnings")

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"== Peak process memory: {peak_mb:.0f} MB")

    print(f"== {'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
