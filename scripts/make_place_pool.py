"""Build transport-to-place handoff pools: cubes carried by the scripted carry to above a target.

Usage: uv run python scripts/make_place_pool.py
For every transport handoff state (training pool and held-out set), a place target is drawn
(x 0.35-0.65 m, y +/-0.25 m, at least 10 cm from where the cube was originally grasped) and the
scripted carry brings the cube to 12 cm above it. When transport has been successful for 10
consecutive steps (the supervisor's switch point) the state is captured. Sources keep the grasp
controller label, so all four grasp controllers are represented. Writes
data/handoff/place_train_v1.npz and eval_sets/place_nominal_v1.npz (goals = pre-place point).
"""

import multiprocessing as mp
import platform
import subprocess
from pathlib import Path

import numpy as np

PRE_PLACE_HEIGHT = 0.12  # m, cube-centre height above the target before placing
MIN_TARGET_DISTANCE = 0.10
KEYS = {"train": 0x5A11, "eval": 0x5A12}


def _worker(args):
    pool_path, indices, key, seed, chunk = args
    import torch

    from polypanda.environments.handoff import HandoffPool, capture_state
    from polypanda.environments.transport import TransportEnv
    from polypanda.evaluation.transport import ScriptedCarryDriver
    from polypanda.simulation.scenes import CUBE_X_RANGE, CUBE_Y_RANGE
    torch.set_num_threads(1)
    pool, env, driver = HandoffPool.load(pool_path), TransportEnv(), ScriptedCarryDriver()
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(key, chunk)))
    out = []
    for i in indices:
        origin = pool.placements[i][:2]
        while True:
            xy = np.array([rng.uniform(*CUBE_X_RANGE), rng.uniform(*CUBE_Y_RANGE)])
            if np.linalg.norm(xy - origin) >= MIN_TARGET_DISTANCE:
                break
        goal = np.array([*xy, PRE_PLACE_HEIGHT])
        obs, _ = env.reset(options={"state": pool.states[i], "goal": goal})
        driver.reset(env)
        streak, state = 0, None
        for _ in range(env.config.episode.horizon):
            obs, _, terminated, truncated, info = env.step(driver(env, obs))
            streak = streak + 1 if info["is_success"] else 0
            if streak >= 10:
                state = capture_state(env.model, env.data)
                break
            if terminated or truncated:
                break
        out.append((i, state, goal))
    return out


def main() -> None:
    import mujoco

    from polypanda.environments.handoff import STATE_SPEC, HandoffPool
    from polypanda.simulation.assets import MENAGERIE_COMMIT
    from polypanda.simulation.scenes import load_pick_scene

    model, _ = load_pick_scene()
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    seed = 20261024
    for name, src, dst in (("train", "data/handoff/transport_train_v1.npz",
                            "data/handoff/place_train_v1.npz"),
                           ("eval", "eval_sets/transport_nominal_v1.npz",
                            "eval_sets/place_nominal_v1.npz")):
        source = HandoffPool.load(src)
        chunks = np.array_split(np.arange(len(source)), 6)
        jobs = [(src, list(c), KEYS[name], seed, k) for k, c in enumerate(chunks)]
        with mp.get_context("spawn").Pool(6) as p:
            results = [r for part in p.map(_worker, jobs) for r in part]
        kept = [(i, s, g) for i, s, g in results if s is not None]
        idx = [i for i, _, _ in kept]
        pool = HandoffPool(
            states=np.array([s for _, s, _ in kept]),
            sources=tuple(source.sources[i] for i in idx),
            placements=source.placements[idx],
            goals=np.array([g for _, _, g in kept]),
            meta={"menagerie_commit": MENAGERIE_COMMIT, "mujoco": mujoco.__version__,
                  "python": platform.python_version(), "git_commit": git,
                  "state_size": mujoco.mj_stateSize(model, STATE_SPEC), "pool": name,
                  "seed": seed, "spawn_key": KEYS[name], "from": src,
                  "pre_place_height": PRE_PLACE_HEIGHT, "transport_attempts": len(results),
                  "transport_successes": len(kept)})
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        pool.save(dst)
        per_source = {s: (sum(source.sources[i] == s for i in idx),
                          sum(source.sources[i] == s for i, _, _ in results))
                      for s in sorted(set(source.sources))}
        print(f"{name}: {len(pool)} / {len(results)} transport carries reached the pre-place "
              f"point -> {dst} (digest {pool.digest})")
        for s, (k, n) in per_source.items():
            print(f"   {s}: {k}/{n}")


if __name__ == "__main__":
    main()
