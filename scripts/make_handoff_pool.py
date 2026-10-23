"""Build grasp-to-transport handoff pools from several grasp controllers.

Usage: uv run python scripts/make_handoff_pool.py --sources scripted \
           runs/grasp_ppo_curr_seed0 runs/grasp_ppo_curr_seed1 runs/grasp_ppo_curr_seed2
Writes data/handoff/transport_train_v1.npz (training starts) and
eval_sets/transport_nominal_v1.npz (held-out starts with goals), each with equal shares per
source. Placements come from SeedSequence streams with dedicated spawn keys per pool and source.
The files depend on trained grasp checkpoints (not in git), so they are committed as artifacts.
"""

import argparse
import multiprocessing as mp
import platform
import subprocess
from pathlib import Path

import numpy as np

TRAIN_KEY, EVAL_KEY = 0x4A11, 0x4A12


def _driver(source: str):
    from polypanda.evaluation.grasp import PolicyDriver, ScriptedDriver
    from polypanda.policies.loader import load_run
    if source == "scripted":
        return ScriptedDriver(), {"source": "scripted"}
    policy = load_run(source)
    meta = policy.metadata
    return PolicyDriver(policy), {"source": source, "policy_id": meta["policy_id"],
                                  "checkpoint": policy.which, "trained_at": meta["git"]["commit"]}


def _worker(args):
    source, index, n, key, seed = args
    import torch

    from polypanda.environments.grasp import GraspEnv
    from polypanda.environments.handoff import capture_handoff
    from polypanda.simulation.scenes import CUBE_X_RANGE, CUBE_Y_RANGE
    torch.set_num_threads(1)
    env = GraspEnv()
    driver, info = _driver(source)
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(key, index)))
    states, placements, attempts = [], [], 0
    while len(states) < n:
        placement = np.array([rng.uniform(*CUBE_X_RANGE), rng.uniform(*CUBE_Y_RANGE),
                              rng.uniform(-np.pi, np.pi)])
        attempts += 1
        state = capture_handoff(env, driver, placement)
        if state is not None:
            states.append(state)
            placements.append(placement)
    info.update(attempts=attempts, captured=n, capture_rate=round(n / attempts, 3))
    return np.array(states), np.array(placements), info


def build(sources, n_per_source, key, seed, workers):
    jobs = [(s, i, n_per_source, key, seed) for i, s in enumerate(sources)]
    with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
        results = pool.map(_worker, jobs)
    states = np.concatenate([r[0] for r in results])
    placements = np.concatenate([r[1] for r in results])
    labels = tuple(info["source"] for (_, _, info) in results for _ in range(n_per_source))
    return states, placements, labels, [r[2] for r in results]


def main() -> None:
    import mujoco

    from polypanda.environments.handoff import STATE_SPEC, HandoffPool
    from polypanda.environments.transport import GoalConfig, sample_goal
    from polypanda.simulation.assets import MENAGERIE_COMMIT
    from polypanda.simulation.scenes import CubeIndices, cube_position, load_pick_scene

    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--train-per-source", type=int, default=300)
    parser.add_argument("--eval-per-source", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train-out", type=Path,
                        default=Path("data/handoff/transport_train_v1.npz"))
    parser.add_argument("--eval-out", type=Path, default=Path("eval_sets/transport_nominal_v1.npz"))
    args = parser.parse_args()

    model, data = load_pick_scene()
    cube = CubeIndices.from_model(model)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    base_meta = {"menagerie_commit": MENAGERIE_COMMIT, "mujoco": mujoco.__version__,
                 "python": platform.python_version(), "git_commit": git,
                 "state_size": mujoco.mj_stateSize(model, STATE_SPEC), "stable_steps": 10,
                 "seed": args.seed}

    for name, n, key, out in (("train", args.train_per_source, TRAIN_KEY, args.train_out),
                              ("eval", args.eval_per_source, EVAL_KEY, args.eval_out)):
        states, placements, labels, infos = build(args.sources, n, key, args.seed, args.workers)
        goals = None
        if name == "eval":
            from polypanda.environments.handoff import restore_state
            rng = np.random.default_rng(np.random.SeedSequence(args.seed, spawn_key=(key, 99)))
            goals = []
            for s in states:
                restore_state(model, data, s)
                goals.append(sample_goal(rng, cube_position(data, cube), GoalConfig()))
            goals = np.array(goals)
        pool = HandoffPool(states, labels, placements, goals,
                           {**base_meta, "pool": name, "spawn_key": key, "sources": infos})
        pool.save(out)
        print(f"{name}: {len(pool)} states -> {out} (digest {pool.digest})")
        for info in infos:
            print(f"   {info['source']}: capture rate {info['capture_rate']:.1%} "
                  f"({info['captured']}/{info['attempts']} grasp attempts)")


if __name__ == "__main__":
    main()
