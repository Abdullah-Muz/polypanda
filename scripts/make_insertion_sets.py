"""v4.1: generate the learned insertion's evaluation sets (docs/v4_1_targets.md), before any
training, and the training pick pool.

Each episode is a full InsertionEnv.draw_episode record (the tester and where it stands, its nest's
clearance and chamfer, the camera's error, the module's placement on the infeed) completed with
the grasp its pick leaves (InsertionEnv.simulate_pick: the cell's pick of that module for that
tester, simulated once here and stored, so the episode replays exactly with
env.reset(options={"episode": ep}) without the pick). The sets are drawn from their own streams:
np.random.SeedSequence(seed, spawn_key=(EVAL_SPAWN_KEY,)), which no env.reset(seed=n) (training)
and no other stream of the project can reproduce.

Usage (the three sets of docs/v4_1_targets.md):
  uv run python scripts/make_insertion_sets.py --name insertion_dev_v1 --seed 20261127 --n 400
  uv run python scripts/make_insertion_sets.py --name insertion_final_v1 --seed 20261128 --n 600
  uv run python scripts/make_insertion_sets.py --name insertion_stretch_v1 --seed 20261129 \\
      --n 400 --stretch
The training pick pool (training draws grasps from it: InsertionConfig.pick_pool), from its own
stream (spawn key PICK_SPAWN_KEY):
  uv run python scripts/make_insertion_sets.py --picks --name insertion_picks_train_v1 \\
      --seed 20261131 --n 3000 --out-dir data/handoff
"""

import argparse
import hashlib
import json
import time
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from polypanda.environments.insertion import STRETCH, InsertionConfig, InsertionEnv
from polypanda.evaluation.insertion import EVAL_SPAWN_KEY, PICK_SPAWN_KEY


def config(stretch: bool) -> InsertionConfig:
    cfg = InsertionConfig()
    if stretch:
        cfg = replace(cfg, variation=replace(cfg.variation, **STRETCH))
    return cfg


def _complete(args) -> list[tuple[int, dict]]:
    import torch
    torch.set_num_threads(1)
    stretch, items = args
    env = InsertionEnv(config(stretch))
    return [(k, env.complete_episode(ep)) for k, ep in items]


def make_episodes(n: int, seed: int, stretch: bool, key: int, jobs: int) -> list[dict]:
    """n episodes drawn from the stream (seed, key), each completed with its pick's grasp."""
    env = InsertionEnv(config(stretch))
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(key,)))
    drawn = [env.draw_episode(rng) for _ in range(n)]
    chunks = [(stretch, [(k, drawn[k]) for k in range(j, n, jobs)]) for j in range(jobs)]
    with Pool(jobs) as pool:
        done = dict(kv for part in pool.map(_complete, chunks) for kv in part)
    return [done[k] for k in range(n)]


def digest_of(episodes: list[dict]) -> str:
    return hashlib.sha256(json.dumps(episodes, sort_keys=True).encode()).hexdigest()[:12]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--stretch", action="store_true",
                   help="insertion_stretch_v1's camera errors (sigma 1.5 mm, clip 4 mm; 1 deg, "
                        "clip 3 deg)")
    p.add_argument("--picks", action="store_true",
                   help="a training pick pool (stream PICK_SPAWN_KEY) instead of a set")
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--out-dir", type=Path, default=Path("eval_sets"))
    args = p.parse_args()
    key = PICK_SPAWN_KEY if args.picks else EVAL_SPAWN_KEY
    start = time.perf_counter()
    episodes = make_episodes(args.n, args.seed, args.stretch, key, args.jobs)
    wall = time.perf_counter() - start
    cfg = config(args.stretch)
    digest = digest_of(episodes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.name}.json"
    if args.picks:
        picks = [{"tester": e["tester"], "pick": e["pick"], "grasp": e["grasp"]}
                 for e in episodes]
        out = {"name": args.name, "seed": args.seed, "spawn_key": key, "n": args.n,
               "what": "real picks for training (InsertionConfig.pick_pool)",
               "digest": digest_of(picks), "picks": picks}
    else:
        out = {"name": args.name, "seed": args.seed, "spawn_key": key, "n": args.n,
               "stretch": args.stretch, "variation": asdict(cfg.variation), "digest": digest,
               "episodes": episodes}
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    testers = {t: sum(e["tester"] == t for e in episodes) for t in cfg.variation.testers}
    cam = np.array([np.hypot(*e["camera_xy"]) for e in episodes]) * 1000
    yaw = np.degrees(np.abs([e["camera_yaw"] for e in episodes]))
    rel = np.array([e["grasp"]["rel_pos"] for e in episodes]) * 1000
    print(f"{path}: {args.n} episodes, digest {out['digest']}, testers {testers}, camera error "
          f"mean {cam.mean():.2f} mm (max {cam.max():.2f}), yaw mean {yaw.mean():.2f} deg "
          f"(max {yaw.max():.2f}); grasp offset in the hand x/y sd "
          f"{rel[:, 0].std():.3f}/{rel[:, 1].std():.3f} mm ({wall:.0f} s)")


if __name__ == "__main__":
    main()
