"""v4.0: generate the door skill's evaluation sets (docs/v4_targets.md), before any training.

Each episode is a full DoorEnv.draw_episode record (command, machine variant, lid start, arm jitter,
camera errors), replayed exactly with env.reset(options={"episode": ep}). Half the episodes are
`open`, half `close`, alternating.

Usage:
  uv run python scripts/make_door_sets.py --name door_dev_v1 --seed 20261124
  uv run python scripts/make_door_sets.py --name door_final_v1 --seed 20261125
  uv run python scripts/make_door_sets.py --name door_stretch_v1 --seed 20261126 --stretch
"""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from polypanda.environments.door import COMMANDS, DoorConfig, DoorEnv

# beyond the training ranges (docs/v4_targets.md). Outward lids keep their handle ranges: their
# handle already ends near the arm's reach.
STRETCH = {
    "handle_offset_side": 0.04,
    "handle_shift_side": 0.03,
    "turn": 15.0,
    "latch_speed": (0.01, 0.15),
    "spring": (10.0, 50.0),
    "friction": (0.2, 15.0),
}


def make_set(n: int, seed: int, stretch: bool) -> tuple[list[dict], dict]:
    cfg = DoorConfig()
    if stretch:
        cfg = replace(cfg, variation=replace(cfg.variation, **STRETCH))
    env = DoorEnv(cfg)
    rng = np.random.default_rng(seed)
    episodes = [env.draw_episode(rng, COMMANDS[k % 2]) for k in range(n)]
    return episodes, asdict(cfg.variation)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--stretch", action="store_true")
    p.add_argument("--out-dir", type=Path, default=Path("eval_sets"))
    args = p.parse_args()
    episodes, variation = make_set(args.n, args.seed, args.stretch)
    body = json.dumps(episodes, sort_keys=True)
    digest = hashlib.sha256(body.encode()).hexdigest()[:12]
    out = {"name": args.name, "seed": args.seed, "n": args.n, "stretch": args.stretch,
           "variation": variation, "digest": digest, "episodes": episodes}
    path = args.out_dir / f"{args.name}.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    springy = sum(e["spring"] > 0 for e in episodes)
    slides = {s: sum(e["slide"] == s for e in episodes) for s in ("right", "left", "out")}
    print(f"{path}: {args.n} episodes, digest {digest}, springy {springy}, slides {slides}")


if __name__ == "__main__":
    main()
