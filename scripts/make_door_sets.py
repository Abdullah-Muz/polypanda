"""v4.0: generate the door skill's evaluation sets (docs/v4_targets.md), before any training.

Each episode is a full DoorEnv.draw_episode record (command, machine variant, lid start, arm jitter,
camera errors), replayed exactly with env.reset(options={"episode": ep}). Half the episodes are
`open`, half `close`, alternating.

Usage:
  uv run python scripts/make_door_sets.py --name door_dev_v1 --seed 20261124
  uv run python scripts/make_door_sets.py --name door_final_v1 --seed 20261125
  uv run python scripts/make_door_sets.py --name door_stretch_v1 --seed 20261126 --stretch
v4.1 (development only): drawers, `--kinds drawer` (or `--kinds lid drawer`), e.g. the 50
drawers the scripts were first measured on (results/v4/door/*_door_drawer_probe_v1.json):
  uv run python scripts/make_door_sets.py --name door_drawer_probe_v1 --seed 4101 --n 50 \\
    --kinds drawer
"""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from polypanda.environments.door import COMMANDS, DOOR_KINDS, DoorConfig, DoorEnv
from polypanda.simulation.machine import ALL_SLIDES

# DoorVariationConfig's v4.1 fields (tester kinds; drawers)
TESTER_FIELDS = ("kinds", "travel_drawer", "drawer_spring_prob", "handle_offset_drawer",
                 "handle_shift_drawer", "handle_height_drawer")

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


def make_set(n: int, seed: int, stretch: bool, kinds=("lid",)) -> tuple[list[dict], dict]:
    cfg = DoorConfig()
    if stretch:
        cfg = replace(cfg, variation=replace(cfg.variation, **STRETCH))
    cfg = replace(cfg, variation=replace(cfg.variation, kinds=tuple(kinds)))
    env = DoorEnv(cfg)
    rng = np.random.default_rng(seed)
    episodes = [env.draw_episode(rng, COMMANDS[k % 2]) for k in range(n)]
    variation = asdict(cfg.variation)
    if tuple(kinds) == ("lid",):    # v4.0's sets record v4.0's fields only (byte-identical files)
        variation = {k: v for k, v in variation.items() if k not in TESTER_FIELDS}
    return episodes, variation


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--stretch", action="store_true")
    p.add_argument("--kinds", nargs="+", default=["lid"], choices=DOOR_KINDS,
                   help="tester kinds drawn (v4.1): lid (v4.0's sets), drawer, or both")
    p.add_argument("--out-dir", type=Path, default=Path("eval_sets"))
    args = p.parse_args()
    episodes, variation = make_set(args.n, args.seed, args.stretch, args.kinds)
    body = json.dumps(episodes, sort_keys=True)
    digest = hashlib.sha256(body.encode()).hexdigest()[:12]
    out = {"name": args.name, "seed": args.seed, "n": args.n, "stretch": args.stretch,
           "variation": variation, "digest": digest, "episodes": episodes}
    path = args.out_dir / f"{args.name}.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    springy = sum(e["spring"] > 0 for e in episodes)
    slides = {s: sum(e["slide"] == s for e in episodes) for s in ALL_SLIDES}
    print(f"{path}: {args.n} episodes, digest {digest}, springy {springy}, slides {slides}")


if __name__ == "__main__":
    main()
