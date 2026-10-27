"""The physical ceiling for recovery: how often can every cube a knock throws off a tower still be
recovered, by any robot? (docs/v3_targets.md)

Usage: uv run python scripts/disturbance_ceiling.py [--n 60] [--seed 1]
Stands towers of 2-4 cubes (shoves on a middle cube: 3-4) on the three fixtures, applies each
kind of push --n times in random directions and strengths (polypanda.simulation.disturbances),
lets everything settle, and classifies the outcome (D1/D2/D3) and every fallen cube: upright,
lying on a side or leaning; within the arm's reach or not; touching another fallen cube. No robot
policy is involved; the arm stands at home. Writes results/disturbances/ceiling.json.
"""

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import mujoco
import numpy as np

from polypanda.simulation.disturbances import (
    PUSH_FORCE,
    PUSHES,
    SHOVE_SPEED,
    Knock,
    apply_knock,
    knock_outcome,
    stand_tower,
)
from polypanda.simulation.workcell import Workcell, WorkcellConfig, WorkOrder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60, help="knocks per kind of push")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("results/disturbances/ceiling.json"))
    args = parser.parse_args()

    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()

    git = {"commit": run("git", "rev-parse", "HEAD"),
           "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}
    cell = Workcell.build(WorkcellConfig(dressing=False))
    m, d = cell.model, cell.data
    order = WorkOrder((("red",) * 4,) * 3, tuple(("red", 0) for _ in range(12)))
    rng = np.random.default_rng(args.seed)

    def step():
        mujoco.mj_step(m, d, nstep=25)

    stats = defaultdict(lambda: {"knocks": 0, "recoverable": 0, "touching": 0,
                                 "poses": Counter(), "reachable": 0, "fallen": 0})
    by_push = Counter()
    for push in PUSHES:
        lo, hi = PUSH_FORCE if push == "push_top" else SHOVE_SPEED
        heights = (3, 4) if push == "shove_middle" else (2, 3, 4)
        for trial in range(args.n):
            cell.reset(order)
            tower = cell.cubes[:heights[trial % len(heights)]]
            stand_tower(cell, trial % 3, tower, rng)
            for _ in range(10):
                step()
            before = [d.xpos[c.body].copy() for c in tower]
            knock = Knock(push, float(rng.uniform(0, 2 * np.pi)), float(rng.uniform(lo, hi)))
            apply_knock(cell, tower, knock, step)
            out = knock_outcome(cell, tower, before)
            by_push[f"{push} -> {out['kind'] or 'nothing fell'}"] += 1
            if out["kind"] is None:
                continue
            s = stats[out["kind"]]
            s["knocks"] += 1
            s["recoverable"] += out["recoverable"]
            s["touching"] += out["touching"]
            s["poses"].update(x["pose"] for x in out["states"])
            s["reachable"] += sum(x["reachable"] for x in out["states"])
            s["fallen"] += len(out["states"])
    result = {"seed": args.seed, "knocks_per_push": args.n, "git": git,
              "outcomes": dict(by_push), "kinds": {}}
    for kind in ("D1", "D2", "D3"):
        s = stats[kind]
        n = s["knocks"]
        if not n:
            continue
        result["kinds"][kind] = {
            "knocks": n, "recoverable": s["recoverable"] / n, "touching": s["touching"] / n,
            "fallen_cubes": s["fallen"], "reachable_cubes": s["reachable"] / s["fallen"],
            "poses": dict(s["poses"])}
        print(f"{kind}: {n} knocks; every fallen cube recoverable in {s['recoverable'] / n:.0%}; "
              f"fallen cubes {dict(s['poses'])}, {s['reachable'] / s['fallen']:.0%} within reach; "
              f"fallen cubes touching each other in {s['touching'] / n:.0%}")
    print("push -> outcome:", dict(by_push))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
