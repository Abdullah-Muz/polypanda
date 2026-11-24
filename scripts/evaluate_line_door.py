"""v4.0 T5: the door actors inside the line's tending cycle, on machines of a door set.

Each actor runs the same cycle on the first --n machines of the set (evaluation/line_door.py):
drive to M1, open, pick, load, close; success = the machine running at the end.

Usage:
  uv run python scripts/evaluate_line_door.py --actor run:runs/door_v2_bc_ppo_r5_seed0@final \
      --set eval_sets/door_final_v1.json --n 100 --jobs 6
Writes results/v4/line/<actor>_<set>_n<n>.json: one row per machine and a summary.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np


def _chunk(args):
    from polypanda.evaluation.line_door import run_cycle
    actor, episodes, indices = args
    return [run_cycle(actor, i, episodes[i]) for i in indices]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--actor", required=True)
    p.add_argument("--set", type=Path, required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    run = lambda *c: subprocess.run(c, capture_output=True, text=True).stdout.strip()  # noqa: E731
    git = {"commit": run("git", "rev-parse", "HEAD"),
           "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}
    data = json.loads(args.set.read_text())
    episodes = data["episodes"][: args.n]
    jobs = [(args.actor, episodes, list(range(j, len(episodes), args.jobs)))
            for j in range(args.jobs)]
    start = time.perf_counter()
    with Pool(args.jobs) as pool:
        rows = sorted((r for rs in pool.map(_chunk, jobs) for r in rs), key=lambda r: r["index"])
    wall = time.perf_counter() - start
    k, n = sum(r["success"] for r in rows), len(rows)
    door = [r[m] for r in rows for m in ("open", "close") if r[m]]
    summary = {
        "n": n, "running": k, "rate": k / n, "wilson95": wilson(k, n),
        "failures": dict(Counter(r["failure"] for r in rows if not r["success"])),
        "errors": sum(r["error"] is not None for r in rows),
        "door_moves": {"opens_ok": sum(1 for r in rows if r["open"] and r["open"]["success"]),
                       "closes_ok": sum(1 for r in rows if r["close"] and r["close"]["success"]),
                       "median_steps": float(np.median([x["steps"] for x in door if x["success"]]))
                       if door else None,
                       "peak_force_median": float(np.median([x["peak_force"] for x in door]))
                       if door else None,
                       "slams": sum(x["slams"] > 0 for x in door)},
        "cycle_s_median": float(np.median([r["cycle_s"] for r in rows])),
    }
    name = args.actor.replace("run:", "").replace("/", "_").replace("@", "_")
    out = args.out or Path("results/v4/line") / f"{name}_{data['name']}_n{n}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"actor": args.actor, "set": data["name"],
                               "set_digest": data["digest"], "git": git, "wall_s": round(wall, 1),
                               "summary": summary, "rows": rows}, indent=1))
    lo, hi = summary["wilson95"]
    print(f"{args.actor} on {data['name']} (first {n}): machine running {k}/{n} "
          f"({100 * k / n:.0f} %, 95 % CI {100 * lo:.0f}-{100 * hi:.0f}); failures "
          f"{summary['failures']}; door moves {summary['door_moves']}; median cycle "
          f"{summary['cycle_s_median']:.1f} s")
    print(f"   wrote {out} ({wall:.0f} s)")


if __name__ == "__main__":
    main()
