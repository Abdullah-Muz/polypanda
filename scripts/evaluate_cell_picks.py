"""v3.2: cell-like pick checks for a grasp (docs/v3_2_targets.md, evaluation/cell_picks.py).

Usage:
  uv run python scripts/evaluate_cell_picks.py --make-set belt --n 200 --seed 20261210 \\
      --set eval_sets/grasp_belt_v1.json
  uv run python scripts/evaluate_cell_picks.py --set eval_sets/grasp_belt_v1.json \\
      --pick run:runs/grasp_anypose6_seed0 [--size-range] [--limit N]
  uv run python scripts/evaluate_cell_picks.py --set ... --pick scripted     # scripted baseline
Each case is staged in a fresh cell and picked with the cell's own pick (the learned grasp
through its station view, or the scripted pick). --size-range scales the test cube by a factor
drawn per case from 0.85-1.15 (its own stream from the set's seed; constant density), as the
robustness suite's size condition. Writes results/grasp/cellpicks_<set>_<pick>[_size].json
(summary by 10-deg rotation bin, provenance) and _episodes.csv.
"""

import argparse
import csv
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import mujoco
import numpy as np
import torch

from polypanda.control.scripted_cell import ScriptedCell
from polypanda.evaluation.cell import skill_from_spec
from polypanda.evaluation.cell_picks import PickSet, generate_pick_set, run_case, summarize
from polypanda.supervisor.cell import LearnedCell


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-set", choices=("belt", "floor"))
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--set", type=Path, required=True)
    parser.add_argument("--pick", default="scripted", help="'scripted' or run:<grasp run dir>")
    parser.add_argument("--size-range", action="store_true",
                        help="scale each test cube by a factor drawn from 0.85-1.15")
    parser.add_argument("--limit", type=int, help="only the first N cases (a smoke test; the "
                                                  "result is marked partial)")
    parser.add_argument("--out-dir", type=Path, default=Path("results/grasp"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.make_set:
        if args.seed is None:
            parser.error("--make-set needs --seed")
        if args.set.exists():
            parser.error(f"{args.set} exists; sets are never overwritten")
        s = generate_pick_set(args.set.stem, args.make_set, args.n, args.seed)
        s.save(args.set)
        print(f"wrote {args.set}: {len(s)} {s.kind} cases, digest {s.digest}")
        return

    torch.set_num_threads(1)
    git = git_state()   # recorded before evaluating, so later commits cannot mislabel the run
    pick_set = PickSet.load(args.set)
    cases = pick_set.cases[:args.limit] if args.limit else pick_set.cases
    sizes = (np.random.default_rng([pick_set.seed, 1]).uniform(0.85, 1.15, len(pick_set))
             if args.size_range else np.ones(len(pick_set)))
    grasp, grasp_desc, grasp_tag, grasp_prov = skill_from_spec(args.pick, "pick")

    def controller(cell):
        if grasp is None:
            return ScriptedCell(cell)
        return LearnedCell(cell, grasp=grasp, aim="fixture")

    rows, t0 = [], time.perf_counter()
    for i, case in enumerate(cases):
        r = run_case(case, controller, size=float(sizes[i]))
        r["index"] = i
        rows.append(r)
        if not args.quiet:
            print(f"case {i}: {'ok  ' if r['success'] else 'FAIL'} yaw_rel {r['yaw_rel']:+6.1f} "
                  f"size {r['size']:.3f} {r['failure'] or ''}", flush=True)
    summary = summarize(rows)
    name = (f"cellpicks_{pick_set.name}_{grasp_tag}{'_size' if args.size_range else ''}"
            f"{f'_first{args.limit}' if args.limit else ''}")
    record = {"name": name, "set": {"path": str(args.set), "name": pick_set.name,
                                    "kind": pick_set.kind, "seed": pick_set.seed,
                                    "digest": pick_set.digest, "n": len(pick_set)},
              "pick": grasp_desc, "pick_provenance": grasp_prov,
              "condition": "size 0.85-1.15" if args.size_range else "nominal",
              "partial": bool(args.limit), "summary": summary,
              "created_utc": datetime.now(UTC).isoformat(timespec="seconds"), "git": git,
              "versions": {"python": platform.python_version(), "mujoco": mujoco.__version__,
                           "torch": torch.__version__},
              "wall_s": round(time.perf_counter() - t0, 1)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    with open(args.out_dir / f"{name}_episodes.csv", "w", newline="") as f:
        cols = ["index", *[k for k in rows[0] if k != "index"]]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({**r, "towers": "-".join(map(str, r["towers"]))} for r in rows)
    print(f"{name}: {summary['success']}/{summary['n']} = {100 * summary['rate']:.1f} % "
          f"(knocked over {summary['knocked_over']}); failures {summary['failures']}")
    for b, s in summary["by_rotation"].items():
        print(f"  {b:>9s} deg: {s['success']:3d}/{s['n']:3d}  knocked over {s['knocked_over']}")


if __name__ == "__main__":
    main()
