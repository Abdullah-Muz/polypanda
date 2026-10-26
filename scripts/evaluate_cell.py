"""Evaluate the Milestone 9 work-order cell on held-out work orders.

Usage:
  uv run python scripts/evaluate_cell.py                                  # scripted cell
  uv run python scripts/evaluate_cell.py --pick run:runs/<run_dir>[@final]  # learned pick
  ... --place run:runs/<run_dir>                                          # learned place
  ... --heights 5 6                                                       # stress test
  ... --compare results/cell/<other>_episodes.csv                         # paired McNemar test
The moves between stations are scripted in every configuration; --pick and --place choose a
scripted or learned skill (the learned ones run through station frames). Writes
results/cell/<name>.json (summary + provenance) and <name>_episodes.csv.
"""

import argparse
import csv
import json
import platform
import subprocess
from dataclasses import asdict, fields
from datetime import UTC, datetime
from pathlib import Path

import mujoco
import numpy as np
import torch

from polypanda.evaluation.cell import CellEpisode, CellEvalConfig, evaluate_cell, skill_from_spec
from polypanda.evaluation.pickplace import mcnemar_exact
from polypanda.simulation.workcell import WorkcellConfig


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pick", default="scripted")
    parser.add_argument("--place", default="scripted")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=CellEvalConfig.seed)
    parser.add_argument("--heights", type=int, nargs=2, default=CellEvalConfig.heights)
    parser.add_argument("--compare", type=Path, help="episodes CSV of another configuration")
    parser.add_argument("--out-dir", type=Path, default=Path("results/cell"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)

    config = CellEvalConfig(n=args.n, seed=args.seed, heights=tuple(args.heights))
    pick, pick_desc, pick_tag, pick_prov = skill_from_spec(args.pick, "pick")
    place, place_desc, place_tag, place_prov = skill_from_spec(args.place, "place")
    lo_h, hi_h = config.heights
    name = f"cell_P-{pick_tag}_L-{place_tag}_h{lo_h}-{hi_h}_n{config.n}"

    def progress(e: CellEpisode):
        if not args.quiet:
            print(f"order {e.index}: {'ok  ' if e.success else 'FAIL'} heights {e.heights} "
                  f"towers {e.towers_ok} {e.failure or ''}", flush=True)

    summary, episodes = evaluate_cell(pick, place, config, progress)
    record = {"name": name,
              "skills": {"pick": pick_desc, "pick_provenance": pick_prov,
                         "place": place_desc, "place_provenance": place_prov,
                         "moves": "scripted (base-only swings in the carry posture)"},
              "eval": asdict(config),
              "cell": asdict(WorkcellConfig()),
              "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
              "git": git_state(),
              "versions": {"python": platform.python_version(), "mujoco": mujoco.__version__,
                           "numpy": np.__version__, "torch": torch.__version__},
              "summary": summary}
    if args.compare:
        with open(args.compare) as f:
            other = [row["success"] == "True" for row in csv.DictReader(f)]
        record["paired_comparison"] = {"against": str(args.compare),
                                       **mcnemar_exact([e.success for e in episodes], other)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    with open(args.out_dir / f"{name}_episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[fld.name for fld in fields(CellEpisode)])
        writer.writeheader()
        writer.writerows(asdict(e) for e in episodes)

    s = summary
    lo, hi = s["success_ci95"]
    print(f"{name}: {s['orders']} orders, {s['cubes']} cubes, {s['wall_clock_s']} s wall")
    print(f"  orders complete {s['success_rate']:.1%} (95% CI {lo:.1%}-{hi:.1%}); towers "
          f"{s['tower_success_rate']:.1%} of {s['towers']}")
    print("  towers by height " + ", ".join(f"{h}: {v['success_rate']:.0%} of {v['towers']}"
                                            for h, v in s["tower_success_by_height"].items()))
    print(f"  failures {s['failures']}; IK failures {s['ik_failures']}; "
          f"{s['sim_s_per_cube']:.1f} s simulated per cube")
    for stage in ("learned_pick", "learned_place"):
        if s[stage]:
            k = s[stage]
            print(f"  {stage}: {k['success_rate']:.1%} of {k['attempts']}, median "
                  f"{k['median_steps']} steps, failures {k['failures']}"
                  + (f", by level {k['by_level']}" if "by_level" in k else ""))
    if args.compare:
        c = record["paired_comparison"]
        print(f"  vs {args.compare.name}: only this {c['only_first']}, only other "
              f"{c['only_second']}, exact McNemar p = {c['p_value']:.3g}")
    print(f"  wrote {args.out_dir / name}.json and _episodes.csv")


if __name__ == "__main__":
    main()
