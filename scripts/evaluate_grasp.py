"""Evaluate a grasp controller on the held-out placement set and archive the results.

Usage: uv run python scripts/evaluate_grasp.py [--set eval_sets/grasp_nominal_v1.json]
       uv run python scripts/evaluate_grasp.py --make-set   # (re)generate the placement set
Writes results/grasp/scripted_<set>.json (summary + provenance) and _episodes.csv.
"""

import argparse
import csv
import json
import platform
import subprocess
from collections import Counter
from dataclasses import asdict, fields
from datetime import UTC, datetime
from pathlib import Path

import mujoco
import numpy as np

from polypanda.control.scripted_grasp import ScriptedGraspConfig
from polypanda.evaluation.grasp import GraspEpisodeResult, evaluate_scripted_grasp
from polypanda.evaluation.sets import GraspEvalSet, generate_grasp_eval_set

DEFAULT_SET = Path("eval_sets/grasp_nominal_v1.json")


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", type=Path, default=DEFAULT_SET)
    parser.add_argument("--make-set", action="store_true")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--out-dir", type=Path, default=Path("results/grasp"))
    args = parser.parse_args()

    if args.make_set:
        eval_set = generate_grasp_eval_set(args.set.stem, args.n, args.seed)
        eval_set.save(args.set)
        print(f"wrote {args.set}: {len(eval_set)} placements, digest {eval_set.digest}, "
              f"regions {dict(Counter(eval_set.regions))}")
        return

    eval_set = GraspEvalSet.load(args.set)
    config = ScriptedGraspConfig()
    summary, episodes = evaluate_scripted_grasp(eval_set, config)
    name = f"scripted_{eval_set.name}"
    record = {
        "name": name,
        "policy": "scripted top-down grasp (PoseIK, privileged cube pose)",
        "controller_config": asdict(config),
        "eval_set": {"name": eval_set.name, "n": len(eval_set), "digest": eval_set.digest},
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(),
        "versions": {"python": platform.python_version(), "mujoco": mujoco.__version__,
                     "numpy": np.__version__},
        "summary": summary,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    with open(args.out_dir / f"{name}_episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[fld.name for fld in fields(GraspEpisodeResult)])
        writer.writeheader()
        writer.writerows(asdict(e) for e in episodes)

    s = summary
    lo, hi = s["success_ci95"]
    print(f"{name}: {len(eval_set)} placements, {s['wall_clock_s']} s wall")
    print(f"  success {s['success_rate']:.1%} (95% CI {lo:.1%}-{hi:.1%}); failures {s['failures']}")
    print(f"  first grasp median {s['first_grasp_s']['median']:.2f} s, p90 "
          f"{s['first_grasp_s']['p90']:.2f} s; episode median {s['episode_s_median']:.2f} s; "
          f"phases {s['phase_s_median']}")
    print(f"  nudge before closing median {s['nudge_mm']['median']:.2f} mm, max "
          f"{s['nudge_mm']['max']:.2f} mm; tilt median {s['max_tilt_deg']['median']:.1f} deg, "
          f"p95 {s['max_tilt_deg']['p95']:.1f}, max {s['max_tilt_deg']['max']:.1f}")
    print(f"  lifted {s['final_lift_cm_median']:.1f} cm, "
          f"finger force {s['finger_force_n_median']:.2f} N each, "
          f"floor-contact episodes {s['floor_contact_episode_rate']:.1%}")
    for region, r in s["by_region"].items():
        print(f"  {region:5s} n={r['n']:3d} success {r['success_rate']:.1%}")
    print(f"  wrote {args.out_dir / name}.json and _episodes.csv")


if __name__ == "__main__":
    main()
