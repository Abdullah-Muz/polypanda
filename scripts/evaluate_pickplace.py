"""Evaluate full pick-and-place through the supervisor on the held-out scene set.

Usage:
  uv run python scripts/evaluate_pickplace.py --make-set               # (re)generate the scenes
  uv run python scripts/evaluate_pickplace.py --grasp scripted         # all-scripted
  uv run python scripts/evaluate_pickplace.py --grasp run:runs/<run_dir>[@final]
  ... --compare results/pickplace/<other>_episodes.csv                 # paired McNemar test
Transport is always the scripted carry and place the scripted place (the skills each stage's
evaluation selected). Writes results/pickplace/<name>.json (summary + provenance) and
<name>_episodes.csv.
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

from polypanda.control.scripted_place import ScriptedPlace
from polypanda.evaluation.pickplace import (
    PickPlaceEpisode,
    evaluate_pickplace,
    grasp_driver_from_spec,
    grasp_env_for,
    mcnemar_exact,
)
from polypanda.evaluation.sets import PickPlaceEvalSet, generate_pickplace_eval_set
from polypanda.evaluation.transport import ScriptedCarryDriver
from polypanda.supervisor.fsm import SupervisorConfig

DEFAULT_SET = Path("eval_sets/pickplace_nominal_v1.json")


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grasp", default="scripted")
    parser.add_argument("--set", type=Path, default=DEFAULT_SET)
    parser.add_argument("--make-set", action="store_true")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--compare", type=Path, help="episodes CSV of another configuration")
    parser.add_argument("--out-dir", type=Path, default=Path("results/pickplace"))
    args = parser.parse_args()
    torch.set_num_threads(1)

    if args.make_set:
        eval_set = generate_pickplace_eval_set(args.set.stem, args.n, args.seed)
        eval_set.save(args.set)
        d = np.linalg.norm(eval_set.targets - eval_set.placements[:, :2], axis=1)
        print(f"wrote {args.set}: {len(eval_set)} scenes, digest {eval_set.digest}; "
              f"start-to-target median {np.median(d):.3f} m (min {d.min():.3f}, "
              f"max {d.max():.3f})")
        return

    eval_set = PickPlaceEvalSet.load(args.set)
    driver, description, tag, provenance = grasp_driver_from_spec(args.grasp)
    config = SupervisorConfig()
    name = f"pickplace_{tag}_{eval_set.name}"
    summary, episodes = evaluate_pickplace(driver, ScriptedCarryDriver(), ScriptedPlace(),
                                           eval_set, config, grasp_env_for(args.grasp))
    record = {"name": name,
              "skills": {"grasp": description, "grasp_provenance": provenance,
                         "transport": "scripted carry (closed-loop IK, gain 0.2)",
                         "place": "scripted place (upright correction, gated release)"},
              "supervisor": asdict(config),
              "eval_set": {"name": eval_set.name, "n": len(eval_set), "digest": eval_set.digest},
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
        writer = csv.DictWriter(f, fieldnames=[fld.name for fld in fields(PickPlaceEpisode)])
        writer.writeheader()
        writer.writerows(asdict(e) for e in episodes)

    s = summary
    lo, hi = s["success_ci95"]
    print(f"{name}: {len(eval_set)} scenes, {s['wall_clock_s']} s wall")
    print(f"  pick-and-place success {s['success_rate']:.1%} (95% CI {lo:.1%}-{hi:.1%}); "
          f"first attempt {s['first_attempt_success_rate']:.1%}")
    print(f"  failures by stage {s['failures_by_stage']}; reasons {s['failures']}")
    print(f"  first failures {s['first_failures']}")
    print(f"  retries {s['retries']}")
    t = s["time_s"]
    if t["median"] is not None:
        print(f"  time to verified placement median {t['median']:.2f} s (p90 {t['p90']:.2f}); "
              f"by stage {t['by_stage_median']}")
        print(f"  final xy error median {s['final_xy_error_mm']['median']:.1f} mm "
              f"(p95 {s['final_xy_error_mm']['p95']:.1f})")
    if args.compare:
        c = record["paired_comparison"]
        print(f"  vs {args.compare.name}: only this {c['only_first']}, only other "
              f"{c['only_second']}, exact McNemar p = {c['p_value']:.3g}")
    print(f"  wrote {args.out_dir / name}.json and _episodes.csv")


if __name__ == "__main__":
    main()
