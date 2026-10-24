"""Run the robustness suite: full pick-and-place under each perturbation condition.

Usage:
  uv run python scripts/evaluate_robustness.py --grasp scripted                  # all conditions
  uv run python scripts/evaluate_robustness.py --grasp run:runs/<run_dir> --conditions push mass
Conditions are defined in polypanda.evaluation.robustness. Scenes are the held-out pick-and-place
set; perturbations are drawn per scene from a fixed stream, identical for every grasp skill.
Writes results/robustness/<tag>_<condition>.json (summary + provenance) and _episodes.csv.
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
    grasp_driver_from_spec,
    grasp_env_for,
)
from polypanda.evaluation.robustness import CONDITIONS, RobustEpisode, run_condition
from polypanda.evaluation.sets import PickPlaceEvalSet
from polypanda.evaluation.transport import ScriptedCarryDriver
from polypanda.supervisor.fsm import SupervisorConfig


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grasp", default="scripted")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument("--set", type=Path, default=Path("eval_sets/pickplace_nominal_v1.json"))
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--out-dir", type=Path, default=Path("results/robustness"))
    args = parser.parse_args()
    torch.set_num_threads(1)

    eval_set = PickPlaceEvalSet.load(args.set)
    driver, description, tag, provenance = grasp_driver_from_spec(args.grasp)
    config = SupervisorConfig()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for condition in args.conditions:
        summary, results = run_condition(driver, ScriptedCarryDriver(), ScriptedPlace(),
                                         eval_set, condition, args.seed, config,
                                         grasp_env_for(args.grasp))
        name = f"{tag}_{condition}"
        record = {"name": name, "condition": condition,
                  "skills": {"grasp": description, "grasp_provenance": provenance,
                             "transport": "scripted carry", "place": "scripted place"},
                  "supervisor": asdict(config),
                  "eval_set": {"name": eval_set.name, "n": len(eval_set),
                               "digest": eval_set.digest},
                  "perturbation_seed": args.seed,
                  "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                  "git": git_state(),
                  "versions": {"python": platform.python_version(),
                               "mujoco": mujoco.__version__, "numpy": np.__version__,
                               "torch": torch.__version__},
                  "summary": summary}
        (args.out_dir / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
        columns = [f.name for f in fields(RobustEpisode) if f.name != "episode"]
        with open(args.out_dir / f"{name}_episodes.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns
                                    + [f.name for f in fields(PickPlaceEpisode)])
            writer.writeheader()
            for r in results:
                row = {c: getattr(r, c) for c in columns}
                writer.writerow({**row, **asdict(r.episode)})
        s = summary
        lo, hi = s["success_ci95"]
        print(f"{name}: success {s['success_rate']:.1%} ({lo:.1%}-{hi:.1%}), "
              f"first attempt {s['first_attempt_success_rate']:.1%}, "
              f"failed at {s['failures_by_stage']}, {s['wall_clock_s']} s")
        for level, v in s["by_level"].items():
            if v["n"]:
                print(f"    {level:12s} n={v['n']:3d} success {v['success_rate']:.1%}")


if __name__ == "__main__":
    main()
