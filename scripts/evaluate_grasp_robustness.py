"""Run the grasp robustness suite: the grasp alone, on held-out grasp sets, under perturbation.

Usage:
  uv run python scripts/evaluate_grasp_robustness.py --grasp run:runs/<run_dir>
  uv run python scripts/evaluate_grasp_robustness.py --grasp scripted-v4 --conditions mass size
Conditions and their perturbation streams are defined in polypanda.evaluation.robustness (paired:
the same draws for every grasp skill). By default both any-pose sets, lying and upright.
Writes results/grasp_robustness/<tag>_<set>_<condition>.json (summary + provenance) and
_episodes.csv.
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

from polypanda.evaluation.grasp import GraspPolicyEpisode
from polypanda.evaluation.pickplace import grasp_driver_from_spec, grasp_env_for
from polypanda.evaluation.robustness import (
    GRASP_CONDITIONS,
    GraspRobustEpisode,
    run_grasp_condition,
)
from polypanda.evaluation.sets import GraspEvalSet

SETS = [Path("eval_sets/grasp_lying_v1.json"), Path("eval_sets/grasp_upright_v1.json")]


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grasp", default="scripted-v4")
    parser.add_argument("--sets", type=Path, nargs="+", default=SETS)
    parser.add_argument("--conditions", nargs="+", default=list(GRASP_CONDITIONS),
                        choices=GRASP_CONDITIONS)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--out-dir", type=Path, default=Path("results/grasp_robustness"))
    args = parser.parse_args()
    torch.set_num_threads(1)

    driver, description, tag, provenance = grasp_driver_from_spec(args.grasp)
    env = grasp_env_for(args.grasp)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for set_path in args.sets:
        eval_set = GraspEvalSet.load(set_path)
        for condition in args.conditions:
            summary, results = run_grasp_condition(driver, eval_set, condition, args.seed, env)
            name = f"{tag}_{eval_set.name}_{condition}"
            record = {"name": name, "condition": condition,
                      "grasp": description, "grasp_provenance": provenance,
                      "protocol": f"{type(env).__name__}, 15 s horizon",
                      "gripper": asdict(env.config.gripper),
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
            columns = [f.name for f in fields(GraspRobustEpisode) if f.name != "episode"]
            with open(args.out_dir / f"{name}_episodes.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns
                                        + [f.name for f in fields(GraspPolicyEpisode)])
                writer.writeheader()
                for r in results:
                    writer.writerow({**{c: getattr(r, c) for c in columns}, **asdict(r.episode)})
            s = summary
            lo, hi = s["success_ci95"]
            print(f"{name}: success {s['success_rate']:.1%} ({lo:.1%}-{hi:.1%}), "
                  f"failures {s['failures']}, {s['wall_clock_s']} s", flush=True)
            for level, v in s["by_level"].items():
                if v["n"]:
                    print(f"    {level:12s} n={v['n']:3d} success {v['success_rate']:.1%}")


if __name__ == "__main__":
    main()
