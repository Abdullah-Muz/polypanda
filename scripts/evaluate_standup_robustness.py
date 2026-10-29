"""Run the stand-up robustness suite: the stand-up alone, on a held-out stand-up set, under
perturbation (v3 step d2).

Usage:
  uv run python scripts/evaluate_standup_robustness.py --policy run:runs/<run_dir> --choose chooser
  uv run python scripts/evaluate_standup_robustness.py --policy scripted --conditions mass size
Conditions and their perturbation streams are defined in polypanda.evaluation.robustness (paired:
the same draws for every stand-up controller). Writes
results/standup_robustness/<name>_<set>_<condition>.json (summary + provenance) and _episodes.csv.
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

from polypanda.evaluation.robustness import (
    STANDUP_CONDITIONS,
    StandUpRobustEpisode,
    run_standup_condition,
)
from polypanda.evaluation.standup import (
    CommandedPolicyDriver,
    StandUpEpisode,
    StandUpEvalSet,
    standup_driver_from_spec,
)


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="scripted", help="scripted | run:DIR[@best|final]")
    parser.add_argument("--choose", default="critic", help="as in evaluate_standup.py")
    parser.add_argument("--set", type=Path, default=Path("eval_sets/standup_v1.json"))
    parser.add_argument("--conditions", nargs="+", default=list(STANDUP_CONDITIONS),
                        choices=STANDUP_CONDITIONS)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--out-dir", type=Path, default=Path("results/standup_robustness"))
    args = parser.parse_args()
    torch.set_num_threads(1)

    driver, env, description, prefix = standup_driver_from_spec(args.policy, args.choose)
    eval_set = StandUpEvalSet.load(args.set)
    suffix = (f"_choose-{args.choose}"
              if isinstance(driver, CommandedPolicyDriver) and args.choose != "critic" else "")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for condition in args.conditions:
        summary, results = run_standup_condition(driver, eval_set, condition, args.seed, env)
        name = f"{prefix}{eval_set.name}{suffix}_{condition}"
        record = {"name": name, "condition": condition, "policy": description,
                  "protocol": f"{type(env).__name__}, 15 s horizon",
                  "eval_set": {"name": eval_set.name, "n": len(eval_set),
                               "digest": eval_set.digest},
                  "perturbation_seed": args.seed,
                  "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                  "git": git_state(),
                  "versions": {"python": platform.python_version(),
                               "mujoco": mujoco.__version__, "numpy": np.__version__,
                               "torch": torch.__version__},
                  "summary": summary}
        (args.out_dir / f"{name}.json").write_text(json.dumps(record, indent=2, default=float)
                                                   + "\n")
        columns = [f.name for f in fields(StandUpRobustEpisode) if f.name != "episode"]
        with open(args.out_dir / f"{name}_episodes.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns
                                    + [f.name for f in fields(StandUpEpisode)])
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
