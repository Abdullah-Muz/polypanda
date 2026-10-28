"""Evaluate a stand-up controller (v3 step d2) on a held-out set of starts and archive the results.

Usage:
  uv run python scripts/evaluate_standup.py --make-set --set eval_sets/standup_v1.json --seed S
  uv run python scripts/evaluate_standup.py --policy scripted --set eval_sets/standup_v1.json
  uv run python scripts/evaluate_standup.py --policy run:runs/<run_dir>[@final] --set ...
Writes results/standup/<name>.json (summary + provenance) and <name>_episodes.csv.
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

from polypanda.environments.standup import StandUpConfig, StandUpEnv
from polypanda.evaluation.grasp import PolicyDriver
from polypanda.evaluation.standup import (
    ScriptedStandUpDriver,
    StandUpEpisode,
    StandUpEvalSet,
    evaluate_standup,
    generate_standup_set,
)
from polypanda.policies.loader import load_run, make_run_env

TASK_CONFIG = Path("configs/standup_v1.yaml")


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", type=Path, required=True)
    parser.add_argument("--policy", help="scripted | run:DIR[@best|final]")
    parser.add_argument("--make-set", action="store_true")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out-dir", type=Path, default=Path("results/standup"))
    args = parser.parse_args()

    if args.make_set:
        if args.seed is None:
            raise SystemExit("--make-set needs --seed")
        env = StandUpEnv(StandUpConfig.from_yaml(TASK_CONFIG).for_evaluation())
        s = generate_standup_set(env, args.set.stem, args.n, args.seed)
        s.save(args.set)
        towers = sum(1 for e in s.episodes if e["tower"])
        print(f"wrote {args.set}: {len(s)} starts ({towers} with a tower), digest {s.digest}")
        return

    eval_set = StandUpEvalSet.load(args.set)
    if args.policy == "scripted":
        driver, env = ScriptedStandUpDriver(), StandUpEnv(
            StandUpConfig.from_yaml(TASK_CONFIG).for_evaluation())
        description, name = (f"scripted stand-up (6-D hand motions), task {TASK_CONFIG}",
                             f"scripted_{eval_set.name}")
    elif args.policy and args.policy.startswith("run:"):
        run_dir, _, which = args.policy[4:].partition("@")
        policy = load_run(run_dir, which or "best")
        meta = policy.metadata
        if meta["skill"] not in ("standup", "standup_v2"):
            raise SystemExit(f"{run_dir} is a {meta['skill']} policy, not standup")
        driver, env = PolicyDriver(policy), make_run_env(run_dir, meta["skill"])
        description = (f"{meta['algorithm']} {policy.which} model from {run_dir} "
                       f"(seed {meta['seed']}, {meta['total_timesteps']:,} steps)")
        name = f"{meta['policy_id']}_{policy.which}_{eval_set.name}"
    else:
        raise SystemExit("use --policy scripted or run:DIR[@which]")

    summary, episodes = evaluate_standup(driver, eval_set, env)
    record = {"name": name, "policy": description, "protocol": "StandUpEnv, 15 s horizon",
              "eval_set": {"name": eval_set.name, "n": len(eval_set), "digest": eval_set.digest},
              "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
              "git": git_state(),
              "versions": {"python": platform.python_version(), "mujoco": mujoco.__version__,
                           "numpy": np.__version__},
              "summary": summary}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{name}.json").write_text(json.dumps(record, indent=2, default=float) + "\n")
    with open(args.out_dir / f"{name}_episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[fld.name for fld in fields(StandUpEpisode)])
        writer.writeheader()
        writer.writerows(asdict(e) for e in episodes)
    s = summary
    lo, hi = s["success_ci95"]
    print(f"{name}: success {s['success_rate']:.1%} (95% CI {lo:.1%}-{hi:.1%}); "
          f"failures {s['failures']}; by tower {s['by_tower']}")


if __name__ == "__main__":
    main()
