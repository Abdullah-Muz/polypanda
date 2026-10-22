"""Evaluate a reach policy on a held-out set and archive the results.

Usage:
  uv run python scripts/evaluate.py --policy ik
  uv run python scripts/evaluate.py --policy random
  uv run python scripts/evaluate.py --policy ppo:path/to/model.zip --name reach_ppo_seed42

Writes results/reach/<name>.json (summary + provenance) and <name>_episodes.csv.
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
import stable_baselines3
import torch

from polypanda.control.classical_baseline import JacobianIKPolicy, RandomPolicy
from polypanda.evaluation.reach import EpisodeResult, evaluate
from polypanda.evaluation.sets import ReachEvalSet


def load_policy(spec: str):
    if spec == "random":
        return RandomPolicy(seed=0), "uniform random actions, seed 0"
    if spec == "ik":
        policy = JacobianIKPolicy()
        return policy, f"Jacobian DLS IK, gain {policy.gain}, damping {policy.damping}"
    algo, _, path = spec.partition(":")
    algos = {"ppo": stable_baselines3.PPO, "sac": stable_baselines3.SAC,
             "td3": stable_baselines3.TD3}
    if algo not in algos or not path:
        raise SystemExit(f"unknown policy spec {spec!r}; use random, ik or ppo|sac|td3:PATH")
    return algos[algo].load(path, device="cpu"), f"{algo.upper()} checkpoint {path}"


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--set", type=Path, default=Path("eval_sets/reach_nominal_v1.json"))
    parser.add_argument("--name", help="result name (default: <policy>_<set>)")
    parser.add_argument("--out-dir", type=Path, default=Path("results/reach"))
    args = parser.parse_args()

    eval_set = ReachEvalSet.load(args.set)
    policy, description = load_policy(args.policy)
    name = args.name or f"{args.policy.split(':')[0]}_{eval_set.name}"

    summary, episodes = evaluate(policy, eval_set)
    record = {
        "name": name,
        "policy": description,
        "eval_set": {"name": eval_set.name, "n": len(eval_set), "digest": eval_set.digest},
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(),
        "versions": {"python": platform.python_version(), "mujoco": mujoco.__version__,
                     "stable_baselines3": stable_baselines3.__version__,
                     "torch": torch.__version__, "numpy": np.__version__},
        "summary": summary,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    with open(args.out_dir / f"{name}_episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[fld.name for fld in fields(EpisodeResult)])
        writer.writeheader()
        writer.writerows(asdict(e) for e in episodes)

    s = summary
    lo, hi = s["success_ci95"]
    print(f"{name}: {description} on {eval_set.name} ({len(eval_set)} targets)")
    print(f"  success {s['success_rate']:.1%} (95% CI {lo:.1%}-{hi:.1%})")
    tts = s["time_to_success_s"]
    if tts["median"] is not None:
        print(f"  time to success median {tts['median']:.2f} s, p90 {tts['p90']:.2f} s; "
              f"held after success {s['held_rate_given_success']:.1%}")
    fd = s["final_distance_mm"]
    print(f"  final distance median {fd['median']:.3f} mm, p95 {fd['p95']:.3f}, "
          f"max {fd['max']:.3f}")
    print(f"  path ratio median {s['path_ratio_median']:.2f}, action jerk median "
          f"{s['action_jerk_median']:.4f}, contact episodes {s['contact_episode_rate']:.1%}, "
          f"limit episodes {s['limit_episode_rate']:.1%}")
    print(f"  inference latency median {s['inference_latency_us']['median']:.0f} us, "
          f"p99 {s['inference_latency_us']['p99']:.0f} us")
    for region, r in s["by_region"].items():
        print(f"  {region:10s} n={r['n']:3d} success {r['success_rate']:.1%}")
    if s["failures"]:
        print(f"  failures: {s['failures']}")
    print(f"  wrote {args.out_dir / name}.json and _episodes.csv")


if __name__ == "__main__":
    main()
