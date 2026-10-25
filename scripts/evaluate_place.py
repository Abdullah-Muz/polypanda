"""Evaluate a place driver on the held-out place set and archive the results.

Usage: uv run python scripts/evaluate_place.py --policy scripted
       uv run python scripts/evaluate_place.py --policy run:runs/<run_dir>[@final]
Writes results/place/<name>.json (summary + provenance) and <name>_episodes.csv.
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

from polypanda.control.scripted_place import ScriptedPlace
from polypanda.environments.handoff import HandoffPool
from polypanda.evaluation.grasp import PolicyDriver
from polypanda.evaluation.place import PlaceEpisode, evaluate_place
from polypanda.policies.loader import load_run, run_env_config


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="scripted", help="scripted | run:DIR[@best|final]")
    parser.add_argument("--set", type=Path, default=Path("eval_sets/place_nominal_v1.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/place"))
    args = parser.parse_args()

    eval_set = HandoffPool.load(args.set)
    if args.policy == "scripted":
        driver, config = ScriptedPlace(), None
        name = f"scripted_{args.set.stem}"
        description = "scripted place (lower via closed-loop IK, gated release, retreat 10 cm)"
    elif args.policy.startswith("run:"):
        run_dir, _, which = args.policy[4:].partition("@")
        policy = load_run(run_dir, which or "best")
        meta = policy.metadata
        if meta["skill"] != "place":
            raise SystemExit(f"{run_dir} is a {meta['skill']} policy, not place")
        driver = PolicyDriver(policy)
        config = run_env_config(run_dir, "place")  # the policy's own task and action interface
        name = f"{meta['policy_id']}_{policy.which}_{args.set.stem}"
        description = (f"{meta['algorithm']} {policy.which} model from {run_dir} (seed "
                       f"{meta['seed']}, {meta['total_timesteps']:,} steps, "
                       f"{meta['action_schema']} actions)")
    else:
        raise SystemExit(f"unknown policy {args.policy!r}")
    summary, episodes = evaluate_place(driver, eval_set, config)
    record = {"name": name, "policy": description,
              "protocol": "PlaceEnv, 6 s horizon",
              "eval_set": {"name": args.set.stem, "n": len(eval_set), "digest": eval_set.digest},
              "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
              "git": git_state(),
              "versions": {"python": platform.python_version(), "mujoco": mujoco.__version__,
                           "numpy": np.__version__},
              "summary": summary}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    with open(args.out_dir / f"{name}_episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[fld.name for fld in fields(PlaceEpisode)])
        writer.writeheader()
        writer.writerows(asdict(e) for e in episodes)

    s = summary
    lo, hi = s["success_ci95"]
    print(f"{name}: {len(eval_set)} cases, {s['wall_clock_s']} s wall")
    print(f"  placed {s['success_rate']:.1%} (95% CI {lo:.1%}-{hi:.1%}); failures {s['failures']}")
    if s["first_placed_s"]["median"] is not None:
        print(f"  verified placed at median {s['first_placed_s']['median']:.2f} s "
              f"(p90 {s['first_placed_s']['p90']:.2f}); release at median "
              f"{s['release_s_median']:.2f} s")
    print(f"  final xy error median {s['final_xy_error_mm']['median']:.1f} mm "
          f"(p95 {s['final_xy_error_mm']['p95']:.1f}); final tilt median "
          f"{s['final_tilt_deg']['median']:.1f} deg (max {s['final_tilt_deg']['max']:.1f}); "
          f"floor-contact episodes {s['floor_contact_episode_rate']:.1%}")
    for source, r in s["by_source"].items():
        print(f"  from {source:28s} n={r['n']:3d} placed {r['success_rate']:.1%}")
    print(f"  wrote {args.out_dir / name}.json and _episodes.csv")


if __name__ == "__main__":
    main()
