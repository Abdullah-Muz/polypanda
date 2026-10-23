"""Evaluate a transport driver on the held-out (handoff state, goal) set and archive results.

Usage: uv run python scripts/evaluate_transport.py --policy scripted
       uv run python scripts/evaluate_transport.py --policy run:runs/<run_dir>[@final]
Writes results/transport/<name>.json (summary + provenance) and <name>_episodes.csv.
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

from polypanda.environments.handoff import HandoffPool
from polypanda.evaluation.grasp import PolicyDriver
from polypanda.evaluation.transport import (
    ScriptedCarryDriver,
    TransportEpisode,
    evaluate_transport,
)
from polypanda.policies.loader import load_run


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, help="scripted | run:DIR[@best|final]")
    parser.add_argument("--set", type=Path, default=Path("eval_sets/transport_nominal_v1.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/transport"))
    args = parser.parse_args()

    eval_set = HandoffPool.load(args.set)
    set_name = args.set.stem
    if args.policy == "scripted":
        driver = ScriptedCarryDriver()
        description = "scripted closed-loop carry (PoseIK, privileged cube position, gain 0.2)"
        name = f"scripted_{set_name}"
    elif args.policy.startswith("run:"):
        run_dir, _, which = args.policy[4:].partition("@")
        policy = load_run(run_dir, which or "best")
        meta = policy.metadata
        if meta["skill"] != "transport":
            raise SystemExit(f"{run_dir} is a {meta['skill']} policy, not transport")
        driver = PolicyDriver(policy)
        description = (f"{meta['algorithm']} {policy.which} model from {run_dir} "
                       f"(seed {meta['seed']}, {meta['total_timesteps']:,} steps)")
        name = f"{meta['policy_id']}_{policy.which}_{set_name}"
    else:
        raise SystemExit(f"unknown policy {args.policy!r}")

    summary, episodes = evaluate_transport(driver, eval_set)
    record = {"name": name, "policy": description, "protocol": "TransportEnv, 10 s horizon",
              "eval_set": {"name": set_name, "n": len(eval_set), "digest": eval_set.digest},
              "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
              "git": git_state(),
              "versions": {"python": platform.python_version(), "mujoco": mujoco.__version__,
                           "numpy": np.__version__},
              "summary": summary}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    with open(args.out_dir / f"{name}_episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[fld.name for fld in fields(TransportEpisode)])
        writer.writeheader()
        writer.writerows(asdict(e) for e in episodes)

    s = summary
    lo, hi = s["success_ci95"]
    print(f"{name}: {description}, {len(eval_set)} cases, {s['wall_clock_s']} s wall")
    print(f"  success {s['success_rate']:.1%} (95% CI {lo:.1%}-{hi:.1%}); drops "
          f"{s['drop_rate']:.1%}; failures {s['failures']}")
    if s["first_success_s"]["median"] is not None:
        print(f"  first success median {s['first_success_s']['median']:.2f} s, "
              f"p90 {s['first_success_s']['p90']:.2f} s")
    print(f"  final distance median {s['final_distance_mm']['median']:.1f} mm, "
          f"p95 {s['final_distance_mm']['p95']:.1f}; max slip speed median "
          f"{s['max_slip_speed_mm_s_median']:.0f} mm/s; cube tilt median "
          f"{s['max_cube_tilt_deg']['median']:.1f} deg (max {s['max_cube_tilt_deg']['max']:.1f})")
    print(f"  floor-contact episodes {s['floor_contact_episode_rate']:.1%}; action jerk median "
          f"{s['action_jerk_median']:.4f}; return {s['return']['mean']:.1f}; latency median "
          f"{s['inference_latency_us']['median']:.0f} us")
    for source, r in s["by_source"].items():
        print(f"  from {source:28s} n={r['n']:3d} success {r['success_rate']:.1%}, "
              f"drops {r['drop_rate']:.1%}")
    print(f"  wrote {args.out_dir / name}.json and _episodes.csv")


if __name__ == "__main__":
    main()
