"""Evaluate a grasp controller on the held-out placement set and archive the results.

Usage:
  uv run python scripts/evaluate_grasp.py                      # standalone scripted runner
  uv run python scripts/evaluate_grasp.py --policy scripted    # scripted, GraspEnv protocol
  uv run python scripts/evaluate_grasp.py --policy run:runs/<run_dir>[@final]
  uv run python scripts/evaluate_grasp.py --make-set           # (re)generate the placement set
Writes results/grasp/<name>.json (summary + provenance) and <name>_episodes.csv.
The GraspEnv protocol (--policy) is the one used to compare learned and scripted grasping.
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
from polypanda.environments.grasp import GraspV2BlindClutterEnv, GraspV2Env, GraspV3Env
from polypanda.evaluation.grasp import (
    GraspEpisodeResult,
    GraspPolicyEpisode,
    PolicyDriver,
    ScriptedDriver,
    evaluate_grasp_policy,
    evaluate_scripted_grasp,
)
from polypanda.evaluation.pickplace import grasp_env_for
from polypanda.evaluation.sets import GraspEvalSet, generate_grasp_eval_set
from polypanda.policies.loader import load_run
from polypanda.simulation.scenes import WIDE_X_RANGE, WIDE_Y_RANGE

DEFAULT_SET = Path("eval_sets/grasp_nominal_v1.json")


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


def write_record(out_dir: Path, name: str, record: dict, episodes, row_type) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    with open(out_dir / f"{name}_episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[fld.name for fld in fields(row_type)])
        writer.writeheader()
        writer.writerows(asdict(e) for e in episodes)


def run_env_protocol(args, eval_set: GraspEvalSet) -> None:
    if args.policy == "scripted":
        driver, description, name = (ScriptedDriver(), "scripted top-down grasp through GraspEnv",
                                     f"scripted_env_{eval_set.name}")
    elif args.policy.startswith("run:"):
        run_dir, _, which = args.policy[4:].partition("@")
        policy = load_run(run_dir, which or "best")
        meta = policy.metadata
        if meta["skill"] not in ("grasp", "grasp_v2", "grasp_v3", "grasp_v4"):
            raise SystemExit(f"{run_dir} is a {meta['skill']} policy, not grasp")
        driver = PolicyDriver(policy)
        description = (f"{meta['algorithm']} {policy.which} model from {run_dir} "
                       f"(seed {meta['seed']}, {meta['total_timesteps']:,} steps)")
        name = f"{meta['policy_id']}_{policy.which}_{eval_set.name}"
    else:
        raise SystemExit(f"unknown policy {args.policy!r}; use scripted or run:DIR[@which]")

    env = grasp_env_for(args.policy)  # the policy's own task env (GraspEnv, V2 or V3)
    if eval_set.distractors is not None and env.distractor_slots == 0:
        # clutter set: put the other cubes in the scene even if the skill cannot observe them
        if args.policy == "scripted":
            env = GraspV3Env(env.config)
        elif isinstance(env, GraspV2Env):
            env = GraspV2BlindClutterEnv(env.config)
        else:
            raise SystemExit("v1 grasp policies cannot be evaluated on clutter sets")
    summary, episodes = evaluate_grasp_policy(driver, eval_set, env=env)
    record = {"name": name, "policy": description,
              "protocol": f"{type(env).__name__}, 15 s horizon",
              "eval_set": {"name": eval_set.name, "n": len(eval_set), "digest": eval_set.digest},
              "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
              "git": git_state(),
              "versions": {"python": platform.python_version(), "mujoco": mujoco.__version__,
                           "numpy": np.__version__},
              "summary": summary}
    write_record(args.out_dir, name, record, episodes, GraspPolicyEpisode)

    s = summary
    lo, hi = s["success_ci95"]
    print(f"{name}: {description} on {eval_set.name} ({len(eval_set)} placements), "
          f"{s['wall_clock_s']} s wall")
    print(f"  success {s['success_rate']:.1%} (95% CI {lo:.1%}-{hi:.1%}); "
          f"failures {s['failures']}")
    if s["first_success_s"]["median"] is not None:
        print(f"  first success median {s['first_success_s']['median']:.2f} s, "
              f"p90 {s['first_success_s']['p90']:.2f} s; time in success state median "
              f"{s['success_fraction_median']:.1%} of episode")
    print(f"  retries {s['retry_rate']:.1%}, cube tilt median "
          f"{s['max_cube_tilt_deg']['median']:.1f} deg (max {s['max_cube_tilt_deg']['max']:.1f}),"
          f" floor-contact episodes {s['floor_contact_episode_rate']:.1%}")
    print(f"  return {s['return']['mean']:.1f} +/- {s['return']['std']:.1f}; latency median "
          f"{s['inference_latency_us']['median']:.0f} us")
    for region, r in s["by_region"].items():
        print(f"  {region:5s} n={r['n']:3d} success {r['success_rate']:.1%}")
    print(f"  wrote {args.out_dir / name}.json and _episodes.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", type=Path, default=DEFAULT_SET)
    parser.add_argument("--policy", help="scripted | run:DIR[@best|final] (GraspEnv protocol)")
    parser.add_argument("--make-set", action="store_true")
    parser.add_argument("--wide", action="store_true",
                        help="with --make-set: placements over the wide (fully reachable) region")
    parser.add_argument("--distractors", type=int, default=0,
                        help="with --make-set: other cubes per placement (clutter set)")
    parser.add_argument("--lying", action="store_true",
                        help="with --make-set: cubes lying on a side (v4 any-pose grasp)")
    parser.add_argument("--x-range", type=float, nargs=2,
                        help="with --make-set: placement x range (m), overrides --wide")
    parser.add_argument("--y-range", type=float, nargs=2,
                        help="with --make-set: placement y range (m), overrides --wide")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--out-dir", type=Path, default=Path("results/grasp"))
    args = parser.parse_args()

    if args.make_set:
        ranges = ({"x_range": WIDE_X_RANGE, "y_range": WIDE_Y_RANGE} if args.wide else {})
        if args.x_range:
            ranges["x_range"] = tuple(args.x_range)
        if args.y_range:
            ranges["y_range"] = tuple(args.y_range)
        eval_set = generate_grasp_eval_set(args.set.stem, args.n, args.seed, **ranges,
                                           distractors=args.distractors, lying=args.lying)
        eval_set.save(args.set)
        print(f"wrote {args.set}: {len(eval_set)} placements, digest {eval_set.digest}, "
              f"regions {dict(Counter(eval_set.regions))}")
        return

    eval_set = GraspEvalSet.load(args.set)
    if args.policy:
        run_env_protocol(args, eval_set)
        return
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
    write_record(args.out_dir, name, record, episodes, GraspEpisodeResult)

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
