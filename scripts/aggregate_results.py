"""Aggregate held-out evaluation results across seeds.

Usage: uv run python scripts/aggregate_results.py results/reach/reach_ppo_v2_seed*_best_*.json
       uv run python scripts/aggregate_results.py results/grasp/grasp_ppo_curr_seed*_best_*.json

Prints one row per run, then mean ± sample std across runs, and the success rate pooled over
all episodes with a 95 % Wilson interval. Reach and grasp (GraspEnv protocol) results are
recognised from their summary fields.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from polypanda.evaluation.reach import wilson_interval

METRICS = [
    ("success %", lambda s: 100 * s["success_rate"]),
    ("held %", lambda s: 100 * (s["held_rate_given_success"] or 0.0)),
    ("t_succ med s", lambda s: s["time_to_success_s"]["median"]),
    ("t_succ p90 s", lambda s: s["time_to_success_s"]["p90"]),
    ("final med mm", lambda s: s["final_distance_mm"]["median"]),
    ("final max mm", lambda s: s["final_distance_mm"]["max"]),
    ("path ratio", lambda s: s["path_ratio_median"]),
    ("jerk", lambda s: s["action_jerk_median"]),
    ("contact %", lambda s: 100 * s["contact_episode_rate"]),
]

GRASP_METRICS = [
    ("success %", lambda s: 100 * s["success_rate"]),
    ("t_succ med s", lambda s: s["first_success_s"]["median"]),
    ("t_succ p90 s", lambda s: s["first_success_s"]["p90"]),
    ("in-success %", lambda s: 100 * s["success_fraction_median"]),
    ("tilt med deg", lambda s: s["max_cube_tilt_deg"]["median"]),
    ("floor ep %", lambda s: 100 * s["floor_contact_episode_rate"]),
    ("return", lambda s: s["return"]["mean"]),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args()

    records = [json.loads(f.read_text()) for f in sorted(args.files)]
    digests = {r["eval_set"]["digest"] for r in records}
    if len(digests) != 1:
        raise SystemExit(f"results come from different evaluation sets: {digests}")

    grasp = ["success_fraction_median" in r["summary"] for r in records]
    if len(set(grasp)) != 1:
        raise SystemExit("cannot mix reach and grasp results")
    metrics = GRASP_METRICS if grasp[0] else METRICS

    header = f"{'run':44s}" + "".join(f"{name:>14s}" for name, _ in metrics)
    print(header)
    table = []
    for r in records:
        values = [fn(r["summary"]) for _, fn in metrics]
        table.append(values)
        dirty = " (dirty)" if r["git"]["dirty"] else ""
        print(f"{r['name'][:44]:44s}" + "".join(f"{v:14.3f}" for v in values) + dirty)
    arr = np.array(table, dtype=float)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1) if len(arr) > 1 else np.zeros(len(metrics))
    print(f"{'mean ± std over ' + str(len(arr)) + ' runs':44s}"
          + "".join(f"{m:8.3f}±{s:<5.3f}" for m, s in zip(mean, std, strict=True)))

    wins = sum(round(r["summary"]["success_rate"] * r["summary"]["episodes"]) for r in records)
    total = sum(r["summary"]["episodes"] for r in records)
    lo, hi = wilson_interval(wins, total)
    print(f"pooled success: {wins}/{total} = {wins / total:.1%} (95% CI {lo:.1%}-{hi:.1%})")


if __name__ == "__main__":
    main()
