"""v3.1 check before any planner training (docs/v3_1_targets.md): does the clutter-aware
symbolic cell reproduce the v3.0 physics results on disturbance_suite_v1?

Usage: uv run python scripts/clutter_check.py [--reps 50] [--out results/planner/clutter_check.json]

The symbolic cell: the v3.0 calibration (odds and action times, disturbance_calib_v1) plus the
clutter model and the clear action fitted on the v3.1 calibration run (random rescue / leave /
clear on the same calibration orders; Calibration.add_clutter). Nothing is fitted on the suite.
It plays every knocked order of disturbance_suite_v1, for each skills seed, with the knock
outcome that seed's physics run saw (where each fallen cube came to rest), `reps` times with
the same random streams for every policy, under the v3.0 task (2 spares, deadline 1.5 x the
order's undisturbed time), with the three v3.0 policies: rescue every cube, the best rule and
the learned planner of that seed. The check passes when it ranks them as physics did (best rule,
then the learned planner, then rescue every cube) with each within 3 points of its physics rate.
The v3.0 symbolic cell (no clutter) is shown beside it.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from polypanda.evaluation.cell import CellEvalConfig
from polypanda.simulation.disturbances import load_suite
from polypanda.supervisor.planner import mixed_order
from polypanda.supervisor.symbolic import (
    BEST_RULE,
    RESCUE_ALL,
    Calibration,
    SymbolicCell,
    Task,
    load_planner,
    undisturbed_time,
)

SUITE = Path("eval_sets/disturbance_suite_v1.json")
RUNS = "results/cell/*disturbance_suite_v1_recover_spares2-{tag}_slack1.5_n200"
V31_CALIBRATION = "results/cell/*disturbance_calib_v1_recover_spares2-random3_slack1.5_n200"
POLICIES = {"rescue every cube": "rescue-all", "best rule": "best-rule",
            "learned planner": "recovery_planner_seed*"}
RANKING = ("best rule", "learned planner", "rescue every cube")   # physics, v3.0 (c52560f)
TOLERANCE = 3.0   # points
KNOCKED = ("D1", "D2", "D3")


def physics(tag: str) -> tuple[dict, dict]:
    """Per (skills seed, order): the physics episode, and the knock outcomes it saw."""
    episodes = {}
    for path in sorted(Path().glob(RUNS.format(tag=tag) + "_episodes.csv")):
        seed = int(str(path).split("_seed")[1][0])
        for r in csv.DictReader(open(path)):
            episodes[(seed, int(r["index"]))] = r
    knocks = Calibration.load(sorted(Path().glob(RUNS.format(tag=tag) + "_skills.csv")),
                              SUITE).knocks
    return episodes, knocks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("results/planner/clutter_check.json"))
    args = parser.parse_args()

    v30 = Calibration.from_calibration_run()
    v31_runs = sorted(Path().glob(V31_CALIBRATION + "_skills.csv"))
    if len(v31_runs) != 3:
        raise SystemExit(f"expected the 3 v3.1 calibration runs, found {len(v31_runs)}")
    v31 = v30.add_clutter(v31_runs)
    cfg = CellEvalConfig(mixed=True)
    orders = [mixed_order(np.random.default_rng([cfg.seed, i]), cfg.heights, cfg.surplus,
                          cfg.solvable_with)[0] for i in range(cfg.n)]
    suite = load_suite(SUITE)
    deadline = [1.5 * undisturbed_time(o, v30) for o in orders]   # as the physics runs had

    report = {"reps": args.reps,
              "clutter_model": {"factors": v31.clutter, "clear_odds": v31.clear_odds,
                                "clear_time_s": v31.times["clear"], **v31.clutter_counts,
                                "runs": [p.name for p in v31_runs]},
              "policies": {}}
    for name, tag in POLICIES.items():
        episodes, knocks = physics(tag)
        cases = sorted(c for c, e in episodes.items() if e["disturbance"] in KNOCKED)
        planners = ({s: load_planner(Path(f"runs/recovery_planner_seed{s}")) for s in range(3)}
                    if name == "learned planner" else None)
        row = {"orders": len(cases),
               "physics": 100 * sum(episodes[c]["success"] == "True" for c in cases) / len(cases)}
        for label, cal in (("symbolic v3.0", v30), ("symbolic with clutter", v31)):
            wins = 0
            for s, i in cases:
                policy = planners[s] if planners else (BEST_RULE if tag == "best-rule"
                                                       else RESCUE_ALL)
                for r in range(args.reps):
                    cell = SymbolicCell(orders[i], suite[i], knocks.get((s, i)), cal,
                                        np.random.default_rng([20261206, s, i, r]), Task(),
                                        deadline[i])
                    wins += cell.result(policy)
            row[label] = 100 * wins / (args.reps * len(cases))
        report["policies"][name] = row
        print(f"{name:18s} ({row['orders']} knocked orders): physics {row['physics']:.1f}, "
              f"symbolic v3.0 {row['symbolic v3.0']:.1f}, "
              f"with clutter {row['symbolic with clutter']:.1f}", flush=True)

    rates = {n: r["symbolic with clutter"] for n, r in report["policies"].items()}
    ranked = all(rates[a] > rates[b] for a, b in zip(RANKING, RANKING[1:], strict=False))
    close = {n: abs(r["symbolic with clutter"] - r["physics"]) <= TOLERANCE
             for n, r in report["policies"].items()}
    report.update(ranked_as_physics=ranked, within_3_points=close,
                  passed=ranked and all(close.values()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"ranked as physics: {ranked}; within {TOLERANCE:g} points: {close}")
    print(f"wrote {args.out}; check {'passed' if report['passed'] else 'FAILED'}")


if __name__ == "__main__":
    main()
