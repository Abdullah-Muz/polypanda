"""v3.1 headroom gate for the recovery planner (docs/v3_1_targets.md), on the calibration orders.

Usage: uv run python scripts/planner_gate_v31.py [--reps 50] [--family-reps 10] [--workers 8]
           [--out results/planner/gate_v31.json]

The clutter-aware symbolic cell (the v3.0 calibration plus Calibration.add_clutter from the v3.1
calibration run, and the time spread from both calibration runs, Calibration.add_time_spread)
plays the knocked calibration orders (each with the knock the v3.0 physics run
saw, per skills seed) under the v3.0 task (2 spares, deadline 1.5 x the undisturbed time).
- Rules: the 2,187 ClearRules (rescue, leave or clear by pose and distance band to the nearest
  tower; and after a failed attempt). Each is scored with `family-reps` rolls per order, the
  best 20 again with `reps`, and the best of those re-scored with fresh rolls.
- Search: for each order, every plan that gives each of its fallen cubes its own answer
  (rescue, leave or clear) and one answer after any failed attempt; the best plan per order is
  re-scored with fresh rolls.
The gate: search at least 5 points above the best rule. Shown beside them: rescue every cube,
and v3.0's best rule (rescue or leave only).
"""

import argparse
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from polypanda.evaluation.cell import CellEvalConfig
from polypanda.simulation.disturbances import load_suite
from polypanda.supervisor.planner import mixed_order
from polypanda.supervisor.symbolic import (
    BEST_RULE,
    CLEAR,
    CLEAR_RULE_FAMILY,
    LEAVE,
    RESCUE,
    RESCUE_ALL,
    Calibration,
    SymbolicCell,
    Task,
    undisturbed_time,
)

SUITE = Path("eval_sets/disturbance_calib_v1.json")
ORDERS_SEED = 20261103
V30_RUNS = "results/cell/*disturbance_calib_v1_recover_n200"
V31_RUNS = "results/cell/*disturbance_calib_v1_recover_spares2-random3_slack1.5_n200"
TASK = Task()
GATE = 5.0   # points
SHORTLIST = 20


def setup():
    global CAL, ORDERS, KNOCKS, DEADLINE, CASES
    v30 = Calibration.load(sorted(Path().glob(V30_RUNS + "_skills.csv")), SUITE)
    runs = sorted(Path().glob(V31_RUNS + "_skills.csv"))
    if len(runs) != 3:
        raise SystemExit(f"expected the 3 v3.1 calibration runs, found {len(runs)}")
    CAL = v30.add_clutter(runs).add_time_spread(sorted(Path().glob(V30_RUNS + "_skills.csv"))
                                                + runs)
    cfg = CellEvalConfig(mixed=True, seed=ORDERS_SEED, n=200)
    ORDERS = [mixed_order(np.random.default_rng([cfg.seed, i]), cfg.heights, cfg.surplus,
                          cfg.solvable_with)[0] for i in range(cfg.n)]
    KNOCKS = load_suite(SUITE)
    DEADLINE = [TASK.slack * undisturbed_time(o, v30) for o in ORDERS]
    CASES = sorted(c for c, fallen in v30.knocks.items() if fallen)   # (seed, order)


def wins(policy_for, reps: int, stream: int, cases=None) -> int:
    total = 0
    for s, i in CASES if cases is None else cases:
        policy = policy_for(s, i)
        for r in range(reps):
            cell = SymbolicCell(ORDERS[i], KNOCKS[i], CAL.knocks[(s, i)], CAL,
                                np.random.default_rng([stream, s, i, r]), TASK, DEADLINE[i])
            total += cell.result(policy)
    return total


def score_rule(job) -> tuple[int, int]:
    k, reps, stream = job
    rule = CLEAR_RULE_FAMILY[k]
    return k, wins(lambda s, i: rule, reps, stream)


def plan_policy(answers, after):
    return lambda d: after if d.retry else (answers[d.cube.index]
                                            if d.cube.index < len(answers) else RESCUE)


def best_plan(job):
    (s, i), reps = job
    k = len(CAL.knocks[(s, i)])
    scored = []
    for answers in itertools.product((RESCUE, LEAVE, CLEAR), repeat=k):
        for after in (RESCUE, LEAVE, CLEAR):
            policy = plan_policy(answers, after)
            scored.append((wins(lambda s_, i_, p=policy: p, reps, 4, [(s, i)]), answers, after))
    _, answers, after = max(scored, key=lambda x: x[0])
    return (s, i), answers, after


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--family-reps", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("results/planner/gate_v31.json"))
    args = parser.parse_args()
    setup()
    n = len(CASES)

    def rate(w, reps):
        return 100 * w / (reps * n)
    with Pool(args.workers, initializer=setup) as pool:
        family = pool.map(score_rule, [(k, args.family_reps, 1)
                                       for k in range(len(CLEAR_RULE_FAMILY))], chunksize=8)
        short = sorted(family, key=lambda x: -x[1])[:SHORTLIST]
        rescored = pool.map(score_rule, [(k, args.reps, 2) for k, _ in short])
        best_k = max(rescored, key=lambda x: x[1])[0]
        best = CLEAR_RULE_FAMILY[best_k]
        plans = dict((c, (a, f)) for c, a, f in pool.map(best_plan, [(c, args.reps)
                                                                    for c in CASES]))
    reps4 = 4 * args.reps
    out = {
        "orders": n, "reps": args.reps, "family_reps": args.family_reps,
        "task": {"spares": TASK.spares, "slack": TASK.slack},
        "clutter_model": {"factors": CAL.clutter, "clear_odds": CAL.clear_odds,
                          "clear_time_s": CAL.times["clear"], **CAL.clutter_counts,
                          "time_sd_s": {a: v ** 0.5 for a, v in CAL.time_var.items()}},
        "rescue_all": rate(wins(lambda s, i: RESCUE_ALL, reps4, 5), reps4),
        "v30_best_rule": str(BEST_RULE),
        "v30_best_rule_rate": rate(wins(lambda s, i: BEST_RULE, reps4, 5), reps4),
        "best_rule": str(best), "best_rule_index": best_k,
        "best_rule_rate": rate(wins(lambda s, i: best, reps4, 5), reps4),
        "search": rate(wins(lambda s, i: plan_policy(*plans[(s, i)]), reps4, 5), reps4),
        "shortlist": [{"index": k, "rule": str(CLEAR_RULE_FAMILY[k]),
                       "rate": rate(w, args.reps)} for k, w in rescored],
    }
    out["headroom"] = out["search"] - out["best_rule_rate"]
    out["passed"] = out["headroom"] >= GATE
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{n} knocked calibration orders: rescue every cube {out['rescue_all']:.1f}, v3.0 best "
          f"rule {out['v30_best_rule_rate']:.1f}, best clear rule {out['best_rule_rate']:.1f} "
          f"({best}), search {out['search']:.1f}: headroom {out['headroom']:+.1f}")
    print(f"wrote {args.out}; gate {'passed' if out['passed'] else 'failed'}")


if __name__ == "__main__":
    main()
