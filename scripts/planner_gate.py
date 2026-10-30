"""The learned recovery planner's headroom gate (v3 step e), on the calibration orders.

Usage: uv run python scripts/planner_gate.py [--reps 50] [--out results/planner/gate.json]

1. Check: the symbolic cell (supervisor.symbolic, calibrated on the physics run's skill attempts)
   plays scripted recovery without spares on the calibration orders, each with the knock the
   physics run saw; its completion is compared with the physics run's, per knock kind.
2. Headroom: with spares (and deadlines), the best of the 32 hand-written rules (RULE_FAMILY)
   against search, which tries every rescue-or-replace plan for each order's fallen cubes (and
   retry or replace after a failure) and keeps the best. Both are re-scored with fresh rolls.
   The gate (docs/v3_targets.md): search at least 5 points above the best rule.
"""

import argparse
import csv
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from polypanda.evaluation.cell import CellEvalConfig
from polypanda.simulation.disturbances import load_suite
from polypanda.supervisor.planner import mixed_order
from polypanda.supervisor.symbolic import (
    RESCUE_ALL,
    RULE_FAMILY,
    Calibration,
    SymbolicCell,
    Task,
    undisturbed_time,
)

SUITE = Path("eval_sets/disturbance_calib_v1.json")
ORDERS_SEED = 20261103
RUNS = "results/cell/*disturbance_calib_v1_recover_n200"
TASKS = [Task(1, None), Task(2, None)] + [Task(2, s, d) for s in (1.25, 1.5, 2.0)
                                         for d in (0.0, 20.0)]


def setup():
    global CAL, ORDERS, KNOCKS, T0, PHYS, CASES
    CAL = Calibration.load(sorted(Path().glob(RUNS + "_skills.csv")), SUITE)
    cfg = CellEvalConfig(mixed=True, seed=ORDERS_SEED, n=200)
    ORDERS = [mixed_order(np.random.default_rng([cfg.seed, i]), cfg.heights, cfg.surplus,
                          cfg.solvable_with)[0] for i in range(cfg.n)]
    KNOCKS = load_suite(SUITE)
    T0 = [undisturbed_time(o, CAL) for o in ORDERS]
    PHYS = {}
    for path in sorted(Path().glob(RUNS + "_episodes.csv")):
        seed = int(str(path).split("_seed")[1][0])
        for r in csv.DictReader(open(path)):
            PHYS[(seed, int(r["index"]))] = r
    CASES = [c for c, r in PHYS.items() if r["disturbance"] in ("D1", "D2", "D3")]


def rate(policy_for, task: Task, reps: int, seed: int, cases=None) -> float:
    rng, wins, cases = np.random.default_rng(seed), 0, CASES if cases is None else cases
    for s, i in cases:
        deadline = None if task.slack is None else task.slack * T0[i]
        policy = policy_for(s, i)
        wins += sum(SymbolicCell(ORDERS[i], KNOCKS[i], CAL.knocks.get((s, i)), CAL, rng, task,
                                 deadline).result(policy) for _ in range(reps))
    return 100 * wins / (reps * len(cases))


def plan_policy(bits, after):
    return lambda d: after if d.retry else (d.cube.index < len(bits) and bits[d.cube.index])


def headroom(job) -> dict:
    task, reps = job
    family = [(rule, rate(lambda s, i, r=rule: r, task, reps, 2)) for rule in RULE_FAMILY]
    best_rule = max(family, key=lambda f: f[1])[0]
    rng, choice = np.random.default_rng(4), {}
    for s, i in CASES:
        deadline = None if task.slack is None else task.slack * T0[i]
        k = len(CAL.knocks.get((s, i)) or [])
        scored = []
        for bits in itertools.product((False, True), repeat=k):
            for after in (False, True):
                pol = plan_policy(bits, after)
                scored.append((sum(SymbolicCell(ORDERS[i], KNOCKS[i], CAL.knocks.get((s, i)),
                                                CAL, rng, task, deadline).result(pol)
                                   for _ in range(reps)), pol))
        choice[(s, i)] = max(scored, key=lambda x: x[0])[1]
    out = {"spares": task.spares, "slack": task.slack, "spare_delay": task.spare_delay,
           "no_spares": rate(lambda s, i: RESCUE_ALL, Task(0, task.slack), 4 * reps, 1),
           "rescue_all": rate(lambda s, i: RESCUE_ALL, task, 4 * reps, 1),
           "best_rule": str(best_rule),
           "best_rule_rate": rate(lambda s, i: best_rule, task, 4 * reps, 3),
           "search": rate(lambda s, i: choice[(s, i)], task, 4 * reps, 5)}
    out["headroom"] = out["search"] - out["best_rule_rate"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("results/planner/gate.json"))
    args = parser.parse_args()
    setup()
    check = []
    for kind in ("D1", "D2", "D3", None):
        for rec in (True, False):
            sel = [c for c in CASES if kind in (None, PHYS[c]["disturbance"])
                   and (PHYS[c]["recoverable"] == "True") == rec]
            check.append({"kind": kind or "all", "recoverable": rec, "orders": len(sel),
                          "physics": 100 * sum(PHYS[c]["success"] == "True" for c in sel)
                          / len(sel),
                          "symbolic": rate(lambda s, i: RESCUE_ALL, Task(0, None), 4 * args.reps,
                                           0, sel)})
            print("check {kind} recoverable={recoverable} (n={orders}): physics {physics:.1f}, "
                  "symbolic {symbolic:.1f}".format(**check[-1]), flush=True)
    with Pool(len(TASKS), initializer=setup) as pool:
        gates = pool.map(headroom, [(t, args.reps) for t in TASKS])
    for g in gates:
        print("spares {spares}, deadline {slack}, spare delay {spare_delay}: no spares "
              "{no_spares:.1f}, rescue all {rescue_all:.1f}, best rule {best_rule_rate:.1f} "
              "({best_rule}), search {search:.1f}: headroom {headroom:+.1f}".format(**g))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "calibration": {"odds": {str(k): v for k, v in CAL.odds.items()},
                        "attempts": {str(k): v for k, v in CAL.counts.items()},
                        "action_times_s": CAL.times},
        "check": check, "gate": gates, "reps": args.reps,
        "passed": any(g["headroom"] >= 5 for g in gates)}, indent=2) + "\n")
    passed = any(g["headroom"] >= 5 for g in gates)
    print(f"wrote {args.out}; gate {'passed' if passed else 'failed'}")


if __name__ == "__main__":
    main()
