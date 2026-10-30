"""Evaluate learned recovery planners (v3 f) in the symbolic cell against the hand-written rules.

Usage: uv run python scripts/evaluate_recovery_planner.py runs/recovery_planner_seed0 ... \
           [--n 2000] [--reps 5] [--out results/planner/symbolic_eval.json]
Held-out orders (their own seed, never used in training or rule selection), each with a knock
whose fallen cubes rest as a calibration knock of that height left them, played `reps` times.
Every policy plays the same orders with the same random streams (per order and repetition), so
results are paired. The best rule is chosen among RULE_FAMILY on a separate selection set.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from polypanda.environments.recovery_planner import HIDDEN
from polypanda.evaluation.cell import CellEvalConfig
from polypanda.evaluation.pickplace import mcnemar_exact
from polypanda.evaluation.reach import wilson_interval
from polypanda.simulation.disturbances import make_suite
from polypanda.supervisor.planner import mixed_order
from polypanda.supervisor.symbolic import (
    DEFAULT_TASK,
    RESCUE_ALL,
    RULE_FAMILY,
    Calibration,
    SymbolicCell,
    undisturbed_time,
)

EVAL_SEED, SELECT_SEED = 20261105, 20261106


def cases(cal: Calibration, seed: int, n: int) -> list:
    rng, cfg, out = np.random.default_rng(seed), CellEvalConfig(), []
    while len(out) < n:
        order, _ = mixed_order(rng, cfg.heights, cfg.surplus, cfg.solvable_with)
        knock = make_suite([order.recipes], rng, no_knock=0.0)[0]
        pool = cal.by_height.get(knock.height) if knock else None
        if pool:
            out.append((order, knock, pool[rng.integers(len(pool))],
                        DEFAULT_TASK.slack * undisturbed_time(order, cal)))
    return out


def play(policy, cal, games, seed, reps) -> list[bool]:
    return [SymbolicCell(order, knock, fallen, cal, np.random.default_rng([seed, i, r]),
                         DEFAULT_TASK, deadline).result(policy)
            for i, (order, knock, fallen, deadline) in enumerate(games) for r in range(reps)]


def learned(run_dir: Path):
    model = PPO.load(run_dir / "model.zip", device="cpu")
    meta = json.loads((run_dir / "meta.json").read_text())
    hidden = [i for b in meta["blind"] for i in HIDDEN[b]]

    def policy(d) -> bool:
        obs = np.clip(d.features, -2.0, 2.0)
        obs[hidden] = 0.0
        return int(model.predict(obs, deterministic=True)[0]) == 1
    return policy


def summary(wins: list[bool]) -> dict:
    k, n = sum(wins), len(wins)
    return {"completed": k, "games": n, "rate": k / n, "ci95": list(wilson_interval(k, n))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("results/planner/symbolic_eval.json"))
    args = parser.parse_args()
    cal = Calibration.from_calibration_run()
    select = cases(cal, SELECT_SEED, args.n // 2)
    best = max(RULE_FAMILY, key=lambda r: sum(play(r, cal, select, SELECT_SEED, 2)))
    games = cases(cal, EVAL_SEED, args.n)
    results = {"rescue_all": play(RESCUE_ALL, cal, games, EVAL_SEED, args.reps),
               "best_rule": play(best, cal, games, EVAL_SEED, args.reps)}
    for run in args.runs:
        results[run.name] = play(learned(run), cal, games, EVAL_SEED, args.reps)
    out = {"task": {"spares": DEFAULT_TASK.spares, "slack": DEFAULT_TASK.slack,
                    "spare_delay": DEFAULT_TASK.spare_delay},
           "orders": args.n, "reps": args.reps, "eval_seed": EVAL_SEED,
           "best_rule": str(best), "policies": {k: summary(v) for k, v in results.items()},
           "paired": {}}
    for name in results:
        if name in ("rescue_all", "best_rule"):
            continue
        for base in ("best_rule", "rescue_all"):
            out["paired"][f"{name} vs {base}"] = mcnemar_exact(results[name], results[base])
    pooled = [w for name, v in results.items() if name.startswith("recovery_planner_seed")
              for w in v]
    if pooled:
        out["policies"]["learned (seeds pooled)"] = summary(pooled)
    for name, s in out["policies"].items():
        lo, hi = s["ci95"]
        print(f"{name}: {100 * s['rate']:.1f} % ({100 * lo:.1f}-{100 * hi:.1f}) of {s['games']}")
    for name, p in out["paired"].items():
        print(f"  {name}: only first {p['only_first']}, only second {p['only_second']}, "
              f"p = {p['p_value']:.3g}")
    print(f"best rule: {best}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
