"""Make the v3 disturbance suite: one knock (or none) for each of the Layer 2 evaluation orders.

Usage: uv run python scripts/make_disturbance_suite.py [--out eval_sets/disturbance_suite_v1.json]
  Development suite (d+: other orders and knocks; run evaluate_cell.py with --seed ORDERS --n N):
  ... --out eval_sets/disturbance_dev_v1.json --orders-seed 20261101 --suite-seed 20261102 --n 50
For every order of the 200 Layer 2 evaluation orders (the same recipes and arrivals as v2), from
its own random stream: with probability 0.2, or when no tower will be 2 cubes tall, no knock (D0,
a control); otherwise a tower that will be at least 2 cubes tall, the height after which it is
knocked (2 up to its full height), and a knock (simulation.disturbances.Knock.sample). Whether a
knock throws off one cube, some or all (D1/D2/D3), and whether that is recoverable, is decided by
the simulation when it happens. The suite is fixed before any recovery is built or tuned.
"""

import argparse
from pathlib import Path

import numpy as np

from polypanda.evaluation.cell import CellEvalConfig
from polypanda.simulation.disturbances import make_suite, save_suite
from polypanda.supervisor.planner import mixed_order

SUITE_SEED = 20261027  # its own stream, independent of the orders' (20261026)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("eval_sets/disturbance_suite_v1.json"))
    parser.add_argument("--no-knock", type=float, default=0.2)
    parser.add_argument("--orders-seed", type=int, default=CellEvalConfig.seed)
    parser.add_argument("--suite-seed", type=int, default=SUITE_SEED)
    parser.add_argument("--n", type=int, default=CellEvalConfig.n)
    args = parser.parse_args()
    cfg = CellEvalConfig(mixed=True, seed=args.orders_seed, n=args.n)
    recipes = [mixed_order(np.random.default_rng([cfg.seed, i]), cfg.heights, cfg.surplus,
                           cfg.solvable_with)[0].recipes for i in range(cfg.n)]
    suite = make_suite(recipes, np.random.default_rng(args.suite_seed), args.no_knock)
    meta = {"orders": f"Layer 2 evaluation orders (CellEvalConfig mixed, seed {cfg.seed})",
            "n": cfg.n, "seed": args.suite_seed, "no_knock": args.no_knock, "version": 1}
    save_suite(args.out, suite, meta)
    knocks = [s for s in suite if s is not None]
    print(f"wrote {args.out}: {len(knocks)} knocks, {len(suite) - len(knocks)} controls (D0); "
          f"pushes {sorted({k.knock.push for k in knocks})}")


if __name__ == "__main__":
    main()
