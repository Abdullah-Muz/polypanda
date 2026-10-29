"""Train the turn chooser of a round-6 stand-up run (training.turn_chooser) and report how well
it chooses, offline, against the PPO critic's choice.

Usage: uv run python scripts/train_turn_chooser.py --run runs/<standup_v3 run>[@which] \
           [--starts 600] [--workers 5]
Runs the policy with every reachable command on --starts starts (env seeds from --seed0; never
an evaluation set), keeps a fifth of the starts aside to compare choosers, then fits on all.
Writes <run>/turn_chooser_<which>.pt, <run>/turn_outcomes_<which>.npz and
results/standup/<policy_id>_<which>_turn_chooser.json.
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from polypanda.policies.loader import load_run
from polypanda.training.turn_chooser import (
    choice_quality,
    collect_outcomes,
    fit_chooser,
    fit_ensemble,
)


def _collect(job):
    run, which, seeds = job
    torch.set_num_threads(1)
    return collect_outcomes(run, which, seeds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="DIR[@best|final]")
    parser.add_argument("--starts", type=int, default=600)
    parser.add_argument("--seed0", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--ensemble", type=int, default=1,
                        help="fit this many choosers and average them (d2 round 8 on: 5)")
    parser.add_argument("--out-dir", type=Path, default=Path("results/standup"))
    args = parser.parse_args()
    run, _, which = args.run.partition("@")
    which = which or "best"
    seeds = list(range(args.seed0, args.seed0 + args.starts))
    jobs = [(run, which, seeds[i::args.workers]) for i in range(args.workers)]
    with Pool(args.workers) as pool:
        parts = pool.map(_collect, jobs)
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez(Path(run) / f"turn_outcomes_{which}.npz", **data)

    policy = load_run(run, which)
    obs_n = (policy.vec_normalize.normalize_obs(data["obs"]) if policy.vec_normalize is not None
             else data["obs"])
    with torch.no_grad():
        critic = policy.model.policy.predict_values(
            torch.as_tensor(obs_n, dtype=torch.float32)).numpy().ravel()
    starts = np.unique(data["start"])
    rng = np.random.default_rng(0)
    held = set(rng.choice(starts, size=len(starts) // 5, replace=False).tolist())
    is_held = np.array([s in held for s in data["start"]])
    train = {k: v[~is_held] for k, v in data.items()}
    test = {k: v[is_held] for k, v in data.items()}
    chooser = (fit_chooser(train["obs"], train["success"]) if args.ensemble == 1
               else fit_ensemble(train["obs"], train["success"], args.ensemble))
    oracle = np.mean([data["success"][data["start"] == s].any() for s in held])
    random_pick = np.mean([data["success"][data["start"] == s].mean() for s in held])
    report = {
        "run": run, "which": which, "ensemble": args.ensemble, "starts": len(starts),
        "rows": int(len(data["success"])),
        "reachable_per_start_median": float(np.median(np.bincount(
            np.searchsorted(starts, data["start"])))),
        "held_out_starts": len(held),
        "held_out": {"chooser": choice_quality(chooser.scores(test["obs"]), test),
                     "critic": choice_quality(critic[is_held], test),
                     "random_reachable": float(random_pick),
                     "oracle_any_command": float(oracle)},
        "failures_by_command_outcome": {f: int((data["failure"] == f).sum())
                                        for f in np.unique(data["failure"]) if f},
    }
    final = (fit_chooser(data["obs"], data["success"]) if args.ensemble == 1
             else fit_ensemble(data["obs"], data["success"], args.ensemble))
    final.save(Path(run) / f"turn_chooser_{which}.pt")
    name = f"{policy.metadata['policy_id']}_{which}_turn_chooser"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
