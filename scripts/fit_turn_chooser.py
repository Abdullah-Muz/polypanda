"""Fit one turn chooser on the outcomes of several runs (scripts/train_turn_chooser.py's
<run>/turn_outcomes_<which>.npz), e.g. to draw training commands for the next round
(StandUpConfig.command.sampler = "chooser").

Usage: uv run python scripts/fit_turn_chooser.py --outcomes A.npz B.npz ... --out FILE.pt
Reports, per run, how well a chooser fitted on the other four fifths of all starts picks on
the fifth held aside (against a random reachable command and the best possible), then fits
on everything.
"""

import argparse
from pathlib import Path

import numpy as np

from polypanda.training.turn_chooser import choice_quality, fit_chooser, fit_ensemble


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcomes", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ensemble", type=int, default=1)
    args = parser.parse_args()
    parts = []
    for k, path in enumerate(args.outcomes):
        d = dict(np.load(path))
        d["start"] = d["start"] * 100 + k          # starts of different runs stay apart
        d["source"] = np.full(len(d["start"]), k)
        parts.append(d)
    data = {key: np.concatenate([p[key] for p in parts]) for key in ("obs", "success", "start",
                                                                    "source")}
    starts = np.unique(data["start"])
    rng = np.random.default_rng(0)
    held = set(rng.choice(starts, size=len(starts) // 5, replace=False).tolist())
    is_held = np.array([s in held for s in data["start"]])
    chooser = fit_chooser(data["obs"][~is_held], data["success"][~is_held])
    for k, path in enumerate(args.outcomes):
        m = is_held & (data["source"] == k)
        test = {key: v[m] for key, v in data.items()}
        ids = np.unique(test["start"])
        best = np.mean([test["success"][test["start"] == s].any() for s in ids])
        rand = np.mean([test["success"][test["start"] == s].mean() for s in ids])
        print(f"{path}: held-out starts {len(ids)}: pooled chooser "
              f"{choice_quality(chooser.scores(test['obs']), test):.1%}, random reachable "
              f"{rand:.1%}, best possible {best:.1%}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    (fit_chooser(data["obs"], data["success"]) if args.ensemble == 1
     else fit_ensemble(data["obs"], data["success"], args.ensemble)).save(args.out)
    print(f"wrote {args.out} ({len(data['success'])} outcomes from {len(args.outcomes)} runs)")


if __name__ == "__main__":
    main()
