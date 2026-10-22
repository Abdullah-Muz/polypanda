"""Generate the versioned held-out reach evaluation set.

Usage: uv run python scripts/make_eval_set.py [--n 200] [--seed 20261021]
Writes eval_sets/<name>.json. Regenerating with the same arguments reproduces the file exactly
(checked in tests); bump the version in the name if anything that affects targets changes.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from polypanda.evaluation.sets import generate_reach_eval_set


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="reach_nominal_v1")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--out-dir", type=Path, default=Path("eval_sets"))
    args = parser.parse_args()

    eval_set = generate_reach_eval_set(args.name, args.n, args.seed)
    path = args.out_dir / f"{args.name}.json"
    eval_set.save(path)
    print(f"wrote {path}: {len(eval_set)} targets, digest {eval_set.digest}")
    for region, count in sorted(Counter(eval_set.regions).items()):
        print(f"  {region:10s} {count:4d}")
    d = np.linalg.norm(eval_set.targets - np.array([0.5545, 0.0, 0.5216]), axis=1)
    print(f"  distance from home TCP: median {np.median(d):.3f} m, min {d.min():.3f}, "
          f"max {d.max():.3f}")


if __name__ == "__main__":
    main()
