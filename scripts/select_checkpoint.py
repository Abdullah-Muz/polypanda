"""Select a grasp run's checkpoint on a larger validation set (polypanda.policies.selection).

Usage:
  uv run python scripts/select_checkpoint.py --make-sets                 # validation sets
  uv run python scripts/select_checkpoint.py runs/<run> [runs/<run> ...]
Each run is scored on the validation set matching its own training region (nominal or wide;
never a test set), and its winner becomes `run:runs/<run>@selected`. Writes
results/selection/<run>.json.
"""

import argparse
import json
from pathlib import Path

import torch

from polypanda.evaluation.pickplace import grasp_env_for
from polypanda.evaluation.sets import GraspEvalSet, generate_grasp_eval_set
from polypanda.policies.selection import select
from polypanda.simulation.scenes import WIDE_X_RANGE, WIDE_Y_RANGE

# Separate seeds from every test set (grasp tests use 20261022, 20261027, 20261029).
SETS = {"nominal": ("data/validation/grasp_select_nominal_v1.json", 20261030, {}),
        "wide": ("data/validation/grasp_select_wide_v1.json", 20261031,
                 {"x_range": WIDE_X_RANGE, "y_range": WIDE_Y_RANGE})}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--make-sets", action="store_true")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=Path("results/selection"))
    args = parser.parse_args()
    torch.set_num_threads(1)

    if args.make_sets:
        for path, seed, ranges in SETS.values():
            s = generate_grasp_eval_set(Path(path).stem, args.n, seed, **ranges)
            s.save(path)
            print(f"wrote {path}: {len(s)} placements, digest {s.digest}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        env = grasp_env_for(f"run:{run}")
        region = "wide" if env.config.placement.x_range == WIDE_X_RANGE else "nominal"
        val = GraspEvalSet.load(SETS[region][0])
        result = select(run, env, val)
        (args.out_dir / f"{run.name}.json").write_text(json.dumps(
            {"run": run.name, **result}, indent=2) + "\n")
        scores = " ".join(f"{k}:{v:.0%}" for k, v in result["scores"].items())
        print(f"{run.name}: {scores} -> selected {result['selected']} ({region} validation)")


if __name__ == "__main__":
    main()
