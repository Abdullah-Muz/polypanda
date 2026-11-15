"""v3.1 skill round: a grasp round against its baseline on the development sets, paired.

Usage: uv run python scripts/grasp_round_report.py --new grasp_anypose6 [--old grasp_anypose5]
           [--out results/grasp/round_<new>.json]

Reads evaluate_grasp.py's and evaluate_grasp_robustness.py's episode files (best checkpoints,
3 seeds) and reports, per development set (grasp_lying_v1, grasp_lying_final_v1 since its one
run, grasp_upright_v1): the new and old success rates pooled over seeds and per seed, with an
exact McNemar test on the episodes paired by seed and index; the upright cubes near the base
(x < 0.40 m, seed 0's weakness); and the size robustness condition (cube size x0.85-1.15) on
lying_v1 and upright_v1, with its gap to the same set at nominal size (the v3.1 size bar is
measured on the one-shot set, not here). One-shot sets are not read.
"""

import argparse
import csv
import json
from pathlib import Path

from polypanda.evaluation.pickplace import mcnemar_exact

SETS = ("grasp_lying_v1", "grasp_lying_final_v1", "grasp_upright_v1")
SIZE_SETS = ("grasp_lying_v1", "grasp_upright_v1")
NEAR = 0.40   # m


def episodes(path: Path) -> list[dict] | None:
    return list(csv.DictReader(open(path))) if path.exists() else None


def runs(prefix: str, set_name: str, size: bool = False) -> dict[int, list[dict]] | None:
    out = {}
    for s in range(3):
        if size:
            path = Path(f"results/grasp_robustness/{prefix}_seed{s}_best_{set_name}_size"
                        "_episodes.csv")
        else:
            path = Path(f"results/grasp/{prefix}_seed{s}_best_{set_name}_episodes.csv")
        rows = episodes(path)
        if rows is None:
            return None
        out[s] = rows
    return out


def compare(new: dict, old: dict, keep=lambda r: True) -> dict:
    a, b, per_seed = [], [], {}
    for s in range(3):
        pairs = [(x["success"] == "True", y["success"] == "True")
                 for x, y in zip(new[s], old[s], strict=True) if keep(x)]
        a += [p[0] for p in pairs]
        b += [p[1] for p in pairs]
        per_seed[s] = {"new": 100 * sum(p[0] for p in pairs) / len(pairs),
                       "old": 100 * sum(p[1] for p in pairs) / len(pairs)}
    return {"cases": len(a), "new": 100 * sum(a) / len(a), "old": 100 * sum(b) / len(b),
            "per_seed": per_seed, "paired": mcnemar_exact(a, b)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", required=True)
    parser.add_argument("--old", default="grasp_anypose5")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = {"new": args.new, "old": args.old, "sets": {}, "size": {}}

    def show(label, c):
        seeds = " / ".join(f"{c['per_seed'][s]['new']:.1f}" for s in range(3))
        print(f"{label:34s} new {c['new']:5.1f} % ({seeds})  old {c['old']:5.1f} %  "
              f"({c['paired']['only_first']} only new, {c['paired']['only_second']} only old, "
              f"p = {c['paired']['p_value']:.3g}; {c['cases']} cases)")
    for name in SETS:
        new, old = runs(args.new, name), runs(args.old, name)
        if new is None or old is None:
            print(f"{name}: missing results")
            continue
        report["sets"][name] = compare(new, old)
        show(name, report["sets"][name])
        if name == "grasp_upright_v1":
            near = compare(new, old, lambda r: float(r["x"]) < NEAR)
            report["sets"]["grasp_upright_v1, x < 0.40 m"] = near
            show("  upright, x < 0.40 m", near)
    for name in SIZE_SETS:
        new, old = runs(args.new, name, size=True), runs(args.old, name, size=True)
        if new is None or old is None:
            print(f"{name} size: missing results")
            continue
        c = compare(new, old)
        nominal = report["sets"].get(name)
        if nominal:
            c["gap_to_nominal"] = {"new": c["new"] - nominal["new"],
                                   "old": c["old"] - nominal["old"]}
        report["size"][name] = c
        show(f"{name}, size x0.85-1.15", c)
        if nominal:
            print(f"  {'':32s} gap to nominal: new {c['gap_to_nominal']['new']:+.1f}, "
                  f"old {c['gap_to_nominal']['old']:+.1f} points")
    out = args.out or Path(f"results/grasp/round_{args.new}.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
