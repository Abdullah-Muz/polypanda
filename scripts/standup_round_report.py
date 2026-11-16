"""v3.1 skill round: a stand-up round against its baseline on the development sets, paired.

Usage: uv run python scripts/standup_round_report.py --new standup_round12 [--old standup_round11]
           [--out results/standup/round_<new>.json]

Reads evaluate_standup.py's and evaluate_standup_robustness.py's episode files (best checkpoints,
3 seeds, the turn direction chosen by each run's turn chooser) and reports, per development set
(standup_v1, standup_final_v1 since its one run): the new and old success rates pooled over seeds
and per seed, with an exact McNemar test on the episodes paired by seed and index, and the
failures by kind; then the size robustness condition (cube size x0.85-1.15) on standup_v1, with
its gap to the same set at nominal size (the v3.1 size bar is measured on the one-shot set, not
here). One-shot sets are not read.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from polypanda.evaluation.pickplace import mcnemar_exact

SETS = ("standup_v1", "standup_final_v1")
SIZE_SET = "standup_v1"
CHOOSE = "choose-chooser"


def runs(prefix: str, set_name: str, size: bool = False) -> dict[int, dict[int, dict]] | None:
    """{seed: {index: row}}, or None if a seed's file is missing."""
    out = {}
    for s in range(3):
        if size:
            path = Path(f"results/standup_robustness/{prefix}_seed{s}_best_{set_name}_{CHOOSE}"
                        "_size_episodes.csv")
        else:
            path = Path(f"results/standup/{prefix}_seed{s}_best_{set_name}_{CHOOSE}"
                        "_episodes.csv")
        if not path.exists():
            return None
        with open(path) as f:
            out[s] = {int(r["index"]): r for r in csv.DictReader(f)}
    return out


def compare(new: dict, old: dict) -> dict:
    a, b, per_seed = [], [], {}
    failures = {"new": Counter(), "old": Counter()}
    for s in range(3):
        if set(new[s]) != set(old[s]):
            raise SystemExit(f"seed {s}: the new and old runs have different cases")
        pairs = [(new[s][i]["success"] == "True", old[s][i]["success"] == "True")
                 for i in sorted(new[s])]
        a += [p[0] for p in pairs]
        b += [p[1] for p in pairs]
        per_seed[s] = {"new": 100 * sum(p[0] for p in pairs) / len(pairs),
                       "old": 100 * sum(p[1] for p in pairs) / len(pairs)}
        for label, rows in (("new", new[s]), ("old", old[s])):
            failures[label].update(r["failure"] for r in rows.values() if r["failure"])
    return {"cases": len(a), "new": 100 * sum(a) / len(a), "old": 100 * sum(b) / len(b),
            "per_seed": per_seed, "paired": mcnemar_exact(a, b),
            "failures": {k: dict(v.most_common()) for k, v in failures.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", required=True)
    parser.add_argument("--old", default="standup_round11")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = {"new": args.new, "old": args.old, "turn": "each run's turn chooser",
              "sets": {}, "size": {}}

    def show(label, c):
        seeds = " / ".join(f"{c['per_seed'][s]['new']:.1f}" for s in range(3))
        olds = " / ".join(f"{c['per_seed'][s]['old']:.1f}" for s in range(3))
        print(f"{label:30s} new {c['new']:5.1f} % ({seeds})  old {c['old']:5.1f} % ({olds})  "
              f"{c['paired']['only_first']} only new, {c['paired']['only_second']} only old, "
              f"p = {c['paired']['p_value']:.3g}; {c['cases']} cases")
        print(f"  {'':28s} failures new {c['failures']['new']}, old {c['failures']['old']}")
    for name in SETS:
        new, old = runs(args.new, name), runs(args.old, name)
        if new is None or old is None:
            print(f"{name}: missing results")
            continue
        report["sets"][name] = compare(new, old)
        show(name, report["sets"][name])
    new, old = runs(args.new, SIZE_SET, size=True), runs(args.old, SIZE_SET, size=True)
    if new is None or old is None:
        print(f"{SIZE_SET} size: missing results")
    else:
        c = compare(new, old)
        nominal = report["sets"].get(SIZE_SET)
        if nominal:
            c["gap_to_nominal"] = {"new": c["new"] - nominal["new"],
                                   "old": c["old"] - nominal["old"]}
        report["size"][SIZE_SET] = c
        show(f"{SIZE_SET}, size x0.85-1.15", c)
        if nominal:
            print(f"  {'':28s} gap to nominal: new {c['gap_to_nominal']['new']:+.1f}, "
                  f"old {c['gap_to_nominal']['old']:+.1f} points")
    out = args.out or Path(f"results/standup/round_{args.new}.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
