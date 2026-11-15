"""v3.1: how tower places fare with loose cubes near the tower (docs/v3_1_targets.md).

Usage: uv run python scripts/clutter_exposure.py [--out results/planner/clutter_exposure.json]

For every learned place on a tower in evaluate_cell's _skills.csv files, it rebuilds which
knocked cubes still lay on the table when the place started, and how far the nearest of them was
from the tower (supervisor.symbolic.tower_place_clutter, which the v3.1 clutter model is fitted
with). Shown per run group: the v3.0 calibration run (rescue every recoverable cube, no spares)
and the three v3.0 physics runs on disturbance_suite_v1.
"""

import argparse
import csv
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

from polypanda.supervisor.symbolic import tower_place_clutter

GROUPS = {
    "calibration (v3.0, rescue every cube)": "*disturbance_calib_v1_recover_n200_skills.csv",
    "suite_v1, rescue every cube": "*disturbance_suite_v1*rescue-all*_skills.csv",
    "suite_v1, best rule": "*disturbance_suite_v1*best-rule*_skills.csv",
    "suite_v1, learned planner": "*disturbance_suite_v1*recovery_planner_seed*_skills.csv",
}
BANDS = (("none within 20 cm", 0.20, float("inf")), ("10-20 cm", 0.10, 0.20),
         ("6-10 cm", 0.06, 0.10), ("under 6 cm", 0.0, 0.06))
def tower_places(path: str) -> list[dict]:
    """Every tower place in one _skills.csv with the nearest loose knocked cube's distance."""
    by_order = defaultdict(list)
    for r in csv.DictReader(open(path)):
        by_order[int(r["order"])].append(r)
    return [p for rows in by_order.values() for p in tower_place_clutter(rows)]


def summarize(places: list[dict]) -> dict:
    out = {"places": len(places), "failed": sum(not p["ok"] for p in places), "bands": {}}
    for name, lo, hi in BANDS:
        rows = [p for p in places if lo <= min(p["nearest"], 9.0) < hi]   # (inf: none at all)
        ok = sum(p["ok"] for p in rows)
        out["bands"][name] = {"places": len(rows), "succeeded": ok,
                              "rate": ok / len(rows) if rows else None,
                              "failures": dict(Counter(p["failure"] for p in rows
                                                       if not p["ok"]))}
    near = [p for p in places if p["nearest"] < 0.20]
    out["within_20cm"] = {"places": len(near), "succeeded": sum(p["ok"] for p in near)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("results/cell"))
    parser.add_argument("--out", type=Path, default=Path("results/planner/clutter_exposure.json"))
    args = parser.parse_args()
    report = {}
    for group, pattern in GROUPS.items():
        files = sorted(glob.glob(str(args.runs / pattern)))
        s = summarize([p for f in files for p in tower_places(f)])
        report[group] = {"files": [Path(f).name for f in files], **s}
        print(f"\n{group}: {s['places']} tower places, {s['failed']} failed ({len(files)} runs)")
        for name, b in s["bands"].items():
            if b["places"]:
                print(f"  nearest loose cube {name:18s} {b['succeeded']:5d}/{b['places']:5d}"
                      f" = {100 * b['rate']:6.2f} %  {b['failures']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
