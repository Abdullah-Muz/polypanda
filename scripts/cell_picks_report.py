"""v3.2: the cell-like pick checks of one or more grasps, against the bars in docs/v3_2_targets.md.

Usage:
  uv run python scripts/cell_picks_report.py --grasps grasp_anypose6 grasp_anypose5 scripted \
      [--belt grasp_belt_v1] [--floor grasp_floor_v1] [--out results/grasp/cellpicks_report.json]
Reads results/grasp/cellpicks_<set>_<grasp>_seed<k>_best[_size]_episodes.csv (scripted: one
run, cellpicks_<set>_scripted[_size]_episodes.csv). Per grasp and seed: success, knock-overs and
the weakest 10-deg rotation bin; each bar checked per seed; the first grasp paired against each
other one, pooled over seeds (exact McNemar on the same cases).
"""

import argparse
import csv
import json
from pathlib import Path

from polypanda.evaluation.pickplace import mcnemar_exact
from polypanda.evaluation.reach import wilson_interval

BELT_MIN, BELT_MAX_KNOCKED = 0.99, 1        # per seed
FLOOR_MIN, FLOOR_BIN_MIN = 0.95, 0.85       # per seed
SIZE_GAP = 0.05                             # pooled, size condition against nominal


def load(results: Path, set_name: str, grasp: str, seed: int | None, size: bool) -> list[dict]:
    run = grasp if seed is None else f"{grasp}_seed{seed}_best"
    path = results / f"cellpicks_{set_name}_{run}{'_size' if size else ''}_episodes.csv"
    if not path.exists():
        return []
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["success"] = r["success"] == "True"
        r["knocked_over"] = r["knocked_over"] == "True"
        r["yaw_rel"] = float(r["yaw_rel"])
    return sorted(rows, key=lambda r: int(r["index"]))


def stats(rows: list[dict]) -> dict:
    n, ok = len(rows), sum(r["success"] for r in rows)
    bins = {}
    for lo in range(-45, 45, 10):
        rs = [r for r in rows if lo <= r["yaw_rel"] < lo + 10 or (lo == 35 and r["yaw_rel"] >= 45)]
        if rs:
            bins[f"{lo:+d}..{lo + 10:+d}"] = (sum(r["success"] for r in rs), len(rs))
    worst = min(bins.items(), key=lambda kv: kv[1][0] / kv[1][1]) if bins else None
    return {"n": n, "success": ok, "rate": ok / n if n else None,
            "ci95": list(wilson_interval(ok, n)) if n else None,
            "knocked_over": sum(r["knocked_over"] for r in rows), "bins": bins,
            "worst_bin": None if worst is None else [worst[0], *worst[1]]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grasps", nargs="+", required=True)
    parser.add_argument("--belt", default="grasp_belt_v1")
    parser.add_argument("--floor", default="grasp_floor_v1")
    parser.add_argument("--results", type=Path, default=Path("results/grasp"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = {"belt_set": args.belt, "floor_set": args.floor, "grasps": {}, "paired": {}}
    pooled: dict = {}
    for grasp in args.grasps:
        seeds = [None] if grasp == "scripted" else [0, 1, 2]
        g = report["grasps"][grasp] = {}
        for check, set_name, size in (("belt", args.belt, False), ("floor", args.floor, False),
                                      ("floor_size", args.floor, True)):
            per_seed = {("all" if s is None else f"seed{s}"):
                        load(args.results, set_name, grasp, s, size) for s in seeds}
            per_seed = {k: v for k, v in per_seed.items() if v}
            if not per_seed:
                continue
            rows = [r for v in per_seed.values() for r in v]
            pooled[(grasp, check)] = per_seed
            entry = {"pooled": stats(rows), "per_seed": {k: stats(v) for k, v in per_seed.items()}}
            ps = entry["per_seed"].values()
            if check == "belt":
                entry["bar_met"] = all(s["rate"] >= BELT_MIN
                                       and s["knocked_over"] <= BELT_MAX_KNOCKED for s in ps)
            elif check == "floor":
                entry["bar_met"] = all(s["rate"] >= FLOOR_MIN
                                       and s["worst_bin"][1] / s["worst_bin"][2] >= FLOOR_BIN_MIN
                                       for s in ps)
            g[check] = entry
        if "floor" in g and "floor_size" in g:
            gap = g["floor_size"]["pooled"]["rate"] - g["floor"]["pooled"]["rate"]
            g["floor_size"]["gap"] = gap
            g["floor_size"]["bar_met"] = gap >= -SIZE_GAP
    first = args.grasps[0]
    for other in args.grasps[1:]:
        for check in ("belt", "floor", "floor_size"):
            a, b = pooled.get((first, check)), pooled.get((other, check))
            if not a or not b:
                continue
            # the scripted pick has one run: each seed of the other grasp is paired with it
            keys = sorted(a)
            sa = [r["success"] for k in keys for r in a[k]]
            sb = [r["success"] for k in keys for r in b.get(k, b.get("all", []))]
            report["paired"][f"{first} vs {other}: {check}"] = mcnemar_exact(sa, sb)

    for grasp, g in report["grasps"].items():
        print(f"== {grasp}")
        for check, e in g.items():
            p = e["pooled"]
            seeds = "; ".join(f"{k} {100 * s['rate']:.1f} % ko {s['knocked_over']} worst bin "
                              f"{s['worst_bin'][0]} {s['worst_bin'][1]}/{s['worst_bin'][2]}"
                              for k, s in e["per_seed"].items())
            bar = {True: "bar met", False: "BAR MISSED"}.get(e.get("bar_met"), "")
            gap = f" gap {100 * e['gap']:+.1f}" if "gap" in e else ""
            print(f"  {check:10s} {p['success']}/{p['n']} = {100 * p['rate']:.1f} % "
                  f"(knocked over {p['knocked_over']}){gap} {bar}\n      {seeds}")
    for k, v in report["paired"].items():
        print(f"  {k}: only first {v['only_first']}, only second {v['only_second']}, "
              f"p = {v['p_value']:.3g}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
