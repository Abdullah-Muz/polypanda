"""v4.1: run shifts of the electronics test cell and report its KPIs (evaluation/test_cell.py).

One shift per seed (development seeds only: the cell's one-shot shifts come with
docs/v4_1_targets.md), in parallel. Each shift draws its modules' hidden truths from its seed,
runs the scripted cell controller with the door and insertion actors given, and records every
module, activity, event and tester state.

Usage:
  uv run python scripts/run_test_cell.py --n 30 --seeds 1 2 --door feel_v2 --jobs 2
  (--insertion search once control/scripted_insertion.py is merged)
Writes results/v4/test_cell/<door>_<insertion>_n<n>_seeds<...>.json (with the git commit) and
prints the KPIs of each shift and of all together.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path


def _shift(args: dict) -> dict:
    from polypanda.evaluation.test_cell import run_shift
    t = time.perf_counter()
    out = run_shift(**args)
    out["wall_s"] = round(time.perf_counter() - t, 1)
    return out


def pooled(shifts: list[dict]) -> dict:
    """The shifts' KPIs pooled: counts summed, rates over the sums."""
    k = [s["kpis"] for s in shifts]
    span = sum(x["shift_s"] for x in k)
    total = {f: sum(x[f] for x in k) for f in ("modules", "taken", "sorted", "tested",
                                              "sort_correct", "escapes", "false_rejects")}
    mods = [m for s in shifts for m in s["modules"]]
    taken = [m for m in mods if m["t_taken"] is not None]
    fp = sum(m["sorted_to"] == "pass_tray" and not m["failures"] and len(m["inserts"]) == 1
             and (m["scan"] or {}).get("tries", 1) == 1 for m in taken)
    inserts = [i for m in mods for i in m["inserts"]]
    first = [i["seated"] for i in inserts if i["attempt"] == 0]
    util = [u for x in k for u in x["utilization"].values()]
    return total | {
        "shift_s": round(span, 1),
        "modules_per_hour": round(total["sorted"] / span * 3600.0, 1) if span else None,
        "utilization_mean": round(sum(util) / len(util), 4) if util else None,
        "first_pass_yield": round(fp / len(taken), 4) if taken else None,
        "sort_correct_rate": round(total["sort_correct"] / max(total["modules"], 1), 4),
        "insertion_first_try_rate": round(sum(first) / len(first), 4) if first else None,
        "failures_by_stage": dict(sum((Counter(x["failures_by_stage"]) for x in k), Counter())),
        "false_rejects_by_reason": dict(sum((Counter(x["false_rejects_by_reason"]) for x in k),
                                            Counter())),
        "caught": dict(sum((Counter(x["caught"]) for x in k), Counter())),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n", type=int, default=30, help="modules per shift")
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--door", default="feel_v2",
                   help="line_camera, a door script (feel_v2, feel, camera, nominal, expert) or "
                        "run:<path>[@final]")
    p.add_argument("--insertion", choices=("straight_down", "search"), default="straight_down",
                   help="the insertion slot (search: control/scripted_insertion.py, once merged)")
    p.add_argument("--handover", choices=("carry", "door_start"), default="door_start")
    p.add_argument("--nest-belief", choices=("camera", "true"), default="camera")
    p.add_argument("--cycle-s", type=float, default=45.0, help="each test's duration (s)")
    p.add_argument("--no-dressing", action="store_true", help="the bare cell (faster)")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    run = lambda *c: subprocess.run(c, capture_output=True, text=True).stdout.strip()  # noqa: E731
    git = {"commit": run("git", "rev-parse", "HEAD"),
           "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}
    jobs = [{"n": args.n, "seed": s, "door": args.door, "insertion": args.insertion,
             "handover": args.handover,
             "nest_belief": args.nest_belief, "cycle_s": args.cycle_s,
             "dressing": not args.no_dressing} for s in args.seeds]
    start = time.perf_counter()
    with Pool(min(args.jobs, len(jobs))) as pool:
        shifts = pool.map(_shift, jobs)
    wall = time.perf_counter() - start
    summary = pooled(shifts)
    insertion = shifts[0]["config"]["insertion"]
    name = args.door.replace("run:", "").replace("/", "_").replace("@", "_")
    seeds = "-".join(str(s) for s in args.seeds)
    out = args.out or (Path("results/v4/test_cell")
                       / f"{name}_{insertion}_n{args.n}_seeds{seeds}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args) | {"out": str(out)}, "git": git,
                               "wall_s": round(wall, 1), "summary": summary,
                               "shifts": shifts}, indent=1, default=str))
    for s in shifts:
        k = s["kpis"]
        print(f"seed {s['config']['seed']}: {k['sorted']}/{k['modules']} sorted in "
              f"{k['shift_s']:.0f} s ({k['modules_per_hour']} /h), utilization "
              f"{k['utilization_mean']:.2f}, first-pass yield {k['first_pass_yield']}, "
              f"correct {k['sort_correct']}, escapes {k['escapes']}, false rejects "
              f"{k['false_rejects']} {k['false_rejects_by_reason']}, failures "
              f"{k['failures_by_stage']} ({s['wall_s']:.0f} s wall)")
    print("all:", json.dumps(summary))
    print(f"   wrote {out} ({wall:.0f} s)")


if __name__ == "__main__":
    main()
