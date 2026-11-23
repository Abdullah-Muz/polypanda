"""v4.0 Phase 1: the variation sweep of the two scripted baselines (evaluation/tending.py).

For every property in evaluation.tending.LEVELS and every level, N trials; each trial's machine
is drawn once (its random direction or sign from the trial's seed) and both scripts run on it,
paired. Success = the machine is running at the end (part seated, lid latched).

Usage:
  uv run python scripts/sweep_machine_variants.py                 # all properties, N = 10
  uv run python scripts/sweep_machine_variants.py --n 2 --props latch --jobs 4 --out-dir /tmp/s
Writes <out-dir>/machine_variant_sweep.json (per property, level and script, with provenance)
and _trials.csv.
"""

import argparse
import csv
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

from polypanda.evaluation.reach import wilson_interval
from polypanda.evaluation.tending import LEVELS, make_variant, run_trial

SEED = 20261123


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


def job(args) -> list[dict]:
    prop, level, trial = args
    seed = int(np.random.SeedSequence([SEED, list(LEVELS).index(prop), level, trial])
               .generate_state(1)[0])
    variant = make_variant(prop, level, np.random.default_rng(seed))
    return [run_trial(variant, script, seed + k) for k, script in enumerate(("nominal", "camera"))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=10, help="trials per property and level")
    parser.add_argument("--props", nargs="+", default=list(LEVELS), choices=list(LEVELS))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=Path("results/v4"))
    args = parser.parse_args()

    git = git_state()   # recorded before running
    jobs = [(p, lv, t) for p in args.props for lv in range(len(LEVELS[p])) for t in range(args.n)]
    t0 = time.perf_counter()
    with Pool(args.jobs) as pool:
        rows = [r for rs in pool.imap(job, jobs, chunksize=1) for r in rs]
    summary: dict = {}
    for p in args.props:
        summary[p] = []
        for lv, value in enumerate(LEVELS[p]):
            entry = {"level": lv, "value": value}
            for script in ("nominal", "camera"):
                rs = [r for r in rows
                      if r["prop"] == p and r["level"] == lv and r["script"] == script]
                ok = sum(r["success"] for r in rs)
                fails: dict = {}
                for r in rs:
                    if not r["success"]:
                        fails[r["failure"]] = fails.get(r["failure"], 0) + 1
                entry[script] = {"n": len(rs), "success": ok, "rate": ok / len(rs) if rs else None,
                                 "ci95": wilson_interval(ok, len(rs)), "failures": fails,
                                 "machine_contacts": sum(r["machine_contacts"] > 0 for r in rs)}
            summary[p].append(entry)
    record = {"name": "machine_variant_sweep", "seed": SEED, "n_per_level": args.n,
              "levels": {p: LEVELS[p] for p in args.props}, "summary": summary,
              "success": "machine running at the end: part seated, lid latched",
              "camera_error": {"xy_m": 0.001, "z_m": 0.001, "yaw_deg": 0.5},
              "created_utc": datetime.now(UTC).isoformat(timespec="seconds"), "git": git,
              "versions": {"python": platform.python_version(), "mujoco": mujoco.__version__},
              "wall_s": round(time.perf_counter() - t0, 1)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "machine_variant_sweep.json").write_text(json.dumps(record, indent=2) + "\n")
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.out_dir / "machine_variant_sweep_trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    for p in args.props:
        print(f"\n{p}")
        for e in summary[p]:
            n, c = e["nominal"], e["camera"]
            print(f"  {str(e['value']):>6}: nominal {n['success']:2d}/{n['n']}"
                  f" {n['failures'] or ''}   camera {c['success']:2d}/{c['n']}"
                  f" {c['failures'] or ''}")
    print(f"\nwrote {args.out_dir}/machine_variant_sweep.json ({record['wall_s']} s)")


if __name__ == "__main__":
    main()
