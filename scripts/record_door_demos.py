"""v4.1: record a door script's demonstrations as a committed clone config says (its `record`
section: training/bc.RecordConfig; e.g. configs/door_v3_bc_r1.yaml).

Each episode is drawn from the config's env (its training distribution as evaluated: without the
training-only focus cells) with its own seed, seed * 1000 + k (a stream apart from the door sets',
which draw from np.random.default_rng(<the set's seed>)); with the configured noise on half of
them (DART-style, as v4.0's), the script's clean action recorded. Successful episodes only.

With a teacher (RecordConfig.teacher, a trained run per clone seed), the teacher's deterministic
policy is tried first on each episode and its successes are kept as they are; the script (with
the noise) runs on the others.

Usage:
  uv run python scripts/record_door_demos.py --config configs/door_v3_bc_r1.yaml
  uv run python scripts/record_door_demos.py --config configs/door_v3_bc_r1_mix.yaml --seed 0
Writes the demonstrations to the config's `demos` path (runs/, not committed; refused if it
exists) and results/v4/door/demos_<config stem>.json (committed: the git state, counts and
success by kind x command, with and without noise, failures, one row per episode).
"""

import argparse
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from polypanda.training.bc import BCRunConfig, record


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--episodes", type=int, help="override (smoke runs only; writes elsewhere)")
    p.add_argument("--jobs", type=int, help="override the config's worker count")
    p.add_argument("--out-dir", type=Path, default=Path("results/v4/door"))
    p.add_argument("--seed", type=int,
                   help="the clone seed, for a config with a per-seed teacher ({seed})")
    args = p.parse_args()
    git = git_state()
    bc = BCRunConfig.from_yaml(args.config)
    if bc.record is None:
        raise SystemExit(f"{args.config} has no record section")
    rec = bc.record
    per_seed = "{seed}" in bc.demos or "{seed}" in (rec.teacher or "")
    if per_seed and args.seed is None:
        raise SystemExit(f"{args.config} records per clone seed: give --seed")
    tag = f"_seed{args.seed}" if per_seed else ""
    if rec.teacher:
        rec = replace(rec, teacher=rec.teacher.format(seed=args.seed))
    demos = Path(bc.demos.format(seed=args.seed))
    out = args.out_dir / f"demos_{args.config.stem}{tag}.json"
    if args.episodes is not None:
        rec = replace(rec, episodes=args.episodes)
        demos = demos.with_name(f"{demos.stem}_smoke{args.episodes}.npz")
        out = out.with_name(f"{out.stem}_smoke{args.episodes}.json")
    if args.jobs is not None:
        rec = replace(rec, jobs=args.jobs)
    if demos.exists():
        raise SystemExit(f"{demos} exists")
    start = time.perf_counter()
    rows: list[dict] = []
    data = record(rec, rows)
    wall = time.perf_counter() - start
    demos.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(demos, **data)
    digest = hashlib.sha256(demos.read_bytes()).hexdigest()[:12]

    def group(key):
        g = defaultdict(lambda: [0, 0])
        for r in rows:
            g[key(r)][0] += r["success"]
            g[key(r)][1] += 1
        return {k: f"{v[0]}/{v[1]}" for k, v in sorted(g.items())}
    failures = defaultdict(int)
    for r in rows:
        if not r["success"]:
            failures[f"{r['kind']}/{r['command']}/{r['failure']}"] += 1
    ok = [r["steps"] for r in rows if r["success"]]
    summary = {
        "episodes": len(rows), "kept": int(data["kept"]), "rate": data["kept"] / len(rows),
        "steps": int(len(data["obs"])),
        "steps_to_success_median": float(np.median(ok)) if ok else None,
        "kind_command": group(lambda r: f"{r['kind']}/{r['command']}"),
        "noisy": group(lambda r: "noisy" if r["noisy"] else "clean"),
        "failures": dict(sorted(failures.items())),
    }
    if rec.teacher:
        summary["kept_by_source"] = group(lambda r: f"{r['source']}/{r['kind']}/{r['command']}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "what": f"{rec.script} demonstrations on training episodes ({rec.env_config}, task "
                f"{rec.task}), for {args.config}",
        "config": str(args.config), "record": asdict(rec), "git": git,
        "demos": str(demos), "demos_sha256_12": digest, "wall_s": round(wall, 1),
        "summary": summary, "rows": rows}, indent=1))
    print(f"{rec.script}: {summary['kept']}/{summary['episodes']} kept "
          f"({100 * summary['rate']:.1f} %), {summary['steps']} steps; "
          f"{summary.get('kept_by_source', '')}; by kind/command "
          f"{summary['kind_command']}; {summary['noisy']}")
    print(f"   failures: {summary['failures']}")
    print(f"   wrote {demos} ({digest}) and {out} ({wall:.0f} s)")


if __name__ == "__main__":
    main()
