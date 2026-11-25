"""v4.0: evaluate a door policy on a committed door set (docs/v4_targets.md).

Policies: the scripted baselines `nominal`, `camera`, `feel`, `feel_v2` (feel plus the door-open
switch; control/door_baselines.py), the privileged `expert` (control/scripted_door.py, an upper
reference), or a trained run (`run:<path>` with its best model; `run:<path>@final` for the final
one), run on the environment of its task (door: DoorEnv, door_v2: DoorV2Env).

Usage:
  uv run python scripts/evaluate_door.py --policy feel --set eval_sets/door_dev_v1.json
Writes results/v4/door/<policy>_<set>.json: one row per episode and a summary (success with its
95 % Wilson interval, per command x slide x spring cell, steps to success, peak force, slams).

v4.1: a set with drawers (episodes with "kind", scripts/make_door_sets.py --kinds) runs in a scene
with the kinds it holds; a drawer's slide is "toward" (the robot) in the cells.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SCRIPTS = ("nominal", "camera", "feel", "feel_v2", "expert")
NOISE_SEED = 7_000_000


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def _make_actor(policy: str, env):
    """A callable (obs) -> action and a reset hook, for one episode at a time."""
    if policy in SCRIPTS:
        from polypanda.control.door_baselines import BaselineDoor
        from polypanda.control.scripted_door import DoorExpert
        script = DoorExpert(env) if policy == "expert" else BaselineDoor(env, policy)
        return (lambda obs: script.act()), script.reset, script
    if policy.startswith("run:"):
        from polypanda.evaluation.policies import load_run_policy
        act = load_run_policy(policy[4:])
        return act, (lambda: None), None
    raise ValueError(f"unknown policy {policy!r}")


def _run_chunk(args) -> list[dict]:
    policy, episodes, indices = args
    from dataclasses import replace

    from polypanda.environments.door import DOOR_KINDS, DoorConfig, DoorEnv
    from polypanda.evaluation.policies import run_env_cls
    # a set with drawers (v4.1: episodes with "kind") needs a scene with them; v4.0's sets have
    # lids only, and run in v4.0's scene
    kinds = tuple(k for k in DOOR_KINDS if any(ep.get("kind", "lid") == k for ep in episodes))
    config = (None if kinds == ("lid",) else
              replace(DoorConfig(), variation=replace(DoorConfig().variation, kinds=kinds)))
    env = run_env_cls(policy[4:])(config) if policy.startswith("run:") else DoorEnv(config)
    act, reset, script = _make_actor(policy, env)
    rows = []
    for i in indices:
        ep = episodes[i]
        # the sensors' noise is seeded per episode, so a result does not depend on how the
        # episodes are split between workers
        obs, info = env.reset(seed=NOISE_SEED + i, options={"episode": ep})
        reset()
        done, steps, total = False, 0, 0.0
        while not done:
            obs, r, term, trunc, info = env.step(act(obs))
            total += r
            steps += 1
            done = term or trunc
        st = info["stats"]
        rows.append({
            "index": i, "command": ep["command"], "slide": ep["slide"],
            "spring": ep["spring"] > 0, "success": bool(info["is_success"]),
            "failure": info["failure"], "steps": steps, "return": round(total, 3),
            "peak_force": round(st["peak_force"], 2), "slams": int(st["slams"]),
            "collision_steps": int(st["collision_steps"]),
            "flipped": bool(getattr(script, "flipped", False)),
            "retries": int(getattr(script, "tries", 0)),
        })
        if "kind" in ep:              # (v4.1 sets: the tester's kind)
            rows[-1]["kind"] = ep["kind"]
    return rows


def summarize(rows: list[dict]) -> dict:
    n, k = len(rows), sum(r["success"] for r in rows)
    cells = defaultdict(lambda: [0, 0])
    for r in rows:
        key = f"{r['command']}/{r['slide']}/{'spring' if r['spring'] else 'plain'}"
        cells[key][0] += r["success"]
        cells[key][1] += 1
    failures = defaultdict(int)
    for r in rows:
        if not r["success"]:
            failures[f"{r['command']}/{r['slide']}/{r['failure']}"] += 1
    ok_steps = [r["steps"] for r in rows if r["success"]]
    out = {
        "n": n, "success": k, "rate": k / n if n else 0.0, "wilson95": wilson(k, n),
        "cells": {c: {"success": v[0], "n": v[1], "rate": v[0] / v[1]}
                  for c, v in sorted(cells.items())},
        "worst_cell_rate": min(v[0] / v[1] for v in cells.values()) if cells else 0.0,
        "steps_to_success_median": float(np.median(ok_steps)) if ok_steps else None,
        "peak_force_median": float(np.median([r["peak_force"] for r in rows])),
        "slam_episodes": sum(r["slams"] > 0 for r in rows),
        "failures": dict(sorted(failures.items())),
    }
    if any("kind" in r for r in rows):
        # v4.1 sets (docs/v4_1_targets.md, D1): per kind x command, and the largest peak force
        groups = defaultdict(list)
        for r in rows:
            groups[f"{r.get('kind', 'lid')}/{r['command']}"].append(r)
        out["kind_command"] = {g: brief(rs) for g, rs in sorted(groups.items())}
        out["peak_force_max"] = float(max(r["peak_force"] for r in rows))
    return out


def brief(rows: list[dict]) -> dict:
    """Success (with its Wilson interval), median steps to success, median and largest peak
    force, and slam episodes of a group of rows."""
    n, k = len(rows), sum(r["success"] for r in rows)
    ok_steps = [r["steps"] for r in rows if r["success"]]
    force = [r["peak_force"] for r in rows]
    return {"n": n, "success": k, "rate": k / n if n else 0.0, "wilson95": wilson(k, n),
            "steps_to_success_median": float(np.median(ok_steps)) if ok_steps else None,
            "peak_force_median": float(np.median(force)) if force else None,
            "peak_force_max": float(max(force)) if force else None,
            "slam_episodes": sum(r["slams"] > 0 for r in rows)}


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--set", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--n", type=int, help="only the first n episodes (smoke runs)")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    git = git_state()          # recorded before running, so later commits cannot mislabel it
    data = json.loads(args.set.read_text())
    episodes = data["episodes"][: args.n] if args.n else data["episodes"]
    jobs = [(args.policy, episodes, list(range(j, len(episodes), args.jobs)))
            for j in range(args.jobs)]
    start = time.perf_counter()
    with Pool(args.jobs) as pool:
        rows = sorted((r for rs in pool.map(_run_chunk, jobs) for r in rs),
                      key=lambda r: r["index"])
    wall = time.perf_counter() - start
    summary = summarize(rows)
    name = args.policy.replace("run:", "").replace("/", "_").replace("@", "_")
    out = args.out or Path("results/v4/door") / f"{name}_{data['name']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"policy": args.policy, "set": data["name"],
                               "set_digest": data["digest"], "git": git,
                               "wall_s": round(wall, 1), "summary": summary, "rows": rows},
                              indent=1))
    lo, hi = summary["wilson95"]
    print(f"{args.policy} on {data['name']}: {summary['success']}/{summary['n']} "
          f"({100 * summary['rate']:.1f} %, 95 % CI {100 * lo:.1f}-{100 * hi:.1f}), worst cell "
          f"{100 * summary['worst_cell_rate']:.0f} %, median steps to success "
          f"{summary['steps_to_success_median']}, peak force median "
          f"{summary['peak_force_median']:.0f} N, slam episodes {summary['slam_episodes']}")
    for g, b in summary.get("kind_command", {}).items():
        print(f"   {g}: {b['success']}/{b['n']}, median steps {b['steps_to_success_median']}, "
              f"peak force median {b['peak_force_median']:.1f} / max {b['peak_force_max']:.1f} N, "
              f"slams {b['slam_episodes']}")
    print("   failures:", summary["failures"])
    print(f"   wrote {out} ({wall:.0f} s)")


if __name__ == "__main__":
    main()
