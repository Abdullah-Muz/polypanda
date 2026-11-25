"""v4.1: evaluate an insertion on a committed insertion set (docs/v4_1_targets.md, Part I).

Policies: the frozen scripts `straight_down` and `search` (control/scripted_insertion.py, run on
the env's episodes through the cell's own control path: evaluation/insertion.run_script), or a
trained run (`run:<path>` with its final model by default, the targets' rule; `run:<path>@best`
for the best one), run on InsertionEnv.

Usage:
  uv run python scripts/evaluate_insertion.py --policy search --set eval_sets/insertion_dev_v1.json
Writes results/v4/insertion/<policy>_<set>.json: the git commit (and whether the tree was dirty),
one row per episode (the episode's factors, success or why not, time to the hand clear, peak nest
force at every physics step and at the control steps, the script's own record) and a summary
(success with its 95 % Wilson interval, success within the 30 s budget, time to success, peak
nest force, failures, by tester, camera error, clearance and chamfer).

The one-shot sets (`*_final_*`, `*stretch*`) are refused unless --one-shot is given: they are run
once, after a lock commit (docs/v4_1_targets.md, Training rules).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

from polypanda.evaluation.insertion import NOISE_SEED, SCRIPTS, episode_row, summarize


def _run_chunk(args) -> list[dict]:
    policy, episodes, indices = args
    import torch
    torch.set_num_threads(1)
    from polypanda.environments.insertion import InsertionEnv
    from polypanda.evaluation.insertion import run_policy, run_script
    env = InsertionEnv()
    act = None
    if policy.startswith("run:"):
        from polypanda.evaluation.policies import load_run_policy
        spec = policy[4:]
        act = load_run_policy(spec if "@" in spec else f"{spec}@final")
    rows = []
    for i in indices:
        ep = episodes[i]
        env.reset(seed=NOISE_SEED + i, options={"episode": ep})
        out = run_script(env, policy) if act is None else run_policy(env, act)
        rows.append(episode_row(i, ep, out))
    return rows


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


def one_shot(name: str) -> bool:
    return "_final_" in name or name.endswith("_final") or "stretch" in name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True,
                   help=f"one of {SCRIPTS}, or run:<path>[@final|@best] (default final)")
    p.add_argument("--set", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--n", type=int, help="only the first n episodes (smoke runs)")
    p.add_argument("--out", type=Path)
    p.add_argument("--one-shot", action="store_true",
                   help="allow a one-shot set (final, stretch): only after the lock commit")
    args = p.parse_args()
    if args.policy not in SCRIPTS and not args.policy.startswith("run:"):
        raise SystemExit(f"unknown policy {args.policy!r}")
    git = git_state()          # recorded before running, so later commits cannot mislabel it
    data = json.loads(args.set.read_text())
    if one_shot(data["name"]) and not args.one_shot:
        raise SystemExit(f"{data['name']} is a one-shot set: run it only after the lock commit "
                         "(--one-shot)")
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
    suffix = f"_n{args.n}" if args.n else ""
    out = args.out or Path("results/v4/insertion") / f"{name}_{data['name']}{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"policy": args.policy, "set": data["name"],
                               "set_digest": data["digest"], "n": len(rows), "git": git,
                               "wall_s": round(wall, 1), "summary": summary, "rows": rows},
                              indent=1))
    s = summary
    lo, hi = s["wilson95"]
    t, f = s["time_to_success_s"] or {}, s["peak_force_n"] or {}
    print(f"{args.policy} on {data['name']}: {s['success']}/{s['n']} ({100 * s['rate']:.1f} %, "
          f"95 % CI {100 * lo:.1f}-{100 * hi:.1f}); within 30 s {s['success_within_budget']}; "
          f"time to success median {t.get('median')} s; peak nest force median "
          f"{f.get('median')} N, max {f.get('max')} N; over 50 N {s['over_damage_limit']}")
    print("   failures:", s["failures"])
    print("   by tester:", {k: f"{v['success']}/{v['n']}" for k, v in s["by_tester"].items()})
    print(f"   wrote {out} ({wall:.0f} s)")


if __name__ == "__main__":
    main()
