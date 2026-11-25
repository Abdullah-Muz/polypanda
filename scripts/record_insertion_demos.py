"""v4.1: record the frozen search script's demonstrations in InsertionEnv's spaces, and check that
they replay (docs/insertion_env.md, "Demonstrations").

On freshly drawn **training** episodes (their own stream: SeedSequence(seed,
spawn_key=(DEMO_SPAWN_KEY,)), never an evaluation set), each episode is run twice:
1. the script on its own control path, unchanged (evaluation/insertion.record_script), its
   commands recorded as the env's actions (the hand target's motion and the release);
2. those actions stepped through the env from the same start (evaluation/insertion.replay).
An episode **matches** when the replay ends as the script did (success or not). Episodes whose
script run went past the env's 30 s budget (the search's own 30 s timeout comes before its
release and rise) cannot match a successful script run; they are counted apart. The replayed
episodes are the demonstrations: their observations (as the env gives them), the recorded
actions, rewards and episode ends, written to runs/insertion_demos/ for behaviour cloning
(successful replays only, by default).

Usage:
  uv run python scripts/record_insertion_demos.py --episodes 200 --jobs 3
Writes results/v4/insertion/demo_replay_<script>_<episodes>_<seed>.json (the match rate, per
episode) and runs/insertion_demos/<script>_<episodes>_<seed>.npz.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from polypanda.evaluation.insertion import DEMO_SPAWN_KEY, NOISE_SEED

DEMO_SEED = 20261132


def _chunk(args) -> list[dict]:
    script, items, pool_path = args
    import torch
    torch.set_num_threads(1)
    from dataclasses import replace

    from polypanda.environments.insertion import InsertionConfig, InsertionEnv
    from polypanda.evaluation.insertion import record_script, replay
    env = InsertionEnv(replace(InsertionConfig(), pick_pool=pool_path))
    out = []
    for k, ep in items:
        ep = env.complete_episode(ep)
        env.reset(seed=NOISE_SEED + k, options={"episode": ep})
        rec = record_script(env, script)
        rp = replay(env, ep, rec["actions"], seed=NOISE_SEED + k)
        s, r = rec["outcome"], rp["outcome"]
        out.append({
            "index": k, "tester": ep["tester"],
            "script_success": s["success"], "script_failure": s["failure"],
            "script_steps": s["steps"], "script_time_s": s["time_s"],
            "replay_success": r["success"], "replay_failure": r["failure"],
            "replay_steps": r["steps"], "replay_time_s": r["time_s"],
            "match": s["success"] == r["success"],
            "identical": bool(abs(s["peak_force"] - r["peak_force"]) < 1e-9
                              and s["seat"] == r["seat"]),
            "over_budget": s["steps"] > env.config.horizon,
            "exceeded": rec["exceeded"], "lead_clips": r["lead_clips"],
            "obs": rp["obs"][:-1], "actions": rp["actions"], "rewards": rp["rewards"],
        })
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--script", default="search", choices=["search", "straight_down"])
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=DEMO_SEED)
    p.add_argument("--pick-pool", default=None,
                   help="draw grasps from this pool (default: simulate each episode's pick)")
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--all", action="store_true", help="keep failed replays in the demos too")
    p.add_argument("--out-dir", type=Path, default=Path("results/v4/insertion"))
    p.add_argument("--demo-dir", type=Path, default=Path("runs/insertion_demos"))
    args = p.parse_args()
    git = {"commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True).stdout.strip(),
           "dirty": bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                        capture_output=True, text=True).stdout.strip())}
    from dataclasses import replace

    from polypanda.environments.insertion import InsertionConfig, InsertionEnv
    env = InsertionEnv(replace(InsertionConfig(), pick_pool=args.pick_pool))
    rng = np.random.default_rng(np.random.SeedSequence(args.seed, spawn_key=(DEMO_SPAWN_KEY,)))
    drawn = [env.draw_episode(rng) for _ in range(args.episodes)]
    chunks = [(args.script, [(k, drawn[k]) for k in range(j, args.episodes, args.jobs)],
               args.pick_pool) for j in range(args.jobs)]
    start = time.perf_counter()
    with Pool(args.jobs) as pool:
        rows = sorted((r for part in pool.map(_chunk, chunks) for r in part),
                      key=lambda r: r["index"])
    wall = time.perf_counter() - start
    n = len(rows)
    within = [r for r in rows if not r["over_budget"]]
    summary = {
        "episodes": n,
        "script_success": sum(r["script_success"] for r in rows),
        "replay_success": sum(r["replay_success"] for r in rows),
        "match": sum(r["match"] for r in rows), "match_rate": sum(r["match"] for r in rows) / n,
        "identical": sum(r["identical"] for r in rows),
        "within_budget": len(within),
        "match_within_budget": sum(r["match"] for r in within),
        "match_rate_within_budget": (sum(r["match"] for r in within) / len(within)
                                     if within else None),
        "steps_beyond_one_action": sum(r["exceeded"] for r in rows),
        "lead_clips": sum(r["lead_clips"] for r in rows),
    }
    keep = [r for r in rows if args.all or r["replay_success"]]
    args.demo_dir.mkdir(parents=True, exist_ok=True)
    demo_path = args.demo_dir / f"{args.script}_{args.episodes}_{args.seed}.npz"
    if keep:
        done = [np.eye(1, len(r["actions"]), len(r["actions"]) - 1, dtype=bool)[0] for r in keep]
        np.savez_compressed(
            demo_path, obs=np.concatenate([r["obs"] for r in keep]),
            act=np.concatenate([r["actions"] for r in keep]).astype(np.float32),
            rew=np.concatenate([r["rewards"] for r in keep]),
            episode=np.concatenate([np.full(len(r["actions"]), r["index"]) for r in keep]),
            done=np.concatenate(done), kept=len(keep), total=n)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"demo_replay_{args.script}_{args.episodes}_{args.seed}.json"
    out.write_text(json.dumps({
        "what": "the search's demonstrations replayed in InsertionEnv (training episodes)",
        "script": args.script, "seed": args.seed, "spawn_key": DEMO_SPAWN_KEY, "git": git,
        "wall_s": round(wall, 1), "summary": summary, "demos": str(demo_path),
        "rows": [{k: v for k, v in r.items() if k not in ("obs", "actions", "rewards")}
                 for r in rows]}, indent=1))
    s = summary
    print(f"{args.script}: {n} training episodes; script {s['script_success']} successes, "
          f"replay {s['replay_success']}; outcome match {s['match']}/{n} "
          f"({100 * s['match_rate']:.1f} %), within the 30 s budget "
          f"{s['match_within_budget']}/{s['within_budget']}; identical runs {s['identical']}; "
          f"steps beyond one action {s['steps_beyond_one_action']}, lead clips {s['lead_clips']}")
    print(f"   wrote {out} and {demo_path} ({len(keep)} episodes, {wall:.0f} s)")


if __name__ == "__main__":
    main()
