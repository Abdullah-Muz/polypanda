"""Oracle check: does the privileged scripted door expert (control/scripted_door.py) solve the door
task through DoorEnv's own action space, on randomized machines?

Usage: uv run python scripts/check_door_expert.py [--episodes 50] [--seed 0] [--config ...]
Prints the success rate per command, the failures (with their machine variants), and writes one
row per episode to --out (CSV) if given.
"""

import argparse
import csv
import time
from collections import Counter
from pathlib import Path

from polypanda.control.scripted_door import run_expert
from polypanda.environments.door import COMMANDS, DoorConfig, DoorEnv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    cfg = DoorConfig.from_yaml(args.config) if args.config else DoorConfig()
    env = DoorEnv(cfg)
    rows, start = [], time.perf_counter()
    for c, command in enumerate(COMMANDS):
        for k in range(args.episodes):
            seed = args.seed + 100_000 * c + k
            info = run_expert(env, seed=seed, options={"command": command})
            ep = info["episode"]
            rows.append({"command": command, "seed": seed, "success": info["is_success"],
                         "failure": info["failure"], "steps": info["steps"],
                         "return": round(info["return"], 2), "phase": info["phase"],
                         "lid": round(info["lid"], 4), "needed": info["needed"],
                         "latched": info["latched"],
                         "collision_steps": info["stats"]["collision_steps"],
                         "slams": info["stats"]["slams"],
                         "peak_force": round(info["stats"]["peak_force"], 1),
                         **{k2: (round(v, 4) if isinstance(v, float) else v)
                            for k2, v in ep.items() if k2 not in ("handle_bias", "jitter",
                                                                  "command")}})
    wall = time.perf_counter() - start
    for command in COMMANDS:
        sel = [r for r in rows if r["command"] == command]
        ok = sum(r["success"] for r in sel)
        steps = sum(r["steps"] for r in sel if r["success"]) / max(ok, 1)
        print(f"{command}: {ok}/{len(sel)} succeed ({100 * ok / len(sel):.0f} %), "
              f"{steps:.0f} steps on average when they do; failures: "
              f"{dict(Counter(r['failure'] for r in sel if not r['success']))}")
        for r in sel:
            if not r["success"]:
                print("   ", {k: r[k] for k in ("seed", "failure", "phase", "lid", "needed",
                                                "slide", "travel", "friction", "damping",
                                                "latch_speed", "handle_height", "turn",
                                                "peak_force")})
    total_steps = sum(r["steps"] for r in rows)
    print(f"{len(rows)} episodes, {total_steps} steps in {wall:.0f} s "
          f"({total_steps / wall:.0f} steps/s)")
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
