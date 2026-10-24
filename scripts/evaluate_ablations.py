"""Supervisor ablations on the held-out pick-and-place scenes, paired with the full system.

Usage: uv run python scripts/evaluate_ablations.py --grasp scripted [--ablations ...]
  no_retries       one grasp attempt: failure detection without recovery
  timed_switching  skills switch after fixed times instead of on verified predicates (so no
                   failure detection or recovery either); the times are each stage's 90th
                   percentile in the full system's successful episodes for the same grasp skill
  no_upright       the place skill releases the cube without first standing it upright
Each result is paired with results/pickplace/pickplace_<tag>_pickplace_nominal_v1 (the full
system) by an exact McNemar test. Writes results/ablations/<tag>_<ablation>.json and
_episodes.csv.
"""

import argparse
import csv
import json
import platform
import subprocess
from dataclasses import asdict, fields, replace
from datetime import UTC, datetime
from pathlib import Path

import mujoco
import numpy as np
import torch

from polypanda.control.scripted_place import ScriptedPlace
from polypanda.evaluation.pickplace import (
    PickPlaceEpisode,
    evaluate_pickplace,
    grasp_driver_from_spec,
    grasp_env_for,
    mcnemar_exact,
)
from polypanda.evaluation.sets import PickPlaceEvalSet
from polypanda.evaluation.transport import ScriptedCarryDriver
from polypanda.supervisor.fsm import SupervisorConfig

ABLATIONS = ("no_retries", "timed_switching", "no_upright")


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


def read_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def stage_times(rows: list[dict]) -> tuple[float, float, float]:
    """90th percentile of each stage's duration in successful first-attempt episodes."""
    wins = [r for r in rows if r["success"] == "True" and r["grasp_attempts"] == "1"]
    return tuple(round(float(np.percentile([float(r[f"{s}_s"]) for r in wins], 90)), 2)
                 for s in ("grasp", "transport", "place"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grasp", default="scripted")
    parser.add_argument("--ablations", nargs="+", default=list(ABLATIONS), choices=ABLATIONS)
    parser.add_argument("--set", type=Path, default=Path("eval_sets/pickplace_nominal_v1.json"))
    parser.add_argument("--nominal-dir", type=Path, default=Path("results/pickplace"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/ablations"))
    args = parser.parse_args()
    torch.set_num_threads(1)

    eval_set = PickPlaceEvalSet.load(args.set)
    driver, description, tag, provenance = grasp_driver_from_spec(args.grasp)
    full = read_rows(args.nominal_dir / f"pickplace_{tag}_{eval_set.name}_episodes.csv")
    full_success = [r["success"] == "True" for r in full]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ablation in args.ablations:
        config, place = SupervisorConfig(), ScriptedPlace()
        if ablation == "no_retries":
            config = replace(config, max_grasp_attempts=1)
        elif ablation == "timed_switching":
            config = replace(config, timed_switching_s=stage_times(full))
        elif ablation == "no_upright":
            place = ScriptedPlace(upright=False)
        summary, episodes = evaluate_pickplace(driver, ScriptedCarryDriver(), place, eval_set,
                                               config, grasp_env_for(args.grasp))
        paired = mcnemar_exact([e.success for e in episodes], full_success)
        name = f"{tag}_{ablation}"
        record = {"name": name, "ablation": ablation,
                  "skills": {"grasp": description, "grasp_provenance": provenance,
                             "transport": "scripted carry",
                             "place": f"scripted place (upright={place.upright})"},
                  "supervisor": asdict(config),
                  "eval_set": {"name": eval_set.name, "n": len(eval_set),
                               "digest": eval_set.digest},
                  "paired_with_full_system": paired,
                  "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                  "git": git_state(),
                  "versions": {"python": platform.python_version(),
                               "mujoco": mujoco.__version__, "numpy": np.__version__,
                               "torch": torch.__version__},
                  "summary": summary}
        (args.out_dir / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
        with open(args.out_dir / f"{name}_episodes.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[fld.name for fld in fields(PickPlaceEpisode)])
            writer.writeheader()
            writer.writerows(asdict(e) for e in episodes)
        s = summary
        lo, hi = s["success_ci95"]
        full_rate = sum(full_success) / len(full_success)
        print(f"{name}: success {s['success_rate']:.1%} ({lo:.1%}-{hi:.1%}) vs full system "
              f"{full_rate:.1%}; only ablation {paired['only_first']}, only full "
              f"{paired['only_second']}, McNemar p = {paired['p_value']:.3g}; "
              f"failed at {s['failures_by_stage']}"
              + (f"; times {config.timed_switching_s}" if config.timed_switching_s else ""))


if __name__ == "__main__":
    main()
