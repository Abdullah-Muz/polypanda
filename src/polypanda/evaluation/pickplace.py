"""End-to-end pick-and-place evaluation through the supervisor.

Success = the supervisor reports a verified placement (PlacedCheck: resting, upright, within
2 cm of the target, still, clear of the gripper, for 0.5 s) within the global time budget,
after at most two re-grasps. Failures are attributed to the stage where the last attempt failed.
"""

import time
from collections import Counter
from dataclasses import dataclass
from math import comb
from pathlib import Path

import numpy as np
import yaml

from polypanda.control.scripted_place import ScriptedPlace
from polypanda.environments.grasp import GraspConfig, GraspEnv, GraspV4Env
from polypanda.evaluation.grasp import PolicyDriver, ScriptedAnyPoseDriver, ScriptedDriver
from polypanda.evaluation.reach import wilson_interval
from polypanda.evaluation.sets import PickPlaceEvalSet
from polypanda.evaluation.transport import ScriptedCarryDriver
from polypanda.policies.loader import load_run, make_run_env
from polypanda.supervisor.fsm import PickPlaceResult, PickPlaceSupervisor, SupervisorConfig

SCRIPTED_V4_CONFIG = Path("configs/grasp_anypose4_cartesian.yaml")  # the any-pose task (d1)


def grasp_env_for(spec: str):
    """The environment a grasp skill is evaluated in: GraspEnv for the scripted grasp,
    GraspV4Env on the any-pose task for the scripted any-pose grasp; for a trained run, its
    task's env with the run's own task configuration, as evaluated."""
    if spec == "scripted":
        return GraspEnv()
    if spec == "scripted-v4":
        return GraspV4Env(GraspConfig.from_yaml(SCRIPTED_V4_CONFIG).for_evaluation())
    run_dir = Path(spec[4:].partition("@")[0])
    meta_file = run_dir / "metadata.yaml"
    skill = (yaml.safe_load(meta_file.read_text())["skill"] if meta_file.exists()
             else yaml.safe_load((run_dir / "config.yaml").read_text())["task"])
    return make_run_env(run_dir, skill)


def grasp_driver_from_spec(spec: str):
    """'scripted', 'scripted-v4' or 'run:<run_dir>[@best|final]' -> (driver, description, tag,
    provenance)."""
    if spec == "scripted":
        return ScriptedDriver(), "scripted top-down grasp", "scripted", {}
    if spec == "scripted-v4":
        return (ScriptedAnyPoseDriver(), "scripted any-pose grasp (4-D cartesian actions)",
                "scripted_v4", {"task_config": str(SCRIPTED_V4_CONFIG)})
    if not spec.startswith("run:"):
        raise ValueError("grasp must be 'scripted', 'scripted-v4' or 'run:<run_dir>[@best|final]'")
    run_dir, _, which = spec[4:].partition("@")
    policy = load_run(run_dir, which or "best")
    meta = policy.metadata
    return (PolicyDriver(policy), f"PPO grasp policy {meta['policy_id']}",
            f"{Path(run_dir).name}_{policy.which}",
            {"run_dir": run_dir, "checkpoint": policy.which, "policy_id": meta["policy_id"],
             # runs still training have provisional metadata without a git record
             "trained_at": meta.get("git", {}).get("commit", "in progress")})


SCRIPTED_SKILLS = {
    "transport": (ScriptedCarryDriver, "scripted carry (closed-loop IK, gain 0.2)"),
    "place": (ScriptedPlace, "scripted place (upright correction, gated release)"),
}


def skill_from_spec(spec: str, stage: str):
    """A transport or place skill, 'scripted' or 'run:<run_dir>[@best|final|selected]' ->
    (driver, env, description, tag, provenance). env is None for the scripted skill (the
    supervisor's default environment); a trained run gets its own task configuration."""
    if spec == "scripted":
        driver_cls, description = SCRIPTED_SKILLS[stage]
        return driver_cls(), None, description, "scripted", {}
    if not spec.startswith("run:"):
        raise ValueError(f"{stage} must be 'scripted' or 'run:<run_dir>[@best|final|selected]'")
    run_dir, _, which = spec[4:].partition("@")
    policy = load_run(run_dir, which or "best")
    meta = policy.metadata
    if not meta["skill"].startswith(stage):
        raise ValueError(f"{run_dir} is a {meta['skill']} policy, not {stage}")
    return (PolicyDriver(policy), make_run_env(run_dir, meta["skill"]),
            f"{meta['algorithm']} {stage} policy {meta['policy_id']}",
            f"{Path(run_dir).name}_{policy.which}",
            {"run_dir": run_dir, "checkpoint": policy.which, "policy_id": meta["policy_id"],
             "trained_at": meta.get("git", {}).get("commit", "in progress")})


@dataclass(frozen=True)
class PickPlaceEpisode:
    index: int
    success: bool
    failure_stage: str | None
    failure_reason: str | None
    grasp_attempts: int
    total_s: float
    final_xy_error_mm: float | None
    grasp_s: float
    transport_s: float
    place_s: float
    home_s: float
    first_failure: str | None  # "stage:outcome" of the first failed attempt, if any
    trace: str                 # e.g. "grasp:ok transport:lost_grip home:ok grasp:ok ..."


def episode_from_result(index: int, r: PickPlaceResult, dt: float) -> PickPlaceEpisode:
    seconds = Counter()
    for rec in r.records:
        seconds[rec.stage] += rec.steps * dt
    first = next((f"{rec.stage}:{rec.outcome}" for rec in r.records
                  if rec.outcome != "ok"), None)
    return PickPlaceEpisode(
        index=index, success=r.success, failure_stage=r.failure_stage,
        failure_reason=r.failure_reason, grasp_attempts=r.grasp_attempts, total_s=r.total_s,
        final_xy_error_mm=r.final_xy_error_mm,
        **{f"{s}_s": round(seconds[s], 3) for s in ("grasp", "transport", "place", "home")},
        first_failure=first, trace=" ".join(f"{rec.stage}:{rec.outcome}" for rec in r.records))


def _pct(values, q):
    values = [v for v in values if v is not None]
    return float(np.percentile(values, q)) if values else None


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict:
    """Exact two-sided McNemar test on paired successes (binomial on the discordant pairs)."""
    only_a = sum(x and not y for x, y in zip(a, b, strict=True))
    only_b = sum(y and not x for x, y in zip(a, b, strict=True))
    n = only_a + only_b
    p = min(1.0, 2 * sum(comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2**n) if n else 1.0
    return {"only_first": only_a, "only_second": only_b, "p_value": p}


def summarize_pickplace(results: list[PickPlaceEpisode]) -> dict:
    n = len(results)
    wins = [r for r in results if r.success]
    retried = [r for r in results if r.grasp_attempts > 1]
    return {
        "episodes": n,
        "success_rate": len(wins) / n,
        "success_ci95": list(wilson_interval(len(wins), n)),
        "first_attempt_success_rate": sum(r.success and r.grasp_attempts == 1
                                          for r in results) / n,
        "failures_by_stage": dict(Counter(r.failure_stage for r in results if not r.success)),
        "failures": dict(Counter(r.failure_reason for r in results if not r.success)),
        "first_failures": dict(Counter(r.first_failure for r in results if r.first_failure)),
        "retries": {"episodes_with_retry": len(retried),
                    "recovered": sum(r.success for r in retried),
                    "attempts": dict(Counter(r.grasp_attempts for r in results))},
        "time_s": {"median": _pct([r.total_s for r in wins], 50),
                   "p90": _pct([r.total_s for r in wins], 90),
                   "by_stage_median": {s: _pct([getattr(r, f"{s}_s") for r in wins], 50)
                                       for s in ("grasp", "transport", "place")}},
        "final_xy_error_mm": {"median": _pct([r.final_xy_error_mm for r in wins], 50),
                              "p95": _pct([r.final_xy_error_mm for r in wins], 95)},
    }


def evaluate_pickplace(grasp_driver, transport_driver, place_driver,
                       eval_set: PickPlaceEvalSet, config: SupervisorConfig | None = None,
                       grasp_env=None, transport_env=None, place_env=None
                       ) -> tuple[dict, list[PickPlaceEpisode]]:
    supervisor = PickPlaceSupervisor(grasp_driver, transport_driver, place_driver, config,
                                     grasp_env, transport_env, place_env)
    start = time.perf_counter()
    results = [episode_from_result(i, supervisor.run(p, t), supervisor.dt)
               for i, (p, t) in enumerate(zip(eval_set.placements, eval_set.targets,
                                              strict=True))]
    summary = summarize_pickplace(results)
    summary["wall_clock_s"] = round(time.perf_counter() - start, 1)
    return summary, results
