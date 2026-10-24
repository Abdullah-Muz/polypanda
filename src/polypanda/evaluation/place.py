"""Evaluate place drivers on held-out transport-to-place handoffs through PlaceEnv.

Success = verified placement at the final step (resting, upright, within 2 cm of the target,
still, clear of the gripper, for 0.5 s). Failures are classified by what went wrong.
"""

import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

from polypanda.environments.handoff import HandoffPool
from polypanda.environments.place import PlaceConfig, PlaceEnv
from polypanda.evaluation.reach import wilson_interval
from polypanda.simulation.scenes import cube_position


@dataclass(frozen=True)
class PlaceEpisode:
    index: int
    source: str
    success: bool
    first_placed_s: float | None
    released: bool
    release_s: float | None
    final_xy_error_mm: float
    final_tilt_deg: float
    final_rest_mm: float
    gripper_contact_at_end: bool
    floor_contact_steps: int
    episode_return: float
    # dropped | toppled | never_released | off_target | tilted | gripper_contact | not_settled
    failure: str | None


def run_place_episode(env: PlaceEnv, driver, state, goal, index: int,
                      source: str) -> PlaceEpisode:
    obs, _ = env.reset(options={"state": state, "target": goal})
    driver.reset(env)
    dt = env.adapter.control_dt
    first = release_at = None
    ret, floor = 0.0, 0
    info = {}
    for _ in range(env.config.horizon):
        obs, reward, terminated, truncated, info = env.step(driver(env, obs))
        ret += reward
        floor += info["floor_contact"]
        if release_at is None and info["released"]:
            release_at = env.elapsed_steps * dt
        if first is None and info["is_success"]:
            first = env.elapsed_steps * dt
        if terminated or truncated:
            break
    cfg = env.config.check
    rest = cube_position(env.data, env.cube)[2] - env.cube.half_size[2]
    success = bool(info["is_success"])
    if success:
        failure = None
    elif info["failure"]:
        failure = info["failure"]
    elif not info["released"]:
        failure = "never_released"
    elif info["xy_error"] >= cfg.tolerance:
        failure = "off_target"
    elif info["cube_tilt"] >= cfg.max_tilt:
        failure = "tilted"
    elif info["gripper_contact"]:
        failure = "gripper_contact"
    else:
        failure = "not_settled"
    return PlaceEpisode(
        index=index, source=source, success=success, first_placed_s=first,
        released=bool(info["released"]), release_s=release_at,
        final_xy_error_mm=info["xy_error"] * 1e3,
        final_tilt_deg=float(np.degrees(info["cube_tilt"])), final_rest_mm=rest * 1e3,
        gripper_contact_at_end=bool(info["gripper_contact"]), floor_contact_steps=int(floor),
        episode_return=float(ret), failure=failure)


def _pct(values, q):
    values = [v for v in values if v is not None]
    return float(np.percentile(values, q)) if values else None


def summarize_place(results: list[PlaceEpisode]) -> dict:
    n = len(results)
    wins = [r for r in results if r.success]
    by_source = {}
    for source in sorted({r.source for r in results}):
        group = [r for r in results if r.source == source]
        k = sum(r.success for r in group)
        by_source[source] = {"n": len(group), "success_rate": k / len(group),
                             "success_ci95": wilson_interval(k, len(group))}
    return {
        "episodes": n,
        "success_rate": len(wins) / n,
        "success_ci95": list(wilson_interval(len(wins), n)),
        "first_placed_s": {"median": _pct([r.first_placed_s for r in wins], 50),
                           "p90": _pct([r.first_placed_s for r in wins], 90)},
        "release_s_median": _pct([r.release_s for r in results], 50),
        "final_xy_error_mm": {"median": _pct([r.final_xy_error_mm for r in results], 50),
                              "p95": _pct([r.final_xy_error_mm for r in results], 95)},
        "final_tilt_deg": {"median": _pct([r.final_tilt_deg for r in results], 50),
                           "max": max(r.final_tilt_deg for r in results)},
        "floor_contact_episode_rate": sum(r.floor_contact_steps > 0 for r in results) / n,
        "return": {"mean": float(np.mean([r.episode_return for r in results]))},
        "by_source": by_source,
        "failures": dict(Counter(r.failure for r in results if r.failure)),
    }


def evaluate_place(driver, eval_set: HandoffPool, config: PlaceConfig | None = None
                   ) -> tuple[dict, list[PlaceEpisode]]:
    env = PlaceEnv(config)
    start = time.perf_counter()
    results = [run_place_episode(env, driver, s, g, i, src) for i, (s, g, src) in
               enumerate(zip(eval_set.states, eval_set.goals, eval_set.sources, strict=True))]
    summary = summarize_place(results)
    summary["wall_clock_s"] = round(time.perf_counter() - start, 1)
    return summary, results
