"""Evaluate any reach policy (baseline or SB3 model) on a held-out target set.

Policies only need SB3's `predict(obs, deterministic=True) -> (action, state)`. Episodes run
the full horizon with success termination disabled, so metrics also capture whether a policy
holds the target after first succeeding. Weights are never updated here.
"""

import time
from dataclasses import asdict, dataclass, replace

import numpy as np

from polypanda.environments.reach import ReachConfig, ReachEnv
from polypanda.evaluation.sets import ReachEvalSet
from polypanda.simulation.panda import tcp_position


@dataclass(frozen=True)
class EpisodeResult:
    index: int
    region: str
    start_distance: float
    success: bool
    steps_to_success: int | None    # policy step at which the dwell criterion was first met
    distance_at_success: float | None
    min_distance: float
    final_distance: float           # at the end of the horizon
    held: bool | None               # never left tolerance after first success
    max_distance_after_success: float | None
    path_ratio: float               # TCP path length / start distance, up to success or horizon
    action_jerk: float              # mean ||a_t - a_{t-1}||^2, up to success or horizon
    contact_steps: int
    limit_steps: int
    return_to_success: float        # training-style return: rewards up to success or horizon
    failure: str | None             # None, "timeout" or "non_finite"


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def evaluation_config(base: ReachConfig | None = None) -> ReachConfig:
    base = base or ReachConfig()
    return replace(base,
                   success=replace(base.success, terminate=False),
                   episode=replace(base.episode, start_joint_noise=0.0))


def run_episode(env: ReachEnv, policy, target: np.ndarray, index: int,
                region: str) -> tuple[EpisodeResult, list[float]]:
    obs, info = env.reset(options={"target": target})
    start_distance = info["distance"]
    tcp_prev = tcp_position(env.data, env.idx)
    prev_action = None
    success_step = distance_at_success = None
    after_success: list[float] = []
    distances, latencies = [], []
    path = jerk = ret = 0.0
    jerk_terms = contact_steps = limit_steps = 0
    failure = None

    for step in range(1, env.config.episode.horizon + 1):
        start = time.perf_counter()
        action, _ = policy.predict(obs, deterministic=True)
        latencies.append(time.perf_counter() - start)
        action = np.asarray(action, dtype=np.float32).reshape(7)
        obs, reward, terminated, _, info = env.step(action)
        if info["failure"]:
            failure = info["failure"]
            break
        distance = info["distance"]
        distances.append(distance)
        contact_steps += bool(info["contact"])
        limit_steps += info["joints_at_limit"] > 0
        if success_step is None:
            tcp = tcp_position(env.data, env.idx)
            path += float(np.linalg.norm(tcp - tcp_prev))
            tcp_prev = tcp
            ret += reward
            if prev_action is not None:
                jerk += float(np.sum((action - prev_action) ** 2))
                jerk_terms += 1
            if info["is_success"]:
                success_step, distance_at_success = step, distance
        else:
            after_success.append(distance)
        prev_action = action

    success = success_step is not None
    tolerance = env.config.success.tolerance
    result = EpisodeResult(
        index=index,
        region=region,
        start_distance=start_distance,
        success=success,
        steps_to_success=success_step,
        distance_at_success=distance_at_success,
        min_distance=min(distances) if distances else float("nan"),
        final_distance=distances[-1] if distances else float("nan"),
        held=(max(after_success, default=0.0) < tolerance) if success else None,
        max_distance_after_success=max(after_success, default=distance_at_success)
        if success else None,
        path_ratio=path / start_distance,
        action_jerk=jerk / jerk_terms if jerk_terms else 0.0,
        contact_steps=contact_steps,
        limit_steps=limit_steps,
        return_to_success=ret,
        failure=failure or (None if success else "timeout"),
    )
    return result, latencies


def _pct(values, q, scale: float = 1.0) -> float | None:
    return float(np.percentile(values, q)) * scale if len(values) else None


def summarize(results: list[EpisodeResult], latencies: list[float], control_dt: float) -> dict:
    n = len(results)
    wins = [r for r in results if r.success]
    lo, hi = wilson_interval(len(wins), n)
    steps = [r.steps_to_success for r in wins]
    finals_mm = [r.final_distance * 1e3 for r in results]
    regions = {}
    for region in sorted({r.region for r in results}):
        group = [r for r in results if r.region == region]
        k = sum(r.success for r in group)
        regions[region] = {"n": len(group), "success_rate": k / len(group),
                           "success_ci95": wilson_interval(k, len(group))}
    failures: dict[str, int] = {}
    for r in results:
        if r.failure:
            failures[r.failure] = failures.get(r.failure, 0) + 1
    returns = [r.return_to_success for r in results]
    lat_us = np.array(latencies) * 1e6
    return {
        "episodes": n,
        "success_rate": len(wins) / n,
        "success_ci95": [lo, hi],
        "time_to_success_s": {"median": _pct(steps, 50, control_dt),
                              "p90": _pct(steps, 90, control_dt)},
        "distance_at_success_mm": {"median": _pct([r.distance_at_success * 1e3 for r in wins], 50)},
        "final_distance_mm": {"median": _pct(finals_mm, 50), "p95": _pct(finals_mm, 95),
                              "max": max(finals_mm)},
        "held_rate_given_success": (sum(r.held for r in wins) / len(wins)) if wins else None,
        "path_ratio_median": _pct([r.path_ratio for r in results], 50),
        "action_jerk_median": _pct([r.action_jerk for r in results], 50),
        "contact_episode_rate": sum(r.contact_steps > 0 for r in results) / n,
        "limit_episode_rate": sum(r.limit_steps > 0 for r in results) / n,
        "return_to_success": {"mean": float(np.mean(returns)), "std": float(np.std(returns))},
        "by_region": regions,
        "failures": failures,
        "inference_latency_us": {"median": _pct(lat_us, 50), "p99": _pct(lat_us, 99)},
    }


def evaluate(policy, eval_set: ReachEvalSet,
             config: ReachConfig | None = None) -> tuple[dict, list[EpisodeResult]]:
    config = evaluation_config(config)
    env = ReachEnv(config)
    results, latencies = [], []
    for i, (target, region) in enumerate(zip(eval_set.targets, eval_set.regions, strict=True)):
        result, lat = run_episode(env, policy, target, i, region)
        results.append(result)
        latencies.extend(lat)
    summary = summarize(results, latencies, env.adapter.control_dt)
    summary["eval_config"] = asdict(config)
    return summary, results
