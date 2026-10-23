"""Evaluate transport drivers on held-out (handoff state, goal) pairs through TransportEnv.

Success = TransportEnv success at the final step: cube centre within tolerance of the goal with
a verified grasp, held for the dwell. Results are broken down by the grasp controller that
produced each handoff state, to expose whether learned-grasp handoffs are harder to carry.
"""

import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

from polypanda.control.ik import PoseIK, closing_yaw, top_down_quat
from polypanda.environments.handoff import HandoffPool
from polypanda.environments.transport import TransportConfig, TransportEnv
from polypanda.evaluation.reach import wilson_interval
from polypanda.simulation.panda import tcp_position
from polypanda.simulation.scenes import cube_position


class ScriptedCarryDriver:
    """Classical baseline: PoseIK steering the hand so the cube's *current* position (privileged)
    goes to the goal, gripper levelled to point down, gain 0.2 as for the other IK baselines."""

    def __init__(self, gain: float = 0.2) -> None:
        self.gain = gain
        self.ik = None
        self._quat = None

    def reset(self, env: TransportEnv) -> None:
        if self.ik is None:
            self.ik = PoseIK()
        R = env.data.site_xmat[env.arm.tcp_site].reshape(3, 3)
        self._quat = top_down_quat(closing_yaw(R))

    def __call__(self, env: TransportEnv, obs: np.ndarray) -> np.ndarray:
        offset = cube_position(env.data, env.cube) - tcp_position(env.data, env.arm)
        q = env.data.qpos[env.arm.arm_qpos]
        return self.ik.action(q, env.goal - offset, self._quat, self.gain, env.adapter.max_delta)


@dataclass(frozen=True)
class TransportEpisode:
    index: int
    source: str                    # grasp controller that produced the handoff state
    success: bool
    first_success_s: float | None
    start_distance_mm: float
    final_distance_mm: float
    min_distance_mm: float
    dropped: bool                  # lost grip: finger contact lost or cube tipped over
    drop_s: float | None
    touched_down: bool             # cube lowered onto the table while still held
    touchdown_s: float | None
    max_slip_speed_mm_s: float
    max_cube_tilt_deg: float
    floor_contact_steps: int
    action_jerk: float             # mean ||a_t - a_{t-1}||^2
    episode_return: float
    # lost_grip | lowered_onto_table | never_reached | drifted_away | grasp_unverified
    failure: str | None


def run_transport_episode(env: TransportEnv, driver, state, goal, index: int,
                          source: str) -> tuple[TransportEpisode, list[float]]:
    obs, info = env.reset(options={"state": state, "goal": goal})
    driver.reset(env)
    dt = env.adapter.control_dt
    start_distance = info["distance"]
    latencies, ret, jerk, jerk_n = [], 0.0, 0.0, 0
    first = drop_at = touchdown_at = None
    min_d, max_slip, max_tilt, floor = start_distance, 0.0, 0.0, 0
    prev_a = None
    for _ in range(env.config.episode.horizon):
        t0 = time.perf_counter()
        a = np.asarray(driver(env, obs), dtype=np.float32)
        latencies.append(time.perf_counter() - t0)
        obs, reward, terminated, truncated, info = env.step(a)
        ret += reward
        if prev_a is not None:
            jerk += float(np.sum((a - prev_a) ** 2))
            jerk_n += 1
        prev_a = a
        if info["failure"] == "lost_grip":
            drop_at = env.elapsed_steps * dt
        elif info["failure"] == "lowered_onto_table":
            touchdown_at = env.elapsed_steps * dt
        else:
            min_d = min(min_d, info["distance"])
            max_slip = max(max_slip, info["slip_speed"])
            max_tilt = max(max_tilt, info["cube_tilt"])
            floor += info["floor_contact"]
        if first is None and info["is_success"]:
            first = env.elapsed_steps * dt
        if terminated or truncated:
            break
    tol = env.config.success.tolerance
    success = bool(info["is_success"])
    if success:
        failure = None
    elif drop_at is not None:
        failure = "lost_grip"
    elif touchdown_at is not None:
        failure = "lowered_onto_table"
    elif min_d >= tol:
        failure = "never_reached"
    elif info["distance"] >= tol:
        failure = "drifted_away"
    else:
        failure = "grasp_unverified"
    return TransportEpisode(
        index=index, source=source, success=success, first_success_s=first,
        start_distance_mm=start_distance * 1e3,
        final_distance_mm=info.get("distance", float("nan")) * 1e3, min_distance_mm=min_d * 1e3,
        dropped=drop_at is not None, drop_s=drop_at,
        touched_down=touchdown_at is not None, touchdown_s=touchdown_at,
        max_slip_speed_mm_s=max_slip * 1e3,
        max_cube_tilt_deg=float(np.degrees(max_tilt)), floor_contact_steps=int(floor),
        action_jerk=jerk / jerk_n if jerk_n else 0.0, episode_return=float(ret), failure=failure,
    ), latencies


def _pct(values, q):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.percentile(values, q)) if values else None


def summarize_transport(results: list[TransportEpisode], latencies: list[float]) -> dict:
    n = len(results)
    wins = [r for r in results if r.success]
    by_source = {}
    for source in sorted({r.source for r in results}):
        group = [r for r in results if r.source == source]
        k = sum(r.success for r in group)
        by_source[source] = {"n": len(group), "success_rate": k / len(group),
                             "success_ci95": wilson_interval(k, len(group)),
                             "drop_rate": sum(r.dropped for r in group) / len(group),
                             "touchdown_rate": sum(r.touched_down for r in group) / len(group)}
    lat_us = [v * 1e6 for v in latencies]
    return {
        "episodes": n,
        "success_rate": len(wins) / n,
        "success_ci95": list(wilson_interval(len(wins), n)),
        "drop_rate": sum(r.dropped for r in results) / n,
        "touchdown_rate": sum(r.touched_down for r in results) / n,
        "first_success_s": {"median": _pct([r.first_success_s for r in wins], 50),
                            "p90": _pct([r.first_success_s for r in wins], 90)},
        "final_distance_mm": {"median": _pct([r.final_distance_mm for r in results], 50),
                              "p95": _pct([r.final_distance_mm for r in results], 95)},
        "max_slip_speed_mm_s_median": _pct([r.max_slip_speed_mm_s for r in results], 50),
        "max_cube_tilt_deg": {"median": _pct([r.max_cube_tilt_deg for r in results], 50),
                              "max": max(r.max_cube_tilt_deg for r in results)},
        "floor_contact_episode_rate": sum(r.floor_contact_steps > 0 for r in results) / n,
        "action_jerk_median": _pct([r.action_jerk for r in results], 50),
        "return": {"mean": float(np.mean([r.episode_return for r in results])),
                   "std": float(np.std([r.episode_return for r in results]))},
        "by_source": by_source,
        "failures": dict(Counter(r.failure for r in results if r.failure)),
        "inference_latency_us": {"median": _pct(lat_us, 50), "p99": _pct(lat_us, 99)},
    }


def evaluate_transport(driver, eval_set: HandoffPool, config: TransportConfig | None = None
                       ) -> tuple[dict, list[TransportEpisode]]:
    if eval_set.goals is None:
        raise ValueError("evaluation pool has no goals")
    env = TransportEnv(config)
    start = time.perf_counter()
    results, latencies = [], []
    for i, (state, goal, source) in enumerate(zip(eval_set.states, eval_set.goals,
                                                  eval_set.sources, strict=True)):
        result, lat = run_transport_episode(env, driver, state, goal, i, source)
        results.append(result)
        latencies.extend(lat)
    summary = summarize_transport(results, latencies)
    summary["wall_clock_s"] = round(time.perf_counter() - start, 1)
    return summary, results
