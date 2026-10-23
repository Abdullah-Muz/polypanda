"""Evaluate grasp controllers on held-out cube placements.

Currently the scripted baseline; learned grasp policies will be evaluated on the same set and
metrics. Success means the verified grasp predicate holds at the end of the episode (after the
controller's hold), not merely at some moment.
"""

import time
from collections import Counter
from dataclasses import dataclass

import mujoco
import numpy as np

from polypanda.control.action_adapter import JointDeltaActionAdapter
from polypanda.control.scripted_grasp import (
    GRIPPER_OPEN,
    Phase,
    ScriptedGrasp,
    ScriptedGraspConfig,
)
from polypanda.evaluation.reach import wilson_interval
from polypanda.evaluation.sets import GraspEvalSet
from polypanda.simulation.grasp import GraspMonitor
from polypanda.simulation.panda import PandaIndices, reset_to_home
from polypanda.simulation.scenes import (
    CubeIndices,
    cube_position,
    cube_tilt,
    load_pick_scene,
    set_cube_pose,
)

MAX_STEPS = 600  # policy steps (30 s) hard cap per episode


@dataclass(frozen=True)
class GraspEpisodeResult:
    index: int
    region: str
    x: float
    y: float
    yaw: float
    success: bool                 # grasped at the end of the episode
    first_grasp_s: float | None   # first time the grasp predicate held
    episode_s: float
    approach_s: float
    descend_s: float
    lift_s: float
    nudge_mm: float | None        # cube xy displacement before the fingers closed
    final_lift_cm: float
    max_tilt_deg: float
    final_slip_mm: float | None
    left_force_n: float
    right_force_n: float
    floor_contact_steps: int      # steps where anything other than the cube touched the floor
    failure: str | None           # None, "timeout:<PHASE>" or "not_grasped"


def _floor_contact(model: mujoco.MjModel, data: mujoco.MjData, cube: CubeIndices) -> bool:
    for i in range(data.ncon):
        c = data.contact[i]
        bodies = (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])
        if 0 in bodies and cube.body not in bodies:
            return True
    return False


def run_scripted_episode(model, data, arm, cube, adapter, monitor, controller,
                         placement, index: int, region: str) -> GraspEpisodeResult:
    x, y, yaw = placement
    reset_to_home(model, data, arm)
    set_cube_pose(model, data, cube, (x, y), yaw)
    adapter.reset_to_measured(data)
    data.ctrl[arm.gripper_actuator] = GRIPPER_OPEN
    monitor.reset()
    controller.reset(data)
    start_xy = cube_position(data, cube)[:2]
    dt = adapter.control_dt
    phase_steps: Counter[Phase] = Counter()
    nudge = first_grasp = None
    max_tilt = 0.0
    floor_steps = 0
    state = None

    for step in range(1, MAX_STEPS + 1):
        action, gripper = controller.act(data)
        phase_steps[controller.phase] += 1
        data.ctrl[arm.gripper_actuator] = gripper
        adapter.step(data, action)
        state = monitor.update(data)
        max_tilt = max(max_tilt, cube_tilt(data, cube))
        floor_steps += _floor_contact(model, data, cube)
        if nudge is None and controller.phase >= Phase.CLOSE:
            nudge = float(np.linalg.norm(cube_position(data, cube)[:2] - start_xy))
        if first_grasp is None and state.grasped:
            first_grasp = step * dt
        if controller.phase == Phase.DONE:
            break

    success = bool(state.grasped and controller.timed_out is None)
    if controller.timed_out is not None:
        failure = f"timeout:{controller.timed_out.name}"
    else:
        failure = None if success else "not_grasped"
    return GraspEpisodeResult(
        index=index, region=region, x=float(x), y=float(y), yaw=float(yaw), success=success,
        first_grasp_s=first_grasp, episode_s=step * dt,
        approach_s=phase_steps[Phase.APPROACH] * dt, descend_s=phase_steps[Phase.DESCEND] * dt,
        lift_s=phase_steps[Phase.LIFT] * dt,
        nudge_mm=None if nudge is None else nudge * 1e3,
        final_lift_cm=state.lift * 100, max_tilt_deg=float(np.degrees(max_tilt)),
        final_slip_mm=None if np.isnan(state.slip) else state.slip * 1e3,
        left_force_n=state.left_force, right_force_n=state.right_force,
        floor_contact_steps=floor_steps, failure=failure,
    )


def _pct(values, q):
    values = [v for v in values if v is not None]
    return float(np.percentile(values, q)) if values else None


def summarize_grasp(results: list[GraspEpisodeResult]) -> dict:
    n = len(results)
    wins = [r for r in results if r.success]
    regions = {}
    for region in sorted({r.region for r in results}):
        group = [r for r in results if r.region == region]
        k = sum(r.success for r in group)
        regions[region] = {"n": len(group), "success_rate": k / len(group),
                           "success_ci95": wilson_interval(k, len(group))}
    return {
        "episodes": n,
        "success_rate": len(wins) / n,
        "success_ci95": list(wilson_interval(len(wins), n)),
        "first_grasp_s": {"median": _pct([r.first_grasp_s for r in wins], 50),
                          "p90": _pct([r.first_grasp_s for r in wins], 90)},
        "episode_s_median": _pct([r.episode_s for r in results], 50),
        "phase_s_median": {"approach": _pct([r.approach_s for r in results], 50),
                           "descend": _pct([r.descend_s for r in results], 50),
                           "lift": _pct([r.lift_s for r in results], 50)},
        "nudge_mm": {"median": _pct([r.nudge_mm for r in results], 50),
                     "max": max((r.nudge_mm for r in results if r.nudge_mm is not None),
                                default=None)},
        "max_tilt_deg": {"median": _pct([r.max_tilt_deg for r in results], 50),
                         "p95": _pct([r.max_tilt_deg for r in results], 95),
                         "max": max(r.max_tilt_deg for r in results)},
        "final_lift_cm_median": _pct([r.final_lift_cm for r in wins], 50),
        "finger_force_n_median": _pct([r.left_force_n + r.right_force_n for r in wins], 50) / 2
        if wins else None,
        "floor_contact_episode_rate": sum(r.floor_contact_steps > 0 for r in results) / n,
        "by_region": regions,
        "failures": dict(Counter(r.failure for r in results if r.failure)),
    }


def evaluate_scripted_grasp(eval_set: GraspEvalSet, config: ScriptedGraspConfig | None = None
                            ) -> tuple[dict, list[GraspEpisodeResult]]:
    model, data = load_pick_scene()
    arm, cube = PandaIndices.from_model(model), CubeIndices.from_model(model)
    adapter = JointDeltaActionAdapter(model, arm)
    monitor = GraspMonitor(model, arm, cube)
    controller = ScriptedGrasp(model, arm, cube, adapter.max_delta, config)
    start = time.perf_counter()
    results = [run_scripted_episode(model, data, arm, cube, adapter, monitor, controller,
                                    placement, i, region)
               for i, (placement, region) in enumerate(zip(eval_set.placements,
                                                           eval_set.regions, strict=True))]
    summary = summarize_grasp(results)
    summary["wall_clock_s"] = round(time.perf_counter() - start, 1)
    return summary, results
