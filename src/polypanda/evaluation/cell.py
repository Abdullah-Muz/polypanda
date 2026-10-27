"""Evaluate the Milestone 9 work-order cell: whole work orders, scripted or learned skills.

Each order is generated from its own random stream (seed, index), which also draws the infeed
jitter, so every configuration sees exactly the same orders and cube poses and the results can
be compared order by order (exact McNemar, evaluation.pickplace.mcnemar_exact).

An order succeeds when every tower is complete and correct (ScriptedCell.check: every cube
upright, at its level, on the fixture, in recipe order). Failures are attributed to the first
failing learned skill ("pick:<reason>", "place:<reason>"), the planner having to reject a cube
that was still needed ("planner:buffer_full", Layer 2), the conveyor, or the final tower check
("tower").

Layer 2 (`mixed`): cubes arrive in random order with 0-2 surplus cubes and are routed by the
greedy planner (supervisor.planner) to a tower, a buffer slot or the reject bin. Orders are
generated solvable with `solvable_with` slots; running them with fewer `slots` measures what the
buffer is worth.

v3: `disturbances` names a suite (eval_sets/, simulation.disturbances) with one knock (or none,
D0) per order; each episode records the outcome (D1/D2/D3, or "none" when nothing fell) and
whether it was recoverable, and failures caused by it are labelled "knock:<kind>".
"""

import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from polypanda.evaluation.pickplace import _pct
from polypanda.evaluation.reach import wilson_interval
from polypanda.simulation.disturbances import load_suite
from polypanda.simulation.workcell import Workcell, WorkcellConfig, WorkOrder
from polypanda.supervisor.cell import LearnedCell, LearnedSkill
from polypanda.supervisor.planner import mixed_order


@dataclass(frozen=True)
class CellEvalConfig:
    n: int = 200
    seed: int = 20261026
    heights: tuple[int, int] = (1, 4)
    lateral: float = 0.015          # m, infeed jitter either way
    yaw: float = float(np.radians(15))  # rad, infeed jitter either way
    # Layer 2: random arrival order with surplus cubes, routed by the greedy planner
    mixed: bool = False
    surplus: tuple[int, int] = (0, 2)
    solvable_with: int = 3          # orders are generated solvable with this many buffer slots
    slots: int = 3                  # buffer slots the controller may use (ablation: 0, 1, 3)
    aim: str = "fixture"            # learned place target: the fixture, or as trained (v3: top)
    batch_choice: bool = False      # v3: take any cube on the belt, not only the front one
    disturbances: str | None = None  # v3: a disturbance suite (eval_sets/), one per order

    @property
    def n_cubes(self) -> int:
        """Cube bodies the cell needs (never fewer than Layer 1's original 12-cube model)."""
        return max(12, 3 * self.heights[1] + (self.surplus[1] if self.mixed else 0))


@dataclass(frozen=True)
class CellEpisode:
    index: int
    success: bool
    failure: str | None
    heights: str                 # tower heights, e.g. "3-1-4"
    towers_ok: str               # per tower, e.g. "110"
    cubes: int
    pick_failures: int           # learned skills only
    place_failures: int
    ik_failures: int
    sim_s: float                 # simulated time for the whole order
    buffered: int = 0            # Layer 2: cubes parked in the buffer,
    rejected: int = 0            # sent to the reject bin,
    needed_rejected: int = 0     # of which still needed (buffer full)
    disturbance: str = ""        # v3: "D0" no knock, "none" nothing fell, or D1/D2/D3
    recoverable: bool = False    # every cube the knock threw off could physically be recovered
    fallen: int = 0              # cubes the knock threw off the tower


def skill_from_spec(spec: str, stage: str):
    """'scripted' or 'run:<run_dir>[@best|final]' -> (LearnedSkill | None, description, tag,
    provenance)."""
    if spec == "scripted":
        return None, f"scripted {stage}", "scripted", {}
    if not spec.startswith("run:"):
        raise ValueError(f"{stage} must be 'scripted' or 'run:<run_dir>[@best|final]'")
    run_dir, _, which = spec[4:].partition("@")
    skill = LearnedSkill(run_dir, which or "best")
    meta = skill.policy.metadata
    if not meta["skill"].startswith({"pick": "grasp"}.get(stage, stage)):
        raise ValueError(f"{run_dir} is a {meta['skill']} policy, not a {stage} policy")
    return (skill, f"{meta['algorithm']} {meta['skill']} policy {meta['policy_id']}",
            f"{Path(run_dir).name}_{skill.policy.which}",
            {"run_dir": run_dir, "checkpoint": skill.policy.which, "policy_id": meta["policy_id"],
             "trained_at": meta.get("git", {}).get("commit", "in progress")})


@lru_cache(maxsize=4)
def _suite(path: str) -> list:
    return load_suite(Path(path))


def run_cell_order(ctl: LearnedCell, config: CellEvalConfig, index: int) -> CellEpisode:
    rng = np.random.default_rng([config.seed, index])
    if config.mixed:
        order, _ = mixed_order(rng, config.heights, config.surplus, config.solvable_with)
    else:
        order = WorkOrder.random(rng, heights=config.heights)
    picks, places, steps, log = len(ctl.picks), len(ctl.places), [0], []

    viewer = ctl.on_step  # (a viewer or a diagnostic hook still sees every step)

    def count():
        steps[0] += 1
        if viewer:
            viewer()

    ctl.on_step = count
    try:
        if config.mixed:
            disturbance = _suite(config.disturbances)[index] if config.disturbances else None
            towers = ctl.run_mixed_order(order, rng, lateral=config.lateral, yaw=config.yaw,
                                         slots=config.slots, batch_choice=config.batch_choice,
                                         disturbance=disturbance, log=log.append)
        else:
            towers = ctl.run_order(order, rng, lateral=config.lateral, yaw=config.yaw,
                                   log=log.append)
    finally:
        ctl.on_step = viewer
    stats = ctl.stats if config.mixed else {}
    knocked = ctl.disturbance_outcome if config.mixed and config.disturbances else None
    ok = ctl.check(towers)
    new_picks, new_places = ctl.picks[picks:], ctl.places[places:]
    failure = None
    if not all(ok):
        skill_failures = [(f"pick:{r['failure'] or 'timeout'}") for r in new_picks
                          if not r["success"]] + [(f"place:{r['failure'] or 'timeout'}")
                                                  for r in new_places if not r["success"]]
        if skill_failures:
            failure = skill_failures[0]
        elif stats.get("needed_rejected"):
            failure = "planner:buffer_full"
        elif any("did not arrive" in line for line in log):
            failure = "conveyor"
        elif knocked and knocked["fallen"]:
            failure = f"knock:{knocked['kind']}"
        else:
            failure = "tower"
    if config.mixed and config.disturbances:
        kind = "D0" if knocked is None else knocked["kind"]
        extra = {"disturbance": kind, "recoverable": bool(knocked and knocked["recoverable"]),
                 "fallen": len(knocked["fallen"]) if knocked else 0}
    else:
        extra = {}
    return CellEpisode(
        index=index, success=all(ok), failure=failure,
        heights="-".join(str(len(r)) for r in order.recipes),
        towers_ok="".join(str(int(x)) for x in ok), cubes=len(order.arrivals),
        pick_failures=sum(not r["success"] for r in new_picks),
        place_failures=sum(not r["success"] for r in new_places),
        ik_failures=ctl.ik_failures, sim_s=round(steps[0] * ctl.dt, 2),
        buffered=stats.get("buffered", 0), rejected=stats.get("rejected", 0),
        needed_rejected=stats.get("needed_rejected", 0), **extra)


def summarize_cell(episodes: list[CellEpisode], ctl: LearnedCell) -> dict:
    n = len(episodes)
    wins = sum(e.success for e in episodes)
    towers = [(int(h), ok == "1") for e in episodes
              for h, ok in zip(e.heights.split("-"), e.towers_ok, strict=True)]
    by_height = {}
    for h in sorted({h for h, _ in towers}):
        good = [ok for hh, ok in towers if hh == h]
        by_height[h] = {"towers": len(good), "success_rate": sum(good) / len(good)}

    def skill(rows, key=None):
        if not rows:
            return None
        out = {"attempts": len(rows), "success_rate": sum(r["success"] for r in rows) / len(rows),
               "median_steps": _pct([r["steps"] for r in rows if r["success"]], 50),
               "failures": dict(Counter(r["failure"] or "timeout" for r in rows
                                        if not r["success"]))}
        if key:
            out["by_level"] = {lv + 1: sum(r["success"] for r in rows if r[key] == lv)
                               / sum(r[key] == lv for r in rows)
                               for lv in sorted({r[key] for r in rows})}
        return out

    cubes = sum(e.cubes for e in episodes)
    disturbances = {}
    for kind in ("D0", "none", "D1", "D2", "D3"):
        rows = [e for e in episodes if e.disturbance == kind]
        if rows:
            rec = [e for e in rows if e.recoverable]
            disturbances[kind] = {
                "orders": len(rows), "success_rate": sum(e.success for e in rows) / len(rows),
                "recoverable": len(rec),
                "success_rate_recoverable": (sum(e.success for e in rec) / len(rec)
                                             if rec else None)}
    return {
        "orders": n,
        "success_rate": wins / n,
        "success_ci95": list(wilson_interval(wins, n)),
        "towers": len(towers),
        "tower_success_rate": sum(ok for _, ok in towers) / len(towers),
        "tower_success_by_height": by_height,
        "cubes": cubes,
        "failures": dict(Counter(e.failure for e in episodes if not e.success)),
        "learned_pick": skill(ctl.picks),
        "learned_place": skill(ctl.places, key="level"),
        "ik_failures": sum(e.ik_failures for e in episodes),
        "sim_s_per_cube": sum(e.sim_s for e in episodes) / cubes,
        "sim_s_per_order": {"median": _pct([e.sim_s for e in episodes], 50),
                            "p90": _pct([e.sim_s for e in episodes], 90)},
        "buffer": {"cubes_buffered": sum(e.buffered for e in episodes),
                   "orders_using_buffer": sum(e.buffered > 0 for e in episodes),
                   "rejected": sum(e.rejected for e in episodes),
                   "needed_rejected": sum(e.needed_rejected for e in episodes)},
        **({"disturbances": disturbances} if disturbances else {}),
    }


def evaluate_cell(grasp: LearnedSkill | None, place: LearnedSkill | None,
                  config: CellEvalConfig | None = None, progress=None
                  ) -> tuple[dict, list[CellEpisode]]:
    config = config or CellEvalConfig()
    cell = Workcell.build(WorkcellConfig(n_cubes=config.n_cubes))
    ctl = LearnedCell(cell, grasp=grasp, place=place, aim=config.aim)
    start, episodes = time.time(), []
    for i in range(config.n):
        episodes.append(run_cell_order(ctl, config, i))
        if progress:
            progress(episodes[-1])
    summary = summarize_cell(episodes, ctl)
    summary["wall_clock_s"] = round(time.time() - start, 1)
    return summary, episodes
