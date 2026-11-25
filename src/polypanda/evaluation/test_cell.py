"""v4.1: a shift of the electronics test cell, and its KPIs (docs/v4_1_plan.md, sections 1 and 7).

`run_shift` builds the cell (simulation/test_cell.py), draws the shift's hidden truths, runs the
controller (control/test_cell.py) and returns everything as plain data: the KPIs (`kpis`), one
record per module, the robot's activities, events (failures, operator interventions) and the
testers' states over time (the data a dashboard overlay draws).

KPIs, all over the shift's simulated time (from the first move to the last module handled, or
the end of the run):
- **modules per hour** (parts per hour): modules the robot sorted (PASS tray or FAIL bin) per
  hour; and **seconds per module** (the shift over the modules sorted);
- **tester utilization**: the share of the shift each tester spent testing (RUNNING), and their
  mean;
- **first-pass yield**: modules that went through once, cleanly (read at the first scan, seated
  at the first insertion, no failed step, passed the test and the vision check) into the PASS
  tray, over the modules the robot took off the conveyor; beside it the **test yield** (the
  tester's passes over its results);
- **sort correctness** against the hidden truth (good: PASS tray; bad: FAIL bin; a module the
  robot did not sort counts as wrong), the confusion matrix, **escapes** (bad modules in the PASS
  tray, with their defects) and **false rejects** (good modules in the FAIL bin), the latter by
  reason (no read, not seated, a test fault, the vision check);
- where bad modules were **caught**: at the scan (no read), by the test, by the vision check;
- **failures by stage**: every failed step (a pick, an insertion attempt, a door, a sort), and
  operator interventions;
- the **cycle-time breakdown**: the robot's total and mean time per activity (drive, open, close,
  fetch (scan and pick at the conveyor), insert, unjam, unload, vision, sort, wait) and their share
  of the shift; the insertions' time, force and first-try rate; each module's lead time (taken
  off the conveyor to sorted).
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from polypanda.control.test_cell import TestCellController
from polypanda.simulation.machine import RUNNING
from polypanda.simulation.test_cell import TestCell, cell_config, draw_truths


def utilization(timeline: list, testers: list[str], t0: float, t1: float) -> dict[str, float]:
    """The share of [t0, t1] each tester spent RUNNING, from the timeline's state changes."""
    out = {}
    span = max(t1 - t0, 1e-9)
    for name in testers:
        rows = [(t, s) for t, n, s, _ in timeline if n == name]
        busy, since = 0.0, None
        for t, s in rows:
            if s == RUNNING and since is None:
                since = t
            elif s != RUNNING and since is not None:
                busy += t - since
                since = None
        if since is not None:
            busy += t1 - since
        out[name] = round(busy / span, 4)
    return out


def first_pass(r) -> bool:
    """Through the cell once, cleanly, into the PASS tray."""
    return (r.sorted_to == "pass_tray" and not r.failures and r.scan is not None
            and r.scan.get("tries", 1) == 1 and len(r.inserts) == 1)


def _median(xs) -> float | None:
    return round(float(np.median(xs)), 2) if len(xs) else None


def kpis(ctl: TestCellController) -> dict:
    recs = ctl.records
    t0, t1 = ctl.t_start, ctl.t_end
    span = max(t1 - t0, 1e-9)
    sorted_ = [r for r in recs if r.sorted_to is not None]
    taken = [r for r in recs if r.t_taken is not None]
    util = utilization(ctl.timeline, list(ctl.cell.machines), t0, t1)
    confusion = Counter(f"{'good' if r.good else 'bad'}->{r.sorted_to or 'not_sorted'}"
                        for r in recs)
    caught = Counter()
    for r in recs:
        if not r.good:
            caught[r.reject_reason if r.sorted_to == "fail_bin" and r.reject_reason
                   else "missed"] += 1
    results = [r.test_result for r in recs if r.test_result in ("pass", "fail")]
    times: dict[str, list] = {}
    for a in ctl.activities:
        times.setdefault(a["what"], []).append(a["t1"] - a["t0"])
    failures = Counter(e["stage"] for e in ctl.events if e["stage"] != "operator")
    operator = Counter(e["why"] for e in ctl.events if e["stage"] == "operator")
    inserts = [i for r in recs for i in r.inserts]
    escapes = [r for r in recs if not r.good and r.sorted_to == "pass_tray"]
    false_rejects = [r for r in recs if r.good and r.sorted_to == "fail_bin"]
    return {
        "modules": len(recs), "taken": len(taken), "sorted": len(sorted_),
        "tested": len(results),
        "shift_s": round(span, 1), "timed_out": bool(ctl.timed_out),
        "modules_per_hour": round(len(sorted_) / span * 3600.0, 1),
        "seconds_per_module": round(span / len(sorted_), 1) if sorted_ else None,
        "utilization": util, "utilization_mean": round(float(np.mean(list(util.values()))), 4),
        "first_pass_yield": round(sum(first_pass(r) for r in taken) / len(taken), 4)
        if taken else None,
        "test_yield": round(results.count("pass") / len(results), 4) if results else None,
        "sort_correct": sum(r.correct for r in recs),
        "sort_correct_rate": round(sum(r.correct for r in recs) / max(len(recs), 1), 4),
        "confusion": dict(confusion),
        "escapes": len(escapes),
        "escaped": [{"serial": r.serial, "defects": [d for d in (r.electrical_defect, r.defect)
                                                     if d], "severity": r.severity}
                    for r in escapes],
        "false_rejects": len(false_rejects),
        "false_rejects_by_reason": dict(Counter(r.reject_reason for r in false_rejects)),
        "caught": dict(caught),
        "failures_by_stage": dict(failures), "operator": dict(operator),
        "insertion": {"attempts": len(inserts), "seated": sum(i["seated"] for i in inserts),
                      "first_try_rate": round(float(np.mean(
                          [i["seated"] for i in inserts if i["attempt"] == 0])), 4)
                      if inserts else None,
                      "median_s": _median([i["s"] for i in inserts]),
                      "peak_force_median": _median([i["peak_force"] for i in inserts]),
                      "peak_force_max": max((i["peak_force"] for i in inserts), default=None)},
        "activity_mean_s": {k: round(float(np.mean(v)), 2) for k, v in sorted(times.items())},
        "activity_total_s": {k: round(float(np.sum(v)), 1) for k, v in sorted(times.items())},
        "activity_share": {k: round(float(np.sum(v)) / span, 4)
                           for k, v in sorted(times.items())},
        "lead_time_median_s": _median([r.t_sorted - r.t_taken for r in recs
                                       if r.t_sorted is not None and r.t_taken is not None]),
        "tray_swaps": ctl.tray_swaps, "bin_emptied": ctl.bin_emptied,
        "ik_failures": ctl.ik_failures,
    }


def run_shift(n: int, seed: int = 0, door="feel_v2", handover: str = "door_start",
              nest_belief: str = "camera", cycle_s: float = 45.0, fail_rate: float = 0.05,
              defect_rate: float = 0.03, residue_rate: float = 0.04, truths=None,
              insertion=None, dressing: bool = True, theme: str = "electronics",
              max_time: float | None = None, log=None, **cell) -> dict:
    """One shift of `n` modules (or the given `truths`), seeded by `seed`. `door` and
    `insertion` fill the controller's actor slots (a door actor's name or the actor; an
    insertion actor, default the straight-down placeholder). `cell`: further TestCellConfig
    fields (e.g. `testers`)."""
    cell_cfg = cell_config(cycle_s=cycle_s, dressing=dressing, theme=theme, **cell)
    built = TestCell.build(cell_cfg)
    truths = (truths if truths is not None
              else draw_truths(n, seed, fail_rate, defect_rate, residue_rate))
    ctl = TestCellController(built, door=door, handover=handover, insertion=insertion,
                             nest_belief=nest_belief, seed=seed, log=log)
    try:
        ctl.run_shift(truths, seed=seed, max_time=max_time)
    finally:
        built.close()
    return {
        "config": {"n": len(truths), "seed": seed, "door": ctl.door.name, "handover": handover,
                   "insertion": ctl.insertion.name, "nest_belief": nest_belief,
                   "cycle_s": cycle_s, "fail_rate": fail_rate, "defect_rate": defect_rate,
                   "residue_rate": residue_rate, "theme": theme, "dressing": dressing,
                   "testers": {s.name: {"stop": s.stop, "angle": s.angle, "kind": s.kind,
                                        "planned": s.planned_kind, "stand_in": s.stand_in,
                                        "clearance": s.config.clearance}
                               for s in cell_cfg.testers}},
        "kpis": kpis(ctl),
        "modules": [r.as_dict() for r in ctl.records],
        "events": ctl.events,
        "activities": ctl.activities,
        "timeline": ctl.timeline,
    }
