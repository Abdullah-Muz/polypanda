"""v4.1: the insertion's scripted baselines, policies and demonstrations on InsertionEnv's episodes
(docs/v4_1_targets.md, Part I; docs/insertion_env.md).

- **The scripts on the same episodes** (`run_script`): from the env's start (the module held, the
  insertion aimed as the cell's slot aims it), the frozen control/scripted_insertion.py scripts
  run on the env's own controller, through the cell's own control path, unchanged:
  * "search": `SearchInsertion()` (its frozen defaults, as the cell runs it:
    control/test_cell.make_insertion), called as NestInsertion calls it: on a drawer, whose nest
    stands on its tray, through the cell's frame adapter (control/test_cell._NestFrame, the force
    read from the real data);
  * "straight_down": `straight_down` (the cell's closed-loop place, ScriptedCell.place, straight
    down onto the believed nest, release, rise). In the insertion slot the wrist is already turned
    so the key meets the slot; the place keeps that turn (`_Aimed`: its carry yaw is the aimed
    yaw), as the cell's straight-down (ScriptedTending.load) keeps it, and on a drawer it runs
    through the same frame adapter.
  The env watches every physics step (the nest force) and judges the end (InsertionEnv.outcome).
- **A policy** (`run_policy`): any (obs) -> action, stepped through the env.
- **Demonstrations** (`record_script`): a script's run recorded as the env's (observation,
  action) pairs: each control step's hand command (the position and yaw the script solved for)
  and release become the action that moves the env's hand target there (InsertionEnv
  .action_from_command). `replay` steps such actions through the env from the same episode;
  docs/insertion_env.md reports how often the replay reproduces the script's outcome.
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np

from polypanda.control.scripted_cell import ScriptedCell
from polypanda.environments.grasp import GRIPPER_OPEN

SCRIPTS = ("straight_down", "search")
EVAL_SPAWN_KEY = 0x1A5E          # the insertion sets' streams (scripts/make_insertion_sets.py)
PICK_SPAWN_KEY = 0x1A5F          # the training pick pool's stream
DEMO_SPAWN_KEY = 0x1A60          # demonstrations' episode stream
NOISE_SEED = 8_000_000           # the sensors' noise, seeded per episode index


class _Aimed:
    """The controller (or its nest-floor frame) as `straight_down` sees it in the insertion slot:
    unchanged, except that its carry yaw is the aimed one, so the cell's place
    (ScriptedCell.place, run unchanged on this view) keeps the wrist turned to the slot."""

    def __init__(self, inner, yaw: float) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_yaw", float(yaw))

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value) -> None:
        setattr(self._inner, name, value)

    def carry_yaw(self, angle: float) -> float:
        return self._yaw

    def place(self, cube, xy, level: int) -> None:
        ScriptedCell.place(self, cube, xy, level)


def run_script(env, script: str) -> dict:
    """Run a frozen script from the env's current start (after reset); returns the outcome
    (InsertionEnv.outcome) with the script's own record under "script"."""
    from polypanda.control.scripted_insertion import SearchInsertion, straight_down
    from polypanda.control.test_cell import _NestFrame, _wrench_on_real_data
    if script not in SCRIPTS:
        raise ValueError(f"unknown script {script!r}; one of {SCRIPTS}")
    ctl, part = env.ctl, env.part
    xy, yaw = np.asarray(env.aim_xy, float), float(env.aim_yaw)
    floor = env.belief.config.nest_floor
    frame = ctl if floor == 0.0 else _NestFrame(ctl, floor)
    with (_wrench_on_real_data() if floor != 0.0 else nullcontext()):
        if script == "search":
            info = SearchInsertion().insert(frame, part, xy, yaw)
        else:
            info = straight_down(_Aimed(frame, yaw), part, xy)
    info = dict(info or {})
    if env.line.attached(part):            # the slot's contract (control/test_cell.NestInsertion)
        info["released_by_slot"] = True
        env.line.detach(part)
        ctl.grip = GRIPPER_OPEN
        ctl.step(ctl.config.grip_steps)
        tcp = ctl.d.site_xpos[ctl.arm.tcp_site][:2].copy()
        ctl.vertical(tcp, ctl.config.carry_height, yaw)
    out = env.outcome()
    out["script"] = {k: v for k, v in info.items() if k != "time_s"}
    return out


def run_policy(env, act) -> dict:
    """Step `act` (obs -> action) from the env's current start to the end of the episode."""
    obs = env.observation()
    info: dict = {}
    total = 0.0
    while True:
        obs, r, term, trunc, info = env.step(act(obs))
        total += r
        if term or trunc:
            break
    out = info.get("outcome") or env.outcome()
    out["return"] = round(total, 3)
    return out


class ScriptRecorder:
    """Records a script's run in the env's spaces (InsertionEnv.recorder): before each control
    step the action reaching the script's command, after it the observation."""

    def __init__(self, env) -> None:
        self.obs = [env.observation()]
        self.actions: list[np.ndarray] = []
        self.exceeded = 0          # steps whose command lies beyond one action

    def before_step(self, env) -> None:
        a = env.action_from_command()
        if np.any(np.abs(a[:4]) > 1.0 + 1e-9):
            self.exceeded += 1
        self.actions.append(a)
        env.adopt_command()

    def after_step(self, env) -> None:
        self.obs.append(env.observation())


def record_script(env, script: str) -> dict:
    """Run `script` from the env's start, recording it: {"obs": (T+1, d), "actions": (T, 5),
    "outcome": ..., "exceeded": steps beyond one action}."""
    rec = ScriptRecorder(env)
    env.recorder = rec
    try:
        out = run_script(env, script)
    finally:
        env.recorder = None
    return {"obs": np.array(rec.obs, dtype=np.float32), "actions": np.array(rec.actions),
            "outcome": out, "exceeded": rec.exceeded}


def replay(env, episode: dict, actions: np.ndarray, seed: int | None = None) -> dict:
    """Step recorded actions through the env from `episode`'s start: {"obs", "actions",
    "rewards", "outcome"} (the steps actually taken: the env may end the episode first)."""
    obs, _ = env.reset(seed=seed, options={"episode": episode})
    obs_l, act_l, rew_l = [obs], [], []
    info: dict = {}
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        obs_l.append(obs)
        act_l.append(a)
        rew_l.append(r)
        if term or trunc:
            break
    out = info.get("outcome") or env.outcome()
    return {"obs": np.array(obs_l, dtype=np.float32), "actions": np.array(act_l),
            "rewards": np.array(rew_l), "outcome": out}


# --- summaries --------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def _stats(values) -> dict | None:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(v):
        return None
    return {"median": round(float(np.median(v)), 3), "p90": round(float(np.percentile(v, 90)), 3),
            "max": round(float(v.max()), 3)}


def _group(rows: list[dict], key) -> dict:
    groups: dict = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    return {g: {"n": len(rs), "success": sum(r["success"] for r in rs),
                "rate": round(sum(r["success"] for r in rs) / len(rs), 4),
                "time_s_median": (_stats(r["time_s"] for r in rs if r["success"]) or {}).get(
                    "median"),
                "peak_force_median": (_stats(r["peak_force"] for r in rs) or {}).get("median")}
            for g, rs in sorted(groups.items())}


def _bin(x: float, edges: tuple[float, ...]) -> str:
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if lo <= x < hi:
            return f"{lo:g}-{hi:g}"
    return f">={edges[-1]:g}"


def summarize(rows: list[dict], budget_s: float = 30.0) -> dict:
    n, k = len(rows), sum(r["success"] for r in rows)
    ok = [r for r in rows if r["success"]]
    within = sum(r["success"] and r["time_s"] <= budget_s for r in rows)
    failures: dict = {}
    for r in rows:
        if not r["success"]:
            failures[r["failure"]] = failures.get(r["failure"], 0) + 1
    return {
        "n": n, "success": k, "rate": k / n if n else 0.0, "wilson95": wilson(k, n),
        "success_within_budget": within, "rate_within_budget": within / n if n else 0.0,
        "wilson95_within_budget": wilson(within, n),
        "time_to_success_s": _stats(r["time_s"] for r in ok),
        "peak_force_n": _stats(r["peak_force"] for r in rows),
        "peak_force_ctl_n": _stats(r["peak_force_ctl"] for r in rows),
        "over_damage_limit": sum(r["peak_force"] > 50.0 for r in rows),
        "over_damage_limit_ctl": sum(r["peak_force_ctl"] > 50.0 for r in rows),
        "failures": dict(sorted(failures.items())),
        "by_tester": _group(rows, lambda r: f"{r['tester']} ({r['kind']})"),
        "by_camera_error_mm": _group(rows, lambda r: _bin(r["camera_error_mm"],
                                                          (0.0, 0.5, 1.0, 1.5, 2.0, 3.01))),
        "by_camera_yaw_deg": _group(rows, lambda r: _bin(abs(r["camera_yaw_deg"]),
                                                         (0.0, 0.25, 0.5, 1.0, 2.01))),
        "by_clearance_mm": _group(rows, lambda r: _bin(r["clearance_mm"],
                                                       (0.25, 0.3125, 0.375, 0.4375, 0.5001))),
        "by_chamfer_mm": _group(rows, lambda r: _bin(r["chamfer_mm"],
                                                     (0.5, 0.625, 0.75, 0.875, 1.0001))),
    }


def episode_row(index: int, ep: dict, out: dict) -> dict:
    """One result row: the episode's factors and the outcome."""
    row = {"index": index, "tester": ep["tester"], "kind": ep["kind"],
           "clearance_mm": round(ep["clearance"] * 1000, 4),
           "chamfer_mm": round(ep["chamfer"] * 1000, 4),
           "camera_error_mm": round(float(np.hypot(*ep["camera_xy"])) * 1000, 4),
           "camera_yaw_deg": round(float(np.degrees(ep["camera_yaw"])), 4)}
    keep = ("success", "failure", "time_s", "steps", "seated", "released", "hand_up",
            "peak_force", "peak_force_ctl", "peak_force_total", "collision_steps", "seat",
            "script", "return", "lead_clips", "turn_clips")
    row |= {k: out[k] for k in keep if k in out}
    return row
