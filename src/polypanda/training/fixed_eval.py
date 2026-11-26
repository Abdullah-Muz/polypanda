"""Validation on a fixed list of episodes (v4.0 door, from dev round 2; v4.1 insertion).

SB3's EvalCallback validates on whatever episodes the eval env draws next, so every
validation meets different machines and checkpoints are not compared on the same cases (the
door's PPO reference swung 52 % -> 33 %, partly that). Here the eval env replays the same
episodes, drawn once with the env's own draw_episode from the validation seed (a stream apart from
the evaluation sets'), each with its own sensor-noise seed, and every validation is also written
per cell to eval/cells.jsonl.

What a row holds and what the cells are is the task's (`CELLS`, chosen by the env's class):
- **door** (DoorEnv, DoorV2Env): cells command x slide x spring, as v4.0 logged them (unchanged,
  line for line, for v4.0's lid-only tasks); with tester kinds drawn (v4.1: episodes with "kind")
  the rows also carry the kind, and the summary adds the targets' kind x command cells (D1).
- **insertion** (InsertionEnv): cells per tester (A lid, B drawer, C pneumatic), the episodes
  over the 50 N damage limit, and success by clearance and by camera error (as
  evaluation/insertion.summarize bins them).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback


# --- the door -------------------------------------------------------------------------------------
def door_row(ep: dict, info: dict) -> tuple[dict, dict]:
    st = info.get("stats", {})
    row = {"command": ep["command"], "slide": ep["slide"], "spring": ep["spring"] > 0}
    out = {"success": bool(info.get("is_success", False)), "failure": info.get("failure"),
           "peak_force": float(st.get("peak_force", 0.0)), "slams": int(st.get("slams", 0))}
    if "kind" in ep:                  # (v4.1: lids and drawers; v4.0's rows stay as they were)
        row["kind"] = ep["kind"]
    return row, out


def _rate_cells(rows: list[dict], key: Callable[[dict], str]) -> dict:
    cells = defaultdict(lambda: [0, 0])
    for r in rows:
        k = key(r)
        cells[k][0] += r["success"]
        cells[k][1] += 1
    return cells


def summarize_cells(rows: list[dict]) -> dict:
    """The door's validation summary (v4.0's, unchanged for lid-only rows)."""
    cells = _rate_cells(rows, lambda r: f"{r['command']}/{r['slide']}/"
                                        f"{'spring' if r['spring'] else 'plain'}")
    ok = [r["steps"] for r in rows if r["success"]]
    out = {
        "n": len(rows), "success": sum(r["success"] for r in rows),
        "cells": {c: f"{v[0]}/{v[1]}" for c, v in sorted(cells.items())},
        "worst_cell_rate": min((v[0] / v[1] for v in cells.values()), default=0.0),
        "steps_to_success_median": float(np.median(ok)) if ok else None,
        "peak_force_median": float(np.median([r["peak_force"] for r in rows])) if rows else None,
        "slam_episodes": sum(r["slams"] > 0 for r in rows),
    }
    if any("kind" in r for r in rows):
        # v4.1 (docs/v4_1_targets.md, D1): kind x command, the targets' cells
        kc = _rate_cells(rows, lambda r: f"{r.get('kind', 'lid')}/{r['command']}")
        out["kind_command"] = {c: f"{v[0]}/{v[1]}" for c, v in sorted(kc.items())}
        out["worst_kind_command_rate"] = min(v[0] / v[1] for v in kc.values())
    return out


# --- the insertion --------------------------------------------------------------------------------
CLEARANCE_BINS_MM = (0.25, 0.3125, 0.375, 0.4375, 0.5001)
CAMERA_BINS_MM = (0.0, 0.5, 1.0, 1.5, 2.0, 3.01)


def _bin(x: float, edges: tuple[float, ...]) -> str:
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if lo <= x < hi:
            return f"{lo:g}-{hi:g}"
    return f">={edges[-1]:g}"


def insertion_row(ep: dict, info: dict) -> tuple[dict, dict]:
    st = info.get("stats", {})
    out = info.get("outcome") or {}
    row = {"tester": ep["tester"], "kind": ep["kind"],
           "clearance_mm": round(ep["clearance"] * 1000, 4),
           "camera_error_mm": round(float(np.hypot(*ep["camera_xy"])) * 1000, 4)}
    res = {"success": bool(info.get("is_success", False)), "failure": info.get("failure"),
           "time_s": out.get("time_s"), "peak_force": float(st.get("peak_force", 0.0))}
    return row, res


def summarize_insertion(rows: list[dict], damage_limit: float = 50.0) -> dict:
    cells = _rate_cells(rows, lambda r: f"{r['tester']} ({r['kind']})")
    ok = [r["steps"] for r in rows if r["success"]]
    failures = defaultdict(int)
    for r in rows:
        if not r["success"]:
            failures[str(r["failure"])] += 1

    def rates(key):
        return {c: f"{v[0]}/{v[1]}" for c, v in sorted(_rate_cells(rows, key).items())}
    return {
        "n": len(rows), "success": sum(r["success"] for r in rows),
        "cells": {c: f"{v[0]}/{v[1]}" for c, v in sorted(cells.items())},
        "worst_cell_rate": min((v[0] / v[1] for v in cells.values()), default=0.0),
        "steps_to_success_median": float(np.median(ok)) if ok else None,
        "peak_force_median": float(np.median([r["peak_force"] for r in rows])) if rows else None,
        "damage_episodes": sum(r["peak_force"] > damage_limit for r in rows),
        "failures": dict(sorted(failures.items())),
        "by_clearance_mm": rates(lambda r: _bin(r["clearance_mm"], CLEARANCE_BINS_MM)),
        "by_camera_error_mm": rates(lambda r: _bin(r["camera_error_mm"], CAMERA_BINS_MM)),
    }


# --- per task -------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskCells:
    """A task's validation rows and cells: row(episode, last info) -> (the episode's keys, the
    outcome's); summarize(rows) -> the summary written to cells.jsonl; scalars: the summary's
    numbers also logged to TensorBoard (eval/<name>)."""
    row: Callable[[dict, dict], tuple[dict, dict]]
    summarize: Callable[[list[dict]], dict]
    scalars: tuple[str, ...]


DOOR = TaskCells(door_row, summarize_cells, ("worst_cell_rate", "slam_episodes"))
INSERTION = TaskCells(insertion_row, summarize_insertion, ("worst_cell_rate", "damage_episodes"))


def cells_for(env: gym.Env) -> TaskCells:
    """The validation cells of the env's task (by its class)."""
    from polypanda.environments.door import DoorEnv
    from polypanda.environments.insertion import InsertionEnv
    if isinstance(env, DoorEnv):
        return DOOR
    if isinstance(env, InsertionEnv):
        return INSERTION
    raise TypeError(f"no fixed-episode validation cells for {type(env).__name__}")


class FixedEpisodes(gym.Wrapper):
    """Plays episodes[0], [1], ... in turn at each reset (cycling); rewind() starts over."""

    def __init__(self, env: gym.Env, n: int, seed: int) -> None:
        super().__init__(env)
        rng = np.random.default_rng(seed)
        self.episodes = [env.unwrapped.draw_episode(rng) for _ in range(n)]
        self.noise_seed = seed * 1000
        self.task_cells = cells_for(env.unwrapped)
        self.rewind()

    def rewind(self) -> None:
        self.next, self.rows, self.k, self.steps = 0, [], None, 0

    def get_rows(self) -> list[dict]:
        return list(self.rows)

    def summary(self, n: int | None = None) -> dict:
        """The task's summary of the first n rows (all by default)."""
        return self.task_cells.summarize(self.rows[:n] if n is not None else list(self.rows))

    def reset(self, *, seed=None, options=None):
        self.k = self.next % len(self.episodes)
        self.next += 1
        self.steps = 0
        return self.env.reset(seed=self.noise_seed + self.k,
                              options={"episode": self.episodes[self.k]})

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self.steps += 1
        if term or trunc:
            keys, out = self.task_cells.row(self.episodes[self.k], info)
            # (v4.0's door rows: index, command, slide, spring, success, failure, steps,
            # peak_force, slams, in that order)
            self.rows.append({"index": self.k, **keys, "success": out.pop("success"),
                              "failure": out.pop("failure"), "steps": self.steps, **out})
        return obs, r, term, trunc, info


class FixedEpisodesEvalCallback(EvalCallback):
    """EvalCallback on an eval env wrapped in FixedEpisodes (one env): each validation replays
    the same episodes, and its per-cell summary goes to <log_path>/cells.jsonl."""

    def _on_step(self) -> bool:
        due = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if due:
            self.eval_env.env_method("rewind")
        keep_going = super()._on_step()
        if due:
            s = self.eval_env.env_method("summary", self.n_eval_episodes)[0]
            scalars = self.eval_env.get_attr("task_cells")[0].scalars
            for name in scalars:
                self.logger.record(f"eval/{name}", s[name])
            if self.log_path is not None:
                path = Path(self.log_path).parent / "cells.jsonl"
                with open(path, "a") as f:
                    f.write(json.dumps({"timesteps": int(self.num_timesteps), **s}) + "\n")
        return keep_going
