"""Validation on a fixed list of episodes (v4.0 door, from dev round 2).

SB3's EvalCallback validates on whatever episodes the eval env draws next, so every
validation meets different machines and checkpoints are not compared on the same cases (the
door's PPO reference swung 52 % -> 33 %, partly that). Here the eval env replays the same
episodes, drawn once with the env's own draw_episode from the validation seed (a stream apart from
the evaluation sets'), each with its own sensor-noise seed, and every validation is also written
per cell (command x slide x spring) to eval/cells.jsonl.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback


class FixedEpisodes(gym.Wrapper):
    """Plays episodes[0], [1], ... in turn at each reset (cycling); rewind() starts over."""

    def __init__(self, env: gym.Env, n: int, seed: int) -> None:
        super().__init__(env)
        rng = np.random.default_rng(seed)
        self.episodes = [env.unwrapped.draw_episode(rng) for _ in range(n)]
        self.noise_seed = seed * 1000
        self.rewind()

    def rewind(self) -> None:
        self.next, self.rows, self.k, self.steps = 0, [], None, 0

    def get_rows(self) -> list[dict]:
        return list(self.rows)

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
            ep, st = self.episodes[self.k], info.get("stats", {})
            self.rows.append({
                "index": self.k, "command": ep["command"], "slide": ep["slide"],
                "spring": ep["spring"] > 0, "success": bool(info.get("is_success", False)),
                "failure": info.get("failure"), "steps": self.steps,
                "peak_force": float(st.get("peak_force", 0.0)),
                "slams": int(st.get("slams", 0)),
            })
        return obs, r, term, trunc, info


def summarize_cells(rows: list[dict]) -> dict:
    cells = defaultdict(lambda: [0, 0])
    for r in rows:
        key = f"{r['command']}/{r['slide']}/{'spring' if r['spring'] else 'plain'}"
        cells[key][0] += r["success"]
        cells[key][1] += 1
    ok = [r["steps"] for r in rows if r["success"]]
    return {
        "n": len(rows), "success": sum(r["success"] for r in rows),
        "cells": {c: f"{v[0]}/{v[1]}" for c, v in sorted(cells.items())},
        "worst_cell_rate": min((v[0] / v[1] for v in cells.values()), default=0.0),
        "steps_to_success_median": float(np.median(ok)) if ok else None,
        "peak_force_median": float(np.median([r["peak_force"] for r in rows])) if rows else None,
        "slam_episodes": sum(r["slams"] > 0 for r in rows),
    }


class FixedEpisodesEvalCallback(EvalCallback):
    """EvalCallback on an eval env wrapped in FixedEpisodes (one env): each validation replays
    the same episodes, and its per-cell summary goes to <log_path>/cells.jsonl."""

    def _on_step(self) -> bool:
        due = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if due:
            self.eval_env.env_method("rewind")
        keep_going = super()._on_step()
        if due:
            rows = self.eval_env.env_method("get_rows")[0][: self.n_eval_episodes]
            s = summarize_cells(rows)
            self.logger.record("eval/worst_cell_rate", s["worst_cell_rate"])
            self.logger.record("eval/slam_episodes", s["slam_episodes"])
            if self.log_path is not None:
                path = Path(self.log_path).parent / "cells.jsonl"
                with open(path, "a") as f:
                    f.write(json.dumps({"timesteps": int(self.num_timesteps), **s}) + "\n")
        return keep_going
