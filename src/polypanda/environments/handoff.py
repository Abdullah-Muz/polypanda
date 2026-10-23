"""Grasp-to-transport handoff states.

A handoff state is the full MuJoCo integration state at the moment a grasp controller has held a
verified grasp (GraspEnv success: grasp predicate true, cube >= 5 cm up) for `stable_steps`
consecutive policy steps, i.e. when a supervisor would switch from the grasp skill to transport.
Pools mix states produced by different grasp controllers (scripted and learned), so transport
trains and is evaluated on the start distribution it will actually receive in pick-and-place.

States are restored bit-exactly with mj_setState (positions, velocities, controls and solver
warm start). They depend on the model layout and MuJoCo version, both recorded and checked.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from polypanda.simulation.assets import MENAGERIE_COMMIT

STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def capture_state(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
    mujoco.mj_getState(model, data, state, STATE_SPEC)
    return state


def restore_state(model: mujoco.MjModel, data: mujoco.MjData, state: np.ndarray) -> None:
    mujoco.mj_setState(model, data, state, STATE_SPEC)
    mujoco.mj_forward(model, data)


def capture_handoff(env, driver, placement, stable_steps: int = 10) -> np.ndarray | None:
    """Run a grasp driver in GraspEnv from `placement`; return the handoff state or None."""
    obs, _ = env.reset(options={"cube": tuple(placement)})
    driver.reset(env)
    streak = 0
    for _ in range(env.config.episode.horizon):
        obs, _, terminated, truncated, info = env.step(driver(env, obs))
        streak = streak + 1 if info["is_success"] else 0
        if streak >= stable_steps:
            return capture_state(env.model, env.data)
        if terminated or truncated:
            return None
    return None


@dataclass(frozen=True)
class HandoffPool:
    states: np.ndarray            # (n, state_size)
    sources: tuple[str, ...]      # grasp controller that produced each state
    placements: np.ndarray        # (n, 3) cube x, y, yaw the grasp started from
    goals: np.ndarray | None = None  # (n, 3) transport goals, for evaluation sets
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.states)

    @property
    def digest(self) -> str:
        h = hashlib.sha256(np.ascontiguousarray(self.states).tobytes())
        if self.goals is not None:
            h.update(np.ascontiguousarray(self.goals).tobytes())
        return h.hexdigest()[:12]

    def subset(self, n: int) -> "HandoffPool":
        goals = None if self.goals is None else self.goals[:n]
        return HandoffPool(self.states[:n], self.sources[:n], self.placements[:n], goals,
                           self.meta)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {**self.meta, "digest": self.digest, "n": len(self)}
        arrays = {"states": self.states, "sources": np.array(self.sources),
                  "placements": self.placements, "meta": np.array(json.dumps(meta))}
        if self.goals is not None:
            arrays["goals"] = self.goals
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path, model: mujoco.MjModel | None = None) -> "HandoffPool":
        with np.load(path) as f:
            meta = json.loads(str(f["meta"]))
            pool = cls(states=f["states"], sources=tuple(str(s) for s in f["sources"]),
                       placements=f["placements"],
                       goals=f["goals"] if "goals" in f else None, meta=meta)
        if pool.digest != meta["digest"]:
            raise ValueError(f"{path}: digest mismatch (file edited or corrupted)")
        if meta.get("menagerie_commit") != MENAGERIE_COMMIT:
            raise ValueError(f"{path}: made with Menagerie {meta.get('menagerie_commit')}")
        if model is not None and pool.states.shape[1] != mujoco.mj_stateSize(model, STATE_SPEC):
            raise ValueError(f"{path}: state size {pool.states.shape[1]} does not match model")
        return pool
