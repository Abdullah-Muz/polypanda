"""Reach targets that are reachable by construction.

Joint configurations are sampled uniformly inside the joint limits (minus a margin) and mapped
through forward kinematics to a TCP position. Samples whose configuration is in contact (self or
floor) or whose TCP falls outside the workspace box are rejected. Every accepted target carries
the joint configuration that reaches it, so an unreachable target cannot be mistaken for a
policy failure.

Uniform joint sampling is not uniform in Cartesian space: some regions of the box are about 10x
denser than others. Evaluate success per region rather than relying on a single average.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from polypanda.simulation.panda import ARM_JOINTS, PandaIndices


@dataclass(frozen=True)
class WorkspaceConfig:
    low: tuple[float, float, float] = (0.25, -0.45, 0.10)   # m, world frame
    high: tuple[float, float, float] = (0.75, 0.45, 0.65)
    joint_margin: float = 0.1  # rad kept clear of every joint limit
    max_tries: int = 10_000


@dataclass(frozen=True)
class ReachTarget:
    position: np.ndarray   # TCP target in the world frame (m)
    witness_q: np.ndarray  # an arm configuration whose TCP is exactly at `position`


class ReachTargetSampler:
    def __init__(self, model: mujoco.MjModel, idx: PandaIndices,
                 config: WorkspaceConfig | None = None) -> None:
        config = config or WorkspaceConfig()
        self.model = model
        self.idx = idx
        self.config = config
        self.low = np.asarray(config.low, dtype=float)
        self.high = np.asarray(config.high, dtype=float)
        if np.any(self.low >= self.high):
            raise ValueError("workspace low must be below high on every axis")
        joint_ids = [model.joint(name).id for name in ARM_JOINTS]
        lo, hi = model.jnt_range[joint_ids].T
        self.q_low, self.q_high = lo + config.joint_margin, hi - config.joint_margin
        if np.any(self.q_low >= self.q_high):
            raise ValueError("joint_margin leaves an empty joint range")
        # Scratch state so sampling never disturbs a running simulation; fingers stay at home.
        self._data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, self._data, idx.home_key)
        self.tries = 0
        self.accepted = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.tries if self.tries else float("nan")

    def sample(self, rng: np.random.Generator, avoid: np.ndarray | None = None,
               min_distance: float = 0.0) -> ReachTarget:
        """Draw one target; optionally at least `min_distance` from the point `avoid`."""
        d = self._data
        for _ in range(self.config.max_tries):
            self.tries += 1
            q = rng.uniform(self.q_low, self.q_high)
            d.qpos[self.idx.arm_qpos] = q
            mujoco.mj_fwdPosition(self.model, d)  # kinematics + collision detection
            if d.ncon:
                continue
            p = d.site_xpos[self.idx.tcp_site].copy()
            if np.any(p < self.low) or np.any(p > self.high):
                continue
            if avoid is not None and np.linalg.norm(p - avoid) < min_distance:
                continue
            self.accepted += 1
            return ReachTarget(position=p, witness_q=q)
        raise RuntimeError(f"no valid target after {self.config.max_tries} tries; "
                           "check the workspace box and joint margin")
