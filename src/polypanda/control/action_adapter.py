"""Joint-increment action interface shared by every arm policy and the classical baseline.

A normalized action a in [-1, 1]^7 nudges seven persistent joint targets:

    q_cmd <- clip(q_cmd + max_delta * a, ctrlrange)

The targets are written to the Menagerie position servos, which then run for
n_substeps physics steps. The gripper actuator is never written here.

Because targets are integrated, they can drift away from the measured joints (gravity sag,
contact, joint limits). Policies therefore observe q_cmd as well as q, targets are reset at
episode boundaries, and an optional max_lead bound stops the target winding up while the arm
is blocked.
"""

import mujoco
import numpy as np

from polypanda.simulation.panda import PandaIndices

ACTION_DIM = 7
ACTION_SCHEMA = "joint-increment-v1-7d"


class JointDeltaActionAdapter:
    def __init__(
        self,
        model: mujoco.MjModel,
        idx: PandaIndices,
        max_delta: float = 0.015,
        n_substeps: int = 25,
        max_lead: float | None = None,
    ) -> None:
        if max_delta <= 0 or n_substeps < 1:
            raise ValueError("max_delta must be > 0 and n_substeps >= 1")
        if max_lead is not None and max_lead <= 0:
            raise ValueError("max_lead must be > 0 or None")
        self.model = model
        self.idx = idx
        self.max_delta = max_delta
        self.n_substeps = n_substeps
        self.max_lead = max_lead
        self.low, self.high = model.actuator_ctrlrange[idx.arm_actuators].T.copy()
        self.q_cmd = np.zeros(ACTION_DIM)

    @property
    def control_dt(self) -> float:
        """Simulated seconds per policy step."""
        return self.n_substeps * self.model.opt.timestep

    @property
    def at_limit(self) -> np.ndarray:
        """Per-joint mask of targets sitting on a ctrlrange bound."""
        return (self.q_cmd <= self.low) | (self.q_cmd >= self.high)

    def reset_to_measured(self, data: mujoco.MjData) -> None:
        """Episode start: targets become the measured joint positions."""
        self.q_cmd = np.clip(data.qpos[self.idx.arm_qpos], self.low, self.high)
        data.ctrl[self.idx.arm_actuators] = self.q_cmd

    def adopt_ctrl(self, data: mujoco.MjData) -> None:
        """Policy hand-off: keep whatever the servos are already commanded, so nothing jumps."""
        self.q_cmd = np.clip(data.ctrl[self.idx.arm_actuators], self.low, self.high)

    def apply(self, data: mujoco.MjData, action: np.ndarray) -> np.ndarray:
        """Integrate one action into the targets and write them to data.ctrl. Does not step."""
        a = np.asarray(action, dtype=np.float64)
        if a.shape != (ACTION_DIM,):
            raise ValueError(f"action must have shape ({ACTION_DIM},), got {a.shape}")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"action contains non-finite values: {a}")
        q_cmd = self.q_cmd + self.max_delta * np.clip(a, -1.0, 1.0)
        if self.max_lead is not None:
            q = data.qpos[self.idx.arm_qpos]
            q_cmd = np.clip(q_cmd, q - self.max_lead, q + self.max_lead)
        self.q_cmd = np.clip(q_cmd, self.low, self.high)
        data.ctrl[self.idx.arm_actuators] = self.q_cmd
        return self.q_cmd.copy()

    def step(self, data: mujoco.MjData, action: np.ndarray) -> np.ndarray:
        """Apply one action and advance physics by one control interval."""
        q_cmd = self.apply(data, action)
        mujoco.mj_step(self.model, data, nstep=self.n_substeps)
        return q_cmd
