"""Scripted top-down grasp: the classical baseline for the grasp skill.

Phases: APPROACH (above the cube) -> DESCEND (around it) -> CLOSE -> LIFT -> HOLD -> DONE.
The arm moves only through the same 7-D joint-increment actions a learned policy uses, steered
by PoseIK (integral action through q_cmd, as in the reach IK baseline). The gripper command is
returned separately. Privileged information: the true cube pose and an exact kinematic model.
"""

from dataclasses import dataclass
from enum import IntEnum

import mujoco
import numpy as np

from polypanda.control.ik import (
    PoseIK,
    closing_yaw,
    grasp_closing_yaw,
    orientation_error,
    top_down_quat,
)
from polypanda.simulation.panda import PandaIndices, tcp_position
from polypanda.simulation.scenes import CubeIndices, cube_position, cube_yaw

GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0


class Phase(IntEnum):
    APPROACH = 0
    DESCEND = 1
    CLOSE = 2
    LIFT = 3
    HOLD = 4
    DONE = 5


@dataclass(frozen=True)
class ScriptedGraspConfig:
    pregrasp_height: float = 0.10  # m above the cube centre
    lift_height: float = 0.10      # m above the grasp position
    gain: float = 0.2              # IK gain, as for the reach baseline
    descend_speed: float = 0.5     # fraction of the full joint speed while descending
    pos_tol: float = 0.004         # m, to leave APPROACH / DESCEND / LIFT
    rot_tol: float = 0.03          # rad
    close_steps: int = 10          # policy steps with the gripper closing (0.5 s)
    hold_steps: int = 20           # policy steps holding at the top (1 s)
    phase_timeout: int = 120       # policy steps allowed per moving phase (6 s)


class ScriptedGrasp:
    def __init__(self, model: mujoco.MjModel, arm: PandaIndices, cube: CubeIndices,
                 max_delta: float, config: ScriptedGraspConfig | None = None) -> None:
        self.model, self.arm, self.cube = model, arm, cube
        self.max_delta = max_delta
        self.config = config or ScriptedGraspConfig()
        self.ik = PoseIK()
        self.phase = Phase.APPROACH
        self.phase_steps = 0
        self.timed_out: Phase | None = None
        self._quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._grasp_pos = np.zeros(3)

    def reset(self, data: mujoco.MjData) -> None:
        R = data.site_xmat[self.arm.tcp_site].reshape(3, 3)
        yaw = grasp_closing_yaw(cube_yaw(data, self.cube), closing_yaw(R))
        self._quat = top_down_quat(yaw)
        self.phase, self.phase_steps, self.timed_out = Phase.APPROACH, 0, None

    def _target(self, data: mujoco.MjData) -> np.ndarray:
        cfg, centre = self.config, cube_position(data, self.cube)
        if self.phase == Phase.APPROACH:
            return centre + [0.0, 0.0, cfg.pregrasp_height]
        if self.phase == Phase.DESCEND:
            return centre
        if self.phase in (Phase.LIFT, Phase.HOLD):
            return self._grasp_pos + [0.0, 0.0, cfg.lift_height]
        return tcp_position(data, self.arm)  # CLOSE / DONE: stay put

    def _at(self, data: mujoco.MjData, target: np.ndarray) -> bool:
        R = data.site_xmat[self.arm.tcp_site].reshape(3, 3)
        pos_err = np.linalg.norm(target - tcp_position(data, self.arm))
        rot_err = np.linalg.norm(orientation_error(R, self._quat))
        return pos_err < self.config.pos_tol and rot_err < self.config.rot_tol

    def _advance(self, phase: Phase) -> None:
        self.phase, self.phase_steps = phase, 0

    def act(self, data: mujoco.MjData) -> tuple[np.ndarray, float]:
        """Return (7-D arm action, gripper control) for the current state."""
        cfg = self.config
        self.phase_steps += 1
        target = self._target(data)
        moving = self.phase in (Phase.APPROACH, Phase.DESCEND, Phase.LIFT)
        if moving and self.phase_steps > cfg.phase_timeout:
            self.timed_out = self.phase
            self._advance(Phase.DONE)
        elif self.phase == Phase.APPROACH and self._at(data, target):
            self._advance(Phase.DESCEND)
        elif self.phase == Phase.DESCEND and self._at(data, target):
            self._grasp_pos = tcp_position(data, self.arm)
            self._advance(Phase.CLOSE)
        elif self.phase == Phase.CLOSE and self.phase_steps > cfg.close_steps:
            self._advance(Phase.LIFT)
        elif self.phase == Phase.LIFT and self._at(data, target):
            self._advance(Phase.HOLD)
        elif self.phase == Phase.HOLD and self.phase_steps > cfg.hold_steps:
            self._advance(Phase.DONE)

        gripper = GRIPPER_OPEN if self.phase <= Phase.DESCEND else GRIPPER_CLOSED
        if self.phase in (Phase.CLOSE, Phase.DONE):
            return np.zeros(7, dtype=np.float32), gripper
        q = data.qpos[self.arm.arm_qpos]
        action = self.ik.action(q, self._target(data), self._quat, cfg.gain, self.max_delta)
        if self.phase == Phase.DESCEND:
            action *= cfg.descend_speed
        return action, gripper
