"""Scripted place: stand the cube upright, lower it onto the target, release, retreat up.

UPRIGHT (optional, on by default) counter-rotates the gripper at the pre-place height so the
cube, which may have pivoted in the weak grip, stands upright: a 4 x 4 x 6 cm cube released
tilted beyond ~34 deg (atan(2/3)) topples. LOWER steers the cube's current centre (privileged)
to just above its resting pose over the target, keeping it upright; PlaceEnv's gate opens the
fingers once the cube is low and on target. After a short WAIT, RETREAT lifts the hand up.
"""

from enum import IntEnum

import mujoco
import numpy as np

from polypanda.control.ik import PoseIK
from polypanda.control.scripted_carry import carry_action, level_quat
from polypanda.simulation.panda import tcp_position
from polypanda.simulation.scenes import cube_tilt, cube_yaw


class PlacePhase(IntEnum):
    UPRIGHT = 0
    LOWER = 1
    WAIT = 2
    RETREAT = 3


def upright_quat(data: mujoco.MjData, arm, cube) -> np.ndarray:
    """Gripper orientation that would hold the cube upright (keeping its yaw) in the current
    in-hand pose: R_tcp' = Rz(yaw) * R_rel^T, with R_rel the cube's orientation in the hand."""
    R_tcp = data.site_xmat[arm.tcp_site].reshape(3, 3)
    R_rel = R_tcp.T @ data.xmat[cube.body].reshape(3, 3)
    yaw = cube_yaw(data, cube)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0],
                   [0.0, 0.0, 1.0]])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (Rz @ R_rel.T).flatten())
    return quat


class ScriptedPlace:
    def __init__(self, gain: float = 0.2, clearance: float = 0.003, wait_steps: int = 8,
                 retreat_height: float = 0.10, upright: bool = True,
                 upright_tol: float = np.radians(5), upright_steps: int = 40) -> None:
        self.gain, self.clearance = gain, clearance
        self.wait_steps, self.retreat_height = wait_steps, retreat_height
        self.upright, self.upright_tol, self.upright_steps = upright, upright_tol, upright_steps
        self.ik = None
        self.phase = PlacePhase.LOWER
        self._waited = self._righting = 0
        self._quat = None
        self._hold = None
        self._retreat_to = None

    def reset(self, env) -> None:
        if self.ik is None:
            self.ik = PoseIK()
        self._quat = level_quat(env.data, env.arm)
        self.phase = PlacePhase.UPRIGHT if self.upright else PlacePhase.LOWER
        self._waited = self._righting = 0
        self._hold = env.data.xpos[env.cube.body].copy()  # cube centre to hold while righting
        self._retreat_to = None

    def __call__(self, env, obs) -> np.ndarray:
        if self.upright and self.phase in (PlacePhase.UPRIGHT, PlacePhase.LOWER):
            self._quat = upright_quat(env.data, env.arm, env.cube)
        if self.phase == PlacePhase.UPRIGHT:
            self._righting += 1
            if (cube_tilt(env.data, env.cube) < self.upright_tol
                    or self._righting > self.upright_steps):
                self.phase = PlacePhase.LOWER
            else:
                return carry_action(self.ik, env.data, env.arm, env.cube, self._hold,
                                    self._quat, self.gain, env.adapter.max_delta)
        if self.phase == PlacePhase.LOWER and env.released:
            self.phase = PlacePhase.WAIT
        if self.phase == PlacePhase.WAIT:
            self._waited += 1
            if self._waited > self.wait_steps:
                self.phase = PlacePhase.RETREAT
                self._retreat_to = tcp_position(env.data, env.arm) + [0, 0, self.retreat_height]
            return np.zeros(7, dtype=np.float32)
        q = env.data.qpos[env.arm.arm_qpos]
        if self.phase == PlacePhase.RETREAT:
            return self.ik.action(q, self._retreat_to, self._quat, self.gain,
                                  env.adapter.max_delta)
        goal = env.target + [0.0, 0.0, self.clearance]
        return carry_action(self.ik, env.data, env.arm, env.cube, goal, self._quat, self.gain,
                            env.adapter.max_delta)
