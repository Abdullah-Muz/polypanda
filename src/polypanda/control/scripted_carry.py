"""Scripted closed-loop carry: the classical transport controller.

PoseIK steers the hand so that the cube's *current* centre (privileged: true cube pose and an
exact kinematic model) moves to the goal, with the gripper levelled to point straight down.
Closing the loop on the cube, not on the pickup-time offset, compensates in-hand slip.
Used directly as the transport baseline and as the base controller for residual RL.
"""

import mujoco
import numpy as np

from polypanda.control.ik import PoseIK, closing_yaw, top_down_quat
from polypanda.simulation.panda import PandaIndices, tcp_position
from polypanda.simulation.scenes import CubeIndices, cube_position


def level_quat(data: mujoco.MjData, arm: PandaIndices) -> np.ndarray:
    """Top-down gripper orientation keeping the current closing direction."""
    return top_down_quat(closing_yaw(data.site_xmat[arm.tcp_site].reshape(3, 3)))


def carry_action(ik: PoseIK, data: mujoco.MjData, arm: PandaIndices, cube: CubeIndices,
                 goal: np.ndarray, quat: np.ndarray, gain: float,
                 max_delta: float) -> np.ndarray:
    """Normalized 7-D action moving the cube's current centre toward `goal`."""
    offset = cube_position(data, cube) - tcp_position(data, arm)
    q = data.qpos[arm.arm_qpos]
    return ik.action(q, goal - offset, quat, gain, max_delta)
