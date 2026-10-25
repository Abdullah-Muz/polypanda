"""Pose (position + orientation) inverse kinematics for the TCP, for top-down grasping.

TCP frame (measured at home): z points out of the gripper (straight down at home) and the
fingers close along y. A top-down grasp therefore fixes z = world -z and turns y to the
desired closing direction, an angle in the horizontal plane called the closing yaw.
"""

import mujoco
import numpy as np

from polypanda.simulation.assets import load_panda_scene
from polypanda.simulation.panda import ARM_JOINTS, PandaIndices


def top_down_quat(closing_yaw: float) -> np.ndarray:
    """TCP orientation (w, x, y, z) pointing straight down, fingers closing along `closing_yaw`."""
    y = np.array([np.cos(closing_yaw), np.sin(closing_yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    R = np.column_stack([np.cross(y, z), y, z])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.flatten())
    return quat


def closing_yaw(R: np.ndarray) -> float:
    """Horizontal direction of the TCP y axis (finger closing axis) for rotation matrix R."""
    return float(np.arctan2(R[1, 1], R[0, 1]))


def grasp_closing_yaw(cube_yaw: float, current: float, symmetry: float = np.pi / 2) -> float:
    """Closing yaw aligned with a pair of cube faces, nearest to `current`.

    A square cross-section repeats every pi/2, so four closing yaws grasp the same cube;
    picking the nearest keeps the wrist rotation to at most 45 degrees.
    """
    k = np.round((current - cube_yaw) / symmetry)
    return float(cube_yaw + k * symmetry)


def orientation_error(R_current: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
    """Rotation vector (world frame, rad) taking the current orientation to the target."""
    current = np.zeros(4)
    mujoco.mju_mat2Quat(current, R_current.flatten())
    local = np.zeros(3)
    mujoco.mju_subQuat(local, target_quat, current)  # expressed in the current frame
    return R_current @ local


class PoseIK:
    """Damped-least-squares pose IK on the TCP site, using a private copy of the arm model.

    Rotation errors are weighted by `rot_weight` (m per rad) so that position and orientation
    share one damped least-squares problem.
    """

    def __init__(self, damping: float = 0.05, rot_weight: float = 0.2) -> None:
        self.damping = damping
        self.rot_weight = rot_weight
        self.model, self.data = load_panda_scene()
        self.idx = PandaIndices.from_model(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.idx.home_key)
        joint_ids = [self.model.joint(n).id for n in ARM_JOINTS]
        self.q_low, self.q_high = self.model.jnt_range[joint_ids].T.copy()
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """TCP position and rotation matrix at arm configuration q."""
        self.data.qpos[self.idx.arm_qpos] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        site = self.idx.tcp_site
        return self.data.site_xpos[site].copy(), self.data.site_xmat[site].reshape(3, 3).copy()

    def error(self, q, target_pos, target_quat) -> tuple[np.ndarray, np.ndarray]:
        pos, R = self.forward(q)
        return np.asarray(target_pos) - pos, orientation_error(R, np.asarray(target_quat))

    def step(self, q, target_pos, target_quat, secondary=None) -> np.ndarray:
        """Full DLS joint step toward the target pose (not scaled or limited).

        `secondary`: an optional joint motion pursued only in the task's null space,
        dq = n + J^+ (e - J n), so it does not disturb the pose (up to damping).
        """
        e_pos, e_rot = self.error(q, target_pos, target_quat)
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.idx.tcp_site)
        cols = self.idx.arm_qvel
        J = np.vstack([self._jacp[:, cols], self.rot_weight * self._jacr[:, cols]])
        e = np.concatenate([e_pos, self.rot_weight * e_rot])
        if secondary is None:
            return J.T @ np.linalg.solve(J @ J.T + self.damping**2 * np.eye(6), e)
        n = np.asarray(secondary, dtype=float)
        return n + J.T @ np.linalg.solve(J @ J.T + self.damping**2 * np.eye(6), e - J @ n)

    def solve(self, q0, target_pos, target_quat, iters: int = 400,
              pos_tol: float = 1e-4, rot_tol: float = 1e-3) -> tuple[np.ndarray, bool]:
        """Iterate to convergence within joint limits. Returns (q, converged)."""
        q = np.clip(np.asarray(q0, dtype=float), self.q_low, self.q_high)
        for _ in range(iters):
            e_pos, e_rot = self.error(q, target_pos, target_quat)
            if np.linalg.norm(e_pos) < pos_tol and np.linalg.norm(e_rot) < rot_tol:
                return q, True
            q = np.clip(q + self.step(q, target_pos, target_quat), self.q_low, self.q_high)
        e_pos, e_rot = self.error(q, target_pos, target_quat)
        return q, bool(np.linalg.norm(e_pos) < pos_tol and np.linalg.norm(e_rot) < rot_tol)

    def action(self, q, target_pos, target_quat, gain: float, max_delta: float) -> np.ndarray:
        """Normalized 7-D joint-increment action toward the target (direction-preserving)."""
        dq = gain * self.step(q, target_pos, target_quat)
        peak = np.abs(dq).max()
        if peak > max_delta:
            dq *= max_delta / peak
        return (dq / max_delta).astype(np.float32)
