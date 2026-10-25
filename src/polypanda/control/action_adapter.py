"""Action interfaces shared by every arm policy and the classical baseline.

Joint increments (the default): a normalized action a in [-1, 1]^7 nudges seven persistent
joint targets:

    q_cmd <- clip(q_cmd + max_delta * a, ctrlrange)

The targets are written to the Menagerie position servos, which then run for
n_substeps physics steps. The gripper actuator is never written here.

Because targets are integrated, they can drift away from the measured joints (gravity sag,
contact, joint limits). Policies therefore observe q_cmd as well as q, targets are reset at
episode boundaries, and an optional max_lead bound stops the target winding up while the arm
is blocked.

Cartesian (Milestone 8): the action moves a persistent target pose of the hand instead, and
damped least-squares IK turns it into the same joint targets. See CartesianActionAdapter.
"""

import mujoco
import numpy as np

from polypanda.control.ik import PoseIK
from polypanda.simulation.panda import PandaIndices

ACTION_DIM = 7
ACTION_SCHEMA = "joint-increment-v1-7d"
CARTESIAN_SCHEMAS = {3: "cartesian-v1-3d", 4: "cartesian-v1-4d", 6: "cartesian-v1-6d"}


def make_action_adapter(model: mujoco.MjModel, idx: PandaIndices, config):
    """The adapter an ActionConfig (polypanda.environments.reach) asks for."""
    if config.space == "joint":
        return JointDeltaActionAdapter(model, idx, config.max_delta, config.n_substeps,
                                       config.max_lead)
    if config.space == "cartesian":
        c = config.cartesian
        return CartesianActionAdapter(
            model, idx, n_substeps=config.n_substeps, dof=c.dof, max_step=c.max_step,
            max_rot=c.max_rot, max_tilt=c.max_tilt, max_lead=c.max_lead, box_low=c.box_low,
            box_high=c.box_high, ik_iters=c.ik_iters, posture_gain=c.posture_gain,
            max_accel=c.max_accel)
    raise ValueError(f"unknown action space {config.space!r}; use 'joint' or 'cartesian'")


ACTION_LIMIT = 5.0  # action-space bound while the bound penalty is on


def action_limit(config) -> float:
    """Action-space bound for an ActionConfig: 1, or ACTION_LIMIT with the bound penalty."""
    return ACTION_LIMIT if config.bound_penalty > 0 else 1.0


def bound_action(action, config) -> tuple[np.ndarray, dict]:
    """Clip a policy output to [-1, 1]; with the bound penalty, also charge for the excess.

    PPO's Gaussian mean can drift beyond the bounds, where every sample clips to the same
    action and training gets no signal to bring it back (seen in the Cartesian transport
    pilots). With bound_penalty > 0 the action space is widened to +-ACTION_LIMIT and outputs
    beyond +-1 cost bound_penalty x squared excess (as rl_games' bounds loss), which gives that
    signal. Returns (clipped action, extra reward terms).
    """
    a = np.asarray(action, dtype=np.float64)
    clipped = np.clip(a, -1.0, 1.0)
    if config.bound_penalty <= 0:
        return clipped, {}
    return clipped, {"action_bound": -config.bound_penalty * float(np.sum((a - clipped) ** 2))}


def action_schema(config) -> str:
    """Action schema name recorded with a policy trained on this ActionConfig."""
    if config.space == "cartesian":
        return CARTESIAN_SCHEMAS[config.cartesian.dof]
    return ACTION_SCHEMA


class JointDeltaActionAdapter:
    action_dim = ACTION_DIM
    schema = ACTION_SCHEMA

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


def _rotation(rotvec: np.ndarray) -> np.ndarray:
    """Rotation matrix for a rotation vector (axis x angle, rad)."""
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3)
    quat, R = np.zeros(4), np.zeros(9)
    mujoco.mju_axisAngle2Quat(quat, rotvec / angle, angle)
    mujoco.mju_quat2Mat(R, quat)
    return R.reshape(3, 3)


def _tilt(R: np.ndarray) -> float:
    """Angle (rad) between the TCP z axis and straight down."""
    return float(np.arccos(np.clip(-R[2, 2], -1.0, 1.0)))


def _limit_tilt(R: np.ndarray, max_tilt: float) -> np.ndarray:
    """Tip R back toward pointing straight down (TCP z = world -z) if it tilts beyond max_tilt."""
    tilt = _tilt(R)
    if tilt <= max_tilt:
        return R
    axis = np.cross(R[:, 2], [0.0, 0.0, -1.0])
    return _rotation(axis / np.linalg.norm(axis) * (tilt - max_tilt)) @ R


class CartesianActionAdapter:
    """Hand-motion actions: the policy moves the gripper, IK moves the joints.

    A persistent target pose of the TCP is integrated from the action, like the joint targets of
    JointDeltaActionAdapter:

        4-D (x, y, z, yaw): position += max_step * a[:3]; the gripper points straight down and
                            its closing direction turns by max_rot * a[3] about world z. A hand
                            that starts tilted (e.g. at a hand-off) is levelled by up to max_rot
                            per step, so the joint targets never jump.
        3-D (x, y, z):      as 4-D with the wrist's yaw held, for tasks that never need to turn
                            it (transport: a 4-D policy learned to spin the wrist and fling the
                            cube out of the weak grip).
        6-D (x, y, z, rx, ry, rz): position as above; orientation turns by the world-frame
                            rotation vector max_rot * a[3:], kept within max_tilt of pointing
                            down.

    Optional max_accel (m/s^2) limits how much the target's velocity may change per step, as
    industrial arms limit Cartesian acceleration: sudden starts and stops shake a weakly held
    cube loose (Milestone 8 transport diagnosis).

    The target stays within max_lead (per axis) of the hand pose the current joint targets
    command, and inside a box, so it cannot wind up while the arm is blocked. Each step,
    ik_iters damped least-squares iterations (PoseIK, exact kinematics on a private model) move
    the joint targets q_cmd to the new pose, after a small null-space pull toward the home
    posture that keeps the redundant elbow from drifting over long episodes. q_cmd goes to the
    same position servos as joint actions, so the physics and the hand-off contract
    (reset_to_measured, adopt_ctrl) are unchanged.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        idx: PandaIndices,
        n_substeps: int = 25,
        dof: int = 4,
        max_step: float = 0.01,
        max_rot: float = 0.05,
        max_tilt: float = 0.6,
        max_lead: float = 0.03,
        box_low=(0.15, -0.5, 0.01),
        box_high=(0.85, 0.5, 0.6),
        ik_iters: int = 2,
        posture_gain: float = 0.02,
        max_accel: float | None = None,
    ) -> None:
        if max_accel is not None and max_accel <= 0:
            raise ValueError("max_accel must be > 0 or None")
        if dof not in CARTESIAN_SCHEMAS:
            raise ValueError(f"dof must be 3, 4 or 6, got {dof}")
        if min(max_step, max_rot, max_lead) <= 0 or n_substeps < 1 or ik_iters < 1:
            raise ValueError("max_step, max_rot, max_lead must be > 0; n_substeps, ik_iters >= 1")
        self.model = model
        self.idx = idx
        self.n_substeps = n_substeps
        self.dof = dof
        self.action_dim = dof
        self.schema = CARTESIAN_SCHEMAS[dof]
        self.max_step, self.max_rot, self.max_tilt = max_step, max_rot, max_tilt
        self.max_lead = max_lead
        self.box_low, self.box_high = np.asarray(box_low, float), np.asarray(box_high, float)
        self.ik_iters, self.posture_gain = ik_iters, posture_gain
        self.max_accel = max_accel
        self._velocity_step = np.zeros(3)  # last step's target displacement (m)
        self.low, self.high = model.actuator_ctrlrange[idx.arm_actuators].T.copy()
        self.ik = PoseIK()
        self.q_home = self.ik.data.qpos[self.ik.idx.arm_qpos].copy()  # PoseIK starts at home
        self.q_cmd = np.zeros(ACTION_DIM)
        self.target_pos = np.zeros(3)
        self.target_R = np.eye(3)

    @property
    def control_dt(self) -> float:
        """Simulated seconds per policy step."""
        return self.n_substeps * self.model.opt.timestep

    @property
    def at_limit(self) -> np.ndarray:
        """Per-joint mask of targets sitting on a ctrlrange bound."""
        return (self.q_cmd <= self.low) | (self.q_cmd >= self.high)

    def _sync_target(self) -> None:
        """Target pose := the pose the joint targets command (no lead, so nothing jumps)."""
        self.target_pos, self.target_R = self.ik.forward(self.q_cmd)
        self._velocity_step = np.zeros(3)  # start from rest

    def reset_to_measured(self, data: mujoco.MjData) -> None:
        """Episode start: joint targets become the measured joints; the hand target follows."""
        self.q_cmd = np.clip(data.qpos[self.idx.arm_qpos], self.low, self.high)
        data.ctrl[self.idx.arm_actuators] = self.q_cmd
        self._sync_target()

    def adopt_ctrl(self, data: mujoco.MjData) -> None:
        """Policy hand-off: keep the servo commands already in place, so nothing jumps."""
        self.q_cmd = np.clip(data.ctrl[self.idx.arm_actuators], self.low, self.high)
        self._sync_target()

    def apply(self, data: mujoco.MjData, action: np.ndarray) -> np.ndarray:
        """Move the target pose by one action, solve IK, write the joint targets. Does not step."""
        a = np.asarray(action, dtype=np.float64)
        if a.shape != (self.dof,):
            raise ValueError(f"action must have shape ({self.dof},), got {a.shape}")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"action contains non-finite values: {a}")
        a = np.clip(a, -1.0, 1.0)
        commanded, _ = self.ik.forward(self.q_cmd)
        move = self.max_step * a[:3]
        if self.max_accel is not None:  # bound the change of velocity per step
            dv = move - self._velocity_step
            limit = self.max_accel * self.control_dt ** 2
            norm = float(np.linalg.norm(dv))
            if norm > limit:
                move = self._velocity_step + dv * (limit / norm)
        pos = np.clip(self.target_pos + move, commanded - self.max_lead, commanded + self.max_lead)
        pos = np.clip(pos, self.box_low, self.box_high)
        self._velocity_step = pos - self.target_pos
        self.target_pos = pos
        if self.dof in (3, 4):
            yaw = self.max_rot * a[3] if self.dof == 4 else 0.0
            R = _rotation(np.array([0.0, 0.0, yaw])) @ self.target_R
            self.target_R = _limit_tilt(R, max(_tilt(R) - self.max_rot, 0.0))
        else:
            self.target_R = _limit_tilt(_rotation(self.max_rot * a[3:]) @ self.target_R,
                                        self.max_tilt)
        self.q_cmd = np.clip(self._solve(self.q_cmd), self.low, self.high)
        data.ctrl[self.idx.arm_actuators] = self.q_cmd
        return self.q_cmd.copy()

    def _solve(self, q: np.ndarray) -> np.ndarray:
        ik, quat = self.ik, np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.target_R.flatten())
        for k in range(self.ik_iters):
            posture = self.posture_gain * (self.q_home - q) if k == 0 else None
            q = np.clip(q + ik.step(q, self.target_pos, quat, posture), ik.q_low, ik.q_high)
        return q

    def step(self, data: mujoco.MjData, action: np.ndarray) -> np.ndarray:
        """Apply one action and advance physics by one control interval."""
        q_cmd = self.apply(data, action)
        mujoco.mj_step(self.model, data, nstep=self.n_substeps)
        return q_cmd
