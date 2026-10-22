"""Non-learned reach controllers with SB3's predict() signature, for like-for-like evaluation.

Both act through the same 7-D joint-increment action interface as RL policies.

JacobianIKPolicy reads only the reach observation (schema reach-v1-27d). Its privileged
information, compared with a learned policy, is an exact copy of the robot's kinematic model,
which it uses to compute the TCP Jacobian. It knows nothing about dynamics, gravity sag or
collisions.
"""

import mujoco
import numpy as np

from polypanda.control.action_adapter import ACTION_DIM
from polypanda.environments.reach import OBS_SLICES
from polypanda.simulation.assets import load_panda_scene
from polypanda.simulation.panda import PandaIndices


class RandomPolicy:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        obs = np.asarray(observation)
        shape = (obs.shape[0], ACTION_DIM) if obs.ndim == 2 else (ACTION_DIM,)
        return self.rng.uniform(-1.0, 1.0, shape).astype(np.float32), state


class JacobianIKPolicy:
    """Damped-least-squares Cartesian controller on the TCP position.

    dq = J^T (J J^T + damping^2 I)^-1 (gain * e), scaled as a whole so no joint exceeds
    max_delta, then expressed as a normalized action. Because the adapter integrates actions
    into q_cmd, this acts as an integral controller on the Cartesian error, which removes the
    steady-state gravity sag of the position servos.
    """

    # gain 0.2: same time-to-2cm as 0.35-1.0 but without their overshoot (tuning seeds 1000-1049).
    def __init__(self, gain: float = 0.2, damping: float = 0.05, max_delta: float = 0.015):
        self.gain = gain
        self.damping = damping
        self.max_delta = max_delta
        self.model, self.data = load_panda_scene()  # private copy: the kinematic model
        self.idx = PandaIndices.from_model(self.model)
        self._jacp = np.zeros((3, self.model.nv))

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """3x7 TCP position Jacobian at arm configuration q (fingers at home)."""
        self.data.qpos[self.idx.arm_qpos] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_jacSite(self.model, self.data, self._jacp, None, self.idx.tcp_site)
        return self._jacp[:, self.idx.arm_qvel].copy()

    def _act(self, obs: np.ndarray) -> np.ndarray:
        q = obs[OBS_SLICES["q"]].astype(np.float64)
        error = obs[OBS_SLICES["target_error"]].astype(np.float64)
        J = self.jacobian(q)
        dq = J.T @ np.linalg.solve(J @ J.T + self.damping**2 * np.eye(3), self.gain * error)
        peak = np.abs(dq).max()
        if peak > self.max_delta:
            dq *= self.max_delta / peak
        return (dq / self.max_delta).astype(np.float32)

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        obs = np.asarray(observation)
        if obs.ndim == 2:
            return np.stack([self._act(o) for o in obs]), state
        return self._act(obs), state
