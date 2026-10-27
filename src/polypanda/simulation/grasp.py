"""Verified grasp predicate: contact alone never counts as a grasp.

The cube counts as grasped only when, for `window` consecutive updates (policy steps):
  1. both fingers are in contact with it,
  2. it is at least `min_lift` above its resting height, and
  3. it is stable in the hand: its pose in the TCP frame moves less than `max_slip` and
     rotates less than `max_slip_rot` over the window.
"""

from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

from polypanda.simulation.panda import PandaIndices, tcp_position
from polypanda.simulation.scenes import CubeIndices, cube_position


@dataclass(frozen=True)
class GraspCheckConfig:
    min_lift: float = 0.02       # m above resting height
    window: int = 5              # consecutive policy steps (0.25 s at 20 Hz)
    max_slip: float = 0.002      # m of cube motion in the hand frame within the window
    max_slip_rot: float = 0.05   # rad (~3 deg) of cube rotation in the hand frame


@dataclass(frozen=True)
class GraspState:
    left_contact: bool
    right_contact: bool
    left_force: float    # N, summed normal force between the finger and the cube
    right_force: float
    lift: float          # m above resting height
    held_steps: int      # consecutive updates with both contacts and lift
    slip: float          # m, max in-hand displacement within the window (nan if too short)
    slip_rot: float      # rad, max in-hand rotation within the window (nan if too short)
    grasped: bool

    @property
    def both_contact(self) -> bool:
        return self.left_contact and self.right_contact


class GraspMonitor:
    def __init__(self, model: mujoco.MjModel, arm: PandaIndices, cube: CubeIndices,
                 config: GraspCheckConfig | None = None) -> None:
        self.model, self.arm, self.cube = model, arm, cube
        self.config = config or GraspCheckConfig()
        self.left = model.body("left_finger").id
        self.right = model.body("right_finger").id
        self._window: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=self.config.window)
        self._held = 0
        self._force = np.zeros(6)
        # Height of the cube's centre at rest, from which lift is measured: standing, by
        # default (GraspV4Env sets it for a cube lying on a side)
        self.rest_z = float(cube.half_size[2])

    def reset(self) -> None:
        self._window.clear()
        self._held = 0

    def _finger_forces(self, data: mujoco.MjData) -> tuple[float, float]:
        forces = {self.left: 0.0, self.right: 0.0}
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if self.cube.body not in (b1, b2):
                continue
            other = b2 if b1 == self.cube.body else b1
            if other in forces:
                mujoco.mj_contactForce(self.model, data, i, self._force)
                forces[other] += float(self._force[0])
        return forces[self.left], forces[self.right]

    def update(self, data: mujoco.MjData) -> GraspState:
        cfg = self.config
        left_f, right_f = self._finger_forces(data)
        left, right = left_f > 0.0, right_f > 0.0
        lift = float(cube_position(data, self.cube)[2] - self.rest_z)

        if left and right and lift >= cfg.min_lift:
            R_tcp = data.site_xmat[self.arm.tcp_site].reshape(3, 3)
            p_rel = R_tcp.T @ (cube_position(data, self.cube) - tcp_position(data, self.arm))
            R_rel = R_tcp.T @ data.xmat[self.cube.body].reshape(3, 3)
            self._window.append((p_rel, R_rel))
            self._held += 1
        else:
            self.reset()

        slip = slip_rot = float("nan")
        if len(self._window) == cfg.window:
            p0, R0 = self._window[0]
            slip = max(float(np.linalg.norm(p - p0)) for p, _ in self._window)
            slip_rot = max(_angle(R0.T @ R) for _, R in self._window)
        grasped = (self._held >= cfg.window and slip <= cfg.max_slip
                   and slip_rot <= cfg.max_slip_rot)
        return GraspState(left, right, left_f, right_f, lift, self._held, slip, slip_rot,
                          grasped)


def _angle(R: np.ndarray) -> float:
    """Rotation angle (rad) of rotation matrix R."""
    return float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
