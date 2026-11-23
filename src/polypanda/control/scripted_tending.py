"""v4.0: the scripted tending controller, the baseline the learned door is compared against.

It extends the scripted cell with the line's moves: drive to a machine's rail stop, open and
close its lid by the handle, and load or unload the pocket (the cell's own pick and place, the
place turned to the pocket's yaw). Everything it does with a machine follows a **belief**: where
it thinks the machine stands, how it is turned, and its layout (which way the lid slides, how far,
where the handle is). Two beliefs are the v4.0 baselines (docs/v4_plan.md, section 3.2;
evaluation/tending.py builds them):
- the **nominal** script believes the machine is exactly as programmed (the nominal machine at
  its nominal place), whatever machine it faces, as a cell programmed for one machine does;
- the **camera-aware** script measures where the machine stands, how it is turned and where its
  handle and pocket are (with the camera's error), but still assumes the nominal slide
  direction, travel and latch, which a camera cannot see.
Without a belief given, the script believes the truth (an oracle, for tests and demonstrations).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from polypanda.control.ik import top_down_quat
from polypanda.control.scripted_cell import CUBE_HEIGHT, CellControllerConfig, ScriptedCell
from polypanda.environments.grasp import GRIPPER_OPEN
from polypanda.simulation.line import Line
from polypanda.simulation.machine import MachineConfig
from polypanda.simulation.scenes import CubeIndices


@dataclass
class Belief:
    """What a script believes about one machine, in the world frame."""
    origin: np.ndarray                # the pocket's centre on the bench (x, y, z)
    yaw: float                        # the machine's turn (rad; its +x points away from the robot)
    config: MachineConfig             # the layout it assumes
    handle: np.ndarray | None = None  # if measured: the handle bar's centre with the lid closed

    @property
    def R(self) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def slide_axis(self) -> np.ndarray:
        return self.R @ self.config.slide_axis

    def handle_at(self, lid_q: float) -> np.ndarray:
        """The handle's centre with the lid open by `lid_q`."""
        if self.handle is not None:
            return self.handle + self.slide_axis() * lid_q
        return self.origin + self.R @ self.config.handle_centre(lid_q)


def true_belief(line: Line, name: str) -> Belief:
    """The truth: the machine as built (for tests, demonstrations and an oracle)."""
    m = line.machines[name]
    R = line.data.xmat[m.body].reshape(3, 3)
    return Belief(origin=line.data.xpos[m.body].copy(), yaw=float(np.arctan2(R[1, 0], R[0, 0])),
                  config=line.spots[name].config)


class ScriptedTending(ScriptedCell):
    door_speed: float = 0.05          # m/s along the slide
    open_margin: float = 0.005        # m short of the open stop: a lid pinned against its stop
                                      # cannot give way to a finger closing off-centre
    latch_push: float = 0.004         # m pushed past closed, so the lid arrives firmly
    finger_span: float = 0.08         # m between the open fingers' pads

    def __init__(self, line: Line, config: CellControllerConfig | None = None,
                 beliefs: dict[str, Belief] | None = None) -> None:
        super().__init__(line, config, on_step=line.update_machines)
        self.cell: Line = line
        self.beliefs = beliefs or {n: true_belief(line, n) for n in line.machines}
        self._finger_qpos = np.array([self.m.joint(f"finger_joint{k}").qposadr[0]
                                      for k in (1, 2)])

    def go_to(self, name: str) -> None:
        """Drive to the machine's rail stop and face it (from the carry posture). The stop and
        the station angle are the line's layout, the same for every script."""
        spot = self.cell.spots[name]
        self.drive_to(spot.stop)
        self.swing_to(np.radians(spot.angle))

    def move_lid(self, name: str, opening: bool) -> None:
        """Hook the handle and slide the lid open (or closed, with a firm last push), then rise
        to the carry height. The fingers stay **open**, straddling the bar with the slide running
        between them, and the hand pushes the bar with the inside of the trailing finger. With
        some 3 cm of slack on each side of the bar, millimetres of error in where the handle is
        believed to be do not matter. (Gripping the bar did: in this simulator the Panda's
        fingertip pads jam on a bar's edge when the hand is 1-2 mm off its centre, measured.)"""
        b = self.beliefs[name]
        travel = b.config.travel - self.open_margin
        q_from, q_to = (0.0, travel) if opening else (travel, -self.latch_push)
        handle, axis = b.handle_at(q_from), b.slide_axis()
        yaw = float(np.arctan2(axis[1], axis[0]))             # the fingers open along the slide
        quat = top_down_quat(yaw)
        c = self.config
        slack = self.finger_span / 2 - b.config.handle_radius   # finger to bar, each side
        direction = np.sign(q_to - q_from)
        self.grip = GRIPPER_OPEN
        self.move_joints(self._solve([handle[0], handle[1], c.carry_height], quat, self.q_cmd))
        self.vertical(handle[:2], handle[2], yaw)
        end = handle + axis * ((q_to - q_from) + direction * slack)
        self.line(handle, end, quat, speed=self.door_speed)
        tcp = self.d.site_xpos[self.arm.tcp_site].copy()
        self.vertical(tcp[:2], c.carry_height, yaw)

    def finger_gap(self) -> float:
        """The gripper's opening (m), as its width sensor would report it."""
        return float(self.d.qpos[self._finger_qpos].sum())

    def load(self, cube: CubeIndices, name: str) -> None:
        """Set a held cube into the pocket where the script believes it is, the cube turned to
        the pocket's believed yaw (the cell's place, which lands the cube's centre within 1 mm of
        its target in closed loop, then lets go)."""
        b = self.beliefs[name]
        xy = b.origin[:2]
        carry = self.carry_yaw(self.angle_of(xy))
        # the cube is square in plan: turn by the smallest angle that squares it to the pocket
        R = self.d.xmat[cube.body].reshape(3, 3)
        cube_yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        turn = (b.yaw - cube_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
        yaw = carry + turn
        rest = CUBE_HEIGHT * 0.5
        tcp = self.d.site_xpos[self.arm.tcp_site]
        offset = tcp[2] - self.d.xpos[cube.body][2]
        self.vertical(xy, rest + self.config.release_clearance + offset, yaw, track=cube.body)
        self.cell.detach(cube)
        self.grip = GRIPPER_OPEN
        self.step(self.config.grip_steps)
        self.vertical(self.d.site_xpos[self.arm.tcp_site][:2].copy(), self.config.carry_height,
                      yaw)

    def unload(self, cube: CubeIndices) -> None:
        """Take the part out of the pocket (the cell's pick: the fingers close above the rim)."""
        self.pick(cube)
