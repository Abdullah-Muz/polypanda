"""v4.0: the scripted tending controller, the baseline the learned door is compared against.

It extends the scripted cell with the line's moves: drive to a machine's rail stop, open and
close its lid by the handle, and load or unload the pocket (the cell's own pick and place). The
door script knows the machine's layout exactly as configured (`MachineConfig`): which way the
lid slides, how far, where the handle is. Two variants of it are the v4.0 baselines
(docs/v4_plan.md, section 3.2):
- the **nominal** script is given one machine's config (the default) whatever machine it faces,
  as a cell programmed for one machine is;
- the **camera-aware** script reads the handle's and pocket's positions from a camera estimate
  but still assumes the nominal slide direction, travel and latch.
Here both read positions from the configs given to them; the camera's error model is applied by
the evaluation that compares them.
"""

from __future__ import annotations

import numpy as np

from polypanda.control.ik import top_down_quat
from polypanda.control.scripted_cell import CellControllerConfig, ScriptedCell
from polypanda.environments.grasp import GRIPPER_CLOSED, GRIPPER_OPEN
from polypanda.simulation.line import Line
from polypanda.simulation.machine import MachineConfig
from polypanda.simulation.scenes import CubeIndices


class ScriptedTending(ScriptedCell):
    door_speed: float = 0.05          # m/s along the slide
    latch_push: float = 0.004         # m pushed past closed, so the lid arrives firmly

    def __init__(self, line: Line, config: CellControllerConfig | None = None,
                 believed: dict[str, MachineConfig] | None = None) -> None:
        super().__init__(line, config, on_step=line.update_machines)
        self.cell: Line = line
        # what the script believes about each machine (by default, the truth)
        self.believed = believed or {n: s.config for n, s in line.spots.items()}

    def _frame(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """The machine's position and rotation (world)."""
        b = self.cell.machines[name].body
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def go_to(self, name: str) -> None:
        """Drive to the machine's rail stop and face it (from the carry posture)."""
        spot = self.cell.spots[name]
        self.drive_to(spot.stop)
        self.swing_to(np.radians(spot.angle))

    def _handle_world(self, name: str, lid_q: float) -> tuple[np.ndarray, np.ndarray]:
        """Where the script believes the handle is, and the slide direction (world)."""
        cfg = self.believed[name]
        p, R = self._frame(name)
        return p + R @ cfg.handle_centre(lid_q), R @ cfg.slide_axis

    def move_lid(self, name: str, opening: bool) -> None:
        """Pinch the handle from above, fingers closing along the slide, slide the lid open (or
        closed, with a firm last push), let go and rise to the carry height."""
        cfg = self.believed[name]
        q_from, q_to = (0.0, cfg.travel) if opening else (cfg.travel, -self.latch_push)
        handle, axis = self._handle_world(name, q_from)
        yaw = float(np.arctan2(axis[1], axis[0]))             # fingers close along the slide
        quat = top_down_quat(yaw)
        c = self.config
        self.grip = GRIPPER_OPEN
        self.move_joints(self._solve([handle[0], handle[1], c.carry_height], quat, self.q_cmd))
        self.vertical(handle[:2], handle[2], yaw)
        self.grip = GRIPPER_CLOSED
        self.step(c.grip_steps)
        end = handle + axis * (q_to - q_from)
        self.line(handle, end, quat, speed=self.door_speed)
        self.grip = GRIPPER_OPEN
        self.step(c.grip_steps)
        tcp = self.d.site_xpos[self.arm.tcp_site].copy()
        self.vertical(tcp[:2], c.carry_height, yaw)

    def load(self, cube: CubeIndices, name: str) -> None:
        """Set a held cube into the machine's pocket (the cell's place, aimed at the pocket)."""
        p, _ = self._frame(name)
        self.place(cube, p[:2], 0)

    def unload(self, cube: CubeIndices) -> None:
        """Take the part out of the pocket (the cell's pick: the fingers close above the rim)."""
        self.pick(cube)
