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
from polypanda.environments.grasp import GRIPPER_CLOSED, GRIPPER_OPEN
from polypanda.simulation.line import Line
from polypanda.simulation.machine import MachineConfig, rib_geom
from polypanda.simulation.scenes import CubeIndices, cube_yaw


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
    wrist_margin: float = 0.05        # rad kept clear of joint 7's limits when turning a part
    regrasp_angle: float = -30.0      # deg: where a keyed part is set down to be regrasped (the
                                      # raw part's spot at a machine's stop, free once picked)

    def __init__(self, line: Line, config: CellControllerConfig | None = None,
                 beliefs: dict[str, Belief] | None = None) -> None:
        super().__init__(line, config, on_step=line.update_machines)
        self.cell: Line = line
        self.beliefs = beliefs or {n: true_belief(line, n) for n in line.machines}
        self._finger_qpos = np.array([self.m.joint(f"finger_joint{k}").qposadr[0]
                                      for k in (1, 2)])
        self.regrasps = 0

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
        its target in closed loop, then lets go). A keyed part going into a pocket believed
        keyed is turned so that its rib faces the slot (`keyed_turn`; pick it with `pick_for`,
        or it may need a regrasp first)."""
        b = self.beliefs[name]
        xy = b.origin[:2]
        carry = self.carry_yaw(self.angle_of(xy))
        if b.config.keyed and rib_geom(self.m, cube) >= 0:
            turn = self.keyed_turn(cube, b, xy, carry, name)
        else:
            # the cube is square in plan: turn by the smallest angle that squares it to the pocket
            turn = (b.yaw - cube_yaw(self.d, cube) + np.pi / 4) % (np.pi / 2) - np.pi / 4
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

    # --- keyed parts (docs/keyed_part.md) ----------------------------------------------------
    def rib_yaw(self, cube: CubeIndices) -> float:
        """Which way the part's rib faces (rad, world). Read from the simulator here; a real
        cell would read it with the camera (the rib stands out of the part's side and is seen
        from above), and a belief could carry it with that camera's error."""
        return cube_yaw(self.d, cube)          # the rib is on the part's local +x face

    def _wrist_turns(self, turn: float, q7: float, hand_yaw: float | None = None,
                     machine_yaw: float = 0.0) -> list[float]:
        """The ways the wrist (at joint 7 angle `q7`) can turn by `turn` (mod 2 pi) within
        joint 7's range, with `wrist_margin` to spare, smallest first. The hand points straight
        down, so turning it by +t about the vertical moves joint 7 by about -t (measured). With
        `hand_yaw` given (the hand's closing direction before the turn), only turns that leave
        the fingers closing along the machine's +-x count: the hand (about 200 x 65 mm) passes
        the lid's opening (240 x 100 mm) only lengthwise."""
        lo, hi = self.ik.q_low[6] + self.wrist_margin, self.ik.q_high[6] - self.wrist_margin
        base = (turn + np.pi) % (2 * np.pi) - np.pi
        options = [base + k * 2 * np.pi for k in (-1, 0, 1)]
        if hand_yaw is not None:
            options = [t for t in options
                       if abs((hand_yaw + t - machine_yaw + np.pi / 2) % np.pi - np.pi / 2)
                       <= np.pi / 4]
        return sorted((t for t in options if lo <= q7 - t <= hi), key=abs)

    def keyed_grasp(self, cube: CubeIndices, name: str) -> tuple[float, float] | None:
        """For a keyed part standing on the bench, to be loaded into machine `name`: the
        closing yaw to grasp it with, so that its rib can then be turned to the slot with the
        hand passing the opening lengthwise, and the wrist turn this needs at the pick. The
        fingers close across one of its four face pairs (on the rib's face too: the rib is
        under them); of the grasps both wrist turns allow (at the pick and at the machine),
        the one with the least turning in all. None if no grasp works."""
        b = self.beliefs[name]
        slot = b.yaw + b.config.key_yaw
        rib = self.rib_yaw(cube)
        carry_pick = self.carry_yaw(self.angle_of(self.d.xpos[cube.body]))
        carry_load = self.carry_yaw(self.angle_of(b.origin))
        best = None
        for k in range(4):
            g = rib + k * np.pi / 2
            for pick_turn in self._wrist_turns(g - carry_pick, self.carry[6]):
                # back in the carry posture the rib sits at (rib - g) from the closing direction
                load_turns = self._wrist_turns(slot - (carry_load + rib - g), self.carry[6],
                                               carry_load, b.yaw)
                if load_turns:
                    cost = abs(pick_turn) + abs(load_turns[0])
                    if best is None or cost < best[0]:
                        best = (cost, carry_pick + pick_turn, pick_turn)
        return None if best is None else best[1:]

    def pick_for(self, cube: CubeIndices, name: str) -> None:
        """Pick a part that is to be loaded into machine `name`: the cell's pick, except that a
        keyed part going into a keyed pocket is grasped by `keyed_grasp` (the wrist may turn up
        to 180 deg at the pick)."""
        b = self.beliefs[name]
        grasp = (self.keyed_grasp(cube, name)
                 if b.config.keyed and rib_geom(self.m, cube) >= 0 else None)
        if grasp is None:
            self.pick(cube)
            return
        yaw, pick_turn = grasp
        c = self.d.xpos[cube.body].copy()
        angle = self.angle_of(c)
        z_carry = self.config.carry_height
        seed = self.q_cmd.copy()
        seed[6] -= pick_turn
        self.move_joints(self._solve([c[0], c[1], z_carry], top_down_quat(yaw), seed))
        self.vertical(c[:2], c[2], yaw)
        self.grip = GRIPPER_CLOSED
        self.step(self.config.grip_steps)
        self.cell.attach(cube)
        self.vertical(c[:2], z_carry, yaw)
        q = self.carry.copy()
        q[0] += angle
        self.move_joints(q, rate=1.0)  # wrist back to the carry yaw (the part turns with it)

    def keyed_turn(self, cube: CubeIndices, b: Belief, xy, carry: float, name: str) -> float:
        """Turn the held part (modulo 360 deg) so that its rib faces the believed slot, at the
        carry height before going down: the shortest turn within joint 7's range that keeps the
        hand lengthwise to the opening. A part held so that no such turn exists (gripped across
        the wrong faces, e.g. by the cell's plain pick) is set down beside the machine and
        picked again with `pick_for` (a regrasp)."""
        slot = b.yaw + b.config.key_yaw
        turns = self._wrist_turns(slot - self.rib_yaw(cube), self.q_cmd[6], carry, b.yaw)
        if not turns:
            self.regrasp(cube, name)
            self.swing_to(self.angle_of(xy))
            turns = self._wrist_turns(slot - self.rib_yaw(cube), self.q_cmd[6], carry, b.yaw)
            if not turns:
                raise RuntimeError("no wrist turn brings the rib to the slot")
        turn = turns[0]
        seed = self.q_cmd.copy()
        seed[6] -= turn
        z = self.d.site_xpos[self.arm.tcp_site][2]
        self.move_joints(self._solve([xy[0], xy[1], z], top_down_quat(carry + turn), seed))
        return turn

    def regrasp(self, cube: CubeIndices, name: str) -> None:
        """Set the held part down on the bench at `regrasp_angle` (at this rail stop) as it is
        held, and pick it up again with `pick_for` machine `name`."""
        self.regrasps += 1
        self.swing_to(np.radians(self.regrasp_angle))
        a = np.radians(self.regrasp_angle)
        spot = self.base_xy() + self.cell.config.radius * np.array([np.cos(a), np.sin(a)])
        yaw = self.carry_yaw(self.angle_of(spot))
        offset = self.d.site_xpos[self.arm.tcp_site][2] - self.d.xpos[cube.body][2]
        self.vertical(spot, CUBE_HEIGHT * 0.5 + self.config.release_clearance + offset, yaw,
                      track=cube.body)
        self.cell.detach(cube)
        self.grip = GRIPPER_OPEN
        self.step(self.config.grip_steps)
        self.vertical(self.d.site_xpos[self.arm.tcp_site][:2].copy(), self.config.carry_height,
                      yaw)
        self.to_carry()
        self.pick_for(cube, name)

    def unload(self, cube: CubeIndices) -> None:
        """Take the part out of the pocket (the cell's pick: the fingers close above the rim)."""
        self.pick(cube)
