"""Scripted controller for the Milestone 9 work-order cell: the baseline the learned skills will
be compared against.

Design (each rule comes from a measured failure of the first, sketch-like controller):

- One carry posture. Cubes are carried at a fixed height with a fixed arm shape, and every move
  between stations is a rotation of the base joint alone: the arm turns rigidly, the hand stays
  level, and no IK runs during a swing. (Interpolating all joints over a 200 deg move tilts the
  hand mid-swing; IK seeded far from the answer fails at tower heights.)
- Alignment for free. Stations sit on one circle and the carry posture faces whichever station
  the base points at, so a cube gripped square to the hand arrives square to the fixture; no
  re-orientation step before placing.
- The wrist turns only at the infeed, to match a slightly rotated cube, and turns back right
  after the lift (spinning a held cube is harmless; measured), so joint 7 stays mid-range.
- Short closed-loop vertical moves from the carry posture: the measured hand (or cube) position
  corrects the IK target, so the arm's sag never puts the grip or the stack off centre.
- Swings take the way round that stays inside joint 1's +-166 deg range, at a moderate rate.
- With the cell's grasp assist on, a verified grip welds the cube to the hand while carrying
  (a firmly held cube otherwise slides along the finger pads whenever the arm moves; see
  WorkcellConfig.grasp_assist); the weld is released before the fingers open.

The controller drives the cell through `step()`, one 0.05 s control step at a time; an optional
callback runs after every step (a viewer, a recorder).

Work orders: `run_order` (Layer 1) stacks cubes that arrive in an always-placeable order;
`run_mixed_order` (Layer 2) takes cubes in random order, with surplus ones, and lets the greedy
planner (supervisor.planner) send each to a tower, a buffer slot or the reject bin. Picking from
a buffer slot and placing onto one are the same pick and place as at the infeed and the towers.
"""

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from polypanda.control.ik import PoseIK, closing_yaw, grasp_closing_yaw, top_down_quat
from polypanda.environments.grasp import GRIPPER_CLOSED, GRIPPER_OPEN
from polypanda.simulation.panda import PandaIndices
from polypanda.simulation.scenes import CubeIndices, cube_yaw
from polypanda.simulation.workcell import Workcell, WorkOrder
from polypanda.supervisor.planner import BUFFER, TOWER, GreedyPlanner

CUBE_HEIGHT = 0.06


@dataclass(frozen=True)
class CellControllerConfig:
    carry_height: float = 0.32      # m, TCP: a carried cube clears a 4-cube tower by ~5 cm
    grip_turn: float = 0.0          # rad: closing direction relative to radial (0 or pi/2)
    swing_rate: float = 0.8         # rad/s, average base rate (60/60 towers at 0.5, 0.8 and 1.0)
    vertical_speed: float = 0.1     # m/s, average speed of vertical moves
    release_clearance: float = 0.004  # m above the resting height when the fingers open
    grip_steps: int = 12            # control steps to close or open the fingers (0.6 s)
    servo_gain: float = 0.5         # closed-loop correction per step
    belt_timeout: float = 20.0      # s to wait for the conveyor


class ScriptedCell:
    def __init__(self, cell: Workcell, config: CellControllerConfig | None = None,
                 on_step: Callable[[], None] | None = None) -> None:
        self.cell, self.config, self.on_step = cell, config or CellControllerConfig(), on_step
        self.m, self.d = cell.model, cell.data
        self.arm = PandaIndices.from_model(self.m)
        self.ik = PoseIK()
        self.home = self.ik.data.qpos[self.ik.idx.arm_qpos].copy()
        self.dt = 25 * self.m.opt.timestep
        self.q_cmd = self.home.copy()
        self.grip = GRIPPER_OPEN
        self.ik_failures = 0
        self.stats: dict[str, int] = {}  # per order (run_mixed_order)
        self.carry = self._carry_posture()

    # --- primitives -----------------------------------------------------------------------
    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.d.ctrl[self.arm.arm_actuators] = self.q_cmd
            self.d.ctrl[self.arm.gripper_actuator] = self.grip
            self.cell.conveyor.update()
            mujoco.mj_step(self.m, self.d, nstep=25)
            if self.on_step:
                self.on_step()

    def _solve(self, pos, quat, seed) -> np.ndarray:
        q, ok = self.ik.solve(seed, pos, quat)
        self.ik_failures += not ok
        return q

    def _carry_posture(self) -> np.ndarray:
        """Joint configuration with the hand at the carry height over the station circle,
        facing straight ahead (base angle 0); other stations: add the angle to joint 1."""
        cfg = self.cell.config
        _, R_home = self.ik.forward(self.home)
        yaw = closing_yaw(R_home) + self.config.grip_turn
        q, ok = self.ik.solve(self.home, [cfg.radius, 0.0, self.config.carry_height],
                              top_down_quat(yaw))
        if not ok:
            raise RuntimeError("carry posture unreachable")
        return q

    @property
    def base_angle(self) -> float:
        """The station direction the carry posture currently points at."""
        return float(self.q_cmd[0] - self.carry[0])

    def carry_yaw(self, angle: float) -> float:
        """Closing direction of the hand in the carry posture pointing at `angle`."""
        _, R = self.ik.forward(self.carry)
        return closing_yaw(R) + angle

    def move_joints(self, q_goal, rate: float = 1.0) -> None:
        """Joint-space move with a smooth (cosine) profile; rate (rad/s) on the largest joint."""
        q0 = self.q_cmd.copy()
        n = max(5, int(np.abs(q_goal - q0).max() / rate / self.dt))
        for k in range(1, n + 1):
            self.q_cmd = q0 + (q_goal - q0) * (1 - np.cos(np.pi * k / n)) / 2
            self.step()

    def swing_to(self, angle: float) -> None:
        """Base-only swing (from the carry posture) to face the station at `angle` (rad), the
        way round that stays inside joint 1's range."""
        q = self.q_cmd.copy()
        delta = (angle - self.base_angle + np.pi) % (2 * np.pi) - np.pi
        target = q[0] + delta
        if abs(target) > 2.89:
            target -= np.sign(target) * 2 * np.pi
        q[0] = target
        self.move_joints(q, rate=self.config.swing_rate)

    def vertical(self, xy, z_to: float, yaw: float, track=None) -> None:
        """Straight vertical move of the hand at (xy) to height z_to with top-down closing
        direction `yaw`. Going down, the target is corrected in closed loop so that `track`'s
        xy (a body id: the cube when placing; by default the hand) lands on `xy`, settled within
        1 mm. Going up, no correction: the constant small nudges would jiggle the hand and a
        held cube creeps along the finger pads (measured: 17 mm over a 0.29 m lift)."""
        cfg = self.config
        quat = top_down_quat(yaw)
        tcp = self.d.site_xpos[self.arm.tcp_site]
        z0 = float(tcp[2])
        if z_to >= z0:
            n = max(5, int((z_to - z0) / cfg.vertical_speed / self.dt))
            for k in range(1, n + 1):
                z = z0 + (z_to - z0) * (1 - np.cos(np.pi * k / n)) / 2
                self.q_cmd = self._solve([xy[0], xy[1], z], quat, self.q_cmd)
                self.step()
            return
        bias = np.zeros(3)
        goal = np.array([xy[0], xy[1], z_to])

        def measured():
            p = tcp.copy()
            if track is not None:
                p[:2] = self.d.xpos[track][:2]
            return p

        n = max(5, int(abs(z_to - z0) / cfg.vertical_speed / self.dt))
        for k in range(1, n + 1):
            z = z0 + (z_to - z0) * (1 - np.cos(np.pi * k / n)) / 2
            target = np.array([xy[0], xy[1], z])
            bias[:2] += cfg.servo_gain * (target - measured())[:2] * (k > n // 2)
            self.q_cmd = self._solve(target + bias, quat, self.q_cmd)
            self.step()
        for _ in range(20):
            error = goal - measured()
            if np.abs(error[:2]).max() < 0.001 and abs(error[2]) < 0.002:
                break
            bias += cfg.servo_gain * error
            self.q_cmd = self._solve(goal + bias, quat, self.q_cmd)
            self.step()

    def to_carry(self) -> None:
        """Straight up to the carry height, then into the carry posture at the current angle."""
        tcp = self.d.site_xpos[self.arm.tcp_site].copy()
        R = self.d.site_xmat[self.arm.tcp_site].reshape(3, 3)
        if tcp[2] < self.config.carry_height - 0.01:
            self.vertical(tcp[:2], self.config.carry_height, closing_yaw(R))
        angle = np.arctan2(tcp[1], tcp[0])
        q = self.carry.copy()
        q[0] += angle
        self.move_joints(q, rate=1.0)

    # --- skills -------------------------------------------------------------------------------
    def pick(self, cube: CubeIndices) -> None:
        """From the carry posture over the pick point: match the cube's yaw, descend, grip,
        lift back, and turn the wrist back to the carry yaw (the cube turns with it)."""
        c = self.d.xpos[cube.body].copy()
        angle = np.arctan2(c[1], c[0])
        carry_yaw = self.carry_yaw(angle)
        yaw = grasp_closing_yaw(cube_yaw(self.d, cube), carry_yaw)
        z_carry = self.config.carry_height
        self.move_joints(self._solve([c[0], c[1], z_carry], top_down_quat(yaw), self.q_cmd))
        self.vertical(c[:2], c[2], yaw)
        self.grip = GRIPPER_CLOSED
        self.step(self.config.grip_steps)
        self.cell.attach(cube)  # grasp assist, only if the grip is verified (cell config)
        self.vertical(c[:2], z_carry, yaw)
        q = self.carry.copy()
        q[0] += angle
        self.move_joints(q, rate=1.0)  # wrist back to the carry yaw

    def place(self, cube: CubeIndices, xy, level: int) -> None:
        """From the carry posture over a station: lower the cube's centre onto (xy) at `level`,
        release, and rise back to the carry posture."""
        rest = CUBE_HEIGHT * (level + 0.5)
        tcp = self.d.site_xpos[self.arm.tcp_site]
        offset = tcp[2] - self.d.xpos[cube.body][2]   # cube centre below the TCP
        yaw = self.carry_yaw(np.arctan2(xy[1], xy[0]))
        self.vertical(xy, rest + self.config.release_clearance + offset, yaw, track=cube.body)
        self.cell.detach(cube)  # the release itself is physical
        self.grip = GRIPPER_OPEN
        self.step(self.config.grip_steps)
        self.vertical(self.d.site_xpos[self.arm.tcp_site][:2].copy(), self.config.carry_height,
                      yaw)

    def reject(self, cube: CubeIndices) -> None:
        """From the carry posture over the reject bin: lower the cube into the bin's mouth, let
        go, and rise back."""
        xy = self.cell.reject_xy
        yaw = self.carry_yaw(np.arctan2(xy[1], xy[0]))
        tcp = self.d.site_xpos[self.arm.tcp_site]
        offset = tcp[2] - self.d.xpos[cube.body][2]
        drop = self.cell.config.bin_height + 0.05 + CUBE_HEIGHT / 2  # cube bottom 5 cm over walls
        self.vertical(xy, drop + offset, yaw)
        self.cell.detach(cube)
        self.grip = GRIPPER_OPEN
        self.step(self.config.grip_steps)
        self.vertical(self.d.site_xpos[self.arm.tcp_site][:2].copy(), self.config.carry_height,
                      yaw)

    # --- work orders ---------------------------------------------------------------------------
    def run_order(self, order: WorkOrder, rng=None, lateral: float = 0.0, yaw: float = 0.0,
                  log: Callable[[str], None] | None = None) -> list[list[int]]:
        """Fulfil a work order from the home pose; returns per tower the arrival indices of the
        cubes placed on it."""
        cell = self.cell
        cfg = cell.config
        cell.reset(order, rng, lateral=lateral, yaw=yaw)
        self.q_cmd, self.grip = self.home.copy(), GRIPPER_OPEN
        self.ik_failures = 0
        cell.conveyor.index()
        self.to_carry()
        towers: list[list[int]] = [[] for _ in order.recipes]
        for k, (colour, t) in enumerate(order.arrivals):
            self.swing_to(np.radians(cfg.infeed_angle))
            for _ in range(int(self.config.belt_timeout / self.dt)):
                if cell.conveyor.ready:
                    break
                self.step()
            cube = cell.conveyor.lead()
            if cube is None or cell.cubes.index(cube) != k:
                if log:
                    log(f"cube {k} ({colour}) did not arrive")
                break
            if log:
                log(f"{colour} cube -> tower {'ABC'[t]}, level {len(towers[t]) + 1}")
            self.pick(cube)
            cell.conveyor.index()  # the belt brings the next cube while the arm is away
            self.swing_to(np.radians(cfg.tower_angles[t]))
            self.place(cube, cell.tower_xy(t), len(towers[t]))
            towers[t].append(k)
        self.swing_to(0.0)
        self.move_joints(self.home)
        self.step(20)
        return towers

    def run_mixed_order(self, order: WorkOrder, rng=None, lateral: float = 0.0,
                        yaw: float = 0.0, slots: int = 3, batch_choice: bool = False,
                        log: Callable[[str], None] | None = None) -> list[list[int]]:
        """Fulfil a Layer 2 work order (random arrival order, surplus cubes) with the greedy
        planner (supervisor.planner) and `slots` buffer slots; returns per tower the arrival
        indices of the cubes placed on it. Counts are left in `self.stats`. With `batch_choice`
        the robot may take any cube on the belt (GreedyPlanner.choose), not only the one at the
        pick point; the others wait 12 and 24 cm upstream, within reach."""
        cell, cfg = self.cell, self.cell.config
        cell.reset(order, rng, lateral=lateral, yaw=yaw)
        self.q_cmd, self.grip = self.home.copy(), GRIPPER_OPEN
        self.ik_failures = 0
        planner = GreedyPlanner(order.recipes, slots=slots)
        self.stats = {"buffered": 0, "from_buffer": 0, "rejected": 0, "needed_rejected": 0,
                      "taken_out_of_order": 0}
        cell.conveyor.index()
        self.to_carry()
        towers: list[list[int]] = [[] for _ in order.recipes]
        parked: list[int | None] = [None] * slots   # arrival index per buffer slot

        def stack(k: int, t: int) -> None:
            self.swing_to(np.radians(cfg.tower_angles[t]))
            self.place(cell.cubes[k], cell.tower_xy(t), len(towers[t]))
            towers[t].append(k)

        handled: set[int] = set()
        for n in range(len(order.arrivals)):
            self.swing_to(np.radians(cfg.infeed_angle))
            for _ in range(int(self.config.belt_timeout / self.dt)):
                if cell.conveyor.ready:
                    break
                self.step()
            cube = cell.conveyor.lead()
            k = n
            if batch_choice:
                if not cell.conveyor.ready:
                    cube = None
            if cube is not None and batch_choice:
                belt = sorted((c for c in cell.cubes if cell.conveyor.on_belt(c)),
                              key=cell.conveyor.upstream_distance)
                j = planner.choose([cell.cube_colour(c) for c in belt])
                cube, k = belt[j], cell.cubes.index(belt[j])
                if j:
                    self.stats["taken_out_of_order"] += 1
                    if log:
                        log(f"took the {cell.cube_colour(cube)} cube, number {j + 1} on the belt")
                    self.swing_to(float(np.arctan2(*self.d.xpos[cube.body][1::-1])))
            if cube is None or k in handled or (not batch_choice and cell.cubes.index(cube) != k):
                if log:
                    log(f"cube {n} ({order.arrivals[n][0]}) did not arrive")
                break
            handled.add(k)
            colour = order.arrivals[k][0]
            self.pick(cube)
            cell.conveyor.index()  # the belt brings the next cube while the arm is away
            kind, where = planner.decide(colour)
            if kind == TOWER:
                if log:
                    log(f"{colour} cube -> tower {'ABC'[where]}, level {len(towers[where]) + 1}")
                stack(k, where)
                while (move := planner.next_buffer_move()) is not None:
                    slot, t = move
                    if log:
                        log(f"  buffer slot {slot + 1} ({order.arrivals[parked[slot]][0]}) -> "
                            f"tower {'ABC'[t]}, level {len(towers[t]) + 1}")
                    self.swing_to(np.radians(cfg.buffer_angles[slot]))
                    self.pick(cell.cubes[parked[slot]])
                    stack(parked[slot], t)
                    parked[slot] = None
                    self.stats["from_buffer"] += 1
            elif kind == BUFFER:
                if log:
                    log(f"{colour} cube -> buffer slot {where + 1} (needed later)")
                self.swing_to(np.radians(cfg.buffer_angles[where]))
                self.place(cube, cell.buffer_xy(where), 0)
                parked[where] = k
                self.stats["buffered"] += 1
            else:
                needed = planner.needed_rejected > self.stats["needed_rejected"]
                if log:
                    log(f"{colour} cube -> reject bin ({'buffer full' if needed else 'surplus'})")
                self.swing_to(np.radians(cfg.reject_angle))
                self.reject(cube)
                self.stats["rejected"] += 1
                self.stats["needed_rejected"] = planner.needed_rejected
        self.swing_to(0.0)
        self.move_joints(self.home)
        self.step(20)
        return towers

    def check(self, towers: list[list[int]]) -> list[bool]:
        """Per tower: every cube upright, at its level, on the fixture, in recipe order."""
        cell, ok = self.cell, []
        for t, recipe in enumerate(cell.order.recipes):
            good = len(towers[t]) == len(recipe)
            for level, k in enumerate(towers[t]):
                cube = cell.cubes[k]
                p = self.d.xpos[cube.body]
                tilt = np.degrees(np.arccos(np.clip(self.d.xmat[cube.body][8], -1, 1)))
                good &= (tilt < 5 and abs(p[2] - CUBE_HEIGHT * (level + 0.5)) < 0.005
                         and np.linalg.norm(p[:2] - cell.tower_xy(t)) < 0.02
                         and cell.cube_colour(cube) == recipe[level])
            ok.append(bool(good))
        return ok
