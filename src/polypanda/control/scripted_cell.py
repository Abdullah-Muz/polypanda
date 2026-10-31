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
from polypanda.simulation.disturbances import Knock, apply_knock, knock_outcome, rest_state
from polypanda.simulation.panda import PandaIndices
from polypanda.simulation.scenes import CubeIndices, cube_yaw
from polypanda.simulation.workcell import Workcell, WorkOrder, station_xy
from polypanda.supervisor.observe import observe
from polypanda.supervisor.planner import BUFFER, TOWER, GreedyPlanner
from polypanda.supervisor.symbolic import Decision, Loose, decision_features

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
    # Recovery (v3): fallen cubes within `clear_radius` of a tower, and every lying cube (stood
    # up there), are first moved to staging spots on an inner ring (between the stations, clear
    # of towers, slots and belt), so the hand never reaches down beside a tower it has just
    # rebuilt, and no regrip happens next to one (measured: both knocked rebuilt towers over).
    clear_radius: float = 0.20
    staging: tuple[tuple[float, float], ...] = ((0.36, -65.0), (0.36, -115.0), (0.36, -15.0),
                                                (0.36, 60.0), (0.36, 15.0))


class ScriptedCell:
    # loose cubes this controller can recover, by how they rest (supervisor.observe)
    recoverable_poses: tuple[str, ...] = ("upright", "lying")

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
        self.disturbance_outcome: dict | None = None  # the order's knock, if any
        self.source = "belt"  # where the cube being handled comes from (belt, buffer, loose, stage)
        # a knock asked for from outside (the viewer's knock key): it hits the next tower a cube
        # is placed on once that tower is at least 2 cubes tall (run_mixed_order)
        self.knock_request: Knock | None = None
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

    def pick_loose(self, cube: CubeIndices, pose: str = "upright", spot=None) -> None:
        """Pick a cube lying loose on the table (after a knock), from the carry posture over it:
        an upright cube as at any station; a cube lying on a side is first stood upright at
        `spot` (stand_up), then picked there. If it does not end up upright, it is not picked."""
        if pose == "lying":
            self.stand_up(cube, self.d.xpos[cube.body][:2].copy() if spot is None else spot)
            if self.cell.attached(cube) or rest_state(self.cell, cube)["pose"] != "upright":
                return
        self.pick(cube)

    def staging_xy(self, s: int) -> np.ndarray:
        r, a = self.config.staging[s]
        return station_xy(a, r)

    def free_staging_spot(self) -> np.ndarray | None:
        """A staging spot with no cube within 8 cm of it, if any."""
        for s in range(len(self.config.staging)):
            xy = self.staging_xy(s)
            if all(np.linalg.norm(self.d.xpos[c.body][:2] - xy) > 0.08 for c in self.cell.cubes):
                return xy
        return None

    def stage(self, cube: CubeIndices, pose: str, spot) -> None:
        """Move a loose cube to a staging spot, standing it up there if it lies on a side."""
        if pose == "lying":
            self.stand_up(cube, spot)
            return
        self.pick(cube)
        if not self.cell.attached(cube):
            return
        self.swing_to(float(np.arctan2(spot[1], spot[0])))
        self.place(cube, spot, 0)

    def line(self, start, end, quat, speed: float | None = None) -> None:
        """Straight move of the hand from `start` to `end` with a fixed orientation `quat`."""
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        speed = speed or self.config.vertical_speed
        n = max(5, int(np.linalg.norm(end - start) / speed / self.dt))
        for k in range(1, n + 1):
            s = (1 - np.cos(np.pi * k / n)) / 2
            self.q_cmd = self._solve(start + (end - start) * s, quat, self.q_cmd)
            self.step()

    def stand_up(self, cube: CubeIndices, spot) -> None:
        """Stand a cube lying on its side upright at `spot` on the table (a regrip): grip it
        from above across its 4 cm width, lift, carry it over the spot, turn the hand up high to
        point sideways along the station circle (the cube's long axis turns vertical; fingers
        toward the nearest tower, so palm and wrist stay clear of it), lower it and let go
        1.5 cm above the table (measured: any lower and the hand touches the table; the cube
        drops flat), back the hand out sideways and return to the carry posture over the spot.
        Pointing the hand toward or away from the robot is out of reach (measured)."""
        held, c, yaw = self.grasp_lying(cube)
        if not held:
            return
        a, rel, R_rel = self.carry_over(cube, spot, c, yaw)
        self.stand_held(cube, spot, a, rel, R_rel, c)

    def grasp_lying(self, cube: CubeIndices) -> tuple[bool, np.ndarray, float]:
        """stand_up's grasp: from above across the lying cube's 4 cm width, then lift to the
        carry height. Returns (held, where the cube lay, the hand's closing yaw)."""
        cfg = self.config
        c = self.d.xpos[cube.body].copy()
        axis = self.d.xmat[cube.body].reshape(3, 3)[:, 2]        # long axis, horizontal
        carry_yaw = self.carry_yaw(float(np.arctan2(c[1], c[0])))
        yaw = float(np.arctan2(axis[1], axis[0])) + np.pi / 2   # fingers close across it
        yaw = carry_yaw + (yaw - carry_yaw + np.pi / 2) % np.pi - np.pi / 2
        self.move_joints(self._solve([c[0], c[1], cfg.carry_height], top_down_quat(yaw),
                                     self.q_cmd))
        self.vertical(c[:2], c[2], yaw)
        self.grip = GRIPPER_CLOSED
        self.step(cfg.grip_steps)
        held = self.cell.attach(cube)
        self.vertical(c[:2], cfg.carry_height, yaw)
        return held, c, yaw

    def carry_over(self, cube: CubeIndices, spot, c, yaw: float):
        """Carry a held cube, lifted to the carry height over where it lay (`c`; the hand at
        closing yaw `yaw`), over the spot, the hand still pointing down. Returns (the spot's
        angle, the cube's position and rotation in the hand)."""
        cfg = self.config
        tcp = self.d.site_xpos[self.arm.tcp_site].copy()
        R_tcp = self.d.site_xmat[self.arm.tcp_site].reshape(3, 3)
        rel = R_tcp.T @ (self.d.xpos[cube.body] - tcp)              # the cube in the hand
        R_rel = R_tcp.T @ self.d.xmat[cube.body].reshape(3, 3)
        a = float(np.arctan2(spot[1], spot[0]))
        # carry the lying cube over the spot first, so the hand turns sideways up high there
        # and never sweeps low past a tower (measured: it knocked rebuilt towers over)
        self.swing_to(a)
        self.move_joints(self._solve([spot[0], spot[1], cfg.carry_height], top_down_quat(yaw
                         + a - float(np.arctan2(c[1], c[0]))), self.q_cmd))
        return a, rel, R_rel

    def stand_held(self, cube: CubeIndices, spot, a: float, rel, R_rel, c) -> None:
        """stand_up's turn, from the hand over the spot: sideways along the station circle,
        down, let go, back out sideways and up to the carry posture over the spot."""
        cfg = self.config
        tangent = np.array([-np.sin(a), np.cos(a), 0.0])
        rest = np.array([spot[0], spot[1], cube.half_size[2] + 0.015])
        towers = [self.cell.tower_xy(t) for t in range(len(self.cell.config.tower_angles))]
        nearest = min(towers, key=lambda xy: np.linalg.norm(xy - spot)) - spot
        # point the fingers toward the nearest tower, so the palm and wrist stay away from it
        signs = sorted((tangent, -tangent), key=lambda z: -float(z[:2] @ nearest))
        for z_axis in signs:
            y_axis = np.cross([0.0, 0.0, 1.0], z_axis)
            R = np.column_stack([np.cross(y_axis, z_axis), y_axis, z_axis])
            if abs((R @ R_rel)[2, 2]) < 0.95:
                continue                                             # (would not stand it up)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, R.flatten())
            goal = rest - R @ rel
            q, ok = self.ik.solve(self.q_cmd, goal + [0.0, 0.0, 0.18], quat)
            if ok:
                break
        else:  # no sideways pose reachable: set the cube down over the spot as it is
            self.ik_failures += 1
            here = self.d.site_xpos[self.arm.tcp_site][:2].copy()
            self.vertical(here, c[2] + 0.002, closing_yaw(self.d.site_xmat[
                self.arm.tcp_site].reshape(3, 3)))
            self.cell.detach(cube)
            self.grip = GRIPPER_OPEN
            self.step(cfg.grip_steps)
            self.to_carry()
            return
        self.move_joints(q, rate=1.0)
        self.line(goal + [0.0, 0.0, 0.18], goal, quat)
        self.cell.detach(cube)
        self.grip = GRIPPER_OPEN
        self.step(cfg.grip_steps)
        self.line(goal, goal - 0.07 * z_axis, quat)                  # fingers slide off sideways
        self.line(goal - 0.07 * z_axis, goal - 0.07 * z_axis + [0.0, 0.0, 0.12], quat)
        q = self.carry.copy()
        q[0] += a
        self.move_joints(q, rate=1.0)

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

    def knock(self, tower: list[int], knock: Knock, log: Callable[[str], None] | None = None
              ) -> None:
        """A disturbance for testing recovery: knock a tower (cube indices, bottom first) and let
        it settle with the arm holding still. The outcome goes to `self.disturbance_outcome`."""
        cubes = [self.cell.cubes[k] for k in tower]
        before = [self.d.xpos[c.body].copy() for c in cubes]
        apply_knock(self.cell, cubes, knock, self.step)
        out = knock_outcome(self.cell, cubes, before)
        self.disturbance_outcome = {
            "kind": out["kind"] or "none", "recoverable": out["recoverable"],
            "fallen": [self.cell.cubes.index(c) for c in out["fallen"]],
            "states": [rest_state(self.cell, c) for c in out["fallen"]]}
        if log:
            log(f"knock ({knock.push}): {self.disturbance_outcome['kind']}, "
                f"{len(out['fallen'])} cubes fell, "
                f"{'recoverable' if out['recoverable'] else 'not recoverable'}")

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
                        disturbance=None, recover: bool = False,
                        log: Callable[[str], None] | None = None, spares: int = 0,
                        replace: Callable[[Decision], bool] | None = None,
                        deadline: float | None = None, times: dict | None = None
                        ) -> list[list[int]]:
        """Fulfil a Layer 2 work order (random arrival order, surplus cubes) with the greedy
        planner (supervisor.planner) and `slots` buffer slots; returns per tower the arrival
        indices of the cubes placed on it. Counts are left in `self.stats`. With `batch_choice`
        the robot may take any cube on the belt (GreedyPlanner.choose), not only the one at the
        pick point; the others wait 12 and 24 cm upstream, within reach. A `disturbance`
        (simulation.disturbances.Disturbance) knocks a tower right after it reaches the given
        height; its outcome is left in `self.disturbance_outcome`. With `recover` the controller
        reads the cell after a knock and at the end (supervisor.observe), brings its plan in line
        with the towers as they stand, and recovers loose cubes (`recover_loose`).

        v3 f, spares: up to `spares` spare cubes per order. A written-off cube the towers still
        need is replaced by a spare of its colour, which comes up the belt as one more arrival
        (Workcell.call_spare). `replace` (a recovery planner, supervisor.symbolic) is asked, as
        in the symbolic cell, whether to replace a cube that could be rescued instead of
        rescuing it, and again after each failed attempt; it sees decision_features, with the
        order's `deadline` (simulated s, or None) and the fitted action `times`."""
        cell, cfg = self.cell, self.cell.config
        cell.reset(order, rng, lateral=lateral, yaw=yaw)
        self.q_cmd, self.grip = self.home.copy(), GRIPPER_OPEN
        self.ik_failures = 0
        planner = GreedyPlanner(order.recipes, slots=slots)
        self.stats = {"buffered": 0, "from_buffer": 0, "rejected": 0, "needed_rejected": 0,
                      "taken_out_of_order": 0}
        if recover:
            self.stats.update(recovered=0, written_off=0, staged=0)
        cfg_c = self.config
        cell.conveyor.index()
        self.to_carry()
        towers: list[list[int]] = [[] for _ in order.recipes]
        parked: list[int | None] = [None] * slots   # arrival index per buffer slot
        self.disturbance_outcome = None

        written_off: set[int] = set()
        attempts: dict[int, int] = {}
        asked: dict[int, int] = {}       # attempts when the planner was last asked, per cube
        staged: set[int] = set()         # cubes moved to a staging spot and still there
        handled: set[int] = set()        # arrivals taken off the belt
        spares_left = [spares]
        if spares:
            self.stats.update(spares_used=0, replaced=0)
        t0 = self.d.time

        def colour_of(k: int) -> str:
            return cell.order.arrivals[k][0]

        def call_spare(k: int) -> None:
            """Replace written-off cube k with a spare, if the towers need it and one is left."""
            if spares_left[0] > 0 and planner.demand(colour_of(k)) > 0:
                spares_left[0] -= 1
                cell.call_spare(colour_of(k))
                self.stats["spares_used"] += 1
                if log:
                    log(f"spare {colour_of(k)} cube called up the belt")

        def ask(k: int, view) -> bool:
            """The recovery planner: replace cube k with a spare instead of rescuing it?"""
            a = attempts.get(k, 0)
            if (replace is None or spares_left[0] <= 0 or planner.demand(colour_of(k)) <= 0
                    or asked.get(k) == a or (k in staged and a == 0)):
                return False
            asked[k] = a
            xy = self.d.xpos[cell.cubes[k].body][:2]
            near = min(np.linalg.norm(xy - cell.tower_xy(t))
                       for t in range(len(order.recipes))) < cfg_c.clear_radius
            cube = Loose(colour_of(k), view.loose[k]["pose"], True, bool(near),
                         staged=k in staged, attempts=a)
            poses = [s["pose"] for j, s in view.loose.items() if j not in written_off]
            elapsed = self.d.time - t0 + times["overhead"]  # (the symbolic clock starts there)
            features = decision_features(
                cube, a > 0, spares_left[0], spares, elapsed, deadline,
                len(cell.order.arrivals) - len(handled), planner, poses, times)
            return bool(replace(Decision(cube, a > 0, features)))

        def stack(k: int, t: int) -> None:
            self.swing_to(np.radians(cfg.tower_angles[t]))
            self.place(cell.cubes[k], cell.tower_xy(t), len(towers[t]))
            towers[t].append(k)
            if (disturbance is not None and self.disturbance_outcome is None
                    and t == disturbance.tower and len(towers[t]) == disturbance.height):
                self.knock(towers[t], disturbance.knock, log)
                if recover:
                    recover_loose()
            elif self.knock_request is not None and len(towers[t]) >= 2:
                knock, self.knock_request = self.knock_request, None
                self.knock(towers[t], knock, log)
                if recover:
                    recover_loose()

        def route(k: int) -> None:
            """Send the cube in the hand (arrival k) to a tower, a buffer slot or the bin."""
            cube, colour = cell.cubes[k], colour_of(k)
            kind, where = planner.decide(colour)
            if kind == TOWER:
                if log:
                    log(f"{colour} cube -> tower {'ABC'[where]}, level {len(towers[where]) + 1}")
                stack(k, where)
                while (move := planner.next_buffer_move()) is not None:
                    slot, t = move
                    if log:
                        log(f"  buffer slot {slot + 1} ({colour_of(parked[slot])}) -> "
                            f"tower {'ABC'[t]}, level {len(towers[t]) + 1}")
                    k_slot, parked[slot] = parked[slot], None  # (before stacking: a knock there
                    self.swing_to(np.radians(cfg.buffer_angles[slot]))  # may refill the slot)
                    self.source = "buffer"
                    self.pick(cell.cubes[k_slot])
                    stack(k_slot, t)
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

        def recover_loose() -> None:
            """Closed loop: read the cell, bring the plan in line with the towers as they stand,
            and recover loose cubes one at a time (cubes a tower needs next first), routing each
            like a new arrival. Cubes this controller cannot recover are written off."""
            while True:
                view = observe(cell)
                for t, stack_now in enumerate(view.towers):
                    if len(stack_now) < len(towers[t]):
                        if log:
                            log(f"tower {'ABC'[t]}: {len(towers[t]) - len(stack_now)} cubes "
                                f"fell; {len(stack_now)} still stand")
                        towers[t][:] = towers[t][:len(stack_now)]
                        planner.placed[t] = len(stack_now)
                usable = []
                for k, state in view.loose.items():
                    if k in written_off:
                        continue
                    if state["recoverable"] and state["pose"] in self.recoverable_poses \
                            and attempts.get(k, 0) < 2:
                        usable.append(k)
                    else:
                        written_off.add(k)
                        staged.discard(k)
                        self.stats["written_off"] += 1
                        if log:
                            log(f"{colour_of(k)} cube written off ({state['pose']}, "
                                f"{'in reach' if state['reachable'] else 'out of reach'})")
                        call_spare(k)
                for k in list(usable):
                    if ask(k, view):
                        usable.remove(k)
                        written_off.add(k)
                        staged.discard(k)
                        self.stats["written_off"] += 1
                        self.stats["replaced"] += 1
                        if log:
                            log(f"{colour_of(k)} cube left for a spare (recovery planner)")
                        call_spare(k)
                if not usable:
                    return
                xy = {k: self.d.xpos[cell.cubes[k].body][:2].copy() for k in usable}
                near = [k for k in usable if view.loose[k]["pose"] == "lying"  # (stood up first)
                        or min(np.linalg.norm(xy[k] - cell.tower_xy(t))
                               for t in range(len(towers))) < cfg_c.clear_radius]
                spot = self.free_staging_spot()
                if near and spot is not None:  # clear the site first
                    k = near[0]
                    if log:
                        log(f"clearing the {colour_of(k)} cube away from the towers")
                    self.swing_to(float(np.arctan2(xy[k][1], xy[k][0])))
                    self.source = "stage"
                    self.stage(cell.cubes[k], view.loose[k]["pose"], spot)
                    moved = self.d.xpos[cell.cubes[k].body][:2]
                    if np.linalg.norm(moved - spot) < 0.03:
                        self.stats["staged"] += 1
                        staged.add(k)
                    else:
                        attempts[k] = attempts.get(k, 0) + 1   # (a failed attempt)
                    continue
                ready = [k for k in usable if any(planner.next_colour(t) == colour_of(k)
                                                  for t in range(len(towers)))]
                k = (ready or usable)[0]
                if log:
                    log(f"recovering the {colour_of(k)} cube")
                self.swing_to(float(np.arctan2(xy[k][1], xy[k][0])))
                self.source = "loose"
                self.pick_loose(cell.cubes[k], view.loose[k]["pose"], spot)
                if not cell.attached(cell.cubes[k]):
                    attempts[k] = attempts.get(k, 0) + 1   # (a failed attempt)
                    self.grip = GRIPPER_OPEN  # (not held: let go, it is looked at again)
                    self.step(self.config.grip_steps)
                    self.to_carry()
                    continue
                self.stats["recovered"] += 1
                staged.discard(k)
                route(k)

        def take(n: int) -> bool:
            """Take arrival n off the belt and route it; False if it did not arrive."""
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
                    log(f"cube {n} ({colour_of(n)}) did not arrive")
                return False
            handled.add(k)
            self.source = "belt"
            self.pick(cube)
            cell.conveyor.index()  # the belt brings the next cube while the arm is away
            route(k)
            return True

        n, stalled = 0, False
        while True:
            while not stalled and n < len(cell.order.arrivals):
                stalled = not take(n)
                n += not stalled
            if recover:
                recover_loose()      # (may call spares up the belt: then the loop goes on)
            if stalled or n >= len(cell.order.arrivals):
                break
        self.swing_to(0.0)
        self.move_joints(self.home)
        self.step(20)
        return towers

    def check(self, towers: list[list[int]]) -> list[bool]:
        """Per tower: every cube upright (either way up: the cubes are symmetric, and a knocked
        cube may be recovered upside down), at its level, on the fixture, in recipe order."""
        cell, ok = self.cell, []
        for t, recipe in enumerate(cell.order.recipes):
            good = len(towers[t]) == len(recipe)
            for level, k in enumerate(towers[t]):
                cube = cell.cubes[k]
                p = self.d.xpos[cube.body]
                tilt = np.degrees(np.arccos(min(1.0, abs(self.d.xmat[cube.body][8]))))
                good &= (tilt < 5 and abs(p[2] - CUBE_HEIGHT * (level + 0.5)) < 0.005
                         and np.linalg.norm(p[:2] - cell.tower_xy(t)) < 0.02
                         and cell.cube_colour(cube) == recipe[level])
            ok.append(bool(good))
        return ok
