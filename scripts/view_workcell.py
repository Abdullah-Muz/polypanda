"""Look at the Milestone 9 work-order cell, running scripted preview orders.

Usage: uv run python scripts/view_workcell.py [--speed 1.0] [--static] [--jitter] [--seed 0]
Each order gives the three towers a random recipe (1-4 cubes, four colours, bottom to top),
shown on the signs behind them. Cubes come out of the hopper through the tunnel in an order that
can always go straight onto a tower; the conveyor indexes each one to the pick point. A scripted
IK controller (not the learned skills) picks it, lifts it clear of the towers, swings the base
to the tower that needs it and stacks it. The stack light runs a lamp test, then shows green.
--static only shows the cell. --jitter loads cubes with small random offsets (+-1.5 cm, +-15 deg).
Close the window to stop.
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from polypanda.control.ik import PoseIK, closing_yaw, grasp_closing_yaw, top_down_quat
from polypanda.environments.grasp import GRIPPER_CLOSED, GRIPPER_OPEN
from polypanda.simulation.panda import PandaIndices
from polypanda.simulation.scenes import cube_yaw
from polypanda.simulation.workcell import TOWERS, Workcell, WorkOrder
from polypanda.visualization.viewer import fast_rendering

APPROACH = 0.15     # m above the grasp / place height
CLEARANCE = 0.004   # m: release the cube this far above its resting height on the tower
CUBE_HEIGHT = 0.06
CLEAR_TOWERS = 0.08  # m: a carried cube's centre stays this far above the tallest tower top


class Preview:
    def __init__(self, cell: Workcell, viewer, speed: float) -> None:
        self.cell, self.viewer, self.speed = cell, viewer, speed
        self.m, self.d = cell.model, cell.data
        self.arm = PandaIndices.from_model(self.m)
        self.ik = PoseIK()
        self.home = self.ik.data.qpos[self.ik.idx.arm_qpos].copy()
        self.dt = 25 * self.m.opt.timestep
        self.q_cmd = self.home.copy()
        self.grip = GRIPPER_OPEN

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            if not self.viewer.is_running():
                raise SystemExit
            start = time.perf_counter()
            self.d.ctrl[self.arm.arm_actuators] = self.q_cmd
            self.d.ctrl[self.arm.gripper_actuator] = self.grip
            self.cell.conveyor.update()
            mujoco.mj_step(self.m, self.d, nstep=25)
            self.viewer.sync()
            time.sleep(max(0.0, self.dt / self.speed - (time.perf_counter() - start)))

    def solve(self, xy, z, yaw, seed=None, quat=None):
        """IK for the TCP at (xy, z): top-down with closing direction `yaw`, or `quat`."""
        seed = self.q_cmd if seed is None else seed
        quat = top_down_quat(yaw) if quat is None else quat
        q, ok = self.ik.solve(seed, [xy[0], xy[1], z], quat)
        if not ok:
            print(f"  (IK did not converge for {np.round(xy, 3)}, z {z:.3f})")
        return q

    def swing(self, q_goal, rate: float = 1.0) -> None:
        """Joint-space move with a smooth (cosine) profile; rate in rad/s on the largest joint."""
        q0 = self.q_cmd.copy()
        n = max(10, int(np.abs(q_goal - q0).max() / rate / self.dt))
        for k in range(1, n + 1):
            self.q_cmd = q0 + (q_goal - q0) * (1 - np.cos(np.pi * k / n)) / 2
            self.step()

    def straight(self, xy, z_from, z_to, yaw, speed: float = 0.1, servo: bool = False) -> None:
        """Vertical straight-line move of the hand, IK every step. With servo, the target is
        corrected in closed loop by the measured hand position (the arm sags a few mm under
        gravity; an off-centre grip lets the cube pivot out of the weak grip)."""
        n = max(5, int(abs(z_to - z_from) / speed / self.dt))
        bias = np.zeros(3)
        for k in range(1, n + 1):
            z = z_from + (z_to - z_from) * (1 - np.cos(np.pi * k / n)) / 2
            goal = np.array([xy[0], xy[1], z])
            if servo:
                bias += 0.5 * (goal - self.d.site_xpos[self.arm.tcp_site])
            self.q_cmd = self.solve(goal[:2] + bias[:2], z + bias[2], yaw)
            self.step()
        for _ in range(20 if servo else 0):  # settle on the goal to within 1 mm
            error = goal - self.d.site_xpos[self.arm.tcp_site]
            if np.abs(error).max() < 0.001:
                break
            bias += 0.5 * error
            self.q_cmd = self.solve(goal[:2] + bias[:2], goal[2] + bias[2], yaw)
            self.step()

    def facing(self, xy) -> np.ndarray:
        seed = self.home.copy()
        seed[0] = np.clip(np.arctan2(xy[1], xy[0]), -2.89, 2.89)
        return seed

    def safe_height(self, heights) -> float:
        """TCP height at which a carried cube clears every tower (and its own approach)."""
        return max(APPROACH + 0.03, max(heights) * CUBE_HEIGHT + CLEAR_TOWERS)

    def lift_to(self, z: float) -> None:
        """Straight up (or down) to height z, keeping the hand's current orientation."""
        p = self.d.site_xpos[self.arm.tcp_site].copy()
        R = self.d.site_xmat[self.arm.tcp_site].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R.flatten())
        n = max(5, int(abs(z - p[2]) / 0.12 / self.dt))
        for k in range(1, n + 1):
            zk = p[2] + (z - p[2]) * (1 - np.cos(np.pi * k / n)) / 2
            self.q_cmd = self.solve(p[:2], zk, 0.0, quat=quat)
            self.step()

    def turn_base(self, to_angle: float, rate: float = 0.8) -> None:
        """Swing on the base joint alone: the arm turns rigidly about the vertical axis, so the
        hand keeps its height and orientation all the way (interpolating all joints tilts it
        mid-swing and a carried cube slides out)."""
        here = np.arctan2(*self.d.site_xpos[self.arm.tcp_site][1::-1])
        q = self.q_cmd.copy()
        q[0] += (to_angle - here + np.pi) % (2 * np.pi) - np.pi
        q[0] = np.clip(q[0], -2.89, 2.89)
        self.swing(q, rate=rate)

    def deliver(self, cube, t: int, level: int, heights) -> None:
        """Pick the cube at the pick point and stack it on tower t at `level`."""
        cell = self.cell
        c = self.d.xpos[cube.body].copy()
        # The closing direction the hand has when the home posture is turned to face the cube:
        # grasping near it keeps the wrist joint mid-range, with room to spare after the swing.
        _, R_home = self.ik.forward(self.home)
        natural = closing_yaw(R_home) + np.arctan2(c[1], c[0])
        yaw = grasp_closing_yaw(cube_yaw(self.d, cube), natural)
        print(f"  {cell.cube_colour(cube):>6s} cube -> tower {TOWERS[t]}, level {level + 1}")
        safe = self.safe_height(heights)
        self.lift_to(safe)
        self.turn_base(np.arctan2(c[1], c[0]))
        self.swing(self.solve(c[:2], c[2] + APPROACH, yaw, self.facing(c[:2])))
        self.straight(c[:2], c[2] + APPROACH, c[2], yaw, servo=True)
        self.grip = GRIPPER_CLOSED
        self.step(12)
        self.lift_to(safe)
        cell.conveyor.index()  # the belt brings the next cube while the arm is away
        self.turn_base(np.radians(cell.config.tower_angles[t]))
        rest = cube.half_size[2] * (2 * level + 1)  # cube-centre height on the tower
        self.place(cube, cell.tower_xy(t), rest, np.radians(cell.config.tower_angles[t]))

    def place(self, cube, xy, rest, fixture: float) -> None:
        """Stand the cube upright with its faces aligned to the fixture (the scripted place's
        upright correction: the cube pivots in the weak grip), then lower the cube's centre
        straight onto (xy, rest), release and retreat."""
        R_tcp = self.d.site_xmat[self.arm.tcp_site].reshape(3, 3)
        R_cube = self.d.xmat[cube.body].reshape(3, 3)
        R_rel = R_tcp.T @ R_cube                                    # cube in the hand
        r_rel = R_tcp.T @ (self.d.xpos[cube.body] - self.d.site_xpos[self.arm.tcp_site])
        hand_yaw = closing_yaw(R_tcp)

        def orientation(yaw):  # hand orientation that holds the cube upright at this cube yaw
            Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0],
                           [0, 0, 1]])
            return Rz @ R_rel.T

        def wrist_turn(yaw):  # how far the hand would have to turn about vertical
            return abs((closing_yaw(orientation(yaw)) - hand_yaw + np.pi) % (2 * np.pi) - np.pi)

        # Any of the four face-aligned yaws stacks the cube; take the one needing the least
        # wrist turn that IK can reach, from the current pose or the home posture turned to
        # face the tower (high on a tall tower, the current pose can lead IK astray).
        candidates = fixture + np.pi / 2 * np.arange(4)
        q_above = None
        for yaw in sorted(candidates, key=wrist_turn):
            R_new = orientation(yaw)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, R_new.flatten())
            above = np.array([*xy, rest + APPROACH]) - R_new @ r_rel
            low = np.array([*xy, rest + CLEARANCE]) - R_new @ r_rel
            for seed in (self.q_cmd, self.facing(xy)):
                q_hi, ok_hi = self.ik.solve(seed, above, quat)
                if ok_hi and self.ik.solve(q_hi, low, quat)[1]:
                    q_above = q_hi
                    break
            if q_above is not None:
                break

        def tcp_for(z):  # hand position that puts the cube's centre at (xy, z)
            return np.array([*xy, z]) - R_new @ r_rel

        p = tcp_for(rest + APPROACH)
        if q_above is None:
            print(f"  (no reachable placing pose over {np.round(xy, 3)} at {rest:.2f} m)")
            q_above = self.solve(p[:2], p[2], 0.0, quat=quat)
        self.swing(q_above, rate=0.6)
        n = int(APPROACH / 0.06 / self.dt)
        bias = np.zeros(2)  # closed loop on the cube's measured position (arm sag, grip offset)
        for k in range(1, n + 1):
            z = rest + APPROACH - (APPROACH - CLEARANCE) * (1 - np.cos(np.pi * k / n)) / 2
            bias += 0.3 * (np.asarray(xy) - self.d.xpos[cube.body][:2])
            p = tcp_for(z)
            self.q_cmd = self.solve(p[:2] + bias, p[2], 0.0, quat=quat)
            self.step()
        for _ in range(20):  # settle over the tower to within 1 mm before letting go
            error = np.asarray(xy) - self.d.xpos[cube.body][:2]
            if np.abs(error).max() < 0.001:
                break
            bias += 0.5 * error
            self.q_cmd = self.solve(p[:2] + bias, p[2], 0.0, quat=quat)
            self.step()
        self.grip = GRIPPER_OPEN
        self.step(12)
        for k in range(1, n + 1):
            p = tcp_for(rest + CLEARANCE + (APPROACH - CLEARANCE) * k / n)
            self.q_cmd = self.solve(p[:2] + bias, p[2], 0.0, quat=quat)
            self.step()

    def run_order(self, order: WorkOrder, rng, jitter: bool) -> list[list[int]]:
        """Fulfil one work order; returns, per tower, the arrival indices of its cubes."""
        cell = self.cell
        # jitter: as left by an infeed with side guides, +-1.5 cm and +-15 deg
        cell.reset(order, rng, lateral=0.015 if jitter else 0.0,
                   yaw=np.radians(15) if jitter else 0.0)
        self.q_cmd, self.grip = self.home.copy(), GRIPPER_OPEN
        for lamps in (("green", "amber", "red"), ("red",), ("amber",), ("green",)):  # lamp test
            cell.set_stack_light(*lamps)
            self.step(8)
        print("  recipes: " + "; ".join(f"{TOWERS[t]} " + "-".join(r)
                                        for t, r in enumerate(order.recipes)))
        cell.conveyor.index()
        towers: list[list[int]] = [[] for _ in order.recipes]
        for k, (_, t) in enumerate(order.arrivals):
            for _ in range(int(20 / self.dt)):  # wait (up to 20 s) for the belt to deliver
                if cell.conveyor.ready:
                    break
                self.step()
            cube = cell.conveyor.lead()
            if cube is None or cell.cubes.index(cube) != k:
                print("  the expected cube did not arrive")
                break
            self.deliver(cube, t, len(towers[t]), [len(s) for s in towers])
            towers[t].append(k)
        self.lift_to(self.safe_height([len(s) for s in towers]))
        self.turn_base(0.0)
        self.swing(self.home)
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
                tilt = np.degrees(np.arccos(np.clip(self.d.xmat[cube.body].reshape(3, 3)[2, 2],
                                                    -1, 1)))
                good &= (tilt < 5 and abs(p[2] - CUBE_HEIGHT * (level + 0.5)) < 0.005
                         and np.linalg.norm(p[:2] - cell.tower_xy(t)) < 0.02
                         and cell.cube_colour(cube) == recipe[level])
            ok.append(bool(good))
        return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    parser.add_argument("--static", action="store_true", help="only show the cell")
    parser.add_argument("--jitter", action="store_true", help="random cube yaw and offset")
    parser.add_argument("--shadows", action="store_true", help="render shadows and reflections")
    parser.add_argument("--seed", type=int, default=0, help="random work orders from this seed")
    args = parser.parse_args()
    cell = Workcell.build()
    if not args.shadows:
        fast_rendering(cell.model)
    rng = np.random.default_rng(args.seed)
    cell.reset(WorkOrder.random(rng))
    with mujoco.viewer.launch_passive(cell.model, cell.data) as viewer:
        cam = viewer.cam  # over the robot's shoulder toward the towers and their signs
        cam.azimuth, cam.elevation, cam.distance, cam.lookat[:] = 200, -28, 2.9, [0, -0.05, 0.15]
        if args.static:
            cell.set_stack_light("green")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)
            return
        preview = Preview(cell, viewer, args.speed)
        try:
            n = 1
            while True:
                print(f"order {n}")
                towers = preview.run_order(WorkOrder.random(rng), rng, args.jitter)
                ok = preview.check(towers)
                print("  towers " + ", ".join(f"{TOWERS[t]} {'OK' if g else 'WRONG'}"
                                              for t, g in enumerate(ok)))
                preview.step(int(3.0 / preview.dt))
                n += 1
        except SystemExit:
            pass


if __name__ == "__main__":
    main()
