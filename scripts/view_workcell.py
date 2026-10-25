"""Look at the Milestone 9 stacking cell, running a scripted preview cycle.

Usage: uv run python scripts/view_workcell.py [--speed 1.0] [--static] [--jitter]
The conveyor indexes each cube to the pick point; a scripted IK controller (not the learned
skills) swings the base to face the infeed, picks the cube, swings ~200 deg to the tower fixture
and stacks it, three times, then starts over. The stack light runs a lamp test, then shows green.
--static only shows the cell. --jitter queues cubes with small random offsets (+-1.5 cm, +-15 deg).
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
from polypanda.simulation.workcell import Workcell
from polypanda.visualization.viewer import fast_rendering

APPROACH = 0.15     # m above the grasp / place height
CLEARANCE = 0.004   # m: release the cube this far above its resting height on the tower


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

    def stack_one(self, cube, level: int) -> None:
        cell = self.cell
        tower = cell.tower_xy
        c = self.d.xpos[cube.body].copy()
        # The closing direction the hand has when the home posture is turned to face the cube:
        # grasping near it keeps the wrist joint mid-range, with room to spare after the swing.
        _, R_home = self.ik.forward(self.home)
        natural = closing_yaw(R_home) + np.arctan2(c[1], c[0])
        yaw = grasp_closing_yaw(cube_yaw(self.d, cube), natural)
        print(f"  pick cube at ({c[0]:+.3f}, {c[1]:+.3f}) -> tower level {level + 1}")
        self.swing(self.solve(c[:2], c[2] + APPROACH, yaw, self.facing(c[:2])))
        self.straight(c[:2], c[2] + APPROACH, c[2], yaw, servo=True)
        self.grip = GRIPPER_CLOSED
        self.step(12)
        self.straight(c[:2], c[2], c[2] + APPROACH, yaw)
        cell.conveyor.index()  # the belt brings the next cube while the arm is away
        # Swing on the base joint alone: the arm turns rigidly about the vertical axis, so the
        # hand stays pointing down all the way (interpolating all joints tilts it mid-swing).
        q = self.q_cmd.copy()
        q[0] += np.radians(cell.config.tower_angle - cell.config.infeed_angle)
        self.swing(q, rate=0.8)
        rest = cube.half_size[2] * (2 * level + 1)  # cube-centre height on the tower
        self.place(cube, tower, rest)

    def place(self, cube, xy, rest) -> None:
        """Stand the cube upright with its faces aligned to the fixture (the scripted place's
        upright correction: the cube pivots in the weak grip), then lower the cube's centre
        straight onto (xy, rest), release and retreat."""
        R_tcp = self.d.site_xmat[self.arm.tcp_site].reshape(3, 3)
        R_cube = self.d.xmat[cube.body].reshape(3, 3)
        R_rel = R_tcp.T @ R_cube                                    # cube in the hand
        r_rel = R_tcp.T @ (self.d.xpos[cube.body] - self.d.site_xpos[self.arm.tcp_site])
        fixture = np.radians(self.cell.config.tower_angle)
        hand_yaw = closing_yaw(R_tcp)

        def orientation(yaw):  # hand orientation that holds the cube upright at this cube yaw
            Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0],
                           [0, 0, 1]])
            return Rz @ R_rel.T

        def wrist_turn(yaw):  # how far the hand would have to turn about vertical
            return abs((closing_yaw(orientation(yaw)) - hand_yaw + np.pi) % (2 * np.pi) - np.pi)

        # Any of the four face-aligned yaws stacks the cube; take the one needing the least
        # wrist turn that IK can reach.
        candidates = fixture + np.pi / 2 * np.arange(4)
        for yaw in sorted(candidates, key=wrist_turn):
            R_new = orientation(yaw)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, R_new.flatten())
            above = np.array([*xy, rest + APPROACH]) - R_new @ r_rel
            low = np.array([*xy, rest + CLEARANCE]) - R_new @ r_rel
            if all(self.ik.solve(self.q_cmd, target, quat)[1] for target in (above, low)):
                break

        def tcp_for(z):  # hand position that puts the cube's centre at (xy, z)
            return np.array([*xy, z]) - R_new @ r_rel

        p = tcp_for(rest + APPROACH)
        self.swing(self.solve(p[:2], p[2], 0.0, quat=quat), rate=0.6)
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

    def run_cycle(self, rng, jitter: bool) -> None:
        cell = self.cell
        # jitter: as left by an infeed with side guides, +-1.5 cm and +-15 deg
        cell.reset(rng, lateral=0.015 if jitter else 0.0, yaw=np.radians(15) if jitter else 0.0)
        self.q_cmd, self.grip = self.home.copy(), GRIPPER_OPEN
        for lamps in (("green", "amber", "red"), ("red",), ("amber",), ("green",)):  # lamp test
            cell.set_stack_light(*lamps)
            self.step(8)
        cell.conveyor.index()
        for level in range(cell.config.n_cubes):
            while cell.conveyor.moving:
                self.step()
            queued = [c for c in cell.cubes if cell.conveyor.on_belt(c)]
            if not queued:
                print("  no cube left on the belt")
                break
            self.stack_one(min(queued, key=cell.conveyor.upstream_distance), level)
        self.swing(self.home)
        self.step(20)
        tilt = [np.degrees(np.arccos(np.clip(self.d.xmat[c.body].reshape(3, 3)[2, 2], -1, 1)))
                for c in cell.cubes]
        heights = [self.d.xpos[c.body][2] for c in cell.cubes]
        off = [1000 * np.linalg.norm(self.d.xpos[c.body][:2] - cell.tower_xy) for c in cell.cubes]
        print(f"  tower: cube heights {np.round(heights, 3)} m, tilts {np.round(tilt, 1)} deg, "
              f"off the fixture centre {np.round(off, 1)} mm")
        self.step(int(2.0 / self.dt))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    parser.add_argument("--static", action="store_true", help="only show the cell")
    parser.add_argument("--jitter", action="store_true", help="random cube yaw and offset")
    parser.add_argument("--shadows", action="store_true", help="render shadows and reflections")
    args = parser.parse_args()
    cell = Workcell.build()
    if not args.shadows:
        fast_rendering(cell.model)
    cell.reset()
    rng = np.random.default_rng(0)
    with mujoco.viewer.launch_passive(cell.model, cell.data) as viewer:
        viewer.cam.distance, viewer.cam.lookat[:] = 2.6, [0.05, 0.0, 0.1]
        if args.static:
            cell.set_stack_light("green")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)
            return
        preview = Preview(cell, viewer, args.speed)
        try:
            cycle = 1
            while True:
                print(f"cycle {cycle}")
                preview.run_cycle(rng, args.jitter)
                cycle += 1
        except SystemExit:
            pass


if __name__ == "__main__":
    main()
