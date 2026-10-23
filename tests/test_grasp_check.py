"""The grasp predicate must only accept real grasps. Each test stages a situation physically."""

import mujoco
import pytest

from polypanda.control.ik import PoseIK, closing_yaw, grasp_closing_yaw, top_down_quat
from polypanda.simulation.grasp import GraspMonitor
from polypanda.simulation.panda import PandaIndices, reset_to_home
from polypanda.simulation.scenes import CubeIndices, load_pick_scene, set_cube_pose

SUBSTEPS = 25  # physics steps per policy step (20 Hz)


@pytest.fixture(scope="module")
def ik():
    return PoseIK()


class Rig:
    """Pick scene with helpers to teleport the arm and drive it at 20 Hz."""

    def __init__(self, ik, xy=(0.5, 0.0), yaw=0.0, cube_offset=(0.0, 0.0)):
        self.model, self.data = load_pick_scene()
        m, d = self.model, self.data
        self.arm, self.cube = PandaIndices.from_model(m), CubeIndices.from_model(m)
        self.monitor = GraspMonitor(m, self.arm, self.cube)
        reset_to_home(m, d, self.arm)
        q_home = d.qpos[self.arm.arm_qpos].copy()
        self.quat = top_down_quat(grasp_closing_yaw(yaw, closing_yaw(ik.forward(q_home)[1])))
        z = self.cube.half_size[2]
        self.q_grasp, ok1 = ik.solve(q_home, [*xy, z], self.quat)
        self.q_lift, ok2 = ik.solve(self.q_grasp, [*xy, z + 0.10], self.quat)
        assert ok1 and ok2
        set_cube_pose(m, d, self.cube, (xy[0] + cube_offset[0], xy[1] + cube_offset[1]), yaw)
        d.qpos[self.arm.arm_qpos] = self.q_grasp
        d.ctrl[self.arm.arm_actuators] = self.q_grasp
        mujoco.mj_forward(m, d)
        self.states = []

    def policy_steps(self, n, arm_target=None, gripper=None, ramp_from=None):
        d = self.data
        if gripper is not None:
            d.ctrl[self.arm.gripper_actuator] = gripper
        for k in range(1, n + 1):
            if arm_target is not None:
                start = ramp_from if ramp_from is not None else arm_target
                d.ctrl[self.arm.arm_actuators] = start + (arm_target - start) * min(1.0, k / n)
            mujoco.mj_step(self.model, d, nstep=SUBSTEPS)
            self.states.append(self.monitor.update(d))
        return self.states[-1]

    def grasp_and_lift(self):
        self.policy_steps(4)                    # settle with fingers open
        self.policy_steps(10, gripper=0.0)      # close (0.5 s)
        self.policy_steps(20, arm_target=self.q_lift, ramp_from=self.q_grasp)  # lift 10 cm
        return self.policy_steps(20)            # hold 1 s


def test_resting_cube_is_not_grasped(ik):
    rig = Rig(ik)
    rig.data.qpos[rig.arm.arm_qpos] = rig.q_lift          # gripper 10 cm above the cube
    rig.data.ctrl[rig.arm.arm_actuators] = rig.q_lift
    state = rig.policy_steps(10)
    assert not state.left_contact and not state.right_contact
    assert state.lift == pytest.approx(0.0, abs=1e-3) and not state.grasped


def test_squeezing_on_the_table_is_not_a_grasp(ik):
    rig = Rig(ik)
    rig.policy_steps(4)
    state = rig.policy_steps(20, gripper=0.0)             # close, but never lift
    assert state.both_contact and state.left_force > 0.5 and state.right_force > 0.5
    assert state.lift < 0.02 and not state.grasped
    assert not any(s.grasped for s in rig.states)


def test_airborne_cube_without_contact_is_not_grasped(ik):
    rig = Rig(ik)
    rig.data.qpos[rig.arm.arm_qpos] = rig.q_lift
    rig.data.ctrl[rig.arm.arm_actuators] = rig.q_lift
    set_cube_pose(rig.model, rig.data, rig.cube, (0.3, 0.3), 0.0)
    rig.data.qpos[rig.cube.qpos + 2] = 0.4                # 37 cm up, nowhere near the hand
    mujoco.mj_forward(rig.model, rig.data)
    state = rig.policy_steps(1)
    assert state.lift > 0.3 and not state.left_contact and not state.right_contact
    assert not state.grasped


def test_single_finger_contact_is_detected_and_not_a_grasp(ik):
    # Cube 1.2 cm off-centre along the closing axis: one finger reaches it first.
    rig = Rig(ik, cube_offset=(0.012, 0.0))
    rig.policy_steps(2)
    rig.policy_steps(10, gripper=0.0)
    singles = [s for s in rig.states if s.left_contact != s.right_contact]
    assert singles, "expected at least one step with exactly one finger touching"
    assert not any(s.grasped for s in singles)


@pytest.mark.parametrize("xy,yaw", [((0.5, 0.0), 0.0), ((0.45, 0.15), 0.6), ((0.4, 0.2), 2.5)])
def test_real_grasp_is_accepted(ik, xy, yaw):
    rig = Rig(ik, xy, yaw)
    state = rig.grasp_and_lift()
    assert state.grasped
    assert state.both_contact and state.lift > 0.08
    assert state.left_force == pytest.approx(1.0, abs=0.3)   # ~2 N squeeze shared by 2 fingers
    assert state.slip < 0.002 and state.slip_rot < 0.05
    first = next(i for i, s in enumerate(rig.states) if s.grasped)
    assert rig.states[first].held_steps >= rig.monitor.config.window


def test_opening_the_gripper_loses_the_grasp(ik):
    rig = Rig(ik)
    assert rig.grasp_and_lift().grasped
    rig.policy_steps(10, gripper=255.0)                   # open while lifted
    state = rig.states[-1]
    assert not state.grasped and not state.both_contact
    assert state.lift < 0.01                              # the cube fell back to the table


def test_slipping_cube_is_rejected(ik):
    rig = Rig(ik)
    assert rig.grasp_and_lift().grasped
    # MuJoCo combines contact friction with max() across the two geoms, so lowering only the
    # cube's friction changes nothing; raising its priority makes its own friction apply.
    rig.model.geom_priority[rig.cube.geom] = 1
    rig.model.geom_friction[rig.cube.geom] = [0.05, 0.0, 0.0]
    rig.policy_steps(10)
    assert not rig.states[-1].grasped
    assert any(not s.grasped for s in rig.states[-10:-5])  # lost within 5 policy steps
