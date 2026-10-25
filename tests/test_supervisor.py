from dataclasses import replace

import mujoco
import numpy as np
import pytest

from polypanda.control.scripted_place import ScriptedPlace
from polypanda.environments.grasp import GRIPPER_OPEN, GraspConfig, GraspV2Env
from polypanda.environments.reach import ActionConfig, CartesianConfig
from polypanda.environments.transport import TransportConfig, TransportEnv
from polypanda.evaluation.grasp import ScriptedDriver
from polypanda.evaluation.transport import ScriptedCarryDriver
from polypanda.simulation.scenes import cube_position
from polypanda.supervisor.fsm import PickPlaceSupervisor, Stage, SupervisorConfig

CUBE, TARGET = (0.45, -0.10, 0.3), (0.55, 0.12)


class DropOnceCarry(ScriptedCarryDriver):
    """Opens the fingers a few steps into the first transport, then carries normally."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def __call__(self, env, obs):
        self.calls += 1
        if self.calls == 5:
            env.data.ctrl[env.arm.gripper_actuator] = GRIPPER_OPEN
        return super().__call__(env, obs)


@pytest.fixture(scope="module")
def supervisor():
    return PickPlaceSupervisor(ScriptedDriver(), ScriptedCarryDriver(), ScriptedPlace())


def test_scripted_pick_and_place_succeeds(supervisor):
    r = supervisor.run(CUBE, TARGET)
    assert r.success and r.grasp_attempts == 1 and r.failure_stage is None
    assert [s.stage for s in r.records] == ["grasp", "transport", "place"]
    assert r.final_xy_error_mm < 20
    env = supervisor.envs[Stage.PLACE]
    np.testing.assert_allclose(env.data.xpos[env.cube.body][:2], TARGET, atol=0.02)


def test_handoff_transfers_the_exact_state(supervisor):
    seen = {}

    def record(stage, env):
        if stage not in seen:  # first step of each stage: arm continues where it was
            seen[stage] = env.data.qpos[env.arm.arm_qpos].copy()

    supervisor.on_step = record
    try:
        supervisor.run(CUBE, TARGET)
    finally:
        supervisor.on_step = None
    grasp_end = supervisor.envs[Stage.GRASP].data.qpos[supervisor.envs[Stage.GRASP].arm.arm_qpos]
    assert np.abs(seen[Stage.TRANSPORT] - grasp_end).max() < 0.02  # one control step apart


def test_dropped_cube_is_regrasped():
    sup = PickPlaceSupervisor(ScriptedDriver(), DropOnceCarry(), ScriptedPlace())
    r = sup.run(CUBE, TARGET)
    stages = [(s.stage, s.outcome) for s in r.records]
    assert stages[:3] == [("grasp", "ok"), ("transport", "lost_grip"), ("home", "ok")]
    assert r.grasp_attempts == 2 and r.success


def test_cube_on_its_side_is_not_regraspable(supervisor):
    env = supervisor.envs[Stage.GRASP]
    env.reset(options={"cube": CUBE})
    q = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]
    env.data.qpos[env.cube.qpos:env.cube.qpos + 7] = [0.5, 0.0, 0.02, *q]
    mujoco.mj_forward(env.model, env.data)
    assert supervisor._regraspable(env)[0] == "cube tipped over"
    env.reset(options={"cube": (0.34, 0.0, 0.0)})  # just outside the placement region: fine
    assert supervisor._regraspable(env)[0] is None


def test_time_budget_ends_the_episode():
    sup = PickPlaceSupervisor(ScriptedDriver(), ScriptedCarryDriver(), ScriptedPlace(),
                              SupervisorConfig(time_budget_s=2.0))
    r = sup.run(CUBE, TARGET)
    assert not r.success and r.failure_stage == "grasp"
    assert "time budget" in r.failure_reason and r.total_s <= 2.0


def test_timed_switching_hands_over_blindly_and_never_retries():
    cfg = SupervisorConfig(timed_switching_s=(9.0, 3.0, 4.0))
    sup = PickPlaceSupervisor(ScriptedDriver(), ScriptedCarryDriver(), ScriptedPlace(), cfg)
    r = sup.run(CUBE, TARGET)
    assert r.success
    assert [(s.stage, s.steps) for s in r.records] == [("grasp", 180), ("transport", 60),
                                                       ("place", 80)]
    short = PickPlaceSupervisor(ScriptedDriver(), ScriptedCarryDriver(), ScriptedPlace(),
                                SupervisorConfig(timed_switching_s=(2.0, 3.0, 4.0)))
    r = short.run(CUBE, TARGET)  # 2 s is too short to grasp: no detection, no retry
    assert not r.success and r.grasp_attempts == 1
    assert "home" not in [s.stage for s in r.records]


class HandMotionCarry:
    """Stand-in for a learned transport with hand-motion actions: move the cube to the goal."""

    def reset(self, env):
        pass

    def __call__(self, env, obs):
        return np.clip((env.goal - cube_position(env.data, env.cube)) / 0.02, -1, 1)


class Idle:
    """A grasp policy that never acts: every attempt times out and needs recovery."""

    def reset(self, env):
        pass

    def __call__(self, env, obs):
        return np.zeros(env.action_space.shape, dtype=np.float32)


def _cartesian(dof):
    return ActionConfig(space="cartesian", cartesian=CartesianConfig(dof=dof))


def test_recovery_works_after_a_hand_motion_skill_fails():
    config = replace(GraspConfig(), action=_cartesian(4),
                     episode=replace(GraspConfig().episode, horizon=20))
    sup = PickPlaceSupervisor(Idle(), ScriptedCarryDriver(), ScriptedPlace(),
                              SupervisorConfig(max_grasp_attempts=2), GraspV2Env(config))
    r = sup.run(CUBE, TARGET)
    assert not r.success and r.grasp_attempts == 2 and r.failure_stage == "grasp"
    assert [s.stage for s in r.records] == ["grasp", "home", "grasp", "home"]
    assert "out of attempts" in r.failure_reason


def test_learned_transport_runs_in_its_own_environment():
    transport_env = TransportEnv(replace(TransportConfig(), action=_cartesian(3)))
    sup = PickPlaceSupervisor(ScriptedDriver(), HandMotionCarry(), ScriptedPlace(),
                              transport_env=transport_env)
    assert sup.envs[Stage.TRANSPORT] is transport_env
    r = sup.run(CUBE, TARGET)
    assert r.success and [s.stage for s in r.records] == ["grasp", "transport", "place"]
