"""v4: the door skill on the rail line: its scripts and its observations are frame-correct when the
robot's base stands on a rail carriage (StationView(origin=...) sets the station and the base)."""

from dataclasses import replace

import numpy as np
import pytest

from polypanda.control.door_baselines import BaselineDoor
from polypanda.control.scripted_door import DoorExpert
from polypanda.control.scripted_tending import ScriptedTending
from polypanda.environments.door import DoorConfig, DoorEnv, DoorSensingConfig
from polypanda.environments.grasp import GRIPPER_OPEN
from polypanda.evaluation.tending import NOMINAL_SPOT, Variant, build
from polypanda.simulation.machine import MachineConfig
from polypanda.simulation.panda import tcp_position
from polypanda.supervisor.stations import StationView

ANGLE = np.radians(NOMINAL_SPOT.angle)       # M1: rail stop -0.6, station angle -90 deg
NOISELESS = replace(DoorConfig(), sensing=replace(
    DoorSensingConfig(), handle_bias=0.0, handle_noise=0.0, bar_yaw_bias=0.0,
    bar_yaw_noise=0.0, force_noise=0.0))


def line_door(config: DoorConfig | None = None):
    line = build(Variant("nominal", 0, 0.0))
    ctl = ScriptedTending(line)
    ctl.go_to("M1")
    env = DoorEnv(config or DoorConfig(), scene=(line.model, line.data),
                  machine=line.machines["M1"])
    view = StationView(env, ANGLE, origin=ctl.base_xy())
    return line, ctl, env, view


def _run(env, view, actor, command: str) -> dict:
    view.reset(command=command)
    actor.reset()
    info = {}
    for _ in range(env.config.horizon):
        _, _, terminated, truncated, info = view.step(actor.act())
        if terminated or truncated:
            break
    return info


def test_the_line_sets_the_door_station_and_the_hand_target_is_in_the_world():
    _, ctl, env, view = line_door()
    np.testing.assert_allclose(ctl.base_xy(), [NOMINAL_SPOT.stop, 0.0], atol=0.005)
    assert env.frame == pytest.approx(ANGLE)
    np.testing.assert_allclose(env.base[:2], ctl.base_xy())
    view.reset(command="open")
    # target_pos is the commanded hand in the world: within the servos' sag (~1 cm at go_to's
    # pose) of the measured TCP, not the base's 0.6 m away
    tcp = tcp_position(env.data, env.arm)
    assert np.linalg.norm(env.adapter.target_pos - tcp) < 0.02


@pytest.mark.parametrize("kind", ["expert", "feel", "camera", "nominal"])
def test_the_door_scripts_open_and_close_the_nominal_machine_on_the_rail(kind):
    line, ctl, env, view = line_door()
    actor = DoorExpert(env) if kind == "expert" else BaselineDoor(env, kind)
    info = _run(env, view, actor, "open")
    assert info["is_success"], (kind, "open", info["failure"])
    assert info["stats"]["peak_force"] < 60.0
    ctl.q_cmd = env.adapter.q_cmd.copy()       # back to the carry posture, as the line does
    ctl.grip = GRIPPER_OPEN
    ctl.to_carry()
    ctl.step(5)
    info = _run(env, view, actor, "close")
    assert info["is_success"], (kind, "close", info["failure"])
    assert line.machines["M1"].latched


def _nominal_episode(env: DoorEnv) -> dict:
    mc = MachineConfig()
    ep = env.draw_episode(np.random.default_rng(0), "open")
    ep.update(slide="right", travel=mc.travel, friction=mc.lid_friction, damping=mc.lid_damping,
              spring=0.0, handle_offset=0.0, handle_shift=0.0, handle_height=mc.handle_height,
              latch_speed=mc.latch_speed, offset=(0.0, 0.0), turn=0.0, lid=0.0,
              jitter=(0.0, 0.0, 0.0, 0.0), handle_bias=(0.0, 0.0, 0.0), bar_yaw_bias=0.0)
    return ep


def test_the_line_observes_as_the_own_scene_at_the_same_machine_pose():
    ref = DoorEnv(NOISELESS)
    ref.reset(seed=0, options={"episode": _nominal_episode(ref)})
    line, _, env, view = line_door(NOISELESS)
    q = ref.data.qpos[ref.arm.arm_qpos].copy()
    q[0] += ANGLE                              # the same arm pose, turned to the station
    d = line.data
    d.qpos[env.arm.arm_qpos] = q
    d.ctrl[env.arm.arm_actuators] = q
    for adr, ref_adr in zip(env._finger_qpos, ref._finger_qpos, strict=True):
        d.qpos[adr] = ref.data.qpos[ref_adr]
    d.ctrl[env.arm.gripper_actuator] = ref.data.ctrl[ref.arm.gripper_actuator]
    d.qvel[env.arm.arm_qpos] = 0.0
    obs_ref, _ = ref.reset(options={"continue": True, "command": "open"})
    obs_line, _ = view.reset(command="open")
    np.testing.assert_allclose(obs_line, obs_ref, atol=2e-3)
    rng = np.random.default_rng(2)
    for _ in range(20):
        a = rng.uniform(-0.3, 0.3, 5).astype(np.float32)
        a[2] = abs(a[2])     # (keep clear of the machine: contacts are not exactly invariant)
        obs_ref, *_ = ref.step(a)
        obs_line, *_ = view.step(a)
    np.testing.assert_allclose(obs_line, obs_ref, atol=5e-3)


def test_the_continue_option_sets_the_camera_errors():
    _, _, env, view = line_door()
    view.reset(command="open", handle_bias=(0.01, 0.0, 0.0), bar_yaw_bias=0.0)
    np.testing.assert_allclose(env._bias, [0.0, -0.01, 0.0], atol=1e-12)   # turned to -90 deg
    assert env._yaw_bias == 0.0


def test_the_base_moves_the_reported_target_and_nothing_else():
    """The adapter's IK, box and lead work relative to the base: the same actions give the same
    joint targets wherever the base is, and target_pos moves with the base exactly."""
    a, b = DoorEnv(), DoorEnv()
    a.reset(seed=7)
    b.reset(seed=7)
    b.adapter.set_base((-0.6, 0.25))
    a.adapter.reset_to_measured(a.data)
    b.adapter.reset_to_measured(b.data)
    rng = np.random.default_rng(1)
    for _ in range(30):
        act = rng.uniform(-1, 1, 4)
        np.testing.assert_array_equal(a.adapter.apply(a.data, act), b.adapter.apply(b.data, act))
        np.testing.assert_allclose(b.adapter.target_pos - a.adapter.target_pos,
                                   [-0.6, 0.25, 0.0], atol=1e-12)
    saved = b.adapter.target_pos.copy()       # assignment (standup's save/restore) round-trips
    b.adapter.target_pos = saved
    np.testing.assert_allclose(b.adapter.target_pos, saved, atol=1e-12)
    np.testing.assert_allclose(b.adapter._target, a.adapter._target, atol=1e-12)
