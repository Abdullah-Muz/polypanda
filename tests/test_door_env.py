"""v4.0: the door skill's env (environments/door.py), its runtime machine variants
(simulation/machine_variation.py) and its privileged expert (control/scripted_door.py)."""

from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from polypanda.control.ik import top_down_quat
from polypanda.control.scripted_door import DoorExpert, run_expert
from polypanda.environments.door import (
    OBS_DIM,
    OBS_LAYOUT,
    OBS_SCHEMA,
    OBS_SLICES,
    DoorConfig,
    DoorEnv,
    DoorSceneConfig,
)
from polypanda.environments.tasks import task_spec
from polypanda.simulation.assets import panda_spec
from polypanda.simulation.machine import MachineConfig, add_machine
from polypanda.simulation.machine_variation import MachineLayout, needed_opening, variant
from polypanda.simulation.panda import tcp_position
from polypanda.supervisor import stations
from polypanda.supervisor.stations import StationView
from polypanda.training.config import TrainConfig


@pytest.fixture(scope="module")
def env():
    return DoorEnv()


def episode(env, command, slide, seed=0, **changes):
    """A drawn episode with its command and slide direction set (travel drawn to match)."""
    ep = env.draw_episode(np.random.default_rng(seed), command)
    v = env.config.variation
    out = slide == "out"
    ep |= {"slide": slide,
           "travel": float(np.mean(v.travel_out if out else v.travel_side)),
           "handle_offset": 0.0, "handle_shift": 0.0, "spring": 0.0}
    if command == "close":
        needed = needed_opening(replace(MachineConfig(), slide=slide))
        ep["lid"] = needed + 0.003
    return ep | changes


# --- spaces and schema ------------------------------------------------------------------------
def test_spaces_schema_and_registry(env):
    assert OBS_SCHEMA == f"door-v1-{OBS_DIM}d" and OBS_DIM == 49
    assert len({name for name, _ in OBS_LAYOUT}) == len(OBS_LAYOUT)
    assert env.observation_space.shape == (OBS_DIM,)
    assert env.action_space.shape == (5,)
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    spec = task_spec("door")
    assert spec.env_cls is DoorEnv and spec.config_cls is DoorConfig
    assert spec.obs_schema == OBS_SCHEMA
    assert stations.LAYOUTS[DoorEnv] is OBS_LAYOUT
    obs, *_ = env.step(env.action_space.sample())
    assert obs.shape == (OBS_DIM,)


def test_the_configs_load_and_v1_is_the_default_task():
    root = Path(__file__).parents[1]
    assert DoorConfig.from_yaml(root / "configs/door_v1.yaml") == DoorConfig()
    train = TrainConfig.from_yaml(root / "configs/door_v1_ppo.yaml")
    assert train.task == "door" and train.env_config == "configs/door_v1.yaml"
    assert train.n_envs == 8 and train.torch_threads == 1 and train.normalize_obs


def test_station_frame_classification():
    rotated = (set(stations.POINT_FIELDS) | set(stations.VECTOR_FIELDS)
               | set(stations.ROT6_FIELDS) | set(stations.PLANAR_FIELDS)
               | set(stations.JOINT_FIELDS))
    names = {name for name, _ in OBS_LAYOUT}
    assert {"handle_rel", "touch_travel", "force"} <= set(stations.VECTOR_FIELDS)
    assert "tcp" in stations.POINT_FIELDS and "tcp_rot6" in stations.ROT6_FIELDS
    # scalars: unchanged by a station turn
    for name in ("dq", "finger_gap", "gripper_closed", "contacts", "bar_align", "door_io",
                 "door_command"):
        assert name in names and name not in rotated


def test_the_observation_reads_the_command_and_the_machine_signals(env):
    obs, _ = env.reset(seed=1, options={"command": "open"})
    np.testing.assert_array_equal(obs[OBS_SLICES["door_command"]], [1.0, 0.0])
    np.testing.assert_array_equal(obs[OBS_SLICES["door_io"]], [1.0, 0.0])   # shut, unlatched
    obs, _ = env.reset(seed=1, options={"command": "close"})
    np.testing.assert_array_equal(obs[OBS_SLICES["door_command"]], [0.0, 1.0])
    np.testing.assert_array_equal(obs[OBS_SLICES["door_io"]], [0.0, 0.0])   # open
    # the camera's handle estimate is the truth within its error (1 mm bias, 0.3 mm noise)
    rel = obs[OBS_SLICES["handle_rel"]]
    true = env.handle_xyz() - tcp_position(env.data, env.arm)
    assert np.abs(rel - true).max() < 0.006


# --- runtime machine variants -----------------------------------------------------------------
def _compiled(cfg: MachineConfig):
    spec = panda_spec()
    add_machine(spec, "M", (0.5, 0.0), 0.0, cfg)
    return spec.compile()


@pytest.mark.parametrize("slide", ["right", "left", "out"])
def test_a_runtime_variant_matches_the_compiled_machine(slide):
    base = MachineConfig()
    cfg = variant(base, slide=slide, travel=0.27 if slide == "out" else 0.13, lid_friction=3.0,
                  lid_damping=7.0, lid_spring=2.0, handle_offset=0.01, handle_shift=-0.015,
                  handle_height=0.045, latch_speed=0.07)
    ref = _compiled(cfg)
    model = _compiled(base)
    layout = MachineLayout(model, "M", base)
    layout.apply(model, cfg)
    j, dof = ref.joint("M_lid").id, int(ref.joint("M_lid").dofadr[0])
    np.testing.assert_allclose(model.jnt_axis[j], ref.jnt_axis[j], atol=1e-12)
    np.testing.assert_allclose(model.jnt_range[j], ref.jnt_range[j], atol=1e-12)
    assert model.jnt_stiffness[j] == pytest.approx(ref.jnt_stiffness[j])
    assert model.dof_frictionloss[dof] == pytest.approx(ref.dof_frictionloss[dof])
    assert model.dof_damping[dof] == pytest.approx(ref.dof_damping[dof])
    for name in ("M_handle", "M_handle_post0", "M_handle_post1"):
        g = ref.geom(name).id
        for f in ("geom_pos", "geom_size", "geom_rbound", "geom_aabb"):
            np.testing.assert_allclose(getattr(model, f)[g], getattr(ref, f)[g], atol=1e-9,
                                       err_msg=f"{name} {f}")
        q, q_ref = model.geom_quat[g], ref.geom_quat[g]   # (q and -q are the same turn)
        assert min(np.abs(q - q_ref).max(), np.abs(q + q_ref).max()) < 1e-9, name
    s = ref.site("M_handle").id
    np.testing.assert_allclose(model.site_pos[s], ref.site_pos[s], atol=1e-9)


def test_the_lid_bvh_is_rebuilt_as_the_compiler_builds_it():
    model = _compiled(MachineConfig())
    layout = MachineLayout(model, "M", MachineConfig())
    b = layout.lid_body
    nodes = slice(model.body_bvhadr[b], model.body_bvhadr[b] + model.body_bvhnum[b])
    compiled = model.bvh_aabb[nodes].copy()
    layout.apply(model, MachineConfig())
    np.testing.assert_allclose(model.bvh_aabb[nodes], compiled, atol=1e-9)


@pytest.mark.parametrize("slide", ["left", "out"])
def test_a_moved_handle_is_inside_the_lid_bvh_and_collides(slide):
    """A handle far from its compiled place: its box is inside its BVH leaf, and a finger-like
    probe touching its far side finds the contact (the midphase used the stale tree before)."""
    model = _compiled(MachineConfig())
    layout = MachineLayout(model, "M", MachineConfig())
    cfg = variant(MachineConfig(), slide=slide, travel=0.27 if slide == "out" else 0.13,
                  handle_shift=-0.02, handle_offset=0.01, handle_height=0.05)
    layout.apply(model, cfg)
    b, adr = layout.lid_body, model.body_bvhadr[layout.lid_body]
    leaf = [k for k in range(adr, adr + model.body_bvhnum[b])
            if model.bvh_nodeid[k] == layout.handle][0]
    R_i = np.zeros(9)
    mujoco.mju_quat2Mat(R_i, model.body_iquat[b])
    centre = (model.geom_pos[layout.handle] - model.body_ipos[b]) @ R_i.reshape(3, 3)
    box = model.bvh_aabb[leaf]
    assert np.all(np.abs(centre - box[:3]) <= box[3:] - 0.007)
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    bar = d.geom_xpos[layout.handle].copy()
    # a free sphere at the bar's new centre (outside its compiled box) touches it
    spec = panda_spec()
    add_machine(spec, "M", (0.5, 0.0), 0.0, MachineConfig())
    probe = spec.worldbody.add_body(name="probe", pos=bar.tolist())
    probe.add_freejoint()
    probe.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.009], pos=[0.0, 0.0, 0.0])
    m2 = spec.compile()
    MachineLayout(m2, "M", MachineConfig()).apply(m2, cfg)
    d2 = mujoco.MjData(m2)
    mujoco.mj_forward(m2, d2)
    handle = m2.geom("M_handle").id
    assert any(handle in (d2.contact[i].geom1, d2.contact[i].geom2) for i in range(d2.ncon))


def test_a_variant_cannot_change_what_needs_a_recompile():
    model = _compiled(MachineConfig())
    layout = MachineLayout(model, "M", MachineConfig())
    with pytest.raises(ValueError):
        layout.apply(model, replace(MachineConfig(), width=0.4))
    with pytest.raises(ValueError):
        variant(MachineConfig(), opening=(0.2, 0.1))


def test_randomization_changes_the_model_every_episode(env):
    m = env.model
    lay = env.layout
    seen = []
    for seed in range(12):
        _, info = env.reset(seed=seed)
        ep = info["episode"]
        cfg = env.machine.cfg
        # the model is the drawn machine
        np.testing.assert_allclose(m.jnt_axis[lay.joint], cfg.slide_axis)
        assert m.jnt_range[lay.joint][1] == pytest.approx(ep["travel"])
        assert m.dof_frictionloss[lay.dof] == pytest.approx(ep["friction"])
        assert m.dof_damping[lay.dof] == pytest.approx(ep["damping"])
        assert cfg.latch_speed == pytest.approx(ep["latch_speed"])
        bar_z = env.handle_xyz()[2]
        assert bar_z == pytest.approx(cfg.lid_top + ep["handle_height"], abs=1e-9)
        R = env.data.xmat[env.machine.body].reshape(3, 3)
        assert np.degrees(np.arctan2(R[1, 0], R[0, 0])) == pytest.approx(ep["turn"])
        np.testing.assert_allclose(env.data.xpos[env.machine.body][:2],
                                   [0.5 + ep["offset"][0], ep["offset"][1]], atol=1e-9)
        seen.append((ep["slide"], m.dof_frictionloss[lay.dof], m.body_pos[lay.body][0],
                     m.geom_pos[lay.handle][2], env.machine.cfg.latch_speed))
    slides, friction, x, height, latch = zip(*seen, strict=True)
    assert set(slides) == {"right", "left", "out"}
    for values in (friction, x, height, latch):
        assert len(set(np.round(values, 9))) == len(values)   # all different


def test_the_lid_slides_the_drawn_way(env):
    """A pull along the drawn slide moves the lid; the joint really is re-axed."""
    for slide in ("right", "left", "out"):
        env.reset(options={"episode": episode(env, "open", slide, friction=0.2, damping=1.0)})
        d = env.data
        dof = env.machine.lid_dof
        body = env.lid_body
        start = d.xpos[body].copy()
        for _ in range(200):
            d.qfrc_applied[dof] = 5.0     # (a generalized force along the joint)
            mujoco.mj_step(env.model, d)
        d.qfrc_applied[dof] = 0.0
        moved = d.xpos[body] - start
        assert env.machine.lid > 0.02
        np.testing.assert_allclose(moved / np.linalg.norm(moved), env.slide_axis(), atol=1e-6)


# --- reset determinism ------------------------------------------------------------------------
def test_reset_is_deterministic_with_a_seed():
    a, b = DoorEnv(), DoorEnv()
    obs_a, info_a = a.reset(seed=7)
    obs_b, info_b = b.reset(seed=7)
    np.testing.assert_array_equal(obs_a, obs_b)
    assert info_a["episode"] == info_b["episode"]
    rng = np.random.default_rng(0)
    for _ in range(15):
        act = rng.uniform(-1, 1, 5).astype(np.float32)
        obs_a, r_a, *_ = a.step(act)
        obs_b, r_b, *_ = b.step(act)
    np.testing.assert_array_equal(obs_a, obs_b)
    assert r_a == r_b
    obs_c, info_c = a.reset(seed=8)
    assert info_c["episode"] != info_a["episode"]


# --- success predicates on staged states ------------------------------------------------------
def stage(env, command, lid, latched=False, above=0.05, closed=False, slide="right"):
    """Reset to a machine, then put the lid at `lid` (latched or not) and the open (or closed)
    hand `above` m over the bar, still."""
    env.reset(options={"episode": episode(env, command, slide)})
    m, d = env.model, env.data
    d.qpos[env.machine.lid_qpos] = lid
    mujoco.mj_forward(m, d)
    target = env.handle_xyz() + [0.0, 0.0, above]
    q, ok = env._ik.solve(d.qpos[env.arm.arm_qpos], target, top_down_quat(0.0))
    assert ok
    d.qpos[env.arm.arm_qpos] = q
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    env.machine.reset()
    if latched:
        d.eq_active[env.machine.latch_eq] = 1
    env.adapter.reset_to_measured(d)
    d.ctrl[env.arm.gripper_actuator] = 0.0 if closed else 255.0
    env._begin(command, (0.0, 0.0, 0.0), 0.0)


def hold(env, steps, gripper=-1.0):
    info, terminated = {}, False
    for _ in range(steps):
        _, _, terminated, _, info = env.step(np.array([0, 0, 0, 0, gripper], np.float32))
        if terminated:
            break
    return info, terminated


def test_open_succeeds_with_the_lid_open_enough_and_the_hand_clear(env):
    needed = needed_opening(MachineConfig())
    stage(env, "open", needed + 0.002)
    info, terminated = hold(env, env.config.success.hold_steps)
    assert terminated and info["is_success"] and info["failure"] is None
    assert info["reward_terms"]["success"] == env.config.reward.success


def test_open_fails_short_of_the_needed_opening(env):
    stage(env, "open", needed_opening(MachineConfig()) - 0.01)
    info, terminated = hold(env, 15)
    assert not terminated and not info["is_success"] and not info["goal_reached"]


def test_success_needs_the_hand_clear_above_the_bar_not_the_fingers_closed(env):
    needed = needed_opening(MachineConfig())
    stage(env, "open", needed + 0.002, above=0.015)        # too low over the bar
    info, terminated = hold(env, 15)
    assert not terminated and info["goal_reached"] and info["failure"] == "not_released"
    # a hook never closes the fingers, and fingers shut on nothing above the bar hold nothing:
    # success asks only that the hand is clear of the bar and above it
    stage(env, "open", needed + 0.002, closed=True)
    info, terminated = hold(env, env.config.success.hold_steps, gripper=1.0)
    assert terminated and info["is_success"] and info["clear_of_bar"]


def test_close_succeeds_only_latched(env):
    stage(env, "close", 0.0, latched=True)
    info, terminated = hold(env, env.config.success.hold_steps)
    assert terminated and info["is_success"]
    stage(env, "close", 0.0, latched=False)                # shut, but not latched
    info, terminated = hold(env, 15)
    assert not terminated and not info["latched"] and not info["goal_reached"]
    obs = env._observation()
    np.testing.assert_array_equal(obs[OBS_SLICES["door_io"]], [1.0, 0.0])


# --- the expert -------------------------------------------------------------------------------
@pytest.mark.parametrize("command", ["open", "close"])
@pytest.mark.parametrize("slide", ["right", "left", "out"])
def test_the_expert_opens_and_closes_every_slide_direction(env, command, slide):
    info = run_expert(env, options={"episode": episode(env, command, slide, seed=3)})
    assert info["is_success"], info["failure"]
    assert info["stats"]["collision_steps"] == 0


def springy(env, command, slide, spring=30.0, seed=0):
    """An episode on a springy lid (with its soft-close damper), a close starting in the detent."""
    ep = episode(env, command, slide, seed=seed, spring=spring)
    ep["damping"] = max(ep["damping"],
                        spring * ep["travel"] / env.config.variation.soft_close_speed)
    if command == "close":
        ep["lid"] = ep["travel"] - 0.002
    return ep


def test_a_springy_lid_shuts_itself_unless_left_in_its_detent(env):
    hold = np.zeros(5, dtype=np.float32)
    hold[4] = -1.0                                     # the hand stays put, fingers open
    for where in ("needed", "detent"):
        env.reset(options={"episode": springy(env, "close", "out")})
        m = env.machine
        q0 = env.needed + 0.002 if where == "needed" else m.cfg.travel - 0.002
        env.data.qpos[m.lid_qpos] = q0
        mujoco.mj_forward(env.model, env.data)
        m.reset()
        for _ in range(40):                            # 2 s
            env.step(hold)
        if where == "needed":                          # creeping shut (its damper slows it)
            assert m.lid < q0 / 2 and not m.held_open
        else:
            assert m.lid > m.cfg.travel - 0.002 and m.held_open


def test_open_on_a_springy_lid_needs_its_detent(env):
    env.reset(options={"episode": springy(env, "open", "out")})
    m = env.machine
    for q, reached in ((env.needed + 0.002, False), (m.cfg.travel - 0.001, True)):
        env.data.qpos[m.lid_qpos] = q
        mujoco.mj_forward(env.model, env.data)
        assert env.goal_reached() is reached
    plain = episode(env, "open", "out")
    env.reset(options={"episode": plain})
    env.data.qpos[env.machine.lid_qpos] = env.needed + 0.002
    mujoco.mj_forward(env.model, env.data)
    assert env.goal_reached()                          # no spring: it stays where it is left


def test_springy_lids_are_drawn_with_their_damper(env):
    rng = np.random.default_rng(11)
    eps = [env.draw_episode(rng) for _ in range(200)]
    v = env.config.variation
    k = [e for e in eps if e["spring"] > 0]
    assert 0.35 < len(k) / len(eps) < 0.65
    for e in k:
        assert v.spring[0] <= e["spring"] <= v.spring[1]
        assert e["spring"] * e["travel"] / e["damping"] <= v.soft_close_speed + 1e-9
        if e["command"] == "close":
            assert e["lid"] == pytest.approx(e["travel"] - 0.002)


@pytest.mark.parametrize("command", ["open", "close"])
@pytest.mark.parametrize("slide", ["right", "left", "out"])
def test_the_expert_handles_springy_lids(env, command, slide):
    info = run_expert(env, options={"episode": springy(env, command, slide, seed=3)})
    assert info["is_success"], info["failure"]
    assert info["stats"]["slams"] == 0


def test_the_expert_succeeds_on_drawn_machines(env):
    wins = [run_expert(env, seed=s)["is_success"] for s in range(6)]
    assert sum(wins) == 6


def test_the_expert_hooks_the_bar_with_open_fingers(env):
    env.reset(options={"episode": episode(env, "open", "right", seed=5)})
    expert = DoorExpert(env)
    for _ in range(env.config.horizon):
        a = expert.act()
        assert a[4] < env.config.gripper.open_below        # never commands the fingers shut
        _, _, terminated, _, info = env.step(a)
        if terminated:
            break
    assert info["is_success"] and info["stats"]["touched"]
    assert not env.closed


def test_the_force_and_touch_travel_follow_the_pushed_lid(env):
    """While the expert pushes a stiff lid open, the felt force points against the slide and
    touch_travel follows the lid's motion."""
    env.reset(options={"episode": episode(env, "open", "out", friction=8.0, damping=8.0)})
    expert = DoorExpert(env)
    along, travel, moved = [], 0.0, 0.0
    for _ in range(200):
        obs, *_ = env.step(expert.act())
        if expert.phase == "push" and env.machine.lid > 0.01:
            axis = env.slide_axis()
            along.append(float(obs[OBS_SLICES["force"]] @ axis))
            travel = float(obs[OBS_SLICES["touch_travel"]] @ axis)
            moved = env.machine.lid
    assert len(along) > 20 and np.mean(along) < -5.0
    assert travel == pytest.approx(moved, abs=0.01)


# --- station frames ---------------------------------------------------------------------------
def test_station_view_at_minus_90_is_the_env_at_0():
    angle = np.radians(-90.0)
    ref = DoorEnv()
    rot = DoorEnv(replace(DoorConfig(), scene=DoorSceneConfig(station_angle=-90.0)))
    ref.reset(seed=11, options={"command": "close"})
    rot.reset(seed=11, options={"command": "close"})
    view = StationView(rot, angle)
    obs_ref, _ = ref.reset(options={"continue": True, "command": "close"})
    obs_rot, _ = view.reset(command="close")
    np.testing.assert_allclose(obs_rot, obs_ref, atol=1e-5)
    rng = np.random.default_rng(2)
    for _ in range(25):
        a = rng.uniform(-0.3, 0.3, 5).astype(np.float32)
        a[2] = abs(a[2])     # (keep clear of the machine: contacts are not exactly invariant)
        obs_ref, r_ref, *_ = ref.step(a)
        obs_rot, r_rot, *_ = view.step(a)
    np.testing.assert_allclose(obs_rot, obs_ref, atol=2e-3)
    assert r_rot == pytest.approx(r_ref, abs=1e-3)


def test_the_expert_works_through_a_station_view():
    rot = DoorEnv(replace(DoorConfig(), scene=DoorSceneConfig(station_angle=-90.0)))
    rot.reset(seed=4, options={"command": "open"})
    view = StationView(rot, np.radians(-90.0))
    view.reset(command="open")
    expert = DoorExpert(rot)
    for _ in range(rot.config.horizon):
        _, _, terminated, truncated, info = view.step(expert.act())
        if terminated or truncated:
            break
    assert info["is_success"], info["failure"]
