"""v4.0 T5: the door skill inside the line's full tending cycle (docs/v4_targets.md, T5).

One trial, on one machine of a door set rebuilt as the line's M1 (its layout, its place on the
bench and its turn, the camera's errors of that episode): the robot starts at rail stop 0 in its
carry posture; a raw part waits on the bench at M1's stop. It drives to M1, **opens the lid**,
picks the part, loads it into the pocket, **closes the lid**. Success = the machine is running
at the end (part seated, lid latched: the machine starts its cycle).

Only the door changes between door actors; driving, picking and loading are the same scripted
code (control/scripted_tending.py) with the same camera estimate of the machine (pocket, turn,
handle; evaluation/tending.camera_belief), drawn from the same seed for every actor:
- `run:<path>[@final]`: a trained door policy, through DoorEnv / DoorV2Env on the line's scene,
  seen from M1's station (supervisor.stations.StationView), from the carry posture at M1;
- `feel_v2`, `feel`, `camera`, `nominal`, `expert`: the door scripts of control/door_baselines.py
  (and the privileged expert), through the same env and view;
- `line_camera`: the line's own camera-aware door move (ScriptedTending.move_lid), as in the
  Phase 1 sweep (evaluation/tending.py): it opens to the believed travel and closes past shut.
A door actor gets the door env's horizon (300 steps, 15 s) for each move; after it, the hand
goes back to the carry posture by the line's scripted move.

**Hand-over** (`handover`): `"carry"` (T5 as first run) starts each door move from the line's
carry posture at M1, the hand about 31 cm above the handle; `"door_start"` (T5b) first lowers
the hand straight down to the door skill's trained start height (DoorStartConfig.carry_height,
TCP 0.32 m) over the believed pocket, where every door training episode began (about 8 cm above
the handle). It applies to every door actor; `line_camera` keeps its own move.
"""

from __future__ import annotations

from dataclasses import replace

import mujoco
import numpy as np

from polypanda.control.scripted_tending import ScriptedTending
from polypanda.environments.grasp import GRIPPER_OPEN
from polypanda.evaluation.tending import NOMINAL_SPOT, PART_ANGLE, camera_belief
from polypanda.simulation.line import Line, LineConfig
from polypanda.simulation.machine import RUNNING, MachineConfig
from polypanda.simulation.machine_variation import needed_opening, variant
from polypanda.simulation.workcell import _yaw_quat
from polypanda.supervisor.stations import StationView

NOISE_SEED = 7_000_000          # scripts/evaluate_door.py: an episode's sensor noise seed
SCRIPTS = ("nominal", "camera", "feel", "feel_v2", "expert")
LINE_SCRIPTS = ("line_camera",)
HANDOVERS = ("carry", "door_start")


def door_start_z() -> float:
    from polypanda.environments.door import DoorStartConfig
    return DoorStartConfig().carry_height


def lower_to_door_start(ctl: ScriptedTending) -> None:
    """The T5b hand-over: straight down to the door skill's trained start height over the
    believed pocket, keeping the carry yaw."""
    b = ctl.beliefs["M1"]
    ctl.vertical(b.origin[:2], door_start_z(), ctl.carry_yaw(ctl.angle_of(b.origin[:2])))


def machine_from_episode(ep: dict, cycle_s: float = 60.0):
    """A door-set episode's machine as the line's M1: its layout, and its offset (given in the
    station frame) and turn about the pocket."""
    cfg = variant(MachineConfig(), slide=ep["slide"], travel=ep["travel"],
                  lid_friction=ep["friction"], lid_damping=ep["damping"], lid_spring=ep["spring"],
                  handle_offset=ep["handle_offset"], handle_shift=ep["handle_shift"],
                  handle_height=ep["handle_height"], latch_speed=ep["latch_speed"])
    a = np.radians(NOMINAL_SPOT.angle)
    c, s = np.cos(a), np.sin(a)
    ox, oy = ep["offset"]
    offset = (c * ox - s * oy, s * ox + c * oy)           # station frame -> world
    return replace(NOMINAL_SPOT, offset=offset, turn=float(ep["turn"]),
                   config=replace(cfg, cycle_s=cycle_s))


def _door_actor(actor: str, line: Line):
    """(env, act(obs), reset()) for a door actor on the line's M1."""
    from polypanda.environments.door import DoorEnv
    if actor.startswith("run:"):
        from polypanda.evaluation.policies import load_run_policy, run_env_cls
        env = run_env_cls(actor[4:])(scene=(line.model, line.data), machine=line.machines["M1"])
        policy = load_run_policy(actor[4:])
        return env, policy, (lambda: None), None
    from polypanda.control.door_baselines import BaselineDoor
    from polypanda.control.scripted_door import DoorExpert
    env = DoorEnv(scene=(line.model, line.data), machine=line.machines["M1"])
    script = DoorExpert(env) if actor == "expert" else BaselineDoor(env, actor)
    return env, (lambda obs: script.act()), script.reset, script


def _door_move(ctl: ScriptedTending, door, command: str, seed: int, ep: dict) -> dict:
    """One door move by a door actor from the carry posture at M1, then back to carry."""
    env, act, reset, _ = door
    view = StationView(env, np.radians(NOMINAL_SPOT.angle), origin=ctl.base_xy())
    obs, _ = env.reset(seed=seed, options={"continue": True, "command": command,
                                           "handle_bias": ep["handle_bias"],
                                           "bar_yaw_bias": ep["bar_yaw_bias"]})
    obs = view.observe(obs)
    reset()
    info, steps = {}, 0
    while steps < env.config.horizon:
        obs, _, terminated, truncated, info = view.step(act(obs))
        steps += 1
        if terminated or truncated:
            break
    ctl.q_cmd = env.adapter.q_cmd.copy()       # the line takes over from where the hand is
    ctl.grip = GRIPPER_OPEN
    ctl.to_carry()
    ctl.step(5)
    st = info.get("stats", {})
    return {"success": bool(info.get("is_success", False)), "failure": info.get("failure"),
            "steps": steps, "peak_force": round(float(st.get("peak_force", 0.0)), 2),
            "slams": int(st.get("slams", 0))}


def run_cycle(actor: str, index: int, ep: dict, handover: str = "carry") -> dict:
    """One tending cycle on door-set episode `ep` (its index `index`) with door actor `actor`,
    the door handed over as `handover` says."""
    if handover not in HANDOVERS:
        raise ValueError(f"handover must be one of {HANDOVERS}, not {handover!r}")
    seed = NOISE_SEED + index
    spot = machine_from_episode(ep)
    line = Line.build_line(LineConfig(machines=(spot,)))
    m, d = line.model, line.data
    m1 = line.machines["M1"]
    belief = camera_belief(line, np.random.default_rng(seed))   # the same for every actor
    ctl = ScriptedTending(line, beliefs={"M1": belief})
    cube = line.cubes[0]
    xy = line.line.station_xy(NOMINAL_SPOT.stop, PART_ANGLE)
    d.qpos[cube.qpos:cube.qpos + 7] = [*xy, cube.half_size[2],
                                       *_yaw_quat(np.radians(PART_ANGLE))]
    mujoco.mj_forward(m, d)
    row = {"index": index, "actor": actor, "handover": handover, "slide": ep["slide"],
           "spring": ep["spring"] > 0, "open": None, "close": None}
    door = None if actor in LINE_SCRIPTS else _door_actor(actor, line)
    opened = picked = seated = False
    error = None
    t0 = float(d.time)
    try:
        ctl.go_to("M1")
        if door is None:
            ctl.move_lid("M1", opening=True)
        else:
            if handover == "door_start":
                lower_to_door_start(ctl)
            row["open"] = _door_move(ctl, door, "open", seed, ep)
        opened = m1.lid
        ctl.swing_to(np.radians(PART_ANGLE))
        ctl.pick(cube)
        picked = line.attached(cube)
        ctl.swing_to(np.radians(NOMINAL_SPOT.angle))
        if picked:
            ctl.load(cube, "M1")
        ctl.step(10)
        seated = m1.in_pocket(cube)[1]
        if door is None:
            ctl.move_lid("M1", opening=False)
        else:
            if handover == "door_start":
                lower_to_door_start(ctl)
            row["close"] = _door_move(ctl, door, "close", seed + 1, ep)
        ctl.step(5)
    except Exception as e:  # noqa: BLE001 (a crashed trial is a failed trial, recorded)
        error = repr(e)[:200]
        opened = m1.lid if opened is False else opened
    running = m1.state == RUNNING
    if running:
        failure = None
    elif error:
        failure = "error"
    elif opened < needed_opening(m1.cfg) - 0.01 and not seated:
        failure = "lid_not_opened"
    elif not picked:
        failure = "pick"
    elif not seated:
        failure = "not_seated"
    elif not m1.latched:
        failure = "not_latched"
    else:
        failure = m1.state
    row |= {"success": running, "failure": failure, "lid_opened": round(float(opened), 4),
            "picked": bool(picked), "seated": bool(seated), "latched": bool(m1.latched),
            "state": m1.state, "ik_failures": ctl.ik_failures, "error": error,
            "cycle_s": round(float(d.time) - t0, 2)}
    return row
