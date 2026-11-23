"""v4.0 Phase 1: how far can a machine differ from the one a script was written for?

The variation sweep (docs/v4_plan.md, section 4, Phase 1): one machine property is varied at a
time, at levels further and further from the nominal machine, and both scripted baselines run the
door-and-load job on the same variants (paired by trial):

- **nominal**: believes the machine is exactly as programmed (nominal layout, nominal place);
- **camera-aware**: measures where the machine stands, its turn, and where its handle and pocket
  are, with the camera's error (default 1 mm per axis and 0.5 deg, from docs/pose_error.md, where
  a station camera's error was under about 1 mm at the 95th percentile), and still assumes the
  nominal slide direction, travel and latch.

One trial: the robot starts at rail stop 0 in its carry posture; a raw part stands on the bench
at the machine's stop (station angle -30 deg); the script drives to the machine, opens the lid,
picks the part, loads it into the pocket and closes the lid. **Success = the machine is running**
(part seated, lid latched) at the end. Each stage's outcome is recorded, with the arm's contacts
with the machine's body (walls, roof, pocket) and IK failures.

No training happens here; this measures the scripts before anything is learned.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import mujoco
import numpy as np

from polypanda.control.scripted_tending import Belief, ScriptedTending
from polypanda.simulation.line import Line, LineConfig, MachineSpot
from polypanda.simulation.machine import RUNNING, MachineConfig
from polypanda.simulation.workcell import _yaw_quat

NOMINAL_SPOT = MachineSpot("M1", -0.6, -90.0)
PART_ANGLE = -30.0                 # deg: where the raw part waits at the machine's stop
CAMERA_XY, CAMERA_Z, CAMERA_YAW = 0.001, 0.001, np.radians(0.5)

# property -> levels, in increasing distance from the nominal machine (the first is nominal)
LEVELS: dict[str, list] = {
    "position": [0.0, 0.01, 0.02, 0.03, 0.05, 0.08],          # m, random direction on the bench
    "turn": [0.0, 2.0, 5.0, 10.0, 15.0, 25.0],                 # deg, random sign
    "handle_along_bar": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05],   # m, random sign
    "handle_along_slide": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05],  # m, random sign
    "handle_height": [0.0, 0.005, 0.01, 0.015, 0.025, 0.04],   # m from 0.035; random sign, >= 0.02
    "slide": ["right", "left", "out"],                         # which way the lid opens
    "friction": [0.4, 2.0, 5.0, 10.0, 20.0, 40.0],             # N, the lid's slide friction
    "latch": [0.02, 0.04, 0.06, 0.1, 0.15, 0.25],              # m/s closing speed to latch
}


@dataclass(frozen=True)
class Variant:
    prop: str
    level: int
    value: float | str
    offset: tuple[float, float] = (0.0, 0.0)
    turn: float = 0.0
    config: MachineConfig = MachineConfig()


def make_variant(prop: str, level: int, rng: np.random.Generator,
                 base: MachineConfig | None = None) -> Variant:
    """The machine for one trial: the nominal one with `prop` set to its `level`-th value
    (a random direction or sign where the property has one, drawn from `rng`)."""
    base = base or MachineConfig()
    value = LEVELS[prop][level]
    sign = 1.0 if rng.random() < 0.5 else -1.0
    if prop == "position":
        a = rng.uniform(0.0, 2 * np.pi)
        return Variant(prop, level, value, offset=(value * np.cos(a), value * np.sin(a)),
                       config=base)
    if prop == "turn":
        return Variant(prop, level, value, turn=sign * value, config=base)
    if prop == "handle_along_bar":
        return Variant(prop, level, value, config=replace(base, handle_offset=sign * value))
    if prop == "handle_along_slide":
        return Variant(prop, level, value, config=replace(base, handle_shift=sign * value))
    if prop == "handle_height":
        h = base.handle_height + sign * value
        if h < 0.02:
            h = base.handle_height + value
        return Variant(prop, level, value, config=replace(base, handle_height=h))
    if prop == "slide":
        # enough travel to clear the whole opening (and the lid's overlap) with the script's
        # 5 mm margin: 0.24 + 0.015 + 0.005 = 0.26 outward; 0.10 + 0.015 + 0.005 = 0.12 sideways
        travel = 0.27 if value == "out" else base.travel
        return Variant(prop, level, value, config=replace(base, slide=value, travel=travel))
    if prop == "friction":
        return Variant(prop, level, value, config=replace(base, lid_friction=value))
    if prop == "latch":
        return Variant(prop, level, value, config=replace(base, latch_speed=value))
    raise KeyError(prop)


def build(variant: Variant, cycle_s: float = 60.0) -> Line:
    spot = replace(NOMINAL_SPOT, offset=variant.offset, turn=variant.turn,
                   config=replace(variant.config, cycle_s=cycle_s))
    return Line.build_line(LineConfig(machines=(spot,)))


def nominal_belief(line: Line) -> Belief:
    """The machine as programmed: the nominal layout at the nominal place."""
    return Belief(origin=np.array([*line.line.station_xy(NOMINAL_SPOT.stop, NOMINAL_SPOT.angle),
                                   0.0]),
                  yaw=np.radians(NOMINAL_SPOT.angle), config=MachineConfig())


def camera_belief(line: Line, rng: np.random.Generator) -> Belief:
    """Measured by a camera with its error: where the machine stands, its turn, its handle (lid
    closed) and its pocket; the nominal slide direction, travel and latch, which it cannot see."""
    m = line.machines["M1"]
    R = line.data.xmat[m.body].reshape(3, 3)
    yaw = float(np.arctan2(R[1, 0], R[0, 0])) + rng.normal(0.0, CAMERA_YAW)
    noise = np.array([CAMERA_XY, CAMERA_XY, CAMERA_Z])
    origin = m.pocket_xyz() + rng.normal(0.0, noise)
    handle = m.handle_xyz() + rng.normal(0.0, noise)
    return Belief(origin=origin, yaw=yaw, config=MachineConfig(), handle=handle)


def run_trial(variant: Variant, script: str, seed: int) -> dict:
    """One door-and-load trial with the `script` ("nominal" or "camera") on `variant`."""
    rng = np.random.default_rng(seed)
    line = build(variant)
    m, d = line.model, line.data
    m1 = line.machines["M1"]
    belief = nominal_belief(line) if script == "nominal" else camera_belief(line, rng)
    ctl = ScriptedTending(line, beliefs={"M1": belief})
    arm = {b for b in range(m.nbody) if m.body(b).name.startswith(("link", "hand"))
           or "finger" in m.body(b).name}
    machine = {m1.body} | {m.body(f"M1_pocket{k}").id for k in range(4)}
    hits = []

    def watch():
        line.update_machines()
        for i in range(d.ncon):
            c = d.contact[i]
            pair = {int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])}
            if pair & arm and pair & machine:
                hits.append(1)

    ctl.on_step = watch
    cube = line.cubes[0]
    spot = line.line.station_xy(NOMINAL_SPOT.stop, PART_ANGLE)
    d.qpos[cube.qpos:cube.qpos + 7] = [*spot, cube.half_size[2],
                                       *_yaw_quat(np.radians(PART_ANGLE))]
    mujoco.mj_forward(m, d)
    row = {**{k: v for k, v in asdict(variant).items() if k != "config"},
           "script": script, "seed": seed}
    try:
        ctl.go_to("M1")
        ctl.move_lid("M1", opening=True)
        opened = m1.lid
        ctl.swing_to(np.radians(PART_ANGLE))
        ctl.pick(cube)
        picked = line.attached(cube)
        ctl.swing_to(np.radians(NOMINAL_SPOT.angle))
        if picked:
            ctl.load(cube, "M1")
        ctl.step(10)
        seated = m1.in_pocket(cube)[1]
        ctl.move_lid("M1", opening=False)
        ctl.step(5)
        error = None
    except Exception as e:  # noqa: BLE001 (a crashed trial is a failed trial, recorded)
        opened, picked, seated, error = m1.lid, line.attached(cube), False, repr(e)[:200]
    running = m1.state == RUNNING
    cfg = m1.cfg
    along = cfg.opening[0] if cfg.slide == "out" else cfg.opening[1]
    needed = along + cfg.lid_overlap        # lid travel that uncovers the whole opening
    if running:
        failure = None
    elif error:
        failure = "error"
    elif opened < needed - 0.01 and not seated:
        failure = "lid_not_opened"
    elif not picked:
        failure = "pick"
    elif not seated:
        failure = "not_seated"
    elif not m1.latched:
        failure = "not_latched"
    else:
        failure = m1.state
    row |= {"success": running, "failure": failure, "lid_opened": opened, "picked": picked,
            "seated": seated, "latched": m1.latched, "state": m1.state,
            "machine_contacts": len(hits), "ik_failures": ctl.ik_failures, "error": error}
    return row
