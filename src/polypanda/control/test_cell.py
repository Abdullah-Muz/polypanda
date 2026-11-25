"""v4.1: the scripted controller of the electronics test cell (docs/v4_1_plan.md, section 1;
simulation/test_cell.py has the cell).

Per module: pick it from the conveyor (after the barcode scan over the pick point), drive to a
tester, load it into the nest, close the tester (the test starts by itself: Machine's interlock),
and serve the other testers while it runs; when its test ends, open the tester, unload the
module, **present it to the vision camera on the way** (`present`: carried to the vision
station, the wrist turned so the module's front looks at the camera, a brief stop, on: no
set-down), and sort it: **PASS tray** if the test passed and the camera saw no defect, **FAIL
bin** otherwise. A module the scanner cannot read goes straight to the FAIL bin, untested.
A module the robot could not get tested goes to the **REWORK tray**, not judged ("not tested":
`rework_reason`): one that would not seat in the nest after its retries ("not_seated"), one
knocked out of its seat between the insertion and the test's start ("not_running"), one whose
test was aborted (lid opened or module unseated while running: "test_aborted") or that has no
result ("no_result"), or one that passed its test but whose front a finger covered at the camera
("not_inspected"; it does not happen with a module taken out of a keyed nest, which is held
across its rib's face). These measure the cell's handling, not the module: they are neither
passed nor rejected (an operator takes them for a retest). One stuck in the nest is cleared by
an operator (counted, not sorted). Every FAIL has its reason (ModuleRecord.reject_reason).

**Serving testers.** The robot has one gripper and opens doors with an empty hand, so a tester is
served in one visit: open it; if it holds a tested module, unload it, inspect and sort it; fetch
the next module from the conveyor; load it; close. The tester stands open, empty, while the robot
inspects, sorts and fetches (a dual gripper or a buffer nest would avoid that; not in v4.1).
**Which tester next:** of the testers that need the robot (test finished, or empty while modules
remain), the one that has waited longest: the one whose test finished first (empty testers count
from when they became empty; at the start of a shift A, then B, then C). If none needs it, the
robot waits at the tester that will finish first. Faulted testers (a test aborted: lid opened or
module unseated while running) are served the same way, the module going to the REWORK tray; a
tester whose door fails twice in a row is taken out of the shift (blocked).

**Actor slots, where learned skills plug in** (everything else is scripted in v4.1):
- the **door** (`door`, a `DoorActor`): opens and closes a lid or drawer tester by its handle.
  `EnvDoor` runs a v4.0 door actor through DoorEnv on the tester, from the hand-over `handover`
  (as evaluation/line_door: "door_start" lowers the hand to the door skill's trained start height
  over the believed nest first): the scripts "feel_v2", "feel", "camera", "nominal", the
  privileged "expert", or a trained policy "run:<path>[@final]" (**the learned door**). `LineDoor`
  is the line's own scripted lid move (ScriptedTending.move_lid with the camera's belief; lids
  only). A string names one of these (`make_door`). Pneumatic testers do not use the slot: the
  cell signals them (Machine.request_open / request_close, track B).
- the **insertion** (`insertion`, an `InsertionActor`): seats the held module in the tester's
  nest. The placeholder is `StraightDown` (ScriptedTending.load: the cell's closed-loop place,
  straight down where the camera puts the nest, the key turned to the slot); the straight-down and
  force-guided search scripts of control/scripted_insertion.py and **the learned insertion** go
  here. An insertion with the call `insert(ctl, part, xy, yaw)` (a `PoseInsertion`, as
  SearchInsertion on main) goes in through `NestInsertion`, which aims it (`aim`: the believed
  nest, the wrist turned so the key meets the slot). `make_insertion` names them.
The contracts are in the two Protocols' docstrings. The controller judges every result itself
(Machine.lid and latched for a door, Machine.in_pocket for a seat), measures the peak force on
the nest during an insertion, and retries: `door_attempts` per door move, `insert_attempts` per
module (the nest measured again each time).

**What the controller knows.** It never reads a module's hidden truth: only the tester's result
signal (TestCell.tester_result), the two cameras' noisy images (perception/inspection.py; a
module the scanner does not read is scanned once more before it is rejected), and its belief
about each tester: `nest_belief="camera"` (default) measures the tester with v4.0's camera error
(1 mm per axis, 0.5 deg) on arrival and before each insertion; "true" uses the truth (an oracle,
for tests and an upper bound). The EnvDoor actors do their own sensing, as in v4.0.

Everything is timed in simulated seconds and logged: the testers' states over time (`timeline`,
for a dashboard), the robot's activities (`activities`), and one record per module (`records`).
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol

import mujoco
import numpy as np

from polypanda.control.ik import closing_yaw, top_down_quat
from polypanda.control.scripted_cell import CUBE_HEIGHT, CellControllerConfig
from polypanda.control.scripted_tending import Belief, ScriptedTending, true_belief
from polypanda.environments.grasp import GRIPPER_OPEN
from polypanda.perception.inspection import NOISE, ImagingNoise, Inspection, inspect
from polypanda.simulation.machine import DONE, EMPTY, FAULT, RUNNING, rib_geom
from polypanda.simulation.machine_variation import needed_opening
from polypanda.simulation.scenes import CubeIndices, cube_yaw
from polypanda.simulation.test_cell import Station, TestCell

DOOR_SCRIPTS = ("feel_v2", "feel", "camera", "nominal", "expert")
LINE_DOOR = "line_camera"
HANDOVERS = ("carry", "door_start")
CAMERA_XY, CAMERA_Z, CAMERA_YAW = 0.001, 0.001, np.radians(0.5)   # as evaluation/tending.py


def camera_belief(cell: TestCell, name: str, rng: np.random.Generator) -> Belief:
    """A tester measured by a camera with v4.0's error (evaluation/tending.camera_belief, for any
    tester): where its nest is and how it is turned, and its handle (none on a pneumatic tester);
    the rest of its layout as programmed (the cell knows its testers' models)."""
    m = cell.machines[name]
    R = cell.data.xmat[m.body].reshape(3, 3)
    yaw = float(np.arctan2(R[1, 0], R[0, 0])) + rng.normal(0.0, CAMERA_YAW)
    noise = np.array([CAMERA_XY, CAMERA_XY, CAMERA_Z])
    origin = m.pocket_xyz() + rng.normal(0.0, noise)
    handle_noise = rng.normal(0.0, noise)            # drawn always: one stream for every kind
    cfg = cell.spots[name].config
    handle = None
    if cell.spots[name].kind != "pneumatic":
        # Belief.handle is the handle with the door shut: seen with the door open by m.lid
        # (along its slide, which the camera sees move), it is referred back to shut
        handle = m.handle_xyz() - R @ cfg.slide_axis * m.lid + handle_noise
    return Belief(origin=origin, yaw=yaw, config=cfg, handle=handle)


# --- the actor slots ------------------------------------------------------------------------------
class DoorActor(Protocol):
    """The door slot. `move(ctl, tester, command)` ("open" or "close") is called at the tester's
    rail stop, facing it (ctl.go_to), the hand **empty** in the carry posture, the tester measured
    (ctl.beliefs[tester]); it moves the lid or drawer by its handle and returns with the hand
    empty, open, back in the carry posture (ctl.q_cmd the arm's command). It may step the
    simulation through ctl (ctl.step runs the rest of the cell) or through its own env, as long as
    it calls `ctl.tick(exclude=tester)` each control step so the other testers and the conveyor run
    on. It returns a dict of whatever it wants recorded; the controller checks the outcome itself
    (open: Machine.lid at least the needed opening; close: Machine.latched)."""
    name: str

    def move(self, ctl: TestCellController, tester: str, command: str) -> dict: ...


class InsertionActor(Protocol):
    """The insertion slot. `insert(ctl, part, tester)` is called with the module held (welded in
    the closed gripper, as the cell's picks leave it), at the tester's rail stop, facing it, in the
    carry posture, the tester open and just measured (ctl.beliefs[tester]: the camera's estimate
    of the nest; nothing about the module's place in the hand). It seats the module in the nest,
    lets go (ctl.cell.detach(part), gripper open) and rises back to the carry height. Returns a
    dict of whatever it wants recorded. The controller then checks the seat
    (Machine.in_pocket), records the time and the peak contact force on the nest, and retries."""
    name: str

    def insert(self, ctl: TestCellController, part: CubeIndices, tester: str) -> dict: ...


class StraightDown:
    """The placeholder insertion: ScriptedTending.load (the cell's closed-loop place, straight
    down into the nest where it is believed to be, the key turned to the slot), then a short
    settle. (control/scripted_insertion.py's straight_down and SearchInsertion replace it.)"""
    name = "straight_down"

    def insert(self, ctl: TestCellController, part: CubeIndices, tester: str) -> dict:
        ctl.load(part, tester)
        ctl.step(10)
        return {}


class PoseInsertion(Protocol):
    """An insertion that is told where to go: `insert(ctl, part, xy, yaw)`, the believed nest's
    centre (world x, y) and the hand's yaw (rad) that brings the module's key to the believed
    slot, called with the hand over the nest at the carry height, the wrist already turned to
    `yaw`. It seats the part, lets go and rises clear. (The call of control/scripted_insertion.py's
    SearchInsertion on main; `NestInsertion` puts one in the slot.)"""

    def insert(self, ctl: TestCellController, part: CubeIndices, xy, yaw: float) -> dict: ...


def aim(ctl: TestCellController, part: CubeIndices, tester: str) -> tuple[np.ndarray, float]:
    """Where an insertion goes and how the hand is turned (ScriptedTending.load's first half):
    the believed nest's centre (ctl.beliefs[tester]) and the carry yaw turned so the held
    module's rib faces the believed slot; the wrist is turned there, over the nest at the carry
    height (a module held the wrong way is regrasped first: ScriptedTending.keyed_turn)."""
    b = ctl.beliefs[tester]
    xy = np.asarray(b.origin[:2], dtype=float).copy()
    carry = ctl.carry_yaw(ctl.angle_of(xy))
    if b.config.keyed and rib_geom(ctl.m, part) >= 0:
        turn = ctl.keyed_turn(part, b, xy, carry, tester)
    else:
        turn = (b.yaw - cube_yaw(ctl.d, part) + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return xy, carry + turn


class NestInsertion:
    """A PoseInsertion in the insertion slot: `aim` (the believed nest and the keyed yaw), the
    actor's insert, and a release if it returned still holding the part (the slot's contract)."""

    def __init__(self, inner: PoseInsertion, name: str | None = None) -> None:
        self.inner = inner
        self.name = name or getattr(inner, "name", None) or type(inner).__name__

    def insert(self, ctl: TestCellController, part: CubeIndices, tester: str) -> dict:
        xy, yaw = aim(ctl, part, tester)
        out = dict(self.inner.insert(ctl, part, xy, yaw) or {})
        if ctl.cell.attached(part):
            out["released_by_slot"] = True
            ctl.cell.detach(part)
            ctl.grip = GRIPPER_OPEN
            ctl.step(ctl.config.grip_steps)
            tcp = ctl.d.site_xpos[ctl.arm.tcp_site][:2].copy()
            ctl.vertical(tcp, ctl.config.carry_height, yaw)
        return out


INSERTIONS = ("straight_down", "search")


def make_insertion(insertion: str | InsertionActor | None = None) -> InsertionActor:
    """An insertion actor from its name, or the actor itself (None: the placeholder).
    "straight_down": the placeholder; "search": control/scripted_insertion.py's SearchInsertion
    behind NestInsertion (that module is written on main; until it is merged here the name is
    refused). The learned insertion goes in as an actor (or a PoseInsertion in NestInsertion)."""
    if insertion is None or insertion == "straight_down":
        return StraightDown()
    if not isinstance(insertion, str):
        return insertion
    if insertion == "search":
        try:
            from polypanda.control.scripted_insertion import SearchInsertion
        except ImportError as exc:
            raise NotImplementedError("the search insertion (control/scripted_insertion.py) "
                                      "is not on this branch yet") from exc
        return NestInsertion(SearchInsertion(), "search")
    raise ValueError(f"unknown insertion {insertion!r} (one of {INSERTIONS}, or an actor)")


class LineDoor:
    """The line's scripted lid move (ScriptedTending.move_lid), with the controller's camera
    belief of the tester (its handle measured; its slide direction, travel and latch as
    programmed). Lid testers only."""
    name = LINE_DOOR

    def move(self, ctl: TestCellController, tester: str, command: str) -> dict:
        if ctl.cell.spots[tester].kind != "lid":
            raise NotImplementedError("the line's scripted door move works on lid testers only")
        ctl.move_lid(tester, opening=command == "open")
        return {}


class EnvDoor:
    """A v4.0 door actor run through DoorEnv on the tester, in the cell's own scene (as
    evaluation/line_door): a door script of control/door_baselines.py, the privileged expert, or a
    trained door policy "run:<path>[@final]". The `handover` is "door_start" (first straight down
    to the door skill's trained start height over the believed nest) or "carry". One env per
    tester, built on first use."""

    def __init__(self, actor: str = "feel_v2", handover: str = "door_start") -> None:
        if actor not in DOOR_SCRIPTS and not actor.startswith("run:"):
            raise ValueError(f"unknown door actor {actor!r}")
        if handover not in HANDOVERS:
            raise ValueError(f"handover must be one of {HANDOVERS}, not {handover!r}")
        self.name, self.actor, self.handover = actor, actor, handover
        self._envs: dict[str, tuple] = {}

    def _env(self, ctl: TestCellController, tester: str) -> tuple:
        """(env, act(obs), reset()) on tester `tester`."""
        if tester in self._envs:
            return self._envs[tester]
        from polypanda.environments.door import DoorEnv
        cell = ctl.cell
        scene, machine = (cell.model, cell.data), cell.machines[tester]
        if self.actor.startswith("run:"):
            from polypanda.evaluation.policies import load_run_policy, run_env_cls
            env = run_env_cls(self.actor[4:])(scene=scene, machine=machine)
            out = (env, load_run_policy(self.actor[4:]), lambda: None)
        else:
            from polypanda.control.door_baselines import BaselineDoor
            from polypanda.control.scripted_door import DoorExpert
            env = DoorEnv(scene=scene, machine=machine)
            script = DoorExpert(env) if self.actor == "expert" else BaselineDoor(env, self.actor)
            out = (env, lambda obs: script.act(), script.reset)
        self._envs[tester] = out
        return out

    def move(self, ctl: TestCellController, tester: str, command: str) -> dict:
        from polypanda.supervisor.stations import StationView
        env, act, reset = self._env(ctl, tester)
        if self.handover == "door_start":
            ctl.lower_to_door_start(tester)
        view = StationView(env, np.radians(ctl.cell.spots[tester].angle), origin=ctl.base_xy())
        obs, _ = env.reset(seed=int(ctl.rng.integers(2**31)),
                           options={"continue": True, "command": command})
        obs = view.observe(obs)
        reset()
        info, steps = {}, 0
        while steps < env.config.horizon:
            obs, _, terminated, truncated, info = view.step(act(obs))
            ctl.tick(exclude=tester)          # the rest of the cell runs on meanwhile
            steps += 1
            if terminated or truncated:
                break
        ctl.q_cmd = env.adapter.q_cmd.copy()  # the cell takes over from where the hand is
        ctl.grip = GRIPPER_OPEN
        ctl.to_carry()
        ctl.step(5)
        st = info.get("stats", {})
        return {"steps": steps, "success": bool(info.get("is_success", False)),
                "failure": info.get("failure"),
                "peak_force": round(float(st.get("peak_force", 0.0)), 2)}


def make_door(door: str | DoorActor, handover: str = "door_start") -> DoorActor:
    """A door actor from its name ("line_camera", a door script, "run:<path>[@final]"), or the
    actor itself."""
    if not isinstance(door, str):
        return door
    return LineDoor() if door == LINE_DOOR else EnvDoor(door, handover)


REJECT_REASONS = ("no_read", "test_fail", "vision")              # to the FAIL bin
REWORK_REASONS = ("not_seated", "not_running", "test_aborted", "no_result",   # not tested
                  "not_inspected")     # (tested, passed, but its front was under a finger)
VERDICTS = {"pass": "pass_tray", "fail": "fail_bin", "not_tested": "rework_tray"}


@dataclass
class ModuleRecord:
    index: int
    serial: str
    good: bool                     # the hidden truth, copied here for scoring only
    electrical_pass: bool
    defect: str | None
    electrical_defect: str | None = None
    severity: float = 1.0
    residue: float = 0.0
    scan: dict | None = None
    tester: str | None = None
    inserts: list = field(default_factory=list)
    test_result: str | None = None
    vision: dict | None = None
    verdict: str | None = None    # "pass", "fail" or "not_tested": the controller's decision
    reject_reason: str | None = None   # why a "fail" verdict (one of REJECT_REASONS)
    rework_reason: str | None = None   # why "not_tested" (one of REWORK_REASONS)
    sorted_to: str | None = None  # "pass_tray", "fail_bin", "rework_tray", or None (not sorted
                                  # by the robot)
    failures: list = field(default_factory=list)
    t_taken: float | None = None
    t_test_start: float | None = None
    t_sorted: float | None = None

    @classmethod
    def of(cls, index: int, truth) -> ModuleRecord:
        """The record of module `index` with hidden truth `truth` (a ModuleTruth)."""
        return cls(index, truth.serial, truth.good, truth.electrical_pass, truth.defect,
                   truth.electrical_defect, truth.severity, truth.residue)

    @property
    def judged(self) -> bool:
        """Sorted PASS or FAIL (not to rework, not cleared by an operator)."""
        return self.sorted_to in ("pass_tray", "fail_bin")

    @property
    def correct(self) -> bool:
        """A judged module sorted as its truth says (a module not judged is not correct)."""
        return (self.sorted_to == "pass_tray" and self.good) or (
            self.sorted_to == "fail_bin" and not self.good)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["judged"], d["correct"] = self.judged, self.correct
        return d


class TestCellController(ScriptedTending):
    belt_wait: float = 30.0         # s to wait for the conveyor to bring a module
    settle_steps: int = 4           # control steps for a module dropped in the FAIL bin to land
    present_steps: int = 4          # control steps the hand stops at the vision camera (0.2 s:
                                    # the arm settles; the image is taken at the end)
    bin_capacity: int = 4           # modules in the FAIL bin before an operator empties it

    pneumatic_timeout: float = 6.0  # s for a pneumatic lid to open or latch on its signal
    scan_tries: int = 2             # barcode reads before a module is rejected as unreadable

    def __init__(self, cell: TestCell, door: str | DoorActor = "feel_v2",
                 handover: str = "door_start", insertion: str | InsertionActor | None = None,
                 nest_belief: str = "camera", seed: int = 0, insert_attempts: int = 2,
                 door_attempts: int = 2, imaging: ImagingNoise = NOISE,
                 config: CellControllerConfig | None = None,
                 log: Callable[[str], None] | None = None) -> None:
        if nest_belief not in ("camera", "true"):
            raise ValueError("nest_belief must be 'camera' or 'true'")
        super().__init__(cell, config)
        self.cell: TestCell = cell
        self.on_step = self._tick
        self.door: DoorActor = make_door(door, handover)
        self.nest_belief = nest_belief
        self.insertion: InsertionActor = make_insertion(insertion)
        self.insert_attempts, self.door_attempts = insert_attempts, door_attempts
        self.imaging = imaging
        self.rng = np.random.default_rng([seed, 3])         # the controller's sensing errors
        self.image_rng = np.random.default_rng([seed, 5])   # the cameras' noise
        self.log = log or (lambda msg: None)
        self._watch: tuple | None = None
        self._reset_logs()

    # --- bookkeeping ------------------------------------------------------------------------
    def _reset_logs(self) -> None:
        self.timeline: list[list] = []
        self.activities: list[dict] = []
        self.records: list[ModuleRecord] = []
        self.events: list[dict] = []
        self._shown: dict[str, tuple] = {}
        self.ready_since: dict[str, float] = {}
        self.blocked: set[str] = set()
        self.taken = 0
        self.tray_swaps = self.bin_emptied = self.rework_collected = 0
        self._peak = 0.0

    @property
    def now(self) -> float:
        return float(self.d.time)

    def tick(self, exclude: str | None = None) -> None:
        """One control step of the cell around an actor that steps the physics itself (a door
        env): the conveyor, the testers except `exclude` (the env updates its own), the logs."""
        self.cell.conveyor.update()
        self._tick(exclude)

    def _tick(self, exclude: str | None = None) -> None:
        """Every control step (after ScriptedCell.step's physics): the testers' cycles (except
        `exclude`), their states into the timeline, and the force watch."""
        for name, m in self.cell.machines.items():
            if name != exclude:
                m.update()
        self._record_states()
        if self._watch is not None:
            self._peak = max(self._peak, self._contact_force(*self._watch))

    def _record_states(self) -> None:
        for name, m in self.cell.machines.items():
            state = (m.state, bool(m.lid > 0.01))
            if self._shown.get(name) != state:
                if self._shown.get(name, (None,))[0] != m.state and m.state in (DONE, FAULT,
                                                                                 EMPTY):
                    self.ready_since[name] = self.now
                self._shown[name] = state
                self.timeline.append([round(self.now, 2), name, m.state, state[1]])

    def _contact_force(self, geoms: set, bodies: set) -> float:
        """The largest normal force between the part's geoms and the nest's bodies (N)."""
        m, d = self.m, self.d
        f6, peak = np.zeros(6), 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 in geoms and m.geom_bodyid[c.geom2] in bodies) or (
                    c.geom2 in geoms and m.geom_bodyid[c.geom1] in bodies):
                mujoco.mj_contactForce(m, d, i, f6)
                peak = max(peak, abs(f6[0]))
        return peak

    @contextmanager
    def doing(self, what: str, tester: str | None = None, module: int | None = None):
        t0 = self.now
        try:
            yield
        finally:
            self.activities.append({"t0": round(t0, 2), "t1": round(self.now, 2), "what": what,
                                    "tester": tester, "module": module})

    def fail(self, rec: ModuleRecord | None, stage: str, **info) -> None:
        self.events.append({"t": round(self.now, 2), "stage": stage,
                            "module": rec.index if rec else None, **info})
        if rec is not None:
            rec.failures.append(stage)
        self.log(f"{self.now:7.1f} s  FAILURE {stage} {info if info else ''}"
                 f"{' (' + rec.serial + ')' if rec else ''}")

    def record_of(self, part) -> ModuleRecord | None:
        k = self.cell.module_of(part)
        return None if k is None else self.records[k]

    # --- moving between stations --------------------------------------------------------------
    def go_station(self, station: Station) -> None:
        self.drive_to(station.stop)
        self.swing_to(np.radians(station.angle))

    def step_until(self, done: Callable[[], bool], timeout: float) -> bool:
        for _ in range(int(timeout / self.dt)):
            if done():
                return True
            self.step()
        return done()

    # --- the door -----------------------------------------------------------------------------
    def lower_to_door_start(self, name: str) -> None:
        """The "door_start" hand-over (evaluation/line_door): straight down to the door skill's
        trained start height over the believed nest, keeping the carry yaw."""
        from polypanda.evaluation.line_door import door_start_z
        b = self.beliefs[name]
        self.vertical(b.origin[:2], door_start_z(), self.carry_yaw(self.angle_of(b.origin[:2])))

    def door_move(self, name: str, command: str) -> dict:
        """One move of tester `name`'s door, at its stop, facing it: a lid or a drawer by the
        door actor (the slot), a pneumatic lid by its signal."""
        if self.cell.spots[name].kind == "pneumatic":
            return self.signal(name, command)
        return self.door.move(self, name, command) or {}

    def signal(self, name: str, command: str) -> dict:
        """A pneumatic tester (track B's Machine.request_open / request_close): signal it and
        wait until it is open, or shut and latched."""
        m = self.cell.machines[name]
        need = needed_opening(m.cfg) - 0.003
        ok = m.request_open() if command == "open" else m.request_close()
        if ok:
            done = (lambda: m.lid >= need) if command == "open" else (lambda: m.latched)
            ok = self.step_until(done, self.pneumatic_timeout)
        return {"signal": command, "success": bool(ok)}

    def opened(self, name: str) -> bool:
        m = self.cell.machines[name]
        return m.lid >= needed_opening(m.cfg) - 0.003

    def open_tester(self, name: str) -> bool:
        """Open tester `name` (at its stop, facing it) unless it is open already."""
        for _ in range(self.door_attempts):
            if self.opened(name):
                return True
            with self.doing("open", name):
                self.door_move(name, "open")
        return self.opened(name)

    def close_tester(self, name: str) -> bool:
        m = self.cell.machines[name]
        for _ in range(self.door_attempts):
            if m.latched:
                return True
            with self.doing("close", name):
                self.door_move(name, "close")
        return m.latched

    # --- the insertion ------------------------------------------------------------------------
    def measure(self, name: str) -> None:
        """Measure tester `name` (its nest's pose and its handle) with the camera, or take the
        truth (nest_belief "true")."""
        self.beliefs[name] = (camera_belief(self.cell, name, self.rng)
                              if self.nest_belief == "camera" else true_belief(self.cell, name))

    def insert(self, part, name: str, rec: ModuleRecord) -> bool:
        """Seat the held part in tester `name`'s nest with the insertion actor, up to
        `insert_attempts` times (a failed attempt: the part is picked out again, the nest
        measured again). Returns whether it is seated; a part that cannot be picked out again
        stays in the nest (the caller handles it)."""
        m = self.cell.machines[name]
        geoms = {g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == part.body
                 and (self.m.geom_contype[g] or self.m.geom_conaffinity[g])}
        nest = getattr(m, "nest_body", m.body)        # a drawer's nest rides on the drawer
        bodies = {nest} | {self.m.body(f"{name}_pocket{k}").id for k in range(4)}
        for attempt in range(self.insert_attempts):
            self.measure(name)
            t0, self._peak, self._watch = self.now, 0.0, (geoms, bodies)
            with self.doing("insert", name, rec.index):
                try:
                    out = self.insertion.insert(self, part, name) or {}
                finally:
                    self._watch = None
            seated = bool(m.in_pocket(part)[1])
            rec.inserts.append({"attempt": attempt, "seated": seated,
                                "s": round(self.now - t0, 2), "peak_force": round(self._peak, 1),
                                **{k: v for k, v in out.items() if k != "seated"}})
            if seated:
                return True
            self.fail(rec, "insert", attempt=attempt)
            if attempt + 1 < self.insert_attempts:
                with self.doing("unjam", name, rec.index):
                    self.unload(part)
                if not self.cell.attached(part):
                    return False
        return False

    # --- stations -----------------------------------------------------------------------------
    def look(self, camera: str) -> Inspection:
        """One image from `camera` ("scanner" or "vision"), with the cameras' noise, read."""
        return inspect(self.imaging.apply(self.cell.camera_image(camera), self.image_rng))

    def scan(self, rec: ModuleRecord) -> bool:
        """Read the barcode of the module at the pick point, trying `scan_tries` times."""
        for k in range(self.scan_tries):
            if k:
                self.step(4)                  # a moment for the next frame
            ins = self.look("scanner")
            rec.scan = {"read": ins.readable, "tries": k + 1, **ins.as_dict()}
            if ins.readable:
                return True
        return False

    def place_at(self, part, xy) -> None:
        """Set a held part down at `xy` on the bench (not necessarily on the station circle):
        over it at the carry height first, then the cell's place, then back to the carry
        posture."""
        yaw = self.carry_yaw(self.angle_of(xy))
        self.swing_to(self.angle_of(xy))
        self.move_joints(self._solve([xy[0], xy[1], self.config.carry_height],
                                     top_down_quat(yaw), self.q_cmd))
        self.place(part, xy, 0)
        self.to_carry()

    def regrip(self, part, rec: ModuleRecord | None, stage: str) -> bool:
        """After a pick: if the grasp did not hold, let go and try once more."""
        if self.cell.attached(part):
            return True
        self.fail(rec, stage)
        self.grip = GRIPPER_OPEN
        self.step(self.config.grip_steps)
        self.to_carry()
        self.pick(part)
        return self.cell.attached(part)

    def operator_clears(self, part, rec: ModuleRecord | None, why: str) -> None:
        """A part the robot cannot handle any more is taken away by an operator (counted; the
        module is not sorted by the robot)."""
        self.events.append({"t": round(self.now, 2), "stage": "operator", "why": why,
                            "module": rec.index if rec else None})
        self.grip = GRIPPER_OPEN
        self.step(self.config.grip_steps)
        self.to_carry()
        self.cell.release(self.cell.cubes.index(part))

    def present(self, part) -> tuple[float, bool]:
        """Inspection on the fly: carry the held module to the vision station and show its
        front to the camera. At the station's rail stop, one joint move from the carry posture
        swings the base to the station and turns the wrist (both about the vertical: the hand
        stays level) so that the module's front looks out at the camera; a brief stop
        (`present_steps`) lets the arm settle. The module stays in the hand, held as it was
        taken out of the nest. How it sits in the hand is the cell's own knowledge (it put the
        keyed module into the nest and took it out). If no wrist turn within joint 7's range
        brings the front round, the wrist turns as far as it can (the front at an angle).
        Returns the wrist's turn (rad) and whether a finger holds the front (then the camera
        sees the finger, not the label: a module taken out of a keyed nest is held across its
        rib's face, which leaves the front free)."""
        st = self.cell.cell_config.vision
        self.drive_to(st.stop)
        a = np.radians(st.angle)
        R = self.d.site_xmat[self.arm.tcp_site].reshape(3, 3)
        front = -self.d.xmat[part.body].reshape(3, 3)[:, 1]      # the module's local -y
        held = bool(abs(front @ R[:, 1]) > np.cos(np.pi / 4))     # the fingers close along y
        rel = float(np.arctan2(front[1], front[0])) - closing_yaw(R)
        turn = (a - self.carry_yaw(a) - rel + np.pi) % (2 * np.pi) - np.pi
        q = self.carry.copy()
        delta = (a - self.base_angle + np.pi) % (2 * np.pi) - np.pi    # as swing_to
        q[0] = self.q_cmd[0] + delta
        if abs(q[0]) > 2.89:
            q[0] -= np.sign(q[0]) * 2 * np.pi
        turns = self._wrist_turns(turn, q[6])
        if turns:
            turn = turns[0]
        q[6] = float(np.clip(q[6] - turn, self.ik.q_low[6] + self.wrist_margin,
                             self.ik.q_high[6] - self.wrist_margin))
        self.move_joints(q, rate=self.config.swing_rate)
        self.step(self.present_steps)
        return float(self.carry[6] - q[6]), held

    def inspect_visual(self, part, rec: ModuleRecord) -> bool:
        """Present the held module to the vision camera (`present`) and image its front
        (`front_held`: a finger is over it, so the image does not count). Returns whether the
        module is still held."""
        with self.doing("vision", None, rec.index):
            turn, held = self.present(part)
            ins = self.look("vision")
            rec.vision = {"defect": ins.defect, "front_held": held,
                          "wrist_turn_deg": round(np.degrees(turn), 1), **ins.as_dict()}
        return self.cell.attached(part)

    def reject(self, cube) -> None:
        """ScriptedCell.reject with the hand turned as it holds the module (a module presented
        to the camera keeps the wrist turned: the drop works either way, and the hand does not
        snap round on the way down)."""
        xy = self.cell.reject_xy
        yaw = closing_yaw(self.d.site_xmat[self.arm.tcp_site].reshape(3, 3))
        offset = self.d.site_xpos[self.arm.tcp_site][2] - self.d.xpos[cube.body][2]
        drop = self.cell.config.bin_height + 0.05 + CUBE_HEIGHT / 2   # bottom 5 cm over the walls
        self.vertical(xy, drop + offset, yaw)
        self.cell.detach(cube)
        self.grip = GRIPPER_OPEN
        self.step(self.config.grip_steps)
        self.vertical(self.d.site_xpos[self.arm.tcp_site][:2].copy(), self.config.carry_height,
                      yaw)
        if abs(self.q_cmd[6] - self.carry[6]) > 0.01:     # the wrist back, the hand empty
            self.to_carry()

    def sort(self, part, rec: ModuleRecord, verdict: str, reason: str | None = None) -> None:
        """PASS tray, FAIL bin, or REWORK tray (`verdict` "pass", "fail" or "not_tested";
        `reason`: why a "fail", one of REJECT_REASONS, or why "not_tested", one of
        REWORK_REASONS); the part's place is checked afterwards."""
        cell, cfg = self.cell, self.cell.cell_config
        where = VERDICTS[verdict]
        rec.verdict = verdict
        if verdict == "not_tested":
            rec.rework_reason = reason
        else:
            rec.reject_reason = reason
        with self.doing("rework" if verdict == "not_tested" else "sort", None, rec.index):
            if verdict == "fail":
                self.go_station(cfg.fail_bin)
                self.reject(part)
                self.step(self.settle_steps)
                ok = cell.in_bin(part)
            else:
                which, station = (("pass", cfg.tray) if verdict == "pass"
                                  else ("rework", cfg.rework))
                pocket = cell.free_pocket(which)
                if pocket is None:
                    self.collect(which)
                    pocket = cell.free_pocket(which)
                self.drive_to(station.stop)
                self.place_at(part, cell.tray(which)[pocket])
                ok = cell.pocket_of(part, which=which) == pocket
        rec.t_sorted = round(self.now, 2)
        if ok:
            rec.sorted_to = where
        else:
            self.fail(rec, f"sort_{where}")
        self.log(f"{self.now:7.1f} s  {rec.serial}: {verdict.upper()}"
                 f"{' (' + reason + ')' if reason else ''} -> {rec.sorted_to or 'lost'} "
                 f"(truth {'good' if rec.good else 'bad'})")
        for which in ("pass", "rework"):
            if cell.free_pocket(which) is None:
                self.collect(which)
        in_bin = [k for k, c in enumerate(cell.cubes) if cell.module_of(c) is not None
                  and cell.in_bin(c)]
        if len(in_bin) >= self.bin_capacity:
            for k in in_bin:
                cell.release(k)
            self.bin_emptied += 1

    def collect(self, which: str = "pass") -> None:
        """An operator exchanges the full PASS tray for an empty one, or takes the REWORK
        tray's modules to the rework bench."""
        for k, c in enumerate(self.cell.cubes):
            if (self.cell.module_of(c) is not None
                    and self.cell.pocket_of(c, which=which) is not None):
                self.cell.release(k)
        if which == "pass":
            self.tray_swaps += 1
        else:
            self.rework_collected += 1
        self.events.append({"t": round(self.now, 2), "stage": "operator",
                            "why": "tray swap" if which == "pass" else "rework collected",
                            "module": None})

    # --- one tester's visit ---------------------------------------------------------------------
    def to_fetch(self) -> int:
        """Modules of the shift not yet taken off the conveyor."""
        return len(self.cell.truths) - self.taken

    def fetch(self, name: str) -> tuple | None:
        """From the conveyor: the next module, scanned and picked for tester `name`, as (part,
        record); modules that cannot be read are rejected on the way. None when none is left."""
        cell = self.cell
        while self.to_fetch() > 0:
            with self.doing("fetch", name):
                self.go_station(cell.cell_config.infeed)
                if not self.step_until(lambda: cell.conveyor.ready, self.belt_wait):
                    self.fail(None, "conveyor")
                    return None
                part = cell.conveyor.lead()
                rec = self.records[cell.module_of(part)]
                self.taken += 1
                rec.t_taken = round(self.now, 2)
                readable = self.scan(rec)
                if readable:
                    self.pick_for(part, name)
                else:
                    self.pick(part)
                held = self.regrip(part, rec, "pick")
                cell.conveyor.index()              # the next module comes up meanwhile
            if not held:
                self.operator_clears(part, rec, "pick failed")
                continue
            if not readable:
                self.log(f"{self.now:7.1f} s  {rec.serial}: no read at the scanner")
                self.sort(part, rec, "fail", "no_read")
                continue
            return part, rec
        return None

    def take_out(self, name: str) -> None:
        """Unload the module of an open tester: a tested one is inspected on the way and
        sorted; one whose test was aborted (or that has no result) goes to the REWORK tray,
        not tested."""
        cell, m = self.cell, self.cell.machines[name]
        part, result = m.part, cell.tester_result(name)
        if part is None:
            return
        rec = self.record_of(part)
        rec.test_result = result
        with self.doing("unload", name, rec.index):
            self.unload(part)
        if not self.regrip(part, rec, "unload"):
            self.operator_clears(part, rec, "unload failed")
            if m.state == FAULT:
                m.reset()
            return
        if m.state == FAULT:
            m.reset()                              # the cell acknowledges the aborted test
        if result not in ("pass", "fail"):
            self.sort(part, rec, "not_tested",
                      "test_aborted" if result == "fault" else "no_result")
            return
        if not self.inspect_visual(part, rec):
            self.operator_clears(part, rec, "dropped at the vision station")
            return
        if rec.vision["front_held"] and result == "pass":
            self.sort(part, rec, "not_tested", "not_inspected")     # its front was not seen
            return
        reason = ("test_fail" if result == "fail"
                  else "vision" if rec.vision["defect"] else None)
        self.sort(part, rec, "fail" if reason else "pass", reason)

    def serve(self, name: str) -> None:
        """One visit to tester `name`: open, take the tested module out, bring the next, load,
        close."""
        cell, m = self.cell, self.cell.machines[name]
        with self.doing("drive", name):
            self.go_to(name)
        self.measure(name)
        if not self.open_tester(name):
            self.fail(None, "door_open", tester=name)
            self.blocked.add(name)
            return
        if m.state in (DONE, FAULT) or m.part is not None:
            self.take_out(name)
        got = self.fetch(name)
        if got is None:
            return
        part, rec = got
        rec.tester = name
        with self.doing("drive", name, rec.index):
            self.go_to(name)
        if not self.insert(part, name, rec):
            # not seated: out again and to the REWORK tray, not tested (a part that cannot be
            # picked out of the nest is cleared by an operator)
            if not cell.attached(part):
                self.pick(part)
            if cell.attached(part):
                self.sort(part, rec, "not_tested", "not_seated")
            else:
                self.operator_clears(part, rec, "jammed in the nest")
            return
        if not self.close_tester(name):
            self.fail(rec, "door_close", tester=name)
            self.blocked.add(name)
            return
        self.step(2)
        if m.state != RUNNING:
            # shut and latched but the test did not start: the module was knocked out of its
            # seat (by the door move); take it out again, not tested, to the REWORK tray
            self.fail(rec, "not_running", state=m.state)
            self.take_back(name, part, rec)
            return
        rec.t_test_start = round(self.now, 2)
        self.log(f"{self.now:7.1f} s  {rec.serial} testing in {name}")

    def take_back(self, name: str, part, rec: ModuleRecord) -> None:
        """A module that went into tester `name` but whose test did not start: open the tester,
        pick the module out and put it in the REWORK tray ("not_running": the same case as one
        that would not seat, found later), or have an operator clear it."""
        m = self.cell.machines[name]
        if not self.open_tester(name):
            self.fail(rec, "door_open", tester=name)
            self.blocked.add(name)
            return
        with self.doing("unload", name, rec.index):
            self.unload(part)
        if self.regrip(part, rec, "unload"):
            self.sort(part, rec, "not_tested", "not_running")
        else:
            self.operator_clears(part, rec, "stuck in the tester")
        if m.state != EMPTY:
            m.reset()

    # --- the shift ------------------------------------------------------------------------------
    def choose(self) -> str | None:
        """The tester to serve next (see the module's docstring), or None."""
        best = None
        for name, m in self.cell.machines.items():
            if name in self.blocked:
                continue
            needs = m.state in (DONE, FAULT) or (m.state == EMPTY and self.to_fetch() > 0)
            if needs and (best is None or self.ready_since[name] < self.ready_since[best]):
                best = name
        return best

    def wait_for_testers(self) -> bool:
        """Nothing to do: go to the tester that will finish first and wait for any to finish.
        False if no test is running (nothing will ever finish)."""
        running = {n: m for n, m in self.cell.machines.items()
                   if m.state == RUNNING and n not in self.blocked}
        if not running:
            return False
        first = min(running, key=lambda n: running[n].started + running[n].cfg.cycle_s)
        with self.doing("drive", first):
            self.go_to(first)
        with self.doing("wait", first):
            self.step_until(lambda: self.choose() is not None, 600.0)
        return True

    def finished(self) -> bool:
        return self.to_fetch() == 0 and not any(
            m.part is not None and n not in self.blocked
            for n, m in self.cell.machines.items())

    def run_shift(self, truths, seed: int = 0, max_time: float | None = None) -> None:
        """Run a shift of modules with hidden `truths` (simulation/test_cell.draw_truths). The
        conveyor's placement noise is seeded by `seed`. Stops early at `max_time` (simulated s;
        default 300 s per module)."""
        cell = self.cell
        cell.reset_shift(truths, np.random.default_rng([seed, 2]), lateral=0.003,
                         yaw=np.radians(5.0))
        self._reset_logs()
        self.records = [ModuleRecord.of(k, t) for k, t in enumerate(truths)]
        self._record_states()
        for k, name in enumerate(cell.machines):
            self.ready_since[name] = self.now + k * 1e-3     # A, then B, then C
        self.q_cmd, self.grip = self.home.copy(), GRIPPER_OPEN
        self.ik_failures = 0
        self.t_start = self.now
        limit = self.t_start + (max_time if max_time is not None else 300.0 * len(truths))
        cell.conveyor.index()
        self.to_carry()
        while not self.finished() and self.now < limit:
            name = self.choose()
            if name is not None:
                self.serve(name)
            elif not self.wait_for_testers():
                break
        self.t_end = self.now
        self.timed_out = self.now >= limit


TestCellController.__test__ = False   # (not a pytest test class)
