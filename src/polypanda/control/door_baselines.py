"""v4.0: the door skill's scripted baselines (docs/v4_targets.md), acting through DoorEnv's own
5-D actions and reading only what the learned policy observes, plus the nominal machine's settings
where stated.

- **nominal:** the script written for one machine. It believes the nominal machine stands at its
  nominal place: the handle where the nominal machine has it, the lid sliding right with 0.12 m of
  travel (0.27 m outward), no spring, and a close at the line script's door speed.
- **camera:** the handle where the camera sees it, and the slide's axis from the bar's seen angle
  (outward if the bar runs across the robot's view, otherwise the nominal sense, right). Travel,
  latch and spring nominal; it opens to the believed travel less 5 mm, as the line's script does.
- **feel:** as camera, and it feels. If the bar does not move when pushed (the hand lags its own
  command: something holds it), it tries the other way. It opens until the lid stops against its
  stop (which also seats a springy lid in its detent). It closes, and if the machine's latched
  signal does not come, pulls the lid back open and pushes again faster, up to 2 retries.
- **feel_v2** (dev round 4, 2026-11-24, when the policy was given the machine's door-open switch,
  DoorV2Env): feel, and it also reads that switch: an open ends as soon as it comes, without
  waiting to feel the stop. `feel` stays the script of rounds 1-3 (and of the demonstrations).

All three hook the bar with open fingers, as the expert (control/scripted_door.py) and the line's
scripted door do, and move along the slide at the line's door speed (0.05 m/s) at first. The
feel script is also the source of the demonstrations: it decides from what the policy can see.

**Drawers (v4.1).** Every script knows which kind of tester it faces (a drawer is plain to see,
and a cell knows its testers), and so that a drawer opens toward the robot: it never tries the
other way. It believes the nominal drawer (MachineConfig(kind="drawer")): its travel, the bar's
16 mm width (where to come down behind it: scripted_door.hook_offset) and its 3.5 cm stand-off
(the slack before the finger meets the plate when closing). The camera script sees the bar; the
nominal one believes it where the nominal drawer has it.
"""

from __future__ import annotations

import numpy as np

from polypanda.control.ik import closing_yaw
from polypanda.control.scripted_door import (
    DoorExpert,
    back_off_for,
    hook_offset,
    hook_slack,
    reopen_reach,
)
from polypanda.simulation.machine import MachineConfig
from polypanda.simulation.panda import tcp_position

MODES = ("nominal", "camera", "feel", "feel_v2")
NOMINAL = MachineConfig()
NOMINAL_DRAWER = MachineConfig(kind="drawer")
NOMINAL_OUT_TRAVEL = 0.27        # the line's outward lid (evaluation/tending.make_variant)


class BaselineDoor(DoorExpert):
    door_speed: float = 0.05      # m/s along the slide (the line's ScriptedTending.door_speed)
    open_margin: float = 0.005    # m short of the believed travel (ScriptedTending.open_margin)
    latch_push: float = 0.004     # m pushed past closed (ScriptedTending.latch_push)
    blocked_lag: float = 0.020    # m the hand lags its command along the push: it is held (the
    #                               arm trails ~1 cm at full reach, so mostly the force decides)
    stop_force: float = 32.0      # N: or the push force passes this (at a stop; a free lid
    #                               resists with at most ~25 N)
    probe_lag: float = 0.008      # m and N: before the lid has moved, the same tighter, and the
    probe_force: float = 28.0     # hand has not moved since touching the bar: the wrong way
    probe_still: float = 0.004    # m
    wrong_way_open: float = 0.01  # m: blocked within this of lid travel when opening: wrong way
    wrong_way_close: float = 0.05  # m: the same when closing (a lid opened wrongly hits its stop)
    max_speed: float = 0.18       # m/s, retries
    back_speed: float = 0.12      # m/s backing off the bar or pulling a lid back for a retry
    reopen: float = 0.04          # m a lid is pulled back open for a retry (a run-up)
    retry_gain: float = 1.6       # each retry pushes this much faster
    max_open: float = 0.30        # m of lid travel the feel script pushes at most
    drawer_stop_force: float = 18.0  # N: a drawer's stop (a free drawer resists with at most
    #                               ~12 N, and the finger pushing it shut gives way: below)

    def __init__(self, env, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        self.open_switch = mode == "feel_v2"     # feel_v2: feel plus the door-open switch
        self.mode = "feel" if self.open_switch else mode
        super().__init__(env)

    def reset(self) -> None:
        super().reset()
        self.belief = None
        self.clear_give = 0.0     # (a drawer: how far the closing finger had given way)
        self.flipped = False      # tried the other way
        self.confirmed = False    # the lid moved: the way is right, never flip again
        self.restart = False      # after backing off: push again from where the hand is
        self.fresh_touch = False  # reversed: wait for the other finger to meet the bar
        self.sign = 1.0

    # --- beliefs ----------------------------------------------------------------------------
    def _station(self) -> tuple[np.ndarray, np.ndarray]:
        """The station frame's radial (away from the robot) and tangential directions."""
        f = self.env.frame
        return np.array([np.cos(f), np.sin(f), 0.0]), np.array([-np.sin(f), np.cos(f), 0.0])

    def _believe(self) -> dict:
        env = self.env
        radial, tangent = self._station()
        if env.machine.cfg.kind == "drawer":
            return self._believe_drawer(radial)
        if self.mode == "nominal":
            cfg = NOMINAL
            travel = cfg.travel
            q = 0.0 if env.command == "open" else travel - self.open_margin
            centre = np.array([env.config.scene.radius, 0.0, 0.0]) + cfg.handle_centre(q)
            c, s = np.cos(env.frame), np.sin(env.frame)
            handle = (np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ centre
                      + env.base)                     # about the base axis (on a rail it moves)
            axis = -tangent                                  # right: the machine's -y
        else:
            handle = env._seen_handle.copy()                 # the camera's estimate
            yaw = env._seen_yaw
            axis = np.array([-np.sin(yaw), np.cos(yaw), 0.0])  # across the bar
            out = abs(axis @ radial) > abs(axis @ tangent)
            if out:
                axis = axis if axis @ radial > 0 else -axis
                travel = NOMINAL_OUT_TRAVEL
            else:
                axis = axis if axis @ -tangent > 0 else -axis  # the nominal sense: right
                travel = NOMINAL.travel
            q = 0.0 if env.command == "open" else travel - self.open_margin
        return {"handle": handle, "axis": axis, "travel": travel, "q": q, "cfg": NOMINAL}

    def _believe_drawer(self, radial: np.ndarray) -> dict:
        """A drawer: it opens toward the robot, so the way is known (`confirmed`); the nominal
        drawer's travel; the handle where the nominal drawer has it (nominal) or where the camera
        sees it, the slide's axis across the seen bar, toward the robot."""
        env, cfg = self.env, NOMINAL_DRAWER
        self.confirmed = True
        q = 0.0 if env.command == "open" else cfg.travel - self.open_margin
        if self.mode == "nominal":
            centre = np.array([env.config.scene.radius, 0.0, 0.0]) + cfg.handle_centre(q)
            c, s = np.cos(env.frame), np.sin(env.frame)
            handle = (np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ centre
                      + env.base)
            axis = -radial
        else:
            handle = env._seen_handle.copy()
            yaw = env._seen_yaw
            axis = np.array([-np.sin(yaw), np.cos(yaw), 0.0])  # across the bar
            axis = axis if axis @ radial < 0 else -axis      # toward the robot
        return {"handle": handle, "axis": axis, "travel": cfg.travel, "q": q, "cfg": cfg}

    # --- policy -----------------------------------------------------------------------------
    def _give(self) -> float:
        """How far the finger pushing a drawer shut has given way (m): it pushes the plate with
        its back, against the hand's finger servo (scene.finger_stiffness, 1000 N/m), not its
        open stop as a pad does, so it yields 1 mm per newton the robot feels. Backing off, the
        hand moves this much more before the finger leaves the plate. 0 on a lid, and opening."""
        if self.belief["cfg"].kind != "drawer" or self.env.command != "close":
            return 0.0
        return float(np.linalg.norm(self.env._force)) / self.env.config.scene.finger_stiffness

    def _lag(self) -> float:
        """How far the measured TCP trails the commanded target along the push."""
        env = self.env
        lag = env.adapter.target_pos - tcp_position(env.data, env.arm)
        return float(lag @ (self.belief["axis"] * self.sign))

    def _begin_push(self) -> None:
        b = self.belief
        cfg = b["cfg"]
        self.slack = hook_slack(cfg, self.env.command)
        if self.start is None:
            self.start = self.env.adapter.target_pos.copy()
        self.s = 0.0
        if self.env.command == "open":
            self.sign = 1.0 if not self.flipped else -1.0
            self.cap = ((b["travel"] - self.open_margin) + self.slack if self.mode != "feel"
                        else self.max_open + self.slack)
        else:
            self.sign = -1.0 if not self.flipped else 1.0
            self.cap = ((b["q"] + self.latch_push) + self.slack if self.mode != "feel"
                        else self.max_open + self.slack)
        if self.tries == 0 and not self.flipped:
            self.speed = self.door_speed
        self.held = 0
        self._next("push")

    def act(self) -> np.ndarray:
        env = self.env
        if self.belief is None:
            self.belief = self._believe()
        b = self.belief
        handle, axis = b["handle"], b["axis"]
        tcp = tcp_position(env.data, env.arm)
        hook = handle                 # where the TCP comes down (a drawer's: off the bar's centre)
        if b["cfg"].kind == "drawer":
            hook = handle + axis * hook_offset(b["cfg"])
        if self.yaw_goal is None:     # fingers across the bar, along the slide: the nearer way
            psi = float(np.arctan2(axis[1], axis[0]))
            cur = closing_yaw(env.adapter.target_R)
            self.yaw_goal = psi + np.pi * np.round((cur - psi) / np.pi)
        self.count += 1
        move, yaw = np.zeros(3), 0.0
        dt = env.adapter.control_dt
        feel = self.mode == "feel"
        if self.phase == "approach":
            goal = hook + [0.0, 0.0, self.hover]
            move, yaw = self._move(goal, self.approach_speed), self._yaw_action()
            if (np.linalg.norm(goal - tcp) < 0.006 and abs(yaw) * env.adapter.max_rot < 0.03) \
                    or self.count > 120:
                self._next("descend")
        elif self.phase == "descend":
            move, yaw = self._move(hook, self.descend_speed), self._yaw_action()
            if np.linalg.norm(hook - tcp) < 0.002 or self.count > 60:
                self.start = env.adapter.target_pos.copy()
                self._begin_push()
        elif self.phase == "push":
            self.s = min(self.s + self.speed * dt, self.cap)
            move = self._move(self.start + axis * self.sign * self.s, 1.0, measured=False)
            # what the robot measures: its hand's travel along the push since a finger came onto
            # the bar (the observation touch_travel), how far it lags its own command, the force
            moved = (float((tcp - env._touch_at) @ (axis * self.sign))
                     if env._touch_at is not None else 0.0)
            lid_travel = moved
            wrong = self.wrong_way_open if env.command == "open" else self.wrong_way_close
            force = float(np.linalg.norm(env._force))
            if feel and env._touch_at is not None and moved >= wrong:
                self.confirmed = True
            stop_force = (self.drawer_stop_force if b["cfg"].kind == "drawer"
                          else self.stop_force)
            limit_lag, limit_force = ((self.blocked_lag, stop_force) if self.confirmed
                                      else (self.probe_lag, self.probe_force))
            if self.fresh_touch and env._touch_at is None:
                self.fresh_touch = False                     # off the bar: the old touch is gone
            blocked = feel and env._touch_at is not None and not self.fresh_touch and (
                self._lag() > limit_lag or force > limit_force) and (
                self.confirmed or moved < (self.probe_still if env.command == "open" else wrong))
            if blocked:
                self.held += 1
            else:
                self.held = 0
            if self.held >= 1:
                if lid_travel < wrong and not (self.flipped or self.confirmed):
                    # the other way, at once and without stopping: what the hand does next can
                    # then be read from its motion alone (a policy without memory can copy it)
                    self.flipped, self.fresh_touch = True, True
                    self.start = self.start + axis * self.sign * self.s
                    self._begin_push()
                elif env.command == "close" and not env.machine.latched \
                        and self.tries < self.retries:
                    self.tries += 1                           # shut, not latched: again, faster
                    self.speed = min(self.speed * self.retry_gain, self.max_speed)
                    self.back_to = self.s - (reopen_reach(b["cfg"]) + self.reopen)
                    if b["cfg"].kind == "drawer":
                        self.back_to -= self._give()
                    self.restart = True
                    self._next("back")
                else:
                    self._next("clear")
            elif env.command == "close" and feel and env.machine.latched:
                self._next("clear")                          # the latched signal came
            elif (env.command == "open" and feel and self.open_switch
                  and env.door_open_switch()):
                self._next("clear")                          # the door-open signal came
            elif self.s >= self.cap:
                self.wait += 1
                if self.wait > self.settle_steps:
                    self._next("clear")
        elif self.phase == "back":
            self.s = max(self.s - self.back_speed * dt, self.back_to)
            move = self._move(self.start + axis * self.sign * self.s, 1.0, measured=False)
            if self.s <= self.back_to:
                self.wait += 1
                if self.wait > 3:
                    if self.restart:                         # push on from here
                        self.start = self.start + axis * self.sign * self.s
                    self._begin_push()
        elif self.phase == "clear":                          # back off the bar, then rise
            back = back_off_for(b["cfg"], self.back_off)
            if b["cfg"].kind == "drawer":                    # (and off a finger that gave way)
                if self.count == 1:
                    self.clear_give = self._give()
                back += self.clear_give
            goal = self.start + axis * self.sign * (self.s - back)
            move = self._move(goal, self.back_speed, measured=False)
            if np.linalg.norm(goal - env.adapter.target_pos) < 1e-4 or self.count > 40:
                self.wait += 1
                if self.wait > 3 or self.count > 40:
                    self.release_point = env.adapter.target_pos.copy()
                    self._next("rise")
        elif self.phase == "rise":
            goal = self.release_point.copy()
            goal[2] = handle[2] + self.rise
            move = self._move(goal, self.descend_speed)
        return self._action(move, yaw)
