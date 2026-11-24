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

All three hook the bar with open fingers, as the expert (control/scripted_door.py) and the line's
scripted door do, and move along the slide at the line's door speed (0.05 m/s) at first. The
feel script is also the source of the demonstrations: it decides from what the policy can see.
"""

from __future__ import annotations

import numpy as np

from polypanda.control.ik import closing_yaw
from polypanda.control.scripted_door import FINGER_SPAN, DoorExpert
from polypanda.simulation.machine import MachineConfig
from polypanda.simulation.panda import tcp_position

MODES = ("nominal", "camera", "feel")
NOMINAL = MachineConfig()
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

    def __init__(self, env, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        self.mode = mode
        super().__init__(env)

    def reset(self) -> None:
        super().reset()
        self.belief = None
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
        return {"handle": handle, "axis": axis, "travel": travel, "q": q}

    # --- policy -----------------------------------------------------------------------------
    def _lag(self) -> float:
        """How far the measured TCP trails the commanded target along the push."""
        env = self.env
        lag = env.adapter.target_pos - tcp_position(env.data, env.arm)
        return float(lag @ (self.belief["axis"] * self.sign))

    def _begin_push(self) -> None:
        b, cfg = self.belief, NOMINAL
        self.slack = FINGER_SPAN / 2 - cfg.handle_radius + 0.004
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
        if self.yaw_goal is None:     # fingers across the bar, along the slide: the nearer way
            psi = float(np.arctan2(axis[1], axis[0]))
            cur = closing_yaw(env.adapter.target_R)
            self.yaw_goal = psi + np.pi * np.round((cur - psi) / np.pi)
        self.count += 1
        move, yaw = np.zeros(3), 0.0
        dt = env.adapter.control_dt
        feel = self.mode == "feel"
        if self.phase == "approach":
            goal = handle + [0.0, 0.0, self.hover]
            move, yaw = self._move(goal, self.approach_speed), self._yaw_action()
            if (np.linalg.norm(goal - tcp) < 0.006 and abs(yaw) * env.adapter.max_rot < 0.03) \
                    or self.count > 120:
                self._next("descend")
        elif self.phase == "descend":
            move, yaw = self._move(handle, self.descend_speed), self._yaw_action()
            if np.linalg.norm(handle - tcp) < 0.002 or self.count > 60:
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
            limit_lag, limit_force = ((self.blocked_lag, self.stop_force) if self.confirmed
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
                    self.back_to = self.s - (FINGER_SPAN - 2 * NOMINAL.handle_radius
                                             + self.reopen)
                    self.restart = True
                    self._next("back")
                else:
                    self._next("clear")
            elif env.command == "close" and feel and env.machine.latched:
                self._next("clear")                          # the latched signal came
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
            goal = self.start + axis * self.sign * (self.s - self.back_off)
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
