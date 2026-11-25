"""v4.0: a scripted door expert that acts through DoorEnv's own action space.

It reads privileged truth (the handle's true position, which way the lid slides, how far it must
go, the latch's speed), so it is an oracle, not a baseline: it shows that the door task can be
solved through the env's interface (5-D actions: hand motion and gripper), and it is the future
source of demonstrations for "demonstrations first" (docs/v4_plan.md, section 3.3).

It **hooks** the bar, as the scripted tending door does: gripping the bar is fragile in this
simulator (fingertip pads 1-2 mm off the bar's centre jam on its edge). The fingers stay open,
turned to straddle the bar along the slide; the hand comes down around the bar and moves along
the slide, so the trailing finger's inside pushes the bar, over the slack (half the finger span
less half the bar) and then the lid's way. Open: until the lid is just past the needed opening,
short of its stop. Close: until the latch catches, at 1.6 x its speed; if the lid shuts without
latching, the hand moves back so the other finger pulls the lid 2.5 cm open, and pushes again
faster. Then the hand backs off the bar and rises clear. The pushes are servoed on the hand's
commanded target (feed-forward); the moves through free space close the loop on the measured
TCP, which takes out the servos' few millimetres of sag.

**Drawers (v4.1).** A drawer's bar stands off its front plate by a few centimetres, and the open
fingers span 8 cm: come down centred on the bar, the finger behind it would land on the plate. So
the hand comes down off-centre, toward the robot (`hook_offset`), with that finger HOOK_MARGIN
behind the bar's back face, in the gap between bar and plate. Opening, it pulls the bar toward
the robot; closing, the same finger pushes the plate (its back) home. The rest is the lid's
script with the drawer's slack (`hook_slack`) and a smaller back-off (half the gap left beside
the finger), so the finger stays clear of both bar and plate before it rises out of the gap.
"""

from __future__ import annotations

import numpy as np

from polypanda.control.ik import closing_yaw
from polypanda.simulation.panda import tcp_position

FINGER_SPAN = 0.08            # m between the open fingers' pads (Menagerie hand)
FINGER_THICK = 0.015          # m, a finger across its pad, over the bar's height (the hand's
#                               collision hull: 10 mm at the tip, 15 mm 1.5 cm up, 27 mm at the top)
HOOK_MARGIN = 0.005           # m: a drawer's hooking finger comes down this far behind the bar


def hook_offset(cfg) -> float:
    """Where the TCP comes down, from the bar's centre along the opening direction (m): over the
    bar for a lid (the fingers straddle it with 3 cm to spare on each side); for a drawer, toward
    the robot, so the finger behind the bar comes down HOOK_MARGIN behind its back face."""
    if cfg.kind != "drawer":
        return 0.0
    return FINGER_SPAN / 2 - cfg.handle_radius - HOOK_MARGIN


def hook_slack(cfg, command: str) -> float:
    """How far the hand moves along its push before the finger meets what it pushes, plus 4 mm:
    on a lid, half the finger span less half the bar; on a drawer, opening, the margin behind the
    bar, and closing, the gap from the finger to the plate (the bar's stand-off less half the bar,
    the margin and the finger)."""
    if cfg.kind != "drawer":
        return FINGER_SPAN / 2 - cfg.handle_radius + 0.004
    if command == "open":
        return HOOK_MARGIN + 0.004
    return cfg.handle_height - cfg.handle_radius - HOOK_MARGIN - FINGER_THICK + 0.004


def reopen_reach(cfg) -> float:
    """How far the hand moves back from a push that shut the lid before its other finger meets
    the bar (to pull it open for a retry): across the finger span less the bar on a lid; on a
    drawer, the finger behind the bar back across the gap from the plate."""
    if cfg.kind != "drawer":
        return FINGER_SPAN - 2 * cfg.handle_radius
    return cfg.handle_height - cfg.handle_radius - FINGER_THICK


def back_off_for(cfg, default: float) -> float:
    """How far the hand backs off what it pushed before rising: `default` on a lid; on a
    drawer, to the middle of the gap left beside the finger between bar and plate."""
    if cfg.kind != "drawer":
        return default
    return max(0.003, (cfg.handle_height - cfg.handle_radius - FINGER_THICK) / 2)


class DoorExpert:
    hover: float = 0.06           # m above the bar before coming down
    approach_speed: float = 0.15  # m/s
    descend_speed: float = 0.08   # m/s
    open_speed: float = 0.08      # m/s along the slide
    close_speed: tuple[float, float] = (0.06, 0.18)  # m/s: 1.6 x the latch speed, within these
    open_past: float = 0.004      # m past the needed opening (and at least 2 mm short of the stop)
    push_past: float = 0.004      # m the hand pushes on once the lid is shut
    settle_steps: int = 15        # steps at the end of a push waiting for the goal
    back_off: float = 0.012       # m the hand moves back off the bar before rising
    rise: float = 0.06            # m above the bar at the end
    retries: int = 2              # close: pull back and push again if the latch did not catch
    reopen: float = 0.025         # m the lid is pulled open for a retry
    detent_push: float = 0.01     # m further a springy lid may be pushed, into its detent

    def __init__(self, env) -> None:
        self.env = env            # a DoorEnv (unwrapped: world-frame truth)
        self.reset()

    def reset(self) -> None:
        self.phase, self.count, self.tries = "approach", 0, 0
        self.yaw_goal: float | None = None
        self.start = self.s = self.cap = self.s_closed = None
        self.wait = 0

    # --- helpers --------------------------------------------------------------------------
    def _move(self, goal, speed: float, measured: bool = True) -> np.ndarray:
        """Hand action (world frame) toward `goal` at `speed`: closed loop on the measured TCP
        (the target integrates half the error each step), or with measured=False on the
        commanded target alone (feed-forward)."""
        ad = self.env.adapter
        if measured:
            step = 0.5 * (np.asarray(goal, dtype=float) - tcp_position(self.env.data,
                                                                        self.env.arm))
        else:
            step = np.asarray(goal, dtype=float) - ad.target_pos
        limit = speed * ad.control_dt
        n = float(np.linalg.norm(step))
        if n > limit:
            step *= limit / n
        return step / ad.max_step

    def _yaw_action(self) -> float:
        ad = self.env.adapter
        err = (self.yaw_goal - closing_yaw(ad.target_R) + np.pi) % (2 * np.pi) - np.pi
        return float(np.clip(err / ad.max_rot, -1.0, 1.0))

    def _action(self, move_world: np.ndarray, yaw: float = 0.0) -> np.ndarray:
        c, s = np.cos(self.env.adapter.frame), np.sin(self.env.adapter.frame)
        to_frame = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])  # station frame
        a = np.concatenate([np.clip(to_frame @ move_world, -1, 1), [yaw, -1.0]])  # fingers open
        return a.astype(np.float32)

    def _next(self, phase: str) -> None:
        self.phase, self.count, self.wait = phase, 0, 0

    def _start_push(self) -> None:
        env, cfg = self.env, self.env.machine.cfg
        lid = env.machine.lid
        if self.start is None:
            self.start = env.adapter.target_pos.copy()
        slack = hook_slack(cfg, env.command)
        if env.command == "open":
            self.sign, self.speed = 1.0, self.open_speed
            self.goal_q = min(env.needed + self.open_past, cfg.travel - 0.002)
            self.cap = (cfg.travel - 0.002 - lid) + slack + self.s
            if cfg.lid_spring > 0:        # into the hold-open detent, or the spring shuts it;
                self.goal_q = max(self.goal_q, cfg.travel - cfg.detent_depth / 2)
                self.cap += self.detent_push  # the hand sags against the spring: push on
        else:
            self.sign = -1.0
            if self.tries == 0:
                self.speed = float(np.clip(1.6 * cfg.latch_speed, *self.close_speed))
            self.cap = lid + slack + self.push_past + self.s
        self.s_closed = None
        self._next("push")

    # --- policy ---------------------------------------------------------------------------
    def act(self) -> np.ndarray:
        env = self.env
        machine = env.machine
        handle = env.handle_xyz()
        axis = env.slide_axis()
        tcp = tcp_position(env.data, env.arm)
        hook = handle                 # where the TCP comes down (a drawer's: off the bar's centre)
        if machine.cfg.kind == "drawer":
            hook = handle + axis * hook_offset(machine.cfg)
        if self.yaw_goal is None:     # fingers across the bar, along the slide: the nearer way
            psi = float(np.arctan2(axis[1], axis[0]))
            cur = closing_yaw(env.adapter.target_R)
            self.yaw_goal = psi + np.pi * np.round((cur - psi) / np.pi)
        self.count += 1
        move, yaw = np.zeros(3), 0.0
        dt = env.adapter.control_dt
        if self.phase == "approach":
            goal = hook + [0.0, 0.0, self.hover]
            move, yaw = self._move(goal, self.approach_speed), self._yaw_action()
            if np.linalg.norm(goal - tcp) < 0.006 and abs(yaw) * env.adapter.max_rot < 0.03:
                self._next("descend")
        elif self.phase == "descend":
            move, yaw = self._move(hook, self.descend_speed), self._yaw_action()
            if np.linalg.norm(hook - tcp) < 0.002 or self.count > 60:
                self.s = 0.0
                self._start_push()
        elif self.phase == "push":
            q = machine.lid
            if env.command == "close" and self.s_closed is None and q <= 0.001:
                self.s_closed = self.s                     # shut: push on only a little
                self.cap = min(self.cap, self.s + self.push_past)
            self.s = min(self.s + self.speed * dt, self.cap)
            move = self._move(self.start + axis * self.sign * self.s, 1.0, measured=False)
            done = q >= self.goal_q if env.command == "open" else machine.latched
            if done:
                self._next("clear")
            elif self.s >= self.cap:
                self.wait += 1
                if self.wait > self.settle_steps:
                    if env.command == "close" and self.tries < self.retries:
                        self.tries += 1                     # not latched: reopen, push again
                        self.speed = min(self.speed * 1.5, self.close_speed[1])
                        self.back_to = self.s - (reopen_reach(machine.cfg) + self.reopen)
                        self._next("back")
                    else:
                        self._next("clear")
        elif self.phase == "back":
            self.s = max(self.s - 0.05 * dt, self.back_to)
            move = self._move(self.start + axis * self.sign * self.s, 1.0, measured=False)
            if self.s <= self.back_to:
                self.wait += 1
                if self.wait > 3:
                    self._start_push()
        elif self.phase == "clear":                       # back off the bar, then rise
            goal = self.start + axis * self.sign * (self.s - back_off_for(machine.cfg,
                                                                          self.back_off))
            move = self._move(goal, 0.05, measured=False)
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


def run_expert(env, seed: int | None = None, options: dict | None = None,
               expert: DoorExpert | None = None) -> dict:
    """One episode of the expert on `env`; returns the final info (with "steps", "return")."""
    expert = expert or DoorExpert(env)
    obs, info = env.reset(seed=seed, options=options)
    expert.reset()
    total, done, steps = 0.0, False, 0
    while not done:
        obs, reward, terminated, truncated, info = env.step(expert.act())
        total += reward
        steps += 1
        done = terminated or truncated
    return {**info, "steps": steps, "return": total, "episode": env.episode,
            "phase": expert.phase, "retries": expert.tries}
