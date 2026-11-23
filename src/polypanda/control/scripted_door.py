"""v4.0: a scripted door expert that acts through DoorEnv's own action space.

It reads privileged truth (the handle's true position, which way the lid slides, how far it must
go, the latch's speed), so it is an oracle, not a baseline: it shows that the door task can be
solved through the env's interface (5-D actions: hand motion and gripper), and it is the future
source of demonstrations for "demonstrations first" (docs/v4_plan.md, section 3.3).

The motion is the scripted tending door's (control/scripted_tending.py, move_lid): turn the
fingers to close along the slide over the handle, come down onto the bar, pinch it, slide the lid
(open: just past the needed opening and short of the stop; close: a few millimetres past shut, at
a speed the latch catches), let go, and rise clear. Positions are servoed on the hand's commanded
target (the adapter's target pose), which the arm follows within a few millimetres.
"""

from __future__ import annotations

import numpy as np

from polypanda.control.ik import closing_yaw
from polypanda.simulation.panda import tcp_position


class DoorExpert:
    hover: float = 0.06           # m above the bar before coming down
    approach_speed: float = 0.15  # m/s
    descend_speed: float = 0.08   # m/s
    open_speed: float = 0.08      # m/s along the slide
    close_speed: tuple[float, float] = (0.06, 0.18)  # m/s: 1.6 x the latch speed, within these
    open_past: float = 0.004      # m past the needed opening (and at least 2 mm short of the stop)
    push_past: float = 0.004      # m pushed past shut
    grip_steps: int = 12          # steps for the fingers to close or open (0.6 s)
    settle_steps: int = 15        # steps waiting at the end of the slide for the goal
    rise: float = 0.06            # m above the bar at the end
    # m the hand stands ahead of the bar along the opening direction when the fingers close, so
    # the finger behind the bar touches it first and pushes the lid the free way (open), never
    # into its closed stop (a pinned bar squeezed by an off-centre finger slips out)
    lead: float = 0.002
    retries: int = 2              # close: pull back and push again if the latch did not catch

    def __init__(self, env) -> None:
        self.env = env            # a DoorEnv (unwrapped: world-frame truth)
        self.reset()

    def reset(self) -> None:
        self.phase, self.count, self.tries = "approach", 0, 0
        self.yaw_goal: float | None = None
        self.grip_point = self.s = self.s_end = self.lid0 = None
        self.gripper = -1.0

    # --- helpers --------------------------------------------------------------------------
    def _move(self, goal, speed: float, measured: bool = True) -> np.ndarray:
        """Hand action (world frame) toward `goal` at `speed`: closed loop on the measured TCP
        (the target integrates half the error each step, so the servos' few millimetres of sag
        are taken out), or with measured=False on the commanded target alone (feed-forward)."""
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
        a = np.concatenate([np.clip(to_frame @ move_world, -1, 1), [yaw, self.gripper]])
        return a.astype(np.float32)

    def _next(self, phase: str) -> None:
        self.phase, self.count = phase, 0

    # --- policy ---------------------------------------------------------------------------
    def act(self) -> np.ndarray:
        env = self.env
        machine = env.machine
        handle = env.handle_xyz()
        axis = env.slide_axis()
        tcp = tcp_position(env.data, env.arm)
        if self.yaw_goal is None:     # fingers close along the slide: the nearer of two ways
            psi = float(np.arctan2(axis[1], axis[0]))
            cur = closing_yaw(env.adapter.target_R)
            self.yaw_goal = psi + np.pi * np.round((cur - psi) / np.pi)
        self.count += 1
        move, yaw = np.zeros(3), 0.0
        if self.phase == "approach":
            goal = handle + [0.0, 0.0, self.hover]
            move, yaw = self._move(goal, self.approach_speed), self._yaw_action()
            if (np.linalg.norm(goal - tcp) < 0.006 and abs(yaw) * env.adapter.max_rot < 0.03):
                self._next("descend")
        elif self.phase == "descend":
            goal = handle + axis * self.lead
            move, yaw = self._move(goal, self.descend_speed), self._yaw_action()
            if np.linalg.norm(goal - tcp) < 0.002 or self.count > 60:
                self._next("grip")
        elif self.phase == "grip":
            self.gripper = 1.0           # (the hand holds still: the fingers centre the bar)
            if self.count >= self.grip_steps:
                self.grip_point = self.env.adapter.target_pos.copy()
                self.lid0, self.s, self.wait = machine.lid, 0.0, 0
                cfg = machine.cfg
                if env.command == "open":
                    target = min(env.needed + self.open_past, cfg.travel - 0.002)
                    self.s_end, self.sign = target - self.lid0, 1.0
                    self.speed = self.open_speed
                else:
                    self.s_end, self.sign = self.lid0 + self.push_past, -1.0
                    self.speed = float(np.clip(1.6 * cfg.latch_speed, *self.close_speed))
                self._next("slide")
        elif self.phase == "slide":
            self.s = min(self.s + self.speed * env.adapter.control_dt, self.s_end)
            move = self._move(self.grip_point + axis * self.sign * self.s, 1.0, measured=False)
            if self.s >= self.s_end:
                self.wait += 1
                if env.goal_reached() and (env.command == "close" or self.wait > 3):
                    self._next("release")
                elif self.wait > self.settle_steps:
                    self.wait = 0
                    furthest = machine.cfg.travel - 0.001 - self.lid0
                    if env.command == "open" and self.s_end < furthest:
                        self.s_end = min(self.s_end + 0.005, furthest)   # reach further
                    elif env.command == "open":
                        self._next("release")
                    elif self.tries < self.retries:   # not latched: pull back, push again
                        self.tries += 1
                        self.speed = min(self.speed * 1.5, self.close_speed[1])
                        self._next("back")
                    else:
                        self._next("release")
        elif self.phase == "back":
            self.s = max(self.s - 0.05 * env.adapter.control_dt, self.lid0 - 0.02)
            move = self._move(self.grip_point + axis * self.sign * self.s, 1.0, measured=False)
            if self.s <= self.lid0 - 0.02:
                self.s_end, self.wait = self.lid0 + self.push_past, 0
                self._next("slide")
        elif self.phase == "release":
            self.gripper = -1.0          # (the hand's target stays where it is)
            if self.count >= self.grip_steps:
                self.release_point = self.env.adapter.target_pos.copy()
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
            "phase": expert.phase}
