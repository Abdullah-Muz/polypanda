"""v4.1: scripted insertions of a held part into a tight nest (docs/v4_1_plan.md, section 4.2).

The two baselines the learned insertion is compared with:

- **straight down** (`straight_down`): the cell's own place (`ScriptedCell.place`): aim at the
  believed nest, descend, release. Phase 0 measured that it tolerates about the nest's chamfer
  of aiming error (results/v4/nest_insertion_probe.json).
- **force-guided search** (`SearchInsertion`): the standard industrial method for tight fits
  (force-controlled search on cobots; the classical stage of connector insertion): descend
  slowly onto the believed nest until the part meets resistance and try it there; then keep a
  light downward press and move the hand on a growing spiral around the first contact; wherever
  the part starts to sink, stop and let it in, compliant sideways and in yaw (the hand gives way
  to the walls' push and twist, as a compliant wrist or an admittance-controlled arm does); then
  seat it and release. It reads only what a real cell can sense: the hand's commanded and
  measured positions and the force and torque on the held part (a wrist force-torque sensor);
  here they are summed from the contacts on the part.

Both start from the carry posture over the nest's station with the part held (the cell's grasp
assist), and finish with the fingers open and the hand back up.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from polypanda.control.ik import top_down_quat
from polypanda.control.scripted_cell import CUBE_HEIGHT, ScriptedCell
from polypanda.environments.grasp import GRIPPER_OPEN
from polypanda.simulation.scenes import CubeIndices


def part_force(model: mujoco.MjModel, data: mujoco.MjData, part: CubeIndices) -> float:
    """The magnitude of the summed contact force on the part's body from everything but the
    robot (what a wrist sensor would feel, less the part's weight)."""
    return float(np.linalg.norm(part_wrench(model, data, part)[:3]))


def part_wrench(model: mujoco.MjModel, data: mujoco.MjData, part: CubeIndices) -> np.ndarray:
    """The summed contact force (N) and torque about the part's origin (N m) on the part, from
    everything but the robot, world frame: [fx, fy, fz, tx, ty, tz]."""
    out, f6 = np.zeros(6), np.zeros(6)
    origin = data.xpos[part.body]
    hand_root = model.body_rootid[model.body("hand").id]
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if part.body not in (b1, b2):
            continue
        other = b2 if b1 == part.body else b1
        if model.body_rootid[other] == hand_root:
            continue                                     # the fingers holding it
        mujoco.mj_contactForce(model, data, i, f6)
        on2 = c.frame.reshape(3, 3).T @ f6[:3]           # on geom2, world frame
        f = on2 if b2 == part.body else -on2
        out[:3] += f
        out[3:] += np.cross(c.pos - origin, f)
    return out


def straight_down(ctl: ScriptedCell, part: CubeIndices, xy, level: int = 0) -> dict:
    """The cell's own place, aimed at `xy` (the believed nest)."""
    t0 = ctl.d.time
    ctl.place(part, xy, level)
    return {"method": "straight_down", "searched": False, "time_s": float(ctl.d.time - t0)}


@dataclass
class SearchInsertion:
    """Force-guided spiral search (see the module docstring). Distances in m, times in s.

    Force-regulated, as a compliant industrial search is: the hand's commanded height is
    adjusted every control step to hold about `press_force` on the part, so the part slides on
    the rim instead of being pinned to it by friction (a position-controlled press of a stiff arm
    pins it: measured, 20 N at 1.5 mm). Commands follow the arm's own commanded pose (forward
    kinematics of its joint targets), not the measured hand, which sags a little under load."""
    rim_z: float = 0.015            # the nest's rim height above the table (MachineConfig)
    hover: float = 0.004            # part bottom this far above the rim before the slow descent
    descend_speed: float = 0.01     # m/s, slow descent onto the nest
    contact_force: float = 3.0      # N: resistance that counts as touching down
    press_force: float = 3.0        # N held on the part while searching
    # m of height per N of force error, per control step: fast enough that an unsupported part
    # (over the opening) is lowered ~0.3 mm per step, below the stability limit of the measured
    # contact stiffness (~13 N/mm: gain x stiffness < 2)
    force_gain: float = 0.0001
    max_step: float = 0.0005        # m: the largest height correction in one step
    pitch: float = 0.0007           # m of radius gained per turn of the spiral
    # m/s along the spiral: 0.15 mm per 50 ms control step, a few steps across the opening
    search_speed: float = 0.003
    max_radius: float = 0.005       # m: give up beyond this from the first contact
    # The spiral stops as soon as the part sinks `sink` below its sliding height (it is over the
    # opening): a moving spiral drags a part that has just started in back out of the ~1 mm
    # capture window (measured: 0.2 s at 6 mm/s, too short for the arm)
    sink: float = 0.0003            # m below the first-contact height that counts as starting in
    release_force: float = 1.0      # N: ...with the rim no longer carrying the part
    # ...or the rim not carrying it for this many control steps in a row whatever the height (a
    # part turned a little from the nest hangs over the opening on the chamfer's edge, sinking
    # less than `sink`; on the flat rim the press holds ~3 N). The cell controls at 20 Hz, and
    # the spiral crosses the ~1 mm capture window in a few steps: stop at once
    unloaded_steps: int = 1
    retry_after: float = 0.001      # m along the spiral before another try
    # A try: hold the spot and press harder, compliant sideways and in yaw, for `engage_time`
    engage_force: float = 8.0       # N pressed while trying
    engage_time: float = 0.6        # s to get in before the spiral resumes (two wiggles)
    # lateral compliance: m per N of sideways force per control step (50 ms): ~0.2 mm/s at 3-4 N
    lateral_gain: float = 0.000003
    max_lateral_step: float = 0.00002
    # rotational wiggle while trying (the industrial answer to a part turned a little from the
    # nest: a twist sensed from friction-laden contacts is too weak and noisy to steer by)
    wiggle: float = 0.026           # rad (1.5 deg) either way
    wiggle_period: float = 0.3      # s (6 control steps)
    drop: float = 0.002             # m the part must sink below its stall height: it is in
    seat_force: float = 15.0        # N: the most the seating push may press with
    # s for the whole insertion, approach included. FROZEN with every parameter above as the
    # v4.1 search baseline (user, 2026-11-24): a generous budget so the baseline is competent;
    # its time is reported beside its success
    timeout: float = 30.0

    def _commanded(self, ctl: ScriptedCell) -> np.ndarray:
        """The hand position the arm's joint targets command (world frame)."""
        pos, _ = ctl.ik.forward(ctl.q_cmd)
        b = ctl.d.xpos[ctl.base_body]
        return np.array([pos[0] + b[0], pos[1] + b[1], pos[2]])

    def insert(self, ctl: ScriptedCell, part: CubeIndices, xy, yaw: float) -> dict:
        d, m = ctl.d, ctl.m
        t0 = d.time
        tcp = d.site_xpos[ctl.arm.tcp_site]
        half = CUBE_HEIGHT / 2
        wrench = np.zeros(6)
        state = {"yaw": float(yaw), "peak": 0.0, "tries": 0}

        def go(p) -> float:
            ctl.q_cmd = ctl._solve(p, top_down_quat(state["yaw"]), ctl.q_cmd)
            ctl.step()
            wrench[:] = part_wrench(m, d, part)
            f = float(np.linalg.norm(wrench[:3]))
            state["peak"] = max(state["peak"], f)
            return f

        def part_z() -> float:
            return float(d.xpos[part.body][2])

        def seated() -> bool:
            return part_z() - half < 0.0015

        def timed_out() -> bool:
            return d.time - t0 >= self.timeout

        def regulate(p, f: float, target: float) -> None:
            """Hold about `target` newtons down on the part."""
            p[2] += float(np.clip(self.force_gain * (f - target), -self.max_step, self.max_step))

        def comply(p) -> None:
            """Give way to the walls' sideways push on the part."""
            p[:2] += np.clip(self.lateral_gain * wrench[:2],
                             -self.max_lateral_step, self.max_lateral_step)

        def try_in(p, stall_z: float) -> bool:
            """Hold this spot, press, comply sideways and wiggle in yaw until the part is in
            (its yaw then stays where it went in) or the try times out (back to the believed
            yaw)."""
            state["tries"] += 1
            t_try = d.time
            while not timed_out() and d.time - t_try < self.engage_time:
                phase = 2 * np.pi * (d.time - t_try) / self.wiggle_period
                state["yaw"] = yaw + self.wiggle * float(np.sin(phase))
                f = go(p)
                regulate(p, f, self.engage_force)
                comply(p)
                if part_z() < stall_z - self.drop or seated():
                    return True
            state["yaw"] = yaw
            return False

        # 1. over the believed nest, the part's bottom `hover` above the rim (the cell's
        #    closed-loop vertical move lands the part, not the hand, on xy)
        offset = float(tcp[2] - part_z())                          # hand above part centre
        ctl.vertical(np.asarray(xy, float), self.rim_z + self.hover + half + offset, yaw,
                     track=part.body)
        ref = self._commanded(ctl)
        # 2. slow descent until it seats or meets resistance
        stall_z = None
        while not timed_out() and not seated():
            ref[2] -= self.descend_speed * ctl.dt
            f = go(ref)
            if f > self.contact_force and part_z() - half > self.rim_z - 0.002:
                stall_z = part_z()
                break
        searched, turns = stall_z is not None, 0.0
        dropped = not searched
        # 3. try where it touched down (the believed nest is the likeliest place); then the
        #    spiral, holding a light force, with a try wherever the part starts to sink
        if searched:
            dropped = try_in(ref, stall_z)
            centre, theta, path, next_try = ref[:2].copy(), 0.0, 0.0, self.retry_after
            unloaded = 0
            while not dropped and not timed_out():
                r = self.pitch * theta / (2 * np.pi)
                if r > self.max_radius:
                    break
                theta += self.search_speed * ctl.dt / max(r, self.pitch / 2)
                path += self.search_speed * ctl.dt
                ref[:2] = centre + r * np.array([np.cos(theta), np.sin(theta)])
                f = go(ref)
                regulate(ref, f, self.press_force)
                unloaded = unloaded + 1 if f < self.release_force else 0
                over = (part_z() < stall_z - self.sink and unloaded > 0) or \
                    unloaded >= self.unloaded_steps
                if over and path >= next_try:
                    unloaded = 0
                    dropped = try_in(ref, stall_z)
                    next_try = path + self.retry_after
            turns = theta / (2 * np.pi)
        # 4. seat it from where it dropped in, the push limited by force, compliant
        if dropped:
            while not timed_out() and not seated():
                f = go(ref)
                comply(ref)
                if f < self.seat_force:
                    ref[2] -= self.descend_speed * ctl.dt
        # 5. release and lift clear
        ctl.cell.detach(part)
        ctl.grip = GRIPPER_OPEN
        ctl.step(ctl.config.grip_steps)
        ctl.vertical(tcp[:2].copy(), ctl.config.carry_height, state["yaw"])
        return {"method": "search", "searched": searched, "dropped_in": dropped,
                "spiral_turns": round(turns, 2), "tries": state["tries"],
                "yaw_correction_deg": round(float(np.degrees(state["yaw"] - yaw)), 2),
                "peak_force_n": round(state["peak"], 2), "time_s": float(d.time - t0)}
