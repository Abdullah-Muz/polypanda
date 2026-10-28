"""The scripted stand-up (v3 step d2) through StandUpEnv's 6-D hand-motion actions.

The hand-written comparison for the learned stand-up (evaluation.standup), and the source of the
demonstration-state starts of its training (StandUpConfig.start.demo_prob): the cell's scripted
stand-up (control.scripted_cell.ScriptedCell.stand_up), with a planner that keeps the arm clear
of the tower.
"""

import mujoco
import numpy as np

from polypanda.control.ik import orientation_error
from polypanda.environments.standup import StandUpEnv, _hand_R, _quat
from polypanda.simulation.panda import tcp_position
from polypanda.simulation.scenes import cube_position, cube_tilt


class ScriptedStandUpDriver:
    """The scripted stand-up through 6-D hand motions: over the spot, turn the hand sideways at
    carry height (the cube's long axis turns vertical), lower the cube until the gate lets go,
    back the hand out along its own axis and rise."""

    pre, back, rise = 0.18, 0.07, 0.12   # m: turn this high above the release; back out; rise
    pos_tol, rot_tol = 0.02, 0.05        # m, rad: reached a waypoint (the turned arm sags ~1 cm)
    # The cell's grasp assist is a soft weld: in a quick turn the cube lags the hand by up to
    # ~25 deg and creeps back over a few seconds (measured), so wait above the spot until it has
    settle = 0.6                         # x gate.max_tilt

    directions = 16     # candidate horizontal directions for the turned hand
    clearance = 0.03    # m, arm to tower along the way down

    def plan(self, env: StandUpEnv, direction: float):
        """The turned-hand poses for one direction of the hand's z axis (rad, world): turn at
        carry height, then down to the release. Returns (joint targets of the poses, or None
        if one is out of reach, stopping at the first; hand rotation; release TCP point)."""
        d, arm = env.data, env.arm
        tcp = tcp_position(d, arm)
        R = d.site_xmat[arm.tcp_site].reshape(3, 3)
        p_rel = R.T @ (cube_position(d, env.cube) - tcp)          # the cube in the hand
        q = d.qpos[arm.arm_qpos].copy()
        R_goal = _hand_R(direction)
        goal = env._release_point() - R_goal @ p_rel
        high = np.array([goal[0], goal[1], tcp[2]])              # turn at carry height first
        poses = [high] + [goal + [0.0, 0.0, self.pre * f] for f in (1.0, 0.66, 0.33, 0.0)]
        qs = []
        for p in poses:
            qk, ok = env._ik.solve(q, p, _quat(R_goal))
            if not ok:
                return None, R_goal, goal
            qs.append(qk)
        return qs, R_goal, goal

    def candidates(self) -> np.ndarray:
        return np.linspace(-np.pi, np.pi, self.directions, endpoint=False)

    def reset(self, env: StandUpEnv, direction: float | None = None) -> None:
        """Plan with privileged knowledge: of the reachable turned-hand poses whose arm stays
        `clearance` from the tower from the turn down to the release, the one needing the least
        joint motion (the cell's scripted stand-up uses fixed directions and could hit a
        tower with the forearm; measured). With `direction`, that direction as commanded
        (StandUpV3Env's demonstration starts), clear of the tower or not."""
        d, arm = env.data, env.arm
        tcp = tcp_position(d, arm)
        q = d.qpos[arm.arm_qpos].copy()
        best = None
        for cand in (self.candidates() if direction is None else [direction]):
            qs, R_goal, goal = self.plan(env, cand)
            if qs is None:
                continue
            if direction is None and min(self._tower_gap(env, qk) for qk in qs) < self.clearance:
                continue
            cost = float(np.abs(qs[0] - q).sum())
            if best is None or cost < best[0]:
                best = (cost, R_goal, goal)
        if best is None:              # nothing clear: the commanded direction, else the one
            a = float(np.arctan2(env.spot[1], env.spot[0]))        # along the station circle
            _, R_goal, goal = self.plan(env, a + np.pi / 2 if direction is None else direction)
            best = (0.0, R_goal, goal)
        _, R_goal, goal = best
        self.quat, self.goal = _quat(R_goal), goal
        self.z = R_goal[:, 2]
        self.waypoints = [np.array([goal[0], goal[1], tcp[2]]), goal + [0.0, 0.0, self.pre], goal]
        self.after = [goal - self.back * self.z, goal - self.back * self.z + [0, 0, self.rise]]
        self.k, self.backing = 0, False

    @staticmethod
    def _tower_gap(env: StandUpEnv, q: np.ndarray) -> float:
        """Smallest distance (m) from the arm's geoms to the tower with the arm at q."""
        if env.tower is None:
            return np.inf
        m = env.model
        if not hasattr(env, "_scratch"):
            env._scratch = mujoco.MjData(m)
        s = env._scratch
        s.qpos[:] = env.data.qpos
        s.qpos[env.arm.arm_qpos] = q
        mujoco.mj_kinematics(m, s)
        mujoco.mj_collision(m, s)   # (updates broad-phase data; distances below are exact)
        gap, fromto = np.inf, np.zeros(6)
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in env.arm_bodies and m.geom_contype[g]:
                gap = min(gap, mujoco.mj_geomDistance(m, s, g, env.tower_geom, 0.2, fromto))
        return float(gap)

    def __call__(self, env: StandUpEnv, obs: np.ndarray) -> np.ndarray:
        d, arm, cart = env.data, env.arm, env.config.action.cartesian
        tcp = tcp_position(d, arm)
        R = d.site_xmat[arm.tcp_site].reshape(3, 3)
        rot = orientation_error(R, self.quat)
        if env.released and not self.backing:
            self.backing, self.k = True, 0
        path = self.after if self.backing else self.waypoints
        settled = cube_tilt(d, env.cube) < self.settle * env.config.gate.max_tilt
        reached = (np.linalg.norm(path[self.k] - tcp) < self.pos_tol
                   and np.linalg.norm(rot) < self.rot_tol and (self.backing or settled))
        if reached and self.k < len(path) - 1:   # waypoints are passed once, never revisited
            self.k += 1
        target = path[self.k]
        a = np.zeros(6, dtype=np.float32)
        a[:3] = np.clip((target - tcp) / cart.max_step, -1, 1)
        a[3:] = np.clip(rot / cart.max_rot, -1, 1)
        return a
