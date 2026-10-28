"""Evaluate the stand-up skill (v3 step d2) on held-out starts.

A StandUpEvalSet is a list of explicit StandUpEnv episodes (spot, hand yaw, grip, tower), all
reachable from the carry posture; drivers act through the env for its full horizon, as the grasp
comparison protocol does. ScriptedStandUpDriver is the hand-written comparison: the cell's
scripted stand-up (control.scripted_cell.ScriptedCell.stand_up) through the policy's own 6-D
hand-motion actions.
"""

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from polypanda.control.ik import orientation_error
from polypanda.environments.standup import StandUpEnv, _hand_R, _quat
from polypanda.evaluation.reach import wilson_interval
from polypanda.simulation.panda import tcp_position
from polypanda.simulation.scenes import cube_position, cube_tilt


@dataclass(frozen=True)
class StandUpEvalSet:
    name: str
    seed: int
    episodes: list[dict]

    def __len__(self) -> int:
        return len(self.episodes)

    @property
    def digest(self) -> str:
        raw = json.dumps(self.episodes, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()[:12]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"name": self.name, "seed": self.seed,
                                          "digest": self.digest,
                                          "episodes": self.episodes}, indent=1) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "StandUpEvalSet":
        raw = json.loads(Path(path).read_text())
        s = cls(raw["name"], raw["seed"], raw["episodes"])
        if s.digest != raw["digest"]:
            raise ValueError(f"{path}: digest mismatch ({s.digest} != {raw['digest']})")
        return s

    def subset(self, n: int) -> "StandUpEvalSet":
        return StandUpEvalSet(self.name, self.seed, self.episodes[:n])


def generate_standup_set(env: StandUpEnv, name: str, n: int, seed: int) -> StandUpEvalSet:
    """n reachable carry-posture starts drawn from the env's (evaluation) start distribution."""
    rng = np.random.default_rng(seed)
    episodes = []
    while len(episodes) < n:
        ep = env.draw_episode(rng)
        if env._start(ep):
            episodes.append(json.loads(json.dumps(ep)))  # plain lists and floats
    return StandUpEvalSet(name, seed, episodes)


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

    def reset(self, env: StandUpEnv) -> None:
        """Plan with privileged knowledge: of the reachable turned-hand poses whose arm stays
        `clearance` from the tower from the turn down to the release, the one needing the least
        joint motion (the cell's scripted stand-up uses fixed directions and could hit a
        tower with the forearm; measured)."""
        d, arm = env.data, env.arm
        tcp = tcp_position(d, arm)
        R = d.site_xmat[arm.tcp_site].reshape(3, 3)
        p_rel = R.T @ (cube_position(d, env.cube) - tcp)          # the cube in the hand
        rest = env._release_point()
        q = d.qpos[arm.arm_qpos].copy()
        best = None
        for direction in np.linspace(-np.pi, np.pi, self.directions, endpoint=False):
            R_goal = _hand_R(direction)
            goal = rest - R_goal @ p_rel
            high = np.array([goal[0], goal[1], tcp[2]])          # turn at carry height first
            poses = [high] + [goal + [0.0, 0.0, self.pre * f] for f in (1.0, 0.66, 0.33, 0.0)]
            qs = [env._ik.solve(q, p, _quat(R_goal)) for p in poses]
            if not all(ok for _, ok in qs):
                continue
            gap = min(self._tower_gap(env, qk) for qk, _ in qs)
            if gap < self.clearance:
                continue
            cost = float(np.abs(qs[0][0] - q).sum())
            if best is None or cost < best[0]:
                best = (cost, R_goal, goal)
        if best is None:              # nothing clear: the direction along the station circle
            a = float(np.arctan2(env.spot[1], env.spot[0]))
            R_goal = _hand_R(a + np.pi / 2)
            best = (0.0, R_goal, rest - R_goal @ p_rel)
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


@dataclass(frozen=True)
class StandUpEpisode:
    index: int
    spot_x: float
    spot_y: float
    tower_cubes: int
    success: bool                 # the placed check holds at the final step
    first_success_s: float | None
    released: bool
    release_s: float | None
    final_tilt_deg: float
    final_xy_cm: float
    episode_return: float
    failure: str | None  # toppled | tower_hit | never_released | not_stood


def run_standup_episode(env: StandUpEnv, driver, ep: dict, index: int
                        ) -> tuple[StandUpEpisode, list[float]]:
    obs, _ = env.reset(options={"episode": ep})
    driver.reset(env)
    dt = env.adapter.control_dt
    latencies, ret, first, release, info = [], 0.0, None, None, {}
    for _ in range(env.config.horizon):
        start = time.perf_counter()
        action = driver(env, obs)
        latencies.append(time.perf_counter() - start)
        obs, reward, terminated, truncated, info = env.step(action)
        ret += reward
        if first is None and info["is_success"]:
            first = env.elapsed_steps * dt
        if release is None and info["released"]:
            release = env.elapsed_steps * dt
        if terminated or truncated:
            break
    success = bool(info["is_success"])
    failure = (None if success else info["failure"] or
               ("not_stood" if info["released"] else "never_released"))
    tower = ep.get("tower")
    return StandUpEpisode(
        index=index, spot_x=float(ep["spot"][0]), spot_y=float(ep["spot"][1]),
        tower_cubes=int(tower[2]) if tower else 0, success=success, first_success_s=first,
        released=bool(info["released"]), release_s=release,
        final_tilt_deg=float(np.degrees(info["cube_tilt"])),
        final_xy_cm=float(info["xy_error"] * 100), episode_return=float(ret),
        failure=failure), latencies


def summarize_standup(results: list[StandUpEpisode], latencies: list[float]) -> dict:
    n = len(results)
    k = sum(r.success for r in results)
    times = [r.first_success_s for r in results if r.success]
    by_tower = {}
    for label, group in (("no tower", [r for r in results if not r.tower_cubes]),
                         ("tower", [r for r in results if r.tower_cubes])):
        if group:
            kg = sum(r.success for r in group)
            by_tower[label] = {"n": len(group), "success_rate": kg / len(group),
                               "success_ci95": list(wilson_interval(kg, len(group)))}
    return {
        "episodes": n,
        "success_rate": k / n,
        "success_ci95": list(wilson_interval(k, n)),
        "first_success_s_median": float(np.median(times)) if times else None,
        "release_rate": sum(r.released for r in results) / n,
        "failures": dict(Counter(r.failure for r in results if r.failure)),
        "by_tower": by_tower,
        "inference_latency_us_median": float(np.median(latencies) * 1e6) if latencies else None,
    }


def evaluate_standup(driver, eval_set: StandUpEvalSet, env: StandUpEnv
                     ) -> tuple[dict, list[StandUpEpisode]]:
    start = time.perf_counter()
    results, latencies = [], []
    for i, ep in enumerate(eval_set.episodes):
        result, lat = run_standup_episode(env, driver, ep, i)
        results.append(result)
        latencies.extend(lat)
    summary = summarize_standup(results, latencies)
    summary["wall_clock_s"] = round(time.perf_counter() - start, 1)
    return summary, results
