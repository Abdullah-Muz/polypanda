"""Evaluate the stand-up skill (v3 step d2) on held-out starts.

A StandUpEvalSet is a list of explicit StandUpEnv episodes (spot, hand yaw, grip, tower), all
reachable from the carry posture; drivers act through the env for its full horizon, as the grasp
comparison protocol does. The hand-written comparison is control.scripted_standup's
ScriptedStandUpDriver (re-exported here).
"""

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from polypanda.control.scripted_standup import ScriptedStandUpDriver  # (also re-exported)
from polypanda.environments.standup import StandUpConfig, StandUpEnv
from polypanda.evaluation.reach import wilson_interval


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
    failure: str | None  # toppled | tower_hit | never_released | not_stood | wrong_turn
    command_deg: float | None = None  # v3: the turn direction the driver commanded (world)


class CommandedPolicyDriver:
    """Drives StandUpV3Env with a direction-commanded policy (d2 round 6). At each start it
    rates every reachable turn direction with the policy's own critic (the value the policy
    expects to earn under that command) and commands the best; then the policy acts.

    `choose`: "critic" (the skill as used); a training.turn_chooser.TurnChooser (the learned
    high level, rating each command's chance of success); for diagnostics, "scripted" (the
    privileged scripted planner's choice, clearing the tower) or a fixed world direction in
    radians."""

    def __init__(self, policy, choose="critic") -> None:
        self.policy, self.choose = policy, choose
        self.choice = None
        self.values = {}

    def reset(self, env) -> None:
        if hasattr(self.choose, "scores"):
            env._chooser = self.choose            # re-planning (ReplanConfig) chooses with it too
            cands = env.reachable_directions() or [env.command]
            obs = []
            for d in cands:
                env.set_command(d)
                obs.append(env.view_observation())
            self.values = dict(zip(cands, self.choose.scores(np.stack(obs)).tolist(),
                                   strict=True))
            d = max(self.values, key=self.values.get)
        elif self.choose == "scripted":
            planner = ScriptedStandUpDriver()
            planner.reset(env)
            d = float(np.arctan2(planner.z[1], planner.z[0]))
        elif self.choose == "critic":
            cands = env.reachable_directions() or [env.command]
            self.values = dict(zip(cands, self._values(env, cands), strict=True))
            d = max(self.values, key=self.values.get)
        else:
            d = float(self.choose)
        env.set_command(d)
        self.choice = float(np.degrees(d))

    def _values(self, env, cands: list[float]) -> list[float]:
        import torch
        obs = []
        for d in cands:
            env.set_command(d)
            obs.append(env.view_observation())
        obs = np.stack(obs)
        if self.policy.vec_normalize is not None:
            obs = self.policy.vec_normalize.normalize_obs(obs)
        with torch.no_grad():
            v = self.policy.model.policy.predict_values(torch.as_tensor(obs, dtype=torch.float32))
        return v.numpy().ravel().tolist()

    def __call__(self, env, obs: np.ndarray) -> np.ndarray:
        # the observation as of the command set in reset (the env's first one predates it), as
        # the policy sees it (at a workcell station: in the station frame)
        return self.policy.predict(env.view_observation(), deterministic=True)[0]


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
        failure=failure, command_deg=getattr(driver, "choice", None)), latencies


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


def standup_driver_from_spec(spec: str, choose: str = "critic",
                             task_config: str | Path = "configs/standup_v1.yaml"):
    """(driver, env, description, name prefix) for "scripted" (on task_config's task) or
    "run:DIR[@best|final]"; for v3 runs `choose` says who picks the turn direction: critic,
    chooser (the run's turn_chooser_<which>.pt), scripted, or a world angle in degrees."""
    from polypanda.evaluation.grasp import PolicyDriver
    from polypanda.policies.loader import load_run, make_run_env
    if spec == "scripted":
        task_config = Path(task_config)
        env = StandUpEnv(StandUpConfig.from_yaml(task_config).for_evaluation())
        tag = "" if task_config == Path("configs/standup_v1.yaml") else f"{task_config.stem}_"
        return (ScriptedStandUpDriver(), env,
                f"scripted stand-up (6-D hand motions), task {task_config}", f"scripted_{tag}")
    if not spec.startswith("run:"):
        raise ValueError("use scripted or run:DIR[@which]")
    run_dir, _, which = spec[4:].partition("@")
    policy = load_run(run_dir, which or "best")
    meta = policy.metadata
    if meta["skill"] not in ("standup", "standup_v2", "standup_v3"):
        raise ValueError(f"{run_dir} is a {meta['skill']} policy, not standup")
    env = make_run_env(run_dir, meta["skill"])
    description = (f"{meta['algorithm']} {policy.which} model from {run_dir} "
                   f"(seed {meta['seed']}, {meta['total_timesteps']:,} steps)")
    name = f"{meta['policy_id']}_{policy.which}_"
    if meta["skill"] != "standup_v3":
        return PolicyDriver(policy), env, description, name
    if choose == "chooser":
        from polypanda.training.turn_chooser import TurnChooser
        how = TurnChooser.load(Path(run_dir) / f"turn_chooser_{policy.which}.pt")
    elif choose in ("critic", "scripted"):
        how = choose
    else:
        how = np.radians(float(choose))
    description += f"; turn direction chosen by: {choose}"
    return CommandedPolicyDriver(policy, how), env, description, name


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
