"""Robustness suite: full pick-and-place under physical variation, disturbances and new scenes.

Every condition reuses held-out scenes and draws its perturbations per scene from a dedicated
SeedSequence stream, so results are reproducible and paired across grasp skills. Perturbations
are evaluation-only: no skill was trained or tuned on them.

- mass:     cube mass x U_log(0.5, 2)            (density change; same size)
- friction: contact friction x U(0.5, 1.5)       (cube, finger pads and table scaled together,
                                                  since MuJoCo combines friction with max())
- size:     cube dimensions x U(0.85, 1.15)      (constant density, so mass scales with volume)
- joints:   start pose offset U(+-0.1) rad per arm joint
- push:     horizontal push on the cube during transport, |F| ~ U(0, 2) N for 0.2 s
            (the grip holds about 2 N of friction, about 1 N of which carries the weight)
- combined: all of the above at once
- ood:      cube starts in a 5 cm band outside the placement region (out of distribution)

The grasp suite (`run_grasp_condition`, v3 any-pose grasp) runs the grasp alone on held-out
grasp sets, lying or upright, under mass, friction, size, joints and combined. Push acts during
transport, which a grasp episode has none of (combined still draws it, unused, so its stream
matches); ood is left out because the any-pose region already reaches the arm's reach limit.
"""

import time
from dataclasses import asdict, dataclass

import numpy as np

from polypanda.evaluation.grasp import (
    GraspPolicyEpisode,
    evaluate_grasp_policy,
)
from polypanda.evaluation.pickplace import (
    PickPlaceEpisode,
    episode_from_result,
    summarize_pickplace,
)
from polypanda.evaluation.reach import wilson_interval
from polypanda.evaluation.sets import GraspEvalSet, PickPlaceEvalSet
from polypanda.simulation.randomization import CubePhysics
from polypanda.simulation.scenes import CUBE_X_RANGE, CUBE_Y_RANGE
from polypanda.supervisor.fsm import PickPlaceSupervisor, Stage, SupervisorConfig

ROBUSTNESS_SPAWN_KEY = 0x8B0D
CONDITIONS = ("mass", "friction", "size", "joints", "push", "combined", "ood")
GRASP_ROBUSTNESS_SPAWN_KEY = 0x8B0E
GRASP_CONDITIONS = ("mass", "friction", "size", "joints", "combined")
PUSH_START, PUSH_STEPS = (5, 15), 4    # transport policy steps; transport needs >= 30 steps
OOD_BAND = 0.05                        # m outside the placement region


@dataclass(frozen=True)
class Perturbation:
    mass_scale: float = 1.0
    friction_scale: float = 1.0
    size_scale: float = 1.0
    arm_offset: tuple[float, ...] = (0.0,) * 7
    push_force_n: float = 0.0          # horizontal magnitude
    push_angle: float = 0.0            # rad, direction in the table plane
    push_step: int = 0                 # transport step the push starts at


def sample_perturbation(rng: np.random.Generator, condition: str) -> Perturbation:
    def draw(name):
        return condition in (name, "combined")
    return Perturbation(
        mass_scale=float(np.exp(rng.uniform(np.log(0.5), np.log(2.0)))) if draw("mass") else 1.0,
        friction_scale=float(rng.uniform(0.5, 1.5)) if draw("friction") else 1.0,
        size_scale=float(rng.uniform(0.85, 1.15)) if draw("size") else 1.0,
        arm_offset=tuple(rng.uniform(-0.1, 0.1, 7).tolist()) if draw("joints") else (0.0,) * 7,
        push_force_n=float(rng.uniform(0.0, 2.0)) if draw("push") else 0.0,
        push_angle=float(rng.uniform(-np.pi, np.pi)) if draw("push") else 0.0,
        push_step=int(rng.integers(*PUSH_START)) if draw("push") else 0,
    )


def ood_scenes(eval_set: PickPlaceEvalSet, rng: np.random.Generator) -> np.ndarray:
    """Cube start poses in a band of OOD_BAND around the placement region (targets unchanged)."""
    (x0, x1), (y0, y1), b = CUBE_X_RANGE, CUBE_Y_RANGE, OOD_BAND
    poses = []
    while len(poses) < len(eval_set):
        x, y = rng.uniform(x0 - b, x1 + b), rng.uniform(y0 - b, y1 + b)
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            poses.append([x, y, rng.uniform(-np.pi, np.pi)])
    return np.array(poses)


class ScenePhysics(CubePhysics):
    """Applies a Perturbation's model changes to one environment, always relative to nominal."""

    def __init__(self, env) -> None:
        super().__init__(env.model, env.cube)

    def apply(self, p: Perturbation) -> None:  # type: ignore[override]
        super().apply(p.mass_scale, p.friction_scale, p.size_scale)


@dataclass(frozen=True)
class RobustEpisode:
    condition: str
    perturbation: str               # JSON-ish summary of the drawn parameters
    mass_scale: float
    friction_scale: float
    size_scale: float
    arm_offset_max: float
    push_force_n: float
    cube_x: float
    cube_y: float
    episode: PickPlaceEpisode


def run_condition(grasp_driver, transport_driver, place_driver, eval_set: PickPlaceEvalSet,
                  condition: str, seed: int, config: SupervisorConfig | None = None,
                  grasp_env=None) -> tuple[dict, list[RobustEpisode]]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    supervisor = PickPlaceSupervisor(grasp_driver, transport_driver, place_driver, config,
                                     grasp_env)
    physics = [ScenePhysics(env) for env in supervisor.envs.values()]
    rng = np.random.default_rng(np.random.SeedSequence(
        seed, spawn_key=(ROBUSTNESS_SPAWN_KEY, CONDITIONS.index(condition))))
    placements = ood_scenes(eval_set, rng) if condition == "ood" else eval_set.placements
    perturbations = [sample_perturbation(rng, condition) for _ in range(len(eval_set))]

    push = {"p": None, "stage": None, "k": 0}

    def on_step(stage, env):
        if stage != push["stage"]:
            push.update(stage=stage, k=0)
        push["k"] += 1
        p = push["p"]
        active = (stage == Stage.TRANSPORT and p.push_force_n > 0
                  and p.push_step <= push["k"] < p.push_step + PUSH_STEPS)
        force = p.push_force_n * np.array([np.cos(p.push_angle), np.sin(p.push_angle), 0.0])
        env.data.xfrc_applied[env.cube.body, :3] = force if active else 0.0
    supervisor.on_step = on_step

    start, results = time.perf_counter(), []
    for i, (pose, target, p) in enumerate(zip(placements, eval_set.targets, perturbations,
                                              strict=True)):
        for ph in physics:
            ph.apply(p)
        push.update(p=p, stage=None, k=0)
        r = supervisor.run(pose, target, grasp_options={"arm_offset": p.arm_offset})
        results.append(RobustEpisode(
            condition=condition,
            perturbation=str({k: v for k, v in asdict(p).items() if k != "arm_offset"}),
            mass_scale=p.mass_scale, friction_scale=p.friction_scale, size_scale=p.size_scale,
            arm_offset_max=float(np.abs(p.arm_offset).max()), push_force_n=p.push_force_n,
            cube_x=float(pose[0]), cube_y=float(pose[1]),
            episode=episode_from_result(i, r, supervisor.dt)))
    for ph in physics:
        ph.apply(Perturbation())  # leave the models nominal
    summary = summarize_pickplace([r.episode for r in results])
    summary["by_level"] = by_level(results, condition)
    summary["wall_clock_s"] = round(time.perf_counter() - start, 1)
    return summary, results


@dataclass(frozen=True)
class GraspRobustEpisode:
    condition: str
    mass_scale: float
    friction_scale: float
    size_scale: float
    arm_offset_max: float
    episode: GraspPolicyEpisode


def run_grasp_condition(driver, eval_set: GraspEvalSet, condition: str, seed: int, env
                        ) -> tuple[dict, list[GraspRobustEpisode]]:
    """The grasp alone on a held-out grasp set (lying or upright) under one condition; the same
    perturbation stream for every skill and both sets (paired)."""
    if condition not in GRASP_CONDITIONS:
        raise ValueError(f"unknown grasp condition {condition!r}")
    rng = np.random.default_rng(np.random.SeedSequence(
        seed, spawn_key=(GRASP_ROBUSTNESS_SPAWN_KEY, GRASP_CONDITIONS.index(condition))))
    perturbations = [sample_perturbation(rng, condition) for _ in range(len(eval_set))]
    physics = ScenePhysics(env)

    def before_episode(i):
        physics.apply(perturbations[i])
        return {"arm_offset": perturbations[i].arm_offset}

    try:
        summary, episodes = evaluate_grasp_policy(driver, eval_set, env=env,
                                                  before_episode=before_episode)
    finally:
        physics.apply(Perturbation())  # leave the model nominal
    results = [GraspRobustEpisode(condition=condition, mass_scale=p.mass_scale,
                                  friction_scale=p.friction_scale, size_scale=p.size_scale,
                                  arm_offset_max=float(np.abs(p.arm_offset).max()), episode=e)
               for p, e in zip(perturbations, episodes, strict=True)]
    summary["by_level"] = by_level(results, condition)
    return summary, results


LEVELS = {  # variable and bin edges reported per condition
    "mass": ("mass_scale", [0.5, 0.71, 1.0, 1.41, 2.0]),
    "friction": ("friction_scale", [0.5, 0.75, 1.0, 1.25, 1.5]),
    "size": ("size_scale", [0.85, 0.925, 1.0, 1.075, 1.15]),
    "joints": ("arm_offset_max", [0.0, 0.05, 0.075, 0.1]),
    "push": ("push_force_n", [0.0, 0.5, 1.0, 1.5, 2.0]),
}


def by_level(results: list[RobustEpisode], condition: str) -> dict:
    if condition not in LEVELS:
        return {}
    key, edges = LEVELS[condition]
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        group = [r for r in results if lo <= getattr(r, key) <= hi]
        k = sum(r.episode.success for r in group)
        out[f"{lo:g}-{hi:g}"] = {"n": len(group), "success_rate": k / len(group) if group else None,
                                 "success_ci95": list(wilson_interval(k, len(group)))}
    return out
