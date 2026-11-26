"""v4.1 insertion round 2's gate before training (docs/v4_1_insertion_r1_rca.md): what simple
behaviours earn under an env config's reward, on the same training-stream episodes.

Behaviours (round 1's root-cause measurements, repeated):
- hover: hold still, never release;
- gauss: an untrained policy's exploration (actions drawn N(0, std), std per action);
- descend_hold: straight down at full speed until the module touches (> 2 N or within 0.5 mm of
  the floor), then hold still; descend_press: straight down at full speed for the whole episode;
- search: the frozen search script on its own control path (evaluation/insertion.run_script),
  scored step by step with the env's reward (InsertionEnv.judge) until the env would have ended
  the episode; search_replay: its recorded actions replayed through env.step (round 1's way; under
  round 2 the compliant layer acts on them).

Starts: "carry" (the targets' start) or "near" (a start height drawn per episode uniformly from
--near-low..--near-high m above the nest's rim; InsertionEnv reset option "start_height").

  uv run python scripts/insertion_reward_gate.py --config configs/insertion_v2.yaml --start carry
Prints a JSON summary per behaviour.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

SEED0 = 900_000
HOLD = np.array([0.0, 0.0, 0.0, 0.0, -1.0])
DOWN = np.array([0.0, 0.0, -1.0, 0.0, -1.0])
R1_NOISE = (0.1, 0.1, 0.1, 0.1, 0.3)          # round 1's initial log std -2.3 / -1.2
R2_NOISE = (0.2, 0.2, 0.2, 0.2, 0.37)         # round 2's -1.6 / -1.0


def make_env(config: str):
    import torch
    torch.set_num_threads(1)
    from polypanda.environments.insertion import InsertionConfig, InsertionEnv
    cfg = InsertionConfig.from_yaml(config)
    return InsertionEnv(replace(cfg, curriculum=replace(cfg.curriculum, enabled=False)))


def start_height(args, k: int) -> float | None:
    if args.start == "carry":
        return None
    return float(np.random.default_rng(1000 + k).uniform(args.near_low, args.near_high))


def reset(env, args, k: int):
    return env.reset(seed=SEED0 + k, options={"start_height": start_height(args, k)})


def rollout(env, act, args, k: int) -> dict:
    obs, _ = reset(env, args, k)
    terms: dict = defaultdict(float)
    total, t, info = 0.0, 0, {}
    first_release = first_contact = None
    min_gap, peak = 1.0, 0.0
    while True:
        obs, r, term, trunc, info = env.step(act(obs, t, env))
        total += r
        for name, v in info["reward_terms"].items():
            terms[name] += v
        min_gap = min(min_gap, env._prev["gap"])
        peak = max(peak, info["nest_force"])
        if first_contact is None and info["nest_force"] > 0.5:
            first_contact = t
        if first_release is None and info["released"]:
            first_release = t
        t += 1
        if term or trunc:
            break
    out = info.get("outcome") or env.outcome()
    return {"return": total, "steps": t, "success": bool(out["success"]),
            "seated": bool(out["seated"]), "failure": out["failure"], "terms": dict(terms),
            "first_release": first_release, "first_contact": first_contact,
            "min_gap_mm": min_gap * 1000, "peak_force": peak,
            "release_blocked": info["stats"].get("release_blocked", 0)}


def gaussian(std, seed: int):
    rng = np.random.default_rng(seed)
    return lambda obs, t, env: np.clip(rng.normal(0.0, std), -1.0, 1.0)


def descend_hold():
    state = {"touch": False}

    def act(obs, t, env):
        if t == 0:
            state["touch"] = False
        if env._step_peak > 2.0 or env._prev["gap"] < 0.0005:
            state["touch"] = True
        return HOLD if state["touch"] else DOWN
    return act


class _Scorer:
    """InsertionEnv.recorder scoring a script's own run with the env's reward."""

    def __init__(self, env) -> None:
        self.total, self.terms, self.done = 0.0, defaultdict(float), False
        self.actions, self.peak, self.steps = [], 0.0, 0
        self.first_contact = None

    def before_step(self, env) -> None:
        a = env.action_from_command()
        self.actions.append(a)
        env.adopt_command()
        self.released_before = env.released
        env._step_peak = 0.0

    def after_step(self, env) -> None:
        if self.done:
            return
        a = np.clip(self.actions[-1], -1.0, 1.0)
        terms, terminated, _, _ = env.judge(a, self.released_before)
        self.total += float(sum(terms.values()))
        for name, v in terms.items():
            self.terms[name] += v
        self.peak = max(self.peak, env._step_peak)
        if self.first_contact is None and env._step_peak > 0.5:
            self.first_contact = env.steps
        self.steps = env.steps
        self.done = terminated or env.steps >= env.config.horizon


def search_own(env, args, k: int) -> dict:
    from polypanda.evaluation.insertion import run_script
    reset(env, args, k)
    sc = _Scorer(env)
    env.recorder = sc
    try:
        out = run_script(env, "search")
    finally:
        env.recorder = None
    return {"return": sc.total, "steps": sc.steps, "success": bool(out["success"]),
            "seated": bool(out["seated"]), "failure": out["failure"], "terms": dict(sc.terms),
            "first_release": None, "first_contact": sc.first_contact, "min_gap_mm": None,
            "peak_force": sc.peak, "release_blocked": 0, "actions": sc.actions}


def job(spec) -> tuple:
    behaviour, args = spec
    env = make_env(args.config)
    rows = []
    for k in range(args.episodes):
        if behaviour == "hover":
            rows.append(rollout(env, lambda o, t, e: HOLD, args, k))
        elif behaviour.startswith("gauss"):
            std = np.array(R1_NOISE if behaviour == "gauss_r1" else R2_NOISE)
            rows.append(rollout(env, gaussian(std, 7 + k), args, k))
        elif behaviour == "descend_hold":
            rows.append(rollout(env, descend_hold(), args, k))
        elif behaviour == "descend_press":
            rows.append(rollout(env, lambda o, t, e: DOWN, args, k))
        elif behaviour == "search":
            own = search_own(env, args, k)
            acts = own.pop("actions")
            rows.append(own)
            rep = rollout(env, lambda o, t, e, acts=acts: acts[t] if t < len(acts) else HOLD,
                          args, k)
            rows.append({**rep, "_replay": True})
    return behaviour, rows


def summarize(rows: list[dict]) -> dict:
    def med(key):
        v = [r[key] for r in rows if r[key] is not None]
        return None if not v else round(float(np.median(v)), 2)
    terms: dict = defaultdict(list)
    for r in rows:
        for name in set(r["terms"]):
            terms[name].append(r["terms"][name])
    fails: dict = defaultdict(int)
    for r in rows:
        fails[str(r["failure"])] += 1
    n = len(rows)
    return {"n": n, "return_mean": round(float(np.mean([r["return"] for r in rows])), 2),
            "return_median": med("return"), "success": sum(r["success"] for r in rows),
            "seated": sum(r["seated"] for r in rows),
            "damage": sum(r["peak_force"] > 50.0 for r in rows),
            "contact_ever": sum(r["first_contact"] is not None for r in rows),
            "released_ever": sum(r["first_release"] is not None for r in rows),
            "release_blocked_mean": round(float(np.mean([r["release_blocked"] for r in rows])),
                                          1),
            "peak_force_median": med("peak_force"), "steps_median": med("steps"),
            "min_gap_mm_median": med("min_gap_mm"), "failures": dict(fails),
            "terms_mean": {k: round(float(np.sum(v) / n), 3) for k, v in sorted(terms.items())}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/insertion_v2.yaml")
    p.add_argument("--start", choices=("carry", "near"), default="carry")
    p.add_argument("--near-low", type=float, default=-0.003)
    p.add_argument("--near-high", type=float, default=0.02)
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--behaviours", nargs="+",
                   default=["hover", "gauss_r1", "gauss_r2", "descend_hold", "descend_press",
                            "search"])
    p.add_argument("--jobs", type=int, default=1)
    args = p.parse_args()
    specs = [(b, args) for b in args.behaviours]
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            results = pool.map(job, specs)
    else:
        results = [job(s) for s in specs]
    report = {"config": args.config, "start": args.start, "episodes": args.episodes}
    if args.start == "near":
        report["near_m"] = [args.near_low, args.near_high]
    for behaviour, rows in results:
        if behaviour == "search":
            report["search"] = summarize([r for r in rows if not r.get("_replay")])
            report["search_replay"] = summarize([r for r in rows if r.get("_replay")])
        else:
            report[behaviour] = summarize(rows)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
