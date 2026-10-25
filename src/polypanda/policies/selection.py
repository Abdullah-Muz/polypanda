"""Checkpoint selection on a larger validation set.

Training keeps the "best" checkpoint by success on 30 validation episodes, which is noisy (±15
points): a lucky peak can be kept over a better policy. This module re-scores a run's saved
checkpoints (best, every intermediate checkpoint, final) on a separate, larger validation set,
never a test set, and records the winner as the run's "selected" checkpoint
(`load_run(run_dir, "selected")`), with every candidate's score in metadata.yaml.
"""

import re
import shutil
from pathlib import Path

import yaml

from polypanda.evaluation.grasp import PolicyDriver, run_policy_episode
from polypanda.evaluation.sets import GraspEvalSet
from polypanda.policies.loader import ALGORITHMS, LoadedPolicy, check_compatible, run_env_config


def candidates(run_dir: Path) -> list[tuple[str, Path, Path]]:
    """(label, model, normalization stats) for every saved checkpoint of a finished run."""
    meta = yaml.safe_load((run_dir / "metadata.yaml").read_text())
    out = [(which, run_dir / meta["checkpoints"][which], run_dir / meta["normalization"][which])
           for which in ("best", "final")]
    for model in sorted((run_dir / "checkpoints").glob("model_*_steps.zip")):
        steps = int(re.search(r"model_(\d+)_steps", model.name).group(1))
        if steps >= meta["total_timesteps"] - 10_000:
            continue  # the final model, already a candidate
        stats = model.with_name(f"model_vecnormalize_{steps}_steps.pkl")
        out.append((f"{steps // 1000}k", model, stats))
    return out


def load_candidate(run_dir: Path, model: Path, stats: Path) -> LoadedPolicy:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from polypanda.control.action_adapter import action_schema
    from polypanda.environments.tasks import TASKS
    meta = yaml.safe_load((run_dir / "metadata.yaml").read_text())
    config = run_env_config(run_dir, meta["skill"])
    check_compatible(meta, action_schema(config.action))
    policy = ALGORITHMS[meta["algorithm"]].load(model, device="cpu")
    env_cls = TASKS[meta["skill"]].env_cls
    norm = VecNormalize.load(str(stats), DummyVecEnv([lambda: env_cls(config)]))
    norm.training, norm.norm_reward = False, False
    return LoadedPolicy(policy, norm, meta, model.stem)


def score(env, policy: LoadedPolicy, val_set: GraspEvalSet) -> float:
    driver, wins = PolicyDriver(policy), 0
    for i, (placement, region) in enumerate(zip(val_set.placements, val_set.regions,
                                                strict=True)):
        others = None if val_set.distractors is None else val_set.distractors[i]
        result, _ = run_policy_episode(env, driver, placement, i, region, others)
        wins += result.success
    return wins / len(val_set)


def select(run_dir: Path, env, val_set: GraspEvalSet) -> dict:
    """Score every candidate, record the winner as "selected" (ties: the later checkpoint)."""
    scores = {label: score(env, load_candidate(run_dir, model, stats), val_set)
              for label, model, stats in candidates(run_dir)}
    def training(label: str) -> int:  # tie-break toward more training: 250k < ... < best < final
        return {"best": 10**9, "final": 10**9 + 1}.get(label) or int(label.rstrip("k"))
    winner = max(scores, key=lambda k: (scores[k], training(k)))
    label, model, stats = next(c for c in candidates(run_dir) if c[0] == winner)
    shutil.copyfile(model, run_dir / "selected_model.zip")
    shutil.copyfile(stats, run_dir / "selected_vecnormalize.pkl")
    meta_file = run_dir / "metadata.yaml"
    meta = yaml.safe_load(meta_file.read_text())
    meta["checkpoints"]["selected"] = "selected_model.zip"
    meta["normalization"]["selected"] = "selected_vecnormalize.pkl"
    meta["selection"] = {"validation_set": val_set.name, "digest": val_set.digest,
                         "episodes": len(val_set), "scores": scores, "selected": winner}
    meta_file.write_text(yaml.safe_dump(meta, sort_keys=False))
    return meta["selection"]
