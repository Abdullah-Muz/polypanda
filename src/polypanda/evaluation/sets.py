"""Versioned held-out sets of reach targets.

Targets are drawn by the same ReachTargetSampler used in training, but from a random stream
derived with a dedicated SeedSequence spawn key. No `env.reset(seed=n)` can ever reproduce
that stream, so training and tuning seeds cannot leak evaluation targets.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from polypanda.environments.reach import OBS_SCHEMA, ReachConfig
from polypanda.environments.targets import ReachTargetSampler
from polypanda.simulation.assets import MENAGERIE_COMMIT, load_panda_scene
from polypanda.simulation.panda import PandaIndices, reset_to_home, tcp_position

EVAL_SPAWN_KEY = 0xE7A1  # separates evaluation streams from every plain integer seed
NEAR_FAR_RADIUS = 0.5    # m, horizontal distance from the base
LOW_HIGH_Z = 0.3         # m


def region_of(position: np.ndarray) -> str:
    reach = "near" if np.hypot(position[0], position[1]) < NEAR_FAR_RADIUS else "far"
    height = "low" if position[2] < LOW_HIGH_Z else "high"
    return f"{reach}-{height}"


@dataclass(frozen=True)
class ReachEvalSet:
    name: str
    seed: int
    targets: np.ndarray    # (n, 3) world-frame TCP targets
    witness_q: np.ndarray  # (n, 7) configurations reaching each target
    regions: tuple[str, ...]
    workspace: dict
    min_start_distance: float
    menagerie_commit: str = MENAGERIE_COMMIT
    obs_schema: str = OBS_SCHEMA

    def __len__(self) -> int:
        return len(self.targets)

    @property
    def digest(self) -> str:
        return hashlib.sha256(np.ascontiguousarray(self.targets).tobytes()).hexdigest()[:12]

    def subset(self, n: int) -> "ReachEvalSet":
        return ReachEvalSet(self.name, self.seed, self.targets[:n], self.witness_q[:n],
                            self.regions[:n], self.workspace, self.min_start_distance,
                            self.menagerie_commit, self.obs_schema)

    def save(self, path: str | Path) -> None:
        payload = {
            "name": self.name,
            "seed": self.seed,
            "spawn_key": EVAL_SPAWN_KEY,
            "n": len(self),
            "digest": self.digest,
            "menagerie_commit": self.menagerie_commit,
            "obs_schema": self.obs_schema,
            "workspace": self.workspace,
            "min_start_distance": self.min_start_distance,
            "start": "home keyframe, no joint noise",
            "targets": [
                {"position": p.tolist(), "witness_q": q.tolist(), "region": r}
                for p, q, r in zip(self.targets, self.witness_q, self.regions, strict=True)
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=1) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ReachEvalSet":
        payload = json.loads(Path(path).read_text())
        entries = payload["targets"]
        loaded = cls(
            name=payload["name"],
            seed=payload["seed"],
            targets=np.array([e["position"] for e in entries]),
            witness_q=np.array([e["witness_q"] for e in entries]),
            regions=tuple(e["region"] for e in entries),
            workspace=payload["workspace"],
            min_start_distance=payload["min_start_distance"],
            menagerie_commit=payload["menagerie_commit"],
            obs_schema=payload["obs_schema"],
        )
        if loaded.digest != payload["digest"]:
            raise ValueError(f"{path}: target digest mismatch (file edited or corrupted)")
        return loaded


def generate_reach_eval_set(name: str, n: int, seed: int,
                            config: ReachConfig | None = None) -> ReachEvalSet:
    config = config or ReachConfig()
    model, data = load_panda_scene()
    idx = PandaIndices.from_model(model)
    reset_to_home(model, data, idx)
    home_tcp = tcp_position(data, idx)
    sampler = ReachTargetSampler(model, idx, config.workspace)
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(EVAL_SPAWN_KEY,)))
    drawn = [sampler.sample(rng, avoid=home_tcp,
                            min_distance=config.episode.min_start_distance) for _ in range(n)]
    targets = np.array([t.position for t in drawn])
    return ReachEvalSet(
        name=name,
        seed=seed,
        targets=targets,
        witness_q=np.array([t.witness_q for t in drawn]),
        regions=tuple(region_of(p) for p in targets),
        workspace={k: list(v) if isinstance(v, tuple) else v
                   for k, v in asdict(config.workspace).items()},
        min_start_distance=config.episode.min_start_distance,
    )
