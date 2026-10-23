"""Versioned held-out sets of reach targets and grasp placements.

Reach targets are drawn by the same ReachTargetSampler used in training, and grasp placements
uniformly over the cube placement region, but always from a random stream derived with a
dedicated SeedSequence spawn key. No `env.reset(seed=n)` can ever reproduce such a stream, so
training and tuning seeds cannot leak evaluation cases.
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
from polypanda.simulation.scenes import CUBE_X_RANGE, CUBE_Y_RANGE

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


GRASP_SPAWN_KEY = 0x6A5B  # distinct from EVAL_SPAWN_KEY: grasp and reach streams never coincide


def grasp_region_of(xy: np.ndarray) -> str:
    return "near" if np.hypot(xy[0], xy[1]) < NEAR_FAR_RADIUS else "far"


@dataclass(frozen=True)
class GraspEvalSet:
    name: str
    seed: int
    placements: np.ndarray  # (n, 3): cube x, y (m) and yaw (rad), upright on the surface
    regions: tuple[str, ...]
    x_range: tuple[float, float] = CUBE_X_RANGE
    y_range: tuple[float, float] = CUBE_Y_RANGE
    menagerie_commit: str = MENAGERIE_COMMIT

    def __len__(self) -> int:
        return len(self.placements)

    @property
    def digest(self) -> str:
        return hashlib.sha256(np.ascontiguousarray(self.placements).tobytes()).hexdigest()[:12]

    def subset(self, n: int) -> "GraspEvalSet":
        return GraspEvalSet(self.name, self.seed, self.placements[:n], self.regions[:n],
                            self.x_range, self.y_range, self.menagerie_commit)

    def save(self, path: str | Path) -> None:
        payload = {
            "name": self.name, "seed": self.seed, "spawn_key": GRASP_SPAWN_KEY,
            "n": len(self), "digest": self.digest, "menagerie_commit": self.menagerie_commit,
            "x_range": list(self.x_range), "y_range": list(self.y_range),
            "start": "home keyframe, cube upright at rest",
            "placements": [{"x": p[0], "y": p[1], "yaw": p[2], "region": r}
                           for p, r in zip(self.placements.tolist(), self.regions, strict=True)],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=1) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "GraspEvalSet":
        payload = json.loads(Path(path).read_text())
        entries = payload["placements"]
        loaded = cls(
            name=payload["name"], seed=payload["seed"],
            placements=np.array([[e["x"], e["y"], e["yaw"]] for e in entries]),
            regions=tuple(e["region"] for e in entries),
            x_range=tuple(payload["x_range"]), y_range=tuple(payload["y_range"]),
            menagerie_commit=payload["menagerie_commit"],
        )
        if loaded.digest != payload["digest"]:
            raise ValueError(f"{path}: placement digest mismatch (file edited or corrupted)")
        return loaded


def generate_grasp_eval_set(name: str, n: int, seed: int) -> GraspEvalSet:
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(GRASP_SPAWN_KEY,)))
    placements = np.column_stack([rng.uniform(*CUBE_X_RANGE, n), rng.uniform(*CUBE_Y_RANGE, n),
                                  rng.uniform(-np.pi, np.pi, n)])
    return GraspEvalSet(name=name, seed=seed, placements=placements,
                        regions=tuple(grasp_region_of(p[:2]) for p in placements))
