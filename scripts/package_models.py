"""Package trained runs for a GitHub Release: policies, normalization statistics and provenance.

Usage: uv run python scripts/package_models.py --tag v2.0
Writes dist/polypanda-models-<tag>.tar.gz containing runs/<name>/ for every run listed in RUNS
(only what loading and auditing need: metadata, configs, best and final checkpoints with their
observation normalization, validation history and learning curves; no TensorBoard logs or
intermediate checkpoints), plus runs/MANIFEST.json with a SHA-256 per file. Runs in STOPPED were
stopped early, so they have no metadata or final checkpoint: only their configs and validation
history are packaged, as evidence for what the docs report about them. Prints the archive's
SHA-256 for scripts/fetch_models.sh.
"""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

# Every run behind a reported result (docs/results.md); pilots are included because their
# failures are reported too.
RUNS = [
    "reach_ppo_pilot_seed0", "reach_ppo_v2_seed0", "reach_ppo_v2_seed1", "reach_ppo_v2_seed2",
    "reach_sac_pilot_seed0", "reach_sac_v2_seed0", "reach_sac_v2_seed1", "reach_sac_v2_seed2",
    "grasp_ppo_pilot_seed0", "grasp_ppo_curr_seed0", "grasp_ppo_curr_seed1",
    "grasp_ppo_curr_seed2", "grasp_v2_seed0", "grasp_v2_seed1", "grasp_v2_seed2",
    "grasp_v2yaw_seed0", "grasp_v2yaw_seed1", "grasp_v2yaw_seed2",
    "transport_ppo_pilot_seed0", "transport_ppo_v2_seed0", "transport_residual_seed0",
    # v2.0: Milestones 7-9
    *(f"{name}_seed{s}" for name in (
        "grasp_wide", "grasp_v3", "grasp_cart", "transport_cart", "transport_slow",
        "place_cart", "place_stack", "place_stack_v2", "place_stack_v3") for s in range(3)),
    "transport_cart3d_nopenalty_seed0",
]
# Stopped early (docs/workcell.md: the three v3 attempts that did not learn)
STOPPED = [f"place_stack_v3{attempt}_seed{s}" for attempt in ("gate1cm", "check1cm", "centring")
           for s in range(3)]
FILES = ["metadata.yaml", "config.yaml", "env_config.yaml", "best_model.zip",
         "best_vecnormalize.pkl", "final_model.zip", "final_vecnormalize.pkl",
         "learning_curves.png", "eval/evaluations.npz"]
STOPPED_FILES = ["config.yaml", "env_config.yaml", "eval/evaluations.npz"]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="v2.0")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"polypanda-models-{args.tag}.tar.gz"
    manifest = {"tag": args.tag, "files": {}}
    with tarfile.open(out, "w:gz") as tar:
        for run, files in [(r, FILES) for r in RUNS] + [(r, STOPPED_FILES) for r in STOPPED]:
            run_dir = args.runs_dir / run
            if files is FILES and not (run_dir / "metadata.yaml").exists():
                raise SystemExit(f"{run_dir}: missing or unfinished run")
            if not (run_dir / "eval" / "evaluations.npz").exists():
                raise SystemExit(f"{run_dir}: no validation history")
            for name in files:
                path = run_dir / name
                if not path.exists():
                    continue
                data = path.read_bytes()
                arcname = f"runs/{run}/{name}"
                manifest["files"][arcname] = sha256(data)
                tar.add(path, arcname=arcname)
        blob = (json.dumps(manifest, indent=1) + "\n").encode()
        info = tarfile.TarInfo("runs/MANIFEST.json")
        info.size = len(blob)
        tar.addfile(info, io.BytesIO(blob))
    digest = sha256(out.read_bytes())
    print(f"wrote {out}: {len(RUNS)} runs and {len(STOPPED)} stopped, "
          f"{len(manifest['files'])} files, "
          f"{out.stat().st_size / 1e6:.1f} MB")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
