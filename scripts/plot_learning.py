"""Overlay validation learning curves of several runs.

Only compare runs of the same task version: v1 counted success at any moment before termination,
v2 counts success still held at the end of the episode.

Usage: uv run python scripts/plot_learning.py runs/reach_ppo_v2_seed0 runs/reach_ppo_v2_seed1 ...
       [--labels A B ...] [--out docs/figures/ppo_reach_learning.png]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--out", type=Path, default=Path("docs/figures/ppo_reach_learning.png"))
    args = parser.parse_args()
    labels = args.labels or [r.name for r in args.runs]
    if len(labels) != len(args.runs):
        raise SystemExit("--labels must match the number of runs")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for run, label in zip(args.runs, labels, strict=True):
        ev = np.load(run / "eval" / "evaluations.npz")
        meta = yaml.safe_load((run / "metadata.yaml").read_text())
        ax.plot(ev["timesteps"], ev["successes"].mean(axis=1) * 100, marker="o", ms=4,
                label=f"{label} ({meta['wall_clock_s'] / 60:.1f} min)")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("validation success (%)")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("PPO on reach v2: validation success during training\n"
                 "(50 targets per point; success = reached and still held at 10 s)", fontsize=10)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"plot saved to {args.out}")


if __name__ == "__main__":
    main()
