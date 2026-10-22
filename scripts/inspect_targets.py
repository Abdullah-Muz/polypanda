"""Summarize and plot the reach target distribution.

Usage: uv run python scripts/inspect_targets.py [--n 2000] [--seed 0] [--out PATH]
Default output: docs/figures/reach_targets.png
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from polypanda.environments.targets import ReachTargetSampler  # noqa: E402
from polypanda.simulation.assets import load_panda_scene  # noqa: E402
from polypanda.simulation.panda import PandaIndices, reset_to_home, tcp_position  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("docs/figures/reach_targets.png"))
    args = parser.parse_args()

    model, data = load_panda_scene()
    idx = PandaIndices.from_model(model)
    reset_to_home(model, data, idx)
    home_tcp = tcp_position(data, idx)

    sampler = ReachTargetSampler(model, idx)
    rng = np.random.default_rng(args.seed)
    start = time.perf_counter()
    P = np.array([sampler.sample(rng).position for _ in range(args.n)])
    elapsed = time.perf_counter() - start

    print(f"{args.n} targets in {elapsed:.2f} s ({elapsed / args.n * 1e3:.2f} ms each), "
          f"acceptance {sampler.acceptance_rate:.1%}")
    for k, axis in enumerate("xyz"):
        q = np.percentile(P[:, k], [0, 5, 50, 95, 100])
        print(f"  {axis}: min {q[0]:+.3f}  p5 {q[1]:+.3f}  median {q[2]:+.3f}  "
              f"p95 {q[3]:+.3f}  max {q[4]:+.3f} m")
    dist = np.linalg.norm(P - home_tcp, axis=1)
    print(f"  distance from home TCP: median {np.median(dist):.3f} m, "
          f"p5 {np.percentile(dist, 5):.3f}, p95 {np.percentile(dist, 95):.3f}")
    bounds = list(zip(sampler.low, sampler.high, strict=True))
    H, _ = np.histogramdd(P, bins=(4, 4, 4), range=bounds)
    print(f"  density over 4x4x4 cells: min {H.min():.0f}, median {np.median(H):.0f}, "
          f"max {H.max():.0f} targets per cell")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (i, j), title in zip(axes, [(0, 1), (0, 2)], ["Top view (x-y)", "Side view (x-z)"],
                                 strict=True):
        ax.scatter(P[:, i], P[:, j], s=3, alpha=0.35, color="#3a7bd5", label="targets")
        lo, hi = sampler.low, sampler.high
        ax.add_patch(plt.Rectangle((lo[i], lo[j]), hi[i] - lo[i], hi[j] - lo[j],
                                   fill=False, ls="--", color="0.4", label="workspace box"))
        ax.scatter(0, 0, marker="s", s=60, color="0.2", label="robot base")
        ax.scatter(home_tcp[i], home_tcp[j], marker="*", s=160, color="#e4572e",
                   label="TCP at home")
        ax.set_xlabel(f"{'xyz'[i]} (m)")
        ax.set_ylabel(f"{'xyz'[j]} (m)")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Reach targets: {args.n} samples, seed {args.seed}")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"plot saved to {args.out}")


if __name__ == "__main__":
    main()
