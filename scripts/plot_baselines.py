"""Plot TCP distance over time for the random and Jacobian-IK baselines on a few targets.

Usage: uv run python scripts/plot_baselines.py [--seeds 1000 1001 1002 1003 1004]
Writes docs/figures/baseline_distance.png. Episodes run the full horizon (termination ignored)
so the curves also show how each controller holds the target.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from polypanda.control.classical_baseline import JacobianIKPolicy, RandomPolicy  # noqa: E402
from polypanda.environments.reach import ReachEnv  # noqa: E402


def rollout(env: ReachEnv, policy, seed: int) -> np.ndarray:
    obs, info = env.reset(seed=seed)
    distances = [info["distance"]]
    for _ in range(env.config.episode.horizon):
        obs, _, _, _, info = env.step(policy.predict(obs)[0])
        distances.append(info["distance"])
    return np.array(distances)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003, 1004])
    parser.add_argument("--out", type=Path, default=Path("docs/figures/baseline_distance.png"))
    args = parser.parse_args()

    env = ReachEnv()
    dt = env.adapter.control_dt
    t = np.arange(env.config.episode.horizon + 1) * dt
    ik, rand = JacobianIKPolicy(), RandomPolicy(seed=0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, seed in enumerate(args.seeds):
        color = plt.cm.tab10(i)
        d_ik, d_rand = rollout(env, ik, seed), rollout(env, rand, seed)
        ax.plot(t, d_ik * 100, color=color, lw=2, label=f"IK, seed {seed}")
        ax.plot(t, d_rand * 100, color=color, lw=1, ls="--", alpha=0.7)
        reached = np.argmax(d_ik < env.config.success.tolerance)
        print(f"seed {seed}: start {d_ik[0]:.3f} m | IK <2 cm at {reached * dt:.2f} s, "
              f"final {d_ik[-1] * 1e3:.3f} mm | random final {d_rand[-1]:.3f} m")
    ax.axhline(env.config.success.tolerance * 100, color="0.3", lw=1, ls=":",
               label="success tolerance (2 cm)")
    ax.plot([], [], color="0.4", ls="--", label="random policy (same seeds)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("TCP distance to target (cm)")
    ax.set_title("Reach baselines: Jacobian IK vs random")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"plot saved to {args.out}")


if __name__ == "__main__":
    main()
