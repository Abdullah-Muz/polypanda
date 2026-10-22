"""Summarize a training run: validation curve, training diagnostics, learning-curve plot.

Usage: uv run python scripts/summarize_run.py runs/<run_dir> [--out PATH]
Default plot path: <run_dir>/learning_curves.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402


def scalars(run_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    acc = EventAccumulator(str(next((run_dir / "tb").rglob("events.out.tfevents.*"))),
                           size_guidance={"scalars": 0})
    acc.Reload()
    out = {}
    for tag in acc.Tags()["scalars"]:
        events = acc.Scalars(tag)
        out[tag] = (np.array([e.step for e in events]), np.array([e.value for e in events]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir
    meta = yaml.safe_load((run_dir / "metadata.yaml").read_text())
    ev = np.load(run_dir / "eval" / "evaluations.npz")
    steps, lengths = ev["timesteps"], ev["ep_lengths"].mean(axis=1)
    success = ev["successes"].mean(axis=1) if "successes" in ev else np.full(len(steps), np.nan)
    tb = scalars(run_dir)

    print(f"{meta['policy_id']}: {meta['algorithm']} seed {meta['seed']}, "
          f"{meta['total_timesteps']:,} steps in {meta['wall_clock_s']:.0f} s "
          f"({meta['steps_per_s']:.0f} steps/s), git dirty {meta['git']['dirty']}")
    print("validation (deterministic, 'eval_episodes' episodes each):")
    for s, sr, ln in zip(steps, success, lengths, strict=True):
        print(f"  {s:>9,d} steps: success {sr:6.1%}  mean episode length {ln:6.1f}")
    best = int(np.argmax(ev["results"].mean(axis=1)))
    print(f"best model (by mean validation return) at {steps[best]:,} steps")
    for tag in ("rollout/success_rate", "rollout/mean_distance_m", "train/std",
                "train/approx_kl", "train/explained_variance", "train/clip_fraction"):
        if tag in tb:
            print(f"  {tag:28s} first {tb[tag][1][0]:9.4f}  last {tb[tag][1][-1]:9.4f}")
    terms = [t for t in tb if t.startswith("reward_terms/")]
    for tag in terms:
        print(f"  {tag:28s} first {tb[tag][1][0]:9.4f}  last {tb[tag][1][-1]:9.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax = axes[0, 0]
    ax.plot(steps, success * 100, marker="o", label="validation (deterministic)")
    if "rollout/success_rate" in tb:
        x, y = tb["rollout/success_rate"]
        ax.plot(x, y * 100, alpha=0.6, label="training rollouts (stochastic)")
    ax.set_ylabel("success (%)")
    ax.set_ylim(-2, 102)
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.plot(steps, lengths * 0.05, marker="o")
    ax.set_ylabel("validation episode length (s)")
    ax = axes[1, 0]
    for tag in terms:
        ax.plot(*tb[tag], label=tag.split("/")[1])
    ax.set_ylabel("mean reward per step")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    if "train/std" in tb:
        ax.plot(*tb["train/std"], label="policy std (exploration)")
    if "rollout/mean_distance_m" in tb:
        ax.plot(*tb["rollout/mean_distance_m"], label="mean TCP distance (m)")
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("environment steps")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{meta['policy_id']} ({meta['algorithm']}, seed {meta['seed']})")
    fig.tight_layout()
    out = args.out or run_dir / "learning_curves.png"
    fig.savefig(out, dpi=100)
    print(f"plot saved to {out}")


if __name__ == "__main__":
    main()
