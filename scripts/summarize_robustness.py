"""Summarize the robustness suite: a markdown table per grasp skill and a success-vs-level plot.

Usage: uv run python scripts/summarize_robustness.py [--groups scripted grasp_ppo_curr grasp_v2]
Pools seeds within a group (e.g. grasp_ppo_curr_seed0..2 -> "grasp_ppo_curr"), reads
results/robustness/ and the nominal results in results/pickplace/, prints the table and writes
docs/figures/robustness.png.
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from polypanda.evaluation.reach import wilson_interval  # noqa: E402
from polypanda.evaluation.robustness import CONDITIONS, LEVELS  # noqa: E402

LABELS = {"scripted": "Scripted grasp", "grasp_ppo_curr": "PPO grasp v1 (3 seeds)",
          "grasp_v2": "PPO grasp v2 (3 seeds)"}
TITLES = {"mass": "Cube mass (x nominal)", "friction": "Contact friction (x nominal)",
          "size": "Cube size (x nominal)", "push": "Push during transport (N)"}


def group_of(tag: str) -> str:
    return re.sub(r"_seed\d+_(best|final)$", "", tag)


def read_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def load(results: Path, nominal: Path) -> dict:
    """{group: {condition: [rows]}} including 'nominal' from the pick-and-place results."""
    data = defaultdict(lambda: defaultdict(list))
    for f in sorted(results.glob("*_episodes.csv")):
        tag, condition = f.name[:-len("_episodes.csv")].rsplit("_", 1)
        data[group_of(tag)][condition] += read_rows(f)
    for f in sorted(nominal.glob("pickplace_*_pickplace_nominal_v1_episodes.csv")):
        tag = f.name[len("pickplace_"):-len("_pickplace_nominal_v1_episodes.csv")]
        data[group_of(tag)]["nominal"] += read_rows(f)
    return data


def rate(rows: list[dict]) -> tuple[float, float, float, int]:
    k, n = sum(r["success"] == "True" for r in rows), len(rows)
    lo, hi = wilson_interval(k, n)
    return k / n, lo, hi, n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", default=["scripted", "grasp_ppo_curr", "grasp_v2"])
    parser.add_argument("--results", type=Path, default=Path("results/robustness"))
    parser.add_argument("--nominal", type=Path, default=Path("results/pickplace"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures/robustness.png"))
    args = parser.parse_args()
    data = load(args.results, args.nominal)
    groups = [g for g in args.groups if g in data]

    print("| Condition | " + " | ".join(LABELS.get(g, g) for g in groups) + " |")
    print("|---|" + "---|" * len(groups))
    for condition in ("nominal", *CONDITIONS):
        cells = []
        for g in groups:
            rows = data[g].get(condition)
            if rows:
                p, lo, hi, n = rate(rows)
                cells.append(f"{p:.1%} ({lo:.1%}–{hi:.1%})")
            else:
                cells.append("—")
        print(f"| {condition} | " + " | ".join(cells) + " |")

    print("\nFinal failures by stage (all episodes in the group):")
    for g in groups:
        for condition in CONDITIONS:
            rows = data[g].get(condition, [])
            stages = defaultdict(int)
            for r in rows:
                if r["success"] != "True":
                    stages[r["failure_stage"]] += 1
            print(f"  {g:16s} {condition:9s} n={len(rows):4d} {dict(stages)}")

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for ax, condition in zip(axes, TITLES, strict=True):
        key, edges = LEVELS[condition]
        centres = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:], strict=True)]
        for g in groups:
            rows = data[g].get(condition, [])
            if not rows:
                continue
            ps, los, his = [], [], []
            for a, b in zip(edges[:-1], edges[1:], strict=True):
                sub = [r for r in rows if a <= float(r[key]) <= b]
                p, lo, hi, _ = rate(sub)
                ps.append(p)
                los.append(max(0.0, p - lo))  # Wilson bounds can round past p at 0 or 100 %
                his.append(max(0.0, hi - p))
            ax.errorbar(centres, ps, yerr=[los, his], marker="o", capsize=3,
                        label=LABELS.get(g, g))
        ax.set_title(TITLES[condition])
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Pick-and-place success")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
