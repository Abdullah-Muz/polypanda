"""Plot tower success by tower height in the work-order cell, scripted vs learned skills.

Usage: uv run python scripts/plot_cell_heights.py [--results results/cell] [--out ...]
Reads every results/cell/*.json written by scripts/evaluate_cell.py. Runs of the same
configuration (scripted or learned pick x scripted or learned place) are pooled: over seeds,
and over height ranges, so a 1-4 main evaluation and a 5-6 stress test form one series. Bars
show the share of towers built correctly at each height with 95 % Wilson intervals.
Writes docs/figures/cell_heights.png.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from polypanda.evaluation.reach import wilson_interval  # noqa: E402

SERIES = {  # (learned pick, learned place) -> label, colour
    (False, False): ("Scripted cell (baseline)", "0.6"),
    (True, False): ("Learned pick, scripted place", plt.cm.tab10(9)),
    (False, True): ("Scripted pick, learned place", plt.cm.tab10(1)),
    (True, True): ("Fully learned skills", plt.cm.tab10(0)),
}
MAIN_MAX_HEIGHT = 4  # heights 1-4: main evaluation; above: stress test


def load(results: Path) -> dict:
    """(learned pick, learned place) -> {"runs": set, "orders": int, "heights": {h: [ok, n]}}"""
    pooled = defaultdict(lambda: {"runs": set(), "orders": 0,
                                  "heights": defaultdict(lambda: [0, 0])})
    for path in sorted(results.glob("*.json")):
        record = json.loads(path.read_text())
        skills, summary = record["skills"], record["summary"]
        key = (bool(skills["pick_provenance"]), bool(skills["place_provenance"]))
        entry = pooled[key]
        entry["runs"].add((skills["pick_provenance"].get("policy_id"),
                           skills["place_provenance"].get("policy_id")))
        entry["orders"] += summary["orders"]
        for h, v in summary["tower_success_by_height"].items():
            entry["heights"][int(h)][0] += round(v["success_rate"] * v["towers"])
            entry["heights"][int(h)][1] += v["towers"]
    return pooled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/cell"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures/cell_heights.png"))
    args = parser.parse_args()
    pooled = load(args.results)
    if not pooled:
        raise SystemExit(f"no results in {args.results}")
    keys = [k for k in SERIES if k in pooled]
    heights = sorted({h for k in keys for h in pooled[k]["heights"]})
    width = 0.8 / len(keys)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, key in enumerate(keys):
        label, colour = SERIES[key]
        entry = pooled[key]
        seeds = len(entry["runs"])
        xs, ps, lo, hi = [], [], [], []
        for j, h in enumerate(heights):
            ok, n = entry["heights"].get(h, (0, 0))
            if not n:
                continue
            a, b = wilson_interval(ok, n)
            xs.append(j + (i - (len(keys) - 1) / 2) * width)
            ps.append(ok / n)
            lo.append(ok / n - a)
            hi.append(b - ok / n)
            print(f"{label}, height {h}: {ok}/{n} = {ok / n:.1%} (95% CI {a:.1%}-{b:.1%})")
        ax.bar(xs, ps, width * 0.9, yerr=[lo, hi], capsize=3, color=colour, ecolor="0.25",
               label=label + (f" ({seeds} seeds pooled)" if seeds > 1 else ""))
    stress = [j for j, h in enumerate(heights) if h > MAIN_MAX_HEIGHT]
    if stress:
        ax.axvspan(stress[0] - 0.5, stress[-1] + 0.5, color="0.93", zorder=0)
        ax.text((stress[0] + stress[-1]) / 2, 1.03, "stress test", ha="center", fontsize=8,
                color="0.35")
    ax.set_xticks(np.arange(len(heights)), [f"{h}" for h in heights])
    ax.set_xlim(-0.5, len(heights) - 0.5)
    ax.set_xlabel("tower height (cubes)")
    ax.set_ylabel("towers built correctly")
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Work-order cell: tower success by height (95 % Wilson intervals)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"plot saved to {args.out}")


if __name__ == "__main__":
    main()
