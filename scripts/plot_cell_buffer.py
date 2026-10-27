"""Plot Layer 2 orders completed against the number of buffer slots.

Usage: uv run python scripts/plot_cell_buffer.py [--results results/cell] [--out ...]
Reads the Layer 2 (--mixed) results/cell/*.json written by scripts/evaluate_cell.py and pools
runs of the same configuration over seeds, as scripts/plot_cell_heights.py does. Points show the
share of orders completed with 95 % Wilson intervals. The dashed line is the share of the same
orders whose greedy plan never needs more than that many slots (`slots_needed`): with fewer
slots than that, the plan is not guaranteed to complete.
Writes docs/figures/cell_buffer.png.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from plot_cell_heights import SERIES, family  # noqa: E402

from polypanda.evaluation.reach import wilson_interval  # noqa: E402
from polypanda.supervisor.planner import mixed_order, slots_needed  # noqa: E402


def load(results: Path):
    """(learned pick, place family) -> {slots: [complete, orders]}, and one eval config."""
    pooled, config = defaultdict(lambda: defaultdict(lambda: [0, 0])), None
    for path in sorted(results.glob("*.json")):
        record = json.loads(path.read_text())
        if not record["eval"].get("mixed"):
            continue
        skills, summary, config = record["skills"], record["summary"], record["eval"]
        key = (bool(skills["pick_provenance"]), family(skills["place_provenance"]))
        entry = pooled[key][config["slots"]]
        entry[0] += round(summary["success_rate"] * summary["orders"])
        entry[1] += summary["orders"]
    return pooled, config


def plan_bound(config: dict, slots: list[int]) -> list[float]:
    """Share of the evaluation's orders whose greedy plan needs at most k slots."""
    needed = []
    for i in range(config["n"]):
        rng = np.random.default_rng([config["seed"], i])
        order, _ = mixed_order(rng, tuple(config["heights"]), tuple(config["surplus"]),
                               config["solvable_with"])
        needed.append(slots_needed(order.recipes, [c for c, _ in order.arrivals]))
    return [float(np.mean([n <= k for n in needed])) for k in slots]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/cell"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures/cell_buffer.png"))
    args = parser.parse_args()
    pooled, config = load(args.results)
    if not pooled:
        raise SystemExit(f"no Layer 2 results in {args.results}")
    slots = sorted({s for entry in pooled.values() for s in entry})

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bound = plan_bound(config, list(range(slots[-1] + 1)))
    ax.plot(range(slots[-1] + 1), bound, "--", color="0.5",
            label="orders whose plan needs at most this many slots")
    print("plan needs at most k slots: " + ", ".join(f"{k}: {b:.1%}" for k, b in enumerate(bound)))
    keys = [k for k in SERIES if k in pooled] + [k for k in pooled if k not in SERIES]
    for i, key in enumerate(keys):
        label, colour = SERIES.get(key, (f"pick learned, place {key[1]}", None))
        xs = sorted(pooled[key])
        ok = np.array([pooled[key][s][0] for s in xs])
        n = np.array([pooled[key][s][1] for s in xs])
        p = ok / n
        ci = np.array([wilson_interval(a, b) for a, b in zip(ok, n, strict=True)])
        x = np.array(xs) + (i - (len(keys) - 1) / 2) * 0.08
        ax.errorbar(x, p, yerr=[np.maximum(0, p - ci[:, 0]), np.maximum(0, ci[:, 1] - p)],
                    fmt="o", color=colour, capsize=3,
                    label=label + (f" ({n[0] // config['n']} seeds pooled)"
                                   if n[0] > config["n"] else ""))
        for s, a, b in zip(xs, ok, n, strict=True):
            print(f"{label}, {s} slots: {a}/{b} = {a / b:.1%}")
    ax.set_xticks(range(slots[-1] + 1))
    ax.set_xlabel("buffer slots the controller may use")
    ax.set_ylabel("orders completed")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Work-order cell, Layer 2: orders completed by buffer size (95 % Wilson)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"plot saved to {args.out}")


if __name__ == "__main__":
    main()
