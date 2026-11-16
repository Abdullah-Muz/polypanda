"""v3.1 skill round: the one-shot results against their targets and paired baselines.

Usage: uv run python scripts/oneshot_report_v31.py [--out results/oneshot_v31.json]

Reads the one-shot runs locked in docs/v3_1_targets.md: the grasp on grasp_lying_final_v2 and
the stand-up on standup_final_v2 (each run's turn chooser), nominal and with the size condition.
Per configuration: success pooled over its seeds with a 95 % Wilson interval, per seed, and the
failures by kind. Against the target (grasp >= 85 %, stand-up >= 90 %) and the size bar (size
condition within 5 points of the nominal one-shot result). Each baseline is paired with the v3.1
skill by seed and case (a scripted skill has one run, paired with every seed); exact McNemar.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from polypanda.evaluation.pickplace import mcnemar_exact
from polypanda.evaluation.reach import wilson_interval

SEEDS = (0, 1, 2)
SKILLS = {
    "grasp": {
        "target": 85.0, "set": "grasp_lying_final_v2", "new": "grasp_anypose6",
        "baselines": ["grasp_anypose5", "grasp_anypose4", "scripted"],
        "nominal": lambda run, st: (f"results/grasp/scripted_v4_{st}_episodes.csv"
                                    if run == "scripted"
                                    else f"results/grasp/{run}_best_{st}_episodes.csv"),
        "size": lambda run, st: (f"results/grasp_robustness/scripted_v4_{st}_size_episodes.csv"
                                 if run == "scripted" else
                                 f"results/grasp_robustness/{run}_best_{st}_size_episodes.csv"),
    },
    "standup": {
        "target": 90.0, "set": "standup_final_v2", "new": "standup_round12",
        "baselines": ["standup_round11", "scripted"],
        "nominal": lambda run, st: (f"results/standup/scripted_{st}_episodes.csv"
                                    if run == "scripted" else
                                    f"results/standup/{run}_best_{st}_choose-chooser_episodes.csv"),
        "size": lambda run, st: (f"results/standup_robustness/scripted_{st}_size_episodes.csv"
                                 if run == "scripted" else
                                 f"results/standup_robustness/{run}_best_{st}_choose-chooser"
                                 "_size_episodes.csv"),
    },
}


def load(path: str) -> dict[int, dict]:
    with open(path) as f:
        return {int(r["index"]): r for r in csv.DictReader(f)}


def config_runs(skill: dict, config: str, condition: str) -> dict[int, dict[int, dict]]:
    """{seed: {index: row}}; a scripted skill's single run stands in for every seed."""
    path = skill[condition]
    if config == "scripted":
        rows = load(path("scripted", skill["set"]))
        return {s: rows for s in SEEDS}
    return {s: load(path(f"{config}_seed{s}", skill["set"])) for s in SEEDS}


def summary(runs: dict[int, dict[int, dict]], single: bool) -> dict:
    seeds = SEEDS[:1] if single else SEEDS
    outcomes = [r["success"] == "True" for s in seeds for r in runs[s].values()]
    lo, hi = wilson_interval(sum(outcomes), len(outcomes))
    return {"cases": len(outcomes), "success": 100 * sum(outcomes) / len(outcomes),
            "ci95": [100 * lo, 100 * hi],
            "per_seed": {s: 100 * sum(r["success"] == "True" for r in runs[s].values())
                         / len(runs[s]) for s in seeds},
            "failures": dict(Counter(r["failure"] for s in seeds for r in runs[s].values()
                                     if r["failure"]).most_common())}


def paired(new: dict, old: dict) -> dict:
    a, b = [], []
    for s in SEEDS:
        if set(new[s]) != set(old[s]):
            raise SystemExit(f"seed {s}: different cases")
        a += [new[s][i]["success"] == "True" for i in sorted(new[s])]
        b += [old[s][i]["success"] == "True" for i in sorted(new[s])]
    return mcnemar_exact(a, b)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/oneshot_v31.json"))
    args = parser.parse_args()
    report = {}
    for name, skill in SKILLS.items():
        r = report[name] = {"set": skill["set"], "target": skill["target"], "configs": {}}
        print(f"\n{name} on {skill['set']} (target >= {skill['target']:.0f} %, size bar: "
              "within 5 points of nominal)")
        new = {c: config_runs(skill, skill["new"], c) for c in ("nominal", "size")}
        for config in [skill["new"], *skill["baselines"]]:
            entry = {}
            for condition in ("nominal", "size"):
                runs = new[condition] if config == skill["new"] else \
                    config_runs(skill, config, condition)
                s = summary(runs, config == "scripted")
                if config != skill["new"]:
                    s["paired_new_vs_this"] = paired(new[condition], runs)
                entry[condition] = s
            entry["size_gap"] = entry["size"]["success"] - entry["nominal"]["success"]
            r["configs"][config] = entry
            for condition in ("nominal", "size"):
                s = entry[condition]
                seeds = " / ".join(f"{v:.1f}" for v in s["per_seed"].values())
                vs = ""
                if "paired_new_vs_this" in s:
                    p = s["paired_new_vs_this"]
                    vs = (f"; v3.1 vs this: {p['only_first']} only v3.1, {p['only_second']} "
                          f"only this, p = {p['p_value']:.3g}")
                print(f"  {config:17s} {condition:7s} {s['success']:5.1f} % "
                      f"({s['ci95'][0]:.1f}-{s['ci95'][1]:.1f}) [{seeds}]{vs}")
            print(f"  {'':17s} size gap {entry['size_gap']:+.1f} points; failures nominal "
                  f"{entry['nominal']['failures']}")
        main_entry = r["configs"][skill["new"]]
        r["target_met"] = main_entry["nominal"]["success"] >= skill["target"]
        r["size_bar_met"] = main_entry["size_gap"] >= -5.0
        print(f"  => target {'MET' if r['target_met'] else 'MISSED'} "
              f"({main_entry['nominal']['success']:.1f} % vs {skill['target']:.0f} %); size bar "
              f"{'MET' if r['size_bar_met'] else 'MISSED'} ({main_entry['size_gap']:+.1f} points)")
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
