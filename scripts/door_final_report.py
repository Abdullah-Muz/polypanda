"""v4.0: the one-shot report on door_final_v1 (docs/v4_targets.md, targets T1-T4).

Reads the evaluate_door.py results of the locked policy (one run per seed) and of the scripted
baselines on the set, and checks each target exactly as docs/v4_targets.md words it:
- T1: success >= 95 % for every seed, and >= 90 % in every command x slide x spring cell, every
  seed;
- T2: better than the camera-aware script, exact two-sided McNemar p < 0.05, pooled;
- T3: success no more than 2 points below the feel script, pooled; and median steps to success at
  least 20 % shorter, pooled, on the episodes both complete;
- T4: median peak force no higher than the feel script's, pooled; slams in at most 1 % of episodes
  for every seed.
"Pooled": every seed's episodes; a scripted baseline has one run, paired with every seed by episode.

Usage:
  uv run python scripts/door_final_report.py \
      --policy 'runs_door_v2_bc_ppo_r5_seed{}_final' --seeds 0 1 2 --feel feel_v2 \
      --set door_final_v1
Writes results/v4/door/final_report_<set>.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from polypanda.evaluation.pickplace import mcnemar_exact

RESULTS = Path("results/v4/door")


def cell(r: dict) -> str:
    return f"{r['command']}/{r['slide']}/{'spring' if r['spring'] else 'plain'}"


def targets(seeds: dict[str, list[dict]], feel: list[dict], camera: list[dict]) -> dict:
    """seeds: seed -> rows (one per episode, sorted by index); feel, camera: rows of one run."""
    out: dict = {"T1": {}, "per_seed": {}}
    for s, rows in seeds.items():
        order = [r["index"] for r in rows]
        if order != [r["index"] for r in feel] or order != [r["index"] for r in camera]:
            raise ValueError("runs must cover the same episodes in the same order")
        n, k = len(rows), sum(r["success"] for r in rows)
        cells: dict[str, list[int]] = {}
        for r in rows:
            c = cells.setdefault(cell(r), [0, 0])
            c[0] += r["success"]
            c[1] += 1
        worst = min(cells.items(), key=lambda kv: kv[1][0] / kv[1][1])
        slams = sum(r["slams"] > 0 for r in rows)
        out["per_seed"][s] = {
            "success": k, "n": n, "rate": k / n, "worst_cell": worst[0],
            "worst_cell_rate": worst[1][0] / worst[1][1], "slam_episodes": slams,
            "slam_rate": slams / n,
            "failures": sorted(f"{cell(r)}:{r['failure']}" for r in rows if not r["success"]),
        }
        out["T1"][s] = k / n >= 0.95 and all(c[0] / c[1] >= 0.90 for c in cells.values())
    out["T1"]["pass"] = all(out["T1"][s] for s in seeds)

    learned = [r for rows in seeds.values() for r in rows]
    feel_p = [r for _ in seeds for r in feel]
    cam_p = [r for _ in seeds for r in camera]
    rate = lambda rows: sum(r["success"] for r in rows) / len(rows)  # noqa: E731

    m = mcnemar_exact([r["success"] for r in learned], [r["success"] for r in cam_p])
    out["T2"] = {"learned": rate(learned), "camera": rate(cam_p), **m,
                 "pass": m["p_value"] < 0.05 and m["only_first"] > m["only_second"]}

    both = [(a["steps"], b["steps"]) for a, b in zip(learned, feel_p, strict=True)
            if a["success"] and b["success"]]
    med_l = float(np.median([a for a, _ in both])) if both else None
    med_f = float(np.median([b for _, b in both])) if both else None
    gap = rate(feel_p) - rate(learned)
    out["T3"] = {"learned": rate(learned), "feel": rate(feel_p), "points_below_feel": 100 * gap,
                 "both_complete": len(both), "median_steps_learned": med_l,
                 "median_steps_feel": med_f,
                 "shorter_by": 1 - med_l / med_f if both else None,
                 "success_pass": gap <= 0.02 + 1e-12,
                 "speed_pass": bool(both) and med_l <= 0.8 * med_f}
    out["T3"]["pass"] = out["T3"]["success_pass"] and out["T3"]["speed_pass"]

    f_l = float(np.median([r["peak_force"] for r in learned]))
    f_f = float(np.median([r["peak_force"] for r in feel_p]))
    slam_ok = {s: out["per_seed"][s]["slam_episodes"] <= 0.01 * out["per_seed"][s]["n"]
               for s in seeds}
    out["T4"] = {"peak_force_median_learned": f_l, "peak_force_median_feel": f_f,
                 "force_pass": f_l <= f_f, "slams_pass_per_seed": slam_ok,
                 "slams_pass": all(slam_ok.values())}
    out["T4"]["pass"] = out["T4"]["force_pass"] and out["T4"]["slams_pass"]
    return out


def load(name: str, set_name: str) -> dict:
    return json.loads((RESULTS / f"{name}_{set_name}.json").read_text())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, help="results name with {} for the seed")
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--feel", default="feel_v2")
    p.add_argument("--camera", default="camera")
    p.add_argument("--nominal", default="nominal")
    p.add_argument("--set", default="door_final_v1")
    args = p.parse_args()
    runs = {str(s): load(args.policy.format(s), args.set) for s in args.seeds}
    base = {k: load(getattr(args, k), args.set) for k in ("feel", "camera", "nominal")}
    digests = {d["set_digest"] for d in [*runs.values(), *base.values()]}
    if len(digests) != 1:
        raise ValueError(f"results come from different sets: {digests}")
    t = targets({s: d["rows"] for s, d in runs.items()}, base["feel"]["rows"],
                base["camera"]["rows"])
    report = {"set": args.set, "set_digest": digests.pop(),
              "policy": {s: d["policy"] for s, d in runs.items()},
              "git": {s: d["git"] for s, d in runs.items()},
              "baselines": {k: {"policy": d["policy"], "git": d["git"], "summary": d["summary"]}
                            for k, d in base.items()},
              **t}
    out = RESULTS / f"final_report_{args.set}.json"
    out.write_text(json.dumps(report, indent=1))
    for s, v in t["per_seed"].items():
        print(f"seed {s}: {v['success']}/{v['n']} ({100 * v['rate']:.1f} %), worst cell "
              f"{v['worst_cell']} {100 * v['worst_cell_rate']:.0f} %, slams {v['slam_episodes']}")
    for k, d in base.items():
        s = d["summary"]
        print(f"{k} ({d['policy']}): {s['success']}/{s['n']}, median steps "
              f"{s['steps_to_success_median']}, peak force {s['peak_force_median']:.1f} N")
    ok = lambda b: "MET" if b else "MISSED"  # noqa: E731
    print(f"T1 {ok(t['T1']['pass'])}: {t['T1']}")
    print(f"T2 {ok(t['T2']['pass'])}: learned {100 * t['T2']['learned']:.1f} % vs camera "
          f"{100 * t['T2']['camera']:.1f} %, discordant {t['T2']['only_first']}/"
          f"{t['T2']['only_second']}, p = {t['T2']['p_value']:.2g}")
    t3 = t["T3"]
    print(f"T3 {ok(t3['pass'])}: success {100 * t3['learned']:.1f} vs {100 * t3['feel']:.1f} % "
          f"({t3['points_below_feel']:+.1f} points below), median steps "
          f"{t3['median_steps_learned']} vs {t3['median_steps_feel']} "
          f"on {t3['both_complete']} pairs "
          f"({100 * t3['shorter_by']:.0f} % shorter)")
    t4 = t["T4"]
    print(f"T4 {ok(t4['pass'])}: peak force {t4['peak_force_median_learned']:.1f} vs "
          f"{t4['peak_force_median_feel']:.1f} N ({ok(t4['force_pass'])}), slams per seed "
          f"{t4['slams_pass_per_seed']} ({ok(t4['slams_pass'])})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
