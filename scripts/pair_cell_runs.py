"""Pair physics-cell runs of two configurations by (seed, order) and test them with exact McNemar.

Usage:
  uv run python scripts/pair_cell_runs.py \
      --a results/cell/<config A, seed 0>_episodes.csv <seed 1> <seed 2> \
      --b results/cell/<config B, seed 0>_episodes.csv <seed 1> <seed 2> \
      [--kinds D1 D2 D3] [--recoverable-only] [--json out.json]

One _episodes.csv per seed per configuration (evaluate_cell.py output). The run JSON next to
each file (same name without _episodes.csv, plus .json) gives its seed (the grasp policy's
`_seed<k>`, else the file name) and its order stream and disturbance suite. Pairing refuses:
two files of one configuration with the same seed, seeds that differ between A and B, a different
suite or order stream (eval.seed, eval.disturbances), or different order indices for a seed.
Episodes are paired by (seed, order index), never by row position, then kept if their knock kind
(CellEpisode.disturbance) is in --kinds (default D1-D3: knocked orders). The kind must agree
between A and B for every pair (the knock comes from the suite, so it always should).
Reports the exact McNemar test (polypanda.evaluation.pickplace.mcnemar_exact) on the pooled pairs,
per seed and per kind, and, where a _skills.csv sits beside the run, the cubes rescued per
configuration (successful picks of a loose cube, source "loose", in the kept orders).
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from polypanda.evaluation.pickplace import mcnemar_exact

KNOCKED = ("D1", "D2", "D3")
# eval fields that must match for a fair pairing (the orders and the knocks)
MUST_MATCH = ("seed", "disturbances", "heights", "surplus", "solvable_with", "mixed")
# eval fields that should match for the v3.1 target ("same skills and spares"): warned about
SHOULD_MATCH = ("spares", "slack", "slots", "recover", "aim", "batch_choice", "lateral", "yaw")


def run_json(episodes_csv: Path) -> dict:
    name = episodes_csv.name
    if not name.endswith("_episodes.csv"):
        raise SystemExit(f"{episodes_csv}: not an _episodes.csv file")
    path = episodes_csv.with_name(name[: -len("_episodes.csv")] + ".json")
    if not path.exists():
        raise SystemExit(f"{episodes_csv}: no run JSON beside it ({path.name})")
    return json.loads(path.read_text())


def run_seed(episodes_csv: Path, record: dict) -> int:
    """The run's seed: its grasp policy's (pick provenance policy_id ..._seed<k>), else the first
    `_seed<k>` in the file name."""
    policy_id = (record.get("skills", {}).get("pick_provenance") or {}).get("policy_id", "")
    for text in (policy_id, episodes_csv.name):
        if m := re.search(r"_seed(\d+)", text):
            return int(m.group(1))
    raise SystemExit(f"{episodes_csv}: cannot tell its seed")


def rescued_by_order(episodes_csv: Path) -> dict[int, int] | None:
    """Successful loose-cube picks per order (stats 'recovered'), from the _skills.csv beside
    the run; None if there is none."""
    path = episodes_csv.with_name(episodes_csv.name.replace("_episodes.csv", "_skills.csv"))
    if not path.exists():
        return None
    out: dict[int, int] = defaultdict(int)
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["skill"] == "pick" and r.get("source") == "loose" and r["success"] == "True":
                out[int(r["order"])] += 1
    return out


def load_group(label: str, files: list[Path]) -> dict:
    """{seed: {"file", "record", "episodes": {index: row}, "rescued"}} for one configuration."""
    group = {}
    for f in files:
        record = run_json(f)
        seed = run_seed(f, record)
        if seed in group:
            raise SystemExit(f"--{label}: two files for seed {seed}: {group[seed]['file']} and {f}")
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        episodes = {int(r["index"]): r for r in rows}
        if len(episodes) != len(rows):
            raise SystemExit(f"{f}: repeated order indices")
        group[seed] = {"file": f, "record": record, "episodes": episodes,
                       "rescued": rescued_by_order(f)}
    return group


def check_pairable(a: dict, b: dict) -> list[str]:
    """Refuse (SystemExit) what would make the pairing unfair; return warnings."""
    if set(a) != set(b):
        raise SystemExit(f"seeds differ: A {sorted(a)}, B {sorted(b)}")
    warnings = []
    for seed in sorted(a):
        ea, eb = a[seed]["record"]["eval"], b[seed]["record"]["eval"]
        for key in MUST_MATCH:
            if ea.get(key) != eb.get(key):
                raise SystemExit(f"seed {seed}: eval.{key} differs: A {ea.get(key)!r}, "
                                 f"B {eb.get(key)!r}")
        if not ea.get("disturbances"):
            raise SystemExit(f"seed {seed}: no disturbance suite in the runs")
        for key in SHOULD_MATCH:
            if ea.get(key) != eb.get(key):
                warnings.append(f"seed {seed}: eval.{key} differs: A {ea.get(key)!r}, "
                                f"B {eb.get(key)!r}")
        sa, sb = a[seed]["record"].get("skills", {}), b[seed]["record"].get("skills", {})
        for skill in ("pick", "place", "standup"):
            if sa.get(skill) != sb.get(skill):
                warnings.append(f"seed {seed}: {skill} skill differs: A {sa.get(skill)!r}, "
                                f"B {sb.get(skill)!r}")
        ia, ib = set(a[seed]["episodes"]), set(b[seed]["episodes"])
        if ia != ib:
            raise SystemExit(f"seed {seed}: order indices differ (only A: "
                             f"{sorted(ia - ib)[:10]}, only B: {sorted(ib - ia)[:10]})")
    seeds = sorted(a)
    suites = {a[s]["record"]["eval"]["disturbances"] for s in seeds}
    if len(suites) > 1:
        warnings.append(f"the seeds use different suites: {sorted(suites)}")
    return warnings


def pairs(a: dict, b: dict, kinds: tuple[str, ...], recoverable_only: bool) -> list[dict]:
    out = []
    for seed in sorted(a):
        for i in sorted(a[seed]["episodes"]):
            ra, rb = a[seed]["episodes"][i], b[seed]["episodes"][i]
            if ra["disturbance"] != rb["disturbance"] or ra["recoverable"] != rb["recoverable"]:
                raise SystemExit(f"seed {seed}, order {i}: knock differs between A "
                                 f"({ra['disturbance']}, recoverable {ra['recoverable']}) and B "
                                 f"({rb['disturbance']}, recoverable {rb['recoverable']})")
            if ra["disturbance"] not in kinds:
                continue
            if recoverable_only and ra["recoverable"] != "True":
                continue
            out.append({"seed": seed, "index": i, "kind": ra["disturbance"],
                        "a": ra["success"] == "True", "b": rb["success"] == "True"})
    return out


def test(rows: list[dict]) -> dict:
    n = len(rows)
    a, b = [r["a"] for r in rows], [r["b"] for r in rows]
    result = mcnemar_exact(a, b)
    return {"pairs": n, "a_rate": sum(a) / n if n else None, "b_rate": sum(b) / n if n else None,
            "only_a": result["only_first"], "only_b": result["only_second"],
            "p_value": result["p_value"]}


def counts(group: dict, rows: list[dict]) -> dict:
    """Per configuration, over the kept orders: cleared, cleared_ok, replaced (left or cleared),
    left (replaced - cleared), spares used, and rescued (from _skills.csv, if every seed has
    one)."""
    keep = {(r["seed"], r["index"]) for r in rows}
    total = defaultdict(int)
    have_skills = all(g["rescued"] is not None for g in group.values())
    for seed, g in group.items():
        for i, e in g["episodes"].items():
            if (seed, i) not in keep:
                continue
            for key in ("cleared", "cleared_ok", "replaced", "spares_used"):
                total[key] += int(e.get(key) or 0)
            if have_skills:
                total["rescued"] += g["rescued"].get(i, 0)
    total["left"] = total["replaced"] - total["cleared"]
    if not have_skills:
        total["rescued"] = None
    return dict(total)


def fmt_row(name: str, t: dict) -> str:
    if not t["pairs"]:
        return f"{name:14s} {0:6d}"
    return (f"{name:14s} {t['pairs']:6d} {100 * t['a_rate']:7.1f} {100 * t['b_rate']:7.1f} "
            f"{t['only_a']:7d} {t['only_b']:7d} {t['p_value']:9.3g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--a", type=Path, nargs="+", required=True,
                        help="configuration A: one _episodes.csv per seed")
    parser.add_argument("--b", type=Path, nargs="+", required=True,
                        help="configuration B: one _episodes.csv per seed")
    parser.add_argument("--kinds", nargs="+", default=list(KNOCKED),
                        help="knock kinds kept (CellEpisode.disturbance); default D1 D2 D3")
    parser.add_argument("--recoverable-only", action="store_true",
                        help="keep only orders whose knock was physically recoverable")
    parser.add_argument("--json", type=Path, help="also write the results here")
    args = parser.parse_args()

    a, b = load_group("a", args.a), load_group("b", args.b)
    warnings = check_pairable(a, b)
    rows = pairs(a, b, tuple(args.kinds), args.recoverable_only)
    if not rows:
        raise SystemExit("no pairs left after filtering")

    pooled = test(rows)
    by_seed = {s: test([r for r in rows if r["seed"] == s]) for s in sorted(a)}
    by_kind = {k: test([r for r in rows if r["kind"] == k]) for k in args.kinds}
    extra = {"a": counts(a, rows), "b": counts(b, rows)}

    for w in warnings:
        print(f"WARNING: {w}")
    first = next(iter(a.values()))["record"]
    print(f"suite {first['eval']['disturbances']}, order stream seed {first['eval']['seed']}, "
          f"kinds {' '.join(args.kinds)}{', recoverable only' if args.recoverable_only else ''}")
    print(f"A: {a[min(a)]['record'].get('skills', {}).get('recovery_planner', '?')}")
    print(f"B: {b[min(b)]['record'].get('skills', {}).get('recovery_planner', '?')}")
    print(f"{'':14s} {'pairs':>6s} {'A %':>7s} {'B %':>7s} {'only A':>7s} {'only B':>7s} "
          f"{'p (exact)':>9s}")
    print(fmt_row("pooled", pooled))
    for s, t in by_seed.items():
        print(fmt_row(f"seed {s}", t))
    for k, t in by_kind.items():
        print(fmt_row(k, t))
    for label in ("a", "b"):
        c = extra[label]
        print(f"{label.upper()} over the kept orders: rescued "
              f"{'n/a (no _skills.csv)' if c['rescued'] is None else c['rescued']}, "
              f"left {c['left']}, cleared {c['cleared']} (into the bin {c['cleared_ok']}), "
              f"spares used {c['spares_used']}")

    if args.json:
        out = {"a": {str(s): str(g["file"]) for s, g in a.items()},
               "b": {str(s): str(g["file"]) for s, g in b.items()},
               "kinds": args.kinds, "recoverable_only": args.recoverable_only,
               "warnings": warnings, "pooled": pooled,
               "by_seed": {str(s): t for s, t in by_seed.items()}, "by_kind": by_kind,
               "counts": extra}
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
