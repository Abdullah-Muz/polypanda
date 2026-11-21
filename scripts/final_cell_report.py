"""The v3.2 final cell run, analysed as fixed in docs/v3_2_targets.md before any result existed.

Usage:
  uv run python scripts/final_cell_report.py [--json results/cell/v32_final_report.json]
  # v3.1's committed run on disturbance_suite_v2, as a check of the analysis (clear rule as
  # configuration 1, best rule as 2, v3.1 no spares as 3):
  uv run python scripts/final_cell_report.py --suite disturbance_suite_v2 \
      --c1 'cell_P-grasp_anypose6_seed*_best_*_{suite}_recover_spares2-clear-rule_slack1.5_n{n}' \
      --c2 'cell_P-grasp_anypose6_seed*_best_*_{suite}_recover_spares2-best-rule_slack1.5_n{n}' \
      --c3 'cell_P-grasp_anypose6_seed*_best_*_{suite}_recover_n{n}' \
      --names 'clear rule' 'best rule' 'v3.1 no spares' 'scripted, recover' 'scripted, none'

The five configurations of the final run are given as glob patterns under --results-dir
(evaluate_cell.py's run names, without the _episodes.csv suffix; {suite} and {n} are filled in):
  1 the v3.2 cell (grasp_anypose9, clear rule, 2 spares, slack 1.5), one run per seed;
  2 the v3.1 cell (grasp_anypose6, same recovery), one run per seed;
  3 v3.2 skills with no spares (--recover only), one run per seed;
  4 the scripted cell with recovery, one run;  5 the scripted cell with no recovery, one run.
A configuration with no matching files is reported as pending, and so is every result that
needs it. Learned runs are loaded and paired with scripts/pair_cell_runs.py (seed from the grasp
policy, refusal of repeated seeds, of a different order stream or suite, of different order
indices); every run must also name --suite as its disturbance suite and the suite's order seed.

Reported:
- The target: 1 against 2 on all orders (every knock kind: D0, none, D1, D2, D3), paired by
  (seed, order index), exact McNemar pooled over the seeds; rates with Wilson 95 % intervals and
  per seed. Met if 1 completes more than 2 with p < 0.05 and both have every --seeds seed. Also
  the knocked orders only (both runs labelled D1-D3), the measure of v3.1's recovery target.
- The regression check, from 3: completion of recoverable D1 / D2 / D3 orders (the cell's knock
  label and its `recoverable` flag, pooled over the seeds; summarize_cell's
  success_rate_recoverable), against 90 / 85 / 80 %.
- D0 (bar 99 %), from 1 and 3: completion of the suite's planned no-knock orders (null entries in
  the suite's `orders`), per seed and pooled; the cell's own D0 label beside it.
- No target: pick knock-overs per seed from each run's _skills.csv (failure knocked_over; belt =
  source belt, floor = source loose, other = stage or buffer), 1 against 2 (and 3, as v3.1
  read them); seed 0 against seeds 1-2; and every configuration side by side, the scripted ones
  as baselines.
"""

import argparse
import csv
import json
import re
from pathlib import Path

from pair_cell_runs import MUST_MATCH, check_pairable, load_group, run_json, test

from polypanda.evaluation.reach import wilson_interval

KINDS = ("D0", "none", "D1", "D2", "D3")
KNOCKED = ("D1", "D2", "D3")
BARS = {"D1": 0.90, "D2": 0.85, "D3": 0.80}
D0_BAR = 0.99
ALPHA = 0.05
DEFAULTS = {
    "c1": "cell_P-grasp_anypose9_seed*_best_*_{suite}_recover_spares2-clear-rule_slack1.5_n{n}",
    "c2": "cell_P-grasp_anypose6_seed*_best_*_{suite}_recover_spares2-clear-rule_slack1.5_n{n}",
    "c3": "cell_P-grasp_anypose9_seed*_best_*_{suite}_recover_n{n}",
    "c4": "cell_P-scripted_L-scripted_*_{suite}_recover_n{n}",
    "c5": "cell_P-scripted_L-scripted_*_{suite}_n{n}",
}
NAMES = ["v3.2 cell", "v3.1 cell", "v3.2 no spares", "scripted, recover", "scripted, none"]
SINGLE = {"c4", "c5"}  # one run each, not one per seed


# ---------------------------------------------------------------- loading

def load_suite(path: Path) -> dict:
    """{"name", "orders" (list, None = planned no-knock), "planned_d0" (set of indices),
    "order_seed" (from meta.orders, or None)}."""
    data = json.loads(path.read_text())
    orders = data["orders"]
    m = re.search(r"seed (\d+)", str(data.get("meta", {}).get("orders", "")))
    return {"name": path.stem, "path": str(path), "orders": orders,
            "planned_d0": {i for i, o in enumerate(orders) if o is None},
            "order_seed": int(m.group(1)) if m else None}


def load_single(label: str, files: list[Path]) -> dict:
    """One run (a scripted configuration), keyed "run", in load_group's layout."""
    if len(files) != 1:
        raise SystemExit(f"--{label}: expected one run, found {len(files)}: "
                         f"{[f.name for f in files]}")
    f = files[0]
    with open(f) as fh:
        rows = list(csv.DictReader(fh))
    episodes = {int(r["index"]): r for r in rows}
    if len(episodes) != len(rows):
        raise SystemExit(f"{f}: repeated order indices")
    return {"run": {"file": f, "record": run_json(f), "episodes": episodes}}


def load_config(label: str, pattern: str, results_dir: Path, suite: str, n: int) -> dict | None:
    pat = pattern.format(suite=suite, n=n)
    if not pat.endswith("_episodes.csv"):
        pat += "_episodes.csv"
    files = sorted(results_dir.glob(pat))
    if not files:
        return None
    return load_single(label, files) if label in SINGLE else load_group(label, files)


def check_suite(label: str, group: dict, suite: dict) -> None:
    """Refuse a run that is not on `suite` (by name and order seed) or has orders it lacks."""
    for key, g in group.items():
        ev = g["record"]["eval"]
        if not ev.get("mixed") or not ev.get("disturbances"):
            raise SystemExit(f"--{label} {key}: not a Layer 2 run with a disturbance suite")
        if Path(ev["disturbances"]).stem != suite["name"]:
            raise SystemExit(f"--{label} {key}: suite {ev['disturbances']}, not {suite['name']}")
        if suite["order_seed"] is not None and ev.get("seed") != suite["order_seed"]:
            raise SystemExit(f"--{label} {key}: order stream seed {ev.get('seed')}, the suite "
                             f"was made for {suite['order_seed']}")
        extra = set(g["episodes"]) - set(range(len(suite["orders"])))
        if extra:
            raise SystemExit(f"--{label} {key}: orders not in the suite: {sorted(extra)[:10]}")


def check_same_orders(label: str, group: dict, ref: dict) -> None:
    """A single (scripted) run must play the same orders as the learned runs."""
    ref_eval = next(iter(ref.values()))["record"]["eval"]
    for key, g in group.items():
        ev = g["record"]["eval"]
        for k in MUST_MATCH:
            if ev.get(k) != ref_eval.get(k):
                raise SystemExit(f"--{label} {key}: eval.{k} {ev.get(k)!r} differs from the "
                                 f"learned runs' {ref_eval.get(k)!r}")


# ---------------------------------------------------------------- measures

def rate(k: int, n: int) -> dict:
    lo, hi = wilson_interval(k, n) if n else (None, None)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": [lo, hi] if n else None}


def ok(row: dict) -> bool:
    return row["success"] == "True"


def completion(group: dict, keep) -> dict:
    """Completion over the episodes `keep(index, row)` selects: per run and pooled."""
    per = {}
    for key, g in group.items():
        rows = [r for i, r in g["episodes"].items() if keep(i, r)]
        per[str(key)] = rate(sum(ok(r) for r in rows), len(rows))
    return {"pooled": rate(sum(v["k"] for v in per.values()), sum(v["n"] for v in per.values())),
            "per_seed": per}


def paired_rows(a: dict, b: dict, suite: dict) -> list[dict]:
    """Every order of every seed, paired by (seed, index); unlike pair_cell_runs.pairs, the two
    runs' knock labels may differ (an order that fails before its knock is labelled D0)."""
    out = []
    for seed in sorted(a):
        for i in sorted(a[seed]["episodes"]):
            ra, rb = a[seed]["episodes"][i], b[seed]["episodes"][i]
            out.append({"seed": seed, "index": i,
                        "planned": "D0" if i in suite["planned_d0"] else "knock",
                        "kind_a": ra["disturbance"], "kind_b": rb["disturbance"],
                        "a": ok(ra), "b": ok(rb)})
    return out


def with_ci(t: dict) -> dict:
    n = t["pairs"]
    for side in ("a", "b"):
        r = t[f"{side}_rate"]
        t[f"{side}_ci95"] = list(wilson_interval(round(r * n), n)) if n else None
    return t


def target(a: dict, b: dict, suite: dict, seeds: list[int]) -> dict:
    rows = paired_rows(a, b, suite)
    pooled = with_ci(test(rows))
    by_seed = {str(s): with_ci(test([r for r in rows if r["seed"] == s])) for s in sorted(a)}
    knocked = [r for r in rows if r["kind_a"] in KNOCKED and r["kind_b"] in KNOCKED]
    disagree = [r for r in rows if r["kind_a"] != r["kind_b"]]
    complete = sorted(a) == sorted(seeds)
    met = bool(complete and pooled["pairs"] and pooled["a_rate"] > pooled["b_rate"]
               and pooled["p_value"] < ALPHA)
    return {"orders": "all (D0, none, D1, D2, D3)", "pooled": pooled, "by_seed": by_seed,
            "seeds_complete": complete, "met": met,
            "knocked_only": {
                "definition": "both runs labelled the order D1-D3, as v3.1's recovery target",
                **(with_ci(test(knocked)) if knocked else {"pairs": 0}),
                "by_kind": {k: with_ci(test([r for r in knocked if r["kind_a"] == k]))
                            for k in KNOCKED
                            if any(r["kind_a"] == k for r in knocked)}},
            "label_disagreements": {"orders": len(disagree),
                                    "examples": [(r["seed"], r["index"], r["kind_a"], r["kind_b"])
                                                 for r in disagree[:10]]}}


def regression(group: dict) -> dict:
    out = {}
    for k, bar in BARS.items():
        c = completion(group, lambda i, r, k=k: r["disturbance"] == k
                       and r["recoverable"] == "True")
        rt = c["pooled"]["rate"]
        out[k] = {**c, "bar": bar, "met": None if rt is None else rt >= bar}
    out["met"] = (None if any(out[k]["met"] is None for k in BARS)
                  else all(out[k]["met"] for k in BARS))
    return out


def d0(group: dict, suite: dict) -> dict:
    planned = completion(group, lambda i, r: i in suite["planned_d0"])
    label = completion(group, lambda i, r: r["disturbance"] == "D0")
    rt = planned["pooled"]["rate"]
    return {"planned": planned, "cell_label": label, "bar": D0_BAR,
            "met": None if rt is None else rt >= D0_BAR}


def knockovers(group: dict) -> dict:
    """Pick knock-overs per seed from the _skills.csv beside each run: belt (source belt),
    floor (source loose: a cube on the table after a knock), other (stage or buffer)."""
    out = {}
    for key, g in group.items():
        f = g["file"]
        path = f.with_name(f.name.replace("_episodes.csv", "_skills.csv"))
        if not path.exists():
            out[str(key)] = None
            continue
        tally = {c: {"knocked_over": 0, "attempts": 0} for c in ("belt", "floor", "other")}
        with open(path) as fh:
            for r in csv.DictReader(fh):
                if r["skill"] != "pick":
                    continue
                cat = {"belt": "belt", "loose": "floor"}.get(r.get("source"), "other")
                tally[cat]["attempts"] += 1
                tally[cat]["knocked_over"] += r.get("failure") == "knocked_over"
        tally["total"] = {k: sum(tally[c][k] for c in ("belt", "floor", "other"))
                          for k in ("knocked_over", "attempts")}
        out[str(key)] = tally
    return out


def side_by_side(group: dict, suite: dict) -> dict:
    eps = [r for g in group.values() for r in g["episodes"].values()]
    out = {"all": completion(group, lambda i, r: True),
           "planned_d0": completion(group, lambda i, r: i in suite["planned_d0"])["pooled"],
           "planned_knock": completion(group, lambda i, r: i not in suite["planned_d0"])["pooled"],
           "knocked_label": completion(group, lambda i, r: r["disturbance"] in KNOCKED)["pooled"],
           "recoverable": {k: completion(group, lambda i, r, k=k: r["disturbance"] == k
                                         and r["recoverable"] == "True")["pooled"]
                           for k in KNOCKED},
           "late": sum(r["failure"] == "late" for r in eps),
           "spares_used": sum(int(r.get("spares_used") or 0) for r in eps)}
    if all(isinstance(s, int) for s in group) and 0 in group and len(group) > 1:
        out["seed0"] = completion(group, lambda i, r: True)["per_seed"]["0"]
        rest = {s: g for s, g in group.items() if s != 0}
        out["other_seeds"] = completion(rest, lambda i, r: True)["pooled"]
    return out


# ---------------------------------------------------------------- printing

def pct(r: dict | None) -> str:
    if not r or not r.get("n"):
        return "n/a"
    lo, hi = r["ci95"]
    return f"{100 * r['rate']:.1f} % ({r['k']}/{r['n']}; {100 * lo:.1f}-{100 * hi:.1f})"


def pct_short(r: dict | None) -> str:
    return "n/a" if not r or not r.get("n") else f"{100 * r['rate']:.1f} % ({r['k']}/{r['n']})"


def verdict(met: bool | None) -> str:
    return {True: "MET", False: "MISSED", None: "PENDING"}[met]


def print_target(t: dict, names: list[str]) -> None:
    p = t["pooled"]
    n = p["pairs"]
    print(f"TARGET: {names[0]} vs {names[1]}, all orders, paired by seed and order")
    print(f"  {names[0]}: {pct(rate(round(p['a_rate'] * n), n))}")
    print(f"  {names[1]}: {pct(rate(round(p['b_rate'] * n), n))}")
    print(f"  only {names[0]} {p['only_a']}, only {names[1]} {p['only_b']}, "
          f"exact McNemar p = {p['p_value']:.3g}")
    for s, v in t["by_seed"].items():
        print(f"  seed {s}: {100 * v['a_rate']:.1f} % vs {100 * v['b_rate']:.1f} % "
              f"({v['pairs']} orders; only {v['only_a']} / {v['only_b']}, p = {v['p_value']:.3g})")
    if not t["seeds_complete"]:
        print("  not every expected seed is present")
    print(f"  -> target {'MET' if t['met'] else 'NOT MET'} (needs {names[0]} > {names[1]}, "
          f"p < {ALPHA})")
    k = t["knocked_only"]
    if k["pairs"]:
        print(f"  check: knocked orders only ({k['definition']}): {k['pairs']} pairs, "
              f"{100 * k['a_rate']:.1f} % vs {100 * k['b_rate']:.1f} %, only {k['only_a']} / "
              f"{k['only_b']}, p = {k['p_value']:.3g}")
    d = t["label_disagreements"]
    print(f"  orders whose knock label differs between the two runs: {d['orders']}"
          + (f" (seed, order, label 1, label 2: {d['examples']})" if d["orders"] else ""))


def print_completion(title: str, c: dict) -> None:
    seeds = ", ".join(f"seed {s} {pct_short(v)}" for s, v in c["per_seed"].items())
    print(f"  {title}: {pct(c['pooled'])}; {seeds}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--suite", default="disturbance_suite_v3",
                        help="the disturbance suite's name (eval_sets/<suite>.json)")
    parser.add_argument("--suite-file", type=Path, help="default eval_sets/<suite>.json")
    parser.add_argument("--n", type=int, default=200, help="orders per run (run name _n<N>)")
    parser.add_argument("--results-dir", type=Path, default=Path("results/cell"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="seeds every learned configuration must have")
    for i, key in enumerate(DEFAULTS, start=1):
        parser.add_argument(f"--{key}", default=DEFAULTS[key],
                            help=f"configuration {i} ({NAMES[i - 1]}): run-name glob, default "
                                 f"{DEFAULTS[key]!r}".replace("%", "%%"))
    parser.add_argument("--names", nargs=5, default=NAMES, help="labels of configurations 1-5")
    parser.add_argument("--json", type=Path, help="also write the results here")
    args = parser.parse_args()

    suite = load_suite(args.suite_file or Path("eval_sets") / f"{args.suite}.json")
    if suite["name"] != args.suite:
        raise SystemExit(f"--suite-file {suite['path']} is not {args.suite}")
    names = args.names
    groups, patterns = {}, {}
    for key in DEFAULTS:
        patterns[key] = getattr(args, key).format(suite=args.suite, n=args.n)
        groups[key] = load_config(key, getattr(args, key), args.results_dir, args.suite, args.n)
        if groups[key] is not None:
            check_suite(key, groups[key], suite)
    learned = next((groups[k] for k in ("c1", "c2", "c3") if groups[k] is not None), None)
    if learned is not None:
        for key in SINGLE:
            if groups[key] is not None:
                check_same_orders(key, groups[key], learned)

    print(f"suite {suite['name']}: {len(suite['orders'])} orders, "
          f"{len(suite['planned_d0'])} planned no-knock; order stream seed {suite['order_seed']}")
    for i, key in enumerate(DEFAULTS):
        g = groups[key]
        print(f"  {i + 1} {names[i]}: "
              + ("PENDING, no files match " + patterns[key] if g is None else
                 ", ".join(f"{s}: {len(v['episodes'])} orders" for s, v in g.items())))
    out = {"suite": {k: suite[k] for k in ("name", "path", "order_seed")},
           "planned_d0": len(suite["planned_d0"]),
           "configurations": {key: {"name": names[i], "pattern": patterns[key],
                                    "files": None if groups[key] is None else
                                    {str(s): str(g["file"]) for s, g in groups[key].items()}}
                              for i, key in enumerate(DEFAULTS)}}

    print()
    if groups["c1"] is not None and groups["c2"] is not None:
        warnings = check_pairable(groups["c1"], groups["c2"])
        for w in warnings:
            print(f"note: {w}")
        out["target"] = target(groups["c1"], groups["c2"], suite, args.seeds)
        out["target"]["notes"] = warnings
        print_target(out["target"], names)
    else:
        out["target"] = None
        print(f"TARGET: PENDING ({names[0]} and {names[1]} are both needed)")

    print()
    if groups["c3"] is not None:
        out["regression"] = regression(groups["c3"])
        print(f"REGRESSION CHECK, {names[2]}: recoverable knocks completed (the cell's label "
              "and recoverable flag)")
        for k in BARS:
            r = out["regression"][k]
            print_completion(f"{k} (bar {100 * r['bar']:.0f} %) {verdict(r['met'])}", r)
    else:
        out["regression"] = None
        print(f"REGRESSION CHECK: PENDING ({names[2]} is needed)")

    print()
    out["d0"] = {}
    print("D0, the suite's planned no-knock orders (bar 99 %, pooled)")
    for key, i in (("c1", 0), ("c3", 2)):
        if groups[key] is None:
            out["d0"][key] = None
            print(f"  {names[i]}: PENDING")
            continue
        r = out["d0"][key] = d0(groups[key], suite)
        print_completion(f"{names[i]}, planned {verdict(r['met'])}", r["planned"])
        print_completion(f"{names[i]}, the cell's own D0 label", r["cell_label"])

    print()
    out["knockovers"] = {}
    print("Pick knock-overs (no target), knocked over / attempts: belt, floor (loose), other")
    for key, i in (("c1", 0), ("c2", 1), ("c3", 2)):
        if groups[key] is None:
            out["knockovers"][key] = None
            print(f"  {names[i]}: PENDING")
            continue
        ko = out["knockovers"][key] = knockovers(groups[key])
        for s, t in ko.items():
            print(f"  {names[i]}, seed {s}: " + ("no _skills.csv" if t is None else ", ".join(
                f"{c} {t[c]['knocked_over']}/{t[c]['attempts']}"
                for c in ("belt", "floor", "other", "total"))))

    print()
    out["side_by_side"] = {}
    print("Side by side (no target): all orders; planned no-knock; planned knock; labelled "
          "D1-D3; recoverable D1 / D2 / D3; late; spares used")
    for i, key in enumerate(DEFAULTS):
        if groups[key] is None:
            out["side_by_side"][key] = None
            print(f"  {i + 1} {names[i]}: PENDING")
            continue
        sb = out["side_by_side"][key] = side_by_side(groups[key], suite)
        print(f"  {i + 1} {names[i]}: {pct_short(sb['all']['pooled'])}; "
              f"{pct_short(sb['planned_d0'])}; {pct_short(sb['planned_knock'])}; "
              f"{pct_short(sb['knocked_label'])}; "
              + " / ".join(pct_short(sb["recoverable"][k]) for k in KNOCKED)
              + f"; late {sb['late']}; spares {sb['spares_used']}")
        if "seed0" in sb:
            print(f"      seed 0 {pct_short(sb['seed0'])}, "
                  f"other seeds {pct_short(sb['other_seeds'])}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
