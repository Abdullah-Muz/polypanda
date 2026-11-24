"""v4.0: the one-shot report's target checks (scripts/door_final_report.py) on synthetic runs."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "door_final_report", Path(__file__).parents[1] / "scripts/door_final_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def rows(n, fail=(), steps=50, force=30.0, slams=(), cmd_of=lambda i: "open"):
    return [{"index": i, "command": cmd_of(i), "slide": "right", "spring": False,
             "success": i not in fail, "failure": None if i not in fail else "not_opened",
             "steps": steps, "peak_force": force, "slams": int(i in slams)}
            for i in range(n)]


def test_all_targets_met():
    seeds = {"0": rows(200), "1": rows(200, fail={3}), "2": rows(200)}
    t = report.targets(seeds, rows(200, steps=180, force=31.0, fail={5}),
                       rows(200, fail=set(range(0, 200, 4))))
    assert t["T1"]["pass"] and t["T2"]["pass"] and t["T3"]["pass"] and t["T4"]["pass"]
    assert t["T2"]["only_first"] == 150 and t["T2"]["only_second"] == 1   # seed 1's episode 3
    try:                                                 # runs must pair episode by episode
        report.targets({"0": rows(200)}, rows(200), rows(200)[::-1])
        raise AssertionError("a misordered baseline was accepted")
    except ValueError:
        pass


def test_each_target_can_miss():
    # T1: one cell below 90 % on one seed (both commands present so the open cell is 85 %)
    cmd = lambda i: "open" if i < 100 else "close"  # noqa: E731
    seeds = {"0": rows(200, fail=set(range(15)), cmd_of=cmd), "1": rows(200, cmd_of=cmd)}
    feel, cam = rows(200, steps=180, cmd_of=cmd), rows(200, cmd_of=cmd)
    t = report.targets(seeds, feel, cam)
    assert not t["T1"]["pass"] and t["T1"]["1"] and not t["T1"]["0"]
    assert not t["T2"]["pass"]                       # no better than the camera script
    assert not t["T3"]["success_pass"]               # 3.75 points below feel
    # T3 speed: only 10 % shorter; T4: harder than feel and 3 slam episodes of 200 (1.5 %)
    t = report.targets({"0": rows(200, steps=162, force=26.0, slams={1, 2, 3})},
                       rows(200, steps=180, force=25.0), rows(200))
    assert t["T3"]["success_pass"] and not t["T3"]["speed_pass"]
    assert not t["T4"]["force_pass"] and not t["T4"]["slams_pass"]


def test_slams_at_exactly_one_percent_pass():
    t = report.targets({"0": rows(400, slams={1, 2, 3, 4})}, rows(400, steps=180, force=40.0),
                       rows(400))
    assert t["T4"]["slams_pass"] and t["T4"]["force_pass"]
