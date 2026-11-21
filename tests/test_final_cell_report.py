"""scripts/final_cell_report.py on tiny synthetic runs (no simulation)."""

import csv
import importlib
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

COLUMNS = ["index", "success", "failure", "disturbance", "recoverable", "spares_used"]
# 6 orders; the suite plans no knock for orders 1 and 4
SUITE = [{"tower": 0, "height": 2, "push": "push_top", "angle": 0.0, "strength": 1.0}, None,
         {"tower": 1, "height": 2, "push": "push_top", "angle": 0.0, "strength": 1.0},
         {"tower": 2, "height": 3, "push": "shove_top", "angle": 1.0, "strength": 1.0}, None,
         {"tower": 0, "height": 1, "push": "push_top", "angle": 2.0, "strength": 1.0}]


@pytest.fixture(scope="module")
def report():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield importlib.import_module("final_cell_report")
    finally:
        sys.path.remove(str(SCRIPTS))


def write_suite(tmp_path: Path, name: str = "suite_t") -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"meta": {"orders": "Layer 2 evaluation orders (seed 7)"},
                                "orders": SUITE}))
    return path


def write_run(d: Path, name: str, seed: int | None, rows: list[tuple], suite: str = "suite_t",
              picks: list[tuple] = ()) -> None:
    """rows: (success, disturbance, recoverable); picks: (source, failure)."""
    policy = {} if seed is None else {"policy_id": f"grasp_t_seed{seed}"}
    record = {"skills": {"pick": "p", "place": "l", "standup": "s", "pick_provenance": policy},
              "eval": {"seed": 7, "disturbances": f"eval_sets/{suite}.json", "mixed": True,
                       "heights": [1, 4], "surplus": [0, 2], "solvable_with": 3}}
    (d / f"{name}.json").write_text(json.dumps(record))
    with open(d / f"{name}_episodes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for i, (ok, kind, rec) in enumerate(rows):
            w.writerow([i, ok, "" if ok else "pick:knocked_over", kind, rec, 0])
    with open(d / f"{name}_skills.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["skill", "success", "failure", "order", "source"])
        for source, failure in picks:
            w.writerow(["pick", failure == "", failure, 0, source])


T, F = True, False
# configuration 1 and 2, per seed: (success, label, recoverable); order 1 of seed 0 fails before
# its (absent) knock, and order 2 of c2 seed 0 fails before its planned knock (label D0)
C1 = {0: [(T, "D1", T), (F, "D0", F), (T, "D2", T), (T, "D3", F), (T, "D0", F), (T, "none", F)],
      1: [(T, "D1", T), (T, "D0", F), (T, "D1", T), (T, "D3", T), (T, "D0", F), (T, "D1", F)]}
C2 = {0: [(F, "D1", T), (F, "D0", F), (F, "D0", F), (T, "D3", F), (T, "D0", F), (T, "none", F)],
      1: [(F, "D1", T), (T, "D0", F), (T, "D1", T), (F, "D3", T), (T, "D0", F), (T, "D1", F)]}


def make_results(tmp_path: Path) -> Path:
    d = tmp_path / "cell"
    d.mkdir()
    for s in (0, 1):
        write_run(d, f"cell_new_seed{s}_suite_t_spares", s, C1[s],
                  picks=[("belt", "knocked_over"), ("belt", ""), ("loose", "knocked_over")])
        write_run(d, f"cell_old_seed{s}_suite_t_spares", s, C2[s], picks=[("stage", "")])
        write_run(d, f"cell_new_seed{s}_suite_t_recover", s, C1[s])
    write_run(d, "cell_scripted_suite_t_recover", None, C1[0])
    return d


def test_target_regression_d0_and_knockovers(report, tmp_path):
    suite = report.load_suite(write_suite(tmp_path))
    assert suite["planned_d0"] == {1, 4} and suite["order_seed"] == 7
    d = make_results(tmp_path)
    c1 = report.load_config("c1", "cell_new_seed*_{suite}_spares", d, "suite_t", 6)
    c2 = report.load_config("c2", "cell_old_seed*_{suite}_spares", d, "suite_t", 6)
    assert sorted(c1) == [0, 1]
    t = report.target(c1, c2, suite, [0, 1])
    p = t["pooled"]
    assert p["pairs"] == 12 and p["only_a"] == 4 and p["only_b"] == 0
    assert p["a_rate"] == pytest.approx(11 / 12) and p["b_rate"] == pytest.approx(7 / 12)
    assert t["by_seed"]["0"]["only_a"] == 2 and t["met"] is False  # p = 0.125
    assert t["label_disagreements"]["orders"] == 1  # seed 0 order 2: D2 against D0
    assert t["knocked_only"]["pairs"] == 6
    assert report.target(c1, c2, suite, [0, 1, 2])["seeds_complete"] is False

    reg = report.regression(c1)
    assert reg["D1"]["pooled"]["k"] == 3 and reg["D1"]["pooled"]["n"] == 3
    assert reg["D3"]["pooled"]["n"] == 1 and reg["met"] is True

    z = report.d0(c1, suite)
    assert (z["planned"]["pooled"]["k"], z["planned"]["pooled"]["n"]) == (3, 4)
    assert z["planned"]["per_seed"]["0"]["rate"] == 0.5 and z["met"] is False
    z2 = report.d0(c2, suite)
    assert z2["cell_label"]["pooled"]["n"] == 5 and z2["planned"]["pooled"]["n"] == 4

    ko = report.knockovers(c1)["0"]
    assert ko["belt"] == {"knocked_over": 1, "attempts": 2}
    assert ko["floor"] == {"knocked_over": 1, "attempts": 1}
    assert ko["total"]["knocked_over"] == 2


def test_refusals_and_cli(report, tmp_path, monkeypatch, capsys):
    suite_file = write_suite(tmp_path)
    d = make_results(tmp_path)
    suite = report.load_suite(suite_file)
    # a run on another suite is refused
    write_run(d, "cell_other_seed0_suite_t_x", 0, C1[0], suite="suite_u")
    other = report.load_config("c1", "cell_other_seed*_{suite}_x", d, "suite_t", 6)
    with pytest.raises(SystemExit, match="suite"):
        report.check_suite("c1", other, suite)
    # two files for one seed are refused (pair_cell_runs.load_group)
    write_run(d, "cell_new_seed0_suite_t_spares_dup", 0, C1[0])
    with pytest.raises(SystemExit, match="two files"):
        report.load_config("c1", "cell_new_seed*_{suite}_spares*", d, "suite_t", 6)

    out = tmp_path / "r.json"
    monkeypatch.setattr(sys, "argv", [
        "final_cell_report.py", "--suite", "suite_t", "--suite-file", str(suite_file),
        "--results-dir", str(d), "--n", "6", "--seeds", "0", "1",
        "--c1", "cell_new_seed*_{suite}_spares", "--c2", "cell_old_seed*_{suite}_spares",
        "--c3", "cell_new_seed*_{suite}_recover", "--c4", "cell_scripted_{suite}_recover",
        "--c5", "cell_missing_{suite}", "--json", str(out)])
    report.main()
    text = capsys.readouterr().out
    assert "target NOT MET" in text and "PENDING" in text
    res = json.loads(out.read_text())
    assert res["target"]["pooled"]["pairs"] == 12
    assert res["side_by_side"]["c5"] is None
    assert res["side_by_side"]["c4"]["all"]["pooled"]["n"] == 6
