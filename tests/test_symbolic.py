"""The symbolic cell (v3 step e): the physics cell's order flow and recovery as a board game, with
spares and the learned planner's rescue-or-replace decisions."""

import numpy as np
import pytest

from polypanda.simulation.disturbances import Disturbance, Knock
from polypanda.simulation.workcell import WorkOrder
from polypanda.supervisor.symbolic import (
    ACTIONS,
    RESCUE_ALL,
    RULE_FAMILY,
    Calibration,
    Decision,
    Loose,
    Rule,
    SymbolicCell,
    Task,
    undisturbed_time,
)

ORDER = WorkOrder((("red", "green", "blue"), ("yellow",), ("red",)),
                  tuple((c, None) for c in ("red", "green", "yellow", "blue", "red")))
KNOCK = Disturbance(0, 2, Knock("shove_top", 0.0, 1.0))  # after the green goes onto tower A
KEYS = ([("pick", s, "upright") for s in ("belt", "buffer", "loose")]
        + [("pick", "loose", "lying"), ("standup",), ("place", "buffer", 0), ("place", "stage", 0)]
        + [("place", "tower", lv) for lv in range(4)])


def calibration(p: float = 1.0) -> Calibration:
    return Calibration({k: p for k in KEYS}, {k: 100 for k in KEYS},
                       {**{a: 10.0 for a in ACTIONS}, "reject": 0.0, "overhead": 5.0})


def fallen(pose="upright", recoverable=True, near=False):
    return [{"pose": pose, "recoverable": recoverable, "near": near}]


NO_SPARES = Task(spares=0, slack=None)


def play(fall, policy=None, cal=None, task=NO_SPARES, deadline=None):
    cell = SymbolicCell(ORDER, KNOCK, fall, cal or calibration(), np.random.default_rng(0), task,
                        deadline)
    return cell.result(policy), cell


def test_an_undisturbed_order_completes_and_takes_its_actions_time():
    ok, cell = play(None)
    assert ok and cell.towers == [["red", "green", "blue"], ["yellow"], ["red"]]
    # 5 belt picks, 5 tower places (no buffer: every cube is placeable on arrival), overhead
    assert cell.t == undisturbed_time(ORDER, calibration()) == 5 + 5 * 10 + 5 * 10


def test_a_knocked_cube_is_recovered_and_the_tower_rebuilt():
    ok, cell = play(fallen())
    assert ok and cell.towers[0] == ["red", "green", "blue"]
    ok, cell = play(fallen("lying", near=True))   # staged and stood up first, then picked
    assert ok and cell.staged == 0


def test_a_cube_lost_for_good_fails_the_order_unless_a_spare_replaces_it():
    assert not play(fallen(recoverable=False))[0]
    ok, cell = play(fallen(recoverable=False), task=Task(spares=2, slack=None))
    assert ok and cell.spares == 1
    assert not play(fallen("leaning"), task=Task(spares=0, slack=None))[0]


def test_the_planner_is_asked_only_when_both_choices_exist_and_can_replace():
    asked = []

    def replace(d: Decision) -> bool:
        asked.append(d)
        return True
    assert play(fallen(), replace)[0] and not asked          # no spare: nothing to decide
    ok, cell = play(fallen(), replace, task=Task(spares=2, slack=None))
    assert ok and len(asked) == 1 and not asked[0].retry and cell.spares == 1
    assert asked[0].features.shape == (SymbolicCell.N_FEATURES,)


def test_after_a_failed_rescue_the_planner_is_asked_again():
    cal = calibration()
    cal.odds[("pick", "loose", "upright")] = 0.0
    asked = []
    ok, _ = play(fallen(), lambda d: asked.append(d.retry) or d.retry, cal,
                 Task(spares=2, slack=None))
    assert ok and asked == [False, True]          # rescue, it fails, then replace


def test_a_deadline_counts_only_orders_completed_in_time():
    t = undisturbed_time(ORDER, calibration())
    assert play(None, deadline=t)[0] and not play(None, deadline=t - 1)[0]


def test_the_rule_family():
    assert len(RULE_FAMILY) == 32 and len(set(RULE_FAMILY)) == 32
    rule = Rule(frozenset({("lying", True)}))
    d = Decision(Loose("red", "lying", True, True), False, np.zeros(SymbolicCell.N_FEATURES))
    assert not RESCUE_ALL(d) and rule(d)
    assert not rule(Decision(d.cube, True, d.features))   # after a failure: retry
    assert "replace lying near a tower" in str(rule)


def clutter_cal(under_6cm: float = 0.0, clear: float = 1.0) -> Calibration:
    from polypanda.supervisor.symbolic import CLEAR_BAND, CLUTTER_BANDS

    cal = calibration()
    cal.clutter = {CLUTTER_BANDS[0][1]: under_6cm, CLUTTER_BANDS[1][1]: 1.0, CLEAR_BAND: 1.0}
    cal.clear_odds = {"upright": clear, "lying": clear}
    cal.times["clear"] = 7.0
    return cal


def at(da: float):
    """A fallen upright cube at the station radius, `da` degrees from the knocked tower."""
    return [{"pose": "upright", "recoverable": True, "near": True, "r": 0.5, "da": da}]


def test_a_cube_left_for_a_spare_stays_on_the_floor_and_gets_in_the_way():
    """v3.1: with a clutter model, a cube left where it fell (3 deg from tower A: 2.6 cm) makes
    the places on that tower riskier (here: certain to fail); rescued, cleared or far away, it
    does not."""
    from polypanda.supervisor.symbolic import CLEAR, LEAVE

    spares = Task(spares=2, slack=None)
    ok, cell = play(at(3.0), lambda d: LEAVE, clutter_cal(), spares)
    assert not ok and len(cell.floor) == 1
    assert cell.nearest_floor_cube(0) < 0.06 and cell.nearest_floor_cube(1) > 0.2
    assert play(at(3.0), lambda d: LEAVE, calibration(), spares)[0]      # (v3.0: no clutter)
    assert play(at(40.0), lambda d: LEAVE, clutter_cal(), spares)[0]     # far from tower A
    assert play(at(3.0), None, clutter_cal(), spares)[0]                 # rescued: moved away
    ok, cell = play(at(3.0), lambda d: CLEAR, clutter_cal(), spares)
    assert ok and not cell.floor and (cell.cleared, cell.cleared_ok) == (1, 1)
    t_leave = play(at(40.0), lambda d: LEAVE, clutter_cal(), spares)[1].t
    assert cell.t == t_leave + 7.0                                       # the clear's time
    ok, cell = play(at(3.0), lambda d: CLEAR, clutter_cal(clear=0.0), spares)
    assert not ok and len(cell.floor) == 1 and (cell.cleared, cell.cleared_ok) == (1, 0)


def test_the_planner_sees_how_near_the_cube_lies_to_a_tower_that_still_needs_cubes():
    asked = []
    play(at(3.0), lambda d: asked.append(d) or False, clutter_cal(), Task(spares=2, slack=None))
    f = asked[0].features
    assert f.shape == (15,) and SymbolicCell.N_FEATURES == 15
    assert f[13] == pytest.approx(2 * 0.5 * np.sin(np.radians(1.5)) / 0.20, abs=1e-4)
    assert f[14] == pytest.approx(2 / 4)       # tower A: 1 of its 3 cubes stands


def test_the_clutter_model_is_fitted_from_a_calibration_run(tmp_path):
    """add_clutter: place odds factor per band (Laplace), clear odds by pose, and the clear's
    time from what the order durations leave over at the calibration's other times."""
    import csv

    from polypanda.supervisor.symbolic import CLEAR_BAND, CLUTTER_BANDS

    cols = ["skill", "success", "failure", "order", "source", "pose", "radius", "angle",
            "loose_before", "t0", "level"]
    near, tower = (0.5, -37.0), (0.5, -40.0)               # tower A at -40 deg; 2.6 cm apart

    def row(skill, order, t0, at, success="True", loose=0, level=0, source="", pose="upright"):
        return dict(zip(cols, [skill, success, "" if success == "True" else "toppled", order,
                               source, pose, *at, loose, t0, level], strict=True))
    rows = [row("knock", 0, "", near, success=""),
            row("place", 0, 1, tower), row("place", 0, 2, tower, "False", loose=1),
            row("clear", 0, 3, near, source="clear"), row("place", 0, 4, tower),
            row("place", 1, 1, tower)]
    path = tmp_path / "run_seed0_skills.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(tmp_path / "run_seed0_episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["index", "rejected", "sim_s"])
        w.writeheader()
        w.writerows([{"index": 0, "rejected": 0, "sim_s": 5 + 3 * 10 + 12},
                     {"index": 1, "rejected": 0, "sim_s": 5 + 10}])
    cal = calibration().add_clutter([path])
    assert cal.clutter[CLUTTER_BANDS[0][1]] == pytest.approx((1 / 3) / (4 / 5))
    assert cal.clutter[CLEAR_BAND] == 1.0
    assert cal.clear_odds == {"upright": pytest.approx(2 / 3)}
    assert cal.times["clear"] == pytest.approx(12.0)
    assert cal.odds == calibration().odds                   # every other odds stays
    assert cal.clutter_counts["places"] == {CLUTTER_BANDS[0][1]: 1, CLUTTER_BANDS[1][1]: 0,
                                            CLEAR_BAND: 3}


def test_the_time_spread_is_fitted_from_calibration_runs(tmp_path):
    """add_time_spread: squared duration residuals (at the mean times) fitted on the action
    counts and a constant: 1 belt pick, residual 3 s; 2 picks, sqrt 13; none, sqrt 5 -> a belt
    pick's variance 4, the overhead's 5."""
    import csv

    cols = ["skill", "success", "order", "source", "pose"]
    picks = [{"skill": "pick", "success": "True", "order": o, "source": "belt", "pose": "upright"}
             for o in (0, 1, 1)]
    path = tmp_path / "run_seed0_skills.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(picks)
    with open(tmp_path / "run_seed0_episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["index", "rejected", "sim_s"])
        w.writeheader()
        w.writerows([{"index": 0, "rejected": 0, "sim_s": 5 + 10 + 3},
                     {"index": 1, "rejected": 0, "sim_s": 5 + 20 + np.sqrt(13)},
                     {"index": 2, "rejected": 0, "sim_s": 5 + np.sqrt(5)}])
    cal = calibration().add_time_spread([path])
    assert cal.time_var["pick_belt"] == pytest.approx(4.0)
    assert cal.time_var["overhead"] == pytest.approx(5.0)
    assert all(v == pytest.approx(0.0, abs=1e-9) for a, v in cal.time_var.items()
               if a not in ("pick_belt", "overhead"))
    assert cal.times == calibration().times and cal.odds == calibration().odds
    assert calibration().time_var is None


def test_a_time_spread_varies_durations_but_not_skill_outcomes():
    """With a spread, each action's time is noisy (its own stream), so the skills' outcomes are
    those of the same cell without it; undisturbed times (deadlines) stay fixed."""
    from dataclasses import replace

    cal = calibration(0.8)
    noisy = replace(cal, time_var={a: 4.0 for a in (*ACTIONS, "overhead")})
    times = []
    for seed in range(40):
        a = SymbolicCell(ORDER, KNOCK, fallen(), cal, np.random.default_rng(seed), NO_SPARES)
        b = SymbolicCell(ORDER, KNOCK, fallen(), noisy, np.random.default_rng(seed), NO_SPARES)
        assert a.result() == b.result() and a.towers == b.towers and a.failed == b.failed
        times.append(b.t - a.t)
    assert np.std(times) > 1.0 and abs(np.mean(times)) < 3.0
    assert undisturbed_time(ORDER, noisy) == undisturbed_time(ORDER, cal)


def test_clear_rules_answer_by_pose_and_distance_band():
    from polypanda.supervisor.symbolic import (
        CLEAR,
        CLEAR_RULE_FAMILY,
        LEAVE,
        RESCUE,
        TOWER_XY,
        ClearRule,
        cube_band,
    )

    assert len(CLEAR_RULE_FAMILY) == 3 ** 7 and len(set(CLEAR_RULE_FAMILY)) == 3 ** 7
    rule = ClearRule((RESCUE, RESCUE, RESCUE, CLEAR, LEAVE, RESCUE), after=LEAVE)
    t = TOWER_XY[0]

    def ask(pose, offset, retry=False, staged=False):
        cube = Loose("red", pose, True, True, xy=t + [offset, 0.0], staged=staged)
        return rule(Decision(cube, retry, np.zeros(SymbolicCell.N_FEATURES)))
    assert [cube_band(Loose("red", "lying", True, True, xy=t + [d, 0.0]))
            for d in (0.03, 0.1, 0.3)] == [0, 1, 2]
    assert ask("lying", 0.03) == CLEAR and ask("lying", 0.1) == LEAVE
    assert ask("lying", 0.3) == RESCUE and ask("upright", 0.03) == RESCUE
    assert ask("lying", 0.03, staged=True) == RESCUE         # staged: away from the towers
    assert ask("lying", 0.03, retry=True) == LEAVE
    assert "lying under 6 cm: clear" in str(rule)


def test_random_choices_repeat_per_order_and_cover_every_answer():
    """v3.1 calibration: rescue, leave or clear at equal odds, the same stream for the same
    order whatever ran before it."""
    from polypanda.supervisor.symbolic import CLEAR, LEAVE, RESCUE, RandomChoice

    d = Decision(Loose("red", "lying", True, True), False, np.zeros(SymbolicCell.N_FEATURES))
    policy = RandomChoice()
    policy.reset(4)
    first = [policy(d) for _ in range(9)]
    policy.reset(5)
    policy(d)
    policy.reset(4)
    assert [policy(d) for _ in range(9)] == first
    draws = [policy(d) for _ in range(3000)]
    assert {RESCUE, LEAVE, CLEAR} == set(draws)
    assert all(abs(draws.count(c) / 3000 - 1 / 3) < 0.03 for c in (RESCUE, LEAVE, CLEAR))
    assert "rescue / leave / clear" in str(policy)


def test_the_planner_env_plays_orders_decision_by_decision():
    """v3 f: one episode per order, one step per decision, reward 1 for an order completed by
    its deadline; orders with nothing to decide are skipped; `blind` hides observation parts."""
    from polypanda.environments.recovery_planner import RecoveryPlannerEnv

    cal = calibration(0.9)
    cal.by_height.update({h: [fallen("lying", near=True) * min(h, 2)] for h in (2, 3, 4)})
    env = RecoveryPlannerEnv(cal, Task(spares=2, slack=2.0), blind=("clock",))
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape and obs[5] == obs[6] == 0.0
    returns = []
    for _ in range(20):
        done, steps = False, 0
        while not done:
            obs, reward, done, _, info = env.step(env.action_space.sample())
            steps += 1
        assert reward in (0.0, 1.0) and info["is_success"] == bool(reward) and steps <= 8
        returns.append(reward)
        env.reset()
    assert 0 < sum(returns) <= 20
