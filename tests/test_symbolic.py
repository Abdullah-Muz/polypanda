"""The symbolic cell (v3 step e): the physics cell's order flow and recovery as a board game, with
spares and the learned planner's rescue-or-replace decisions."""

import numpy as np

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
