"""The Layer 2 greedy planner and mixed work orders (pure logic, no physics)."""

from collections import Counter

import numpy as np

from polypanda.supervisor.planner import (
    BUFFER,
    REJECT,
    TOWER,
    GreedyPlanner,
    mixed_order,
    simulate_routing,
    slots_needed,
)

RECIPES = (("red", "green"), ("blue",), ("yellow",))


def test_routes_to_tower_buffer_and_reject():
    p = GreedyPlanner(RECIPES, slots=3)
    assert p.decide("green") == (BUFFER, 0)           # tower A needs it, after a red
    assert p.decide("yellow") == (TOWER, 2)
    assert p.next_buffer_move() is None               # green still waits for the red
    assert p.decide("red") == (TOWER, 0)
    assert p.next_buffer_move() == (0, 0)              # now the green goes onto A
    assert p.next_buffer_move() is None
    assert p.decide("blue") == (TOWER, 1)
    assert p.complete
    assert p.decide("red") == (REJECT, None)           # surplus
    assert p.needed_rejected == 0


def test_does_not_buffer_more_than_still_needed():
    p = GreedyPlanner((("red", "green"), ("blue",), ("yellow",)), slots=3)
    assert p.decide("green") == (BUFFER, 0)
    assert p.decide("green") == (REJECT, None)  # one green is needed, one is already parked
    assert p.needed_rejected == 0               # a surplus, not a loss


def test_full_buffer_rejects_a_needed_cube():
    p = GreedyPlanner((("red", "green", "blue"), ("yellow",), ("red",)), slots=1)
    assert p.decide("green") == (BUFFER, 0)
    assert p.decide("blue") == (REJECT, None)
    assert p.needed_rejected == 1


def test_prefers_the_tower_with_more_to_build():
    p = GreedyPlanner((("red",), ("red", "green", "blue"), ("yellow",)), slots=3)
    assert p.decide("red") == (TOWER, 1)


def test_batch_choice_takes_a_cube_a_tower_needs_next():
    p = GreedyPlanner(RECIPES, slots=3)
    assert p.choose(["green", "yellow", "red"]) == 1  # tower C needs yellow now
    assert p.choose(["green", "green"]) == 0          # none needed now: the front cube
    p.decide("yellow")
    assert p.choose(["green", "red"]) == 1


def test_symbolic_routing_on_the_layer2_evaluation_orders():
    """The routing the cell gives the 200 Layer 2 evaluation orders (docs/v3_targets.md): the
    physics cell with scripted skills completed 64, 144 and 200 of them front cube first."""
    from polypanda.evaluation.cell import CellEvalConfig
    cfg = CellEvalConfig(mixed=True)
    orders = []
    for i in range(cfg.n):
        order, _ = mixed_order(np.random.default_rng([cfg.seed, i]), cfg.heights, cfg.surplus,
                               cfg.solvable_with)
        orders.append((order.recipes, [c for c, _ in order.arrivals]))

    def completed(slots, batch_choice):
        return sum(simulate_routing(r, c, slots, batch_choice=batch_choice)["complete"]
                   for r, c in orders)

    assert [completed(s, False) for s in (0, 1, 3)] == [64, 144, 200]
    assert [completed(s, True) for s in (0, 1, 3)] == [106, 170, 200]


def test_enough_slots_guarantee_completion():
    rng = np.random.default_rng(0)
    short = 0
    for _ in range(300):
        order, _ = mixed_order(rng, solvable_with=6)
        colours = [c for c, _ in order.arrivals]
        need = slots_needed(order.recipes, colours)
        for slots in range(4):
            p = GreedyPlanner(order.recipes, slots=slots)
            for c in colours:
                p.decide(c)
                while p.next_buffer_move() is not None:
                    pass
            if need <= slots:
                assert p.complete and p.needed_rejected == 0
            else:
                assert p.needed_rejected > 0
                short += not p.complete
    assert short > 0  # too small a buffer does lose orders (unless a surplus cube rescues it)


def test_mixed_orders_are_solvable_and_hold_the_needed_cubes_plus_surplus():
    rng = np.random.default_rng(1)
    for _ in range(100):
        order, _ = mixed_order(rng, heights=(1, 4), surplus=(0, 2), solvable_with=3)
        needed = Counter(c for r in order.recipes for c in r)
        arriving = Counter(c for c, _ in order.arrivals)
        assert all(arriving[c] >= n for c, n in needed.items())
        assert 0 <= arriving.total() - needed.total() <= 2
        assert all(t is None for _, t in order.arrivals)
        assert slots_needed(order.recipes, [c for c, _ in order.arrivals]) <= 3
