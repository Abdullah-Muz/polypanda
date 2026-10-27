"""Greedy routing of arriving cubes for the work-order cell (Milestone 9, Layer 2).

Cubes arrive in random order, including surplus cubes no tower needs. For each arriving cube the
planner decides, in this order of preference:

1. a tower whose next cube is this colour: place it there (several: the one with the most cubes
   still to build, then the first);
2. a colour some tower still needs later, beyond what the buffer already holds, and a free
   buffer slot: park it in the buffer;
3. otherwise: the reject bin (surplus, or the buffer is full).

After every tower placement, buffered cubes that have become placeable are moved to their towers
(`next_buffer_move`, repeated until it returns None). The planner tracks the plan, not the
physics: the controller reports nothing back, and a physical failure shows up in the final
tower check.

With batch choice (v3), the robot may take any cube of the current belt batch, not only the
front one (`choose`): a cube some tower needs next if the belt has one, otherwise the front cube.
That one rule completes 53 % instead of 32 % of the Layer 2 evaluation orders without a buffer,
within 3-6.5 points of a planner that knew every future arrival (docs/v3_targets.md).

With an unlimited buffer the greedy plan always completes an order. With `slots` slots it is
guaranteed to when the unlimited plan never holds more than `slots` cubes at once
(`slots_needed`), so work orders can be generated solvable for a given buffer. (It may also
complete with fewer: a surplus cube arriving later can replace a needed cube that was rejected.)
"""

from dataclasses import dataclass, field

import numpy as np

from polypanda.simulation.workcell import COLOURS, WorkOrder

TOWER, BUFFER, REJECT = "tower", "buffer", "reject"


@dataclass
class GreedyPlanner:
    recipes: tuple[tuple[str, ...], ...]
    slots: int | None = 3                # None: unlimited
    placed: list[int] = field(init=False)
    buffer: list[str | None] = field(init=False)
    needed_rejected: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.placed = [0] * len(self.recipes)
        self.buffer = [] if self.slots is None else [None] * self.slots

    def next_colour(self, t: int) -> str | None:
        r = self.recipes[t]
        return r[self.placed[t]] if self.placed[t] < len(r) else None

    def demand(self, colour: str) -> int:
        """Cubes of this colour the towers still need, beyond those already buffered."""
        still = sum(r[self.placed[t]:].count(colour) for t, r in enumerate(self.recipes))
        return still - self.buffer.count(colour)

    def _tower_for(self, colour: str) -> int | None:
        ready = [t for t in range(len(self.recipes)) if self.next_colour(t) == colour]
        if not ready:
            return None
        return max(ready, key=lambda t: (len(self.recipes[t]) - self.placed[t], -t))

    def decide(self, colour: str) -> tuple[str, int | None]:
        """Route an arriving cube: (TOWER, tower), (BUFFER, slot) or (REJECT, None)."""
        t = self._tower_for(colour)
        if t is not None:
            self.placed[t] += 1
            return TOWER, t
        if self.demand(colour) > 0:
            if self.slots is None:
                self.buffer.append(colour)
                return BUFFER, len(self.buffer) - 1
            if None in self.buffer:
                slot = self.buffer.index(None)
                self.buffer[slot] = colour
                return BUFFER, slot
            self.needed_rejected += 1
        return REJECT, None

    def choose(self, colours: list[str]) -> int:
        """Batch choice: which of the cubes on the belt (front first) to take next."""
        for j, colour in enumerate(colours):
            if self._tower_for(colour) is not None:
                return j
        return 0

    def next_buffer_move(self) -> tuple[int, int] | None:
        """A buffered cube that can go onto its tower now: (slot, tower), or None."""
        for slot, colour in enumerate(self.buffer):
            if colour is None:
                continue
            t = self._tower_for(colour)
            if t is not None:
                self.buffer[slot] = None
                self.placed[t] += 1
                return slot, t
        return None

    @property
    def held(self) -> int:
        return sum(c is not None for c in self.buffer)

    @property
    def complete(self) -> bool:
        return all(p == len(r) for p, r in zip(self.placed, self.recipes, strict=True))


def simulate_routing(recipes, colours, slots: int | None = 3, batch: int = 3,
                     batch_choice: bool = False) -> dict:
    """The routing a work order gets in the cell, without physics: cubes arrive in belt batches
    of `batch`; the planner takes the front cube, or with batch choice any cube of the batch.
    Returns whether the plan completes the order, and its buffer and reject counts. The physics
    cell with scripted skills reproduces it (tests/test_planner.py, tests/test_workcell.py)."""
    planner = GreedyPlanner(recipes, slots=slots)
    counts = {"buffered": 0, "rejected": 0, "taken_out_of_order": 0}
    for start in range(0, len(colours), batch):
        belt = list(colours[start:start + batch])
        while belt:
            j = planner.choose(belt) if batch_choice else 0
            counts["taken_out_of_order"] += j > 0
            kind, _ = planner.decide(belt.pop(j))
            counts["buffered"] += kind == BUFFER
            counts["rejected"] += kind == REJECT
            while planner.next_buffer_move() is not None:
                pass
    return {"complete": planner.complete, "needed_rejected": planner.needed_rejected, **counts}


def slots_needed(recipes, colours) -> int:
    """Most cubes the greedy plan holds in the buffer at once for this arrival sequence (with an
    unlimited buffer); the order is solvable with at least that many slots."""
    planner, most = GreedyPlanner(recipes, slots=None), 0
    for colour in colours:
        planner.decide(colour)
        while planner.next_buffer_move() is not None:
            pass
        most = max(most, planner.held)
    return most


def mixed_order(rng: np.random.Generator, heights: tuple[int, int] = (1, 4),
                surplus: tuple[int, int] = (0, 2), solvable_with: int = 3,
                colours: tuple[str, ...] = tuple(COLOURS), towers: int = 3
                ) -> tuple[WorkOrder, int]:
    """A Layer 2 work order: random recipes (as WorkOrder.random), the cubes they need plus
    `surplus` extra cubes of random colours, arriving in random order. Arrival orders the greedy
    plan cannot complete with `solvable_with` buffer slots are reshuffled (the recipes and
    surplus are kept). Returns the order (arrivals as (colour, None): the planner routes them)
    and how many shuffles were rejected."""
    recipes = WorkOrder.random(rng, heights, colours, towers).recipes
    extra = [str(c) for c in rng.choice(colours, size=rng.integers(surplus[0], surplus[1] + 1))]
    cubes = [c for r in recipes for c in r] + extra
    rejected = 0
    while True:
        sequence = [cubes[i] for i in rng.permutation(len(cubes))]
        if slots_needed(recipes, sequence) <= solvable_with:
            return WorkOrder(recipes, tuple((c, None) for c in sequence)), rejected
        rejected += 1
