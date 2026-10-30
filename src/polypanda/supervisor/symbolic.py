"""The work-order cell without physics (v3 step e): a symbolic cell for the learned recovery
planner.

It plays an order as ScriptedCell.run_mixed_order does with recovery on (3 buffer slots, no
batch choice): the greedy planner routes every arrival; after a knock the cell recovers loose
cubes as recover_loose does (cubes lying or near a tower are first moved to a staging spot, then
picked and routed; two attempts per cube; the rest written off). Every skill attempt succeeds with
the odds measured in the physics cell on the calibration orders (evaluate_cell's _skills.csv),
by situation, and costs the time fitted on those orders' durations. A knock throws off the top
cubes of its tower, which come to rest as a calibration knock left them.

Spares (v3 f): a written-off cube the towers still need is replaced by a spare of its colour,
which rides the belt like an arrival, while spares last. The planner's decisions: when a fallen
cube could be rescued and a spare could replace it, rescue or replace; after a failed rescue
attempt, retry or replace. `play` yields a Decision at each, and is sent True to replace.
"""

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from polypanda.simulation.workcell import WorkcellConfig, WorkOrder, station_xy
from polypanda.supervisor.planner import BUFFER, TOWER, GreedyPlanner

_CFG = WorkcellConfig()
TOWER_XY = [station_xy(a, _CFG.radius) for a in _CFG.tower_angles]
BUFFER_XY = [station_xy(a, _CFG.radius) for a in _CFG.buffer_angles]
CLEAR_RADIUS = 0.20   # m, as CellControllerConfig.clear_radius: closer cubes are staged first
STAGING_SPOTS = 5     # as CellControllerConfig.staging
ATTEMPTS = 2          # rescue attempts per cube, as recover_loose
ACTIONS = ("pick_belt", "pick_buffer", "pick_up", "lying", "place_tower", "place_buffer",
           "place_stage", "reject")
# the calibration run (step e, fcefde0): all learned skills, 3 seeds, 200 calibration orders
CALIBRATION_SUITE = Path("eval_sets/disturbance_calib_v1.json")
CALIBRATION_RUNS = "results/cell/*disturbance_calib_v1_recover_n200_skills.csv"


def _xy(row) -> np.ndarray:
    a = np.radians(float(row["angle"]))
    return float(row["radius"]) * np.array([np.cos(a), np.sin(a)])


def target_kind(xy) -> str:
    """Where a place puts its cube: a tower, a buffer slot or a staging spot."""
    if min(np.linalg.norm(xy - t) for t in TOWER_XY) < 0.03:
        return "tower"
    if min(np.linalg.norm(xy - b) for b in BUFFER_XY) < 0.03:
        return "buffer"
    return "stage"


def near_tower(xy) -> bool:
    return bool(min(np.linalg.norm(xy - t) for t in TOWER_XY) < CLEAR_RADIUS)


def situation(row) -> tuple:
    """The situation a skill attempt's odds are kept by (a _skills.csv row)."""
    if row["skill"] == "pick":
        source = row["source"] if row["source"] in ("belt", "buffer") else "loose"
        return ("pick", source, row["pose"])
    if row["skill"] == "place":
        kind = target_kind(_xy(row))
        return ("place", kind, int(row["level"]) if kind == "tower" else 0)
    return ("standup",)


def action(row) -> str | None:
    """Which timed action a skill attempt is (a lying cube's grasp is part of its stand-up)."""
    if row["skill"] == "pick":
        if row["source"] in ("belt", "buffer"):
            return f"pick_{row['source']}"
        return "pick_up" if row["pose"] == "upright" else None
    if row["skill"] == "standup":
        return "lying"
    if row["skill"] == "place":
        return "place_" + target_kind(_xy(row))
    return None


def _time_key(key: tuple) -> str | None:
    if key[0] == "pick":
        return {"belt": "pick_belt", "buffer": "pick_buffer"}.get(
            key[1], "pick_up" if key[2] == "upright" else None)
    return "lying" if key[0] == "standup" else f"place_{key[1]}"


@dataclass
class Calibration:
    """Odds per situation (Laplace-smoothed), the time each action costs, and knock outcomes."""
    odds: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    times: dict = field(default_factory=dict)
    knocks: dict = field(default_factory=dict)   # (seed, order) -> fallen cubes' rest states
    by_height: dict = field(default_factory=lambda: defaultdict(list))  # knocked height -> those

    def p(self, key: tuple) -> float:
        return self.odds[key]

    def perfect(self) -> "Calibration":
        """Every skill always succeeds (undisturbed times; an ablation)."""
        return Calibration({k: 1.0 for k in self.odds}, self.counts, self.times, self.knocks,
                           self.by_height)

    @classmethod
    def from_calibration_run(cls, root: Path = Path()) -> "Calibration":
        return cls.load(sorted(root.glob(CALIBRATION_RUNS)), root / CALIBRATION_SUITE)

    @classmethod
    def load(cls, skills_csvs: list[Path], suite: Path) -> "Calibration":
        """From evaluate_cell's _skills.csv files of runs on `suite` (each with its
        _episodes.csv): one file per seed, named ..._seed<k>_..."""
        cal, ok = cls(), defaultdict(list)
        heights = [r and r["height"] for r in json.loads(Path(suite).read_text())["orders"]]
        X, y = [], []
        for path in sorted(skills_csvs):
            seed = int(str(path).split("_seed")[1][0])
            fallen, used = defaultdict(list), defaultdict(Counter)
            for r in csv.DictReader(open(path)):
                if r["skill"] == "knock":
                    fallen[int(r["order"])].append(
                        {"pose": r["pose"], "recoverable": r["recoverable"] == "True",
                         "near": near_tower(_xy(r))})
                    continue
                ok[situation(r)].append(r["success"] == "True")
                if (a := action(r)) is not None:
                    used[int(r["order"])][a] += 1
            for i, states in fallen.items():
                cal.knocks[(seed, i)] = states
                cal.by_height[heights[i]].append(states)
            episodes = Path(str(path).replace("_skills.csv", "_episodes.csv"))
            for e in csv.DictReader(open(episodes)):
                i = int(e["index"])
                if heights[i] is not None and i not in fallen:   # (a knock that threw nothing)
                    cal.knocks[(seed, i)] = []
                    cal.by_height[heights[i]].append([])
                c = used[i]
                c["reject"] = int(e["rejected"])
                X.append([c[a] for a in ACTIONS] + [1.0])
                y.append(float(e["sim_s"]))
        cal.odds = {k: (sum(v) + 1) / (len(v) + 2) for k, v in ok.items()}
        cal.counts = {k: len(v) for k, v in ok.items()}
        coef, *_ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)
        cal.times = dict(zip(ACTIONS + ("overhead",), map(float, coef), strict=True))
        return cal


@dataclass
class Loose:
    """A fallen cube."""
    colour: str
    pose: str            # upright, lying or leaning
    recoverable: bool
    near: bool           # within CLEAR_RADIUS of a tower
    index: int = 0       # which of the knock's fallen cubes (top first)
    attempts: int = 0
    staged: bool = False
    asked: int = -1      # attempts when the planner was last asked about it


def decision_features(cube: Loose, retry: bool, spares: int, max_spares: int, t: float,
                      deadline: float | None, queue: int, planner: GreedyPlanner,
                      loose_poses: list[str], times: dict) -> np.ndarray:
    """What the planner sees at a decision, scaled to about [0, 1]; the symbolic and the physics
    cell (ScriptedCell.run_mixed_order) both build it here. `queue`: arrivals (and spares) still
    to come up the belt; `loose_poses`: every fallen cube still loose, this one included."""
    horizon = deadline or 1e9
    left = sum(len(r) - p for r, p in zip(planner.recipes, planner.placed, strict=True))
    work = left * (times["pick_belt"] + times["place_tower"])
    ready = any(planner.next_colour(k) == cube.colour for k in range(len(planner.recipes)))
    return np.array([
        cube.pose == "lying", cube.near, retry, cube.staged, spares / max(1, max_spares),
        t / horizon, (horizon - t - work) / horizon, queue / 14, left / 12,
        loose_poses.count("lying") / 4, loose_poses.count("upright") / 4, ready,
        deadline is not None], dtype=np.float32)


@dataclass
class Decision:
    """A question for the planner: replace `cube` with a spare (True) or rescue it (False)?"""
    cube: Loose
    retry: bool          # asked after a failed rescue attempt
    features: np.ndarray


@dataclass(frozen=True)
class Task:
    spares: int = 2
    slack: float = 1.5           # deadline: slack x the order's undisturbed time (None: none)
    spare_delay: float = 0.0     # s a spare costs on top of a belt arrival


DEFAULT_TASK = Task()  # the learned planner's task (docs/v3_targets.md, 2026-10-30)


class SymbolicCell:
    """One order. `play()` is a generator: it yields a Decision whenever the planner has a
    choice and must be sent True (replace) or False (rescue); it returns whether the order was
    completed (by the deadline, if any). `result()` drives it with a policy."""

    N_FEATURES = 13

    def __init__(self, order: WorkOrder, knock, fallen: list[dict] | None, cal: Calibration,
                 rng: np.random.Generator, task: Task = DEFAULT_TASK,
                 deadline: float | None = None) -> None:
        self.order, self.knock, self.fallen, self.cal, self.rng = order, knock, fallen, cal, rng
        self.task, self.deadline = task, deadline
        self.planner = GreedyPlanner(order.recipes, slots=3)
        self.towers: list[list[str]] = [[] for _ in order.recipes]
        self.queue = [(c, False) for c, _ in order.arrivals]
        self.loose: list[Loose] = []
        self.slot: dict[int, str] = {}
        self.spares, self.staged, self.failed, self.knocked = task.spares, 0, False, False
        self.t = cal.times["overhead"]

    # --- skills -----------------------------------------------------------------------------
    def roll(self, key: tuple) -> bool:
        self.t += self.cal.times.get(_time_key(key) or "", 0.0)
        return bool(self.rng.random() < self.cal.p(key))

    def needed(self, colour: str) -> bool:
        return self.planner.demand(colour) > 0

    def features(self, c: Loose, retry: bool) -> np.ndarray:
        return decision_features(
            c, retry, self.spares, self.task.spares, self.t, self.deadline, len(self.queue),
            self.planner, [x.pose for x in self.loose], self.cal.times)

    # --- the order --------------------------------------------------------------------------
    def play(self) -> Generator[Decision, bool, bool]:
        while self.queue and not self.failed:
            colour, spare = self.queue.pop(0)
            if spare:
                self.t += self.task.spare_delay
            if not self.roll(("pick", "belt", "upright")):
                return False
            yield from self.route(colour)
            if not self.queue and not self.failed:
                yield from self.recover()
        return (not self.failed
                and all(len(t) == len(r) for t, r in zip(self.towers, self.order.recipes,
                                                         strict=True))
                and (self.deadline is None or self.t <= self.deadline))

    def result(self, policy: Callable[[Decision], bool] | None = None) -> bool:
        """Play the order with `policy` (None: always rescue)."""
        game = self.play()
        try:
            decision = next(game)
            while True:
                decision = game.send(bool(policy and policy(decision)))
        except StopIteration as end:
            return end.value

    def stack(self, colour: str, t: int):
        if not self.roll(("place", "tower", len(self.towers[t]))):
            self.failed = True
            return
        self.towers[t].append(colour)
        k = self.knock
        if k is not None and not self.knocked and t == k.tower and len(self.towers[t]) == k.height:
            self.knocked = True
            fallen = self.fallen or []
            for j, (st, c) in enumerate(zip(fallen, reversed(self.towers[t]), strict=False)):
                self.loose.append(Loose(c, st["pose"], st["recoverable"], st["near"], j))
            del self.towers[t][len(self.towers[t]) - len(fallen):]
            self.planner.placed[t] = len(self.towers[t])
            yield from self.recover()

    def route(self, colour: str):
        kind, where = self.planner.decide(colour)
        if kind == TOWER:
            yield from self.stack(colour, where)
            while not self.failed and (move := self.planner.next_buffer_move()) is not None:
                slot, t = move
                if not self.roll(("pick", "buffer", "upright")):
                    self.failed = True
                    return
                yield from self.stack(self.slot.pop(slot), t)
        elif kind == BUFFER:
            if not self.roll(("place", "buffer", 0)):
                self.failed = True
                return
            self.slot[where] = colour
        else:
            self.t += self.cal.times["reject"]

    def write_off(self, c: Loose) -> None:
        self.loose.remove(c)
        if c.staged:
            self.staged -= 1
        if self.needed(c.colour) and self.spares > 0:
            self.spares -= 1
            self.queue.append((c.colour, True))

    def recover(self):
        while not self.failed:
            for c in list(self.loose):
                if not c.recoverable or c.pose not in ("upright", "lying") \
                        or c.attempts >= ATTEMPTS:
                    self.write_off(c)
                elif (self.spares > 0 and self.needed(c.colour) and c.asked != c.attempts
                      and not (c.staged and c.attempts == 0)):
                    c.asked = c.attempts
                    retry = c.attempts > 0
                    if (yield Decision(c, retry, self.features(c, retry))):
                        self.write_off(c)
            if not self.loose:
                return
            near = [c for c in self.loose if not c.staged and (c.pose == "lying" or c.near)]
            if near and self.staged < STAGING_SPOTS:
                c = near[0]
                if c.pose == "lying":
                    ok = self.roll(("pick", "loose", "lying")) and self.roll(("standup",))
                else:
                    ok = self.roll(("pick", "loose", "upright")) and \
                        self.roll(("place", "stage", 0))
                if ok:
                    c.pose, c.near, c.staged = "upright", False, True
                    self.staged += 1
                else:
                    c.attempts += 1
                continue
            ready = [c for c in self.loose if any(self.planner.next_colour(t) == c.colour
                                                  for t in range(len(self.towers)))]
            c = (ready or self.loose)[0]
            if c.pose == "lying":       # (no staging spot free: stood up where it lies)
                if not (self.roll(("pick", "loose", "lying")) and self.roll(("standup",))):
                    c.attempts += 1
                    continue
                c.pose = "upright"
            if not self.roll(("pick", "loose", "upright")):
                c.attempts += 1
                continue
            self.loose.remove(c)
            if c.staged:
                self.staged -= 1
            yield from self.route(c.colour)


def undisturbed_time(order: WorkOrder, cal: Calibration) -> float:
    """An order's time with every skill succeeding and no knock."""
    cell = SymbolicCell(order, None, None, cal.perfect(), np.random.default_rng(0),
                        Task(spares=0))
    cell.result()
    return cell.t


# --- hand-written rules -------------------------------------------------------------------------
SITUATIONS = (("upright", False), ("upright", True), ("lying", False), ("lying", True))


@dataclass(frozen=True)
class Rule:
    """Replace a fallen cube in one of the `replace` situations (pose, near a tower) instead of
    rescuing it; after a failed rescue attempt, replace if `after_failure`, else retry."""
    replace: frozenset = frozenset()
    after_failure: bool = False

    def __call__(self, d: Decision) -> bool:
        if d.retry:
            return self.after_failure
        return (d.cube.pose, d.cube.near) in self.replace

    def __str__(self) -> str:
        what = ", ".join(f"{p}{' near a tower' if n else ''}" for p, n in sorted(self.replace))
        return (f"replace {what or 'nothing'}; after a failed attempt "
                f"{'replace' if self.after_failure else 'retry'}")


RESCUE_ALL = Rule()
RULE_FAMILY = tuple(Rule(frozenset(s for s, b in zip(SITUATIONS, bits, strict=True) if b), after)
                    for bits in np.ndindex(2, 2, 2, 2) for after in (False, True))
# the best of RULE_FAMILY on the selection orders (results/planner/symbolic_eval.json, 52111b3)
BEST_RULE = Rule(frozenset({("lying", False), ("lying", True), ("upright", True)}), False)


def load_planner(run_dir: Path) -> Callable[[Decision], bool]:
    """A trained recovery planner (scripts/train_recovery_planner.py) as a policy: True to
    replace. It sees what it was trained to see (its run's `blind`)."""
    from stable_baselines3 import PPO

    from polypanda.environments.recovery_planner import HIDDEN

    model = PPO.load(Path(run_dir) / "model.zip", device="cpu")
    meta = json.loads((Path(run_dir) / "meta.json").read_text())
    hidden = [i for b in meta["blind"] for i in HIDDEN[b]]

    def policy(d: Decision) -> bool:
        obs = np.clip(d.features, -2.0, 2.0)
        obs[hidden] = 0.0
        return int(model.predict(obs, deterministic=True)[0]) == 1
    return policy
