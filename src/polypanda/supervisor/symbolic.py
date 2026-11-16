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

Clutter (v3.1, docs/v3_1_targets.md): a fallen cube rests where a calibration knock left it,
turned to the tower knocked here, and a cube left for a spare stays there on the floor. With a
clutter model (Calibration.add_clutter), a place on a tower succeeds with its odds times a
factor set by how near the nearest cube on the floor lies to that tower (CLUTTER_BANDS). The
planner may also answer CLEAR: the cube is taken to the reject bin (its own odds and time), and
off the floor if that succeeds, then replaced. Without a clutter model the cell plays exactly as
in v3.0.

Time spread (v3.1, amendment of 2026-11-15): with Calibration.add_time_spread, every action's
time gets Gaussian noise around its mean (its variance fitted on calibration runs), drawn from a
stream of its own; without it every action takes its mean time, as in v3.0.
"""

import csv
import itertools
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Generator
from dataclasses import dataclass, field, replace
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


# v3.1 clutter: the nearest cube on the floor to a tower, by distance band (scripts/
# clutter_exposure.py: under 6 cm places fail about 1 in 5; further out they barely do)
CLUTTER_BANDS = ((0.06, "under 6 cm"), (0.20, "6-20 cm"))
CLEAR_BAND = "none within 20 cm"
MATCH = 0.06   # m: a loose pick or a clear this close to a fallen cube's rest takes it away


def clutter_band(distance: float) -> str:
    for edge, name in CLUTTER_BANDS:
        if distance < edge:
            return name
    return CLEAR_BAND


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def tower_place_clutter(rows: list[dict]) -> list[dict]:
    """Every place on a tower in one order's _skills.csv rows, with the distance from that tower
    to the nearest knocked cube still on the table when the place started. Rows are taken in
    time order (t0): a knocked cube is on the table from the knock until a successful pick of a
    loose cube, or a clear, within MATCH of where it lay takes it away (a cube moved by an
    attempt that failed counts where it first came to rest). A place the cell itself saw with no
    other loose cube (loose_before = 0) has none (distance inf)."""
    loose = [_xy(r) for r in rows if r["skill"] == "knock"]
    out = []
    for r in sorted((r for r in rows if r["skill"] != "knock"), key=lambda r: float(r["t0"])):
        if ((r["skill"] == "pick" and r["source"] not in ("belt", "buffer")
             or r["skill"] == "clear") and r["success"] == "True" and loose):
            d = [float(np.linalg.norm(_xy(r) - c)) for c in loose]
            if min(d) < MATCH:
                loose.pop(int(np.argmin(d)))
        if r["skill"] == "place" and target_kind(_xy(r)) == "tower":
            tower = TOWER_XY[int(np.argmin([np.linalg.norm(_xy(r) - t) for t in TOWER_XY]))]
            seen = int(r["loose_before"] or 0) > 0
            d = min((float(np.linalg.norm(c - tower)) for c in loose), default=float("inf"))
            out.append({"ok": r["success"] == "True", "failure": r["failure"] or "timeout",
                        "nearest": d if seen else float("inf"), "level": int(r["level"])})
    return out


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
    """Which timed action a skill attempt is (a lying cube's grasp is part of its stand-up, a
    clear's grasp part of the clear)."""
    if row["skill"] == "clear":
        return "clear"
    if row["skill"] == "pick":
        if row["source"] == "clear":
            return None
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
    # v3.1 (add_clutter): tower-place odds factor per clutter band, the clear's odds by pose
    clutter: dict | None = None
    clear_odds: dict = field(default_factory=dict)
    clutter_counts: dict = field(default_factory=dict)   # places, clears behind the fit
    # v3.1 (add_time_spread): the variance (s^2) of each action's time; None: fixed times
    time_var: dict | None = None

    def p(self, key: tuple) -> float:
        return self.odds[key]

    def perfect(self) -> "Calibration":
        """Every skill always succeeds (undisturbed times; an ablation): clutter costs nothing,
        every clear succeeds, and every action takes its mean time."""
        return Calibration({k: 1.0 for k in self.odds}, self.counts, self.times, self.knocks,
                           self.by_height, None, {k: 1.0 for k in self.clear_odds},
                           self.clutter_counts)

    def add_clutter(self, skills_csvs: list[Path]) -> "Calibration":
        """v3.1 (docs/v3_1_targets.md, 2026-11-14): this calibration plus a clutter model and
        the clear action, from the _skills.csv (and _episodes.csv) files of the v3.1
        calibration run (random rescue / leave / clear). Only these come from that run:
        - per clutter band, the factor on a tower place's odds: its success rate there over the
          rate with no cube on the floor within 20 cm (Laplace-smoothed, pooled over levels);
        - the clear's odds by the cube's pose (ending in the bin, Laplace-smoothed);
        - the clear's time: what the orders' durations leave over after every other action at
          this calibration's times, fitted on the number of clears (least squares, no
          intercept). Every other odds and time stays this calibration's."""
        places, clears, n_clears, residual = defaultdict(list), defaultdict(list), [], []
        for path in sorted(skills_csvs):
            by_order, used = defaultdict(list), defaultdict(Counter)
            for r in csv.DictReader(open(path)):
                by_order[int(r["order"])].append(r)
                if r["skill"] == "clear":
                    clears[r["pose"]].append(r["success"] == "True")
                if r["skill"] != "knock" and (a := action(r)) is not None:
                    used[int(r["order"])][a] += 1
            for rows in by_order.values():
                for p in tower_place_clutter(rows):
                    places[clutter_band(p["nearest"])].append(p["ok"])
            episodes = Path(str(path).replace("_skills.csv", "_episodes.csv"))
            for e in csv.DictReader(open(episodes)):
                c = used[int(e["index"])]
                c["reject"] = int(e["rejected"])
                known = sum(self.times[a] * c[a] for a in ACTIONS) + self.times["overhead"]
                n_clears.append(c["clear"])
                residual.append(float(e["sim_s"]) - known)

        def rate(v):
            return (sum(v) + 1) / (len(v) + 2)
        base = rate(places[CLEAR_BAND])
        n, res = np.array(n_clears, dtype=float), np.array(residual)
        times = dict(self.times, clear=float(n @ res / (n @ n)) if n.any() else 0.0)
        return Calibration(
            self.odds, self.counts, times, self.knocks, self.by_height,
            {band: rate(places[band]) / base for _, band in CLUTTER_BANDS} | {CLEAR_BAND: 1.0},
            {pose: rate(v) for pose, v in clears.items()},
            {"places": {b: len(v) for b, v in places.items()},
             "clears": {pose: len(v) for pose, v in clears.items()}})

    def clutter_factor(self, distance: float) -> float:
        return 1.0 if self.clutter is None else self.clutter[clutter_band(distance)]

    def add_time_spread(self, skills_csvs: list[Path]) -> "Calibration":
        """v3.1 (docs/v3_1_targets.md, amendment of 2026-11-15): this calibration plus a spread
        on each action's time, from the _skills.csv (and _episodes.csv) files of calibration
        runs. An order's duration less its actions at this calibration's mean times (the clear's
        too, if fitted) leaves a residual; its square is fitted on the order's action counts
        (and a constant, the overhead's) by non-negative least squares, which gives each action
        a variance. The mean times, and everything else, stay as they are."""
        from scipy.optimize import nnls

        names = ACTIONS + ("clear", "overhead")
        X, y = [], []
        for path in sorted(skills_csvs):
            used = defaultdict(Counter)
            for r in csv.DictReader(open(path)):
                if r["skill"] != "knock" and (a := action(r)) is not None:
                    used[int(r["order"])][a] += 1
            episodes = Path(str(path).replace("_skills.csv", "_episodes.csv"))
            for e in csv.DictReader(open(episodes)):
                c = used[int(e["index"])]
                c["reject"], c["overhead"] = int(e["rejected"]), 1
                known = sum(self.times.get(a, 0.0) * c[a] for a in names)
                X.append([c[a] for a in names])
                y.append((float(e["sim_s"]) - known) ** 2)
        var, _ = nnls(np.array(X, dtype=float), np.array(y))
        return replace(self, time_var=dict(zip(names, map(float, var), strict=True)))

    @classmethod
    def from_calibration_run(cls, root: Path = Path()) -> "Calibration":
        return cls.load(sorted(root.glob(CALIBRATION_RUNS)), root / CALIBRATION_SUITE)

    @classmethod
    def load(cls, skills_csvs: list[Path], suite: Path) -> "Calibration":
        """From evaluate_cell's _skills.csv files of runs on `suite` (each with its
        _episodes.csv): one file per seed, named ..._seed<k>_..."""
        cal, ok = cls(), defaultdict(list)
        orders = json.loads(Path(suite).read_text())["orders"]
        heights = [r and r["height"] for r in orders]
        towers = [r and r["tower"] for r in orders]
        X, y = [], []
        for path in sorted(skills_csvs):
            seed = int(str(path).split("_seed")[1][0])
            fallen, used = defaultdict(list), defaultdict(Counter)
            for r in csv.DictReader(open(path)):
                if r["skill"] not in ("pick", "place", "standup", "knock"):
                    continue                         # (v3.1 rows, e.g. clear: not v3.0 odds)
                if r["skill"] == "knock":
                    i = int(r["order"])
                    fallen[i].append(
                        {"pose": r["pose"], "recoverable": r["recoverable"] == "True",
                         "near": near_tower(_xy(r)),
                         # (v3.1) where it lay: radius, and angle from the knocked tower
                         "r": float(r["radius"]),
                         "da": _wrap(float(r["angle"]) - _CFG.tower_angles[towers[i]])})
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
    xy: np.ndarray | None = None   # (v3.1) where it lies; None: unknown, or moved to staging


def decision_features(cube: Loose, retry: bool, spares: int, max_spares: int, t: float,
                      deadline: float | None, queue: int, planner: GreedyPlanner,
                      loose_poses: list[str], times: dict) -> np.ndarray:
    """What the planner sees at a decision, scaled to about [0, 1]; the symbolic and the physics
    cell (ScriptedCell.run_mixed_order) both build it here. `queue`: arrivals (and spares) still
    to come up the belt; `loose_poses`: every fallen cube still loose, this one included.
    The first 13 are v3.0's (a v3.0 planner sees only those); v3.1 adds the cube's distance to
    its nearest tower (in 20 cm, at most 1) and the cubes that tower still needs (in 4)."""
    horizon = deadline or 1e9
    left = sum(len(r) - p for r, p in zip(planner.recipes, planner.placed, strict=True))
    work = left * (times["pick_belt"] + times["place_tower"])
    ready = any(planner.next_colour(k) == cube.colour for k in range(len(planner.recipes)))
    if cube.xy is None or cube.staged:
        gap, needs = 1.0, 0.0
    else:
        d = [float(np.linalg.norm(cube.xy - t)) for t in TOWER_XY[:len(planner.recipes)]]
        k = int(np.argmin(d))
        gap, needs = min(1.0, d[k] / CLEAR_RADIUS), len(planner.recipes[k]) - planner.placed[k]
    return np.array([
        cube.pose == "lying", cube.near, retry, cube.staged, spares / max(1, max_spares),
        t / horizon, (horizon - t - work) / horizon, queue / 14, left / 12,
        loose_poses.count("lying") / 4, loose_poses.count("upright") / 4, ready,
        deadline is not None, gap, needs / 4], dtype=np.float32)


@dataclass
class Decision:
    """A question for the planner: replace `cube` with a spare (True) or rescue it (False)?
    In the physics cell (v3.1) a policy may also answer CLEAR: take the cube to the reject bin,
    then replace it with a spare."""
    cube: Loose
    retry: bool          # asked after a failed rescue attempt
    features: np.ndarray


# a recovery planner's answers; a bool answer is RESCUE (False) or LEAVE (True)
RESCUE, LEAVE, CLEAR = 0, 1, 2
CHOICE_SEED = 20261205   # v3.1 calibration: the random choices' own stream (RandomChoice)


class RandomChoice:
    """v3.1 calibration (docs/v3_1_targets.md, 2026-11-14): rescue, leave or clear, drawn with
    equal odds at every decision, from a stream seeded per order (`reset(order index)`), so
    the calibration run sees the floors any planner can leave behind."""

    def __init__(self, seed: int = CHOICE_SEED, choices: tuple[int, ...] = (RESCUE, LEAVE, CLEAR)
                 ) -> None:
        self.seed, self.choices = seed, choices
        self.reset(0)

    def reset(self, index: int) -> None:
        self.rng = np.random.default_rng([self.seed, index])

    def __call__(self, d: Decision) -> int:
        return int(self.choices[self.rng.integers(len(self.choices))])

    def __str__(self) -> str:
        names = {RESCUE: "rescue", LEAVE: "leave", CLEAR: "clear"}
        return (f"random {' / '.join(names[c] for c in self.choices)}, equal odds "
                f"(seed {self.seed}, per order)")


@dataclass(frozen=True)
class Task:
    spares: int = 2
    slack: float = 1.5           # deadline: slack x the order's undisturbed time (None: none)
    spare_delay: float = 0.0     # s a spare costs on top of a belt arrival


DEFAULT_TASK = Task()  # the learned planner's task (docs/v3_targets.md, 2026-10-30)


class SymbolicCell:
    """One order. `play()` is a generator: it yields a Decision whenever the planner has a
    choice and must be sent RESCUE (or False), LEAVE (or True) or CLEAR; it returns whether the
    order was completed (by the deadline, if any). `result()` drives it with a policy."""

    N_FEATURES = 15        # (decision_features; a v3.0 planner sees the first 13)
    N_FEATURES_V30 = 13

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
        # (v3.1) with a time spread, its own stream: the skills' outcomes stay those without it
        self.time_rng = rng.spawn(1)[0] if cal.time_var else None
        self.t = 0.0
        self.spend("overhead")
        self.floor: list[np.ndarray] = []   # (v3.1) written-off cubes left where they lie
        self.cleared = self.cleared_ok = 0

    # --- skills -----------------------------------------------------------------------------
    def spend(self, action: str | None) -> None:
        """The time `action` takes: its mean, plus Gaussian noise with a time spread."""
        if not action:
            return
        self.t += self.cal.times.get(action, 0.0)
        if self.time_rng is not None and (var := self.cal.time_var.get(action, 0.0)) > 0:
            self.t += float(self.time_rng.normal(0.0, np.sqrt(var)))

    def roll(self, key: tuple, factor: float = 1.0) -> bool:
        self.spend(_time_key(key))
        p =self.cal.p(key) if factor == 1.0 else min(1.0, self.cal.p(key) * factor)
        return bool(self.rng.random() < p)

    def nearest_floor_cube(self, t: int) -> float:
        """Distance from tower t to the nearest cube on the floor: written off, or loose and
        not yet moved to a staging spot (inf: none)."""
        cubes = self.floor + [c.xy for c in self.loose if c.xy is not None and not c.staged]
        return min((float(np.linalg.norm(xy - TOWER_XY[t])) for xy in cubes),
                   default=float("inf"))

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

    def result(self, policy: Callable[[Decision], bool | int] | None = None) -> bool:
        """Play the order with `policy` (None: always rescue)."""
        game = self.play()
        try:
            decision = next(game)
            while True:
                decision = game.send(RESCUE if policy is None else int(policy(decision)))
        except StopIteration as end:
            return end.value

    def stack(self, colour: str, t: int):
        factor = (1.0 if self.cal.clutter is None
                  else self.cal.clutter_factor(self.nearest_floor_cube(t)))
        if not self.roll(("place", "tower", len(self.towers[t])), factor):
            self.failed = True
            return
        self.towers[t].append(colour)
        k = self.knock
        if k is not None and not self.knocked and t == k.tower and len(self.towers[t]) == k.height:
            self.knocked = True
            fallen = self.fallen or []
            for j, (st, c) in enumerate(zip(fallen, reversed(self.towers[t]), strict=False)):
                xy = (station_xy(_CFG.tower_angles[t] + st["da"], st["r"]) if "da" in st
                      else None)
                self.loose.append(Loose(c, st["pose"], st["recoverable"], st["near"], j, xy=xy))
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
            self.spend("reject")

    def write_off(self, c: Loose, gone: bool = False) -> None:
        """Give up on cube c (and call a spare, if one is needed and left). It stays on the
        floor where it lies unless `gone` (cleared into the bin) or moved to a staging spot."""
        self.loose.remove(c)
        if c.staged:
            self.staged -= 1
        elif c.xy is not None and not gone:
            self.floor.append(c.xy)
        if self.needed(c.colour) and self.spares > 0:
            self.spares -= 1
            self.queue.append((c.colour, True))

    def clear(self, c: Loose) -> None:
        """v3.1: take cube c to the reject bin (its odds by pose, its fitted time), then write
        it off; if the clear fails, it stays on the floor."""
        self.spend("clear")
        ok = bool(self.rng.random() < self.cal.clear_odds.get(c.pose, 0.0))
        self.cleared += 1
        self.cleared_ok += ok
        self.write_off(c, gone=ok)

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
                    choice = yield Decision(c, retry, self.features(c, retry))
                    choice = RESCUE if choice is None else int(choice)
                    if choice == LEAVE:
                        self.write_off(c)
                    elif choice == CLEAR:
                        self.clear(c)
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
                    c.pose, c.near, c.staged, c.xy = "upright", False, True, None
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


BAND_NAMES = ("under 6 cm", "6-20 cm", "further")   # (v3.1 rules: from the nearest tower)
ANSWER_NAMES = {RESCUE: "rescue", LEAVE: "leave", CLEAR: "clear"}


def cube_band(cube: Loose) -> int:
    """v3.1: 0 within 6 cm of a tower, 1 within 20 cm, 2 further (or moved to staging)."""
    if cube.xy is None or cube.staged:
        return 2
    d = min(float(np.linalg.norm(cube.xy - t)) for t in TOWER_XY)
    return 0 if d < CLUTTER_BANDS[0][0] else 1 if d < CLUTTER_BANDS[1][0] else 2


@dataclass(frozen=True)
class ClearRule:
    """v3.1 (docs/v3_1_targets.md): rescue, leave or clear a fallen cube by its pose (upright,
    lying) and how near it lies to a tower (cube_band); after a failed rescue attempt, `after`
    (RESCUE: retry). `answers[3 * lying + band]`."""
    answers: tuple[int, ...] = (RESCUE,) * 6
    after: int = RESCUE

    def __call__(self, d: Decision) -> int:
        if d.retry:
            return self.after
        return self.answers[3 * (d.cube.pose == "lying") + cube_band(d.cube)]

    def __str__(self) -> str:
        parts = [f"{pose} {band}: {ANSWER_NAMES[self.answers[3 * i + j]]}"
                 for i, pose in enumerate(("upright", "lying"))
                 for j, band in enumerate(BAND_NAMES)]
        return "; ".join(parts) + f"; after a failed attempt: {ANSWER_NAMES[self.after]}"


CLEAR_RULE_FAMILY = tuple(ClearRule(tuple(a), after)
                          for a in itertools.product((RESCUE, LEAVE, CLEAR), repeat=6)
                          for after in (RESCUE, LEAVE, CLEAR))


def load_planner(run_dir: Path) -> Callable[[Decision], bool]:
    """A trained recovery planner (scripts/train_recovery_planner.py) as a policy: True to
    replace. It sees what it was trained to see (its run's `blind`)."""
    from stable_baselines3 import PPO

    from polypanda.environments.recovery_planner import HIDDEN

    model = PPO.load(Path(run_dir) / "model.zip", device="cpu")
    meta = json.loads((Path(run_dir) / "meta.json").read_text())
    n = model.observation_space.shape[0]           # (a v3.0 planner: the first 13 features)
    hidden = [i for b in meta["blind"] for i in HIDDEN[b] if i < n]
    two_way = model.action_space.n == 2            # (v3.0: rescue or leave)

    def policy(d: Decision) -> bool | int:
        obs = np.clip(d.features[:n], -2.0, 2.0)
        obs[hidden] = 0.0
        a = int(model.predict(obs, deterministic=True)[0])
        return a == 1 if two_way else a
    return policy
