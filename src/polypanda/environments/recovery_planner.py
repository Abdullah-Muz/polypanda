"""The learned recovery planner's environment (v3 f): rescue or replace, in the symbolic cell.

One episode is one work order played in the symbolic cell (supervisor.symbolic) with spares and
a deadline (Task). Each step is one of the planner's decisions about a fallen cube, as
SymbolicCell yields them: action 0 rescues it (or retries after a failed attempt), action 1
writes it off and calls a spare of its colour. The reward is 1 when the order ends complete by
its deadline, else 0. Orders are drawn as the cell's Layer 2 orders are (mixed_order), each with
a knock drawn as the disturbance suites draw theirs (make_suite), whose fallen cubes come to rest
as a calibration knock of the same height left them. Orders that give the planner no decision are
skipped: no choice of it can change them.

`blind` hides parts of the observation, for the ablations in docs/v3_targets.md: "situation"
(the cube's pose and nearness to a tower) and "clock" (time used and time left).
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from polypanda.evaluation.cell import CellEvalConfig
from polypanda.simulation.disturbances import make_suite
from polypanda.supervisor.planner import mixed_order
from polypanda.supervisor.symbolic import (
    DEFAULT_TASK,
    Calibration,
    Decision,
    SymbolicCell,
    Task,
    undisturbed_time,
)

HIDDEN = {"situation": (0, 1), "clock": (5, 6)}   # feature indices (SymbolicCell.features)


class RecoveryPlannerEnv(gym.Env):
    def __init__(self, calibration: Calibration, task: Task = DEFAULT_TASK,
                 blind: tuple[str, ...] = ()) -> None:
        self.cal, self.task = calibration, task
        self.hidden = [i for name in blind for i in HIDDEN[name]]
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(SymbolicCell.N_FEATURES,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self._game = None
        self.cell: SymbolicCell | None = None

    def _obs(self, decision: Decision) -> np.ndarray:
        obs = np.clip(decision.features, -2.0, 2.0)
        obs[self.hidden] = 0.0
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        cfg, rng = CellEvalConfig(), self.np_random
        while True:
            order, _ = mixed_order(rng, cfg.heights, cfg.surplus, cfg.solvable_with)
            knock = make_suite([order.recipes], rng, no_knock=0.0)[0]
            pool = self.cal.by_height.get(knock.height) if knock else None
            if not pool:
                continue
            fallen = pool[rng.integers(len(pool))]
            deadline = (None if self.task.slack is None
                        else self.task.slack * undisturbed_time(order, self.cal))
            self.cell = SymbolicCell(order, knock, fallen, self.cal, rng, self.task, deadline)
            self._game = self.cell.play()
            try:
                return self._obs(next(self._game)), {}
            except StopIteration:
                continue

    def step(self, action):
        try:
            return self._obs(self._game.send(bool(action))), 0.0, False, False, {}
        except StopIteration as end:
            done = bool(end.value)
            return (np.zeros(SymbolicCell.N_FEATURES, dtype=np.float32), float(done), True, False,
                    {"is_success": done})
