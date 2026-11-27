import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class RewardTermsCallback(BaseCallback):
    """Log per-step means of each reward term and the TCP distance once per rollout."""

    def __init__(self) -> None:
        super().__init__()
        self._sums: dict[str, float] = defaultdict(float)
        self._count = 0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            for name, value in info.get("reward_terms", {}).items():
                self._sums[name] += value
            if np.isfinite(info.get("distance", np.nan)):
                self._sums["_distance"] += info["distance"]
            self._count += 1
        return True

    def _on_rollout_end(self) -> None:
        if not self._count:
            return
        for name, total in self._sums.items():
            key = "rollout/mean_distance_m" if name == "_distance" else f"reward_terms/{name}"
            self.logger.record(key, total / self._count)
        self._sums.clear()
        self._count = 0


class StartCurriculumCallback(BaseCallback):
    """v4.1 insertion round 2: the start-height curriculum (IndustReal's sampling-based one;
    environments/insertion.InsertionCurriculumConfig). Counts the training episodes' successes
    (every env); after each `window` episodes it moves the lowest start: up `step_up` if their
    success rate is above `raise_above`, down `step_down` if below `lower_below`, within
    [low_min, high], and sets it in every training env (env_method "set_start_low"). Logs
    curriculum/low_mm and curriculum/success_rate, and appends each decision to `log_path`
    (JSON lines) if given."""

    def __init__(self, curriculum, high: float = 0.3, log_path: Path | None = None) -> None:
        super().__init__()
        self.cfg = curriculum
        self.low = float(curriculum.low)
        self.high = float(high)
        self.log_path = None if log_path is None else Path(log_path)
        self.successes: list[bool] = []

    def _on_training_start(self) -> None:
        self.training_env.env_method("set_start_low", self.low)

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if done:
                self.successes.append(bool(info.get("is_success", False)))
        if len(self.successes) >= self.cfg.window:
            self.decide()
        return True

    def decide(self) -> None:
        c = self.cfg
        rate = float(np.mean(self.successes))
        before = self.low
        if rate > c.raise_above:
            self.low = min(self.high, self.low + c.step_up)
        elif rate < c.lower_below:
            self.low = max(c.low_min, self.low - c.step_down)
        if self.low != before:
            self.training_env.env_method("set_start_low", self.low)
        self.logger.record("curriculum/low_mm", self.low * 1000)
        self.logger.record("curriculum/success_rate", rate)
        if self.log_path is not None:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"timesteps": int(self.num_timesteps),
                                    "episodes": len(self.successes), "success_rate": rate,
                                    "low_before_m": before, "low_m": self.low}) + "\n")
        self.successes = []


HEIGHT_BINS_MM = (-1e9, 0.0, 5.0, 20.0, 50.0, 100.0, 1e9)


def height_bin(height: float | None) -> str:
    """A start height's bin (m above the rim; None: the carry start), for logging."""
    if height is None:
        return "carry"
    mm = height * 1000
    for lo, hi in zip(HEIGHT_BINS_MM[:-1], HEIGHT_BINS_MM[1:], strict=True):
        if lo <= mm < hi:
            return ("inside" if hi == 0.0 else f">{lo:g}mm" if hi == 1e9
                    else f"{lo:g}-{hi:g}mm")
    return "carry"


class ReverseCurriculumCallback(BaseCallback):
    """v4.1 insertion round 2, after pilot 2: the reverse start curriculum
    (InsertionCurriculumConfig, mode "reverse"; Florensa et al. 2017). The training envs draw
    starts uniformly between the fixed deepest start `low` and `high`, plus a share from the carry
    start. After each `window` curriculum episodes (carry starts not counted), `high` grows x
    `grow` if their success rate is above `raise_above`, shrinks / `grow` below `lower_below`,
    within [high_min, cap], and is set in every training env (env_method "set_start_high").
    Logs curriculum/high_mm, curriculum/success_rate, curriculum/carry_success_rate and success
    per start-height bin (curriculum/success_<bin>), and appends each decision with its bins to
    `log_path` (JSON lines)."""

    def __init__(self, curriculum, cap: float = 0.3, log_path: Path | None = None) -> None:
        super().__init__()
        self.cfg = curriculum
        self.high = float(curriculum.high)
        self.cap = float(cap)
        self.log_path = None if log_path is None else Path(log_path)
        self.successes: list[bool] = []
        self.carry: list[bool] = []
        self.bins: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    def _on_training_start(self) -> None:
        self.training_env.env_method("set_start_low", float(self.cfg.low))
        self.training_env.env_method("set_start_high", self.high)

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if not done:
                continue
            ok = bool(info.get("is_success", False))
            height = info.get("start_height")
            b = self.bins[height_bin(height)]
            b[0] += ok
            b[1] += 1
            (self.carry if height is None else self.successes).append(ok)
        if len(self.successes) >= self.cfg.window:
            self.decide()
        return True

    def decide(self) -> None:
        c = self.cfg
        rate = float(np.mean(self.successes))
        before = self.high
        if rate > c.raise_above:
            self.high = min(self.cap, self.high * c.grow)
        elif rate < c.lower_below:
            self.high = max(c.high_min, self.high / c.grow)
        if self.high != before:
            self.training_env.env_method("set_start_high", self.high)
        self.logger.record("curriculum/high_mm", self.high * 1000)
        self.logger.record("curriculum/success_rate", rate)
        carry = float(np.mean(self.carry)) if self.carry else None
        if carry is not None:
            self.logger.record("curriculum/carry_success_rate", carry)
        bins = {k: f"{v[0]}/{v[1]}" for k, v in self.bins.items()}
        for k, v in self.bins.items():
            self.logger.record(f"curriculum/success_{k}", v[0] / v[1])
        if self.log_path is not None:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"timesteps": int(self.num_timesteps),
                                    "episodes": len(self.successes), "success_rate": rate,
                                    "high_before_m": before, "high_m": self.high,
                                    "carry_episodes": len(self.carry),
                                    "carry_success_rate": carry, "bins": bins}) + "\n")
        self.successes, self.carry = [], []
        self.bins = defaultdict(lambda: [0, 0])


class SaveVecNormalizeCallback(BaseCallback):
    """Used as EvalCallback's callback_on_new_best: store normalization stats with the best model.

    Without this, best_model.zip would later be paired with the *final* running statistics,
    which drift after the best checkpoint was taken.
    """

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = Path(path)

    def _on_step(self) -> bool:
        vec_normalize = self.model.get_vec_normalize_env()
        if vec_normalize is not None:
            vec_normalize.save(str(self.path))
        return True
