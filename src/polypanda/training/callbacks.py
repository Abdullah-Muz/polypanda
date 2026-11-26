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
