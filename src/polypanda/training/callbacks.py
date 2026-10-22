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
