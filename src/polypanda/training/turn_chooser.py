"""The turn chooser (d2 round 6): which way a commanded stand-up policy should turn the hand.

StandUpV3Env's policy turns the commanded way; its PPO critic, rating every reachable command,
picked directions that put the tower behind the wrist about as often as the one-sided rounds
did (it estimates the discounted, shaped return, not whether the cube ends up standing). The
chooser is the high level of a two-level policy, learned from outcomes as a contextual bandit:
for many starts (never the evaluation sets), the frozen low-level policy is run with every
reachable command and each outcome (stood or not) recorded; a small network learns the chance
of success from the policy's own observation at the start under that command. In use, the
command with the highest predicted chance is taken (evaluation.standup.CommandedPolicyDriver).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


def collect_outcomes(run_dir: str | Path, which: str, seeds: list[int]) -> dict:
    """Run the policy of `run_dir` with every reachable command on the start of each seed
    (StandUpV3Env as evaluated: nominal cube, no training devices). Returns the starts'
    observations under each command, outcomes, failures and indices."""
    from polypanda.evaluation.standup import CommandedPolicyDriver, run_standup_episode
    from polypanda.policies.loader import load_run, make_run_env

    policy = load_run(run_dir, which)
    env = make_run_env(run_dir, "standup_v3")
    obs, success, failure, start, command = [], [], [], [], []
    for seed in seeds:
        _, info = env.reset(seed=seed)
        ep = info["episode"]
        for d in env.reachable_directions():
            env.reset(options={"episode": ep, "command": d})
            obs.append(env._observation())
            result, _ = run_standup_episode(env, CommandedPolicyDriver(policy, d), ep, seed)
            success.append(result.success)
            failure.append(result.failure or "")
            start.append(seed)
            command.append(d)
    return {"obs": np.array(obs, dtype=np.float32), "success": np.array(success),
            "failure": np.array(failure), "start": np.array(start),
            "command": np.array(command)}


class _Net(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.body(x).squeeze(-1)


@dataclass
class TurnChooser:
    """P(stood | observation under a command), on observations normalized as the policy's."""
    net: _Net
    mean: np.ndarray
    std: np.ndarray

    def scores(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor((obs - self.mean) / self.std, dtype=torch.float32)
        with torch.no_grad():
            return torch.sigmoid(self.net(x)).numpy()

    def state(self) -> dict:
        return {"state": self.net.state_dict(), "n_in": int(self.mean.size),
                "mean": self.mean, "std": self.std}

    def save(self, path: str | Path) -> None:
        torch.save(self.state(), path)

    @classmethod
    def from_state(cls, d: dict) -> "TurnChooser":
        net = _Net(d["n_in"])
        net.load_state_dict(d["state"])
        net.eval()
        return cls(net, d["mean"], d["std"])

    @classmethod
    def load(cls, path: str | Path) -> "TurnChooser | TurnChooserEnsemble":
        d = torch.load(path, weights_only=False)
        if "members" in d:
            return TurnChooserEnsemble([cls.from_state(m) for m in d["members"]])
        return cls.from_state(d)


@dataclass
class TurnChooserEnsemble:
    """Several choosers fitted from different initializations; the mean of their P(stood).
    On round 7's held-out practice starts 5 of them picked a command that stood 86.1 % of the
    time, one 84.7 % (the PPO critic 85.3 %; best possible 92.2 %)."""
    members: list[TurnChooser]

    def scores(self, obs: np.ndarray) -> np.ndarray:
        return np.mean([m.scores(obs) for m in self.members], axis=0)

    def save(self, path: str | Path) -> None:
        torch.save({"members": [m.state() for m in self.members]}, path)


def fit_chooser(obs: np.ndarray, success: np.ndarray, epochs: int = 300, lr: float = 3e-3,
                weight_decay: float = 1e-4, seed: int = 0) -> TurnChooser:
    torch.manual_seed(seed)
    mean, std = obs.mean(0), obs.std(0) + 1e-6
    x = torch.as_tensor((obs - mean) / std, dtype=torch.float32)
    y = torch.as_tensor(success, dtype=torch.float32)
    net = _Net(obs.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss(net(x), y).backward()
        opt.step()
    net.eval()
    return TurnChooser(net, mean, std)


def fit_ensemble(obs: np.ndarray, success: np.ndarray, n: int = 5, **kwargs
                 ) -> TurnChooserEnsemble:
    return TurnChooserEnsemble([fit_chooser(obs, success, seed=k, **kwargs) for k in range(n)])


def choice_quality(scores: np.ndarray, data: dict) -> float:
    """Of the starts in `data`, the fraction where the command with the best score stood."""
    hits = []
    for s in np.unique(data["start"]):
        m = data["start"] == s
        hits.append(bool(data["success"][m][np.argmax(scores[m])]))
    return float(np.mean(hits))
