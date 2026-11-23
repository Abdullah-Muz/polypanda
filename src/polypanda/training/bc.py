"""v4.0 "demonstrations first" (docs/v4_plan.md 3.3, docs/v4_targets.md): record the feel script
(control/door_baselines.py) on training machines, and clone it into a PPO policy that PPO then
refines (a run directory that `init_from` warm-starts from).

- **Recording.** Each episode is drawn from the training distribution with its own seed (a stream
  apart from the evaluation sets'). With `noise`, the hand actions executed get Gaussian noise
  while the script's own, clean action is recorded (DART-style noise injection): the clone then
  sees states a little off the script's path, and what the script does from them. The script
  servos its pushes on its own commanded target and its free moves on the measured hand, so it
  recovers from the noise. Only successful episodes are kept.
- **Cloning.** The policy network (the run's PPO MlpPolicy, same sizes) is fitted to the
  recorded actions (mean squared error on the Gaussian's mean, actions clipped to [-1, 1]) on
  observations normalized with the demonstrations' own statistics, which become the run's
  VecNormalize. The value network is fitted to the discounted returns of the recorded episodes,
  scaled as VecNormalize scales rewards, so PPO starts with a critic that knows the task.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

DEMO_SEED = 20261130          # the demonstrations' episode stream (evaluation sets: 20261124-26)


@dataclass(frozen=True)
class RecordConfig:
    episodes: int = 3000
    seed: int = DEMO_SEED
    noise: float = 0.2            # std of the noise on executed hand actions (of the max step)
    noise_share: float = 0.5      # share of episodes recorded with noise
    jobs: int = 6


def _record_chunk(args) -> dict:
    seeds, noise_share, noise = args
    from polypanda.control.door_baselines import BaselineDoor
    from polypanda.environments.door import DoorEnv
    env = DoorEnv()
    obs_l, act_l, rew_l, ep_l, done_l = [], [], [], [], []
    kept, total = 0, 0
    for seed in seeds:
        rng = np.random.default_rng(seed)
        noisy = rng.uniform() < noise_share
        ep = env.draw_episode(rng)
        script = BaselineDoor(env, "feel")
        obs, _ = env.reset(seed=seed, options={"episode": ep})
        script.reset()
        obs_ep, act_ep, rew_ep = [], [], []
        done = False
        while not done:
            a = script.act()
            executed = a.copy()
            if noisy:
                executed[:4] = np.clip(a[:4] + rng.normal(0.0, noise, 4), -1.0, 1.0)
            obs_ep.append(obs)
            act_ep.append(a)
            obs, r, term, trunc, info = env.step(executed)
            rew_ep.append(r)
            done = term or trunc
        total += 1
        if info["is_success"]:
            kept += 1
            obs_l.append(np.array(obs_ep))
            act_l.append(np.array(act_ep))
            rew_l.append(np.array(rew_ep))
            ep_l.append(np.full(len(obs_ep), seed))
            d = np.zeros(len(obs_ep), dtype=bool)
            d[-1] = True
            done_l.append(d)
    return {"obs": obs_l, "act": act_l, "rew": rew_l, "episode": ep_l, "done": done_l,
            "kept": kept, "total": total}


def record(cfg: RecordConfig) -> dict:
    """Feel-script demonstrations on freshly drawn training machines (successes only)."""
    seeds = [cfg.seed * 1000 + k for k in range(cfg.episodes)]
    chunks = [(seeds[j::cfg.jobs], cfg.noise_share, cfg.noise) for j in range(cfg.jobs)]
    with Pool(cfg.jobs) as pool:
        parts = pool.map(_record_chunk, chunks)
    out = {k: np.concatenate([x for p in parts for x in p[k]]) for k in
           ("obs", "act", "rew", "episode", "done")}
    out["kept"] = sum(p["kept"] for p in parts)
    out["total"] = sum(p["total"] for p in parts)
    return out


def discounted_returns(rew: np.ndarray, done: np.ndarray, gamma: float) -> np.ndarray:
    ret = np.zeros_like(rew, dtype=np.float64)
    g = 0.0
    for t in range(len(rew) - 1, -1, -1):
        if done[t]:
            g = 0.0
        g = rew[t] + gamma * g
        ret[t] = g
    return ret


@dataclass(frozen=True)
class CloneConfig:
    epochs: int = 40
    batch_size: int = 1024
    learning_rate: float = 1e-3
    value_epochs: int = 20
    log_std: float = -1.0         # the clone's action noise (std ~0.37): PPO explores from here
    val_share: float = 0.05       # episodes held out to watch over-fitting
    seed: int = 0


def clone(demos: dict, train_cfg, env_config, cfg: CloneConfig) -> tuple:
    """A PPO model whose policy imitates the demonstrations, and the VecNormalize that goes with
    it. Returns (model, vecnormalize, log)."""
    from polypanda.environments.door import DoorEnv
    from polypanda.training.runner import build_model

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    gamma = train_cfg.algo_params.gamma
    venv = VecNormalize(DummyVecEnv([lambda: DoorEnv(env_config)]), norm_obs=True,
                        norm_reward=True, gamma=gamma)
    rms = RunningMeanStd(shape=demos["obs"].shape[1:])
    rms.update(demos["obs"].astype(np.float64))
    venv.obs_rms = rms
    ret = discounted_returns(demos["rew"], demos["done"], gamma)
    ret_rms = RunningMeanStd(shape=())
    ret_rms.update(ret)
    venv.ret_rms = ret_rms
    model = build_model(train_cfg, venv, None)
    policy = model.policy
    x = torch.as_tensor(np.clip((demos["obs"] - rms.mean) / np.sqrt(rms.var + venv.epsilon),
                                -venv.clip_obs, venv.clip_obs), dtype=torch.float32)
    y = torch.as_tensor(np.clip(demos["act"], -1.0, 1.0), dtype=torch.float32)
    v = torch.as_tensor(ret / np.sqrt(ret_rms.var + venv.epsilon), dtype=torch.float32)
    episodes = np.unique(demos["episode"])
    held = set(rng.choice(episodes, int(len(episodes) * cfg.val_share), replace=False))
    is_val = np.isin(demos["episode"], list(held))
    tr, va = np.flatnonzero(~is_val), np.flatnonzero(is_val)

    def mean_action(xb):
        latent_pi, _ = policy.mlp_extractor(policy.extract_features(xb))
        return policy.action_net(latent_pi)

    def value(xb):
        _, latent_vf = policy.mlp_extractor(policy.extract_features(xb))
        return policy.value_net(latent_vf).squeeze(-1)

    pi_params = (list(policy.mlp_extractor.policy_net.parameters())
                 + list(policy.action_net.parameters()))
    vf_params = (list(policy.mlp_extractor.value_net.parameters())
                 + list(policy.value_net.parameters()))
    log = {"train_mse": [], "val_mse": [], "value_mse": []}
    opt = torch.optim.Adam(pi_params, lr=cfg.learning_rate)
    for _ in range(cfg.epochs):
        rng.shuffle(tr)
        for k in range(0, len(tr), cfg.batch_size):
            b = tr[k: k + cfg.batch_size]
            loss = ((mean_action(x[b]) - y[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            log["train_mse"].append(float(((mean_action(x[tr]) - y[tr]) ** 2).mean()))
            if len(va):
                log["val_mse"].append(float(((mean_action(x[va]) - y[va]) ** 2).mean()))
    opt = torch.optim.Adam(vf_params, lr=cfg.learning_rate)
    for _ in range(cfg.value_epochs):
        rng.shuffle(tr)
        for k in range(0, len(tr), cfg.batch_size):
            b = tr[k: k + cfg.batch_size]
            loss = ((value(x[b]) - v[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            ev = va if len(va) else tr
            log["value_mse"].append(float(((value(x[ev]) - v[ev]) ** 2).mean()))
    with torch.no_grad():
        policy.log_std.fill_(cfg.log_std)
    return model, venv, log


def save_vecnormalize(venv: VecNormalize, path: Path) -> None:
    venv.training = False
    with open(path, "wb") as f:
        pickle.dump(venv, f)
