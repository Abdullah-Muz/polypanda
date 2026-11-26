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

v4.1 (docs/v4_1_targets.md): the same for any task and script.
- **Recording** (the door): `RecordConfig.task`, `env_config` (a YAML; the training distribution
  as evaluated, without the training-only focus cells) and `script` (feel_v2 for task door_v2);
  the defaults are v4.0's recording, draw for draw. The insertion's demonstrations come from
  scripts/record_insertion_demos.py in the same format (obs, act, rew, episode, done, kept, total).
- **Cloning** works on any task's demonstrations (the task of the training config), and can
  **warm-start** (`CloneConfig.init_from`): the clone starts from another run's network (v4.0's
  door round 5), its inputs re-expressed in the demonstrations' normalization without changing
  what it computes (training.runner.renormalize_inputs), its return normalization and its action
  noise kept (`log_std: null`), so cloning refreshes what the source knows rather than starting
  over. scripts/train_bc.py runs a clone from a committed config (configs/*_bc_r*.yaml).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from polypanda.config import load_yaml

DEMO_SEED = 20261130          # the demonstrations' episode stream (evaluation sets: 20261124-26)


@dataclass(frozen=True)
class RecordConfig:
    episodes: int = 3000
    seed: int = DEMO_SEED
    noise: float = 0.2            # std of the noise on executed hand actions (of the max step)
    noise_share: float = 0.5      # share of episodes recorded with noise
    jobs: int = 6
    # v4.1: the task (its env class), its env config (None: the env's defaults) and the script
    task: str = "door"
    env_config: str | None = None
    script: str = "feel"
    # v4.1: a trained run ("{seed}": the clone's seed) tried first on every episode, clean and
    # deterministic: its successful episodes are kept as they are, the script runs on the rest
    # (so a warm-started clone is taught the source's own behaviour where it already works)
    teacher: str | None = None


def _record_env(task: str, env_config: str | None):
    from polypanda.environments.tasks import task_spec
    spec = task_spec(task)
    if env_config is None:
        return spec.env_cls()
    cfg = spec.config_cls.from_yaml(env_config)
    # the distribution as evaluated: a training-only focus on some cells is not copied
    return spec.env_cls(cfg.for_evaluation() if hasattr(cfg, "for_evaluation") else cfg)


def _play(env, ep: dict, seed: int, act, rng, noise: float | None, on_reset=None) -> tuple:
    """One episode from `ep` acted by `act` (obs -> action; on_reset() called after the env's
    reset); with `noise`, Gaussian noise (from rng) on the executed hand actions while the clean
    action is recorded."""
    obs, _ = env.reset(seed=seed, options={"episode": ep})
    if on_reset is not None:
        on_reset()
    obs_ep, act_ep, rew_ep = [], [], []
    done = False
    while not done:
        a = np.asarray(act(obs))
        executed = a.copy()
        if noise is not None:
            executed[:4] = np.clip(a[:4] + rng.normal(0.0, noise, 4), -1.0, 1.0)
        obs_ep.append(obs)
        act_ep.append(a)
        obs, r, term, trunc, info = env.step(executed)
        rew_ep.append(r)
        done = term or trunc
    return obs_ep, act_ep, rew_ep, info


def _record_chunk(args) -> dict:
    seeds, noise_share, noise, *rest = args
    task, env_config, mode, teacher = (list(rest) + [None])[:4] if rest else (
        "door", None, "feel", None)
    from polypanda.control.door_baselines import BaselineDoor
    env = _record_env(task, env_config)
    teacher_act = None
    if teacher is not None:
        from polypanda.evaluation.policies import load_run_policy
        teacher_act = load_run_policy(teacher)
    obs_l, act_l, rew_l, ep_l, done_l, rows = [], [], [], [], [], []
    kept, total = 0, 0
    for seed in seeds:
        rng = np.random.default_rng(seed)
        noisy = rng.uniform() < noise_share
        ep = env.draw_episode(rng)
        source = "script"
        if teacher_act is not None:
            # v4.1: the teacher policy first, clean; its successful episodes are kept as they are
            obs_ep, act_ep, rew_ep, info = _play(env, ep, seed, teacher_act, rng, None)
            source = "teacher"
        if teacher_act is None or not info["is_success"]:
            script = BaselineDoor(env, mode)
            obs_ep, act_ep, rew_ep, info = _play(env, ep, seed, lambda o, s=script: s.act(),
                                                 rng, noise if noisy else None, script.reset)
            source = "script"
        total += 1
        rows.append({"seed": int(seed), "kind": ep.get("kind", "lid"), "command": ep["command"],
                     "slide": ep["slide"], "noisy": bool(noisy and source == "script"),
                     "source": source,
                     "success": bool(info["is_success"]), "failure": info["failure"],
                     "steps": len(act_ep)})
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
            "kept": kept, "total": total, "rows": rows}


def record(cfg: RecordConfig, rows: list | None = None) -> dict:
    """Feel-script demonstrations on freshly drawn training machines (successes only). With
    `rows` (a list), one row per recorded episode (kept or not) is appended to it."""
    seeds = [cfg.seed * 1000 + k for k in range(cfg.episodes)]
    chunks = [(seeds[j::cfg.jobs], cfg.noise_share, cfg.noise, cfg.task, cfg.env_config,
               cfg.script, cfg.teacher) for j in range(cfg.jobs)]
    with Pool(cfg.jobs) as pool:
        parts = pool.map(_record_chunk, chunks)
    out = {k: np.concatenate([x for p in parts for x in p[k]]) for k in
           ("obs", "act", "rew", "episode", "done")}
    out["kept"] = sum(p["kept"] for p in parts)
    out["total"] = sum(p["total"] for p in parts)
    if rows is not None:
        rows.extend(sorted((r for p in parts for r in p["rows"]), key=lambda r: r["seed"]))
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
    # the clone's action noise (std ~0.37): PPO explores from here. One value for every action,
    # one per action (v4.1), or None: keep the network's (the source's, when warm-started)
    log_std: float | tuple[float, ...] | None = -1.0
    val_share: float = 0.05       # episodes held out to watch over-fitting
    seed: int = 0
    # v4.1: warm start from this run ("{seed}": the clone's seed) and checkpoint (best/final)
    init_from: str | None = None
    init_checkpoint: str = "final"
    # v4.1: an action that cannot be undone (the insertion's release, action 4), on once the
    # observation at latch_obs is (its "released" input): from then on its label is +1, which
    # the env treats exactly as the recorded -1 (ignored once released), so the demonstrations'
    # behaviour is unchanged and the label agrees with "let go" instead of contradicting it
    latch_action: int | None = None
    latch_obs: int | None = None


def latch_labels(act: np.ndarray, obs: np.ndarray, action: int, obs_index: int) -> np.ndarray:
    """The labels with `action` set to +1 wherever obs[:, obs_index] is on (above 0.5)."""
    out = np.array(act, copy=True)
    out[:, action] = np.where(obs[:, obs_index] > 0.5, 1.0, out[:, action])
    return out


@dataclass(frozen=True)
class BCRunConfig:
    """A clone, as committed before it runs (configs/*_bc_r*.yaml, scripts/train_bc.py): the
    training config whose task, env config, network and gamma the clone takes (the PPO round
    that starts from it), the demonstrations file, and how to clone (the seed comes from the
    command line)."""
    training_config: str
    demos: str
    clone: CloneConfig = field(default_factory=CloneConfig)
    # how the demonstrations are recorded: a door script's (scripts/record_door_demos.py records
    # exactly this), or None with the recording described in demos_source (the insertion's:
    # scripts/record_insertion_demos.py)
    record: RecordConfig | None = None
    demos_source: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> BCRunConfig:
        return load_yaml(cls, path)


def _demo_rms(obs: np.ndarray) -> RunningMeanStd:
    rms = RunningMeanStd(shape=obs.shape[1:])
    rms.update(obs.astype(np.float64))
    return rms


def warm_start(policy, venv: VecNormalize, demo_rms: RunningMeanStd, cfg: CloneConfig) -> dict:
    """Load cfg.init_from's network into `policy` and re-express its inputs in the
    demonstrations' normalization (`demo_rms`), so that it computes what the source computed
    (inputs inside the clipping range); venv takes the source's return normalization."""
    from stable_baselines3 import PPO

    from polypanda.training.runner import renormalize_inputs

    source = Path(cfg.init_from.format(seed=cfg.seed))
    which = cfg.init_checkpoint
    loaded = PPO.load(source / f"{which}_model.zip", device="cpu")
    if (loaded.observation_space.shape != venv.observation_space.shape
            or loaded.action_space.shape != venv.action_space.shape):
        raise ValueError(f"{source}: observation/action sizes differ from this task's")
    policy.load_state_dict(loaded.policy.state_dict())
    with open(source / f"{which}_vecnormalize.pkl", "rb") as f:
        stats = pickle.load(f)
    renormalize_inputs(policy, stats.obs_rms, demo_rms, venv.epsilon)
    venv.ret_rms = stats.ret_rms
    meta = yaml.safe_load((source / "metadata.yaml").read_text())
    return {"run": str(source), "checkpoint": which, "policy_id": meta["policy_id"],
            "git": meta.get("git", {}).get("commit"),
            "log_std": [round(float(x), 4) for x in policy.log_std.detach()]}


def clone(demos: dict, train_cfg, env_config, cfg: CloneConfig) -> tuple:
    """A PPO model whose policy imitates the demonstrations, and the VecNormalize that goes with
    it. Returns (model, vecnormalize, log)."""
    from polypanda.environments.tasks import task_spec
    from polypanda.training.runner import build_model

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    gamma = train_cfg.algo_params.gamma
    env_cls = task_spec(train_cfg.task).env_cls
    venv = VecNormalize(DummyVecEnv([lambda: env_cls(env_config)]), norm_obs=True,
                        norm_reward=True, gamma=gamma)
    rms = _demo_rms(demos["obs"])
    venv.obs_rms = rms
    ret = discounted_returns(demos["rew"], demos["done"], gamma)
    ret_rms = RunningMeanStd(shape=())
    ret_rms.update(ret)
    venv.ret_rms = ret_rms
    model = build_model(train_cfg, venv, None)
    policy = model.policy
    init = warm_start(policy, venv, rms, cfg) if cfg.init_from else None
    ret_rms = venv.ret_rms
    x = torch.as_tensor(np.clip((demos["obs"] - rms.mean) / np.sqrt(rms.var + venv.epsilon),
                                -venv.clip_obs, venv.clip_obs), dtype=torch.float32)
    act = demos["act"]
    if cfg.latch_action is not None:
        act = latch_labels(act, demos["obs"], cfg.latch_action, cfg.latch_obs)
    y = torch.as_tensor(np.clip(act, -1.0, 1.0), dtype=torch.float32)
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
    if init is not None:              # how far the source is from the demonstrations at first
        with torch.no_grad():
            init["start_train_mse"] = float(((mean_action(x[tr]) - y[tr]) ** 2).mean())
            if len(va):
                init["start_val_mse"] = float(((mean_action(x[va]) - y[va]) ** 2).mean())
                init["start_value_mse"] = float(((value(x[va]) - v[va]) ** 2).mean())
        log["init"] = init
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
    if isinstance(cfg.log_std, tuple | list):
        with torch.no_grad():
            policy.log_std.copy_(torch.as_tensor(cfg.log_std, dtype=policy.log_std.dtype))
    elif cfg.log_std is not None:
        with torch.no_grad():
            policy.log_std.fill_(cfg.log_std)
    return model, venv, log


def save_vecnormalize(venv: VecNormalize, path: Path) -> None:
    venv.training = False
    with open(path, "wb") as f:
        pickle.dump(venv, f)
