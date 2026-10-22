"""Measure ReachEnv throughput with random actions, single env and SB3 vectorized envs.

Usage: uv run python scripts/benchmark_env.py [--seconds 4] [--envs 1 2 4 8]
"""

import argparse
import os
import time

import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from polypanda.environments.reach import ReachEnv


def rate(vec_env, seconds: float) -> float:
    vec_env.seed(0)
    vec_env.reset()
    rng = np.random.default_rng(0)
    n, steps = vec_env.num_envs, 0
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        vec_env.step(rng.uniform(-1, 1, (n, 7)).astype(np.float32))
        steps += n
    return steps / (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--envs", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()

    print(f"logical CPUs visible: {os.cpu_count()}")
    env, steps = ReachEnv(), 0
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    start = time.perf_counter()
    while time.perf_counter() - start < args.seconds:
        _, _, term, trunc, _ = env.step(rng.uniform(-1, 1, 7).astype(np.float32))
        steps += 1
        if term or trunc:
            env.reset()
    print(f"raw ReachEnv:             {steps / (time.perf_counter() - start):8,.0f} steps/s")

    for n in args.envs:
        for cls in (DummyVecEnv, SubprocVecEnv):
            if cls is SubprocVecEnv and n == 1:
                continue
            vec = make_vec_env(ReachEnv, n_envs=n, vec_env_cls=cls)
            try:
                print(f"{cls.__name__:13s} x {n:<2d}:      {rate(vec, args.seconds):8,.0f} steps/s")
            finally:
                vec.close()


if __name__ == "__main__":
    main()
