# PolyPanda

Hierarchical reinforcement learning for Franka Panda manipulation: modular PPO/SAC skills, MuJoCo environments, a rule-based task supervisor, and reproducible robotics-control benchmarks.

> **Status: pre-implementation.** This repository was just created. Nothing below is implemented yet; the roadmap describes planned work and will be updated as each milestone is actually completed and tested.

## Overview

PolyPanda trains several task-specialized reinforcement-learning policies (reach, grasp, transport, place) for a simulated 7-DOF Franka Emika Panda arm and coordinates them with a rule-based finite-state-machine supervisor to perform autonomous tabletop pick-and-place. Learned policies are benchmarked against a classical inverse-kinematics baseline on held-out scenarios across multiple seeds.

## Planned stack

- [MuJoCo](https://mujoco.org/) physics with the Panda model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [Gymnasium](https://gymnasium.farama.org/) custom environments
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (PPO, SAC) on CPU PyTorch
- Python 3.12 on Ubuntu 24.04, CPU-first; no NVIDIA GPU required

## Roadmap

| Milestone | Scope | Status |
|---|---|---|
| 1 | Reproducible environment, pinned Panda model, headless simulation, tests | Planned |
| 2 | Custom reaching environment and classical IK baseline | Planned |
| 3 | PPO and SAC reaching policies with benchmark | Planned |
| 4 | Grasp and transport skills | Planned |
| 5 | Multi-policy FSM supervisor and full pick-and-place | Planned |
| 6 | Robustness, evaluation, documentation | Planned |

## License

MIT, see [LICENSE](LICENSE). Third-party assets such as the MuJoCo Menagerie Panda model keep their own licenses.
