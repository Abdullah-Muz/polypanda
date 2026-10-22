# PolyPanda

Hierarchical reinforcement learning for Franka Panda manipulation: modular PPO/SAC skills, MuJoCo environments, a rule-based task supervisor, and reproducible robotics-control benchmarks.

> **Status: early development.** Milestone 1 (reproducible simulator foundation) is complete. No RL policies have been trained yet. The roadmap is updated only when a milestone is actually completed and tested.

## Overview

PolyPanda trains several task-specialized reinforcement-learning policies (reach, grasp, transport, place) for a simulated 7-DOF Franka Emika Panda arm and coordinates them with a rule-based finite-state-machine supervisor to perform autonomous tabletop pick-and-place. Learned policies are benchmarked against a classical inverse-kinematics baseline on held-out scenarios across multiple seeds.

## Planned stack

- [MuJoCo](https://mujoco.org/) physics with the Panda model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [Gymnasium](https://gymnasium.farama.org/) custom environments
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (PPO, SAC) on CPU PyTorch
- Python 3.12 on Ubuntu 24.04, CPU-first; no NVIDIA GPU required

## Setup

Requires Linux (developed on Ubuntu 24.04 under WSL2), git, and [uv](https://docs.astral.sh/uv/). No GPU needed.

```bash
git clone https://github.com/Abdullah-Muz/polypanda.git
cd polypanda
scripts/fetch_assets.sh                       # pinned MuJoCo Menagerie Panda model
uv sync                                       # Python 3.12 env from uv.lock (CPU PyTorch)
uv run python scripts/verify_install.py       # versions, model contract, headless stepping
uv run pytest                                 # test suite
```

Optional interactive viewer:

```bash
uv run python -m mujoco.viewer --mjcf=external/mujoco_menagerie/franka_emika_panda/scene.xml
```

## Roadmap

| Milestone | Scope | Status |
|---|---|---|
| 1 | Reproducible environment, pinned Panda model, headless simulation, tests | Done ([details](docs/environment.md)) |
| 2 | Custom reaching environment and classical IK baseline | Planned |
| 3 | PPO and SAC reaching policies with benchmark | Planned |
| 4 | Grasp and transport skills | Planned |
| 5 | Multi-policy FSM supervisor and full pick-and-place | Planned |
| 6 | Robustness, evaluation, documentation | Planned |

## License

MIT, see [LICENSE](LICENSE).

The Franka Emika Panda model comes from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Apache-2.0). It is downloaded by `scripts/fetch_assets.sh` at a pinned commit and is not redistributed in this repository.
