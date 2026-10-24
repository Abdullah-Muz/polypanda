# PolyPanda

Hierarchical reinforcement learning for Franka Panda manipulation: modular PPO/SAC skills, MuJoCo environments, a rule-based task supervisor, and reproducible robotics-control benchmarks.

> **Status: early development.** Milestones 1–2 are complete: a reproducible simulator, the reaching environment, and classical baselines evaluated on a held-out set. Milestone 3 is complete: PPO and SAC reaching policies are trained and benchmarked against the baselines over 3 seeds each. Milestone 4 is complete: learned grasping (80.5 % held-out success) and transport (scripted closed-loop carry, 92.0 %, after plain and residual RL were evaluated). Milestone 5 (place skill and full pick-and-place supervisor) is next. The roadmap is updated only when a milestone is actually completed and tested.

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

## Current results

Held-out reach evaluation (200 targets, [protocol](docs/evaluation_protocol.md)):

| Policy | Success (95 % CI) | Holds target | Time to success (median) | Final error (median) |
|---|---|---|---|---|
| Random | 0 % (0–1.9 %) | — | — | 375 mm |
| Jacobian IK baseline | 100 % (98.1–100 %) | 100 % | 1.65 s | < 0.001 mm |
| **PPO** (3 seeds, ~5 min CPU training each) | 100 % (99.4–100 %) | 100 % | 1.70 ± 0.09 s | 1.5 ± 0.3 mm |
| SAC (3 seeds, same 200k-step budget) | 63.2 % (59.2–66.9 %) | 43 ± 19 % | 1.98 ± 0.28 s | 27 ± 6 mm |

PPO matches the model-based IK controller on success and holding using only its observations. IK stays more precise and smoother. With default settings and the same budget, SAC lagged far behind: its automatic entropy tuning collapsed exploration before it discovered the hold reward. Details, learning curves and the reward-design lesson behind task v2 are in [docs/results.md](docs/results.md).

Held-out grasping (200 cube placements, any position and rotation in the workspace; success = verified grasp lifted ≥ 5 cm):

| Policy | Success (95 % CI) | Time to first grasp (median) |
|---|---|---|
| Scripted top-down grasp (privileged cube pose, IK) | 100 % (98.1–100 %) | 6.98 s |
| PPO, plain | 0 %: learned to close on the cube, never to lift | — |
| **PPO + pre-grasped training starts** (3 seeds) | **80.5 %** (77.1–83.5 %) | **4.31 s** |

Details, failure analysis and the curriculum story: [docs/grasp_task.md](docs/grasp_task.md).

Held-out transport (200 grasp-to-transport handoffs from scripted and learned grasps; carry the held cube to a 3-D goal and hold it there with a verified grasp):

| Policy | Success (95 % CI) |
|---|---|
| **Scripted closed-loop carry** (privileged cube pose, IK) | **92.0 %** (87.4–95.0 %) |
| PPO, plain (two diagnosed attempts) | 0 %: sinks the cube onto the table, then loses the grip |
| PPO residual on the scripted carry | 92.5 % (88.0–95.4 %), statistically identical (same 14 failures) |

The remaining transport failures are cubes creeping out of off-centre grips while held, mostly from handoffs of one learned grasp policy. Transport reliability is limited by grasp quality, not by the carry. Details: [docs/transport_task.md](docs/transport_task.md).

Evaluate a baseline yourself: `uv run python scripts/evaluate.py --policy ik`

## Roadmap

| Milestone | Scope | Status |
|---|---|---|
| 1 | Reproducible environment, pinned Panda model, headless simulation, tests | Done ([details](docs/environment.md)) |
| 2 | Custom reaching environment and classical IK baseline | Done ([task](docs/reach_task.md), [results](docs/results.md)) |
| 3 | PPO and SAC reaching policies with benchmark | Done ([results](docs/results.md)) |
| 4 | Grasp and transport skills | Done ([grasp](docs/grasp_task.md), [transport](docs/transport_task.md)) |
| 5 | Multi-policy FSM supervisor and full pick-and-place | Planned |
| 6 | Robustness, evaluation, documentation | Planned |

## License

MIT, see [LICENSE](LICENSE).

The Franka Emika Panda model comes from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Apache-2.0). It is downloaded by `scripts/fetch_assets.sh` at a pinned commit and is not redistributed in this repository.
