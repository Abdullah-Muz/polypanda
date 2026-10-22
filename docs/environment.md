# Environment and simulator baseline

Measured results from Milestone 1. Unless noted, numbers come from `scripts/verify_install.py` on the development laptop (AMD Ryzen 7 250, 8 cores / 16 threads, WSL2 with Ubuntu 24.04 and a 14 GiB memory cap, no CUDA).

## Software

| Component | Version |
|---|---|
| Python | 3.12.3 |
| MuJoCo | 3.16.0 |
| Gymnasium | 1.4.0 |
| Stable-Baselines3 | 2.9.0 |
| PyTorch | 2.14.1+cpu |
| NumPy | 2.5.4 |
| uv | 0.13.5 |

Exact versions of every dependency are in `uv.lock`. PyTorch is resolved from the CPU-only wheel index, so no CUDA libraries are installed.

## Robot model

Franka Emika Panda from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) at commit `cda5236f98383f6fbaa4820c6b18dba892cd087f` (2026-10-19), Apache-2.0. `scripts/fetch_assets.sh` downloads only `franka_emika_panda/` and Menagerie's root files (about 44 MB).

- Physics timestep: 0.002 s (500 Hz)
- `nq` = 9, `nv` = 9 (7 arm joints + 2 finger joints), `nu` = 8 (7 arm actuators + 1 gripper)
- `home` keyframe: arm `q = [0, 0, 0, -1.5708, 0, 1.5708, -0.7853]` rad; its arm control targets equal its joint positions

### Tool centre point (TCP)

Menagerie's Panda has no end-effector site. PolyPanda adds one named `tcp` to the `hand` body when the model is loaded (`add_tcp_site`, via `mujoco.MjSpec`), so the pinned Menagerie files stay unmodified.

- Location: 0.1029 m along the hand's z axis, the midpoint between the two main fingertip pads (finger body offset 0.0584 m plus pad centre 0.0445 m). Franka documents 0.1034 m from flange to TCP.
- At `home`, the TCP is at (0.5545, 0.0, 0.5216) m in the world frame, with the gripper pointing straight down.
- Tests check that it stays midway between the pads at any finger opening and that it moves correctly under joint rotation.
- `uv run python scripts/view_panda.py` opens the viewer with the TCP drawn as a red sphere.

### Arm actuators

Each arm actuator is a position servo: `tau = kp * (q_cmd - q) - kd * qdot`, clamped to the force range.

| Actuator | Joint | kp (Nm/rad) | kd (Nm·s/rad) | ctrlrange (rad) | forcerange (Nm) |
|---|---|---:|---:|---|---|
| actuator1 | joint1 | 4500 | 450 | [-2.8973, 2.8973] | ±87 |
| actuator2 | joint2 | 4500 | 450 | [-1.7628, 1.7628] | ±87 |
| actuator3 | joint3 | 3500 | 350 | [-2.8973, 2.8973] | ±87 |
| actuator4 | joint4 | 3500 | 350 | [-3.0718, -0.0698] | ±87 |
| actuator5 | joint5 | 2000 | 200 | [-2.8973, 2.8973] | ±12 |
| actuator6 | joint6 | 2000 | 200 | [-0.0175, 3.7525] | ±12 |
| actuator7 | joint7 | 2000 | 200 | [-2.8973, 2.8973] | ±12 |

Gripper: `actuator8` drives both fingers through a tendon, with ctrlrange [0, 255] (255 = fully open).

### Steady-state gravity offset

The model has no gravity compensation, so a servo holding a pose under load settles where its spring force balances gravity. The steady-state error is `gravity torque / kp`. After holding `home` for 10 s:

| Joint | Offset (mrad) | Holding torque (Nm) |
|---|---:|---:|
| joint2 (shoulder) | +6.58 | -29.62 |
| joint4 (elbow) | -6.31 | +22.10 |
| joint6 | -1.14 | +2.28 |
| joint5 | -0.33 | +0.67 |
| joints 1, 3, 7 | ~0 | ~0 |

For joint2, 29.62 / 4500 = 6.58 mrad, which matches the measurement. The offset is real behaviour, not a simulation fault, and it changes with pose. As a result, the commanded targets `q_cmd` never exactly equal the measured positions `q`, which is one reason both are part of the reach observation. Task success must be measured from the actual end-effector position, never from the commands.

## Performance

| Metric | Value |
|---|---|
| Headless physics, arm only, no contacts | ~42,000 steps/s on one core |
| Real-time factor | ~85x |
| Equivalent policy rate at 25 substeps per action | ~1,700 policy steps/s |
| Peak process memory (imports + model) | ~525 MB |

These are upper bounds for a bare `mj_step` loop. A Gymnasium environment, observation building, reward computation, object contacts and the RL update all reduce real training throughput, which will be measured separately.

## Viewer

`uv run python -m mujoco.viewer --mjcf=external/mujoco_menagerie/franka_emika_panda/scene.xml` opens and renders correctly through WSLg on the Radeon 780M. Training and tests never depend on the viewer.

## Reproducibility checks

- A fresh clone, followed by `fetch_assets.sh`, `uv sync --locked`, `verify_install.py` and `pytest`, passed locally.
- The same steps pass in GitHub Actions on `ubuntu-24.04`, starting from an empty cache.
