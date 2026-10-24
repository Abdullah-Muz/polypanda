# Model cards

Trained policies are published as GitHub Release assets: `scripts/fetch_models.sh` downloads them into `runs/` and checks the archive's SHA-256. Every run directory holds:
- `metadata.yaml`: skill, observation and action schemas, robot model revision, git commit, seed, steps, wall clock, package-lock hash;
- the training and task configurations;
- the best (validation-selected) and final checkpoints, each with its observation-normalization statistics;
- the validation history.

`polypanda.policies.loader.load_run` refuses a policy whose schemas or robot model revision do not match the code.

**All policies:**
- act on a simulated Franka Panda in MuJoCo 3.16 (Menagerie model `cda5236f`);
- output 7 joint-position increments (`joint-delta-v1`, at most 0.015 rad per joint per 50 ms step);
- are SB3 PPO or SAC MLPs (128 × 128, ReLU) trained on CPU.

**None has been tested on hardware.** They are research artefacts for this simulation, not controllers for a real robot.

## Reach: PPO v2 (`reach_ppo_v2_seed{0,1,2}`), recommended

| | |
|---|---|
| Task | move the gripper's tool point within 2 cm of a 3-D target and hold it (ReachEnv, task v2, observation `reach-v1-27d`) |
| Training | 200,704 steps, 8 envs, about 5 min each on a laptop CPU |
| Held-out result | 100 % success and hold on `reach_nominal_v1` (200 targets); final error 1.5 ± 0.3 mm; 1.70 s median to success |
| Compared with | Jacobian IK baseline: 100 %, under 0.001 mm error, smoother actions |
| Limitations | about 3× jerkier than IK; settles about 1.5 mm off target; trained without obstacles |

## Reach: SAC v2 (`reach_sac_v2_seed{0,1,2}`), comparison only

Same task and budget as PPO; default SAC settings with automatic entropy tuning. **63.2 %** held-out success: the entropy temperature collapsed before the hold reward was discovered ([results.md](results.md)). Kept for the PPO vs SAC comparison.

## Grasp: PPO v2 with committed grasp yaw (`grasp_v2yaw_seed{0,1,2}`), recommended

| | |
|---|---|
| Task | from the home pose, pick up a 4 × 4 × 6 cm cube placed anywhere in x 0.35–0.65 m, y ±0.25 m, any yaw, and hold it lifted ≥ 5 cm with a verified grasp (GraspV2Env, `grasp-v2-41d`). A gate closes the fingers when the hand is centred and aligned |
| Training | 1M steps, 8 envs, 30 % pre-grasped starts (training only), about 40 min each (3 in parallel) |
| Held-out result | **87.0 %** pooled over 3 seeds on `grasp_nominal_v1` (200 placements); 4.4 s median to first grasp |
| In pick-and-place | **89.2 %** end to end with the supervisor, scripted transport and scripted place (`pickplace_nominal_v1`) |
| Limitations | weakest when the wrist must turn 40–45° (65 %); much worse outside the training region (see [robustness.md](robustness.md)); touches the floor in about 10 % of episodes; the scripted grasp with the privileged cube pose reaches 100 % |

## Grasp: PPO v1 (`grasp_ppo_curr_seed{0,1,2}`), superseded

The first grasp policy (GraspEnv, `grasp-v1-41d`; same training recipe). **80.5 %** held-out grasp and 86.0 % pick-and-place, with 54 % at the 45° tie. Kept because all of Milestone 5's and the first robustness suite's results use it.

## Grasp: full v2 (`grasp_v2_seed{0,1,2}`), negative result

GraspV2Env with the committed yaw **plus** a pushing penalty and cube mass/friction randomization: 4.5–26 % held-out grasp. The policies learned to hover about 1 cm short of the cube ([grasp_task.md](grasp_task.md#task-v2-milestone-62)). Published for transparency; do not use.

## Transport: residual PPO (`transport_residual_seed0`), comparison only

Learned corrections (scaled by 0.3) on top of the scripted closed-loop carry (`transport-residual-v1-54d`). **92.5 %** vs the scripted carry's 92.0 % on `transport_nominal_v1`, with the same 14 failures (McNemar p = 1.0). The pick-and-place system uses the scripted carry.

## Pilots and failed attempts

| Run | Outcome |
|---|---|
| `reach_ppo_pilot_seed0` | task v1 (episodes ended on success): reached all targets, held only 14.5 % |
| `reach_sac_pilot_seed0` | first SAC pilot, 100k steps |
| `grasp_ppo_pilot_seed0` | no curriculum: learned to close the fingers, never to lift (0 %) |
| `transport_ppo_pilot_seed0`, `transport_ppo_v2_seed0` | plain PPO transport: 0 %, sank the cube onto the table, then lost grip |

The failures are documented in the task pages and kept so every reported number can be reproduced.
