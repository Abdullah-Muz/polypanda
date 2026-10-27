# Model cards

Trained policies are published as GitHub Release assets: `scripts/fetch_models.sh` downloads them into `runs/` and checks the archive's SHA-256. Every run directory holds:
- `metadata.yaml`: skill, observation and action schemas, robot model revision, git commit, seed, steps, wall clock, package-lock hash;
- the training and task configurations;
- the best (validation-selected) and final checkpoints, each with its observation-normalization statistics;
- the validation history.

`polypanda.policies.loader.load_run` refuses a policy whose schemas or robot model revision do not match the code.

**All policies:**
- act on a simulated Franka Panda in MuJoCo 3.16 (Menagerie model `cda5236f`), at 20 Hz;
- output either 7 joint-position increments (`joint-increment-v1-7d`, at most 0.015 rad per joint per step; Milestones 1–7), or a motion of the hand that inverse kinematics turns into joint targets (`cartesian-v1-4d` / `-3d` / `-6d`, Milestones 8–9; [cartesian_actions.md](cartesian_actions.md));
- are SB3 PPO or SAC MLPs (128 × 128, ReLU) trained on CPU.

**None has been tested on hardware.** They are research artefacts for this simulation, not controllers for a real robot.

| Skill | Recommended policy | Use it with |
|---|---|---|
| Reach | `reach_ppo_v2_seed{0,1,2}` | ReachEnv |
| Grasp | `grasp_cart_seed{0,1,2}` (hand motion); `grasp_v2yaw_seed{0,1,2}` (joint increments) | pick-and-place, and the work-order cell's infeed and buffer |
| Transport | `transport_slow_seed{0,1,2}` | pick-and-place (the cell's moves are scripted) |
| Place on a table | `place_cart_seed{0,1,2}` | pick-and-place |
| Place on a tower | `place_stack_v3_seed{0,1,2}`, **aimed at the fixture** | the work-order cell |

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

## Grasp: wide region (`grasp_wide_seed{0,1,2}`), not adopted (Milestone 7)

The v2 grasp task on the whole feasible table, nothing else changed (1M steps, about 50 min each). With fair checkpoint selection it is no better on the wide set than the v2 grasp (81.3 vs 81.8 %) and worse on the original set ([generalization.md](generalization.md)).

## Grasp: among other cubes, v3 (`grasp_v3_seed{0,1,2}`), not adopted (Milestone 7)

Grasping one cube with two others nearby, observing all three (1.5M steps, about 90 min each). 65.0 / 56.2 / 41.5 % on the original, wide and clutter sets, with seeds from 21.5 to 67 % on clutter; the v2 grasp, blind to the other cubes, does better ([generalization.md](generalization.md#73-grasping-among-other-cubes-task-v3)).

## Grasp: hand motion (`grasp_cart_seed{0,1,2}`), recommended (Milestone 8)

| | |
|---|---|
| Task | the v2 grasp task (any cube pose in the original region, lift ≥ 5 cm with a verified grasp), with 4-D hand-motion actions (`cartesian-v1-4d`: x, y, z and wrist yaw; the gripper points down) |
| Training | 1M steps, 8 envs, 100-episode validation, about 60–75 min each |
| Held-out result | **91.3 %** original, **85.0 %** wide, **84.7 %** clutter (3 seeds pooled; per seed 81.5 / 97.5 / 95.0 % original). Meets the Milestone 7 target without training on the wide region or on clutter |
| In pick-and-place | 91.8 % with scripted transport and place; 88.5 % as part of the learned chain (grasp, transport and place all learned) |
| In the work-order cell | at the infeed and the buffer slots through station frames, without retraining: 1515/1515 picks per seed in Layer 1 and 2010/2010 per seed in Layer 2 ([workcell.md](workcell.md)) |
| Limitations | seed 0 is the weakest (81.5 %); failures are never lining up or knocking the cube over during the approach. The scripted grasp, planning from the true cube pose, reaches 100 % |

## Transport: hand motion, 0.12 m/s (`transport_slow_seed{0,1,2}`), recommended (Milestone 8)

| | |
|---|---|
| Task | carry a held cube to a 3-D goal and hold it there (TransportEnv v2) with 3-D hand-motion actions (`cartesian-v1-3d`; the wrist is held), a 0.12 m/s top speed and the action bound penalty |
| Training | 1M steps, 8 envs, about 73 min each |
| Held-out result | **90.2 %** (89.0 / 89.5 / 92.0) on `transport_nominal_v1`, against 92.0 % for the scripted carry (p = 0.16) |
| In pick-and-place | 99.5 % with the other skills scripted; part of the 88.5 % learned chain |
| Limitations | all failures are the weak grip's slow creep, as for the scripted carry. Not used in the work-order cell, whose moves between stations are scripted |

## Transport: hand motion, 0.2 m/s (`transport_cart_seed{0,1,2}`), superseded

The same task at 0.2 m/s top speed: 76.2 % (83.5 / 60.0 / 85.0), losing cubes while carrying 1.8× faster than the scripted carry. `transport_cart3d_nopenalty_seed0` is the 3-D pilot without the bound penalty: about 0 %, its mean action stuck beyond the bounds ([cartesian_actions.md](cartesian_actions.md#83-learned-transport-two-pilots-and-two-fixes)).

## Place on a table: hand motion (`place_cart_seed{0,1,2}`), recommended for pick-and-place (Milestone 8)

| | |
|---|---|
| Task | stand a held cube upright on a target on the table and release it (PlaceEnv, `place-v2-48d`) with 6-D hand-motion actions (`cartesian-v1-6d`) and a cube-tilt penalty |
| Training | 1M steps, 8 envs, about 25 min each |
| Held-out result | 95.1 % (96.5 / 98.0 / 90.9) on the place set; **99.5 %** inside pick-and-place, statistically indistinguishable from the scripted place |
| Limitations | trained on the table only: in the work-order cell it builds every 1-high tower but topples every cube set on another cube. Use the stacking place there |

## Place on a tower: stacking v1 (`place_stack_seed{0,1,2}`) (Milestone 9)

| | |
|---|---|
| Task | the table place task, with each episode setting the cube on a tower of 0–5 cubes at the target (`PlaceConfig.stack`); moving a tower cube by more than 1 cm fails the episode |
| Training | 2M steps, 8 envs, about 2.2 h each |
| In the work-order cell | Layer 1: **599/600** orders (3 seeds); Layer 2: **587/600** |
| Limitations | every failed order is a 4-high tower: a placement nudges the stack by a few millimetres and the stack then creeps in the simulator |

## Place on a tower: stacking v2 (`place_stack_v2_seed{0,1,2}`), negative result

v1 plus a charge for every millimetre a tower cube moves, failing at 5 mm. Not clearly gentler in the training scene; in the cell some seeds set the first cube 9–16 mm off and the overhang crept. 8 of 30 Layer 1 and 7 of 21 Layer 2 orders failed before the evaluations were stopped ([workcell.md](workcell.md#v2-penalise-every-nudge)). Do not use.

## Place on a tower: stacking v3 (`place_stack_v3_seed{0,1,2}`), recommended for the cell

| | |
|---|---|
| Task | v1 plus a bonus for placements within 1 cm and a softer nudge charge; trained to aim at the top tower cube's actual position |
| Training | 2M steps, 8 envs, about 80 min each; best validation 98 / 98 / 99 % |
| In the work-order cell, **aimed at the fixture** | Layer 1: **599/600** orders (3 seeds); Layer 2: **595/600** (p = 0.057 against v1) |
| Use | the cell aims it at the fixture by default (`LearnedCell(aim="fixture")`). Aimed at the cube below, as trained, its small repeatable error adds up into a leaning stack: 20 of 165 Layer 1 and 38 of 124 Layer 2 orders failed ([workcell.md](workcell.md#v3-in-the-cell-a-root-cause-analysis)) |
| Limitations | seed 0 nudges towers most and has 4 of the 5 Layer 2 failures; a first cube placed 6–8 mm off on the table can leave an overhang that creeps |

## Pilots and failed attempts

| Run | Outcome |
|---|---|
| `reach_ppo_pilot_seed0` | task v1 (episodes ended on success): reached all targets, held only 14.5 % |
| `reach_sac_pilot_seed0` | first SAC pilot, 100k steps |
| `grasp_ppo_pilot_seed0` | no curriculum: learned to close the fingers, never to lift (0 %) |
| `transport_ppo_pilot_seed0`, `transport_ppo_v2_seed0` | plain PPO transport: 0 %, sank the cube onto the table, then lost grip |
| `place_stack_v3gate1cm_seed*`, `place_stack_v3check1cm_seed*`, `place_stack_v3centring_seed*` | three stacking v3 attempts stopped at 0.3–0.5M steps with about 0 % validation (an unbounded centring cost). Published without checkpoints or metadata: their configs and validation history only |

The failures are documented in the task pages and kept so every reported number can be reproduced.
