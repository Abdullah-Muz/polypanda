# Hand-motion actions and a fully learned chain (Milestone 8)

Until Milestone 8 every policy commanded seven joint increments. Learned grasping worked (about 90 %), but plain PPO never learned to carry a cube (0 % in Milestone 4), and the scripted carry succeeded precisely because it plans in hand coordinates. Milestone 8 gives policies that option and uses it to learn all three skills.

**Outcome:**
- **Every skill is now learned.** The learned grasp and the learned place match or beat their earlier versions, and transport is learned for the first time.
- **The fully learned chain** (grasp, transport and place all trained with PPO) completes **83.3 %** of the 200 held-out pick-and-place scenes; its best seed completes 91.5 %. It is about 40 % faster than the scripted chain.
- **Scripted baselines.** The scripted controllers are reported next to every result as reference baselines; the all-scripted chain completes 100 %.

## 8.1 The interface (`polypanda.control.action_adapter.CartesianActionAdapter`)

Selected per task with `action.space: cartesian` (`ActionConfig.cartesian` holds the settings). The action moves a persistent **target pose of the hand**, and inverse kinematics turns it into the same joint targets the position servos always received:

| Variant | Action | Used for |
|---|---|---|
| 4-D (`cartesian-v1-4d`) | x, y, z (±1 cm per step) and wrist yaw (±0.05 rad); the gripper is kept pointing down | grasp |
| 3-D (`cartesian-v1-3d`) | x, y, z; the wrist is held | transport |
| 6-D (`cartesian-v1-6d`) | x, y, z and a rotation vector, within 0.6 rad of pointing down | place (standing the cube upright needs the hand to tilt) |

How it works:

- **IK solver.** Each step runs two damped least-squares iterations (`PoseIK`, exact kinematics on a private copy of the arm). A small null-space pull toward the home posture keeps the redundant elbow from drifting.
- **No wind-up, no jumps.** The target stays within 3 cm of the commanded hand and inside a box, so it cannot wind up while the arm is blocked. A hand that starts tilted is levelled at no more than 0.05 rad per step, so nothing jumps at a skill handoff.
- **Accuracy.** A 10 cm command moves the hand 100–107 mm with under 1.2 mm of sideways drift; a yaw command turns the wrist exactly. The remaining offset is the position servos' constant gravity sag of about 7 mm, which joint actions have too.
- **Cost.** About 0.5 ms per step, so environments run at about 85 % of their joint-action speed.
- **Run metadata.** Runs record their action schema, and loading a policy into an environment with a different interface is refused. All 27 earlier runs still load.
- **Recovery.** The supervisor's recovery has its own joint-increment controller, since it would otherwise send 7-joint commands to a hand-motion interface. The first version did exactly that and would have crashed at the first retry; a test now covers it.

A one-line rule, "move the cube toward the goal", carries 15 of 20 cubes with hand-motion actions. That was the first sign that learning transport would become feasible.

## 8.2 Grasp with hand-motion actions

`configs/grasp_cartesian_ppo.yaml`: the adopted grasp task (v2, committed yaw) with 4-D actions and nothing else changed, plus 100-episode validation. 3 seeds, trained at `4a66a0b` (seed 0) and `9ae56da` (seeds 1–2), and evaluated at `9ae56da` (all clean). The two commits differ only in the transport options added in between.

| Held-out set (3 seeds, 600 episodes) | Hand-motion grasp | Joint-action grasp (v2, selected) | Paired test |
|---|---|---|---|
| Original | **91.3 %** (88.8–93.3 %); 81.5 / 97.5 / 95.0 | 89.8 % | p = 0.40 |
| Wide (whole feasible table) | **85.0 %** (81.9–87.6 %); 76.5 / 86.5 / 92.0 | 81.8 % | p = 0.12 |
| Clutter (2 other cubes) | **84.7 %** (81.6–87.3 %); 72.5 / 90.5 / 91.0 | 78.8 % | **p = 0.004** |
| Pick-and-place, original | **91.8 %** | 89.0 % | p = 0.09 |
| Pick-and-place, wide | **86.7 %** | 81.2 % | **p = 0.004** |

- **This meets the Milestone 7 target** (at least 85 % wide and at least 87 % original). Neither wide-region training nor distractor training had reached it ([generalization.md](generalization.md)); changing the action interface did, without training on either.
- **The gains grow where the task is harder.** Moving the hand directly makes the motion consistent across the table and around other cubes.
- **This is the learned grasp.** The remaining failures on the original set split between never lining up (23 of 52) and knocking the cube over during the approach (22).

## 8.3 Learned transport: two pilots and two fixes

`configs/transport_cartesian_ppo.yaml`: TransportEnv (task v2) with hand-motion actions; all other settings match the Milestone 4 PPO runs, apart from 100-episode validation.

1. **4-D pilot: 0 %. The policy spun the wrist.** Half to three quarters of its actions were at full strength, and it turned the wrist 0.3–2.3 rad while carrying, flinging the cube out of the roughly 2 N grip. Transport never needs to turn the wrist. *Fix:* the 3-D variant, which holds the wrist (stopped at 600k steps; files kept outside the repository).
2. **3-D pilot: 0 % (1 % at the end). The hand kept rising.** The policy commanded "up" at full speed the whole episode, carrying cubes meant for 15 cm to 52 cm while they slid out of the fingers. The cause was a PPO pathology, not the task. The policy's mean y and z actions sat beyond the ±1 bound in about half of all steps (up to 2.6). Every sample there clips to the same action, so the returns carry no signal to bring the mean back. *Fix:* the **action bound penalty** (`action.bound_penalty`, as in rl_games' bounds loss). The action space widens to ±5, the environment still clips to ±1, and the excess costs 0.1 × excess². Inside the bounds nothing changes. The run is kept as `runs/transport_cart3d_nopenalty_seed0`.
3. **With the penalty: it learned.** Validation went 1 → 2 → 4 → 38 → 68 → 70 → 76 → 80 → 87 → 82 %.

3 seeds, trained at `9ae56da` (seed 0) and `bf278d9` (seeds 1–2), and evaluated at `9ae56da` and `8a67dc0` (clean):

| Held-out transport set (200 cases) | Learned | Scripted carry |
|---|---|---|
| Success, 3 seeds pooled | 76.2 % (72.6–79.4 %); per seed 83.5 / 60.0 / 85.0 | 92.0 % |
| Main failure | lost grip (120 of 143) | lost grip (15 of 16) |
| Time to first success, median (seed 0) | 1.8 s | 2.05 s |

- **Seeds disagree.** Seed 1 was a slow starter: its validation stayed at 0–5 % for half of training before climbing to 73 %.
- **The weak grip is the limit, as in Milestone 4.** Cubes held off-centre creep out of the fingers. The learned carry is faster and less gentle than the scripted one.
- **Effect of the bound penalty on grasp: unknown.** The grasp policy also saturates its vertical action (78 % of steps), but there "full speed down, then full speed up" is roughly right, and grasp learned well without the penalty. Its effect on grasp is untested.

## 8.4 Learned place

The place task existed as an environment with a gated release; RL for placing had not been attempted. Changes before the first training run:

- **Registered as a task** (`place`).
- **Observation v2** (`place-v2-48d`) adds the cube's up axis. The v1 observation gave the tilt angle but not its direction, and standing a cube upright needs both; the scripted place uses the full orientation.
- **Optional cube-tilt penalty** while the cube is held (0.3 per rad in the learned configuration). It is larger than the existing gripper-tilt penalty of 0.1, which it would otherwise cancel, since righting the cube means tilting the hand.
- **6-D hand-motion actions with the bound penalty** (`configs/place_cartesian.yaml`). Everything else is the scripted place's task.

A trivial rule ("lower the cube onto the target, lift away after release") places 18 of 20 cubes, so the task is learnable. PPO learned it quickly and reliably: validation reached 96–99 % for all 3 seeds. Trained at `522d16a` (seed 0) and `8a67dc0` (seeds 1–2), and evaluated at `8a67dc0` (clean):

| | Learned place (3 seeds) | Scripted place |
|---|---|---|
| Held-out place set (198 cases per seed) | 95.1 % (93.1–96.6 %); 96.5 / 98.0 / 90.9 | 97.5 % |
| Inside pick-and-place (200 scenes per seed) | **99.5 %** (597 / 600); 99.0 / 99.5 / 100 | 100 % |
| Paired test inside pick-and-place, per seed | p = 0.5 / 1 / 1 | |

The place set is harder than what place sees in pick-and-place, where the scripted carry hands over a steadily held cube. In the full task the learned place is **statistically indistinguishable from the scripted place**. It is adopted.

## 8.5 The fully learned chain

The supervisor now takes a learned skill for every stage, each running in its own task configuration (`scripts/evaluate_pickplace.py --grasp ... --transport ... --place ...`). Seed *s* of each skill forms chain *s*. Evaluated at `8a67dc0` (clean), 200 scenes per seed:

| System | Original scenes | Wide scenes | Time to placed, median | Final xy error, median |
|---|---|---|---|---|
| **Fully learned** (3 seeds pooled) | **83.3 %** (80.1–86.1 %); 77.0 / 81.5 / 91.5 | 70.7 % (66.9–74.2 %); 75.0 / 56.0 / 81.0 | 7.0–7.8 s | 4–7 mm |
| Learned grasp + scripted transport and place | 91.8 % | 86.7 % | | |
| Learned transport only | 86.7 %; 90.5 / 78.0 / 91.5 | | | |
| Learned place only | 99.5 % | | | |
| **All scripted (baseline)** | **100 %** | **98.5 %** | 12.4 s | 1.7 mm |
| v1.0 system (joint-action learned grasp, selected checkpoints, + scripted transport and place) | 89.0 % | 81.2 % | | |

- **Where the chain loses.** Of the 100 failed scenes on the original set, the final attempt failed at the grasp in 46, transport in 31 and place in 23. The supervisor's retries recover part of the losses: first-attempt success is 70.5 / 71.5 / 88.5 % per seed.
- **Speed.** It is faster than scripted (about 7 s against 12.4 s), mostly because the learned grasp takes about 4 s against 7.8 s. It is less precise (4–7 mm against 1.7 mm), well within the 2 cm tolerance.
- **The wide scenes are harder** (70.7 %): transport and place were trained only on the original region.
- **Against the v1.0 system** (one learned skill), the fully learned chain is 5.7 points lower (83.3 % vs 89.0 %, paired p = 0.003). That is the cost of making transport learned too. Its best seed (91.5 %) is above the v1.0 system.

## What Milestone 8 showed

1. **The action interface mattered more than the reward or the training region.** The same PPO settings that could not learn transport with joint increments learned it with hand-motion actions, and the grasp improved most where the task was hardest.
2. **Two failure diagnoses were needed before transport learned, each from looking at what the policy actually did.** First the wrist spin, then a policy mean stuck outside the action bounds. The second fix (the bound penalty) is a general PPO remedy, now available to every task.
3. **Seed-to-seed variation remains the main weakness of the learned skills** (transport 60–85 %, chain 77–91.5 %). Reporting 3 seeds is essential.
4. **Next:** improve transport, the weakest learned link, then multi-cube stacking (Milestone 9).
