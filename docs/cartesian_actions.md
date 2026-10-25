# Hand-motion actions and a fully learned chain (Milestone 8)

Until Milestone 8 every policy commanded seven joint increments. Learned grasping worked (about 90 %), but plain PPO never learned to carry a cube (0 % in Milestone 4), and the scripted carry succeeded precisely because it plans in hand coordinates. Milestone 8 gives policies that option and uses it to learn all three skills.

**Outcome:**
- **Every skill is now learned.** The learned grasp beats the earlier learned grasp, transport is learned for the first time, and learned transport and learned place are statistically indistinguishable from their scripted versions inside the full task.
- **The fully learned chain** (grasp, transport and place all trained with PPO) completes **88.5 %** of the 200 held-out pick-and-place scenes, and 96 % for its best seed. That equals the v1.0 system, which had only a learned grasp (89.0 %, paired p = 0.84). It is about 40 % faster than the scripted chain.
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

### Follow-up: a lower top speed (adopted)

Failure analysis of the 0.2 m/s policies:

- **Where cubes were lost.** Cubes lost while holding at the goal matched the scripted carry (12–16 per seed against 11): this is the weak grip's slow creep. The extra losses came *while carrying*: 13–47 per seed against 4 for the scripted carry, early in the episode (median 2.5–3.6 s against 6.2 s).
- **Not grasp quality.** The learned carry also failed with clean scripted handoffs (29–45 of 50 against 49 of 50).
- **What the hand did.** Traces of the failed carries showed two behaviours:
  - **Seed 1 took large detours upward** (median height range 33 cm against 8 cm for the scripted carry on the same cases). The telescoping progress reward pays back on the return leg whatever a detour costs, so detours are free.
  - **All seeds carried about 1.8× faster than the scripted carry** (0.25 against 0.14 m/s at peak).

Two mechanism checks on the existing policies, without retraining:

| Change at test time (3 seeds pooled) | Transport success |
|---|---|
| None (0.2 m/s) | 76.2 % |
| Acceleration limit 2 / 1 / 0.5 m/s² (`cartesian.max_accel`) | 76.7 / 77.2 / 76.8 %: no effect |
| Top speed 0.16 / **0.12** / 0.08 m/s | 82.2 / **84.7** / 85.8 % |

Sudden starts and stops were therefore not the cause; the sustained speed was. The acceleration limit stays available but off.

`configs/transport_cartesian_slow_ppo.yaml` retrains with a top speed of 0.12 m/s, about the scripted carry's pace, and nothing else changed. 3 seeds, trained and evaluated at `9be3dfa` (clean):

| | Learned, 0.12 m/s | Learned, 0.2 m/s | Scripted carry |
|---|---|---|---|
| Held-out transport (200 cases) | **90.2 %** (87.5–92.3 %); 89.0 / 89.5 / 92.0 | 76.2 % | 92.0 % |
| Paired test | vs 0.2 m/s: p = 2.5 × 10⁻²⁰ | | vs learned 0.12 m/s: **p = 0.16** |
| Time to first success, median | 2.2–2.3 s | 1.7–2.0 s | 2.05 s |
| Inside pick-and-place (learned transport only) | **99.5 %** (597 / 600; p = 0.25 vs 100 %) | 86.7 % | 100 % |

- **The seeds now agree.** All three learned quickly, reaching 71–75 % by 200k steps except seed 0, and peaked at 95–97 % in validation; seed 1 is no longer a slow starter.
- **The detours disappeared without a reward change.** All remaining failures are the grip's slow creep, as for the scripted carry. The detour loophole in the progress reward is real, but it no longer costs anything measurable.
- **The 0.12 m/s transport is the learned transport from here on.**

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

The supervisor now takes a learned skill for every stage, each running in its own task configuration (`scripts/evaluate_pickplace.py --grasp ... --transport ... --place ...`). Seed *s* of each skill forms chain *s*. 200 scenes per seed; the chain with the 0.12 m/s transport was evaluated at `9be3dfa`, the others at `8a67dc0` (clean):

| System | Original scenes | Wide scenes | Time to placed, median | Final xy error, median |
|---|---|---|---|---|
| **Fully learned** (0.12 m/s transport; 3 seeds pooled) | **88.5 %** (85.7–90.8 %); 78.5 / 96.0 / 91.0 | **80.8 %** (77.5–83.8 %); 71.0 / 86.5 / 85.0 | 7.3–7.6 s | 4–7 mm |
| Fully learned, first transport (0.2 m/s) | 83.3 %; 77.0 / 81.5 / 91.5 | 70.7 % | 7.0–7.8 s | 4–7 mm |
| Learned grasp + scripted transport and place | 91.8 % | 86.7 % | | |
| Learned transport only (0.12 m/s) | 99.5 % | | | |
| Learned place only | 99.5 % | | | |
| **All scripted (baseline)** | **100 %** | **98.5 %** | 12.4 s | 1.7 mm |
| v1.0 system (joint-action learned grasp, selected checkpoints, + scripted transport and place) | 89.0 % | 81.2 % | | |

- **The fully learned chain equals the v1.0 system**, which had only one learned skill: 531 vs 534 of 600 on the original scenes (paired p = 0.84) and 485 vs 487 on the wide scenes (p = 0.93). The slower transport added 5.2 points on the original scenes (p = 3 × 10⁻⁶) and 10.1 on the wide ones.
- **The grasp is now the weakest link.** The most common final failure is a cube knocked over during the grasp (32 of 69), and the weakest chain seed is the one with the weakest grasp (grasp seed 0: 81.5 %).
- **Speed and precision.** About 7.5 s per cube against 12.4 s for scripted, mostly because the learned grasp takes about 4 s against 7.8 s. Less precise (4–7 mm against 1.7 mm), well within the 2 cm tolerance.
- **The wide scenes are harder** (80.8 %): transport and place were trained only on the original region.

## What Milestone 8 showed

1. **The action interface mattered more than the reward or the training region.** The same PPO settings that could not learn transport with joint increments learned it with hand-motion actions, and the grasp improved most where the task was hardest.
2. **Three failure diagnoses got transport from 0 % to the scripted level, each from looking at what the policy actually did:** the wrist spin, a policy mean stuck outside the action bounds, and carrying too fast. The bound penalty is a general PPO remedy, now available to every task.
3. **Cheap test-time checks before retraining paid off.** Running the existing policies with a changed controller ruled out acceleration and confirmed speed in about 20 minutes, before a 1.5-hour retraining.
4. **A forgiving task makes seeds agree.** At 0.2 m/s the transport seeds spread from 60 to 85 %; at 0.12 m/s from 89 to 92 %. Reporting 3 seeds is what showed it.
5. **Next:** multi-cube stacking (Milestone 9). The grasp is now the weakest learned skill.
