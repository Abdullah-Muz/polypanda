# Transport task (Milestone 4)

Carry a grasped cube to a 3-D goal and hold it there, following the brief: *"assume an object is already grasped, then learn to move it toward a placement target while keeping it secure."* Obstacles are a planned extension and are not part of v1.

## T1: can the grip survive carrying? (physics check)

The scripted grasp lifted the cube, then IK steered it to random goals, 20 carries per speed:

| IK gain | Real drops (finger contact lost) | Success at goal (verified grasp, within 2 cm) | Final cube error (median) | Cube tilt (median / max) |
|---|---|---|---|---|
| 0.2 | 0 / 20 | 20 / 20 | 12.6 mm | 21° / 39° |
| 1.0 | 0 / 20 | 14 / 20 | 13.6 mm | 19° / 33° |
| 3.0 | 0 / 20 | 16 / 20 | 16.7 mm | 15° / 34° |

- The about-2 N grip never lost contact while carrying.
- The cube **slides about 13 mm in the hand**, so a mover that aims the hand using the cube offset measured at pickup lands 1–2 cm off. The baseline therefore steers in closed loop on the cube's current position.
- The cube **pivots 15–39° in the grip**: the clamping force is small and torsional friction is 0.03. This is kept as a documented property of the model's gripper rather than changed.

## T2: handoff start states (`polypanda.environments.handoff`)

Transport starts where a grasp controller hands over. A **handoff state** is the full MuJoCo integration state (positions, velocities, controls, solver warm start; restored bit-exactly with `mj_setState`) captured when a grasp controller has held a verified grasp (GraspEnv success: grasp predicate true, cube ≥ 5 cm up) for 10 consecutive policy steps. That is the moment a supervisor would switch skills.

Pools mix four grasp controllers in equal shares: the scripted grasp and PPO grasp seeds 0–2. Transport is therefore trained and evaluated on the start distribution it will actually receive in pick-and-place, not only on tidy scripted starts.

| Pool | States | Use | Grasp capture rate per source (scripted / PPO 0 / 1 / 2) |
|---|---|---|---|
| `data/handoff/transport_train_v1.npz` | 1200 | training starts | 100 % / 85.0 % / 86.2 % / 83.8 % |
| `eval_sets/transport_nominal_v1.npz` | 200, with goals | held-out evaluation | 100 % / 89.3 % / 89.3 % / 79.4 % |

Placements come from SeedSequence streams with a separate spawn key per pool and source. Each file stores a digest, the Menagerie commit and the state size, all checked on load. The pools depend on grasp checkpoints that are not in git, so they are committed as artifacts. Evaluation goals are uniform in x 0.35–0.65 m, y ±0.25 m, z 0.10–0.25 m (cube centre) and at least 15 cm from the cube at handoff.

## T3: environment (`polypanda.environments.transport.TransportEnv`, `PolyPanda/Transport-v0`)

Configuration: `configs/transport.yaml`. The dataclass defaults match the file (tested).

- **Reset** restores a handoff state and keeps the grasp skill's joint commands (`adopt_ctrl`, so nothing jumps at the switch). The gripper stays closed throughout; releasing belongs to the place skill.
- **Observation** (`transport-v1-47d`): arm `q`/`dq`/`q_cmd`, TCP pose (6-D rotation), finger gap, cube relative to the TCP, **cube slip velocity in the hand frame**, cube height and tilt, finger contacts, goal, and goal − cube.
- **Reward:**

| Term | Value | Purpose |
|---|---|---|
| progress | 20 × reduction in cube–goal distance | telescoping |
| hold | 0.5 × (1 − d / 2 cm) per step at the goal with a verified grasp | arrive and stay |
| action / **action rate** | −0.005‖a‖², **−0.05‖aₜ − aₜ₋₁‖²** | smooth motion slips less; random actions drop the cube every time |
| gripper / cube tilt | −0.1 per rad each | keep things upright |
| floor | −0.1 per step | |
| drop | −5, terminate | both-finger contact lost for 5 steps, cube back on the table, or tipped past 80° |

- **Success:** cube centre within 2 cm of the goal with a verified grasp, for 5 steps, and **not dropped**, at the end of the 10 s episode. (The "not dropped" condition was added after the baseline evaluation found a cube at the goal tipped 80.1° in the fingers; a regression test now covers it.)
- **Reward scale** on 10 held-out starts: scripted carry +78.9 (10/10), random −24.8 (drops 10/10), standing still −5.8.
- **Throughput:** about 310 env steps/s single process and about 1,230 with 8 workers. Finger–cube contact is active on every step.

## T4: scripted carry baseline

`ScriptedCarryDriver`: `PoseIK`, gain 0.2, steering the hand so the cube's **current** position (privileged) goes to the goal, with the gripper levelled. Evaluated on all 200 held-out cases at commit `7ec2e71` (clean):

| Metric | Value |
|---|---|
| Success | **92.0 %** (95 % CI 87.4–95.0 %) |
| Drops | 7.5 % (15); plus 1 case at the goal without a verified grasp |
| Time to first success | median 2.05 s, p90 2.95 s |
| Final distance | median 0.2 mm |
| Cube tilt | median 18.5°, max 80.1° |
| Floor contact / action jerk | 0 % / 0.0015 |

By the grasp controller that produced the handoff:

| Handoff source | Success | Drops |
|---|---|---|
| scripted grasp | **98 %** | 2 % |
| PPO grasp seed 1 | 96 % | 4 % |
| PPO grasp seed 2 | 88 % | 10 % |
| PPO grasp seed 0 | **86 %** | **14 %** |

**Findings:**
- **Handoffs from learned grasps are harder to carry.** They hold the cube less centrally and more tilted, which justifies building the pool from learned as well as scripted grasps.
- **Drops happen while holding, not while moving.** 13 of 15 drops occur after 2.9 s, after the cube has typically reached the goal (median 2.05 s). Dropped cubes had tilted far more (median 37° vs 17°). Under the weak grip the cube slowly creeps and rotates out under gravity, more so on longer carries (median start distance 374 mm for drops vs 253 mm for the rest). This matters for the place skill: the sooner a cube is set down, the better.

## T5: PPO pilot (stalled), diagnosis and planned improvements

`transport_ppo_pilot_seed0`: `configs/transport_ppo.yaml` (same PPO settings as grasp, including `ent_coef 0.005`), 1M steps, 8 envs, 41 min, trained at `8e7a955` (clean).

- **Result: 0 % success.** Every validation test was 0 % (20 tests, 30 episodes each), no training episode succeeded (about 45k episodes), and the held-out evaluation scored 0 / 200 with every episode ending in a "drop".
- **Learning did happen, just not enough.** Training episodes lengthened from about 10 to 22 steps before a terminating event, and in the last 100k steps validation episodes reached 33–40 steps (1.7–2 s). Action noise fell from 1.0 to 0.26, and the progress reward rose (the cube moved toward the goal; closest median approach 11 cm, best 3.4 cm). The hold reward never appeared.

**Diagnosis: the robot mostly lowered the cube onto the table.** Instrumenting 40 held-out episodes showed the terminating condition was **"cube within 1 cm of the table" in 40 / 40**, never "finger contact lost for 5 steps" or "tipped over". Re-scoring all 200 held-out episodes with the v2 labels below (which also check whether both fingers still held the cube at touchdown) refines this: **171 (85.5 %) were set down while still held**, and 29 (14.5 %) reached the table without both fingers in contact, i.e. the cube was slipping out as it came down. Sinking is the dominant failure, though not the only one. By episode end the hand had moved down a median 6.3 cm (3.4–8.8 cm). Yet the goal is *above* the handoff height in 93 % of cases: the cube starts at a median 11.1 cm and goals sit at 10–25 cm, a median 6.3 cm higher. The policy learned to swing the cube sideways toward the goal with joint motions that trade height for reach, and nothing told it it was sinking until the binary termination at 1 cm. Moving sideways at constant height needs several joints coordinated, which random exploration rarely discovers.

The first hypothesis, too-violent initial exploration noise, was ruled out by this check: noise had already fallen to 0.26 while the "drops" continued. Recording the specific terminating condition settled it.

**Improvements, implemented as task v2** (`configs/transport.yaml`; v1 preserved as `configs/transport_v1.yaml`):
1. **Height-keeping signal.** `low_height`: −10 per metre the cube is below a safe carrying height (`safe_lift` 6 cm above the table), every step. Sinking is now signalled long before touchdown; v1 had only the terminal −5 at 1 cm. All handoffs (lift ≥ 6.3 cm) and goals (≥ 7 cm) are above the safe height.
2. **Honest failure labels.** The single `dropped` label is split, with separate reward terms:
   - **lowered onto table**: the cube touches down while *both fingers still hold it* (`touchdown`, −5);
   - **lost grip**: finger contact lost for 5 steps, the cube tipped past 80°, or the cube reached the table unheld (`drop`, −5).

   A test writing the v2 code caught a mislabel: a cube released by opening the fingers reached the table before 5 contact-free steps had passed, so it was briefly counted as "lowered". The definition now requires both fingers in contact at touchdown.
3. Keep all other settings unchanged, so the effect of the fix is attributable. Then pilot → 3 seeds → held-out comparison with the scripted baseline (92.0 %).
4. Possible later improvements: a goal-height range that also exercises upward moves more strongly, gentler initial exploration (`log_std_init`), or Cartesian end-effector actions as a separately labelled ablation.

### T5b: v2 pilot (also stalled)

`transport_ppo_v2_seed0`: identical settings on task v2, 1M steps, trained at `df08683` (clean). The laptop slept for about 45 min mid-run, which paused but did not affect training.

- **Held-out: 0 / 200.** Failures: 102 lowered onto table, 98 lost grip. Closest approach 1.7 cm; median closest 12.7 cm.
- **The height signal worked as designed, but was not sufficient.** During training the touchdown penalty per step fell from −0.28 to −0.085 and the low-height penalty from −0.14 to −0.06. Episode lengths still matched v1 (about 22 steps, around 1.1 s at 800k steps), and failures shifted toward lost grip. The hold reward never appeared.
- **Interpretation:** carrying a weakly held cube smoothly toward a goal, by discovering coordinated 7-joint increments through trial and error, is beyond plain PPO at this budget. The scripted carry does it easily because it plans in hand coordinates with an exact kinematic model. Fixing one failure mode exposed the next: the bottleneck is discovering smooth, coordinated motion at all, not a single reward bug.

### T5c: residual RL on the scripted carry

`TransportResidualEnv` (task `transport_residual`): the policy outputs a correction added to the scripted closed-loop carry, applied = clip(base + 0.3 × policy). The observation appends the base action (`transport-residual-v1-54d`), and a −0.01‖output‖² penalty discourages needless deviation. A zero output reproduces the baseline exactly (tested). Initial exploration noise was set to std ≈ 0.5 (`log_std_init −0.7`) after measuring the base carry under residual noise: 95 % success with no noise, 80 % at std 0.5, and 32 % at the SB3 default std 1.0 (60 % lost grip).

`transport_residual_seed0`: 1M steps, trained and evaluated at `de474a4` (clean). Wall clock was 66 min, inflated by background load on the laptop (OneDrive sync, viewer) and the per-step base IK; results are unaffected. Training behaved completely differently from plain PPO: episodes ran almost the full 200 steps, validation stayed at 87–100 %, the hold reward rose from 0.29 to 0.38 per step, and correction size and noise shrank steadily.

**Held-out (200 cases):**

| | Scripted carry | Residual RL |
|---|---|---|
| Success (95 % CI) | 92.0 % (87.4–95.0 %) | 92.5 % (88.0–95.4 %) |
| Lost grip | 7.5 % | 6.5 % |
| Time to first success (median) | 2.05 s | 1.90 s |
| Final distance (median) | 0.2 mm | 0.6 mm |
| Action jerk (median) | 0.0015 | 0.0175 |

Paired on the same 200 cases, both succeed in 183, **both fail in the same 14**, only scripted succeeds in 1, and only residual succeeds in 2 (exact McNemar p = 1.0). **The residual matches the baseline but does not improve it.** It arrives slightly sooner with jerkier actions and changes the outcome in only 3 cases.

**Why no gain: transport failures are limited by grasp quality, not by carrying.** The shared failures are cubes creeping out of the grip while held at the goal (all are `lost_grip`), and they concentrate on handoffs from PPO grasp seed 0 (84 % success from its handoffs vs 100 % from the scripted grasp's). A cube grasped off-centre in the about-2 N grip slowly works loose during the hold, and no arm correction within the residual's range repairs how the fingers hold it. This is a cross-skill dependency: improving end-to-end reliability means improving grasp quality, or placing the cube sooner, rather than refining the carry.

### Transport conclusion (Milestone 4)

| Approach | Held-out success |
|---|---|
| Scripted closed-loop carry | **92.0 %** |
| Plain PPO v1 | 0 % (sinks the cube onto the table) |
| Plain PPO v2 (height-keeping signal) | 0 % (sinking reduced; lost grip instead) |
| Residual PPO on the scripted carry | 92.5 % (equivalent to the baseline, p = 1.0) |

The scripted closed-loop carry is the transport skill used in pick-and-place: it is as reliable as the best learned variant, much smoother, and simpler. The learning results are reported as findings: plain PPO cannot discover smooth coordinated carrying under this weak grip at a 1M-step budget, and residual RL cannot fix failures that originate in the grasp.
