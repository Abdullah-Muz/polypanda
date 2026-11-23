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
- are SB3 PPO or SAC MLPs (128 × 128, ReLU) trained on CPU; the v3 turn chooser is an ensemble of 5 small supervised networks, and the v3 recovery planner a 64 × 64 PPO policy over two discrete actions.

**None has been tested on hardware.** They are research artefacts for this simulation, not controllers for a real robot.

| Skill | Recommended policy | Use it with |
|---|---|---|
| Reach | `reach_ppo_v2_seed{0,1,2}` | ReachEnv |
| Grasp | `grasp_cart_seed{0,1,2}` (hand motion); `grasp_v2yaw_seed{0,1,2}` (joint increments) | pick-and-place, and the work-order cell's infeed and buffer |
| Transport | `transport_slow_seed{0,1,2}` | pick-and-place (the cell's moves are scripted) |
| Place on a table | `place_cart_seed{0,1,2}` | pick-and-place |
| Place on a tower | `place_stack_v3_seed{0,1,2}`, **aimed at the fixture** | the work-order cell |
| Grasp, upright or lying (v3) | `grasp_anypose9_seed{0,1,2}` (v3.2, adopted after the development results; the v3.2 grasp target is not met: seed 0 still misses standing cubes near the base turned close to −45°, see its card). `grasp_anypose6_seed{0,1,2}` (v3.1) is the grasp of every v3.1 cell result | the cell with recovery |
| Stand a cube upright (v3) | `standup_round12_seed{0,1,2}` (v3.1) with its own turn chooser, shield and re-planning | the cell with recovery |
| Rescue, replace or clear (v3) | a hand-written rule: the clear rule (v3.1, `symbolic.CLEAR_NEAR_RULE`) or the best rule (`symbolic.BEST_RULE`), level in physics (p = 0.84); `recovery_planner_seed{0,1,2}` is published for comparison | the cell with spares and a deadline |

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

## Grasp a cube upright or lying on its side: round 4 (`grasp_anypose4_seed{0,1,2}`) and round 5 (`grasp_anypose5_seed{0,1,2}`) (Milestone 10, superseded by v3.1)

| | |
|---|---|
| Task | grasp a cube upright or lying on a side anywhere in the cell's reach (0.30–0.78 m) and lift it 5 cm (`GraspV4Env`, `grasp_v4`; hand motion `cartesian-v1-4d`); the cell's gripper (1000 N/m, torsional friction); randomized cube mass and friction |
| Training | round 4: 2M steps warm-started from `grasp_cart`, with function-preserving input renormalization; round 5: 1M more steps with more lying cubes (70 %) and practice near the base (x 0.30–0.40, 30 % of episodes), learning rate 3e-4 → 3e-5 |
| Held-out result | round 4: **84.8 %** on the one-shot lying set (79.0 / 84.0 / 91.5; target 85 %), scripted 84.0 % (p = 0.74); upright 92.2 %. Round 5, development sets: lying 92.7 % (vs 86.2 %, p = 8e-5), upright 95.2 % |
| In the work-order cell | round 5 does every pick in v3 recovery: 96.6–99.6 % of about 2400 picks per seed |
| Limitations | not trained with randomized cube size: on upright cubes it loses ground to a different size or friction (84 % vs scripted 92 %); seed 0 knocks over and times out on cubes far more often than seeds 1–2; misses are mostly fingers that never close on a lying cube |

## Stand a held cube upright beside towers: round 11 (`standup_round11_seed{0,1,2}`) (Milestone 10, superseded by v3.1)

| | |
|---|---|
| Task | from the carry posture, turn a lying cube held in the hand upright and set it on a spot, with a tower of 1–4 cubes 20–26 cm away; touching the tower fails (`StandUpV3Env`, `standup`; hand motion `cartesian-v1-6d`, tilt up to 100°) |
| Policy | commanded: it observes the direction its hand should turn to (one of 16), the nearest tower, the clearance from it and the direction away from it. In use, a turn chooser (`turn_chooser_best.pt`, 5 networks trained on the policy's own outcomes) sets the command; a safety shield slides the arm along obstacles; after 20 blocked steps it backs off and re-plans (up to 4 times) |
| Training | 11 rounds; round 11: 600k steps warm-started from round 10, learning rate 3e-4 → 3e-5, with the shield (and its cut penalty) and re-planning in training |
| Held-out result | **87.7 %** on the one-shot set (89.5 / 90.5 / 83.0; target 90 %), scripted 93.0 % (p = 0.001); development set 91.5 %. Robustness: mass 90.3 %, friction 90.0 %, size 77.3 % (scripted 92.5 %), combined 71.5 % |
| In the work-order cell | runs on the cell's scene through the spot's station frame: 81.8 / 95.7 / 91.5 % of stand-ups per seed on the calibration orders |
| Limitations | most failures are freezes in front of a tower; not trained with randomized cube size; seed 0's stand-ups time out more in the cell, where the cube comes from the learned grasp rather than the trained start |

## Grasp a cube upright or lying on its side: v3.1 round 1 (`grasp_anypose6_seed{0,1,2}`) (Milestone 11), replaced in the cell by `grasp_anypose9` from v3.2

| | |
|---|---|
| Task | as round 5 (`GraspV4Env`, `grasp_v4`, hand motion `cartesian-v1-4d`, the cell's gripper), now with cube size randomized ×0.85–1.15 at constant density (not observed) |
| Training | 1M steps warm-started from `grasp_anypose5` (best), learning rate 3e-4 → 3e-5; 40 % of cubes near the base (x 0.30–0.40 m, was 30 %); otherwise as round 5 (`configs/grasp_anypose6_cartesian{,_ppo}.yaml`) |
| Files (sha256 prefix) | seed 0 `best_model.zip` `897b90bf9e2a`, `best_vecnormalize.pkl` `c70cdec7c2e8`; seed 1 `d17f8a091596`, `754b19539a97`; seed 2 `4cae67a2bc88`, `10762481f437` (locked at 13f690c) |
| Held-out result | **97.8 %** (96.3–98.7) on the one-shot lying set `grasp_lying_final_v2` (95.5 / 99.5 / 98.5; target 85 %); size varied 96.7 % (bar met, −1.2). Paired on the same cases: `anypose5` 95.5 % (p = 0.034), `anypose4` 87.7 %, scripted 84.5 %. Development: lying 97.0–97.8 % (v3.0 92.7–94.8 %), upright 95.0 % (v3.0 95.2 %) |
| In the work-order cell | `disturbance_suite_v2`, rescue every cube, no spares: 91.1 / 99.5 / 99.5 % of 2384 / 2378 / 2403 picks per seed (v3.0's `anypose5` on the same orders: 97.4 / 99.2 / 99.8 %). Seed 0 knocks over 32 of 1590 belt picks and 35 of 224 upright floor picks; seeds 1 and 2 knock over 2 picks each |
| Limitations | **seed 0, in the cell:** upright cubes turned 5–15° clockwise on the belt, or 35–45° on the floor, get knocked over (a finger lands on the top edge while the gripper is open, or the hand turns too little and never closes); v3.0's seed 0 had the same floor weakness on the other side of 45°. Upright cubes smaller than about ×0.95: the hand rests more than 1 cm below the centre and the fixed closing window never fires (seed 2: 62 % on these cubes). Seed 0 also knocks over the largest upright cubes (79.5 %). See [recovery.md](recovery.md#seed-0s-grasp-in-the-cell-what-the-skill-tests-missed) and [v3_2_candidates.md](v3_2_candidates.md) |

## Stand a held cube upright beside towers: v3.1 round 12 (`standup_round12_seed{0,1,2}`) (Milestone 11), recommended

| | |
|---|---|
| Task | as round 11 (`StandUpV3Env`, `standup`, hand motion `cartesian-v1-6d`), now with cube size randomized ×0.85–1.15 at constant density, and 30 % of training episodes starting from the grips the v3.1 grasp really hands over (`data/handoff/standup_handoffs_grasp_anypose6_seed{0,1,2}.npz`: 193 / 195 / 200 recorded hand-offs) |
| Policy | commanded, as round 11; each run has its own turn chooser (`turn_chooser_best.pt`, 5 networks, 600 of the run's starts, `scripts/train_turn_chooser.py --ensemble 5`): on held-out starts 94.2 / 95.8 / 95.0 % (a random reachable command: 93.4 / 96.5 / 89.9 %); shield and re-planning as round 11 |
| Training | 600k steps warm-started from round 11 (best), learning rate 3e-4 → 3e-5 (`configs/standup_v3_r12{,_ppo}.yaml`) |
| Files (sha256 prefix: model, normalizer, chooser) | seed 0 `e5c25dc9f7ce`, `43ef8fdd1d99`, `1476c38bc477`; seed 1 `62eae05022ea`, `57545a722b2e`, `220d2079f565`; seed 2 `ce418cfd1874`, `bfb6f230dd22`, `2461153519d2` (locked at 13f690c) |
| Held-out result | **90.0 %** (87.3–92.2) on the one-shot set `standup_final_v2` (94.5 / 90.5 / 85.0; target 90 %); size varied 88.3 % (bar met, −1.7; round 11: 76.8 %). Paired: round 11 88.2 % (p = 0.2), scripted 90.5 % (p = 0.81). Development: `standup_final_v1` 92.8 % (round 11 87.7 %, p = 0.0001) |
| In the work-order cell | `disturbance_suite_v2`, rescue every cube, no spares: 89.0 / 94.9 / 90.6 % of 209 / 177 / 191 stand-ups per seed (round 11 on the same orders: 83.5 / 95.2 / 91.4 %) |
| Limitations | most failures are still freezes where the cube is never released (48 of 60 one-shot failures); seed 2 is the weakest on the one-shot set (85.0 %); the chooser beats a random reachable command only for seed 2 |

## Grasp a cube upright or lying on its side: v3.2 round 3 (`grasp_anypose9_seed{0,1,2}`), the cell's grasp from v3.2 (adopted post hoc; v3.2 grasp target not met)

| | |
|---|---|
| Task | as v3.1 (`GraspV4Env`, `grasp_v4`, observation `grasp-v4-41d`, hand motion `cartesian-v1-4d`, the cell's gripper, cube size ×0.85–1.15 at constant density, not observed), now with the cell's own pick situations in training: 12 % of episodes stand an upright cube near the base (x 0.30–0.40 m) at any rotation, half of them turned 35–45° either way; 8 % stand it at belt height where the infeed delivers it, turned up to ±15°; 15 % of cubes are large (×1.05–1.15); a penalty for each step a finger touches the cube's top while the gripper is open, and for each step the open hand is low beside the cube (`configs/grasp_anypose8_cartesian.yaml`, v3.2 round 2's task, unchanged) |
| Training | 1M steps warm-started from `grasp_anypose6` (best), learning rate 1.5e-4 → 1.5e-5 (half of v3.1's); validation keeps the cell-like starts (12 % near-base upright, 8 % belt, 80 % whole table), 400 episodes every 100k steps (`configs/grasp_anypose9_cartesian_ppo.yaml`) |
| Checkpoint | each seed's best by that mixed validation (`best_model.zip`), the rule fixed before training: seed 0 at 900k, seed 1 at 1M, seed 2 at 900k steps (validation 98.0 / 98.3 / 99.0 %) |
| Files (sha256 prefix) | seed 0 `best_model.zip` `36fac25ed6fd`, `best_vecnormalize.pkl` `4aae3f914938`; seed 1 `dc8a1e251e15`, `1b7d615d586c`; seed 2 `729c236585b6`, `762c4d0d5b9b` (trained at 4d960a6; adopted at b5e9368, not locked) |
| Held-out result | one-shot cell-like pick sets, run once (per seed 0 / 1 / 2, v3.1's `anypose6` on the same cases): belt `grasp_belt_final_v1` **100 / 100 / 100 %**, none knocked over (v3.1 97.5 / 100 / 100 %, 5 knocked over; p = 0.063; bar met); floor `grasp_floor_final_v1` **94.5** / 100 / 100 %, seed 0's −45..−35° bin 14/23 (v3.1 89.5 / 100 / 100 %, 12/23; p = 0.021; **bar missed** on seed 0); floor with size varied 94.0 / 100 / 100 %, gap −0.2 (v3.1 81.5 / 99.5 / 99.0 %; p = 4e-6; bar met). Lying `grasp_lying_final_v2`: **98.3 %** (97.0–99.1; 98.0 / 98.5 / 98.5), v3.1 97.8 % (p = 0.65; no regression, bar met). The scripted pick: 100 % on belt and floor |
| Development | belt `grasp_belt_v1` 100 / 100 / 100 %, none knocked over (v3.1 98.0 / 100 / 100 %, 4); floor `grasp_floor_v1` 94.5 / 100 / 100 %, seed 0's −45..−35° bin 9/18 (v3.1 94.0 %, 14/18; p = 1); floor with size varied 94.0 / 99.5 / 100 % (v3.1 84.5 / 100 / 99.0 %; p = 0.0005); lying `grasp_lying_final_v1` 97.7 % (v3.1 97.8 %, p = 1); upright `grasp_upright_v1` 94.8 % (v3.1 95.0 %, p = 1) |
| Status | **the cell's grasp from v3.2 on, by a user decision made after seeing the development results (post hoc). It is an adoption, not a locked v3.2 grasp: the v3.2 grasp target is not met** (seed 0 misses the floor bar on the development and one-shot sets; round 3 was the last allowed round). See [v3_2_targets.md](v3_2_targets.md#decision-2026-11-20-user-the-cell-uses-round-3s-grasp) |
| In the work-order cell | held-out `disturbance_suite_v3`, run once from `50afb4c` (v3.1's place, stand-up and clear-rule recovery): **91.5 %** of all orders against 85.7 % with `grasp_anypose6` (47 / 12, p = 5e-6; seed 0 89.5 against 71.0 %, seeds 1–2 level); pick knock-overs per seed 3 / 2 / 3 against 50 / 2 / 3; planned no-knock orders 99.3 %. Without spares: recoverable D1 / D2 / D3 96.2 / 95.8 / 92.3 %, no-knock 98.7 % (bar 99 %, missed by one order) ([v3_2_targets.md](v3_2_targets.md#the-final-cell-run-as-it-came-2026-11-22)) |
| Limitations | **seed 0, standing cubes near the base turned close to −45°**, where either finger pair could close: on the one-shot floor set all 11 of its failures are cubes 0.31–0.38 m from the base axis turned −35 to −43° (1 knocked over); on the development set 9 of its 11 are turned −38 to −45° (the other two +25° and +42°; 3 knocked over). Most are picks that never complete. v3.1's seed 0 had the same weakness, and the scripted pick does better on these cases (0 / 11 on the one-shot floor set, p = 0.001). The v3.1 stand-up (`standup_round12`) was trained on hand-offs from `grasp_anypose6`, so in the cell it now follows a grasp it was not trained after. The closing window for upright cubes is unchanged (candidate F1 was not applied) |

## Grasp: v3.2 rounds 1 and 2 (`grasp_anypose7_seed{0,1,2}`, `grasp_anypose8_seed{0,1,2}`), not adopted

Both warm-started from `grasp_anypose6`, 1M steps, on the development sets only ([v3_2_targets.md](v3_2_targets.md#grasp-round-1-as-it-came-2026-11-1718-development-sets)). Round 1 (`configs/grasp_anypose7_cartesian.yaml`) fixed the belt but moved seed 0's floor weakness to the other side of 45°, and lying cubes regressed (94.2 against 97.8 %, p = 0.001). Round 2 (`configs/grasp_anypose8_cartesian.yaml`, the task round 3 kept) broke seed 0 on the belt at its best checkpoint (77.0 %); its final checkpoints met every pick bar, but lying cubes regressed again (93.5 %, p = 0.0002). Published because their results are reported; do not use.

## Rescue or replace: recovery planner (`recovery_planner_seed{0,1,2}`) (Milestone 10)

| | |
|---|---|
| Task | after a knock, for each fallen cube that could be rescued while a spare is left: rescue it or call a spare up the belt; after a failed rescue: retry or replace. 2 spares per order; an order counts only if complete within 1.5 × its undisturbed time ([recovery.md](recovery.md#the-learned-recovery-planner-f)) |
| Policy | PPO, 64 × 64, 13 observations (`SymbolicCell.features`), no observation normalization; `model.zip` and `meta.json` (git commit, task, calibration odds and action times) |
| Training | 500k decisions in the symbolic cell (`supervisor/symbolic.py`), γ = 1, 8 envs, about 10 min each |
| Held-out result | symbolic cell: **94.3 %** (94.2 / 94.5 / 94.3) of 10,000 held-out games each, against the best hand-written rule 92.5 % (p < 1e-29 per seed); physics cell, `disturbance_suite_v1`: **86.2 %** of 470 knocked orders, against the best rule 88.3 % (p = 0.08) |
| Limitations | trained in a symbolic cell that does not model a cube left for a spare as clutter; in physics those cubes get in the way of later placements, and the planner leaves more of them than it should. Use `symbolic.BEST_RULE` in physics. On `disturbance_suite_v2` with the v3.1 skills (v3.1 final run): 84.3 % of 465 knocked orders, against the best rule 86.5 % and the v3.1 clear rule 86.0 %. Ablations (`recovery_planner_blind-situation_*`, `_blind-clock_*`, `_perfect_*`) are published for the comparison |

## Pilots and failed attempts

| Run | Outcome |
|---|---|
| `reach_ppo_pilot_seed0` | task v1 (episodes ended on success): reached all targets, held only 14.5 % |
| `reach_sac_pilot_seed0` | first SAC pilot, 100k steps |
| `grasp_ppo_pilot_seed0` | no curriculum: learned to close the fingers, never to lift (0 %) |
| `transport_ppo_pilot_seed0`, `transport_ppo_v2_seed0` | plain PPO transport: 0 %, sank the cube onto the table, then lost grip |
| `place_stack_v3gate1cm_seed*`, `place_stack_v3check1cm_seed*`, `place_stack_v3centring_seed*` | three stacking v3 attempts stopped at 0.3–0.5M steps with about 0 % validation (an unbounded centring cost). Published without checkpoints or metadata: their configs and validation history only |

The failures are documented in the task pages and kept so every reported number can be reproduced.
