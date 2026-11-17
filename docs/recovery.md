# Recovering from knocks (Milestones 10–11, v3.0 and v3.1)

In v2 the cell built its towers blind: once a cube was placed, the controller assumed it stayed there. v3 closes the loop. The cell reads the towers as they physically stand, notices when one is knocked, and rebuilds it. Recovery needs two skills the robot did not have: grasping a cube that lies on its side, and standing it upright beside the other towers. Both are learned with PPO. On top of them, a learned planner decides, for each fallen cube, whether to rescue it or to replace it with a spare.

Every target was committed before the work it measures ([v3_targets.md](v3_targets.md)), including two dated amendments made when measurements showed a target could not mean what it said. Every result is reported as it came, misses included.

*v3.1 (Milestone 11) retrained both new skills, which now meet their targets, and replaced the planner with a hand-written rule that clears clutter, which did not beat the best rule (86.0 % against 86.5 % of knocked orders, p = 0.84). It has its own section: [v3.1](#v31-milestone-11-new-skills-and-clearing-clutter). The sections before it describe v3.0.*

**Headline:**
- **The cell recovers from knocks with learned skills only.** On a held-out suite of 200 orders with one knock each (3 seeds), the learned cell completes 93.2 % of orders after a single cube falls, 87.5 % after part of a tower falls and 91.5 % after a whole tower falls, counting the knocks a robot can physically recover from. Targets: 90 / 85 / 80 %. That is level with the scripted cell on the same orders (77.5 % against 78.5 % of all orders, p = 0.58).
- **Two new skills**, each just short of its own target on a one-shot test set: grasping a lying cube 84.8 % (target 85 %), standing a cube upright beside towers 87.7 % (target 90 %).
- **A learned recovery planner** finds a time-aware strategy that beats every hand-written rule in a calibrated symbolic cell (94.3 % against 92.5 %, p < 1e-29). In the physics cell it does not (86.2 % against 88.3 %, p = 0.08), and the reason was found: a cube left for a spare stays where it fell, often beside a tower, and gets in the way. The symbolic cell does not model it.

| Step | What | Result |
|---|---|---|
| a | Batch choice: take any cube on the belt | learned cell 600/600 orders at 3 buffer slots |
| b | Closed loop, knocks and the physical ceiling | at most 63–73 % of knocks can be recovered by any robot |
| c | Scripted recovery (baseline) | recoverable knocks: D1 96 %, D2 8/8, D3 87 % |
| d1 | Learned grasp of a lying cube | 84.8 % one-shot (target 85 %) |
| d2 | Learned stand-up beside towers | 87.7 % one-shot (target 90 %) |
| d+ | Recovery with the learned skills | every recovery target met |
| e | Skill odds by situation; symbolic cell | 18,000 attempts; the symbolic cell matches physics (93.2 % = 93.2 %) |
| f | Learned recovery planner | beats the best rule in the symbolic cell |
| g | The planner in the physics cell | does not beat the best rule; a sim-to-physics gap, explained |

## Knocks, and what can be recovered at all (step b)

**A knock** (`simulation/disturbances.py`) hits a tower right after it reaches a given height:
- a shove on the top or a middle cube, 0.6–1.5 m/s in a random direction; or
- a steady push of 1.0–2.5 N on the top cube, as a hand would push, stopping once the cube has moved 3 cm.

It is classified after everything settles: **D1** one cube left the tower, **D2** some, **D3** all of them. A knock is **recoverable** when every fallen cube rests flat (upright or lying on a side, not leaning on anything), within the arm's reach (0.30–0.78 m from the base axis, inside its base rotation range), and not on the belt or in the reject bin.

**The physical ceiling.** Before any recovery was built, towers of 2–4 cubes were knocked with no robot involved, and every fallen cube's resting place was classified (`scripts/disturbance_ceiling.py`, `results/disturbances/ceiling.json`). Even a perfect robot could recover every fallen cube after only about **68 % of D1, 73 % of D2 and 63 % of D3 knocks**. Single cubes are thrown out of reach; cubes of a toppled tower stay close but land leaning on each other. A low rim around each tower made it worse. The recovery targets were amended accordingly, before any recovery code: they count the recoverable knocks, and everything else is still run and reported.

**The disturbance suite** (`eval_sets/disturbance_suite_v1.json`, committed before any recovery existed): the 200 held-out Layer 2 orders, each with one knock (158) or none (42 controls, D0), from its own random stream. Without recovery, the v2 controller completes 44/200: every knock that throws a cube off fails the order.

## The closed loop and scripted recovery (steps a–c)

**Batch choice (a).** The robot may take any of the three cubes on the belt, not only the front one. The scripted cell then completes 106 / 170 / 200 of 200 orders at 0 / 1 / 3 buffer slots, exactly what the symbolic routing predicts, and the learned cell 600/600 at 3 slots (against 595/600 front cube only).

**The closed loop (b, c).** After a knock and at the end of every order, `supervisor/observe.py` reads the cell as it physically is: which cubes stand on each tower (upright either way up, at their level, on the fixture), which stand in buffer slots, and which are loose, with how each one rests. The planner's plan is brought in line with the towers as they stand, and loose cubes are recovered one at a time (`ScriptedCell.recover_loose`):
- cubes lying on a side, or within 20 cm of a tower, are first moved to a staging spot on an inner ring, so the hand never works down beside a rebuilt tower;
- a lying cube is stood upright there (grasped across its width, carried over the spot, turned, released);
- then each cube is picked and routed like a new arrival, with the cubes a tower needs next first;
- two attempts per cube; cubes that cannot be recovered are written off.

**Scripted recovery, held-out suite:** 157/200 orders (no recovery: 44/200). Of the recoverable knocks: **D1 76/79 = 96 %, D2 8/8, D3 26/30 = 87 %**, already above the targets with scripted skills. This is the baseline the learned skills are compared with.

## Two new learned skills (step d)

### Grasping a cube lying on its side (d1)

`GraspV4Env` (`grasp_v4`): the hand-motion grasp of Milestone 8 (4-D actions: x, y, z, wrist yaw), now for cubes upright or lying on a side anywhere in the cell's reach (0.30–0.78 m), with randomized mass and friction. Warm-started from the upright grasp. Four rounds of training; the two decisive fixes were root causes outside the algorithm:
- **Observation normalization.** The warm start inherited its observation statistics, in which a lying cube read 10.9 standard deviations out and was clipped. `renormalize_inputs` rewrites the first layer so the policy computes exactly the same function on renormalized inputs (tested to 1e-4).
- **The gripper.** In the original 2 N Menagerie gripper, a lying cube held across its width swings in the fingers when lifted, and the grasp check never confirms it. The cell's stiffer gripper with torsional friction holds it (22/25 lifts against 3/25). A scripted controller through the same 4-D actions confirmed the task was feasible (30/30 lying, 30/30 upright).

| Held-out set (3 seeds, 600 cases) | Learned (PPO) | Scripted (same actions) |
|---|---|---|
| **One-shot lying set** (`grasp_lying_final_v1`) | **84.8 %** (79.0 / 84.0 / 91.5); target 85 %, one cube short | 84.0 % (p = 0.74) |
| Lying, development set | 86.2 % | 84.0 % |
| Upright, the same region | 92.2 % | 89.5 % |

Robustness: on lying cubes the learned grasp holds 87–88 % under every perturbation (mass, friction, size, joint offsets, combined). On upright cubes it loses ground under friction and size changes; cube size was never randomized in training, against what the targets asked.

For recovery in the cell (d+), a fifth round added practice near the robot's base, where the cell's staging ring stands cubes, and more lying cubes: lying 92.7 % against 86.2 % on the development set (p = 8e-5). Its reported one-shot result stays round 4's 84.8 %.

### Standing a cube upright beside towers (d2)

`StandUpEnv` (`standup`): the arm holds a lying cube at the carry height over a spot; 6-D hand actions (position and a rotation of up to 100°) must turn it upright and set it down, with a tower of 1–4 cubes 20–26 cm away in most episodes. Touching the tower fails the episode. Success is the placed check: within 3 cm of the spot, tilted under 10°.

It took 11 rounds. Early policies learned one turn direction each and hit towers behind the wrist, or froze in front of them. What made it work:
- **A commanded turn** (`StandUpV3Env`): the policy observes a command, the direction its hand should turn to, and learns to turn every reachable way.
- **A learned turn chooser:** an ensemble of 5 small networks trained on the policy's own outcomes predicts, from the start state, which of 16 turn directions will succeed; the policy is commanded the best one.
- **A safety shield** that slides the arm along obstacles: an action that would move the arm into a tower loses only its component toward the tower, and the policy pays for every cut. Tower hits on the development set went from 57 to 0 (a handful remain on the one-shot set; most failures there are freezes).
- **Re-planning:** after 20 fully blocked steps, the arm backs off and switches to the chooser's next-best direction (up to 4 times).
- The policy also observes the tower: its position, the clearance from the arm, and the direction away from it.

| Held-out set (3 seeds, 600 starts) | Learned (PPO + chooser + shield + re-planning) | Scripted (16-direction planner) |
|---|---|---|
| **One-shot set** (`standup_final_v1`) | **87.7 %** (89.5 / 90.5 / 83.0); target 90 % | 93.0 % (p = 0.001) |
| Development set (round 11) | 91.5 % | 93.0 % (p = 0.37) |

The drop from 91.5 to 87.7 % is what selecting over 11 rounds on one development set costs. Robustness: mass 90.3 %, friction 90.0 %, cube size 77.3 % (scripted 92.5 %; size never randomized), all combined 71.5 %.

## Recovery with the learned skills (d+)

In the cell, the stand-up runs on the cell's own scene through the spot's station frame, sees the nearest real tower, and its shield checks every obstacle in the cell. The configuration was chosen on a separate development suite (50 other orders) and locked in a commit before the held-out suite was run once (`959d2f2`).

| Knocks (recoverable only) | Target | **Learned skills** (3 seeds, `bf8ad83`) | Scripted skills |
|---|---|---|---|
| D1, one cube | ≥ 90 % | **93.2 %** (218/234) | 96.2 % (76/79) |
| D2, part of a tower | ≥ 85 % | **87.5 %** (28/32) | 8/8 |
| D3, the whole tower | ≥ 80 % | **91.5 %** (75/82) | 86.7 % (26/30) |
| No knock (D0) | | 126/126 | 42/42 |
| All 200 orders, every knock | | 77.5 % | 78.5 % (paired p = 0.58) |

Every recovery target is met, and none is significantly worse than scripted recovery on the same orders (p = 0.6 / 1 / 0.24). The seeds differ: seeds 1 and 2 recover 97–100 % of recoverable knocks, seed 0 73–81 %, because its grasp knocks cubes over and its stand-up times out.

## Skill odds and a symbolic cell (step e)

The planner needs to know what its choices cost: how likely each skill is to succeed in each situation, and how long it takes. A calibration run (200 other orders, 90 % of them knocked, `eval_sets/disturbance_calib_v1.json`, all learned skills, 3 seeds) logged every skill attempt with its situation: where the cube came from, how it rested, where it was, how near the closest stack was, the loose cubes before and after, and its start and end times (`<run>_skills.csv`, about 18,000 attempts, `fcefde0`).

| Situation | Success |
|---|---|
| Pick from the belt | about 100 % (5031) |
| Pick from a buffer slot | 99.9 % (797) |
| Place on a tower, levels 1–4 | 99.3–99.9 % |
| Pick an upright fallen cube | 91.7 % (768) |
| Grasp a lying fallen cube | 93.4 % (682) |
| Stand it upright | 88.6 % (638) |

Each action's time comes from a least-squares fit of every order's duration against the actions it took (error 9 s on 214 s orders): a belt pick 16 s, a pick from the floor 8.5 s, a stand-up 20 s, a place 2–13 s.

**The symbolic cell** (`supervisor/symbolic.py`) plays an order as the physics cell does, with the same greedy routing and the same recovery rules, but every skill is a dice roll at the measured odds and costs its fitted time. Knocks leave their cubes where calibration knocks left them. It plays an order in well under a millisecond. **It reproduces the physics cell:** recoverable knocks completed 93.2 % in both, unrecoverable 15.8 % against 12.9 %, undisturbed orders 98.0 % against 100 % (`results/planner/gate.json`).

## Is there anything for a learned planner to decide? (the headroom gates)

The original v3 plan put a learned planner at the top: it would route cubes and run recovery, and beat the hand-written rules. Twice, before training, a search for the best possible decisions showed how little room there was.

1. **Routing (step 1, before any v3 code).** Even a planner that knew every future arrival completes at most 3–9 points more orders than the greedy rule, and a simple batch rule gets within 3–6.5 points of it. Routine routing is nearly solved by rules.
2. **Recovery.** After d+, scripted recovery with the learned skills already completes 92 % of the recoverable knocks, and 26 % of knocks are not recoverable by anyone. The task got a limited stock of **spare cubes** (a dated amendment, before any planner code): for each fallen cube, rescue it or call a spare of its colour up the belt. In the symbolic cell, on the calibration orders:
   - with 1 or 2 spares, the best of 32 hand-written rules completes 97.0–97.5 % of knocked orders, and search over every rescue-or-replace plan does no better;
   - with a deadline (an order counts only if done within 1.25–2 × its undisturbed time), search beats the best rule by at most 2.3 points.

The gate asked for 5 points. It failed: rules are near-optimal for these decisions. The planner was trained anyway, with the target left where it was and expected to be missed.

## The learned recovery planner (f)

**The task** (fixed before training): 2 spares per order, a spare rides the belt like any arrival, and an order counts only if it is complete within 1.5 × its undisturbed time.

**The environment** (`environments/recovery_planner.py`): one episode is one order in the symbolic cell, and each step is one decision. After a knock, for each fallen cube that could be rescued while a spare is left: rescue it, or leave it and call a spare; after a failed rescue attempt: retry, or replace. The policy sees 13 numbers: the cube's pose and whether it lies near a tower, whether this is a retry, spares left, time used and time left against the deadline, arrivals still to come, cubes still needed, the fallen cubes by pose, and whether a tower needs this colour next. Reward 1 for an order completed by its deadline. PPO with no discount (γ = 1), 3 seeds, 500,000 decisions each. Every action offered is valid, so no action masking is needed.

**In the symbolic cell** (2000 held-out orders × 5, the same random streams for every policy, `results/planner/symbolic_eval.json`):

| Policy | Orders completed by the deadline |
|---|---|
| Rescue every cube it can, replace the rest | 89.0 % |
| The best of 32 hand-written rules (chosen on separate orders) | 92.5 % |
| **Learned planner** (3 seeds) | **94.3 %** (94.2 / 94.5 / 94.3); each seed above the best rule, p < 1e-29 |
| Ablation: blind to the cube's situation | 94.2 % |
| Ablation: blind to the clock | 93.7 % |
| Ablation: trained as if every skill always succeeded | 94.0 % |

The learned planner captures about all of the headroom search found (+1.9 points), and the ablations say where it comes from: **the clock.** It replaces cubes when the order runs late and rescues them when there is time; a rule keyed on the situation cannot do that.

## The planner in the physics cell (g): a sim-to-physics gap

Run once on the held-out suite with spares and the deadline (all learned skills, 3 seeds, configuration locked at `29add44`, results `c52560f`):

| Recovery planner | Knocked orders completed (470) | Late orders |
|---|---|---|
| No spares, no deadline (d+, for reference) | 71.3 % | |
| Rescue every cube | 84.3 % | 63 |
| **The best hand-written rule** | **88.3 %** | 25 |
| **Learned planner** | **86.2 %** (83.3 / 87.3 / 87.9) | 30 |

- **The learned planner does not beat the best rule:** 8 orders only it completed, 18 only the rule (p = 0.08). The A1 target is missed, as the gate predicted.
- **Spares and a good rule matter:** the best rule beats rescuing every cube (38 / 19, p = 0.016); the learned planner is ahead of it but not significantly (30 / 21, p = 0.26). Both cut late orders by more than half.

**Why the symbolic edge did not carry over.** Almost every skill failure under the best rule and the learned planner (28 and 29 orders, against 7 when every cube is rescued) is in an order where a cube was left for a spare, and they are mostly placements that topple or drop. A cube left for a spare stays where it fell, often right beside a tower, and is in the way of later placements there. The symbolic cell treats a written-off cube as gone. Its odds match physics on average, but this mechanism only appears when the policy changes what is left on the floor. The natural fixes, sweeping a written-off cube into the reject bin or calibrating placement odds on clutter, need a fresh held-out suite to test fairly, and are left for v3.1.

## What I learned

1. **Measure the physical ceiling before setting a target.** A third of knocks cannot be recovered by any robot. Targets over all knocks would have measured the dice, not the robot.
2. **Check for headroom before training a planner.** Twice, search showed that hand-written rules were near-optimal. That changed what the planner was for, and it was known before any training was spent on it.
3. **Root causes were in the setup, not the algorithm.** The lying-cube grasp failed for three rounds because of inherited observation statistics and a gripper too weak for the task. Neither was a reward or hyperparameter problem.
4. **Decompose a hard skill.** One policy that had to pick its own turn direction collapsed onto one direction. A commanded policy with a learned chooser, a sliding safety shield and re-planning stood cubes up beside towers and nearly stopped hitting them.
5. **A simulator that matches the averages can still miss a mechanism.** The symbolic cell reproduced every rate it was calibrated on, but not the clutter that the planner's own choices created.
6. **Selection on a development set costs points.** The stand-up went from 91.5 % on its development set, where 11 rounds were compared, to 87.7 % on its one-shot set.

## v3.1 (Milestone 11): new skills, and clearing clutter

v3.0 left three things short. Both new skills missed their one-shot targets by a little, neither was trained with a varied cube size, and the learned recovery planner lost to the best hand-written rule in physics, because cubes left for a spare got in the way. v3.1 is one more round on each, under the same rules. The targets were committed before the work ([v3_1_targets.md](v3_1_targets.md)), the one-shot sets and the cell's suite are new, every run uses 3 seeds, and every result is reported as it came.

**Headline:**
- **Both skills now meet their targets on fresh one-shot sets.**
  - Grasping a lying cube: **97.8 %** (target 85 %; v3.0 84.8 % on its own one-shot set).
  - Standing a cube upright beside towers: **90.0 %** (target 90 %; v3.0 87.7 %).
  - Each lost under 2 points when the cube size was varied.
- **The recovery target was missed.** The cell's recovery is a hand-written rule that clears clutter. It is the v3.0 best rule, except that a cube it would leave for a spare is taken to the reject bin when it lies within 6 cm of a tower that still needs cubes.
  - On the new held-out suite, the clear rule completed **86.0 %** of knocked orders, against **86.5 %** for the best rule (p = 0.84). The target asked for a significant gain.
  - The rule did remove the clutter: places with a loose cube within 6 cm of the tower fell from 243 to 19. But clearing costs time, and the orders it saved from a failed place were matched by orders that ran late.
- **The v3 recovery targets still hold for recoverable knocks.** Without spares, the cell completes D1 90.6 %, D2 89.2 % and D3 89.8 % (bars 90 / 85 / 80 %).
- **D0 missed its bar.** Orders with no knock completed 95.4–96.3 % (bar 99 %), every miss on seed 0. The v3.0 cell misses it on the same orders too (96.3 %).
- **A weakness found in the cell, not in the skill tests.** Seed 0's new grasp knocks over upright cubes:
  - on the belt: 32 of 1,590 picks;
  - on the floor: 35 of 224 picks.
  - Both were traced to one cause, the cube's rotation relative to the hand. On seeds 1 and 2, the v3.1 cell completes 93.5 % of all orders, and every order with no knock (72/72).

| Part | What | Result |
|---|---|---|
| Grasp, round 1 | size randomized, more practice near the base | one-shot **97.8 %** (target 85 %); size varied 96.7 % |
| Stand-up, round 12 | size randomized, starts from the new grasp's real hand-offs | one-shot **90.0 %** (target 90 %); size varied 88.3 % |
| Recovery planner | clutter-aware symbolic cell, then a hand-written clearing rule | the symbolic cell failed its check twice; no planner was trained |
| Final cell run | `disturbance_suite_v2`, 7 configurations | clear rule 86.0 % vs best rule 86.5 % of knocked orders (p = 0.84): **target missed**. Regression bars D1–D3 met, D0 missed |

### The grasp, round 1 (`grasp_anypose6`)

Round 1 continued training from v3.0's cell grasp (`grasp_anypose5`) for 1M steps, with the same reward and schedule (learning rate 3e-4 → 3e-5). Two settings changed:
- **Cube size randomized**, ×0.85–1.15 at constant density, as the v3 targets had asked. The policy is not told the size.
- **More practice near the base:** 40 % of training cubes at x 0.30–0.40 m, up from 30 %, aimed at v3.0 seed 0 knocking over upright cubes there.

**Development sets** (3 seeds, 600 cases each; seeds 0 / 1 / 2 in brackets; p from an exact paired McNemar test against v3.0's grasp on the same cases):

| Set | v3.1 (round 1) | v3.0 (`grasp_anypose5`) | p |
|---|---|---|---|
| Lying, `grasp_lying_v1` | **97.0 %** (93.0 / 98.0 / 100) | 92.7 % (90.5 / 93.5 / 94.0) | 0.0004 |
| Lying, `grasp_lying_final_v1` (v3.0's one-shot set) | **97.8 %** (95.5 / 99.0 / 99.0) | 94.8 % (94.0 / 93.5 / 97.0) | 0.008 |
| Upright, `grasp_upright_v1` | 95.0 % (91.5 / 95.5 / 98.0) | 95.2 % (91.5 / 97.0 / 97.0) | 1.0 |
| Upright near the base, x < 0.40 m (117 cases) | 94.0 % (87.2 / 100 / 94.9) | 88.9 % (74.4 / 97.4 / 94.9) | 0.11 |
| Lying, size ×0.85–1.15 | **97.0 %** (94.0 / 97.5 / 99.5) | 92.3 % (90.5 / 91.5 / 95.0) | 0.0001 |
| Upright, size ×0.85–1.15 | 87.8 % (85.5 / 95.0 / 83.0) | 91.3 % (86.5 / 95.5 / 92.0) | 0.02 |

- Lying cubes, the skill's job, improved clearly and lose nothing when the size changes (97.0 % both ways).
- Upright cubes of normal size did not change.
- Upright cubes of varied size got worse: 7.2 points below normal size, against 3.8 for v3.0, mostly on the smallest cubes.

**Why small upright cubes fail.**
- The gripper closes by a fixed rule, not by the policy's choice: the hand must be within 1 cm of the cube's centre, both sideways and in height.
- The policies come down until the palm rests on the cube's top. On a normal cube that leaves the hand just inside the window. On the smallest cubes it is more than 1 cm below the centre, and the hand waits there until time runs out.
- Below ×0.925, seed 2 succeeds 57 % of the time. v3.0's grasp has the same trap (73 % and 82 % on these cubes for seeds 0 and 2).

*What-if, evaluation only, not a result:* the window was widened to accept a hand up to 1.5 cm below the centre, as the rule for lying cubes already does. With no retraining, the 74 small upright cases then succeed 95.9 % for seed 2 (62.2 % as is) and 93.2 % for seed 0 (78.4 % as is). This is a candidate for v3.2.

**Decision:** round 1 met the target on development. Under the stopping rule it was locked (13f690c) and run once on the fresh one-shot set.

### The stand-up, round 12 (`standup_round12`)

Round 12 continued from v3.0's stand-up (round 11) for 600k steps, as round 11 was trained. Two settings changed:
- **Cube size randomized**, ×0.85–1.15 at constant density.
- **Starts from real hand-offs.** 30 % of training episodes hold the cube exactly as the new grasp hands it over in the cell, not only as v3.0's synthetic grip holds it. The hand-offs were recorded from `grasp_anypose6` after the grasp was locked: 193, 195 and 200 held grasps of 200 per seed. *Amended before training: the plan first used v3.0's grasp.*

**The turn chooser.** Each run gets its own, an ensemble of 5 small networks trained on 600 of the run's own starts.
- On held-out starts, it picks a direction that works 94.2 / 95.8 / 95.0 % of the time (seeds 0 / 1 / 2).
- Picking a random reachable direction does nearly as well (93.4 / 96.5 / 89.9 %). Only seed 2's chooser clearly beats random.

**Development sets** (3 seeds, 600 starts each, each run's chooser; p against round 11, paired):

| Set | v3.1 (round 12) | v3.0 (round 11) | p |
|---|---|---|---|
| `standup_v1` | **93.5 %** (96.0 / 95.0 / 89.5) | 91.5 % (94.5 / 90.5 / 89.5) | 0.11 |
| `standup_final_v1` (v3.0's one-shot set) | **92.8 %** (95.0 / 95.0 / 88.5) | 87.7 % (89.5 / 90.5 / 83.0) | 0.0001 |
| `standup_v1`, size ×0.85–1.15 | **89.5 %** (88.5 / 92.5 / 87.5) | 77.3 % (78.5 / 79.5 / 74.0) | 3e-12 |

- Varying the size now costs 4.0 points instead of 14.2.
- Most failures are still freezes, where the cube is never released (31 against 62 on `standup_final_v1`).

### One-shot results

Each skill was run once on its fresh one-shot set, with the locked configuration (6d50e22). Wilson 95 % intervals; p from exact McNemar tests paired on the same cases.

| Grasp a lying cube, `grasp_lying_final_v2` | Nominal (600) | Size ×0.85–1.15 | Paired p (nominal) |
|---|---|---|---|
| **v3.1 grasp (`grasp_anypose6`)** | **97.8 %** (96.3–98.7); 95.5 / 99.5 / 98.5 | 96.7 % (94.9–97.8); gap −1.2 | |
| v3.0 cell grasp (`grasp_anypose5`) | 95.5 % (93.5–96.9) | 95.7 % | 0.034 (26 only v3.1, 12 only v3.0) |
| v3.0 one-shot-tested grasp (`grasp_anypose4`) | 87.7 % | 87.2 % | 3e-13 |
| Scripted any-pose grasp (same actions, 200) | 84.5 % | 85.0 % | 5e-16 |

- Target 85 %: met.
- Size bar (within 5 points): met (−1.2).
- The remaining misses are fingers that never close (13 of 600).

| Stand a cube upright, `standup_final_v2` | Nominal (600) | Size ×0.85–1.15 | Paired p (nominal) |
|---|---|---|---|
| **v3.1 stand-up (round 12, its chooser)** | **90.0 %** (87.3–92.2); 94.5 / 90.5 / 85.0 | 88.3 % (85.5–90.7); gap −1.7 | |
| v3.0 stand-up (round 11) | 88.2 % (85.3–90.5) | 76.8 % (p = 1e-10 against v3.1) | 0.2 (36 only v3.1, 25 only v3.0) |
| Scripted 16-direction planner (200) | 90.5 % | 90.0 % | 0.81 |

- Target 90 %: met, exactly.
- Size bar: met (−1.7).
- The stand-up is now level with the scripted planner. In v3.0 it was 5 points behind (p = 0.001), and it no longer falls apart when the size changes.
- Seed 2 is the weakest (85.0 %).
- Most failures are still freezes (48 of 60).

### The recovery planner and clutter

The plan was to make the symbolic cell see clutter, then train a new planner in it that could also clear a cube into the reject bin.
- **New calibration run.** v3.0's calibration log held almost no clutter: its recovery rescued every reachable cube, so only 3 of its 5,066 tower places had a knocked cube within 6 cm of the tower. A new physics calibration run on the 200 calibration orders drew rescue, leave or clear at random for every decision, so the data covers the floors any planner can leave behind. The rate of failed places beside clutter was fitted from that run.
- **The check failed twice.** Before any planner training, the clutter-aware symbolic cell had to rank v3.0's three physics results on `disturbance_suite_v1` in the right order, each within 3 points: best rule 88.3 %, learned planner 86.2 %, rescue every cube 84.3 %.
  - First run: 85.7 / 86.4 / 86.6 %. Each was within 3 points, but in the wrong order.
  - The diagnosis found that the clutter part matched physics, and the error was in v3.0's time model, which gave every action a fixed time. One revision added each action's measured time spread.
  - Re-run: 85.5 / 86.2 / 86.3 %, wrong again.
  - The remaining cause is that in physics a failed skill is usually retried at a cost in time, while the symbolic cell ends the order. That revision is left for v3.2.
- **Decision.** Per the amendment written before the re-run, no planner was trained on the symbolic cell, and the user chose v3.1's recovery: **the clear rule**.
  - The clear rule is the v3.0 best rule, except that a cube it would leave for a spare is taken to the reject bin and then replaced by a spare, when it lies within 6 cm of a tower of the order that still needs cubes.
  - The 6 cm line comes from the clutter measurements. Under 6 cm, places on that tower fail about 1 in 5 (v3.0 suite_v1 runs) or about 1 in 8 (v3.1 calibration run). From 6 to 20 cm they barely do.
  - The rule was fixed in advance, not tuned in the symbolic cell or on any suite (`ClutterClearRule`; `evaluate_cell.py --recovery-planner clear-rule`).

### The final cell run (`disturbance_suite_v2`)

Every configuration ran the suite's 200 held-out orders once (`--seed 20261203 --n 200`). The learned configurations used 3 seeds, the scripted ones 1 run. The configuration was locked in 37230c4 before any run, and the results are in 285e445. Every run recorded the lock commit with a clean tree.

**The target:** the clear rule (1) against the best rule (2), with the same v3.1 skills, 2 spares and the 1.5× deadline. Knocked orders D1–D3 were pooled, with an exact McNemar test paired by seed and order (`scripts/pair_cell_runs.py`, `results/cell/v31_final_target_clear-rule_vs_best-rule.json`). Both runs labelled the same 465 orders as knocked, so no order had to be dropped.

| Knocked orders | Clear rule (v3.1 cell) | Best rule | Only clear / only best | p |
|---|---|---|---|---|
| **Pooled, 3 seeds (465)** | **86.0 %** (82.6–88.9) | **86.5 %** (83.0–89.3) | 11 / 13 | **0.84** |
| Seed 0 (150) | 74.0 % | 74.0 % | 4 / 4 | 1 |
| Seed 1 (157) | 93.0 % | 91.7 % | 5 / 3 | 0.73 |
| Seed 2 (158) | 90.5 % | 93.0 % | 2 / 6 | 0.29 |
| D1, one cube (329) | 94.2 % | 94.2 % | 0 / 0 | 1 |
| D2, part of a tower (53) | 86.8 % | 94.3 % | 0 / 4 | 0.13 |
| D3, the whole tower (83) | 53.0 % | 50.6 % | 11 / 9 | 0.82 |

**The target is missed.** The clear rule is not better than the best rule, and not significantly worse.

**What the clear rule did** (over the 465 knocked orders):

| | Clear rule | Best rule |
|---|---|---|
| Cubes cleared to the reject bin | 90 (86 landed in the bin) | 0 |
| Cubes rescued | 119 | 175 |
| Cubes left for a spare | 397 | 487 |
| Spares used | 581 | 581 |
| Tower places with a loose cube within 6 cm (all 600 orders) | 19 (14 succeeded) | 243 (194 succeeded) |
| Failed tower places (all 600 orders) | 71 | 119 |
| Knocked orders failed only by the deadline | 37 | 26 |
| Knocked orders ended by a failed place | 1 | 5 |

**Why it did not help.**
- The rule removed the mechanism it was written for. Places beside a loose cube almost disappeared, and failed places fell by 40 %.
- But a failed place is usually retried, so few orders were lost to clutter in the first place.
- Each clearing costs a pick and a drop, and that time showed up as late orders.
- Clutter was a real cost in v3.0's physics runs, but on this suite and with these skills it was too small to win orders back.

**The regression check** (v3's recovery targets, read from configuration 4: v3.1 skills, rescue every cube, no spares):

| Knocks (recoverable only) | Target | v3.1 skills (4) | v3.0 cell (5) | Scripted (6) |
|---|---|---|---|---|
| D1, one cube | ≥ 90 % | **90.6 %** (242/267) | 94.8 % (254/268) | 94.0 % (78/83) |
| D2, part of a tower | ≥ 85 % | **89.2 %** (33/37) | 79.5 % (35/44) | 76.5 % (13/17) |
| D3, the whole tower | ≥ 80 % | **89.8 %** (53/59) | 88.2 % (60/68) | 92.3 % (24/26) |
| No knock (D0), the suite's planned controls | ≥ 99 % | **95.4 %** (103/108); v3.1 cell (1): **96.3 %** (104/108) | 96.3 % (104/108) | 36/36 |

- **D1, D2 and D3 are met.** The v3.0 cell misses D2 on these orders.
- **D0 is missed.** It is read, as in v3.0, on the suite's 36 planned no-knock orders per seed. Every miss is on seed 0, in v3.1 and in v3.0 alike:
  - v3.1 cell: 4 of 4 began with the grasp knocking a cube over;
  - configuration 4: 4 of 5;
  - v3.0 cell: 2 of 4, the other 2 a tower out of tolerance.
- The cell's own D0 label is lower: 88.9 % for the v3.1 cell, 83.7 % for configuration 4. It also counts planned-knock orders that failed before their knock was due, and those are again seed 0's knock-overs.

**Side by side** (all 200 orders per seed):

| Configuration | All orders | No knock (planned) | Knocked D1–D3 | Late | Spares used |
|---|---|---|---|---|---|
| **1 v3.1 cell: clear rule, spares, deadline** | **87.0 %** (522/600) | 96.3 % | 86.0 % | 37 | 599 |
| 2 best rule (v3.1 skills, spares, deadline) | 87.2 % (523/600) | 95.4 % | 86.5 % | 26 | 599 |
| 3 v3.0 learned planner (v3.1 skills, spares, deadline) | 85.7 % (514/600) | 96.3 % | 84.3 % | 29 | 508 |
| 4 v3.1 skills, rescue every cube, no spares | 78.0 % (468/600) | 95.4 % | 75.4 % | | |
| 5 v3.0 cell (no spares) | 81.5 % (489/600) | 96.3 % | 77.5 % | | |
| 6 scripted cell, recovery (1 run) | 82.0 % (164/200) | 36/36 | 77.2 % | | |
| 7 scripted cell, no recovery (v2, 1 run) | 21.5 % (43/200) | 36/36 | 0.6 % | | |

- **Spares and a rule matter most.** With spares and the deadline, the v3.1 cell completes 87.0 % of all orders, against 78.0 % without spares and 82.0 % for the scripted cell. Without recovery, only 21.5 % are completed.
- **Seed by seed, all orders:**

  | Seed | Clear rule (1) | v3.1, no spares (4) | v3.0 cell (5) |
  |---|---|---|---|
  | 0 | 74.0 % | 62.0 % | 72.0 % |
  | 1 | 94.5 % | 86.0 % | 85.0 % |
  | 2 | 92.5 % | 86.0 % | 87.5 % |

  Like for like (4 against 5), the v3.1 skills trail the v3.0 cell only because of seed 0. Seeds 1 and 2 are level.
- **Seeds 1 and 2 only** (a side note; the results above use all 3 seeds): the v3.1 cell completes 93.5 % of all orders and 72/72 orders with no knock.

### Seed 0's grasp in the cell: what the skill tests missed

The one-shot set tests lying cubes in the grasp's own scene. In the cell, seed 0's new grasp knocks over cubes the other two seeds never touch:

| Pick (configuration 4, 200 orders) | v3.1 seed 0 | v3.1 seeds 1, 2 | v3.0 seed 0 |
|---|---|---|---|
| Upright cube on the belt | 32 / 1,590 knocked over (2.0 %) | 0 / 1,696 each | 0 / 1,688 |
| Upright cube on the floor, after a knock | 35 / 224 (16 %) | 0 / 205, 0 / 212 | 52 / 278 (19 %) |

Every failure was replayed in a fresh cell, with the contacts between the fingers and the cube recorded. The belt ones reproduced 31 of 32, the floor ones 35 of 35. The cause is one thing: **how far the cube is turned relative to the hand.**
- **Belt cubes** arrive turned up to ±15°.
  - 29 of the 32 knock-overs were cubes turned 5–15° clockwise.
  - At that rotation, the hand comes down 2.3–2.8 cm outward of the cube's centre, as it always does slightly. A finger lands on the cube's top corner while the gripper is still open, and the cube tips over at once.
  - Successful picks at the same rotation follow almost the same path, so this is a thin margin rather than a wrong strategy.
- **Floor cubes** near the base are mostly cubes the stand-up has just turned upright on the staging ring, 0.36 m from the base, now picked again.
  - 31 of the 35 knocked picks follow a successful stand-up at the same spot in the same order.
  - Those cubes can be at any rotation. Turned under 10°: 0 knocked over of 57. Turned 35–45°: 28 of 41.
  - It fails in two ways. In 22 cases the hand turns 8–11° too little, stops 1.4–1.9 cm off-centre, and never gets to close before the cube falls. In 13 cases a finger lands on the top edge on the way down, as on the belt.
- **v3.0's seed 0 had the same floor weakness, on the other side of 45°.** The same 228 standing-cube poses were run in the grasp's own scene. Of cubes turned +26..+45°, v3.0 seed 0 knocked over 18 of 48. Of cubes turned −45..−36°, v3.1 seed 0 knocked over 10 of 24. Round 1's training moved the weak side; it did not remove it. v3.1 seeds 1 and 2 knocked over 0 and 5 of 228.

The fixes are proposed for v3.2 in [v3_2_candidates.md](v3_2_candidates.md): a cell-like pick check per seed before any grasp is locked, plus training changes.

### What I learned (v3.1)

1. **Removing a mechanism is not the same as winning orders.** The clear rule did exactly what it was designed to do: clutter beside towers almost vanished and failed places fell by 40 %. But failed places were mostly retried anyway, and clearing cost time, so completed orders did not move. A fix should be judged on the outcome that counts before it is built. Here, that meant measuring how many orders the clutter actually cost.
2. **Train on what the task varies.** Randomizing the cube size cost nothing on lying cubes, and cut the stand-up's size loss from 14 to 4 points.
3. **A fixed closing rule is part of the policy's world.** The small-cube failures were a 1 cm window, not a learning problem. Widening it fixes most of them without retraining.
4. **Test a skill where it will run, seed by seed.** The one-shot grasp set was met easily. But the cell feeds cubes at rotations and heights that set does not cover, and one seed of three had a thin margin there. Pooled over seeds, it hid behind the other two until the cell run.
5. **Hand-offs between learned skills matter.** Training the stand-up on the grips the real grasp produces is part of why it no longer loses to the scripted planner.

## Reproduce

```bash
uv run python scripts/evaluate_cell.py --mixed --recover \
    --disturbances eval_sets/disturbance_suite_v1.json \
    --pick run:runs/grasp_anypose5_seed1 --place run:runs/place_stack_v3_seed1 \
    --standup run:runs/standup_round11_seed1                       # d+, one seed
uv run python scripts/planner_gate.py                              # e: the check and the gate
uv run python scripts/train_recovery_planner.py --seed 0           # f: about 10 minutes on CPU
uv run python scripts/evaluate_recovery_planner.py runs/recovery_planner_seed{0,1,2}
uv run python scripts/evaluate_cell.py ... --spares 2 --slack 1.5 \
    --recovery-planner run:runs/recovery_planner_seed1             # g, one seed
# v3.1: the final cell run, one seed (configurations 1 and 4; all 7 in docs/v3_1_targets.md)
uv run python scripts/evaluate_cell.py --mixed --slots 3 --aim fixture --recover \
    --disturbances eval_sets/disturbance_suite_v2.json --seed 20261203 --n 200 \
    --pick run:runs/grasp_anypose6_seed1 --place run:runs/place_stack_v3_seed1 \
    --standup run:runs/standup_round12_seed1 \
    --spares 2 --slack 1.5 --recovery-planner clear-rule              # 1; without the last line: 4
uv run python scripts/pair_cell_runs.py \
    --a results/cell/*anypose6*spares2-clear-rule*_episodes.csv \
    --b results/cell/*anypose6*spares2-best-rule*_episodes.csv     # the target: 1 against 2
```
