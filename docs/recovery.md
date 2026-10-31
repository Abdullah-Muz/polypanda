# Recovering from knocks (Milestone 10, v3.0)

In v2 the cell built its towers blind: once a cube was placed, the controller assumed it stayed there. v3 closes the loop. The cell reads the towers as they physically stand, notices when one is knocked, and rebuilds it. Recovery needs two skills the robot did not have: grasping a cube that lies on its side, and standing it upright beside the other towers. Both are learned with PPO. On top of them, a learned planner decides, for each fallen cube, whether to rescue it or to replace it with a spare.

Every target was committed before the work it measures ([v3_targets.md](v3_targets.md)), including two dated amendments made when measurements showed a target could not mean what it said. Every result is reported as it came, misses included.

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

**The closed loop (b, c).** After each placement and after a knock, `supervisor/observe.py` reads the cell as it physically is: which cubes stand on each tower (upright either way up, at their level, on the fixture), which stand in buffer slots, and which are loose, with how each one rests. The planner's plan is brought in line with the towers as they stand, and loose cubes are recovered one at a time (`ScriptedCell.recover_loose`):
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
```
