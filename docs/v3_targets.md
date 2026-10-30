# v3 targets (Milestone 10), set before any v3 training

v3 makes the work-order cell hierarchical RL and self-correcting: a learned planner decides where every cube goes, it sees the real towers, and the robot recovers from dropped cubes and knocked-over towers. This page fixes, before any v3 policy is trained, what counts as success. Results will be reported against it whatever they are, including targets that are missed.

The rules are the project's usual ones: held-out evaluation sets that training never sees, 3 training seeds per learned component, 95 % Wilson intervals, exact McNemar tests on paired orders, and a scripted baseline for every learned component.

## Evaluation sets

- **Routing:** the same 200 held-out Layer 2 orders as v2 (`CellEvalConfig`, seed 20261026), at 0, 1 and 3 buffer slots.
- **Disturbance suite** (new, held out, its own random seed): the same kind of orders, each with one injected disturbance, labelled by type:
  - **D0** no disturbance (control);
  - **D1** a single cube knocked off a tower or dropped near a station;
  - **D2** a tower partly knocked over (its top cubes fall);
  - **D3** a tower knocked over completely.

  Disturbances vary in strength and timing (early or late in the build) and in which tower is hit. The suite is generated and committed before recovery is trained or tuned.

## A1: learned planner

*Amended on 2026-10-26, before any v3 training or v3 code. The original A1 targets asked the learned planner to beat the greedy planner at 0 or 1 buffer slots. An analysis of the routing decisions on the 200 orders, with no learning involved, showed this cannot be a meaningful RL result. Even a planner that knew every future arrival could complete at most 3–9 points more orders than greedy. Letting the robot pick any of the 3 cubes on the belt (batch choice) completes far more orders without a buffer (53 % instead of 32 %), but a simple hand-written rule already gets that gain, within 3–6.5 points of perfect planning, and within about 3 s of the fastest possible order time. Routine routing is nearly solved by simple rules in this cell. The decisions that are hard for rules, with uncertain outcomes, come with recovery, so the learned planner runs the whole cell, recovery included, and its targets are set against the best hand-written rules.*

The planner chooses which cube on the belt to take next, and where it goes: a tower, a buffer slot or the reject bin. After a disturbance it also decides which fallen cubes to recover, in what order, and what to reject. It sees the recipes, the towers as they are (A2), the buffer and the cubes on the belt.

**Baselines:** the v2 greedy planner (front cube only) and a hand-written rule with batch choice (take a cube some tower needs next if the belt has one; otherwise route as greedy does). Under disturbances, the scripted recovery rules (A3).

| Target | Measured on |
|---|---|
| Routine orders (D0): not significantly worse than the hand-written batch rule at 0, 1 or 3 buffer slots | routing orders, symbolic simulation and physics cell |
| Routine orders at 3 slots in the physics cell, with the learned skills: at least 99 % | routing orders |
| **Under disturbances (D1–D3 pooled): significantly more orders completed (p < 0.05) than the scripted recovery rules, with the same skills** | disturbance suite, physics cell |

**Ablations** (reported, no target): the planner trained without seeing the cubes on the belt, without seeing the towers, and with skill outcomes assumed always successful in training.

## A2: closed loop

The planner observes the towers as they physically are, not as planned.

- **No regression:** undisturbed orders (D0) at 3 slots stay at or above 99 %.
- **Symbolic-vs-physics check:** on undisturbed orders, the planner's decisions in the physics cell match its decisions in the symbolic simulation in at least 99 % of steps. Mismatches are listed and explained.

## A3: recovery

*Amended on 2026-10-26, before any recovery code or training. A physical check, with no robot policy involved, knocked towers of 2–4 cubes over with realistic disturbances and classified where every fallen cube came to rest. The robot can only recover a cube that lies within its reach and rests flat. Even a perfect robot could recover every fallen cube after only about 68 % of single-cube knocks, 73 % of partial topples and 63 % of full topples. Cubes slide out of reach (inward under the arm, or beyond 0.78 m), land leaning on something, or land touching other cubes. A low rim around each tower did not help. So the targets below count the orders where recovery is physically possible. The orders where it is not are still run, and are reported separately.*

**Disturbances** (each knock is classified by its outcome, after everything has settled):
- a shove on the top cube, or on a middle cube, at 0.6–1.5 m/s in a random direction;
- a steady push of 1.0–2.5 N on the top cube in a random direction, as a hand would push, stopping once the cube has moved 3 cm.

Outcome: **D1** exactly one cube leaves the tower, **D2** some but not all, **D3** all of them. A knock is **recoverable** when every fallen cube rests flat (upright, or lying on a side, not leaning on anything), lies 0.30–0.78 m from the robot's base axis within its base rotation range, and is not on the belt or in the reject bin. Whether a knock is recoverable is decided by the simulation when it happens, before the robot acts.

**Baseline:** scripted recovery rules (re-pick fallen cubes, rebuild in recipe order, reject what cannot be used), with the same skills.

| Disturbance | Target: orders completed, of the recoverable knocks (3 seeds pooled) |
|---|---|
| D1 single cube | at least 90 %, and not significantly worse than the scripted recovery |
| D2 partial topple | at least 85 %, and not significantly worse than the scripted recovery |
| D3 full topple | at least 80 %, and not significantly worse than the scripted recovery |

Also reported, with no target: orders completed over all knocks, recoverable or not, and the share of knocks that were recoverable.

*A possible extension, not a target: scrap and replace. Cubes that cannot be recovered are written off, and a limited stock of spares can be called up the conveyor, which would let the planner choose between rescuing a hard cube and spending time and a spare on a replacement.*

**New skills** (trained with randomized cube mass, friction and size):

| Skill | Target (held-out cases, 3 seeds pooled) |
|---|---|
| Grasp a cube lying on its side | at least 85 % |
| Stand a held cube upright before placing | at least 90 % |

Each new skill is also run through the robustness suite and compared with a scripted version.

*Results, reported as they came (no target moved): the lying-cube grasp (d1) ran 84.8 % on its one-shot set, the stand-up (d2) 87.7 % on its one-shot set (91.5 % on its development set), each close to its scripted version. Neither was trained with a randomized cube size, against the line above; the stand-up's size robustness is weak (77.3 % against the scripted 92.5 %).*

### Recovery with the learned skills (d+), protocol set on 2026-10-29 before any d+ code

- **Targets unchanged:** the D1/D2/D3 table above, now with the learned lying-cube grasp and stand-up doing the recovery (3 seeds pooled); the scripted recovery is reported beside it on the same orders.
- **Development and final sets:** all d+ development uses `eval_sets/disturbance_dev_v1.json` (50 other orders, orders seed 20261101, knocks seed 20261102). `disturbance_suite_v1` is run once, at the end, for the reported result.
- **Headline: learned only.** When a learned skill fails, the cell retries with the learned skills (its usual attempts per cube); a variant that falls back to the scripted skills may be reported beside it, labelled.
- **First integration, no retraining:** the stand-up observes the nearest real tower, and its safety shield checks every obstacle in the cell. A gate test (isolated recoveries in the cell) decides whether the stand-up needs a round in cell-like scenes; any stand-up retraining first gets a fresh one-shot stand-up set.

*Result, reported as it came (configuration locked in 959d2f2; d1 round 5 was added for d+ on the development suite): on `disturbance_suite_v1`, 3 seeds pooled, all learned, recoverable D1 93.2 % (218/234), D2 87.5 % (28/32, a small sample), D3 91.5 % (75/82), so every target is met. Orders completed: 77.5 %, against 78.5 % for the scripted cell; the paired difference per order is not significant (p = 0.58). Seeds differ: seeds 1 and 2 recover 97–100 % of recoverable knocks, seed 0 only 73–81 %, because its pick knocks cubes over and its stand-up times out (79.4 % of stand-ups).*

## Deliverables

- An interactive demo: press a key in the viewer to knock a tower over, and watch the robot notice and rebuild it.
- A demo video with a panel showing the planner's decisions, and a side-by-side of v2 (does not notice a knock) and v3 (rebuilds).
- Model cards and a v3.0 release, as for v1.0 and v2.0.
