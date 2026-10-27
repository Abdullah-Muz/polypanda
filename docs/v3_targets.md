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

The planner chooses a tower, a buffer slot or the reject bin for each arriving cube. It sees the recipes, the towers' progress, the buffer and the cubes visible on the conveyor.

| Target | Measured on |
|---|---|
| Not significantly worse than the greedy planner at any buffer size | routing orders, symbolic simulation and physics cell |
| **Significantly better than greedy (p < 0.05) at 0 or 1 buffer slots** | routing orders, symbolic simulation |
| At 3 slots in the physics cell, with the learned grasp and place: at least 99 % of orders | routing orders |

**Ablations** (reported, no target): the planner trained without the conveyor look-ahead, and without seeing the towers.

## A2: closed loop

The planner observes the towers as they physically are, not as planned.

- **No regression:** undisturbed orders (D0) at 3 slots stay at or above 99 %.
- **Symbolic-vs-physics check:** on undisturbed orders, the planner's decisions in the physics cell match its decisions in the symbolic simulation in at least 99 % of steps. Mismatches are listed and explained.

## A3: recovery

**Baseline:** scripted recovery rules (re-pick fallen cubes, rebuild in recipe order, reject what cannot be used), with the same skills.

| Disturbance | Target: orders completed (3 seeds pooled) |
|---|---|
| D1 single cube | at least 90 %, and not significantly worse than the scripted recovery |
| D2 partial topple | at least 85 %, and not significantly worse than the scripted recovery |
| D3 full topple | at least 80 %, and not significantly worse than the scripted recovery |

**New skills** (trained with randomized cube mass, friction and size):

| Skill | Target (held-out cases, 3 seeds pooled) |
|---|---|
| Grasp a cube lying on its side | at least 85 % |
| Stand a held cube upright before placing | at least 90 % |

Each new skill is also run through the robustness suite and compared with a scripted version.

## Deliverables

- An interactive demo: press a key in the viewer to knock a tower over, and watch the robot notice and rebuild it.
- A demo video with a panel showing the planner's decisions, and a side-by-side of v2 (does not notice a knock) and v3 (rebuilds).
- Model cards and a v3.0 release, as for v1.0 and v2.0.
