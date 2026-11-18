# v3.2 candidates (a proposal, not decisions)

*The grasp round and the pick checks were adopted on 2026-11-17, with their bars, in [v3_2_targets.md](v3_2_targets.md). It differs from this proposal in one way: the floor set holds nominal cubes, and cube size is a separate condition on the same cases with its own bar.*

*Written on 2026-11-16 and 2026-11-17, from the v3.1 diagnoses and the final cell run. Nothing here is decided. It collects what the v3.1 diagnoses point at, a check that would have caught the cell failures before the grasp was locked, and how each fix could be measured, in the style of [v3_1_targets.md](v3_1_targets.md). If a candidate is adopted, its bar is committed before any v3.2 training, under the project's usual rules: 3 seeds, held-out sets drawn before training, Wilson 95 % intervals, exact McNemar tests on paired cases, a scripted baseline and the previous version beside every result.*

## What the v3.1 final run says

- **The skills, not the recovery decisions, are where the points are.** The clear rule and the best rule completed the same share of knocked orders (86.0 % against 86.5 %, p = 0.84), and the v3.0 learned planner was close behind (84.3 %). On seeds 1 and 2 every configuration with spares completes about 93.5–94 % of orders. Seed 0 completes 70–74 %. Most of that gap is its grasp knocking cubes over: with the clear rule, 36 of seed 0's 52 failed orders began with such a knock-over.
- **D0 missed its bar** (99 %): 95.4–96.3 % of the planned no-knock orders, every miss on seed 0.

## What the v3.1 diagnoses found

All four are grasp problems. Each was reproduced in replays of the cell or the grasp's own scene, and none shows up in the one-shot lying set the grasp was locked on.

| # | Problem | Who | How big | Mechanism (evidence) |
|---|---|---|---|---|
| G1 | **Small upright cubes: the gripper never closes** | all seeds, worst seed 2 | upright, size under ×0.95: seed 2 46/74 (62 %), seed 0 58/74 (78 %); v3.0 54/74 and 61/74 | the hand comes down until the palm rests on the cube's top. On a small cube that leaves it more than 1 cm below the centre, outside the fixed 1 cm closing window, and it waits there until time runs out (replays of the failed cases) |
| G2 | **Under-rotation near 45°: no close, then a knock-over** (mechanism A) | seed 0 (v3.1 and v3.0) | upright floor cubes near the base turned −35..−45°: 24/26 knocked over in replays; 22 of seed 0's 35 floor knock-overs in the cell | the hand turns 8–11° too little and stops 1.4–1.9 cm off-centre, outside both the 8.6° and the 1 cm windows. It shuffles on the cube's top edge until the cube falls (replays with contacts) |
| G3 | **Outward descent with the fingers open** (mechanism B, and the belt) | seed 0 v3.1 | belt: 32/1,590 knocked over, 29 of them cubes turned −5..−15°; floor cubes turned +25..+40°: 13 of 35 knock-overs | every policy comes down 1.5–2 cm outward of the cube's centre and centres only once it is low. Seed 0 is 2.3–2.8 cm out, and the right finger lands on the cube's top corner at step 46 with the gripper open. In all 9 belt replays with contacts, that finger touches the cube first |
| G4 | **Large upright cubes knocked over** | seed 0 v3.1 | size over ×1.075: 31/39 (79.5 %, 7 knocked over), against v3.0 seed 0 38/39 | not replayed. *Hypothesis:* G3 again. A wider cube puts its top edge under an outward finger. A contact replay would confirm or reject this |

*Estimate, not measured:* if about a third of belt cubes are turned −5..−15° (the infeed's yaw spread is ±15°), then G3 knocks over roughly 5 % of those picks and well under 1 % of the others.

The same 228 standing-cube poses run in the grasp's own scene show that **v3.0's seed 0 had the G2/G3 weakness too**, on the other side of 45°: v3.0 seed 0 knocked over 18 of 48 cubes turned +26..+45°, v3.1 seed 0 10 of 24 turned −45..−36°. Round 1's training moved the weak side; it did not remove it. v3.1 seeds 1 and 2 knocked over 0 and 5 of 228. That scene is gentler than the cell (at −45..−36°, 10/24 there against 24/26 in cell replays), so a v3.2 floor check is better run through a cell station view.

## Candidate fixes

| Fix | Addresses | What changes | Retraining? |
|---|---|---|---|
| F1 **Wider closing window** for upright cubes: accept a hand up to 1.5 cm below the centre, as the lying-cube rule already does | G1 | the gate (`GateConfig`), in training and in use | not required (evaluation-only what-if: seed 2 46/74 → 71/74, seed 0 → 69/74 on the small cubes). Retraining under the new gate is cleaner, since the policy was trained with the old one |
| F2 **Rotation coverage**: upright cubes over the full ±45° range at the near-base ring (x 0.30–0.40 m) in a fixed share of training, with an extra share within 10° of ±45° | G2 | `resting` / placement sampling | yes |
| F3 **No finger on the top while open**: a penalty for finger contact with the cube's top face while the gripper is open (or an episode end, as for knock-overs), and upright cubes raised to belt height with belt-like yaw (±15°) in a share of episodes | G3, G4 | reward term; scene sampling (raised cubes) | yes |
| F4 **Large upright cubes** in a share of training (size ×1.05–1.15 upright) | G4 | sampling within the existing size range | yes (with F3) |

F1 alone fixes no knock-overs. F2–F4 are one retraining round with three changes. Their effects can only be told apart through the per-problem checks below, not by ablation, unless ablation rounds are paid for.

## A cell-like pick check, before any grasp is locked

v3.1's grasp met its one-shot target easily (97.8 %) and still had a 2 % belt knock-over rate and a 16 % floor knock-over rate on one seed in the cell. A check made of the cell's own pick situations would have shown both in minutes. Proposed sets, drawn and committed before v3.2 training like the other sets:

| Set | Contents | Why |
|---|---|---|
| `grasp_belt_v1` | upright cubes standing on the belt at the pick point, raised as the belt raises them, lateral jitter ±1.5 cm, yaw ±15° (the infeed's spread), 200 per seed | the cell's most frequent pick; G3 |
| `grasp_floor_upright_v1` | upright cubes on the floor near the base, radius 0.31–0.40 m, rotation uniform over −45..+45° relative to the hand, sizes ×0.85–1.15, 200 per seed, run through a cell station view | where cubes are picked again right after the stand-up turns them upright on the 0.36 m staging ring (31 of v3.1 seed 0's 35 floor knock-overs in the cell); G1, G2, G4 |

Each is run per seed (not only pooled), so a single weak seed cannot hide behind two good ones.

**Bars (proposed, in the v3.1 style):**
- **Belt set:** at most 1 knock-over in 200 per seed, and at least 99 % success per seed. *(At v3.1 seed 0's cell rate of 2 %, about 4 of 200 would be expected, if the set's rotations match the belt's.)*
- **Floor set:** at least 95 % per seed, and no 10° rotation bin under 85 % for any seed. *(v3.1 seed 0, cubes turned −45..−35° near the base: 24 of 26 knocked over in the cell replays.)*
- The one-shot lying target stays (≥ 85 % pooled, size bar within 5 points). A new round must not lose on it: paired against `grasp_anypose6`, no significant drop (exact McNemar, p < 0.05).
- In the cell: on a development suite (not a held-out suite), pick knock-overs per seed reported against v3.1, with no bar until the sets above are met.

## How each fix would be measured

| Fix | Primary measure | Secondary |
|---|---|---|
| F1 | upright success at size < ×0.95 on `grasp_floor_upright_v1` and the size condition of `grasp_upright_v1`, per seed, paired against v3.1 | lying one-shot unchanged (the lying window is already 1.5 cm) |
| F2 | floor set, rotation bins over \|rotation\| ≥ 30°, per seed; the number of "never closed then knocked over" episodes | hand-to-cube rotation error at grasp height (replay trace), against 8–11° in v3.1 seed 0 |
| F3 | belt set knock-overs per seed; first-contact body and place (finger on the top face while open) in contact replays | outward offset at 3.5 cm above the centre, against 2.3–2.8 cm for v3.1 seed 0 |
| F4 | upright success at size > ×1.075, per seed | knock-overs among them |

## Also worth noting

- **The stand-up's turn chooser** beats a random reachable command only for seed 2 (95.0 % against 89.9 %). For seeds 0 and 1 it is level with random (94.2 / 93.4 %, 95.8 / 96.5 %). A v3.2 stand-up round could test whether the chooser earns its place, or whether re-planning alone does the work.
- **Stand-up freezes** (never released) are still most failures: 48 of 60 on the v3.1 one-shot set.
- **The symbolic cell** (v3.1 amendment): physics retries a failed skill at a cost in time, and the symbolic cell ends the order. Revising that is a precondition for any planner work.
- **A D0 definition:** the cell labels every order whose knock never happened as `D0`, including orders that failed before their knock was due. v3.1's bar was read on the suite's planned controls (36 per seed), as in v3.0, and the cell's label is reported beside it. The v3.2 targets should say which one counts.
- **Seed 0's place:** in the cell without spares, seed 0's tower places succeed 95.9 % (seeds 1–2: 99.6–99.9 %; v3.0 seed 0, same place policy: 99.7 %). Not diagnosed; the cubes its grasp knocks over lie near the towers it builds, which is a likely cause to check first.
