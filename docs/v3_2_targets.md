# v3.2 targets, set before any v3.2 training

v3.1 ([v3_1_targets.md](v3_1_targets.md), [recovery.md](recovery.md#v31-milestone-11-new-skills-and-clearing-clutter)) showed where the remaining points are:
- **The skills, not the recovery decisions.** The clear rule and the best rule completed the same share of knocked orders (86.0 % against 86.5 %, p = 0.84).
- **Seed 0's grasp drags the cell down.** On seeds 1 and 2, the v3.1 cell completes 93.5 % of all orders, and every order with no knock. On seed 0 it completes 74.0 %, and 36 of its 52 failed orders began with the grasp knocking an upright cube over. Orders with no knock completed 95.4–96.3 %, against a 99 % bar.
- **The grasp's tests did not see it.** The one-shot lying set was met easily (97.8 %). Seed 0's knock-overs happen on upright cubes the cell turns in ways that set does not cover. On the belt they are cubes turned 5–15° one way; on the floor near the base, cubes turned close to 45°. v3.0's seed 0 had the floor weakness too, on the other side of 45°.

v3.2 is one grasp round aimed at those failures, under the project's usual rules: targets committed before the work they measure, held-out sets that training never sees, 3 training seeds, 95 % Wilson intervals, exact McNemar tests on paired cases, a scripted baseline and the previous version beside every result, and every result reported as it comes, misses included. The candidates behind it are in [v3_2_candidates.md](v3_2_candidates.md). The symbolic cell's revision (a failed skill costs time instead of ending the order) and cube positions from a camera are left for later milestones, with their own targets.

## Cell-like pick checks

A new check runs a grasp the way the cell runs it (`scripts/evaluate_cell_picks.py`, `evaluation/cell_picks.py`). Each case stages one upright cube in a fresh cell and calls the cell's own pick: the same base swing, station view, closing gates and grasp assist as in a work order. Towers stand partly built (0–3 cubes each) around it.
- **Belt:** the cube stands on the belt at the pick point, offset sideways up to ±1.5 cm and turned up to ±15°, as the infeed delivers it, with 0–2 more cubes queued behind it.
- **Floor:** the cube stands on the table at one of the cell's five staging angles, where the stand-up sets cubes upright and they are picked again, 0.31–0.40 m from the base axis, turned uniformly −45..+45° relative to the hand.
- **Size condition:** the same cases with the test cube scaled by a factor drawn per case from ×0.85–1.15 at constant density, as the robustness suite's size condition. The cell's own cubes are nominal, so the bars below are on nominal cubes; the size condition has its own bar.

Every case reports success, whether the cube was knocked over (tipped past 45°, or the grasp env's knock-over check), and the cube's rotation relative to the hand. Results are given per seed and by 10° rotation bin, not only pooled, so a single weak seed cannot hide behind two good ones.

## Evaluation sets, generated and committed before any v3.2 training

| Set | Contents, digest | Seed | Used for |
|---|---|---|---|
| `grasp_belt_v1` | 200 belt cases, `df27cd6acc2b` | 20261210 | development: comparing rounds, and the baselines |
| `grasp_floor_v1` | 200 floor cases, `3e4bf79d8493` | 20261211 | development |
| `grasp_belt_final_v1` | 200 belt cases, `60b305744abd` | 20261212 | the one-shot result, run once |
| `grasp_floor_final_v1` | 200 floor cases, `2b5c90a4a278` | 20261213 | the one-shot result, run once |
| `disturbance_suite_v3` | 200 Layer 2 orders (150 knocks, 50 no-knock controls), file sha256 `d3c9566d4734` | orders 20261214, knocks 20261215 | the final cell run |

```bash
uv run python scripts/evaluate_cell_picks.py --make-set belt --n 200 --seed 20261210 --set eval_sets/grasp_belt_v1.json
uv run python scripts/evaluate_cell_picks.py --make-set floor --n 200 --seed 20261211 --set eval_sets/grasp_floor_v1.json
uv run python scripts/evaluate_cell_picks.py --make-set belt --n 200 --seed 20261212 --set eval_sets/grasp_belt_final_v1.json
uv run python scripts/evaluate_cell_picks.py --make-set floor --n 200 --seed 20261213 --set eval_sets/grasp_floor_final_v1.json
uv run python scripts/make_disturbance_suite.py --out eval_sets/disturbance_suite_v3.json \
    --orders-seed 20261214 --suite-seed 20261215 --n 200   # evaluate_cell.py ... --seed 20261214 --n 200
```

- **Baselines, before any training:** on the two development sets, the v3.1 grasp (`grasp_anypose6`), the v3.0 grasp (`grasp_anypose5`) and the scripted pick, nominal on both sets and with the size condition on the floor set. They are committed before any v3.2 training, and they also check that the sets catch what the cell showed (seed 0's knock-overs). If they do not, the sets are revised, and that is reported, before anything is trained on them.
- `grasp_lying_final_v2` (v3.1's one-shot set) and the other v3.1 sets are development sets now.

## The grasp round

**What changes in training** (the candidates F1–F4; exact values are fixed in the round's config commit, before training):
- **F1, a wider closing window for upright cubes:** the gate accepts a hand up to 1.5 cm below the cube's centre, as the lying-cube gate already does. It applies in training and in use, for every grasp run with the new configuration.
- **F2, rotation coverage:** a share of training episodes stands upright cubes near the base (x 0.30–0.40 m) at any rotation over −45..+45°, with an extra share within 10° of ±45°.
- **F3, no finger on the cube's top while the gripper is open:** a penalty for that contact, and upright cubes raised to belt height with the belt's yaw spread (±15°) in a share of episodes.
- **F4, large upright cubes** (×1.05–1.15) in a share of training.

Warm-started from `grasp_anypose6`, 3 seeds.

**Targets** (each seed separately, on the one-shot sets, nominal cubes unless stated):

| Check | Target |
|---|---|
| Belt, `grasp_belt_final_v1` | at least 99 % success **and** at most 1 cube knocked over in 200, for every seed |
| Floor, `grasp_floor_final_v1` | at least 95 % success for every seed, and no 10° rotation bin under 85 % for any seed |
| Floor, size condition | within 5 points of the floor result, pooled over the 3 seeds |
| Lying cubes, `grasp_lying_final_v2` | no significant drop against `grasp_anypose6` (exact McNemar, p < 0.05), pooled; run once at the lock as a regression check |

- **Baselines on the one-shot sets,** run beside the new grasp at the lock: `grasp_anypose6`, `grasp_anypose5`, and the scripted pick.
- **Stopping rule:** at most 3 training rounds. Each round starts from a root-cause diagnosis of the previous round's failures, and rounds are compared on the development sets only. After the last round, or earlier if a round meets every target on development, the chosen configuration is locked in a commit and run once on the one-shot sets. The result is reported as it comes, whatever it is.

## The final cell run

The v3.2 cell (the new grasp, the v3.1 stand-up and place, and v3.1's recovery: the clear rule, 2 spares, the 1.5× deadline) runs `disturbance_suite_v3` once, 3 seeds, with its configuration locked in a commit beforehand. The v3.1 cell runs the same orders beside it, so that only the grasp differs between the two. The scripted cell runs them too.

**Target:**

| Target | Measured on |
|---|---|
| **All orders: the v3.2 cell completes significantly more (p < 0.05, exact McNemar paired by seed and order) than the v3.1 cell** | `disturbance_suite_v3`, physics cell, 3 seeds |

**Regression check**, the v3 targets, as in v3.1:
- recoverable knocks completed without spares (rescue every cube): D1 at least 90 %, D2 at least 85 %, D3 at least 80 %;
- **orders with no knock (D0): at least 99 %**, read from the v3.2 cell and from the no-spares configuration. **D0 means the suite's planned no-knock orders** (50 per seed in `disturbance_suite_v3`), as in v3.0 and v3.1. The cell's own D0 label, which also counts planned-knock orders that failed before their knock was due, is reported beside it.

**Also reported, with no target:** pick knock-overs per seed in the cell (belt and floor), against v3.1 on the same orders, and seeds 1–2 next to seed 0.
