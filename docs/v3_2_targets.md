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

## Baselines, as they came (2026-11-17, before any v3.2 training)

On the development sets, from `c3141a3` with a clean tree (`results/grasp/cellpicks_baseline_v32.json`, `scripts/cell_picks_report.py`). The scripted pick is one run. One run (v3.1 seed 0, floor, size condition) was killed by the out-of-memory killer while another project's jobs shared the machine; it was re-run unchanged from the same commit.

| Grasp, per seed 0 / 1 / 2 | Belt (`grasp_belt_v1`) | Floor (`grasp_floor_v1`) | Floor, size ×0.85–1.15 |
|---|---|---|---|
| v3.1 (`grasp_anypose6`) | 98.0 / 100 / 100 %; seed 0 knocks over **4**, all turned −15..−5° | 94.0 / 100 / 100 %; seed 0 knocks over **11**, weakest bin −45..−35° (14/18) | 84.5 / 100 / 99.0 %; seed 0 knocks over 19, 13 of them on cubes over ×1.075 |
| v3.0 (`grasp_anypose5`) | 100 / 100 / 100 % | 86.5 / 100 / 100 %; seed 0 knocks over **24**, weakest bin +35..+45° (4/25) | 85.5 / 98.0 / 99.0 % |
| Scripted pick | 100 % | 100 % | 100 % |

- **The sets catch what the cell showed, in minutes:**
  - seed 0's belt knock-overs, at the predicted rate (the cell's 2 % would be about 4 in 200) and in the same rotation band;
  - its floor knock-overs near 45°;
  - v3.0 seed 0's mirror-image weakness on the other side of 45°.
  - Seeds 1 and 2 are clean. No revision of the sets is needed.
- **Against the bars:** v3.1 misses the belt and floor bars on seed 0 only, and meets the size bar (pooled gap −3.5 points). Paired with v3.0, pooled: belt 0 / 4 (p = 0.13), floor 26 / 11 (p = 0.02), floor with size 29 / 27 (p = 0.89).
- **Not seen through the cell's pick: the small-cube closing trap (G1).** In the grasp's own scene, seed 2 grasped only about 6 in 10 upright cubes under ×0.95. Through the cell's pick, it grasps 46 of 47 under ×0.925. Not yet explained. Until it is, F1 (the wider closing window) is not needed for the cell's failures, which are F2–F4's: rotation near 45°, and seed 0's large cubes (32 of 45 over ×1.075).

## Grasp round 1, as it came (2026-11-17/18, development sets)

`grasp_anypose7` (F2–F4 from `0ca34e6`, warm-started from `grasp_anypose6`, 1M steps per seed, clean tree for all three). Reports: `scripts/cell_picks_report.py` and `results/grasp/round_grasp_anypose7.json`.

| Development check, per seed 0 / 1 / 2 | v3.2 round 1 (`grasp_anypose7`) | v3.1 (`grasp_anypose6`) | Paired, pooled (new only / old only) |
|---|---|---|---|
| Belt, `grasp_belt_v1` | **100 / 100 / 100 %, no knock-overs** (bar met) | 98.0 / 100 / 100 %, seed 0 knocks over 4 | 4 / 0, p = 0.13 |
| Floor, `grasp_floor_v1` | 96.0 / 100 / 100 %; seed 0 knocks over 8, weakest bins +25..+35° (19/23) and +35..+45° (21/25): **bar missed** on the 85 % bin rule | 94.0 / 100 / 100 %, seed 0 weakest bin −45..−35° (14/18) | 12 / 8, p = 0.50 |
| Floor, size ×0.85–1.15 | 89.0 / 99.5 / 98.5 %, pooled gap −3.0 points (bar met) | 84.5 / 100 / 99.0 %, gap −3.5 | 27 / 20, p = 0.38 |
| Lying, `grasp_lying_final_v1` | 96.0 / 94.5 / 92.0 % = 94.2 % | 95.5 / 99.0 / 99.0 % = 97.8 % | 11 / 33, **p = 0.001** |
| Lying, `grasp_lying_v1` | 96.3 % | 97.0 % | 14 / 18, p = 0.60 |
| Lying, size ×0.85–1.15 | 94.5 % | 97.0 % | 16 / 31, **p = 0.04** |
| Upright, `grasp_upright_v1` | 93.5 / 98.0 / 87.5 % = 93.0 % | 91.5 / 95.5 / 98.0 % = 95.0 % | 22 / 34, p = 0.14 |
| Upright, size ×0.85–1.15 | 88.3 % (gap −4.7) | 87.8 % (gap −7.2) | 44 / 41, p = 0.83 |

- **The belt is fixed.** Seed 0's four belt knock-overs are gone, with none on any seed.
- **Seed 0's floor weakness moved instead of going away.** Its negative side is now clean (−45..−25°: 40/40, from 33/40), but the positive side got worse (+25..+45°: 40/48, from 45/48, all failures knock-overs). This is the same mirror image seen between v3.0 and v3.1. With the size condition, the same side drops to 32/48.
- **Lying cubes regressed on seeds 1 and 2.** On the earlier one-shot lying set (grasp_lying_final_v1) the drop is significant (p = 0.001), and seed 2 also fell on upright cubes (87.5 % from 98.0 %). The lock-time regression check on `grasp_lying_final_v2` would likely fail. Round 1 is therefore not a lock candidate.

## Grasp round 1 diagnosis (2026-11-18/19)

Probes on the round 1 and v3.1 policies (replays that only observe; outcomes reproduce the committed results exactly):

- **Lying regression:** the cell-like 20 % (F2+F3) came out of the whole pool. Sampled training starts: lying 0.69 → 0.57, far (x ≥ 0.45 m) lying 0.29 → 0.24. Nearly every new lying failure is a far cube that is never closed on.
- **Not the finger-on-top penalty:** v3.1's successful far grasps rarely trigger it (4/84, 11/86, 4/87 per seed on lying cubes).
- **Seed 2's far upright regression is not explained by exposure.** Far upright training rose from 0.135 to 0.171.
- **Seed 0's one-sided knock-overs are not a data imbalance.** Training rotations are symmetric by construction, and no policy turned the long way round (0 of 1,800 traced episodes).
- **What does happen:** at the first finger–cube contact, the open hand is about 2 cm radially outside the cube's centre and 1–3 cm to the side, so a finger lands on the top edge. Rotation is nearly done by then (3–9° left). finger_top acts only once that contact has happened.
- **A cheap dev check:** the plain grasp environment reproduces the one-sided failures (an upright cube at x 0.30–0.40 m, y ±0.02 m, turned 20–45° either way), with no cell needed.

## Grasp round 2, as it came (2026-11-19, development sets)

**Setup:** `grasp_anypose8` (`f2d774e`), warm-started from `grasp_anypose6`, 1M steps per seed, clean tree.
- **R1:** `lying_prob` 0.875 (near_prob stays 0.4). Sampled lying, far lying, near lying and far upright shares match v3.1.
- **R2:** `reward.corridor` 0.2 per step while the open hand's TCP is below the cube's top + 1 cm and more than 2 cm off its centre. A pre-check on v3.1 found that its successful grasps cross this corridor about 1.3 steps per episode, its failures 33–70.

**Results with the round's checkpoint rule as it stood ("best" by validation reward):** seed 0 at 400k, seed 1 at 300k, seed 2 at 1M.

| Development check, per seed 0 / 1 / 2 | Round 2 (`grasp_anypose8`, best) | v3.1 (`grasp_anypose6`) | Paired, pooled (new only / old only) |
|---|---|---|---|
| Belt, `grasp_belt_v1` | **77.0** / 100 / 100 %; seed 0 knocks over 46 (bar missed) | 98.0 / 100 / 100 % | 2 / 44, p = 3e-11 |
| Floor, `grasp_floor_v1` | 93.0 / 100 / 100 %; seed 0 weakest bin +35..+45° 20/25 (bar missed) | 94.0 / 100 / 100 % | 12 / 14, p = 0.85 |
| Floor, size ×0.85–1.15 | 86.0 / 100 / 100 %, gap −2.3 points (bar met) | 84.5 / 100 / 99.0 % | 24 / 19, p = 0.54 |
| Lying, `grasp_lying_final_v1` | 97.5 / 97.0 / 98.5 % = 97.7 % | 97.8 % | 9 / 10, p = 1 (round 1 vs this: 94.2 %, p = 0.0015) |
| Lying, `grasp_lying_v1` | 97.0 % | 97.0 % | 17 / 17, p = 1 |
| Lying, size ×0.85–1.15 | 96.8 % | 97.0 % | 17 / 18, p = 1 |
| Upright, `grasp_upright_v1` | 91.5 / 93.0 / 93.5 % = 92.7 % | 95.0 % | 22 / 36, p = 0.087 |
| Upright, size ×0.85–1.15 | 85.5 % | 87.8 % | 31 / 45, p = 0.14 |

- **The lying regression is gone.** Seeds 1 and 2 are clean on every cell check.
- **Seed 0's "best" checkpoint broke on the belt.** To check whether that is the checkpoint or the lineage, seed 0's other checkpoints were run on the same development sets (output kept outside `results/`):

| Round 2 seed 0 | Validation | Belt | Floor | Near-base probe (neg / pos) |
|---|---|---|---|---|
| 250k | 92.0 % | 57.0 %, 86 knocked over | 68.5 %, 61 | 76/103, 53/97 |
| 400k (best) | 97.5 % | 77.0 %, 46 | 93.0 %, 8 | 88/103, 90/97 |
| 500k | 95.5 % | 96.0 %, 8 | 96.0 %, 8 | 98/103, 87/97 |
| 750k | 87.5 % | 99.0 %, 2 | 99.0 %, 2 | 100/103, 96/97 |
| 1M checkpoint | 93.0 % | 99.0 %, 1 | 100 %, 0 | 103/103, 96/97 |

- **The warm-started policy is disrupted early on and then keeps improving on the cell's cases.** Validation does not see this: its starts are whole-table placements with no belt or standing near-base cubes, and 200 episodes is noisy.

### Checkpoint rule, changed 2026-11-19 (user decision), before any final-checkpoint result

**From round 2 on, a round's policy is each seed's final checkpoint** (`final_model.zip`, the learning rate fully decayed), not the best-validation one. The rule applies to all three seeds alike.

- **This is a post-hoc change:** it was made after seeing seed 0's checkpoints on the development sets above.
- **Safeguards:** it is recorded here before any final-checkpoint result exists. The development sets choose nothing further. As before, the lock is judged only on the untouched one-shot sets.
- **Reports:** `grasp_round_report.py` and `cell_picks_report.py` take `<run>@final`.

### Round 2 final checkpoints, as they came (2026-11-20, development sets)

Each seed's `final_model.zip` (1,003,520 steps), evaluated from `1133a67`.

| Development check, per seed 0 / 1 / 2 | Round 2 (`grasp_anypose8@final`) | v3.1 (`grasp_anypose6`) | Paired, pooled (new only / old only) |
|---|---|---|---|
| Belt, `grasp_belt_v1` | 100 / 100 / 100 %, none knocked over (bar met) | 98.0 / 100 / 100 % | 4 / 0, p = 0.13 |
| Floor, `grasp_floor_v1` | 100 / 100 / 100 % (bar met) | 94.0 / 100 / 100 % | 12 / 0, p = 0.0005 |
| Floor, size ×0.85–1.15 | 96.0 / 100 / 100 %, gap −1.3 points (bar met) | 84.5 / 100 / 99.0 % | 32 / 7, p = 7e-5 |
| Near-base probe (neg / pos) | 103/103, 96/97 · 103/103, 96/97 · 103/103, 97/97 | | |
| Lying, `grasp_lying_final_v1` | 93.0 / **88.0** / 99.5 % = **93.5 %** | 95.5 / 99.0 / 99.0 % = 97.8 % | 11 / 37, **p = 0.0002** |
| Lying, `grasp_lying_v1` | 93.5 / **85.0** / 100 % = 92.8 % | 97.0 % | 13 / 38, p = 0.0006 |
| Lying, size ×0.85–1.15 | 94.0 / 83.5 / 100 % = 92.5 % | 97.0 % | 11 / 38, p = 0.0001 |
| Upright, `grasp_upright_v1` | 90.5 / 97.0 / 93.0 % = 93.5 % | 95.0 % | 27 / 36, p = 0.31 |
| Upright, size ×0.85–1.15 | 84.0 / 78.0 / 82.0 % = 81.3 %, gap −12.2 points | 87.8 %, gap −7.2 | 42 / 81, p = 0.0006 |

- **Every cell-like bar is met on every seed**, better than v3.1 and round 1.
- **The lying regression is back, on seeds 1 and 0.** Round 2 is therefore not a lock candidate, as with round 1. The one-shot sets stay untouched.
- **Where the failures are:** 35 of the 39 lying_final_v1 failures are far cubes (r ≥ 0.6 m) that are never closed on (seed 1: 23 of 24). A probe of the far cases shows the open hand comes down to the table 7–17 mm off the cube's centre and 5–20° misaligned, then stays low beside the cube for the rest of the episode without re-centring. v3.1's few far failures look the same; the final checkpoints just have many more. The corridor term fires throughout these failures (seed 1: median 37 steps), so it did not prevent this mode. Whether it contributes is not established.
- **Training is unstable on whole-table cases.** Seed 1's validation success swings 0.80–0.955 over the run (300k 0.955, 800k 0.80, final 0.91). v3.1 seed 1 stayed at 0.90–0.97 with the same learning rate. Seed 2 ends at its best. The two checkpoint rules fail in opposite ways: "best" missed seed 0's late cell-like gains, while "final" catches seed 1 in a whole-table trough.

## Grasp round 3 setup (2026-11-20, user approved; the last allowed round)

**Run:** `grasp_anypose9` (`configs/grasp_anypose9_cartesian_ppo.yaml`). It uses round 2's task unchanged (`configs/grasp_anypose8_cartesian.yaml`) and is warm-started from `grasp_anypose6` (best), 1M steps, 3 seeds.

Round 2's two checkpoint rules failed in opposite ways, so round 3 makes two changes:

- **Mixed validation.** Validation keeps training's cell-like starts (new `eval_cell_like` option; 12 % near-base upright, 8 % belt, 80 % whole table with no extra near-base draws). It runs 400 episodes every 100k steps, and a checkpoint is saved every 100k.
- **Learning rate halved:** 1.5e-4 → 1.5e-5 (was 3e-4 → 3e-5), to damp the drift away from the warm start that seed 1 showed.

**Round 3 checkpoint rule, fixed before training:** the round's policy is each seed's **best checkpoint by this mixed validation** (`best_model.zip`, mean validation reward), for all three seeds. The development sets then decide whether round 3 is a lock candidate, with the same bars as rounds 1–2. If it is, the lock is judged once on the untouched one-shot sets. If it is not, the v3.2 grasp target is recorded as not met and the cell keeps v3.1's `grasp_anypose6`.
