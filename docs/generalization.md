# Generalizing the learned grasp (Milestone 7)

The robustness suite ([robustness.md](robustness.md)) showed that the learned grasp does not extrapolate: with cubes placed in a 5 cm band just outside the training region, pick-and-place with the v2 grasp fell from 89.2 % to 58.0 %, while the scripted grasp held 99.0 %. Milestone 7 asked whether training can close that gap, changing one thing at a time.

**Targets** (set before any training): at least 85 % grasp success on a wide held-out set covering the whole feasible table, without falling below 87 % on the original set.

**Outcome:** not met. Neither a wider training region nor training among distractor cubes produced a better grasp than v2. The most useful finding was about checkpoint selection, not about the task.

## 7.1 Diagnosis: where the v2 grasp fails outside its region

From the out-of-region robustness condition (pick-and-place, v2 grasp, 3 seeds):

| Where the cube is | Success | Typical failure |
|---|---|---|
| Too close to the robot base | 49 % | the arm sweeps through the cube and knocks it over |
| Too far | 63 % | timeout, never lines up |
| Left or right of the region | 63–72 % | timeout, never lines up |

The effect starts inside the region: cubes within 3 cm of its edge succeeded 81 % of the time, interior cubes 90 %. The policy has learned motions for the states it saw, not a general way to reach a cube.

## Wide held-out sets

All placements come from the whole IK-verified feasible box, x 0.30–0.70 m and y −0.30–0.30 m, with any yaw. This is the original region (x 0.35–0.65 m, y −0.25–0.25 m) plus the 5 cm band of the robustness test, 1.6 times the area. The draws use their own spawn keys, as all held-out sets do ([evaluation_protocol.md](evaluation_protocol.md)).

| Set | Contents | Seed | Digest |
|---|---|---|---|
| `grasp_wide_v1` | 200 placements | 20261027 | `c8ba609e4d2e` |
| `pickplace_wide_v1` | 200 scenes, targets at least 10 cm away | 20261028 | `f425f96980b7` |
| `grasp_clutter_v1` | 200 placements, each with 2 other cubes at least 10 cm apart | 20261029 | `4ffba3e2e598` |

In the wide pick-and-place evaluation the supervisor's retry region is the wide box, so it only refuses to re-grasp cubes that left the table area it can reach.

Baselines at `81b43f4` (clean):

| Grasp | `grasp_wide_v1` | `pickplace_wide_v1` |
|---|---|---|
| Scripted | 100 % | 98.5 % |
| PPO v2 (best checkpoints, pooled 600) | 76.2 % (72.6–79.4 %) | 80.7 % (77.3–83.6 %) |

## 7.2 Training on the wide region

`configs/grasp_v2_wide_ppo.yaml`: the adopted v2 task (committed yaw only) with the wide placement region and nothing else changed. 3 seeds, trained at `81b43f4` (clean).

| Policy (pooled 600) | Original grasp | Wide grasp | Clutter grasp | Pick-and-place original | Pick-and-place wide |
|---|---|---|---|---|---|
| v2, best checkpoints | 87.0 % | 76.2 % | 77.7 % | 89.2 % | 80.7 % |
| wide, best checkpoints | 85.7 % | 81.5 % | 79.3 % | 85.8 % | 80.8 % |
| per seed (wide) | 76.5 / 94.0 / 86.5 | 68.0 / 89.5 / 87.0 | 66.0 / 89.5 / 82.5 | 84.5 / 90.5 / 82.5 | 72.0 / 90.0 / 80.5 |

Wide training improved the wide grasp by 5 points, lost a little on the original set, and did nothing end to end. The seeds disagree badly: one wide seed reached 89.5 % on the wide set (above the target), another 68 %.

## Checkpoint selection: the 30-episode validation was the bigger problem

Training keeps the "best" checkpoint according to a validation run during training. Until here that validation used 30 episodes. With 30 episodes the 95 % interval on an 85 % rate is about ±13 points, so "best" mostly meant "luckiest". Seed 1 of the v1 grasp, for example, scored 93 % in validation and 79 % on the held-out set.

`scripts/select_checkpoint.py` re-scores every saved checkpoint (best, final, and the intermediate ones every 250k steps) on a 100-placement validation set and copies the winner to `selected_model.zip`. The validation sets (`data/validation/grasp_select_nominal_v1.json`, `grasp_select_wide_v1.json`, seeds 20261030 and 20261031) are separate from every test set. The choice is recorded in `results/selection/` and the run's metadata, and used with `run:runs/<name>@selected`.

| Run | Validation scores (best / final / 250k / 500k / 750k) | Selected |
|---|---|---|
| v2 seed 0 | 80 / 73 / 56 / 78 / **85** | 750k |
| v2 seed 1 | 84 / 74 / 48 / **89** / 84 | 500k |
| v2 seed 2 | 91 / 67 / 49 / **94** / 93 | 500k |
| wide seed 0 | **72** / 66 / 30 / 53 / 72 | best |
| wide seed 1 | 91 / **93** / 52 / 78 / 83 | final |
| wide seed 2 | **83** / 61 / 51 / 68 / 66 | best |

Re-tested at `cfa939a` (clean):

| Policy (pooled 600) | Original grasp | Wide grasp | Clutter grasp | Pick-and-place original | Pick-and-place wide |
|---|---|---|---|---|---|
| v2, selected | **89.8 %** (87.2–92.0 %) | **81.8 %** (78.5–84.7 %) | 78.8 % | 89.0 % | **81.2 %** |
| wide, selected | 85.7 % | 81.3 % | **80.5 %** | 86.8 % | 80.8 % |
| Scripted | 100 % | 100 % | 100 % | 100 % | 98.5 % |

- **Fair selection gave the v2 policies as much on the wide set as wide training did** (76.2 → 81.8 %), without any new training. The final checkpoints are worse than mid-training ones for every v2 seed: these PPO runs drift after their peak.
- **Wide training is not adopted.** With selection applied to both, it is no better on the wide set (81.3 vs 81.8 %) and worse on the original set.
- **The adopted grasp stays v2.** End to end on the original set, selected and best checkpoints are equivalent (89.0 vs 89.2 %), so the published v1.0 numbers stand. Training now validates on 100 episodes.

## 7.3 Grasping among other cubes (task v3)

`GraspV3Env` (task `grasp_v3`, schema `grasp-v3-49d`) puts up to two other cubes on the table, at least 10 cm apart. The policy observes, for each slot, whether a cube is present and its position relative to the gripper. Moving another cube costs −20 per metre; knocking one over costs −5 and ends the episode (`knocked_distractor`). Training draws 0, 1 or 2 distractors per episode; validation and the clutter test set always use 2. Otherwise the task is the wide-region v2 task.

`configs/grasp_v3_clutter_ppo.yaml`: 1.5M steps (wider and harder task), 100-episode validation every 100k steps, 3 seeds, trained at `5b3c4f1` and evaluated at `b3a1786` (both clean). For comparison, the v2 grasp and the wide grasp ran on the clutter set without seeing the other cubes (`GraspV2BlindClutterEnv`: v2 observations in the cluttered scene).

| Policy (pooled 600) | Original grasp | Wide grasp | Clutter grasp |
|---|---|---|---|
| v3, best checkpoints | 65.0 % (61.1–68.7 %) | 56.2 % (52.2–60.1 %) | 41.5 % (37.6–45.5 %) |
| per seed (v3) | 62.0 / 86.0 / 47.0 | 57.5 / 79.0 / 32.0 | 36.0 / 67.0 / 21.5 |
| v2, selected, blind to the other cubes | **89.8 %** | **81.8 %** | **78.8 %** |
| Scripted | 100 % | 100 % | 100 % |

- **v3 is worse everywhere, including the clutter it was trained for.** The v2 policy, which has never seen another cube, does nearly twice as well among them.
- **Most v3 failures are hesitation.** On the clutter set, 258 of 351 failures never closed the fingers (v2: 75 of 127). The disturbance penalty most likely taught the policy to avoid the contact a grasp needs near other cubes. This is the same failure as the pushing penalty in the full v2 task ([grasp_task.md](grasp_task.md#full-v2-all-three-changes-at-once-failed)).
- **Seeds disagree even more than for the wide task:** 21.5–67 % on the clutter set. Validation during training showed the same pattern: seed 1 peaked at 83 % and the other two plateaued near 45 % and 35 %.
- End-to-end evaluation of v3 needs a cluttered scene in the transport and place environments too (the supervisor hands over exact physics states, so all stages must share one model). That scene arrives with stacking in Milestone 9.

**Decision:** v3 is not adopted.

## What Milestone 7 showed

1. **The learned grasp's limits are seed-to-seed reliability and validation noise more than task coverage.** The same configuration produces policies between 66 % and 94 %. Measuring that took 3 seeds per configuration and 100-episode validation; a single run would have produced either conclusion.
2. **Penalizing contact with objects the gripper must get close to makes PPO hesitate.** This happened twice: pushing the target cube (v2 full) and disturbing other cubes (v3).
3. **The scripted grasp, planning from the measured cube pose, is still better everywhere tested** (100 % on all three sets). Under the adoption rule it remains the default; the learned grasp is reported alongside it.

Possible next steps that were not tried: widening the region gradually as a curriculum, a gentler or delayed disturbance penalty, and an action space in which the policy commands the gripper's motion directly instead of joint increments (Milestone 8).
