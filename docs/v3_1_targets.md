# v3.1 targets, set before any v3.1 training

v3.0 left two results short ([v3_targets.md](v3_targets.md), [recovery.md](recovery.md)):
- **The two new skills** each missed their target by a little on a one-shot set: the lying-cube grasp 84.8 % (target 85 %), the stand-up 87.7 % (target 90 %). Neither was trained with a randomized cube size, against what the v3 targets asked. The stand-up loses ten points when the cube size changes (77.3 %).
- **The learned recovery planner** beat the best hand-written rule in the symbolic cell but not in the physics cell (86.2 % against 88.3 %, p = 0.08). The cause was found: a cube left for a spare stays where it fell, often beside a tower, and later placements there fail. The symbolic cell, where the planner was trained, treats that cube as gone.

v3.1 is another round on both, under the project's usual rules: targets committed before the work they measure, held-out sets that training never sees, 3 training seeds per learned component, 95 % Wilson intervals, exact McNemar tests on paired cases, a scripted baseline beside every learned result, and every result reported as it comes, misses included. Cube positions from a camera are left for a later milestone.

## Evaluation sets, generated and committed before any v3.1 training

The v3.0 one-shot sets have each been run, so they become development sets. Fresh one-shot sets are drawn the same way, from new seeds:

| Set | Made like | Seed | Contents, digest | Used for |
|---|---|---|---|---|
| `grasp_lying_final_v2` | `grasp_lying_final_v1` (lying cubes, x 0.30–0.78 m, y ±0.20 m) | 20261201 | 200 cubes (73 near, 127 far), `f6bacc6ad73a` | the grasp's one-shot result |
| `standup_final_v2` | `standup_final_v1` | 20261202 | 200 starts (168 with a tower), `b711d50d5406` | the stand-up's one-shot result |
| `disturbance_suite_v2` | `disturbance_suite_v1` (Layer 2 orders, one knock each or none) | orders 20261203, knocks 20261204 | 200 orders (164 knocks, 36 controls) | the final cell result |

The same commands with the v1 seeds reproduce each v1 set byte for byte (checked before the v2 sets were drawn):

```bash
uv run python scripts/evaluate_grasp.py --make-set --lying --x-range 0.30 0.78 --y-range -0.2 0.2 \
    --n 200 --seed 20261201 --set eval_sets/grasp_lying_final_v2.json
uv run python scripts/evaluate_standup.py --make-set --n 200 --seed 20261202 \
    --set eval_sets/standup_final_v2.json
uv run python scripts/make_disturbance_suite.py --out eval_sets/disturbance_suite_v2.json \
    --orders-seed 20261203 --suite-seed 20261204 --n 200   # evaluate_cell.py ... --seed 20261203 --n 200
```

- **Development:** the grasp on `grasp_lying_v1`, `grasp_lying_final_v1` and `grasp_upright_v1`; the stand-up on `standup_v1` and `standup_final_v1`; the cell on `disturbance_dev_v1` and `disturbance_suite_v1`.
- Each one-shot set is run once, with a configuration locked in a commit beforehand.

## The skill round

**What changes in training:**
- **Cube size randomized**, ×0.85–1.15 (the robustness suite's range), in both skills, alongside the mass and friction already randomized.
- **Stand-up starts from real hand-offs:** a share of training episodes starts from the cube poses in the fingers that the learned lying grasp actually produces, not only from the synthetic grasp of v3.0.
- **The grasp near other cubes:** seed 0's grasp knocked neighbouring cubes over in the cell. Its failures are diagnosed first. Any fix is reported, with no separate target.

**Targets** (unchanged from v3; 3 seeds pooled, 600 cases):

| Skill | Target on its fresh one-shot set | Size bar (new) |
|---|---|---|
| Grasp a cube lying on its side | at least 85 % | with the cube size varied ×0.85–1.15, within 5 points of its one-shot result |
| Stand a held cube upright beside towers | at least 90 % | with the cube size varied ×0.85–1.15, within 5 points of its one-shot result |

- The size bar is measured on the same one-shot set, with the robustness suite's size condition (constant density). The other robustness conditions (mass, friction, joint offsets, combined) are reported with no target.
- **Baselines**, paired on the same cases: the scripted version of each skill, and the v3.0 skill.

**Stopping rule:** at most 3 training rounds per skill. Each round starts from a root-cause diagnosis of the previous round's failures, and rounds are compared on the development sets only. After the last round, or earlier if a round meets the target on development, the chosen configuration is locked and run once on the one-shot set. The result is reported as it comes, whatever it is.

## The recovery planner and clutter

**The task is v3.0's, unchanged:** 2 spares per order, a spare rides the belt like any arrival, and an order counts only if it is complete within 1.5 × its undisturbed time.

**Two changes:**
1. **Clutter in the symbolic cell.** The odds of a pick or place depend on the loose cubes near it: how many, and how close to the tower. These odds are fitted from calibration orders (`disturbance_calib_v1`), never from a suite (see the amendment below for which run). Cubes left behind stay on the floor of the symbolic cell where they fell.
2. **A third choice for the planner: clear the cube away.** Besides rescuing a fallen cube or leaving it for a spare, the planner may pick it up and drop it in the reject bin, then call a spare. Clearing needs no stand-up. Its odds and time are measured in physics on the calibration orders before any planner training. The drop into the bin is scripted and disclosed, as for surplus cubes.

*Amended on 2026-11-14, before any v3.1 code or training: a new calibration run. The v3.0 calibration log turned out to hold almost no clutter. Its recovery rescued every reachable cube at once, so only 71 of its 5,066 tower places had a knocked cube lying within 20 cm of the tower, and just 3 within 6 cm. The v3.0 physics runs on `disturbance_suite_v1` show the mechanism: with the best rule, places with a loose cube within 6 cm of the tower succeeded 79.7 % of the time (339 places), against 99.9 % with none within 20 cm (4,628 places); with the learned planner, 79.3 % (256) against 99.6 % (4,745). Cubes 6–20 cm away cost little (96–100 %) (`scripts/clutter_exposure.py`, `results/planner/clutter_exposure.json`). Odds fitted on the v3.0 log would leave the symbolic cell blind to clutter, and fitting them on the suite_v1 runs would make the check below circular. So the clutter odds, and the clearing action's odds and time, come from **a new physics calibration run**:*
- *the 200 calibration orders of `disturbance_calib_v1` (not a suite), all learned skills, 3 seeds, with spares and the deadline of the task above;*
- *for each fallen cube the planner is asked about, the choice of rescue, leave for a spare, or clear, is drawn at random (equal odds, from the run's own seeded stream), so the data covers the floors any planner can leave behind;*
- *run once the clearing action exists and before any planner is trained; the v3.0 calibration log keeps supplying everything else (the other skill odds, action times, knock outcomes).*

**Checks before any planner training, in this order:**
- **The symbolic cell reproduces the v3.0 gap.** On `disturbance_suite_v1`, now a development suite whose physics results are known, the clutter-aware symbolic cell must rank the three v3.0 policies as physics did (best rule, then the v3.0 learned planner, then "rescue every cube"), each within 3 points of its physics rate (88.3 / 86.2 / 84.3 % of knocked orders). If it does not, the clutter model is revised, or the user decides, before anything is trained on it.

  *Amended on 2026-11-15, after the check failed once and before any planner training: one revision of the time model, then one re-run.*
  - *The first run (`results/planner/clutter_check.json`, 30920a3) got each policy within 3 points (best rule 85.7, learned planner 86.4, rescue every cube 86.6) but ranked them wrongly.*
  - *Diagnosis (on the suite_v1 runs, to locate the error; nothing is fitted on them): the clutter part matches physics. Orders lost to a failed tower place within 20 cm of a loose cube: best rule 8.7 % in physics, 8.3 % symbolic; learned planner 7.0 % and 8.3 %. The error is in v3.0's part of the cell. On the other orders, "rescue every cube" succeeds 84.4 % in physics against 87.7 % symbolic, chiefly from lateness (13.4 % against 9.9 % of orders).*
  - *Cause: the symbolic cell gives every action a fixed time, so an order's duration has no spread. Physics durations scatter around the fitted times by about 9.6 s (sd) on knocked calibration orders, while the means match (ratio 0.99–1.01 by number of rescue actions). Rescuing every cube runs close to the deadline, where that spread turns into late orders.*
  - *The revision: each action's time in the symbolic cell gets Gaussian noise, with a variance per action fitted (non-negative least squares) to the squared duration residuals of the two calibration runs (v3.0 and v3.1, `disturbance_calib_v1`). The mean times are unchanged, and the noise has its own random stream so skill outcomes are not reshuffled.*
  - *The check is then re-run once, unchanged. If it fails again, it is reported, no planner is trained on the symbolic cell in v3.1, and the user decides what v3.1 ships for recovery.*
- **The headroom gate, as in v3.0:** in the clutter-aware symbolic cell, search over every rescue, leave or clear plan per order must beat the best hand-written rule by at least 5 points of knocked orders. The rule family grows to include clearing rules (for example, "clear a left-behind cube within 20 cm of a tower"), and the best rule is chosen in the symbolic cell, not on a suite. If the gate fails, it is reported, and the user decides whether the planner is still trained.

**Target** (unchanged from v3):

| Target | Measured on |
|---|---|
| **Knocked orders (D1–D3 pooled), with spares and the deadline: significantly more completed (p < 0.05) than the best hand-written rule, with the same skills and spares** | `disturbance_suite_v2`, physics cell, 3 seeds |

Also reported, with no target: the learned planner and the rules in the symbolic cell; the v3.0 planner on the new task; how often each policy clears, rescues or leaves a cube; and the ablations of v3.0 (blind to the situation, blind to the clock, trained with skills that never fail).

## The final cell run

The v3.1 cell (new skills, new planner) runs `disturbance_suite_v2` once, 3 seeds, with its configuration locked in a commit beforehand. The scripted cell and the v3.0 cell run the same orders beside it. The v3 recovery and routine targets still hold, as a regression check:
- recoverable knocks completed: D1 at least 90 %, D2 at least 85 %, D3 at least 80 %;
- undisturbed orders (D0): at least 99 %.

## Deliverables

- Updated `recovery.md`, `results.md` and the model cards, with every v3.1 result next to its v3.0 result.
- A v3.1 release: models archive, release notes, verified from a fresh clone.
