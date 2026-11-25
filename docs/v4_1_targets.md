# v4.1 targets, set before any v4.1 training

*Approved by the user on 2026-11-25, before any v4.1 training and before any evaluation set below exists. Binding from this commit: every change after it is written in openly, with its reason and what it costs, as in [v4_targets.md](v4_targets.md).*

v4.1 is an electronics test cell ([v4_1_plan.md](v4_1_plan.md)) with two learned skills: **precision insertion** of a keyed module into a tight test nest (new), and **the door** extended from v4.0's sliding lids to tester lids and drawers, from any hand-over. Everything else in the cell (picking, driving, scanning, inspection on the fly, sorting, scheduling) is scripted.

The project's usual rules hold:
- These targets are committed before the work they measure.
- Evaluation sets are generated and committed (with digests) before training, and training never sees them.
- 3 training seeds, each judged separately unless stated.
- 95 % Wilson intervals; exact McNemar tests on paired episodes.
- Scripted baselines run on the same episodes.
- Every result is reported as it comes, misses included.
- **At most 3 development rounds per skill**, fixed here, not raised later (the lesson of v4.0's fifth round).

## What the scripts showed (Phase 2, development nests and drawers)

**Insertion** (`results/v4/insertion_baselines_dev.json`, `insertion_search_dev_rev1.json`; stiff contacts; clearance 0.25/0.5 mm, chamfer 0.5/1 mm, nest turned 0/1/2°, aim off 0–3 mm; 20 trials per condition):

| | Straight-down | Force-guided search (frozen, `8e884b2`) |
|---|---|---|
| Seated, all 960 | 40.4 % | 75.7 % |
| 0 / 1 / 2 / 3 mm aim error | 99 / 59 / 3 / 1 % | 94 / 64 / 79 / 66 % |
| Nest turned 2° | 40 % | 54 % |
| Median peak force on the nest | 16 N | 6 N |
| Median time (carry posture to hand clear) | 6.4 s | 17.5 s; 26 % of trials hit its 30 s budget |

The search is the fair baseline: the standard industrial method, tuned at the cell's 20 Hz control rate, revised once (wiggle ±2.5°) and frozen before any target. Straight-down is the cell's naive reference.

**Door on drawers** (`eval_sets/door_drawer_probe_v1.json`, 50 drawers): expert 50/50, `feel_v2` 50/50 (median peak 12.6 N), `camera` 41/50 (all 9 failures are closes), `nominal` 37/50.

**The cell** (scripted, the real tester kinds A lid, B drawer, C pneumatic; 3 dev shifts × 30 modules, `feel_v2` door, inspection on the fly, 45 s tests; `results/v4/test_cell/*_planned_n30_seeds1-2-3.json`, at `d67ecee`):

| | Straight-down | Search |
|---|---|---|
| Judged modules per hour | 42.3 | 49.1 |
| First-try seat | 71 % | 96 % |
| Not tested (to rework) | 14 of 90 | 0 |
| Operator interventions | 9 (6 modules tipped off a drawer's tray) | 1 |
| Nest force, median / max | 25 / 266 N | 7 / 12 N |
| Insert time, median | 8.0 s | 11.7 s |
| Escapes / false rejects | 0 / 1 | 0 / 2 (vision) |

In the cell the search seats far more often than on the development grid's hard corners (the cell's camera errors are mostly near 1 mm). The scripted cell with the search is therefore a strong reference: the learned cell has to win on time without losing modules.

## Part I: the learned insertion

### The task, as the policy and the scripts face it

- **Start:** the module held (the cell's grasp, its offset in the hand as a real pick leaves it), the hand in the carry posture over the nest's believed place: the same start for the policy and both scripts.
- **Observed:** the arm's joints, the hand pose, the **camera's estimate** of the nest's pose, and the force and torque on the held module (a wrist force-torque sensor; `part_wrench`). **Not observed:** the nest's true pose, the module's place in the hand, clearance, chamfer.
- **Acts** at the cell's 20 Hz, as the scripts do.
- **Success:** seated (on the nest's floor, flat within 2°, inside the clearance), released, hand back at the carry height, and **no contact force on the nest above 50 N at any step** (a damage limit for pogo pins and housings; straight-down jams reach 15–330 N).
- **Time:** from the start to the hand clear, as the scripts are timed. **Budget 30 s**, the search's own.

### The episode distribution (training, development and final sets alike)

| Factor | Drawn as | Why |
|---|---|---|
| Clearance per side | uniform 0.25–0.5 mm | Phase 0: stiff contacts are trustworthy from 0.25 mm |
| Chamfer | uniform 0.5–1.0 mm | tester nests' small lead-ins |
| Camera error, position | each axis normal, σ 0.8 mm (mean 1.0 mm radial), clipped at 3 mm | the plan's "about 1 mm" camera error |
| Camera error, yaw | normal, σ 0.5°, clipped at 2° | the plan's "about 0.5°" |
| Module in the hand | as the cell's pick leaves it (pick jitter of the infeed) | the real source of the hand offset |
| Nest place | the cell's tester stops and nest heights, drawn per tester as in v4.0 | unseen testers |

The development grid above used fixed levels up to 3 mm and 2°; this distribution puts most episodes near 1 mm and 0.5°, where the scripts differ most.

### Evaluation sets

| Set | Contents | Seed | Used for |
|---|---|---|---|
| `insertion_dev_v1` | 400 episodes, the distribution above | 20261127 | development rounds; the scripts' dev numbers |
| `insertion_final_v1` | **600** episodes, the same distribution | 20261128 | the one-shot result, run once |
| `insertion_stretch_v1` | 400 episodes: camera error σ 1.5 mm (clip 4 mm), yaw σ 1° (clip 3°) | 20261129 | reported, no bar |

*Recorded 2026-11-25 (after approval; no target changed):* generated by `scripts/make_insertion_sets.py` at `c109e72` (`docs/insertion_env.md`): `insertion_dev_v1` digest `693e45d96e37`, `insertion_final_v1` `0d1623e33b82`, `insertion_stretch_v1` `9bd043477747`. Training draws episodes from `env.reset(seed=n)` and grasps from a separate pick pool (`data/handoff/insertion_picks_train_v1.json`, seed 20261131), never from the sets' streams.

600 final episodes because industry states insertion reliability as "99.5 % at 95 % confidence", which needs about 600 trials with no failure. That check is **reported** for every seed; it is not a bar (see I1).

### Targets (on `insertion_final_v1`)

| # | Check | Target |
|---|---|---|
| I1 | Success | at least **95 %** for every seed |
| I2 | Against straight-down | significantly better (exact McNemar, p < 0.05), pooled |
| I3 | Against the search | success **no more than 2 points lower**, pooled; **and faster:** median time to success at least **40 % shorter**, pooled, on the episodes both complete |
| I4 | Gentleness | median peak force on the nest **no higher than the search's**, pooled; the 50 N damage limit is part of success (I1) |

- **Why I3 asks for 40 %, not v4.0's 20 %:** the search is slow by construction (median 17.5 s, a quarter of trials at its 30 s budget), and reading the first contact and correcting in one move is exactly where a learned insertion can win. A smaller bar would be met by a policy that only trims the search's approach.
- **What a miss means:** I1 or I2 missed: the skill does not work. I3 missed on success: the search is more reliable, reported as the finding. I3 missed on time, or I4 missed: the trade-off is reported as it is, not avoided.
- **Before any bar is final:** both scripts run on `insertion_dev_v1`, committed. If the search seats less than 85 % there, I1's 95 % is kept (a learned skill may beat a weak search), but I3's "2 points lower" becomes "not lower" (no credit for matching a script that fails often). This rule is fixed now, before the dev numbers.
  - **Outcome (recorded 2026-11-25, results at `45e8d24`, run at clean `c109e72`):** on `insertion_dev_v1` the search seats **330/400 = 82.5 %** (95 % CI 78.5–85.9), below 85 %, so **I3 reads: success not lower than the search's, pooled; and median time to success at least 40 % shorter**. Straight-down seats 197/400 (49.2 %; 184 of its failures break the 50 N limit). The search's median time to success in the env is **8.7 s**, not the probe's 17.5 s: the env's clock starts after the cell's aim, where the probe's included the approach from the carry height. The 40 % bar is relative and unchanged; against this search it means a median of about 5.2 s or less.

### Training rules

- **Demonstrations from the search script** on freshly drawn training episodes (never the evaluation sets); **behaviour cloning**, then **PPO**. The clone's dev result is reported.
- **Rounds:** at most **3** development rounds on `insertion_dev_v1`; each round's config committed before it trains.
- **The policy used for a seed:** the **final checkpoint of its last round** (v4.0's rule after round 2: best-validation checkpoints proved a lottery).
- **The one-shot run** on `insertion_final_v1` happens once, after a lock commit names the policy, the seeds and the scripts' versions.

## Part II: the door on testers

### Changes from v4.0

- **Testers:** lids (v4.0's machine family, drawn as tester lids) and **drawers** sliding toward the robot (track B). The pneumatic tester has no door to work.
- **Any hand-over:** starts drawn with the hand from a few centimetres to about 35 cm above the handle (v4.0's T5: 67 % of machines running from the line's hand-over, 95 % from the trained start).
- **Retrained from v4.0's round 5** final checkpoints (same observation, `door-v2-50d`; the tester kind is not observed: drawers are told apart by feel and by the bar's look, as slide direction was in v4.0).
- **Baselines:** `feel_v2` (the fair one) and `camera`, both extended to drawers; `nominal` and the privileged expert reported for reference.

### Evaluation sets

| Set | Contents | Seed | Used for |
|---|---|---|---|
| `door_dev_v2` | 400 episodes: 200 lids, 200 drawers, half open and half close, hand-over starts drawn wide | 20261130 | development |
| `door_final_v2` | 400 episodes, the same mix | 20261201 | the one-shot result, run once |

*Recorded 2026-11-25 (after approval; no target changed):* generated at `be2362c` with `DoorStartConfig.wide` (the hand 3–35 cm above the bar, −8 to +20 cm radially, ±8 cm sideways, wrist ±90°; [door_env.md](door_env.md)): `door_dev_v2` digest `231dbe288150`, `door_final_v2` `1db12384e8da`, 100 episodes per kind × command. On `door_dev_v2` (`28b855b`): `feel_v2` 398/400 (passes the 90 % demonstration check), expert 400, `camera` 242, `nominal` 149; v4.0's round-5 final policies 140, 174 and 122 (no drawer training; weakest from turned wrists and high starts).

`door_drawer_probe_v1` (the 50 drawers of Phase 2) stays a probe, not an evaluation set.

### Targets (on `door_final_v2`)

| # | Check | Target |
|---|---|---|
| D1 | Success | at least **95 %** for every seed; at least **90 %** in every cell of kind (lid, drawer) × command, for every seed |
| D2 | Against `camera` | significantly better (exact McNemar, p < 0.05), pooled |
| D3 | Against `feel_v2` | success no more than 2 points lower, pooled; median time to success at least **20 % shorter**, pooled, on the episodes both complete |
| D4 | Gentleness | median peak force no higher than `feel_v2`'s, pooled; slams in at most 1 % of episodes for every seed |
| D5 | No forgetting (reported, no bar) | the v4.1 policies on `door_dev_v1` (v4.0's lid set: lids only, v4.0's starts) beside v4.0's round 5 |

D4 is v4.0's T4, which v4.0 missed (31 vs 26 N). It is kept as strict: if missed again, it is reported as the speed/force trade-off.

### Training rules

As v4.0's, with at most **3** development rounds on `door_dev_v2`, the policy = the final checkpoint of the last round, and one one-shot run after a lock commit.

## Part III: the cell

**Compared on the same shifts:** the **learned cell** (learned door on A and B, learned insertion on all three testers) against the **scripted cell** (`feel_v2` door, search insertion), both with the real tester kinds (lid, drawer, pneumatic), inspection on the fly, 45 s tests. **Final shifts:** seeds 1001–1006, 30 modules each (180 modules; truths drawn per seed), run once, after both skills are locked. Development shifts use seeds 1–99.

| # | Check | Target |
|---|---|---|
| C1 | Throughput | judged modules per hour (PASS + FAIL) higher than the scripted cell's, pooled over the six shifts |
| C2 | Modules not tested | no more than the scripted cell's, pooled |
| C3 | Quality (reported, no bar) | escapes, false rejects, sort correctness against the hidden truth, peak nest force, failures by stage, robot time breakdown, tester utilization |

C2 is the cell-level form of the insertion's job: every module that will not seat goes to rework untested. The all-straight-down cell is reported beside both, as the naive reference.

## Not targets

- Picking, scanning, inspection and scheduling stay scripted; their numbers are reported in C3.
- The rigid grasp assist (no give in the fingers) is kept for consistency with v4.0 and stated beside every insertion result.
- Clearances below 0.25 mm are not claimed (Phase 0).
