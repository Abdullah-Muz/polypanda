# The learned insertion's training environment (v4.1)

The env for v4.1's new learned skill: seat a held, keyed electronic module in a tight test nest, from the cell's carry posture, with only a camera's estimate of where the nest is ([v4_1_targets.md](v4_1_targets.md), Part I, binding; [v4_1_plan.md](v4_1_plan.md), section 4.2). Code: `src/polypanda/environments/insertion.py` (`InsertionEnv`, task `insertion`), `src/polypanda/simulation/machine_variation.py` (`NestLayout`: a nest re-cut without recompiling), `src/polypanda/evaluation/insertion.py` (the scripts, policies and demonstrations on the env's episodes), `scripts/make_insertion_sets.py`, `scripts/evaluate_insertion.py`, `scripts/record_insertion_demos.py`, `configs/insertion_v1.yaml`, `tests/test_insertion_env.py`. Nothing has been trained: this is the environment, the sets and the scripted baselines on them.

## The task

- **Scene.** The test cell's robot on its rail, with the cell's three testers at their rail stops: A, a lid tester (−0.9 m), B, a drawer that opens toward the robot (−0.45 m), C, a pneumatic lid (0 m), each with a tight, keyed nest with stiff contacts (`simulation/test_cell.planned_testers`, `machine.tight_nest`), the infeed conveyor, and one keyed module (the cell's 40 × 40 × 60 mm part with its rib, stiff contacts, held by the cell's grasp assist). No stations, dressing or hall, for speed. As in the test cell, the two fingers never collide with each other.
- **Start** (the same for the policy and both scripts): the robot at the tester's stop in the cell's carry posture facing it, the module in the hand as its pick left it, the tester open (a lid slid open, a drawer pulled out and held at its stop by its index pin); the tester measured by the camera; the insertion **aimed** as the cell's insertion slot aims every insertion (`control/test_cell.aim`): the hand over the believed nest at the carry height, the wrist turned so the module's key meets the believed slot (about 90° on the drawer, whose fingers close across it). Time and force are counted from there, as the search times itself.
- **Success** (the targets' definition): the module **seated** (its bottom within 1 mm of the nest's floor, tilted less than 2°, inside the clearance plus 0.2 mm, its key in the slot within 10°), **released**, the **hand back at the carry height** (within 1 cm) and clear of the module, and the **nest force never above 50 N**. The policy's episode ends at success; at the damage (the force over 50 N); released with the hand back up but not seated; or at the **30 s** budget (600 steps).
- **The nest force** is the largest normal force of any contact between the module and the tester (the nest's walls and chamfers, a drawer's tray and front, the tester's body) or the nest's floor (the bench, under a lid or pneumatic tester's nest), checked **after every 2 ms physics step**. The cell and the Phase 2 probe report the same measure but sample it once per 50 ms control step and leave the bench under the nest out; both peaks are recorded (`peak_force`, `peak_force_ctl`), with the summed force at control steps (`peak_force_total`). Why every physics step: a learned policy can find short impacts a 20 Hz sample misses, and the nest's floor is where a test nest's pogo pins are.

## The episode distribution (`draw_episode`)

As [v4_1_targets.md](v4_1_targets.md) fixes it:

| Factor | Drawn as | How |
|---|---|---|
| Tester | A (lid), B (drawer), C (pneumatic), uniformly: its rail stop and its nest's height (a drawer's nest stands on its tray, 1 cm up) | the cell's three testers in one scene; the robot at the drawn stop |
| Where it stands | ±1.5 cm on each axis of its station frame, turned ±10° (v4.0's machines) | `MachineLayout.place` |
| Clearance per side | uniform 0.25–0.5 mm | `NestLayout`: the pocket's geoms as `machine._add_pocket` draws them for that clearance (`nest_geoms` reads them from that very function), their bounding boxes, the pocket bodies' BVH; checked against a compiled nest, field by field (test) |
| Chamfer | uniform 0.5–1.0 mm | as the clearance (`pocket_straight`) |
| Camera error, position | normal, σ 0.8 mm per horizontal axis, clipped at 3 mm radially (mean 1.0 mm) | added to the true nest centre (`camera_belief`) |
| Camera error, yaw | normal, σ 0.5°, clipped at 2° | added to the tester's true yaw |
| Module in the hand | as the cell's pick leaves it: the module arrives on the infeed belt ±3 mm across it and ±5° (the test cell's placement noise) and the cell's controller picks it for that tester (`ScriptedTending.pick_for`, the keyed grasp) | `simulate_pick`; the grasp (the grasp assist's weld and the fingers) is stored in the episode |

The insertion's camera does not see the clearance, the chamfer or the module's place in the hand; the cell knows its testers' models (the nominal nest), which the scripts use.

**The module's place in the hand comes out almost exact.** The cell's pick lands the hand on the module in closed loop and the fingers centre and square it: over the sets' 1,400 picks the module sits in the hand within ±0.03 mm and 0.01° of the same pose (sd 0.013 mm along the fingers' faces, 0 across them), turned by the keyed grasp's quarter turns. The factor is drawn as the targets ask, but it adds next to nothing; the camera's error is the episode's real error. (Phase 0's probe saw more because it picked from the bench with ±15 mm, ±15° of jitter and the plain pick.)

**Streams.** The evaluation sets draw from `SeedSequence(seed, spawn_key=(0x1A5E,))` (seeds 20261127–29); the training pick pool from spawn key `0x1A5F` (seed 20261131); demonstrations from spawn key `0x1A60` (seed 20261132). Training draws its episodes from `env.reset(seed=n)`, gymnasium's plain `SeedSequence(n)` stream, which no spawn-keyed stream can reproduce, and its grasps from the pick pool (`InsertionConfig.pick_pool`, `configs/insertion_v1.yaml`; without a pool, reset simulates each episode's pick, about 0.7 s).

## What the robot observes (schema `insertion-v1-42d`)

Everything in the **believed nest frame**: origin the camera's estimate of the nest's centre on its floor (the floor's height as the cell has it programmed), axes along the believed nest.

| Field | Size | Why it is fair |
|---|---|---|
| joints q, velocities dq, commanded targets q_cmd | 21 | its own arm |
| TCP position and rotation (6-D) | 9 | forward kinematics, relative to where the camera puts the nest |
| the hand target (position) and its yaw relative to the aimed yaw | 4 | its own command |
| wrist force and torque on the held module | 6 | a wrist force-torque sensor: `control/scripted_insertion.part_wrench`, what the search reads (from every contact on the module but the fingers), plus 0.1 N and 0.002 N m of noise per axis per step |
| released | 1 | its own gripper command |
| elapsed share of the 30 s budget | 1 | its own clock (the budget ends the episode) |

**Not observed:** the nest's true pose, the module's place in the hand, the clearance and the chamfer. (The scripts' closed-loop place reads the module's true position to land it on the believed nest: `ScriptedCell.vertical(track=part)`, frozen; see "Design choices".)

## Actions (schema `insertion-v1-5d`)

Five numbers in [−1, 1]: a hand motion dx, dy, dz in the believed nest frame (up to 10 mm per step each: 0.2 m/s, the cell's vertical moves peak at 7.9 mm per step), a wrist turn dyaw (up to 0.05 rad per step; the search's wiggle needs 0.046), and **release** (above 0.5 the grasp assist lets go and the fingers open; it cannot be undone). They move a persistent **hand target** (kept within 3 cm of the measured hand on each axis and within 0.35 rad of the aimed yaw), which the cell's own IK solves to the arm's joint targets every step (`ScriptedCell._solve`: the same solver, tolerances and seed as the scripts), at the cell's 20 Hz (25 × 2 ms). The hand points straight down; tilting is not an action.

This is the form the scripts' own commands take: every control step they solve the arm for one hand position and yaw, and they open the fingers once. So a script's run converts into the env's (observation, action) pairs exactly (`InsertionEnv.action_from_command`; "Demonstrations" below).

## Reward (`InsertionRewardConfig`; terms in `info["reward_terms"]`)

Shaped toward seating gently and fast. The potentials use the true nest (training only; the policy never observes it):

| Term | Value | When |
|---|---|---|
| `insert` | +10 per m the module's bottom nears the nest's floor, +100 per m its centre nears the nest's centre (counted within 1 cm) | while it is held |
| `seat` | +3 | once, the first time it is seated |
| `release` | +1 if released seated, −2 if not | once, at the release |
| `retreat` | +5 per m the hand rises (up to the carry height) | after the release |
| `success` | +10 | once (the episode ends) |
| `force` | −0.05 per N of nest force above 20 N (the step's peak) | every step |
| `damage` | −10 | the nest force above 50 N (the episode ends) |
| `failure` | −5 | released, the hand back up, not seated (the episode ends) |
| `collision` | −0.05 | each step the arm (not the module) touches the tester |
| `time` | −0.01 | every step (−6 over the whole budget) |
| `action` | −0.001 × \|hand action\|² | every step |

Potential terms (insert, retreat) are paid only between two steps on the same side of the release, so letting go is not penalized by them. The force penalty starts well under the damage limit (20 N; the search presses with 3–15 N) so that the policy learns the trade-off before it ends an episode.

## The scripts on the same episodes

`evaluation/insertion.run_script` runs the frozen scripts of `control/scripted_insertion.py` from the env's start on the env's own controller (`InsertionController`: the cell's `ScriptedTending` with hooks; its physics step is `ScriptedCell.step`'s, bit for bit, test), through the cell's own control path, unchanged:
- **search**: `SearchInsertion()` with its frozen defaults, called as the cell's insertion slot calls it (`NestInsertion`): on the drawer, whose nest stands on its tray, through the cell's frame adapter `_NestFrame` (the force read from the real data).
- **straight-down**: `straight_down`, the cell's closed-loop place (`ScriptedCell.place`) straight down onto the believed nest, release, rise. In the slot the wrist is already turned to the slot; the place keeps that turn (a view whose carry yaw is the aimed yaw, `_Aimed`), as the cell's own straight-down (`ScriptedTending.load`) keeps it, and on the drawer it runs through the same `_NestFrame`.

The env watches every physics step and judges the end (`InsertionEnv.outcome`) exactly as it judges a policy. The scripts are not stopped at 30 s: the search's own 30 s timeout comes before its release and rise (about 3 s), so a search that seats in its last seconds ends past 30 s. `scripts/evaluate_insertion.py` reports both success as the script ends and success within the 30 s budget.

## Demonstrations

`evaluation/insertion.record_script` records a script's run in the env's spaces: before each control step, the action that moves the env's hand target to the script's command (the position and yaw it solved for) and releases if it has just let go; after it, the env's observation. `replay` steps those actions through the env from the same episode. `scripts/record_insertion_demos.py` does both on freshly drawn training episodes and writes the replayed episodes (observations as the env gives them, the recorded actions, rewards) to `runs/insertion_demos/` for behaviour cloning.

## Evaluation sets

Generated and committed before any training (`c109e72`, by `scripts/make_insertion_sets.py` at `893a952`); each episode carries the grasp its simulated pick leaves, and the digest covers it (`tests/test_insertion_sets.py`).

| Set | Episodes | Seed | Distribution | Digest | Testers A / B / C | Camera error, mean (max) |
|---|---|---|---|---|---|---|
| `insertion_dev_v1` | 400 | 20261127 | the targets' | `693e45d96e37` | 121 / 129 / 150 | 0.99 mm (2.56), 0.37° (1.52) |
| `insertion_final_v1` | 600 | 20261128 | the targets' | `0d1623e33b82` | 199 / 182 / 219 | 1.03 mm (2.70), 0.39° (1.43) |
| `insertion_stretch_v1` | 400 | 20261129 | camera σ 1.5 mm (clip 4 mm), σ 1° (clip 3°) | `9bd043477747` | 122 / 155 / 123 | 1.79 mm (4.00), 0.78° (2.71) |

Training pick pool: `data/handoff/insertion_picks_train_v1.json`, 1,000 picks, seed 20261131, digest `a047bf40b75c`.

`scripts/evaluate_insertion.py` refuses the final and stretch sets without `--one-shot`. Nothing has run on them.

## The scripts on `insertion_dev_v1`

Run once each at clean `c109e72` (`results/v4/insertion/search_insertion_dev_v1.json`, `straight_down_insertion_dev_v1.json`):

| | Straight-down | Search (frozen) |
|---|---|---|
| Success | 197/400, **49.2 %** (CI 44.4–54.1) | 330/400, **82.5 %** (CI 78.5–85.9) |
| Success within 30 s | 197 | 329 |
| Seated (whatever the force) | 266 | 330 |
| Over 50 N (every physics step / at control steps) | 184 / 124 | 1 / 0 |
| Time to success, median (p90) | 6.35 s (6.4) | 8.7 s (16.4) |
| Peak nest force, median / max | 43.5 / 202 N | 12.9 / 53.8 N |
| Failures | 184 damage, 19 not seated | 69 not seated, 1 damage |
| Lid / drawer / pneumatic | 75/121, 41/129, 81/150 | 103/121, 103/129, 124/150 |

The search by clearance: 67/96 at 0.25–0.31 mm, 70/94, 94/110, 99/100 at 0.44–0.5 mm; by camera error: 65/74 under 0.5 mm, 112/135, 101/126, 44/55, 8/10 at 2–3 mm. It seats on its first try in 154 episodes; the median time grows from 8.4 s under 1 mm of error to 13.9 s at 1.5–2 mm. Straight-down seats two thirds of the modules but presses most of them too hard: its closed-loop place pushes the module toward the believed nest, into the wall.

**The rule fixed in advance applies:** the search seats less than 85 % on `insertion_dev_v1` (82.5 %), so I3's "no more than 2 points lower" becomes "**not lower**" ([v4_1_targets.md](v4_1_targets.md), "Before any bar is final"; to be recorded there by the main session).

## Demonstrations: do they replay?

`scripts/record_insertion_demos.py --episodes 200 --pick-pool data/handoff/insertion_picks_train_v1.json` at `45e8d24`, on 200 freshly drawn **training** episodes (seed 20261132, spawn key `0x1A60`; `results/v4/insertion/demo_replay_search_200_20261132.json`):

- the search succeeded on its own control path in 158; replaying its recorded actions in the env succeeded in 155;
- **the replay ends as the script did in 197 of 200 episodes (98.5 %)**, and in **155 of 155 (100 %)** of the episodes whose script run ended within the env's 30 s budget. The 3 mismatches are searches that seated after 30 s (their own timeout comes before the release and rise), which the env's 600-step budget truncates;
- 156 replays are identical to the script's run (the same peak force and seat to the last digit): the env's control path is the script's, so the replay is exact up to the floating-point rounding of the actions, which did not change any outcome;
- no step needed more than one action (no clipping), and the lead bound never acted.

The 155 replayed successes (their observations, actions, rewards) are in `runs/insertion_demos/search_200_20261132.npz` (not committed; training will record its own, larger set the same way).

## Speed

One env step (50 ms simulated) takes about 6 ms on this machine (25 physics steps with the nest-force watch, the IK solve, the observation): random actions 6.1 ms per step, a policy rollout 6–8 ms; mj_step itself is about 4.5 ms of it. A reset with a pool grasp takes about 80 ms (layout, settle, the aim's turn); with a simulated pick about 0.7 s more. A plain SB3 PPO (2 envs in one process, n_steps 512) stepped the env for 4,096 steps in 34 s (8.2 ms per step including its updates); `check_env` passes.

## Round 2's options (`configs/insertion_v2.yaml`)

Round 1 (pure PPO, `configs/insertion_ppo_r1.yaml`) settled on hovering; the root causes and the round-2 plan the user approved are in [v4_1_insertion_r1_rca.md](v4_1_insertion_r1_rca.md). Round 2 adds four env options, all **off by default** (the defaults, `configs/insertion_v1.yaml`, are round 1's env exactly: `tests/test_insertion_r2.py` replays round 1's returns and observations to the last digit). Unchanged in every option: the episode distribution, the observation, the sensing, the success definition and the 50 N damage limit (checked every physics step), the 30 s budget, and the evaluation start. The frozen scripts never pass through the policy's action layer (they command the arm through their own control path), so none of this changes them.

- **Release interlock** (`action.release_window`, 3 mm): a release lets go only with the module's bottom within 3 mm of the nest's floor (12 mm and more inside the 15 mm pocket); elsewhere it is ignored, unpenalised, and counted (`stats["release_blocked"]`). The policy still decides when to let go. The floor's height is the cell's programmed one (the camera errs only horizontally) and the module sits in the hand within ±0.03 mm, so the cell knows this depth from its own kinematics; the env reads it from the true pose, the same number.
- **Compliant action layer** (`compliance`, `InsertionEnv.comply`): the learned skill's controller is an admittance on the hand target, as industrial insertion runs on impedance or admittance control (IndustReal, FORGE). Its reference is the **rest target**, the target that would hold the hand still where it is in free air (the measured TCP plus the arm's static deflection under its own and the module's weight, J (τ/kp): the arm's servos give ~5.5 mm under load; `rest_target`). Each control step:
  - near the nest (the module's bottom less than 15 mm above the rim, or inside) the target stays within 5 mm of the rest target on each axis, so the hand slows to about 50 mm/s before it can touch (the servos lag 0.1 s × speed; a touch-down at round 1's 0.2 m/s alone gave 35–80 N on the stiff nest);
  - in contact (the wrist sensor's force on the module above 2 N, along n) the target may lead the rest target into the contact by at most 1 mm, and above 10 N it backs off along n by 0.05 mm/N (at most 1 mm per step; the gain × the ~13 N/mm contact stiffness = 0.65, stable).
- **Start-height curriculum** (`curriculum`, training only; IndustReal's sampling-based curriculum): each training episode starts with the module's bottom at a height above the rim drawn uniformly from [`low`, the carry start's]. Below the rim the module starts engaged: on the true nest's axis with its key in the slot (it fits only there; no interpenetration: the tests check the nest force at the start, the offset within the clearance, the key and the tilt), still held; above it, over the believed nest as the aim leaves it (the camera's error kept). The arm is put there by the cell's IK and commanded at its rest target, so it does not sag from the start height. `low` starts 3 mm inside the rim and moves with the rolling training success over 100-episode windows: +5 mm above 80 %, −3 mm below 10 %, within [−8 mm, the carry start] (`training/callbacks.StartCurriculumCallback`, set in every training env, logged as `curriculum/low_mm` and to the run's `curriculum.jsonl`). `InsertionConfig.for_evaluation` turns it off: validation and evaluation start at the carry height. A start height can also be given (reset option `start_height`).
- **Keypoint progress reward** (`reward.progress: keypoints`): the held progress term is the change of a potential on the mean distance d of the module's 4 bottom corners from where they sit seated (true nest, key in the slot): φ(d) = Σ wᵢ (1 − tanh(d/sᵢ)), scales 100, 10 and 1 mm, weights 5, 1, 5 (IndustReal / Isaac Lab's multi-scale keypoint reward): about +5.8 for the approach from the carry height to 2 mm over the floor and +5.1 for the last 2 mm. Potential-based and paid only while held, so it cannot pay for dawdling (a positive per-step reward would, with episodes ending at success). Force penalty −0.02 per N per step above 15 N (round 1: −0.05 above 20 N); time −0.005 per step (round 1: −0.01); the arm touching the tester −0.02 per step (round 1: −0.05; set by the gate: untrained noise from near-nest starts drifts the hand into the testers' roofs and walls for ~240 steps); no penalty for an early release (the interlock); seat, success, damage and failure terms unchanged.

**A trained run is evaluated with its own action layer** (`evaluation/insertion.policy_env_config`: the run's `env_config.yaml`, curriculum off, checked to keep the targets' distribution, sensing, success, start and budget); the scripts keep the default env.

**The gate before round 2's training** (`scripts/insertion_reward_gate.py`): see [v4_1_insertion_r1_rca.md](v4_1_insertion_r1_rca.md), "As built".

## Design choices to check

- **Straight-down is the cell's place keeping the slot's turn.** The frozen `straight_down` turns the wrist to the carry yaw (it was written for the probe's unkeyed nest). In the insertion slot the wrist is already turned to the key; `_Aimed` keeps that turn, as the cell's own straight-down (`ScriptedTending.load`) does; without it the drawer's module would be turned back by 90°. Nothing in the script changes.
- **The start includes the cell's aim** (the keyed turn over the believed nest), not timed, the same for all three. The insertion starts as the aim's last step ends, the arm still settling (up to ~3 mm and ~3° off for a moment), as in the cell.
- **The nest force is checked every 2 ms and includes the nest's floor.** Stricter than the cell's and the probe's (every 50 ms, walls only). It decides straight-down's numbers (sampled at control steps it would pass 244, not 197) and not the search's. Chosen because a learned policy can find short impacts a 20 Hz sample misses.
- **The scripts' time is not cut at 30 s.** The search's own 30 s timeout comes before its release and rise; 1 dev success ended past 30 s. Both counts are reported; the policy's episode ends at 600 steps.
- **The module's place in the hand is nearly exact** (±0.03 mm): drawn as the targets ask, but it adds no error. The scripts' closed-loop place reads the module's true position (`ScriptedCell.vertical(track=part)`, frozen) while the policy does not observe it; with the offset this small the asymmetry does not matter.
- **Testers are moved ±1.5 cm and turned ±10°** ("drawn per tester as in v4.0"); the cell itself has them at nominal places. The camera's error is what the insertion sees either way.
- **Action scale.** 10 mm and 0.05 rad per step at full action, so the scripts' fastest moves fit one step and their demonstrations convert exactly. For precise work the policy's exploration noise must be small (a log std near −2 to −3 after cloning); the next phase chooses it.
- **The wrist sensor** reads `part_wrench` plus 0.1 N / 0.002 N m of noise, and zero once the module is released; the scripts read it without noise (frozen).
- **Drawers are in** (tester B, with the cell's index pin): the nest 1 cm higher and the hand turned 90°. The search runs there through the cell's `_NestFrame` as in the cell.
- **Damage ends the policy's episode** at once (its outcome is decided); the scripts run to their end and are judged after.
