# v4.1 insertion, round 1: why pure PPO collapsed, and what round 2 changes

*2026-11-25. Round 1 (`configs/insertion_ppo_r1.yaml`, pure PPO from scratch, launched at `d657eb7`) was stopped by the user at about 200–250k of 3M steps per seed, after every seed had settled on the same behaviour. This note is the root-cause analysis the user asked for, the research behind the fix, and the round-2 plan the user approved. Round 1 counts as one of the 3 development rounds of [v4_1_targets.md](v4_1_targets.md).*

## What happened

| Seed | Validation at 100k | At 200k |
|---|---|---|
| 0 | 0/60; holds the module above the nest, never touches (all 60 run out of time; peak nest force 0) | the same |
| 1 | 0/60; goes down: 20 damage (> 50 N), 39 run out of time, 1 not seated | 0/60; now hovers like seed 0 |
| 2 | 0/60; hovers | (stopped before 200k) |

Training curves (all seeds alike): episode length rose from ~30 to ~580 steps and the mean return from −25 to about −7, the return of hovering for the whole budget. Success happened once, by chance, in the first 4k steps (1.2 %), then never.

## Measurements (`~/scratch/claude/rca.py`, 24 training-stream episodes, the round-1 reward)

**What each kind of behaviour earns:**

| Behaviour | Mean return | Outcome | Largest terms |
|---|---|---|---|
| The frozen search (the expert) | **+12.1** | 19/24 seated | success +10, insert +3.0, seat +2.4 |
| Hover, never release | −5.9 | never touches | time −6.0 |
| Straight down, keep pressing | −11.3 | 24/24 damage | damage −10, force −3.5 |
| Untrained policy (round 1's initial noise) | −24.9 | 16 damage, 8 not seated | force −14.1, damage −10, failure −5, release −2 |
| Down until touching, then hold still | **−74.2** | 8 damage, 16 out of time | force −69.3 |

**Untrained exploration:** with the hand's noise at round 1's initial std (0.1 of full scale, about 1 mm per step), the module never came within 257 mm of the nest in 600 steps. With the release's noise (std 0.3 on a threshold at 0.5), the untrained policy let go after a median of 12 steps, dropping the module from ~25 cm: median peak 297 N on the nest, up to 720 N.

**What the checkpoints learned:** a deterministic mean of +0.05 to +0.18 on dz (upward) in every seed; release mean −0.09 to −0.17 with its log std falling from −1.2 to about −1.6; the hand's log std unchanged at −2.3.

**Optimisation:** approx KL 0.02–0.06 and clip fraction 0.30–0.39 throughout (healthy PPO is about 0.01–0.02 and below 0.2); entropy coefficient 0; explained variance 0.7–0.96 (the critic learned the hovering value well).

## Root causes

1. **A valley around the safe behaviour.** The reward ranks success highest (+12 against −6 for hovering: the environment and the reward are not wrong), but every unskilled way of touching the nest scores far below hovering (−11 to −74). The worst is the most natural first attempt, going down and holding at contact: the arm is position-controlled and stiff, moves up to 10 mm per step, and by the time it touches its target is several millimetres into the rim, so it keeps pressing at 30–50 N and the per-step force penalty charges for every step of it. The only way to a positive return is a long, precise sequence (~200 steps) that exploration does not produce.
2. **Exploration never reaches the nest.** About 1 mm of random motion per step random-walks; it does not descend 25 cm. The reward for descending (+10 per m, about +0.01 per mm) is the size of the time penalty per step and is drowned by the large, noisy penalties of early releases and crashes.
3. **The release poisons early learning.** As a threshold on a noisy action, it fires within about 12 steps, from height. The drop alone breaks the 50 N limit. PPO's first strong lesson is therefore "never let go", its next "stay away" (dz drifts upward).
4. **PPO too aggressive, nothing keeping exploration alive.** Learning rate 3e-4 with 10 epochs gave KL and clip fractions about twice the healthy range; with no entropy bonus the policy locked onto the first safe behaviour within ~100k steps and its noise only shrank.

Ruled out: a bug in the environment (the search runs through the same environment and actions and scores +12 with 19/24 seated; recorded search actions replay to the same outcome), the observation or its normalisation (the critic fits well), the action scaling.

## What the research says (a research report kept outside the repo, `research_rl_insertion_industry.md`, with its sources)

- The one well-documented recipe in which pure PPO from scratch learns insertion is NVIDIA's simulation line (Factory, IndustReal, Isaac Lab's `PegInsert`, FORGE). It works because the setup removes the traps above: the part starts a few centimetres above the hole (IndustReal's start-height curriculum begins inside it and widens with success); the gripper never opens (no release decision in exploration; FORGE learns a separate "done" output); the reward is a squashed keypoint distance, strongest in the last millimetres; an impedance controller makes damaging forces hard to reach; and the budget is millions of steps. IndustReal reports insertion at about 12 % without these components.
- Even NVIDIA's own broader work leans on more than pure RL at scale (AutoMate: 38 % pure RL against 81.5 % with a disassembly-derived imitation reward), and industrial manipulation (Tesla, Figure, Boston Dynamics/TRI) is trained mainly from teleoperated demonstrations; HIL-SERL reports 0 % from scratch on every task. Staying pure RL, as the user chose, is supported only with that scaffolding.

## Round 2 (approved by the user, 2026-11-25): still pure PPO

| Root cause | Change | Source |
|---|---|---|
| 2, exploration never reaches the nest | **Start-height curriculum, training only:** starts drawn between slightly inside the nest and the full carry height; the lowest start rises 5 mm when training success is above 80 %, falls 3 mm below 10 % (the full range is always sampled). Evaluation keeps the targets' start. | IndustReal |
| 3, the release | **Release interlock:** a release takes effect only within ~3 mm of seated; elsewhere it is ignored, unpenalised. The policy still decides when to let go. | Isaac Lab, FORGE |
| 1, the valley | **Compliant action layer:** above a contact force threshold the hand target stops advancing into the contact and backs off (admittance), so a naive "down and hold" gives ~10–20 N, not 200 N. The 50 N damage check is unchanged. | IndustReal, FORGE |
| 1 and 2, guidance too weak | **Reward:** progress (potential-based) on a squashed 4-keypoint distance to the seated pose, ~+5 for the approach and ~+5 for the last 2 mm; force penalty only above 15 N at −0.02/N per step (was 20 N, −0.05); time −0.005 per step (was −0.01); no penalty for an early release (the interlock); seat and success bonuses unchanged. Not a positive per-step reward: with episodes ending at success it would pay the policy to dawdle, against the speed target. | IndustReal / Isaac keypoint reward; Ng et al. potential shaping |
| 4, optimisation | **PPO:** learning rate 1e-4, 5 epochs, target KL 0.02, entropy 0.001, gamma 0.995, hand noise ~2 mm per step (safe once contact is compliant), release noise std ~0.37. | Isaac Lab settings |

**Gate before any training:** the round-1 measurements above are repeated under the round-2 environment; "down until touching, then hold" must now score clearly above hovering, untrained exploration must no longer drop the module, and the search must still score highest. **Then a one-seed pilot** (1M steps) with a go/no-go on contact, first seats and a rising success rate, before the three seeds.

Unchanged: the targets, the evaluation sets and their start, the success definition and the 50 N limit, the frozen scripts.

## As built (2026-11-26, branch `v41-insertion-r2`, before any round-2 training)

Code and design: [insertion_env.md](insertion_env.md), "Round 2's options"; env `configs/insertion_v2.yaml`, PPO `configs/insertion_ppo_r2.yaml`. Every round-2 option is off by default: `configs/insertion_v1.yaml` is round 1's env exactly (a test replays round 1's returns and observations to the last digit).

**As planned:** release interlock 3 mm from the floor (no penalty, `release_unseated` 0); compliance threshold 10 N, lead 1 mm, back-off 0.05 mm/N; curriculum from 3 mm inside the rim, +5 mm above 80 % / −3 mm below 10 % training success (100-episode windows), the full range always sampled, deepest −8 mm, training only; time −0.005 per step; seat and success bonuses unchanged; PPO lr 1e-4 (constant), 5 epochs, target KL 0.02, entropy 0.001, gamma 0.995, log std −1.6 (hand) / −1.0 (release), 8 envs, validation as round 1 at the carry start.

**Changed or added by the gate** (each because a measured behaviour showed the trap the plan meant to remove):

| What | As built | Why (measured) |
|---|---|---|
| Compliance reference | the arm's **rest target** (measured TCP + the servos' static give under the weights, J τ/kp) | the arm hangs ~5.5 mm below its target; a lead measured from the hand allowed ~85 N |
| Slow zone | module's bottom < 15 mm above the rim (or inside): target within 3 mm of the rest target (hand ≤ ~30 mm/s) | a touch-down at 0.2 m/s gave 35–80 N before the 20 Hz layer could act |
| What the layer senses | the **external force and twist on the whole robot and the module** (as the Panda's joint-torque estimate), with a yaw admittance (0.2 / 0.5 N m, 0.5 mrad lead) | untrained noise near the nest pushed the hand into the testers' roofs with 100–300 N; the wrist is ~2000 N m/rad stiff in yaw |
| Contact reflex | between control steps, every physics step of a policy's step: external force > 12 N or twist > 3 N m holds the arm at its rest pose until the next control step | a contact met at speed built up force for up to 50 ms |
| Progress term | φ(d) = 4(1−tanh(d/100 mm)) + 2(1−tanh(d/30 mm)) + 5(1−tanh(d/1 mm)) (planned 5/1/5 at 100/10/1 mm); paid on **new bests** of φ only (`keypoints_best`) | +5.9 for the approach and +5.0 for the last 2 mm as planned, +1.5 for touching down from 2 cm over the rim; with the plain difference untrained noise near the nest lost ~1.5 (d's random walk drifts outward). A straight approach earns exactly the potential difference, so dawdling is still not paid |
| Force penalty | −0.02 per N per step above **20 N** (planned 15 N) | sustained pressing is held near 12 N by the reflex, so the penalty only charges impacts; from 15 N it charged noise near the nest −0.5 for 15–25 N touches |
| Arm on the tester | −0.02 per N of its push above 20 N (round 1: −0.05 per step of any touch) | noise near the nest brushed the testers' roofs for ~240 steps (−12) |
| Action penalty | −0.0002 × \|a\|² (round 1 −0.001) | −0.1 of pure noise per episode at std 0.2 |

**The gate** (`scripts/insertion_reward_gate.py`, the same 24 training-stream episodes, seeds 900000–900023; mean return, successes, damage episodes (> 50 N), median peak nest force; "near" = start heights drawn 3 mm inside to 20 mm above the rim, the curriculum's first range). Round 1's carry-start numbers reproduce this note's table above.

| Behaviour | Round 1 env, carry start | Round 2 env, carry start | Round 1 env, near start | Round 2 env, near start |
|---|---|---|---|---|
| The frozen search (its own control path) | +12.13; 20/24; 0 dmg; 10.5 N | **+16.74**; 20/24; 0; 10.5 N | +3.17; 13/24; 0; 13.0 N | **+6.98**; 13/24; 0; 13.0 N |
| Hover, never release | −5.91 | −3.00 | −6.00 | −2.97 |
| Untrained noise, round 1's (std 0.1, release 0.3) | −24.92; 16 dmg; 297 N; all drop the module | −2.99; 0 dmg; no release | −36.04; 10 dmg | −2.78; 0 dmg; 13.2 N |
| Untrained noise, round 2's (std 0.2, release 0.37) | −22.30; 10 dmg | −2.98; 0 dmg; no release | −28.43; 19 dmg | −2.91; 0 dmg; 18.2 N |
| Down until touching, then hold | −74.15; 8 dmg; 46.5 N | **+1.39**; 0 dmg; 9.1 N | −47.48; 3 dmg | −2.07; 0 dmg; 7.7 N |
| Straight down, keep pressing (never release) | −11.30; 24 dmg; 70 N | +3.31; 0 dmg; 11.2 N; 6 seated | −94.97; 18 dmg | +2.07; 0 dmg; 9.7 N; 14 seated |

- **Met:** the search scores highest from both starts; untrained exploration never lets go away from the floor and never breaks 50 N, and from near-nest starts scores at or above hovering (−2.78 / −2.91 against −2.97); down-and-hold scores above hovering (+4.4 at the carry start, +0.9 near the nest, where little distance is left to gain); no naive behaviour damages the nest.
- **Pressing and waiting** (straight down, never releasing) scores above holding because it seats the module (6 and 14 of 24) and collects the seat bonus; its median peak force is 9.7–11.2 N, under the 12 N reflex and the 20 N penalty start, so there is no "press hard" to learn. It still forgoes the success bonus (+10, plus the release and retreat): with the interlock a release near the floor is the next step, and the release noise (std 0.37) requests one on ~9 % of steps.
- **The search's recorded actions replayed through the round-2 action layer** fail (0/24 at the carry start): the layer is the learned skill's controller, not the scripts'; the frozen search keeps its own control path (and is scored on it).
- **The curriculum's share of near starts:** drawn uniformly from 3 mm inside the rim to the carry start (~+190 to +250 mm), about 10 % of early training episodes start within 2 cm of the rim; most start high, where the approach is learned. This is the approved rule (IndustReal's); the pilot shows whether it is enough.

### Pilot 1 and release-and-retract (2026-11-26)

**Pilot 1** (`runs/insertion_ppo_r2_pilot1_stopped_seed0`, one seed, stopped by the user at ~300k of 1M): validation at 100k and 200k 0/60 successes, 0 seated, contact 21 → 41 of 60, no damage, every episode `not_released`; the curriculum went to its deepest start (−8 mm) by 125k with training success 0. From −8 mm a scripted gentle push, release and lift succeeds 12/12 (the env is sound); the 250k policy never releases deterministically, and stochastically 8 of its 12 releases ended in **damage**: having learned to press down while holding, it kept pressing with the open hand after the release and drove the fingers into the freed module (> 50 N). Letting go looked dangerous, so it was suppressed.

**Fix (approved by the user, before the 3 seeds): release-and-retract** (`action.release_retract`, off by default, on in `configs/insertion_v2.yaml`). A release that takes effect (inside the interlock's 3 mm window) opens the fingers and retracts the hand straight up to the carry height as one built-in motion, the industrial release-and-retract macro: the cell's own (`grip_steps` for the fingers, then `ScriptedCell.vertical` at the cell's speed), as the frozen search's retreat runs it. The episode then ends, judged as always (seated, released, hand clear, no damage). The policy still chooses where to seat and when to let go. The retract's control steps count in the episode's time (and in the time penalty); the nest-force watch runs through it; the post-release `retreat` shaping no longer pays anything under it; seat and success bonuses unchanged.

**The gate after it** (24 episodes each; the carry and near tables above are unchanged, the search runs its own path): from the curriculum's **deepest start (−8 mm, 7 mm over the floor)**:

| Behaviour | Return | Success | Damage | Peak nest force, median | Time to success, median |
|---|---|---|---|---|---|
| Press down (full speed), a release asked every step (the pilot's habit) | +14.18 | **24/24** | 0 | 3.2 N | 3.75 s |
| Gently down until seated, hold 5 steps, release | **+17.86** | **24/24** | 0 | 5.2 N | 4.1 s |
| Gently down until seated, hold, never release | +4.89 | 0 (24 seated) | 0 | 5.4 N | |
| Untrained noise around a downward push (round 2's noise, release noise included) | +14.87 | 24/24 | 0 | 17.5 N | 5.4 s |
| Hold still, a release asked every step (outside the window) | −2.97 | 0 (600 blocked per episode) | 0 | 0 | |
| Untrained noise (round 2's) | −1.53 | 3/24 | 0 | 23.3 N | |
| Hover | −2.97 | 0 | 0 | 0 | |
| The frozen search (own path) | +15.55 | 24/24 | 0 | 12.1 N | |

The first pilot's 250k policy from the same −8 mm starts (seeds 700000–700011, as the main session's probe): stochastic **8/12 succeed, 4 not released, 0 damage** (peaks 15–31 N; before the fix 8 of 9 releases were damage); deterministic 0/12 (it never releases: its release mean is what pilot 1 learned).

**What it costs on time:** the retract (0.6 s for the fingers, then ~0.3 m at the cell's 0.1 m/s) is about 3.4 s of every success, as it is of the search's (whose 8.7 s median on `insertion_dev_v1` includes its own release and rise). I3 asks for a median time to success at least 40 % shorter than the search's (about 5.2 s): the learned skill has about 1.8 s left for the approach (≥ ~1.25 s at the 0.2 m/s step limit, plus the slow zone's last ~3 cm at ~30 mm/s) and the seat. The comparison stays fair (the same retreat for both); whether the bar is still reachable is a question for the user (a faster built-in retract would favour the policy over the frozen search's retreat).

### Pilot 2 and the changes before pilot 3 (2026-11-26; the user's option A, still pure PPO, still round 2)

**Pilot 2** (`runs/insertion_ppo_r2_pilot2_seed0`, 1M steps, NO-GO): validation 0/60 at every 100k, 0 seated, contact 1–59 of 60; the curriculum's low bound sat at −8 mm from 124k on with training success 0–4 %. The main session's probe: success 0/12 from −8, 0, +5, +20, +50 mm and the carry start, with 190–300 blocked release requests per episode. Two causes:
- **The curriculum** (IndustReal's rule, uniform from `low` to the carry start): only ~5 % of episodes started near the nest, and its +5 mm step (above 80 % success) could never fire.
- **The policy pulled out of the nest.** Its final policy from −8 mm (12 episodes, our probe with diagnostics): mean action dz **+0.10** and dx −0.1 per step: it rises out of the pocket (closest 2–7 mm over the floor early on) and drifts 30–50 cm away, with the old and the new layer alike (1–2 of 12 succeed). With `keypoints_best` progress, leaving cost nothing, while staying near the nest risked force and contact penalties. (The main session's reading, a stall against the walls, was tested too: scripted pushes down with the policy's 2 mm lateral noise and a release asked every step succeed 12/12 from −8 mm with the old layer as with the new one.)

**Changes for pilot 3:**

| What | As built | Why |
|---|---|---|
| Progress | back to the **plain potential difference** (`keypoints`), φ unchanged | leaving the nest must cost what entering earned (pilot 2). The price: untrained noise near the nest is again charged for d's outward drift (−1.2 near-nest; offset there now by its own successes) |
| Compliance along the insertion axis (`compliance.axis_split`, user's item 1) | across the axis (x, y, yaw) the admittance as before; along it the policy may always push: the target stops advancing into a vertical contact at **15 N** (no back-off), backs off only above **30 N**; the reflex acts on **20 N across**, 3 N m twist, or **35 N along** (was 12 N on the whole force) | the whole-force layer backed off along the walls' push and its 12 N reflex held the arm whenever the module rubbed a wall; pushing down inside a tight nest needs the axis free (force-limited). The slow zone stays: it limits speed (~30 mm/s over the last ~3 cm), not progress (the scripted pushes seat through it) |
| Curriculum (`curriculum.mode: reverse`, user's item 2) | a **reverse curriculum** (Florensa et al.): starts uniform from the fixed deepest start (−8 mm) to `high`, which starts at +5 mm and grows ×1.5 above 70 % success of the curriculum episodes, shrinks ÷1.5 below 20 % (floor 5 mm), capped at the carry start; **15 %** of episodes from the carry start (not in the rule). `ReverseCurriculumCallback` logs `curriculum/high_mm`, the curriculum and carry success, and success per start-height bin (inside, 0–5, 5–20, 20–50, 50–100, > 100 mm, carry) to TensorBoard and `curriculum.jsonl` | about 85 % of early episodes now start within 5 mm of the rim or inside; ~9 raises reach the carry start |

**The gate before pilot 3** (`28802dc`; 24 episodes each; the search on its own path):

| Behaviour | Carry start | Near start | −8 mm | +2 mm |
|---|---|---|---|---|
| The frozen search | **+16.71**, 20/24 | **+6.91**, 13/24 | +15.55, 24/24 | −1.17, 6/24 |
| Hover | −3.00 | −2.97 | −2.97 | −2.97 |
| Untrained noise, round 2's | −3.01, 0 dmg | −2.95, 2/24 succeed, 0 dmg | +6.91, 15/24 | |
| Untrained noise, round 1's | −3.00 | +0.04, 5/24 | | |
| Down until touching, hold | +1.39 | −2.14 | | |
| Straight down, keep pressing | +2.89 (5 seated), 10.0 N | +1.59 (14 seated), 9.3 N | | |
| Push down with lateral noise (2 mm/step), release asked every step | | | +14.13, **24/24**, 0 dmg, 10.8 N, 3.9 s | +11.50, **20/24**, 0 dmg, 15.8 N, 5.4 s |
| Gently down, seated, hold 5 steps, release | | | +17.38, 24/24, 4.1 s | |
| Seated, hold forever | | | +3.94 (24 seated) | +1.00 (12 seated) |
| Press, release asked every step | | | +14.17, 24/24 | |
| Untrained noise around a downward push | | | +16.37, 24/24 | |
| Hold still, release asked every step (outside the window) | | | −2.97, all blocked | |

No damage in any row. The search scores highest from the targets' carry start and from near-nest starts; from engaged starts (−8 mm, +2 mm) a direct push-and-release beats it (the search lifts the module back to its own hover height first). Hover stays below down-and-hold (+4.4 carry, +0.8 near). Pilot 2's final policy from −8 mm under the new layer: 1/12 stochastic, 2/12 deterministic (it still rises out: a learned habit, not the layer).

### Pilot 3 and the release fixes before the 3 seeds (2026-11-26; user-approved)

**Pilot 3** (`runs/insertion_ppo_r2_pilot3_seed0`, stopped by plan at ~520k; checkpoints 250k and 500k) **learned the insertion**: the reverse curriculum's high bound climbed from 5 to 300 mm (the carry start); training success in the last quarter 74 % from the carry start, 80–100 % from lower starts, no damage. But deterministic validation (carry start) seated 0, 0, 14, 45 of 60 at 100–400k and **released 0**. The main session's diagnostic (500k, carry start): stochastic 19/20 success (median 10.45 s); deterministic 20/20 seated, 0 released, its release mean while seated about 0.045 (p90 0.25, max 0.46) against the 0.5 threshold. Seated, the potential is flat, so nothing pushed the release's mean over an off-centre threshold (research note `research_rl_release_decision.md`, kept outside the repo: Isaac Lab and robosuite put a gripper's threshold at 0).

**Fixes** (round 1's defaults unchanged):

| What | As built | Why |
|---|---|---|
| Release threshold | `action.release_above` **0.0** (round 1: 0.5) | the midpoint of the action; at 0.5 pilot 3's mean release seated was ~0.045 |
| Waiting seated | `reward.seated_wait` **0.03** per step seated and still held | a gradient toward letting go: 1 s seated costs 0.6 against the release's +1 and the success's +10. It accrues only while seated, so a release before seating gains nothing from it (and forgoes the last millimetres' potential) |
| Steps | **2M** per seed (`configs/insertion_ppo_r2.yaml`; was 5M) | pilot 3 reached 74 % training success from the carry start by ~450k; the machine is shared by three seeds |
| Evaluation | `scripts/evaluate_insertion.py`: **deterministic stays binding**; `--stochastic` (seeded per episode, its own file `_stochastic`) reported beside it; `--auto-release` a **side ablation** (`action.auto_release`: the cell releases on seated detection, the industrial PLC way; its own file `_autorelease`), never in training or a binding result | the targets' result is the deterministic policy's, as declared before training |

**DIAGNOSTIC (not a result):** pilot 3's 500k policy, its own env, the carry start, seeds 700000–700019:

| Variant | Success | Seated | Time to success, median | Peak force, median / max |
|---|---|---|---|---|
| Deterministic, threshold 0.5 (as trained) | 0/20 (20 not released) | 19 | | 15.9 / 26.4 N |
| **Deterministic, threshold 0** | **19/20** | 19 | 8.0 s | 15.3 / 26.4 N |
| Stochastic, threshold 0 | 20/20 | 20 | 9.5 s | 20.3 / 34.1 N |
| Deterministic, threshold 0.5, auto-release ablation | 19/20 | 19 | 7.95 s | 15.3 / 26.4 N |

The threshold alone closes the gap for this policy (its mean release seated is above 0 on most steps).

**The gate after the fixes** (`da2df0a`; 24 episodes each; the search on its own path; no damage in any row):

| Behaviour | Carry start | Near start | −8 mm | +2 mm |
|---|---|---|---|---|
| The frozen search | **+16.71**, 20/24 | **+6.91**, 13/24 | +15.55, 24/24 | −1.17, 6/24 |
| Hover | −3.00 | −2.97 | −2.97 | −2.97 |
| Down until touching, hold | +1.39 | −2.88 | | |
| Straight down, keep pressing (never releases) | +1.17 (5 seated) | −7.61 (14 seated; waiting −9.2) | | |
| Untrained noise, round 2's | −3.01 | −2.99, 2/24 | +6.76, 15/24 | |
| Gently down, seated, hold 5 steps, release | | | **+17.20**, 24/24, 4.1 s | **+7.02**, 12/24 |
| **Seated, hold forever** | | | **−13.91** (waiting −17.9) | **−5.99** |
| **Release asked from the start, takes effect in the window before seating** | | | +14.17, 24/24 | +6.66, 13/24 |
| Push down with 2 mm lateral noise, release asked every step | | | +14.13, 24/24 | +11.50, 20/24 |
| Press, release asked every step | | | +14.17, 24/24 | |
| Untrained noise around a downward push | | | +14.73, 24/24 | |
| Release asked outside the window | | | −2.97, all blocked | |

Holding seated forever now loses clearly to releasing (−13.9 against +17.2 from −8 mm; −6.0 against +7.0 from +2 mm); a release in the window before seating still loses to seating first (+14.2 against +17.2; +6.7 against +7.0: it forgoes the last millimetres' potential, and from +2 mm it drops unaligned modules). Down-and-hold stays above hovering (+4.4 at the carry start; +0.1 near the nest, where the waiting cost now charges its occasional seat).
