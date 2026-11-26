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
