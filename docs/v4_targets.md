# v4.0 targets, set before any v4.0 training

v4.0's headline is a learned door skill: open and close the sliding lid of a benchtop machine the robot has never seen, with no reprogramming ([v4_plan.md](v4_plan.md), section 3.2; the env in [door_env.md](door_env.md)). Phase 1 measured the scripts first ([v4_plan.md](v4_plan.md), section 4):
- A camera fixes everything that can be seen: where the machine stands, how it is turned, where the handle is.
- The camera-aware script fails, every time, only on what a camera cannot see: which way the lid slides, and how hard the latch must be pushed.

These targets follow the project's usual rules:
- They are committed before the work they measure.
- The evaluation sets are generated and committed before training, and training never sees them.
- There are 3 training seeds, each judged separately.
- Results use 95 % Wilson intervals, and exact McNemar tests on paired episodes.
- Scripted baselines are run on the same episodes.
- Every result is reported as it comes, misses included.

## Decisions behind them (the user, 2026-11-23)

- **The bar's angle is observed** (`bar_align`). A camera sees which way the handle bar runs. That tells a sideways lid from an outward one, but not left from right, nor the travel, spring or latch.
- **Closing springs are included:** half the machines have one, with a soft-close damper and a hold-open detent (`5e49427`). On a springy lid, `open` succeeds only with the lid in its detent.
- **Demonstrations first, with a plain PPO run as a reference.** The policy starts by copying a script, then PPO refines it. A PPO-from-scratch run is trained beside it and reported whatever it shows.

## The baselines: three door scripts, run in the door env on the same episodes

All three act through the env's own 5-D actions, as the policy does, and read only what the policy observes, plus the nominal machine's settings where stated. They hook the bar as the expert does (`control/scripted_door.py`).

| Script | What it believes | What it does when the machine differs |
|---|---|---|
| **Nominal** | the nominal machine at its nominal place: handle where the nominal machine has it, slides right, 0.12 m travel, latch at 0.02 m/s, no spring | nothing: it is the script written for one machine |
| **Camera-aware** | the handle where the camera sees it, the slide's axis from the bar's angle (outward if the bar runs sideways; otherwise right, the nominal sense); nominal travel, latch and spring | opens to the believed travel less 5 mm, as the line's script does |
| **Feel** (the fair baseline, and the demonstration source) | as camera-aware | pushes the believed way and, if the bar does not move (force above a threshold with no hand travel), tries the other way. It opens until the lid stops against its stop, which also finds a springy lid's detent. It closes, and if the machine's latched signal does not come, pulls the lid back open and pushes again faster, up to 2 retries |

- **The privileged expert** reads the truth, such as the slide direction. It is reported as an upper reference, never as a baseline.
- **Why the demonstrations come from the feel script, not the expert:** the expert pushes the right way from the first step, using information the policy cannot observe. A policy copying it would have to guess left or right, and would learn to average the two. The feel script decides from what the policy sees, so it can be copied.
- **Built and committed before training:**
  - the three scripts;
  - their results on the development set;
  - a check that the feel script succeeds on at least 90 % of the development set, so its demonstrations are worth copying. If it does not, the script is fixed and that is reported, before any demonstration is recorded.

## Evaluation sets, generated and committed before any training

Each episode is a full `draw_episode` record: command, machine variant, lid start, arm jitter and camera errors. It is replayed exactly through the reset option `episode`.

| Set | Contents | Seed | Used for |
|---|---|---|---|
| `door_dev_v1` | 400 episodes (200 open, 200 close), the training ranges; digest `f865b89d2df1` | 20261124 | development: comparing rounds, baselines, the feel-script check |
| `door_final_v1` | 400 episodes, the training ranges; digest `1bf1bd3f850f` | 20261125 | the one-shot result, run once |
| `door_stretch_v1` | 400 episodes, ranges beyond training: handle ±4 cm along the bar and ±3 cm along the slide, turn ±15°, latch to 0.15 m/s, spring to 50 N/m, friction to 15 N (outward lids keep their handle ranges: the arm's reach); digest `8f64548983f8` | 20261126 | reported, no bar: how far it extrapolates, against the scripts |

Every machine in these sets is new to the policy: training draws a fresh variant every episode from a different random stream.

```bash
uv run python scripts/make_door_sets.py --name door_dev_v1 --seed 20261124
uv run python scripts/make_door_sets.py --name door_final_v1 --seed 20261125
uv run python scripts/make_door_sets.py --name door_stretch_v1 --seed 20261126 --stretch
```

## Targets (on `door_final_v1`, each seed separately unless stated)

| # | Check | Target |
|---|---|---|
| T1 | Success | at least **95 %** for every seed; and at least **90 %** in every cell of command × slide direction × spring or none, for every seed |
| T2 | Against the camera-aware script | significantly better (exact McNemar, p < 0.05), pooled |
| T3 | Against the feel script | success **no more than 2 points lower**, pooled; **and faster:** median time to success at least **20 % shorter**, pooled, on the episodes both complete |
| T4 | Gentleness | median peak force no higher than the feel script's, pooled; slams in at most 1 % of episodes for every seed |
| T5 | In the line (reported, no bar) | the learned door inside the full scripted tending cycle on the line's M1, on 100 machines from `door_final_v1`'s variants: machine running at the end, against the camera-aware and feel scripts. This needs the hand-motion adapter to account for the rail first ([door_env.md](door_env.md), open questions) |

- **What a miss means:**
  - **T1 or T2 missed:** the skill does not work on unseen machines.
  - **T3 missed:** a well-written script with simple feel logic does the job as well. That would be reported as the finding, with where each wins.
- **Why T3 asks for speed:** the feel script will likely succeed on nearly everything, because it can try both ways and retry the latch. Where a learned policy can plausibly beat it is in not wasting moves. It can learn to tell the slide direction within a few millimetres, and to close a lid firmly the first time.

## Training rules

- **Demonstrations:** the feel script on freshly drawn training machines, recording observations and actions. These are machines from the training stream, never from the evaluation sets.
- **Behaviour cloning** of those demonstrations first, then **PPO fine-tuning** from the cloned policy. The cloned policy's own result on `door_dev_v1` is reported too.
- **PPO-from-scratch reference:** `configs/door_v1_ppo.yaml`, 3M steps, 1 seed, launched right after this commit. Reported as it comes; it is not a candidate for the final policy.
- **Rounds:** at most 4 development rounds on `door_dev_v1`. Each round's config is committed before it trains.
- **The policy used for a seed** is its best validation checkpoint (mixed open and close on randomized machines), the rule fixed here, before any training.
  - **Changed by the user on 2026-11-23, after dev round 3 and before any seed 1 or 2 result:** the policy used for a seed is the **final checkpoint of its last round** (`final_model.zip`). The recipe per seed is unchanged: clone the feel demonstrations (clone seed = the seed), then `configs/door_v1_bc_ppo.yaml`, `door_v1_bc_ppo_r2.yaml` and `door_v1_bc_ppo_r3.yaml` in turn, each warm-started from the previous run's best checkpoint as the configs say. No fourth round.
  - **Why, and what it costs:** round 3's best checkpoint (900k) scored 395/400 with 6 slams (1.5 %) on `door_dev_v1`; its final checkpoint, evaluated afterwards, 400/400 with 2 slams (0.5 %) (`96fbe36`, `daa91b3`). Seed 0's dev numbers under this rule were therefore chosen after seeing them and flatter it. Seeds 1 and 2, and the one-shot run, are judged under the rule fixed here.
  - **Changed again by the user on 2026-11-24, after the seed 1 and 2 dev results (`9512e41`) and before any round 4 result: a fourth round** (the last of the 4 allowed above), on every seed. The policy used for a seed is the **final checkpoint of round 4**: `configs/door_v2_bc_ppo_r4.yaml`, warm-started from the seed's round 3 best checkpoint. `door_final_v1` is still untouched.
  - **Why (the diagnosis on `door_dev_v1`):** seed 2's round 3 final checkpoint scored 392/400, with open/right/spring at 16/21, below T1's 90 % cell bar. Its earlier checkpoints opened all 21. All 8 failures are opens that time out. Six of them let go of the bar before the springy lid reached its hold-open detent (the last 8 mm of travel), sometimes while the lid still moved at 0.15–0.2 m/s, letting it coast ("fling"). The spring then pulled the lid shut, and the hand, parked away from the bar, never went back. One more (episode 196) pushed the lid into its closed stop at about 28 N for 270 steps and never tried the other way. The policy cannot see the lid once its hand is off the bar: v1 observes the door-closed switch and latched, nothing at the open end. All three seeds fling in round 3 (spring-lid opens released before the detent: seed 0 23/98, all of which still coasted home; seed 1 9/98, 2 failures; seed 2 11/98, 6 failures), so the weakness is shared and not only seed 2's.
  - **The one change:** the machine's **door-open switch** joins the observation (`DoorV2Env`, task `door_v2`, schema `door-v2-50d`): on while the lid is within 2 mm of its open stop. That is a signal real machines expose next to the door-closed switch, and with it on the lid is open far enough on every variant and, if springy, in its detent. It is appended as the last input, and the round 3 network is widened to it with zero weights, so round 4 starts acting exactly as round 3's best (checked on 776 dev steps: largest action difference 6e-7). Reward, variation, focus, horizon and learning rates are round 3's.
  - **The baselines get the same signal:** `feel_v2` is the feel script that also ends an open as soon as the door-open switch comes, as it ends a close when the latched signal comes. T3 and T4 compare the learned policy with `feel_v2`. `feel` stays as it was (rounds 1–3, the demonstrations). `camera` and `nominal` read no machine signals and are unchanged.
  - **Confirmed by the user on 2026-11-24, after `feel_v2`'s dev result (`11069bf`) and before any round 4 result:** T3 and T4 stay against `feel_v2`, although it makes T4 much stricter: with the switch the script no longer presses the lid into its stop, and its median peak force on `door_dev_v1` fell from 45 N (`feel`) to 25 N, while round 3's policies sit at 36–39 N. If T4's force bar is missed, it is reported as the trade-off it is (the policy about 3× faster, pressing harder), not avoided by comparing with a script denied a signal the policy has.
  - **Changed by the user on 2026-11-24, after round 4's dev results (`c855145`) and before any round 5 result: a fifth round, one more than the 4 this document allowed.** The limit is broken openly, as an engineering decision: `door_final_v1` is still untouched, and round 5 is the last round whatever it shows. The policy used for a seed is the **final checkpoint of round 5** (`configs/door_v2_bc_ppo_r5.yaml`, warm-started from the seed's round 4 best checkpoint), **unless the fallback applies:** if any seed's round 5 final checkpoint fails T1 on `door_dev_v1` (success below 95 %, or any cell below 90 %), every seed's policy is its round 4 final checkpoint.
  - **Why (round 4 on `door_dev_v1`):** the door-open switch did its job: 1197/1200 pooled (round 3: 1189), seed 2's 8 failures all fixed, and a hand that let go of a springy lid early went back to the bar 37 times out of 38. But slams were 1, 11 and 5 per 400 (T4 allows 4), and the median peak force stayed at 37–40 N against `feel_v2`'s 25 N. A replay of every policy (`~/scratch/anat`, not committed) showed:
    - **The slam count of one checkpoint is a draw.** Pooled over seeds: round 3 best 11, round 3 final 5, round 4 best 9, round 4 final 17 slams per 1200. Validation slams swing between 0 and 7 per 200 from one check to the next within a run. The true rate of every round 3 and 4 policy is about 0.5–1.5 %, on the 1 % bar. The switch is not the cause, and round 3 was not safely passing either.
    - **What a slam is:** in all 17 round 4 slams the hand is on the bar, driving the lid at its own top speed (0.2 m/s), and brakes a step or two late; the sideways ones are on machines with a short travel (0.122–0.130 of 0.12–0.14 m), where the stop comes early. 15 of 17 hit at 0.17 m/s or less.
    - **How the margin acts:** round 3's margin of 0.03 cut arrivals over 0.15 m/s from 3.2 % to 1.0 %. In rounds 2, 3 and 4 alike, 0.24–0.30 % of arrivals are more than 0.05 m/s over where the penalty starts, so a margin of 0.05 predicts about 1 slam per seed per 400 episodes (an extrapolation).
    - **Where the force comes from:** pushing a lid takes about 6 N. On opens the peak is the fingers meeting the bar (half the episodes, 41 N) or the lid meeting its stop (45 %, 37 N), and the reward's force penalty only started at 40 N. `feel_v2` peaks at 19 N on opens and 39 N on closes; the policy at 38 N on both. The gap is on opens only.
  - **The two changes (`configs/door_v2_r5.yaml`), both asking for a gentler arrival:** `reward.slam_margin` 0.03 → 0.05 (penalties from 0.10 m/s at the open stop; slams still counted at 0.15) and `reward.force_limit` 40 → 20 N (at which round 4's policy would pay 0.37 per open, more than its whole time penalty of 0.27; the rate per newton is unchanged). Observation, variation, focus, horizon and learning rates are round 4's. The policy has time to spend: 51 steps against the roughly 142 that T3 needs.
  - **A pilot first, because no round has changed the force limit before:** seed 0, 400k steps, the same config (`runs/door_v2_r5_pilot_seed0`). It is not a candidate policy. It is read on the training validation episodes and `door_dev_v1` for one question: do peak force and slams fall without success falling? The three seeds are launched only if they do.
  - **Round 5's outcome (`1d7b399`):** every seed's round 5 final checkpoint passes T1 on `door_dev_v1` (399, 394 and 400 of 400; worst cells 98, 91 and 100 %), so the fallback does not apply. The pilot passed its check first (396/400, 34 N, 2 slams).
- **The one-shot run** on `door_final_v1` happens once, after the lock commit names the policy, the seeds and the baselines' versions.

## The lock (2026-11-24, approved by the user)

Committed before anything has been run on `door_final_v1`. That set is as created in `bed1e4f` (digest `1bf1bd3f850f`, file sha256 `0589d4c29a4f46db…`), and no result on it exists.

- **The policy:** each seed's round 5 final checkpoint, all three seeds, trained at `d3b085c` (task `door_v2`):

  | Seed | Run | `final_model.zip` sha256 | `final_vecnormalize.pkl` sha256 |
  |---|---|---|---|
  | 0 | `runs/door_v2_bc_ppo_r5_seed0@final` | `08ce989d544a5902…` | `36d430d7706734ed…` |
  | 1 | `runs/door_v2_bc_ppo_r5_seed1@final` | `cbad1f0c4e2852ad…` | `0f5538d1b08ef5e2…` |
  | 2 | `runs/door_v2_bc_ppo_r5_seed2@final` | `dd06a82662ac3e51…` | `950bfce8a568aade…` |

- **The baselines,** at this commit: `feel_v2` (T3, T4), `camera` (T2) and `nominal` (reported).
- **What to expect, from `door_dev_v1` (flattering after five rounds of development on it):** T1 met on every seed (seed 1's worst cell 91 %, close to the bar), T2 met, T3 met (99.4 vs 98.2 %, 71 % fewer steps), T4 missed: median peak force 31.0 vs 25.4 N, and seed 2's slams 5/400. `scripts/door_final_report.py` run on the dev results gives exactly this.
- **The run, once:** six evaluations, then the report. The evaluation is deterministic (each episode's noise is seeded by its index), so a job that crashes for a reason outside the policy (out of memory, a WSL restart) is rerun with the same command; nothing else is ever rerun.

  ```
  for s in 0 1 2; do uv run python scripts/evaluate_door.py --policy run:runs/door_v2_bc_ppo_r5_seed$s@final --set eval_sets/door_final_v1.json --jobs 6; done
  for b in feel_v2 camera nominal; do uv run python scripts/evaluate_door.py --policy $b --set eval_sets/door_final_v1.json --jobs 6; done
  uv run python scripts/door_final_report.py --policy 'runs_door_v2_bc_ppo_r5_seed{}_final' --seeds 0 1 2 --feel feel_v2 --set door_final_v1
  ```

- **The scoring** is `scripts/door_final_report.py` (tests: `tests/test_door_final_report.py`), committed here: each target as worded above; "pooled" means all three seeds' episodes, each scripted baseline's single run paired with every seed by episode; T2 is an exact two-sided McNemar test, and the policy must also be the better one.
- **Whatever the result,** it is the v4.0 door result, reported as it comes, misses included. T5 (the learned door in the line, reported with no bar) follows separately.
