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
- **The one-shot run** on `door_final_v1` happens once, after the lock commit names the policy, the seeds and the baselines' versions.
