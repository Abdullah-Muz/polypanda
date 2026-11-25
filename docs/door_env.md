# The door skill's training environment (v4.0)

The env for v4.0's headline skill: a learned policy that opens and closes the sliding lid of a benchtop machine it has never seen (`docs/v4_plan.md`, sections 3.2, 3.3 and 3.5). Code: `src/polypanda/environments/door.py` (`DoorEnv`, task `door`), `src/polypanda/simulation/machine_variation.py` (new machines without recompiling), `src/polypanda/control/scripted_door.py` (the privileged expert), `scripts/check_door_expert.py`, `configs/door_v1.yaml`, `configs/door_v1_ppo.yaml`, `tests/test_door_env.py`. Nothing has been trained yet: targets come first.

## The task

- **Command:** `open` or `close`, given per episode, as the stand-up is told which way to turn. Half the episodes each.
- **Scene:** a minimal one for speed: the Panda on the bench and one machine straight in front of it, its pocket at the cell's 0.5 m radius. No rail, conveyor or hall. In the line the skill runs at the machine's angle through `StationView` (reset option `continue`).
- **Start:** `open` starts with the lid shut and unlatched, as a machine leaves it after a cycle. `close` starts with the lid open somewhere between the opening that uncovers the pocket ("needed": the opening along the slide plus the lid's overlap) and its travel less 5 mm. The arm starts in the cell's carry posture above the machine (TCP 0.32 m up), jittered by up to 2 cm and 10°, fingers open.
- **Horizon:** 300 steps (15 s at 20 Hz). The expert needs about 125 (open) and 140 (close).

## What the robot observes, and why

Only what a real robot could know (schema `door-v1-49d`):

| Field | Size | Why it is fair |
|---|---|---|
| joints q, velocities dq, commanded targets q_cmd | 21 | its own arm, as in the grasp |
| TCP position and rotation (6-D) | 9 | forward kinematics |
| finger gap, gripper commanded closed | 2 | the gripper's own state |
| each finger touching the handle bar | 2 | the grasp's contact flags (a stand-in for a fingertip sensor or a gap check) |
| `handle_rel`: handle bar's centre relative to the TCP | 3 | a **camera estimate**: true position plus an error drawn once per episode (1 mm per axis, the station camera of `docs/pose_error.md`) plus 0.3 mm noise per step. Within 6 cm of the hand (horizontally) the camera cannot see the bar and the last estimate is held, as the estimator withholds occluded estimates |
| `bar_align`: how far the fingers' closing direction is from the bar's normal (sin, cos of twice the error) | 2 | a camera sees which way the bar runs (error 0.5° per episode); see the open questions |
| `touch_travel`: how far the hand has moved since a finger came onto the bar (reset after 5 steps off it) | 3 | the robot's own motion; it works whether the policy hooks or grips the bar |
| `force`: external force on the arm | 3 | a real Panda estimates it from its joint torque sensors; here the sum of contact forces on the robot plus 1 N noise per axis |
| `door_io`: door-closed switch, latched | 2 | the signals a real machine exposes |
| `door_command`: open or close (one-hot) | 2 | the task |
| `door_open_switch` (**v2 only**: `DoorV2Env`, task `door_v2`, schema `door-v2-50d`, from dev round 4): the lid within 2 mm of its open stop | 1 | the door-open switch a real machine has next to its door-closed switch. Without it the policy could not tell that a springy lid it had let go of short of its detent was being pulled shut ([v4_targets.md](v4_targets.md), the round 4 change) |

**Not observed:** which way the lid slides, how far it travels, its friction, damping and spring (whether it has one, and so whether it must be pushed into its detent), the latch speed, and the lid's displacement. A real robot knows the lid's displacement only through its own hand while it moves the handle (`touch_travel`, and the held handle estimate), and from the camera once the hand is clear. Finding these by feel is the point of the skill.

## Actions

Five numbers: the `cartesian-v1-4d` hand motion of `CartesianActionAdapter` (x, y, z, yaw; the hand points down), and a gripper command. The gripper closes above +0.25 and opens below −0.25, and keeps its state in between. I chose a policy-commanded gripper over a grasp-style gate so the policy may learn either way to move the lid:
- **hook** the bar: fingers open, straddling it, the trailing finger pushing. The scripted door and the expert do this, because gripping the bar is fragile here: pads 1–2 mm off centre jam on its edge;
- **grip** it.

Either way, letting go stays the policy's own decision. A gate would need privileged knowledge to decide when the lid has gone far enough. The dead band stops exploration noise from chattering the fingers. The schema is recorded as `cartesian-v1-4d+gripper`.

The fingers move at most 0.1 m/s each. Commanded straight shut, the cell's stiff finger servo (1000 N/m) slammed them through the bar within one control step. That was seen in the first (gripping) expert runs, and a real Franka Hand is no faster.

## What is randomized (every episode, without recompiling)

| Property | Range | How |
|---|---|---|
| slide direction | right, left, out (uniform) | joint axis `jnt_axis`; the bar turns to run across the slide |
| travel | sideways 0.12–0.14 m, outward 0.265–0.275 m | `jnt_range` |
| friction | 0.2–10 N, log-uniform (nominal 0.4) | `dof_frictionloss` |
| damping | 1–10 N s/m, log-uniform (nominal 4) | `dof_damping` |
| closing spring | half the machines: 10–40 N/m, uniform; the rest none | `jnt_stiffness`; a springy lid gets a soft-close damper (damping raised so it shuts itself no faster than 0.2 m/s) and a hold-open detent (`Machine._hold_open`) |
| handle along the bar | ±3 cm sideways lids, ±1 cm outward | `geom_pos`/`geom_quat` of bar and posts, `site_pos` |
| handle along the slide | ±2 cm sideways, ±1 cm outward | as above |
| handle height (bar centre above lid) | 2.5–5 cm (nominal 3.5) | as above, plus the posts' size and bounding boxes |
| latch speed | 0.01–0.1 m/s, log-uniform (nominal 0.02) | `Machine.cfg` |
| where the machine stands | ±1.5 cm each axis | `body_pos` |
| its turn | ±10° | `body_quat` |

The geometry stays sensible for each direction. The lid covers the opening the same way whatever the direction. Outward lids need 0.255 m to uncover the opening, so their travel starts at 0.265. A handle shifted along an outward lid's 13 cm width is kept within ±1 cm, so its posts stay on the lid. The outward ranges are tighter than the sideways ones because of the arm's reach: top-down IK fails beyond about 0.8 m out at handle height, and an outward lid's handle ends up to 0.79 m out.

A runtime variant is checked field by field against the same variant compiled by `add_machine` (test). One extra field had to be rebuilt: MuJoCo's per-body bounding-volume tree (`bvh_aabb`) is built at compile time, and the collision midphase prunes with it. A handle moved outside its compiled box was passed through by a finger (3 of 50 expert failures before the fix). `MachineLayout.refresh_bvh` rebuilds it, and reproduces the compiler's boxes exactly for the nominal machine. The lid body's inertia is not updated: the handle's 60 g moving a few centimetres.

## Reward

Shaped, with every term in `info["reward_terms"]`:
- approach: 10 per metre the TCP nears the handle, until the lid's goal is reached;
- touch: +0.5 while a finger touches the bar (potential-based), until the goal;
- progress: 10 per unit of progress toward the commanded end. For open, that is the opening divided by "needed", capped at 1; for close, the share of the starting opening closed. Falls give it back;
- retreat: once the goal is reached, 20 per metre the hand, off the bar, rises above it, capped at 5 cm;
- success bonus: 10, once;
- penalties:
  - 0.01 per newton of force above 40 N;
  - 0.1 per step the arm touches the machine's walls, roof, pocket or the bench;
  - 20 per m/s the lid hits a stop faster than allowed (0.15 m/s open, 0.3 m/s closing; a latch needs at most 0.1);
  - 0.05 per step pushing harder than 25 N with the lid at a stop;
  - 0.005 per step;
  - 0.001 × |hand action|².

Phase terms are paid only between two steps in the same phase, so reaching the goal is never penalized. The expert scores about +21 to +22. The barely trained smoke-run policy scored about −2 (its validation, 20k steps).

## Success and failures

- **open:** the lid opened at least "needed".
- **close:** the latch caught.
- **both:** the hand is clear of the handle (no finger touching the bar) and the TCP is at least 3 cm above the bar, all held for 0.5 s. The episode then ends. Nothing requires the fingers to have been closed, since a hook never closes them.

Otherwise it is cut off at the horizon, and `info["failure"]` gives one of `never_touched`, `not_opened`, `not_closed`, `closed_not_latched` (shut too slowly to latch), `not_released` or `non_finite`. `info["stats"]` counts collision steps and slams, and records the peak force.

## The expert: is the task solvable through this interface?

`DoorExpert` acts through the env's own 5-D actions but reads the truth: where the handle is, which way the lid slides, how far, and the latch speed. Like the scripted tending door, it **hooks** the bar and never closes the fingers:
1. turn the open fingers to straddle the bar along the slide;
2. come down around the bar;
3. move along the slide so the trailing finger pushes the bar, first over the slack (half the finger span less half the bar), then the lid's way:
   - open: until the lid is just past "needed";
   - close: until the latch catches, at 1.6 × the latch speed. If the lid shuts without latching, the other finger pulls it 2.5 cm open and the hand pushes again faster;
4. back off the bar 12 mm and rise.

It is also the planned demonstration source (plan section 3.3).

| Run (hooking expert) | open | close |
|---|---|---|
| 100 + 100 episodes, seed 5000 | 100/100 (about 123 steps) | 100/100 (about 139 steps) |

Every slide direction succeeded (right 38 + 33, left 29 + 37, out 33 + 30). There were no collisions, no slams and no close retries. The peak force was 14 N opening. Closing it was 47 N, from the last push into the stop to latch; that is just over the 40 N threshold, so the penalty is a few hundredths.

An earlier **gripping** version of the expert (pinch the bar, then slide) also reached 100/100 + 100/100 on the same seeds, and 50/50 + 50/50 on another seed, once the env had two fixes. Before those fixes it failed often:
- the fingers tunnelled through the bar: 4 to 6 of 10 open episodes failed in the first trials;
- the stale collision tree caused 3 failures in 50 open episodes, all on left-sliding lids with handles moved far from nominal.

Both were env faults, and both are fixed and tested. The expert was switched to hooking to match the scripted door: the main session measured gripping as fragile on the updated bar (24 mm tall).

## Training pipeline

A 20k-step smoke run of `door_v1_ppo` (2 subprocess envs, under `nice`) ran end to end at about 150 steps/s including validation. It wrote metadata with `door-v1-49d` and `cartesian-v1-4d+gripper`. A 41k-step run with the real 8 envs made 143 steps/s. That was at `nice 19` on a machine loaded by other work (load about 11), so it is a floor; at that rate 3M steps take about 6 hours. Both runs used the gripping version of the env, before the hook changes, which do not affect speed. Their directories under `/tmp` were deleted.

## Open questions

- **`bar_align`.** The camera's view of the bar gives the slide's axis, though not its sense (left or right), travel, stiffness or latch. An outward lid's bar is the only one running sideways, so for outward lids it gives the direction too. I think it is fair: a camera sees it, and the fingers must close across the bar. A stricter variant is one flag away (`sensing.observe_bar_yaw: false`). Please decide before targets are set.
- **After merging main's new bar** (`handle_half_height`, 24 mm tall): `MachineLayout` copies `_add_lid`'s placement of the bar and posts. The compiled-versus-runtime test will flag any change in that placement, and the expert should be re-checked (`scripts/check_door_expert.py`).
- **Pure PPO will likely struggle with exploration** (find the bar, push it the right way, back off). The PPO config is plain, from scratch. The plan's behaviour cloning from `DoorExpert` (state inputs) is not built yet. A demonstration recorder over `run_expert` is a few lines.
- **The expert is an oracle, not the baseline.** The nominal and camera-aware scripts of `evaluation/tending.py` are the fair baselines. They run in the line, not in this env.
- **In the line:** `CartesianActionAdapter` solves IK for a base at the origin. On the rail, the base moves, and the adapter does not yet take that into account, so the skill has not been run on the line's M1. It is tested through `StationView` at −90° in its own scene.
- **Closing spring (decided 2026-11-23: included).** Half the machines have one. A one-armed robot cannot hold a lid open while it loads, so a springy lid has a **hold-open detent**, as springy guards do: in the last 8 mm of its travel the spring is cancelled and the lid is held against its stop with 1 N. Pushed out of it, the spring pulls it shut. `open` on a springy lid therefore succeeds only with the lid in its detent (failure `not_held_open` otherwise), and `close` episodes start with it there. Without a damper, a springy lid pushed out of its detent ran ahead into the leading finger (100–390 N spikes, measured) and the detent snapped it into its stop (slams), so springy lids get a soft-close damper. The expert pushes on 1 cm further on springy lids (the hand sags against the spring at full reach): 120/120 on a mixed spring check, no slams, peak force at most 53 N.
- **Force estimate:** an idealized estimate (contact forces plus Gaussian noise). A real torque-based estimate is biased by model errors and is worse when the arm moves fast.

## Training-only devices (dev round 2)

- `focus` (DoorFocusConfig): a share `prob` of training episodes drawn from chosen cells
  ("command/slide/spring|plain"); `DoorConfig.for_evaluation()` drops it, and at prob 0 the
  episode stream is the plain one draw for draw (the eval sets regenerate with their digests).
- `reward.slam_hit`: a penalty once per slam on top of `slam` per m/s (0 in door_v1.yaml).
- Validation on fixed episodes (`eval_fixed_episodes` in a training config,
  training/fixed_eval.py): the same `eval_episodes` episodes, drawn once from `eval_seed`, every
  validation, logged per cell to `eval/cells.jsonl`.

## Hand-over starts drawn wide (v4.1)

v4.0's T5 showed the skill depends on starting where it was trained: 67 % of machines running
when the line handed the door over from its carry posture (hand about 31 cm over the handle),
95 % from the trained start (about 8 cm over it; [v4_targets.md](v4_targets.md), T5 and T5b).
v4.1 trains and evaluates from **any hand-over** ([v4_1_targets.md](v4_1_targets.md), Part II;
[v4_1_plan.md](v4_1_plan.md), section 4.1): `DoorStartConfig.wide`.

- **Off by default**, so v4.0's start, draws, sets and policies are untouched
  (`tests/test_lid_identity.py`, unchanged, passes; `door_dev_v1`, `door_final_v1`,
  `door_stretch_v1` and `door_drawer_probe_v1` redraw byte for byte).
- **On**, each episode's hand start is drawn, uniformly and independently:

  | | Range | Measured from |
  |---|---|---|
  | TCP height | 0.03–0.35 m above the handle bar's centre (`wide_height`) | the true bar, as the episode places it (lid open or shut) |
  | along the station's radial | −0.08 to +0.20 m (`wide_radial`; + away from the robot) | the carry point over the station (a lid's pocket, an open drawer's nest) |
  | across it | ±0.08 m (`wide_side`) | the same |
  | wrist yaw | ±90° (`wide_yaw`: the fingers any way round) | the carry posture's |

  A draw is kept only if the arm reaches it (the IK converges) and every moving link of the
  arm (hand and fingers included) is at least 1 cm (`wide_clearance`) from the machine, its lid
  or drawer and the bench (`DoorEnv.arm_clearance`, MuJoCo's geom distance); otherwise it is drawn
  again (`DoorEnv._draw_wide_start`; about 1 draw in 70 is redrawn). The start is stored as the
  episode's `jitter` (from the carry point), so a wide episode replays exactly like any other,
  in any version of the env.
- **The cell's real hand-overs, and how the range covers them** (measured in the test cell,
  `control/test_cell.py`, planned testers A lid, B drawer, both hand-overs, dev shifts 1 and 2,
  8 modules each; the TCP relative to the true handle and to the machine's origin, machine
  frame, x away from the robot):

  | Hand-over | Kind, command | TCP over the bar | TCP from the machine's origin (x) | Wrist yaw from the carry posture |
  |---|---|---|---|---|
  | `door_start` (the cell's default) | lid, open / close | 0.10 m | 0.00 | 0° |
  | | drawer, open (shut drawer) | 0.15 m | **+0.11 m** (over the believed nest inside the box; 8.5 cm beyond the bar) | 0° |
  | | drawer, close (open drawer) | 0.15 m | 0.00 (9.5 cm beyond the bar) | up to ±17° |
  | `carry` | lid, open / close | 0.10 m | 0.00 | 0° |
  | | drawer, open | 0.15 m | 0.00 | 0° |
  | | drawer, close (after loading it) | 0.15 m | 0.00 | **89°** (the wrist still turned to reach into the drawer: `MachineConfig.hand_axis`) |
  | the line's carry (v4.0's T5) | lid | 0.31 m | 0.00 | 0° |

  v4.0's own start (TCP 0.32 m ± 2 cm over the pocket, ±2 cm, ±10°) is 8–13 cm over a lid's
  bar and 14–18 cm over a drawer's, within ±3 cm of the origin. It never started over a shut
  drawer's nest (+0.11 m) nor with the wrist turned more than 10°. The wide range holds every
  row above: heights 0.10, 0.15 and 0.31 m inside 0.03–0.35, the shut drawer's +0.11 m inside
  +0.20 (a drawer's nest sits at most its travel, 0.14 m, behind the origin), the cell's lateral
  errors (under 1 cm) inside ±8 cm, and the 89° wrist inside ±90°.
- `scripts/make_door_sets.py --wide-start` draws a set with wide starts and records the start
  config in the file (`start`); with several `--kinds` it takes them in turn, two episodes
  each (open, close), so 400 episodes are exactly 100 per kind × command.

### The v2 door sets (v4.1, generated before any v4.1 door training)

As [v4_1_targets.md](v4_1_targets.md), Part II: lids and drawers, half `open` and half `close`
within each kind, hand-over starts drawn wide; every other range as v4.0's training ranges
(drawers as `door_drawer_probe_v1`'s). Generated at `4d05cf2`:

```bash
uv run python scripts/make_door_sets.py --name door_dev_v2 --seed 20261130 --kinds lid drawer --wide-start
uv run python scripts/make_door_sets.py --name door_final_v2 --seed 20261201 --kinds lid drawer --wide-start
```

| Set | Episodes | Seed | Digest | File sha256 | Used for |
|---|---|---|---|---|---|
| `door_dev_v2` | 400: 100 each of lid/open, lid/close, drawer/open, drawer/close; lids right 57, left 74, out 69, 95 springy | 20261130 | `231dbe288150` | `60640238011d0377…` | development |
| `door_final_v2` | 400, the same mix (lids right 55, left 73, out 72, 92 springy) | 20261201 | `1db12384e8da` | `653f031533797728…` | the one-shot result, run once |

Both redraw byte for byte (`tests/test_door_wide_start.py` redraws `door_dev_v2`). In
`door_dev_v2` the TCP starts 0.03–0.35 m over the bar (median 0.19 m; 48 % above 0.20 m, 18 % in
v4.0's 8–13 cm), 41 of the 100 shut drawers with the hand beyond +0.09 m (over the nest inside
the box), and 137 of 400 with the wrist turned more than 60°.

**Training never sees them.** As in v4.0, training draws a fresh episode every reset from the
env's own generator (`env.np_random`), seeded by the trainer with the run's `seed` plus the env's
rank, and validates on episodes drawn from its `eval_seed`; the sets come from
`np.random.default_rng(20261130)` and `(20261201)`. No door config's `seed` or `eval_seed` (with
64 ranks above it) is a set's seed (`tests/test_door_wide_start.py`, for every
`configs/door_*.yaml`). v4.1's door training configs need `start.wide: true` and
`variation.kinds: [lid, drawer]` to train on this distribution.

### The baselines and v4.0's policies on `door_dev_v2` (before any v4.1 door training)

Run once each from `be2362c` (clean), `scripts/evaluate_door.py --set eval_sets/door_dev_v2.json
--jobs 3`; results in `results/v4/door/<policy>_door_dev_v2.json`. v4.0's policies are the locked
round-5 final checkpoints (`runs/door_v2_bc_ppo_r5_seed{0,1,2}@final`, the starting point of
v4.1's retraining), read from the main checkout's `runs/` unchanged. `door_final_v2` has not been
run on.

| Policy | Lid open | Lid close | Drawer open | Drawer close | All | Median steps to success | Peak force median / max | Slam episodes |
|---|---|---|---|---|---|---|---|---|
| expert (privileged) | 100 | 100 | 100 | 100 | 400/400 | 119 | 8.7 / 346 N | 0 |
| **`feel_v2`** | 100 | 98 | 100 | 100 | **398/400 (99.5 %)** | 146 | 21.2 / 325 N | 0 |
| `camera` | 43 | 48 | 99 | 52 | 242/400 (60.5 %) | 144 | 6.6 / 325 N | 0 |
| `nominal` | 17 | 15 | 53 | 64 | 149/400 (37.2 %) | 148 | 17.5 / 566 N | 3 |
| v4.0 r5, seed 0 | 59 | 80 | 0 | 1 | 140/400 (35.0 %) | 66 | 88.5 / 1325 N | 12 |
| v4.0 r5, seed 1 | 78 | 86 | 0 | 10 | 174/400 (43.5 %) | 68 | 113 / 668 N | 15 |
| v4.0 r5, seed 2 | 62 | 60 | 0 | 0 | 122/400 (30.5 %) | 68 | 111 / 531 N | 39 |

(Each kind × command cell is out of 100; per-cell steps and forces are in each file's
`summary.kind_command`.)

- **The demonstration check passes:** `feel_v2` succeeds on 398 of 400 (99.5 %, 95 % CI
  98.2–99.9), every kind × command cell at 98 % or more. Its two failures are springy
  left-sliding lids to close (episodes 25 and 329: hand 18 and 28 cm over the bar, wrist turned
  49° and 87°, stiff latches of 0.089 and 0.099 m/s): it tries right first, flips, and pushes
  three times without the latch catching (`not_closed` at the horizon). Its worst
  command × slide × spring cell is close/left/spring.
- **The wide starts cost the scripts almost nothing** (by start height, wrist turn and radial
  place `feel_v2` is 98–100 % everywhere). One start (episode 78: a shut drawer, hand 20 cm up
  and 19 cm beyond the origin, wrist turned 82°) makes the expert, `feel_v2` and `camera` land
  a finger on top of the drawer's bar on their way down (325–346 N for a few steps); all but
  `camera` then succeed. It is the expert's and `feel_v2`'s only peak above 110 N.
- **v4.0's policies do not work from these starts, nor on drawers.** On drawers they succeed 0–10
  of 200 (never trained on them; per-cell median peak forces 95–315 N); on lids
  59–86 % against 98.5–100 % on `door_dev_v1`, losing most with the wrist turned more than 45°
  (52–74 of 95) and from high starts. This is what v4.1's retraining starts from.
- **The v4.0 policies are unchanged:** on `door_dev_v1`, at `be2362c`, each seed's 400 rows are
  identical to its committed result from `d3b085c` (399, 394 and 400 of 400).
