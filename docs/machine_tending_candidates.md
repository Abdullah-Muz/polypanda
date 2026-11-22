# Machine tending: candidates for v4.0 (a proposal, not decisions)

*Written on 2026-11-21, while v3.2's final cell run is still pending. Nothing here is decided. It sets out what a machine-tending milestone could contain, what each piece would reuse from the current code, whether it should be learned or scripted, what could go wrong, and three scopes to choose from, with one recommended. If a scope is adopted, its targets are committed in a `v4_targets.md` before any v4 training, under the project's usual rules: 3 seeds, held-out sets drawn before training, Wilson 95 % intervals, exact McNemar tests on paired cases, a scripted baseline beside every learned result, and every result reported as it comes, misses included.*

*Numbering: the README's roadmap numbers milestones up to 11 (v3.1) and lists v3.2 without a number. This doc counts v3.2 as Milestone 12 and calls the next one **v4.0 / Milestone 13**.*

## What machine tending is

In a factory, many machines (a CNC mill, a lathe, a test station) need someone to put a raw part in, close the door, start the cycle and take the finished part out when it ends. Machine tending gives that job to a robot. The hard parts are not just the motions. The robot has to open and close a door, set the part into a fixture that only accepts it when it is aligned to within a millimetre or two, and juggle several machines that finish at different times so that none of them stands idle. Here the machine is simulated: a box with a door, a fixture inside, a timer that runs the cycle, and a sensor that says whether a part is seated.

## What the current code gives a tending cell

Most of what tending needs already exists. Grounded in the code as of `50afb4c`:

| Need | What exists | Where |
|---|---|---|
| A scene with stations around the robot | `build_workcell_spec` composes the cell in memory with `MjSpec`; station angles are fields of `WorkcellConfig` (`tower_angles`, `buffer_angles`, `reject_angle`); `station_xy` places anything on the 0.5 m circle | `simulation/workcell.py` |
| A device that runs every control step | `Conveyor`: `update()` once per step, `index()`, `ready`: the pattern a machine's timer and interlock would copy | `simulation/workcell.py` |
| Constraints that switch on and off | the grasp assist: one `mjEQ_WELD` per cube, toggled through `eq_active` (`Workcell.attach` / `detach`); the same mechanism can latch a door | `simulation/workcell.py` |
| A skill trained in front of the robot, used at any station | `StationView` rotates observations by −θ and actions by +θ; it knows each env's observation layout through `LAYOUTS` and `VECTOR_FIELDS` | `supervisor/stations.py` |
| A learned skill inside the cell | `LearnedSkill` (policy plus its env config), `LearnedCell._run_skill` (runs a policy until its success predicate, failure or horizon), `LearnedCell.place` (lower to the hand-off gap, then the policy, detach on release) | `supervisor/cell.py` |
| Skill envs that run on a shared scene | every skill env takes `scene=(model, data)` and a cube body name, and starts from the current state with reset option `continue` (e.g. `PlaceEnv.__init__`, `PlaceEnv.reset`) | `environments/place.py`, `grasp.py`, `standup.py` |
| A new skill gets trained | register it in `TASKS` (`environments/tasks.py`); write a task YAML and a PPO YAML (`configs/<task>.yaml`, `<task>_ppo.yaml`); `scripts/train.py --config ... --seed k` runs `training/runner.train`, with warm starts (`init_from`) and function-preserving input renormalization (`renormalize_inputs`); `policies/loader.load_run` refuses mismatched schemas | `training/runner.py` |
| Scripted moves | `ScriptedCell.swing_to` (base-only swings), `vertical` (closed-loop descent that lands a tracked body within 1 mm), `line`, `pick`, `place`, `reject` | `control/scripted_cell.py` |
| Reading the cell as it is | `observe()` returns towers, buffer slots and loose cubes with their `rest_state` (upright, lying, leaning; reachable) | `supervisor/observe.py`, `simulation/disturbances.py` |
| Recovery | `recover_loose` inside `run_mixed_order` (stage, stand up, re-pick, route), `Workcell.call_spare`, the staging ring `CellControllerConfig.staging` | `control/scripted_cell.py` |
| Fault suites | `Knock`, `make_suite`, `save_suite` / `load_suite`, `knock_outcome` with a `recoverable` flag | `simulation/disturbances.py` |
| Planners and their gates | the symbolic cell and its `Calibration` (skill odds and fitted action times, `add_time_spread`), `RecoveryPlannerEnv` (PPO over discrete actions), the headroom gate | `supervisor/symbolic.py`, `environments/recovery_planner.py`, `scripts/planner_gate.py` |
| Evaluation and provenance | `scripts/evaluate_cell.py`, `evaluation/cell.py` (`CellEpisode`, per-attempt skill records), `evaluation/cell_picks.py` (stage one case in a fresh cell, call the cell's own skill), `mcnemar_exact`, `wilson_interval`; every result records the commit and whether the tree was clean | `evaluation/` |
| Hand-off pools, videos | `scripts/make_place_pool.py`, `make_standup_handoffs.py`, `data/handoff/*.npz`; `scripts/record_recovery.py`, `record_cell.py`, `view_workcell.py` | `scripts/` |

**One constraint shapes everything below.** Every work surface is at table height (z = 0 is the bench), and every station sits on the 0.5 m circle. That is why a skill trained in front of the robot works at any station by a rotation about the base axis (`StationView`). A machine fixture placed **at table height on that circle** keeps the existing grasp and place usable as they are. A raised fixture, or one at another radius, would need retraining.

## Where the machines go

The 0.5 m circle is already full within the base's reach (joint 1 ±166°, `disturbances.BASE_RANGE` 162°):
- infeed pick point at +90°, with the belt housing running along y = 0.5 m and covering roughly +55° to +125° of the circle (from `_add_conveyor`: housing from x = +0.98 m to x = −0.86 m, ±9 cm wide);
- towers A, B and C at −40°, −90° and −140°;
- buffer slots at −10°, +10° and +30°;
- the reject bin at +150°.

A machine is about 32 cm wide, about 40° of arc at 0.5 m. Two options:

- **Layout A, a tending variant of the cell (recommended for the small and medium scopes).** M1 takes tower B's place (−90°) and M2 takes tower C's (−140°). Tower A's fixture (−40°) becomes the outfeed stack for finished parts. Infeed, buffer slots and bin stay. In code, this means a new `WorkcellConfig` field (e.g. `machine_angles`) and `tower_angles=(-40.0,)`; the builder already loops over `cfg.tower_angles`. Two staging spots, at −65° and −115° on the 0.36 m ring, would touch M1's enclosure and have to move. The existing three-tower cell stays as it is, as the default configuration.
- **Layout B, machines beside the full cell.** Keep all three towers and put the machines on an outer ring (fixture at about 0.68 m) in the front arc, behind the buffer slots. The old cell is untouched, but the fixture would be off the radius the place was trained at (`docs/workcell.md`: "trained in front of the robot at that distance"), the carry posture is built for `cfg.radius` (`ScriptedCell._carry_posture`), and a hood sliding outward would leave the 0.78 m reach (`disturbances.REACH`). That means more retraining and more reach risk.

**Top-down sketch, Layout A** (not to scale; angles are base angles, 0° = straight ahead; machines are drawn square to the page, but really each faces the robot):

```
                                +90 deg: infeed
  belt housing (downstream)      pick point                    tunnel        hopper
  =============================[   P   ]=======================[########]   [::::::]
                                   ^     raw parts ride the belt this way: <----
    [ bin ] +150 deg               |
    reject / scrap                 |                                  [b2] +30 deg
                                   |                                  buffer slots
                         .  .  staging ring  .  .                     [b1] +10 deg
                       .        (r 0.36 m)        .
                      .         ( PANDA )          .  -------------->  front, 0 deg
                       .                          .                   [b0] -10 deg
                         .  .                .  .
        M2 (slow)                                              [ A ] -40 deg
        -140 deg                                               outfeed stack
     +====H=====+                                              (tower A fixture)
     |  [pkt]   |            M1 (fast)
     |          |            -90 deg
     +----------+         +====H=====+     H     handle on the hood's robot-side edge
                          |  [pkt]   |     pkt   pocket fixture, on the 0.5 m circle
                          |          |     ====  hood edge; the hood slides outward,
                          +----------+           away from the robot, to open
```

## Candidate pieces

A **round** below means one 3-seed training run, its development evaluation and the diagnosis before the next run. At v3.2's pace (grasp rounds 1–3 ran 2026-11-17 to 11-20) that is about one day. Training itself is 1–2.5 h per seed on CPU, the 3 seeds in parallel. Day counts are project days at that pace. They are rough, and the project's history says skills take more rounds than planned: the grasp needed 9 rounds over v3–v3.2, and the stand-up 12.

### C1. The machine model

- **What it is:** an enclosure on the bench, about 32 × 28 × 20 cm, with its fixture on the 0.5 m circle at table height. It has a hood over the fixture on a slide joint, which opens radially outward by about 26 cm (a hinge is the alternative, see C2), and a handle bar on the hood's robot-side edge. A **latch** holds the hood when it comes within 2 mm of closed. A **cycle timer** runs the machine through `EMPTY → LOADED → RUNNING → DONE`, with `FAULT` when the interlock is broken. A **part-present sensor** reports whether a part is seated, and a lamp per machine shows its state. Machining is shown by recolouring the part from raw grey to a finished colour, as `Workcell.reset` already recolours cubes through `geom_rgba`.
- **New:** all of it. That includes a `Machine` class with a per-step `update()` (the `Conveyor` pattern) and an interlock: the machine starts only with a part seated and the hood latched, and opening the hood mid-cycle is a fault. The latch is a weld toggled like the grasp assist, released at `DONE`. The sensor is a verified predicate, in keeping with the project's rule that transitions are verified predicates, never timers: part bottom within 1 mm of the pocket floor, tilt under 3°, inside the pocket. A MuJoCo touch sensor could be added beside it. `observe()` must learn about machine fixtures (a new `CellView.machines`); otherwise a seated part reads as a loose cube and triggers recovery.
- **Reused:** `MjSpec` composition, `station_xy`, `_visual` dressing, the stack-light pattern (`Workcell.set_stack_light`).
- **Learned or scripted:** scripted. It is part of the environment, not the robot.
- **Risks:** tight box-in-box contacts in MuJoCo can jam or jitter at 1–2 mm clearance, so measure a scripted insertion before anything is trained on it. The hood's slide friction and damping need tuning. The enclosure must stay under about 0.2 m so a carried cube clears it during base swings (`CellControllerConfig.carry_height` is 0.32 m at the TCP).
- **Cost:** no training; 2–3 days with tests and a scripted tending controller.

### C2. A learned door skill: grasp the handle, open or close the hood

- **What it is:** one **commanded** policy, told "open" or "close" (as the v3 stand-up is told which way to turn). It comes down from the carry posture, pinches the handle bar with a closing gate, slides the hood to its open stop or until it latches, lets go and rises. Success: the hood within 5 mm of the open stop, or latched, with the fingers released and the hand clear, held for 0.5 s.
- **New:** `DoorEnv` (e.g. `environments/door.py`) with its own observation schema: arm state as in the grasp, the handle relative to the TCP, hood position and speed, the command, finger contacts and a latched flag. It needs a `TASKS` entry, a layout in `stations.LAYOUTS` (with the handle offset in `VECTOR_FIELDS`), configs, and a scripted baseline: IK above the handle, `vertical` down, close, `line` along the slide axis, open, up.
- **Reused:** the closing-gate logic of `GraspEnv` (`GateConfig`: close when centred and aligned, reopen after closing on nothing), the `cartesian-v1-4d` action adapter with `set_frame`, the runner, the loader, `LearnedCell._run_skill`.
- **Learned or scripted:** **learned, as the headline**: the project's first articulated object, contact-rich, the kind of task RL handles well without an exact model of the mechanism. The scripted door runs beside it and, knowing the slide axis exactly, will probably reach about 100 %.
- **A design choice that avoids a known simulator problem.** In this simulator a firmly held cube slides along the finger pads whenever the arm moves (`WorkcellConfig.grasp_assist`, `docs/workcell.md`). If the handle bar runs tangentially and the fingers close radially, then pulling or pushing the hood is taken by the finger faces, not by friction. A slide therefore needs no new assist.
- **Risks:**
  - Training starts from scratch: the observation layout differs from every existing policy, so there is no warm start. Exploration is the likely bottleneck, as it was for the grasp (`docs/grasp_task.md`). A share of episodes would start with the handle already pinched, as the grasp's 30 % pre-grasped starts did.
  - Latching needs a firm last push, and a slammed hood could bounce back open.
  - A hinged lid would sweep an arc beyond the 6-D adapter's 0.6 rad tilt limit (`max_tilt`), and its handle would turn with it. It would need a free-turning handle or an assist.
- **Cost:** 2–4 rounds of about 1M steps each; 4–6 days.

### C3. Precise place into the fixture

- **What it is:** set a held raw part into a pocket with 2 mm clearance per side (40 mm part, 44 mm pocket), 15 mm deep, with a 45° lead-in chamfer. Success: seated (bottom within 1 mm of the floor, tilt under 3°), released, gripper clear, for 0.5 s. This is a peg-in-hole task with a cube.
- **New:**
  - a fixture option in `PlaceConfig`: pocket geometry and a clearance curriculum;
  - a seated check, much stricter than `PlacedCheckConfig` (2 cm, 10°, 3 mm);
  - a release gate that opens once the part is **inserted** (its bottom below the pocket's rim) rather than a tighter xy window;
  - a hand-off pool recorded from real grasps in the cell, as `make_standup_handoffs.py` did for the stand-up.
- **Reused:** `PlaceEnv` and its observations (`place-v2-48d`; the target is the pocket centre); a warm start from `place_stack_v3` (with `init_renormalize_steps`); the `LearnedCell.place` flow.
- **Learned or scripted:** learned. Insertion is a classic RL case, and the learned stacking place is already the cell's placer. The scripted baseline is `ScriptedCell.place`, whose closed-loop `vertical(track=cube)` settles within 1 mm. Expect it to be strong.
- **Risks:**
  - **A known exploration trap:** a first stacking-v3 run with a 1 cm release gate never learned to release at all (comment in `configs/place_stack_v3_cartesian.yaml`). That is why the gate opens on insertion, and why the clearance shrinks from 10 mm to 2 mm over training.
  - In the cell, the deterministic place repeats the same 2–3 mm error (`docs/workcell.md`). That is more than a 1 mm clearance would forgive, so the chamfer has to capture it.
  - A plus: the pocket stops the slow sliding (creep) that ended every learned cell failure traced so far.
- **Cost:** 1–3 rounds (`place_stack_v3` took about 80 min per seed for 2M steps); 3–4 days.

### C4. Unload from the fixture

- **What it is:** with the hood open, grasp the finished part in the pocket and lift it clear of the rim and the enclosure. A scripted swing then takes it to the outfeed, where the learned stacking place sets it on tower A's fixture.
- **New:** little if the grasp transfers: a success check of the part lifted 5 cm above the rim, and a `fixture_unload` pick set built like `evaluation/cell_picks.py`.
- **Reused:** `grasp_anypose9` through `StationView`. In Milestone 9, the learned grasp of the time (`grasp_cart`) ran at the belt and the buffer unchanged, 2010 of 2010 picks per seed. `place_stack_v3` is aimed at the fixture for the outfeed.
- **Learned or scripted:** learned, with the existing policy. Try it zero-shot first, and fine-tune with the pocket and enclosure in the grasp's training scene only if the bar is missed.
- **Risks:** the grasp has never seen walls beside the cube or a frame above it, and its observation has no obstacle input. The fingertips must close above the rim, so the pocket stays at most 15 mm deep, or gets finger reliefs. A plus: the fixture fixes the part's yaw, so seed 0's known weakness on standing cubes turned near −45° does not arise.
- **Cost:** 0–2 rounds; 1–3 days.

### C5. Scheduling two machines with different cycle times

- **What it is:** deciding what the robot does next when M1 (fast, e.g. 30–45 s) and M2 (slow, e.g. 60–90 s) compete for one gripper. The robot itself needs about 20 s of simulated time per pick-and-place today (`docs/workcell.md`). The decisions are which machine to serve, when to pre-fetch a raw part into a buffer slot, and where to park a finished part during a swap.
- **The idle-time metric:** for each machine, the fraction of the order's duration it is not cutting. That splits into **starved** (empty, waiting for a part) and **blocked** (done, waiting to be unloaded), reported separately. Beside it: throughput (finished parts per simulated hour) and makespan (time to finish an N-part order). All three come from the `Machine` state log, in physics.
- **New:**
  - a discrete-event tending model in the style of `supervisor/symbolic.py`, with action times fitted from logged physics attempts (as `Calibration.add_time_spread` does);
  - a rule family: first-come-first-served, serve whichever machine finishes soonest, pre-fetch when a machine is within T seconds of done, swap through a buffer slot;
  - a per-order search for the best plan, for the headroom gate;
  - idle time in the cell's episode records.
- **Reused:** the `GreedyPlanner` pattern, the symbolic cell's calibration, the `RecoveryPlannerEnv` pattern (a 64 × 64 PPO policy over discrete actions, minutes to train), the headroom gate of `scripts/planner_gate.py`.
- **Learned or scripted:** **scripted first; learned only if a headroom gate passes.** Twice before (routing, then recovery), a search showed hand-written rules already near-optimal. The v3 planner's edge in its symbolic cell also vanished in physics. With two machines and one gripper, a good rule may leave nothing to learn.
- **Risks:** no headroom; a symbolic-to-physics gap; idle time mixed up with skill failures (each retry costs time). The cycle-time ranges above are guesses, to be calibrated from scripted tending runs before any suite is drawn.
- **Cost:** scripted 2–3 days. A learned scheduler adds 1–3 short rounds and 3–4 days, including its physics check.

### C6. Recovery

| Fault | Detected by | Response | Reuses |
|---|---|---|---|
| Hood not latched after closing | latch flag, hood position | close again (up to 2 tries), then the machine faults and the order fails | new |
| Part seated badly (resting on the chamfer, tilted) | the seated predicate, before closing | lift it out with the unload skill and place it again (up to 2 tries), then to the reject bin and a spare up the belt | `ScriptedCell.reject`, `Workcell.call_spare` |
| Part dropped on the bench during a transfer | `observe()`: a loose cube and its `rest_state` | the existing `recover_loose`: stand it up if lying, pick, route; written off if leaning or out of reach | all existing |
| Part dropped inside an enclosure, off the fixture | new zone in `observe()` | recoverable only with the hood open and the part in reach; otherwise written off, spare called | new |
| Outfeed stack knocked | the existing knock check | the existing rebuild | all existing |

- **Faults must be injected.** With the grasp assist on, a part never drops by itself. A fault suite would mirror `disturbance_suite`: one fault per order or none. The types: the latch refuses once, the part is nudged onto the chamfer after release, the weld is released mid-carry, and a `Knock` hits the outfeed.
- **Measure the physical ceiling first.** In v3, about a third of knocks were unrecoverable by any robot. Run the scripted cell on the fault suite first, and count only the faults it shows to be recoverable, as v3 did.
- **Learned or scripted:** decisions scripted, skills learned. In v3.0 and v3.1 the learned recovery planner did not beat the hand-written rules in physics.
- **Cost:** 0 rounds; 3–4 days.

### C7. Camera-based part pose (optional)

- **What it is:** estimate the part's position and yaw from a simulated fixed overhead camera, using MuJoCo offscreen rendering (the videos already use it), and feed the estimate to the policies in place of the true pose. Today every policy reads the cube's pose from the simulator (README limitation: "none uses vision").
- **New:** cameras in the scene, a pose estimator (colour segmentation plus depth is enough for coloured cubes in simulation), an error model, and evaluation with the estimated pose paired against the true one.
- **Learned or scripted:** a classical estimator first. Policies are fine-tuned with matching observation noise.
- **Risks:** rendering on CPU is slow, so the camera would be used at evaluation only, with training on noise. Simulated images are too clean to say anything about real vision. Yaw is ambiguous every 90°, which is harmless for the grasp: it already reads alignment as `align_sin4` / `align_cos4`.
- **Cost:** 0–1 rounds; 4–6 days. Already on the roadmap as "later".

### Summary

| Piece | Learned or scripted | Reuses most of | Rounds | Days |
|---|---|---|---|---|
| C1 machine model | scripted (environment) | workcell builder, `Conveyor` pattern, weld toggling | 0 | 2–3 |
| C2 door open/close | **learned** (new env), scripted baseline | grasp gate, action adapter, runner | 2–4 | 4–6 |
| C3 precise place | **learned** (warm start from `place_stack_v3`), scripted baseline | `PlaceEnv`, `LearnedCell.place` | 1–3 | 3–4 |
| C4 unload | **learned** (`grasp_anypose9`, zero-shot first) | `StationView`, cell pick checks | 0–2 | 1–3 |
| C5 scheduling | scripted; learned only past a headroom gate | symbolic cell, planner gate | 0 (+1–3) | 2–3 (+3–4) |
| C6 recovery | scripted decisions, learned skills | `observe`, `recover_loose`, spares, suites | 0 | 3–4 |
| C7 camera pose | classical estimator, noise-tuned policies | offscreen rendering | 0–1 | 4–6 |

Evaluation runs, documents and the video add 3–4 days to any scope.

## Scope options

| | **Small** | **Medium (recommended)** | **Large** |
|---|---|---|---|
| Machines | 1 (M1) | 2, different cycle times | 2 |
| Layout | A | A | A, or B |
| Door | scripted | **learned** (slide), scripted baseline | learned slide and a hinged-lid variant |
| Place into fixture | learned | learned | learned, plus a 1 mm clearance condition with its own bar |
| Unload | learned (zero-shot) | learned (zero-shot, fine-tune if needed) | as medium |
| Scheduling | none (one machine) | scripted rules, idle metric, headroom gate | as medium, plus a learned scheduler if the gate passes |
| Recovery | re-seat only | latch, seat and drop faults with an injected-fault suite | as medium, plus drops inside the enclosure |
| Camera | no | no | yes (C7) |
| Towers | outfeed stack only | outfeed stack only | towers built from machined parts: tending feeds tower A's recipe |
| New learned skills | 1 (plus an unload check) | 2 (plus unload) | 2–3, plus a scheduler and a camera |
| Rough length | 2–3 weeks | 4–5 weeks | 7–9 weeks |

**Recommendation: medium.**
- It is the smallest scope that is really machine tending: a door, a precise fixture, two machines competing for one robot, and faults to recover from.
- It adds two genuinely new learned skills (an articulated door and a peg-in-hole place), which keeps the learned chain as the headline.
- Unload and outfeed reuse the policies the cell already trusts.
- Scheduling stays scripted, measured with the idle metric, and gets the headroom gate before anyone trains a scheduler. That is the lesson of v3 and v3.1.
- Layout A keeps every station on the circle the skills were trained on, so station frames apply unchanged.
- The small scope is a good fallback if the door or the pocket contacts turn out to be hard in MuJoCo (C1's first measurement).
- The large scope stacks three uncertain pieces (hinge, camera, learned scheduler) on top of medium. It is better done as v4.1 once medium's results are in.

## Draft targets for the medium scope

*Drafts for the user to approve. Nothing below is committed. The final numbers, set seeds and digests would go into `v4_targets.md` before any v4 training, in the v3.2 style.*

### Evaluation sets, generated and committed before any training

| Set | Contents | Used for |
|---|---|---|
| `door_v1` | 200 cases: 100 open, 100 close, from the carry posture over M1 or M2. Hood start: closed or partly open (0–5 cm for close cases); machine placed with ±3 mm and ±1° jitter; slide friction ×0.8–1.2 | development |
| `door_final_v1` | 200 cases, the same distribution, another seed | the one-shot result, run once |
| `fixture_place_v1`, `fixture_place_final_v1` | 200 hand-offs each, recorded from the cell's own grasps (grip offsets and tilts as they really come), into M1's or M2's pocket, 2 mm clearance. Condition: the same cases at 1 mm clearance | development; one-shot |
| `fixture_unload_v1`, `fixture_unload_final_v1` | 200 seated parts each, offset within the clearance, turned ±3°, hood open | development; one-shot |
| `tending_suite_v1`, `tending_suite_final_v1` | 200 tending orders each, 4–8 parts, cycle times drawn from calibrated ranges (M1 fast, M2 slow); 150 with one injected fault (latch, seat, drop, outfeed knock), 50 with none | development; the final cell run, once |

- **Baselines, before any training,** on the development sets: the scripted door, the scripted fixture place and the scripted unload. Beside them, the existing learned policies zero-shot: `place_stack_v3` aimed into the pocket, and `grasp_anypose9` for the unload. As in v3.2, this also checks that each set catches failures before anything is trained on it.
- **The physical ceiling of each fault type,** from the scripted cell on the development suite, before the bars below are fixed.
- **Checkpoint rule, fixed before training** (v3.2 lesson): each seed's best checkpoint by a validation that mixes the cell's own starts with the training distribution.

### Skill targets (each seed separately, on the one-shot sets)

| Check | Draft bar | Beside it |
|---|---|---|
| Door, `door_final_v1` | open ≥ 95 % **and** close-and-latch ≥ 95 %, for every seed; the hood never driven past its stops | scripted door, paired (exact McNemar, pooled) |
| Fixture place, `fixture_place_final_v1` | seated ≥ 95 % for every seed | scripted place and `place_stack_v3` zero-shot, paired; median xy error at release; the 1 mm condition reported, no bar |
| Unload, `fixture_unload_final_v1` | ≥ 98 % pooled and ≥ 95 % for every seed, no part left tilted in the pocket | scripted unload, paired. If `grasp_anypose9` meets the bar zero-shot, no unload training is done |

**Stopping rule:** at most 3 training rounds per learned skill. Each round starts from a diagnosis of the last. Rounds are compared on the development sets only; the chosen configuration is locked in a commit and run once on the one-shot set. The result is reported as it comes.

### Tending cell targets (`tending_suite_final_v1`, physics, 3 seeds)

The **learned tending cell** uses `grasp_anypose9` at the belt, the learned door, the learned fixture place, the learned unload and `place_stack_v3` at the outfeed. Its scheduler, swings and recovery decisions are scripted. The scripted tending cell runs the same orders beside it.

| Target | Draft bar |
|---|---|
| Orders with no fault | ≥ 97 % complete, pooled; per seed reported |
| Recoverable faults (as measured by the ceiling run) | ≥ 85 % of orders complete, pooled; per fault type reported |
| All orders, learned against scripted cell | reported with exact McNemar, paired by seed and order; no bar |
| Machine idle time: the chosen rule against first-come-first-served | lower mean idle fraction, significant (paired by order; a Wilcoxon signed-rank test, since idle time is not pass/fail. This test would be new to the project) |

- **Headroom gate for a learned scheduler:** in the calibrated tending model, a per-order search must beat the best rule by at least 5 % of throughput on 1,000 model orders. If it does not, that is reported and no scheduler is trained. If it does, the user decides whether one is trained, with its own target.
- **Regression check:** the three-tower cell is not changed by the tending variant. The test suite passes, and the v3.2 cell reproduces its committed per-order outcomes on a few orders of its development run on `disturbance_suite_v2`.
- **Also reported, no target:** starved and blocked time per machine, throughput, makespan, time per tending cycle against the scripted cell, and per-seed skill rates inside the cell.

## Video: a new 45-second clip

A replay of a held-out tending order, recorded like `scripts/record_recovery.py` (a `record_tending.py` would be new). As with the v3 clip, the caption says which fault was chosen for the demo and that everything after it is the cell's own.

| Time | Shot | On screen |
|---|---|---|
| 0–4 s | Wide shot of the hall and the cell, both machine lamps on | "PolyPanda v4: machine tending with learned skills" |
| 4–9 s | A raw grey part rides the belt; the learned grasp picks it | "grasp: PPO" |
| 9–15 s | Close-up on M1: the hand pinches the handle and slides the hood open | "door: PPO, new in v4" |
| 15–21 s | Low angle on the pocket: the part slides down the chamfer and seats; the lamp turns on | "place into a 2 mm fixture: PPO" |
| 21–25 s | The hood closes and latches; M1's lamp goes amber, a cycle timer counts down | "machine running" |
| 25–32 s | M2 finishes; the robot opens it, unloads the now-coloured part and stacks it at the outfeed | a two-row timeline at the bottom: each machine's running, starved and blocked time |
| 32–39 s | Fault: the next part lands tilted on the chamfer; the log reads "part not seated"; the robot lifts it out and places it again | the robot's log, as in the v3 clip |
| 39–44 s | Top view, 4× speed: several cycles across both machines | idle fraction per machine |
| 44–48 s | End card | learned against scripted results, "simulation only", repository link |

## Open questions

1. **Layout:** A (machines replace towers B and C, outfeed on tower A) or B (an outer ring, the full cell kept)? A is recommended.
2. **Door:** a sliding hood (recommended) or a hinged lid? Should the door be learned in v4.0 (recommended, the headline) or scripted, leaving one new learned skill?
3. **Two machines in parallel or in series?** Parallel means the same operation at different speeds, the classic tending case, and is recommended for v4.0. In series, every part goes through M1 and then M2, which is a flow shop with a buffer and a harder scheduling problem.
4. **Clearance:** 2 mm per side as the bar, with 1 mm as a reported condition? Or 1 mm as the bar?
5. **What machining does to a part:** a colour change from raw grey to a finished colour is proposed. Should finished parts feed tower recipes (the large scope)?
6. **Start after v3.2's release?** v3.2's final cell run is pending. Starting v4 after it keeps the cell's skills fixed while tending is built on them.
7. **Unrecoverable machine faults:** is "the order fails" acceptable, or should the cell model an operator call?
8. **Grasp assist:** keep the weld for carried parts (disclosed, as now), and no assist on the door handle. Agreed?
9. **Numbering:** v4.0 / Milestone 13, counting v3.2 as Milestone 12?
10. **Compute:** medium needs about 6–10 three-seed runs. Is the shared machine's capacity fine for that pace alongside the other projects?
