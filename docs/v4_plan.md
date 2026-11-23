# v4.0 plan: a learned machine-tending skill for machines it has not seen

*Adopted on 2026-11-22, after the v3.2 release. This plan builds on [machine_tending_candidates.md](machine_tending_candidates.md) and takes its medium scope. It adds two things that doc did not have: a learned skill whose job is machines it was not written for, and training that starts from demonstrations.*

*Amended the same day, after Phase 0 and the camera measurement (section 3.1). The plan first bet that realistic sensing error would make the scripted place fail, so that a learned place could win. Both measurements say it does not. On the user's decision, the learned skill's job is now **machine variation that a camera does not reveal**: one learned door-and-load skill for machines it has not seen, with no reprogramming.*

*Amended again the same day (user decisions, section 3.4): **the robot moves to a linear rail and serves a straight production line.** Raw keyed blanks go through M1 (op 1), are flipped, go through M2 (op 2), and are checked at a quality station that scans them. The work ships in three releases: v4.0 is the rail line with one machine and the learned door on machines it has not seen; v4.1 adds the second machine and the flip; v4.2 adds the quality station.*

*Nothing here is a target yet. Targets, evaluation sets and their digests go into `v4_targets.md` before any v4 training, under the project's usual rules: 3 seeds, held-out sets drawn before training, Wilson 95 % intervals, exact McNemar tests on paired cases, a scripted baseline beside every learned result, and every result reported as it comes, misses included.*

## 1. The problem

Machine tending means a robot loads raw parts into a machine, such as a CNC mill, and unloads the finished ones. It is an established use of robots, and demand is growing: one analyst forecasts collaborative robot shipments to grow about 17 % a year from 2025 to 2030 ([Interact Analysis, via Robotics 24/7](https://www.robotics247.com/article/report-17.3-growth-forecast-for-collaborative-robot-shipments-from-2025-30)). Market-size estimates for tending robots differ by a factor of about three between firms, so they are not quoted here ([one of them](https://persistencemarketresearch.com/market-research/machine-tending-robots-market.asp)).

What goes wrong in practice, according to vendor and trade sources:

- **Tending depends on parts being exactly where the program expects them.** A vendor guide names part presentation as a limit: reliable tending needs consistent fixturing and part orientation, and a misaligned part interrupts the workflow ([Standard Bots](https://standardbots.com/guides/guide-to-cobot-machine-tending)).
- **Custom fixturing is costly for small batches.** It is one of the things that makes high-mix, low-volume work inefficient ([NIST](https://www.nist.gov/node/1631246)).
- **Every new machine or part means reprogramming.** A trade article argues that a changeover taking hours cannot pay for a short run, and that changeover in minutes is what makes small-batch automation pay ([Modern Machine Shop](https://www.mmsonline.com/articles/the-future-of-machine-shop-automation-why-flexibility-remains-the-competitive-advantage)). Small shops often lack a dedicated robot programmer ([NIST](https://www.nist.gov/node/1631246)), and flexibility is named as a key requirement for pre-engineered tending systems in high-mix shops ([Automate](https://www.automate.org/robotics/industry-insights/low-barriers-to-entry-help-pre-engineered-robotic-machine-tending-solve-the-labor-shortage-gap/aph)).
- **Unattended running does not remove faults.** Shops that run overnight plan set responses for misaligned parts, failed grip checks and full output bins ([Mitsubishi Electric](https://us.mitsubishielectric.com/fa/en/resources/blog/assets/lights-out-manufacturing/)), and still need a person who can be called in ([MSC Direct](https://www.mscdirect.com/knowledge-center/articles/definitive-guide-lights-out-manufacturing)).

These are vendor and trade sources, not measurements, so they are directional. No study of why tending projects fail was found.

**What v4.0 sets out to show:** a door-and-load skill, learned with RL, that tends machines it was not trained on and was not programmed for, inside a tending cell that checks its own work and recovers from faults without a person.

**The limit, stated up front:** this is simulation only. v4.0 can show the approach with honest numbers. It cannot claim to have fixed a factory problem, and nothing here has controlled hardware.

## 2. Decisions (2026-11-22)

The user said to proceed with the candidates doc's recommended picks. They stay revisable until `v4_targets.md` is committed.

| Question in the candidates doc | Decision |
|---|---|
| Scope | **medium** |
| Layout | ~~**A:** machines M1 and M2 take the places of towers B and C; tower A's fixture becomes the outfeed stack~~ **Superseded (2026-11-22, user): a rail line** (section 3.4). The circle around the fixed base was full |
| The robot | **on a linear rail (a "7th axis")**, serving a straight line of stations; rail moves scripted (section 3.4) |
| The part | **a keyed blank** that fits its fixture only one way round (section 3.4) |
| The process | **two operations with a flip between them** (M1, then M2), visible machining, then **a quality station** that scans the part (section 3.4) |
| Releases | **v4.0** rail line, one machine, the learned door on unseen machines, keyed part; **v4.1** second machine and flip; **v4.2** quality station (section 4) |
| Door | **a sliding hood, opened and closed by a learned skill**, with a scripted door beside it |
| The learned skill's job (amended 2026-11-22, user) | **machines it has not seen**: the door and load skill is trained with RL on randomized machine variants and tested on held-out ones, against a script written for one machine and a script that locates the handle with the camera (section 3.2) |
| Two machines | ~~**in parallel**, with different cycle times~~ **in series** from v4.1: op 1 on M1, op 2 on M2 (section 3.4) |
| Clearance of the fixture pocket | **2 mm per side as the bar**; 1 mm reported as a condition |
| Numbering | v4.0 / Milestone 13 |

The candidates doc's other proposals stand as written: an unrecoverable machine fault fails the order, and the grasp assist stays for carried parts and is not used on the door handle. Machining no longer only recolours the part: from v4.1 the finished part shows what was cut (section 3.4).

Added to that scope: the items of section 3.

## 3. Additions to the candidates doc

### 3.1 Imperfect sensing: measured, and why it no longer carries the plan

**The result (2026-11-22, [pose_error.md](pose_error.md), `results/v4/pose_error.json`, measured from `e2f9f53`).** A classical estimator works on simulated depth images: background removal, clustering and the top face's rectangle, with no learning. When it returns a pose, the cube's position is off by a median of 0.25–1.8 mm, depending on the depth noise assumed. The noise levels run from none, through a Kinect-v1 model, to a D435-class sensor 1.5 m away. Apart from about 1 % of estimates where two neighbouring cubes merged into one cluster (10–35 mm off), no estimate was more than about 4 mm off. A camera near the machine does better, under 1 mm at the 95th percentile at every noise level. Phase 0 showed the scripted place still seats the part at 4 mm of aiming error. **So under realistic sensing, the scripted place does not fail, and a learned place has no success rate to win.** It would not fix merged clusters either; better segmentation would. The simulated sensor is cleaner than a real one, and the probe varied position, not yaw, so a rotated part remains a cheap open check.

**What sensing still does in v4.0:** the camera estimate stays as the source of part, fixture and handle positions, with its measured error model, so that no controller reads exact positions from the simulator. It is no longer where the learned skill is expected to win.

The original reasoning, kept for the record:

- **Why:** the scripted baselines win in this project because they read the simulator's exact positions. A real robot sees parts through a camera, with error. Work on insertion reports that controllers jam at narrow clearances when the socket's position is uncertain, and that learned or residual policies are the usual remedy ([variable socket position](https://arxiv.org/pdf/1810.01531), [tight insertion under perception noise](https://arxiv.org/pdf/2505.11858)). With realistic pose error, the tight fixture place becomes a task where a learned skill has a real reason to win. Without it, the candidates doc itself expects the scripted door to reach about 100 % and the scripted place to be strong.
- **What:** part and fixture positions come from a simulated camera estimate with a measured error model, not from the simulator. The estimator's error is measured on rendered images of held-out scenes. As candidate C7 proposed, training uses a noise model fitted to that error, because rendering is too slow to put the camera inside training.
- **Feasibility, measured on the development machine** (no NVIDIA GPU; an integrated Radeon 780M; 2026-11-22, the v3.2 cell scene):

  | Measurement | Result |
  |---|---|
  | Offscreen depth rendering, 128 × 128, one process | about 51 frames per second |
  | Offscreen colour rendering, 84 × 84 to 224 × 224, one process | about 7 frames per second |
  | Physics, single thread | about 4,500 steps per second (9 × real time) |

  So training a pose estimator on rendered images is feasible, and training a policy directly from pixels is not. Distilling a state-based policy into a visual one is the common alternative ([state-to-visual DAgger](https://arxiv.org/html/2412.13662v1)); at these frame rates it would be slow here, so it is not planned for v4.0.
- **Evaluation:** every case is run with exact poses and with estimated poses, paired. Beside it, an error-tolerance curve: success against pose error, for the scripted skill and the learned one.
- **Status change:** this replaces candidate C7's "optional, large scope only" status. Sensing is part of v4.0.

### 3.2 The learned skill's job: machines it has not seen

- **The problem it stands for:** a scripted tending program is written for one machine: its door, its handle, its fixture. A new machine, or a machine moved a few centimetres, means reprogramming (section 1).
- **What varies between machines, in simulation:** where the machine stands and how it is turned; where the handle sits on the door and how high; which way the door slides (outward or sideways); how far it travels; how stiff it is to slide (friction and damping); how hard it must be pushed to latch; and where the fixture sits inside. Training draws these at random (domain randomization). The final test uses held-out variants, including some further from the training range.
- **What a camera reveals and what it does not:** a camera can find the handle and the fixture (section 3.1). It cannot see which way the door slides, how stiff it is or where it latches. Those have to be found by feel, which is where a policy that reacts to what it feels has a reason to beat a fixed program.
- **Two scripted baselines, so the comparison is fair:**
  1. the **nominal script**: written for one machine (the default variant), the way a cell is programmed today;
  2. the **camera-aware script**: the same program, but it reads the handle's and fixture's positions from the camera estimate. It still assumes the nominal slide direction, travel and latch force.
- **The learned skill:** the door policy (candidate C2) observes the handle's estimated position, its own arm, the gripper's contact and the door's motion while it pulls. It is trained with PPO, starting from demonstrations (section 3.3), on randomized machines.
- **Honest risk:** a camera-aware script given a little more logic (e.g. "pull, and if the door does not move, try sideways") may cope with much of this variation. The variation sweep in Phase 1 measures the scripts before any training, and if the camera-aware script copes, that is reported.

### 3.3 Demonstrations first

- **What:** record the scripted controller doing each new task, train the policy to copy it (behaviour cloning or DAgger, state inputs, on CPU), then refine it with PPO, or with a residual policy on top.
- **Why:**
  - Recent sources describe imitation for initialisation and RL for refinement as the common recipe, and a review states that imitation alone is capped by the quality of its demonstrations ([review](https://arxiv.org/pdf/2607.06706)).
  - One paper reports about 97 % success for a residual RL policy on top of an imitation policy, against about 80 % for imitation alone ([ResiP](https://arxiv.org/html/2407.16677v4)).
  - This project's grasp took 9 training rounds and its stand-up 12. The door has no policy to warm-start from, and the candidates doc names exploration as its likely bottleneck.
- **Caveat:** no paper was found in which the refined policy beats a scripted expert that has perfect information. That is why this addition pairs with machine variation the script was not written for (section 3.2). The demonstrations come from the script running on the variants it can still handle.
- **A proving ground before it is relied on:** one existing skill, with a check committed before the run.

### 3.4 The line: a rail, a keyed part, two operations and a quality station

*Decided 2026-11-22 by the user.*

**The rail.** The robot is mounted on a carriage that slides along a straight linear rail (a "7th axis"), as in real tending lines where one robot serves several machines. It replaces the fixed base, whose 0.5 m circle of stations was full (the infeed's belt housing, three towers, three buffer slots and the reject bin). The line is laid out so that **every station sits at the same standard pose in front of its rail stop: 0.5 m in front of the arm's base axis, at table height.** From the arm's point of view, a station on the line looks exactly like a station on the old circle, so every learned skill runs in its station frame unchanged (`StationView`). Moving along the rail is scripted ("drive to M2's stop"), as the base swings between stations are today. **The existing three-tower cell stays as it is, as its own configuration**, so v1–v3.2 results still reproduce. The line is a new configuration beside it.

```
  side A of the rail (stations at the standard pose in front of their stops)

   infeed          M1             M2           quality       outfeed
  conveyor       (op 1)         (op 2)         station        stack
   [==>]        [ ▭▭▭ ]        [ ▭▭▭ ]         [ ⌖ ]          [ ▮ ]
     |             |              |              |              |      stops
  ═══●═════════════●══════════════●══════════════●══════════════●═══   rail
     |        [ carriage + arm ] ───▶ drives between stops ◀───        |
   [bin]                      [buffer]
  side B of the rail (reject bin and buffer, reached by turning the arm's base)
```

- **What the rail costs:** about 1–1.5 weeks of groundwork, with no RL in it:
  - the carriage on a slide joint with a position servo;
  - IK and the scripted moves working from a moving base;
  - drive-to-stop moves;
  - the cell observer knowing where every station is;
  - the new line layout.
- **What it buys:** room for every station below, a layout that reads instantly as a production line, and travel time along the rail that makes machine idle time a real scheduling problem.

**A keyed part (v4.0).** The blank gets a notch or chamfered corner, and the pocket a matching key, so the part fits only one of its four orientations. Why: a plain square block fits any way round, so in Phase 0 the lead-in rescued every aiming error up to 4 mm. A keyed part brings back orientation errors a lead-in cannot absorb: a part turned 90° the wrong way does not go in at all. The footprint stays that of the blocks the skills were trained on, so the grasp mostly carries over. Placing the part in the right orientation is new.

**Two operations with a flip (v4.1).**
- **The process:** M1 cuts op 1. The robot unloads the part, flips it over and loads it into M2, which cuts op 2 on the other face. This is how two-sided parts are made: op 1 machines one side, op 2 the other, often on a second machine.
- **The flip:** reuses v3's learned turning and stand-up skill. Flipping onto a *specific* face is new for it.
- **Visible machining:** a finished part shows what was cut, e.g. a step or pocket after op 1 and a chamfer after op 2, not just a colour change. It is modelled by swapping the part's visual (and, where the scan needs it, collision) geometry when the operation finishes.

**A quality station (v4.2).**
- **The station:** a nest (a shallow fixture like the machine's pocket) with a scanner bar that sweeps over the part.
- **The scan is real computation, not a flag:** the simulated depth camera and the estimator of section 3.1 ([pose_error.md](pose_error.md)) render the part in the nest and measure the features each operation should have cut: the step or pocket depth, the keyed corner, the chamfer, the overall size.
- **Routing:** a pass goes to the outfeed. A fail goes to the reject bin, or back to the machine for rework when an operation was skipped.
- **Defects are injected,** as v3 injected knocks: a skipped operation; a wrong flip, so op 2 cut the wrong face; a part damaged after a drop; a dimension off.
- **What is measured:** defects caught, false rejects (good parts failed; these rise with camera noise, using the noise model already measured) and escapes (bad parts that reach the outfeed).
- **Honest note:** this adds a complete process and quality numbers a factory would recognise. It does not add RL: the scan is classical measurement, and loading the nest reuses the place skill.

**Later (v4.x), not planned in detail:** a mix of part types, such as round bars and flanged parts, each machine with its own fixture, and the robot routing each part to the right machine. This needs the grasp retrained on new shapes, and the cube grasp alone took 9 rounds.

### 3.5 The machine

The machine is a benchtop machine sized for the arm's reach, like a small desktop mill or a test station. It is built in simulation from simple shapes with `MjSpec`, as the conveyor, towers and bin are, so that every property a variant changes is a number.
- **Body:** about 320 mm wide, 280 mm deep and 170 mm tall.
- **Lid:** a lid on top, over an opening of 240 × 100 mm, slides sideways or outward depending on the variant. The opening is sized for the hand reaching straight down, which measures about 200 × 65 mm. The robot pinches a handle bar on the lid.
- **Inside:** the pocket fixture of Phase 0 (44 mm square, 15 mm deep, a 45° lead-in 5 mm wide), keyed for the keyed part.
- **Status lamp:** green ready, amber running, blue done, red fault.
- **Physical parts** are only the walls, roof, lid, handle and fixture. The window, control panel, lamps, hazard trim and a spindle head (which lowers and spins over the part while the machine runs, for the video) are visual only.
- **The cycle:** the machine starts only with the part seated and the lid latched, keeps the lid locked while it runs, and unlocks it when done.
- **Variants:** handle position and height, slide direction and travel, slide friction, a spring, latch force, and the machine's position and turn, size and colour. These are drawn at random in training and held out for testing (section 3.2).

## 4. Releases and phases

Each phase has a go/no-go gate, and each release ends with something shown publicly. Durations are rough, and this project's estimates have been optimistic before. Across v4.0–v4.2 they come to roughly 8–10 weeks.

| Release | Phase | Work | Gate | Shown publicly | Rough length |
|---|---|---|---|---|---|
| v4.0 | 0 (done) | Tight-fit feasibility probe: the scripted place puts a cube into a pocket at 2 mm and 1 mm clearance | the contacts are stable and the part seats; otherwise the fixture is loosened or the scope drops to small | the probe's result | days (done 2026-11-22) |
| v4.0 | 1 | The rail and the line layout, as a new configuration beside the three-tower cell; the machine model with adjustable variants and the keyed pocket; the scripted door and load (nominal and camera-aware); the variation sweep: both scripts on machines drawn further and further from the nominal one, no training | the full test suite still passes and v3.2 results reproduce; then **the make-or-break experiment:** is there machine variation that the camera-aware script cannot handle? (amended 2026-11-22; the first version asked about sensing error, answered no in section 3.1) | the line, and the variation curve: where each script starts to fail | 2.5–3.5 weeks (the rail 1–1.5) |
| v4.0 | 2 | The learned door and load: demonstrations, then PPO on randomized machines; `v4_targets.md` and its held-out variants committed first; then the one-machine line run once on a held-out suite; the v4.0 release | the learned skill meets its development bar on held-out variants within the stopping rule; the final run is locked in a commit before it is made | the learned door on machines it has not seen | 2–3 weeks |
| v4.1 | 3 | The second machine; the flip between operations; visible machining; unload, scheduling with rail travel, recovery; the two-machine line run once on a held-out suite; the release | the final run is locked in a commit before it is made | the two-operation line | 2–2.5 weeks |
| v4.2 | 4 | The quality station: the nest, the depth-camera scan of the cut features, injected defects, pass and fail routing; the full line run once on a held-out suite; the release | the scan's checks and the defect suite are committed before the run | the full line with quality inspection | about 1 week |

**Phase 0 result:**

*Measured 2026-11-22 from `a444ae8` with a clean tree (`scripts/probe_fixture_insertion.py`, `results/v4/fixture_insertion_probe.json`).* The scripted cell picked an upright cube from the bench, carried it to a pocket at tower B's station and set it in with the unchanged `ScriptedCell.place`. The pocket was 15 mm deep: 10 mm of straight wall under a 45° lead-in 5 mm wide. The place was aimed at the pocket's centre plus an error of 0–6 mm in a random direction. Each condition ran the same 50 trials, 1,200 in all.

| Pocket (clearance per side, wall friction) | 0 mm | 1 mm | 2 mm | 3 mm | 4 mm | 6 mm aiming error |
|---|---|---|---|---|---|---|
| 2 mm, cube-like (1.0) | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 |
| 2 mm, steel-like (0.3) | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 |
| 1 mm, cube-like | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 42/50 |
| 1 mm, steel-like | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 35/50 |

- **Gate: passed.** The contacts are stable. A seated cube moves 0.0 mm over the next second, every seated cube was unloaded again with the scripted pick (1,177 of 1,177), and there were no pick or IK failures. The 2 mm pocket is workable as designed.
- **The scripted place is much more tolerant of position error than expected.** The lead-in and the arm's give guide the cube in at every aiming error up to 4 mm, and at 6 mm in the 2 mm pocket. It fails only at 1 mm clearance and 6 mm error: the cube rests tilted on the lead-in, or tips over off it.
- **What the error does cost:** wall forces grow with the error. The median peak goes from 0 N with exact aim to 30–50 N at 6 mm, with maxima up to 90 N. The fingers strike the pocket's rim in up to 21 of 50 trials at 4–6 mm. The probe counts these as successes; on a real machine they are collisions.
- **Caveats:**
  - MuJoCo's soft contacts let the cube press up to 1.9 mm into cube-like walls (under 1 mm into steel-like ones), which is more give than real steel has.
  - The cube is held rigidly by the grasp assist until it is released 4 mm above the pocket's floor.
  - The scripted place still reads the cube's true position in the hand; only the target is wrong.
- **What this means for Phase 1:**
  - If a camera's realistic error stays within about 4 mm, imperfect sensing does not make the scripted place fail. The learned place would then have no success-rate advantage to win, and that will be reported.
  - Its realistic chances are elsewhere: lower insertion force and no rim strikes under the same error, both measurable here. Or errors larger than the lead-in can absorb.
  - The realistic error range is being measured next, with a simulated depth camera.
  - *Measured the same day (section 3.1): realistic error stays within about 4 mm. The phase plan was amended accordingly.*

- `v4_targets.md` is committed before any training, so before the training in Phase 2.
- Phases 0 and 1 need no training.
- If Phase 1 shows that the camera-aware script copes with all the variation drawn, the learned door is still built and compared, and the result is reported as it is.

## 5. What will be measured

Drafts until `v4_targets.md`.

| Question | Measure |
|---|---|
| Does it work on machines it has not seen? | door-and-load success on held-out machine variants, learned against the nominal and the camera-aware script, paired by variant |
| How far can a machine differ before each approach fails? | success against how far the variant is from the nominal machine, per varied property (handle position, slide direction, travel, stiffness, latch force) |
| How gently does it work? | peak forces on the door and in the fixture, and collisions with the machine, on the same variants |
| Does sensing cost anything? | cases with exact and with estimated positions, paired (measured for the estimate itself in section 3.1) |
| How many faults are fixed with no person? | the share of injected, recoverable faults after which the order still completes; the physical ceiling is measured first with the scripted cell, as v3 did |
| Do the machines stand idle? | idle time per machine, split into starved and blocked, for the chosen rule against first-come-first-served, with the robot's travel along the rail counted (v4.1) |
| Does the quality station catch bad parts? | defects caught, false rejects (good parts failed, against camera noise) and escapes (bad parts that reach the outfeed), on an injected-defect suite (v4.2) |
| Learned against scripted | paired by case, each seed separately, 3 seeds |

## 6. Not in v4.0, and why

| Idea | For | Against |
|---|---|---|
| **Mixed-size box packing** (the strongest alternative) | palletizing and depalletizing were 83 % of the robotic picking market in 2025 ([Interact Analysis, via MMH](https://www.mmh.com/article/robotic_picking_market_to_reach_4.6_billion_in_2030/Robotics)); learned packing planners beat heuristics, 76.1 % against 57.9 % space use in one benchmark ([GOPT](https://arxiv.org/html/2409.05344v2)); stability in physics is an open problem this project could measure ([CoPack](https://arxiv.org/html/2511.19932v1)) | the research robots use suction, and a two-finger gripper needs gaps beside each box |
| **Bin picking from a heap** | the fastest-growing segment, about 36 % a year ([Interact Analysis, via Robotics 24/7](https://www.robotics247.com/article/report-robotic-bin-picking-robotic-trailer-unloading-drive-strong-market-growth)) | this project's own clutter attempt scored 21.5–67 % across seeds ([generalization.md](generalization.md)); the methods that work rely on camera images and GPUs |
| **A real arm** (a candidate for the milestone after) | SO-101 kits cost roughly $120–300 ([cost guide](https://physicalailab.io/en/so-101-robot-arm-kit-cost-buy-setup/)); [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) publishes a recipe that trains in simulation and runs on the real arm ([SO-101 pick-cube](https://github.com/google-deepmind/mujoco_playground/tree/main/mujoco_playground/tasks/manipulation/pick_cube/so101/deploy)) | it needs a rented GPU, a different robot and a different training stack; no published success rates were found for it |
| **Removing the grasp assist** | cell results would measure grip retention ([workcell.md](workcell.md#grasp-assist-a-disclosed-simplification)) | the fixes tried in Milestone 9 did not stop the slip; worth a two-day, time-boxed attempt with fixes not yet tried, such as a smaller time step |
| **A browser demo** (dropped 2026-11-22, user decision) | anyone could try the policies from a link; MuJoCo has official WebAssembly bindings ([mujoco_wasm](https://github.com/zalo/mujoco_wasm/)) | it does not advance the problem v4.0 is about; the IK, gates and observations would have to be re-implemented in JavaScript and kept identical |
| **Work orders in plain English** | quick to build | not a learned skill; adds an outside dependency |
| **Cycle time** | throughput is what a shop buys | partly covered by the idle-time measure |

## 7. Risks

- **Tight-fit contact in the simulator.** Box-in-box contact at 1–2 mm can jam or jitter. Phase 0 measures it before anything is designed around it.
- **The camera-aware script may cope with most machine variation,** especially with a little retry logic. Phase 1 measures this before any training, and the result is reported either way.
- **Breadth against cost.** The wider the randomized machines, the longer the door takes to train on CPU. The ranges are set from Phase 1's sweep, not guessed.
- **Sensing error in realistic ranges did not hurt the script** (section 3.1). The learned place is no longer expected to win on success rate.
- **The rail touches tested code.** The scripted moves, IK and observer assume a fixed base. Mitigation: the line is a separate configuration, the three-tower cell is left as it is, and the full test suite and a reproduction of committed v3.2 results gate Phase 1.
- **Scope growth.** The plan grew from one machine to a rail line with two operations and a quality station. Mitigation: the release split. v4.0 alone carries the RL result, and each later release stands on its own.
- **Estimates.** Day counts have been optimistic before: skills have taken more rounds than planned.
- **Simulated images are clean.** A camera estimate in simulation says little about real vision; the error model, not the images, is what carries over.

## 8. Sources

- [Standard Bots: guide to cobot machine tending](https://standardbots.com/guides/guide-to-cobot-machine-tending) (vendor)
- [NIST: cobots and custom fixturing in high-mix, low-volume work](https://www.nist.gov/node/1631246)
- [Modern Machine Shop: the future of machine shop automation and changeover](https://www.mmsonline.com/articles/the-future-of-machine-shop-automation-why-flexibility-remains-the-competitive-advantage) (trade)
- [Automate: pre-engineered robotic machine tending](https://www.automate.org/robotics/industry-insights/low-barriers-to-entry-help-pre-engineered-robotic-machine-tending-solve-the-labor-shortage-gap/aph) (trade)
- [Mitsubishi Electric: lights-out manufacturing](https://us.mitsubishielectric.com/fa/en/resources/blog/assets/lights-out-manufacturing/) (vendor)
- [MSC Direct: a guide to lights-out manufacturing](https://www.mscdirect.com/knowledge-center/articles/definitive-guide-lights-out-manufacturing) (vendor)
- [Interact Analysis: collaborative robot shipments, 2025–30](https://www.robotics247.com/article/report-17.3-growth-forecast-for-collaborative-robot-shipments-from-2025-30)
- [Interact Analysis: bin picking and trailer unloading](https://www.robotics247.com/article/report-robotic-bin-picking-robotic-trailer-unloading-drive-strong-market-growth)
- [Interact Analysis: the robotic picking market](https://www.mmh.com/article/robotic_picking_market_to_reach_4.6_billion_in_2030/Robotics)
- [Persistence Market Research: machine tending robots](https://persistencemarketresearch.com/market-research/machine-tending-robots-market.asp)
- [GOPT: online 3D bin packing with a learned policy](https://arxiv.org/html/2409.05344v2)
- [CoPack: bin packing in a physics engine and the real world](https://arxiv.org/html/2511.19932v1)
- [ResiP: residual RL for precise assembly](https://arxiv.org/html/2407.16677v4)
- [A review of vision-language-action models](https://arxiv.org/pdf/2607.06706)
- [State-to-visual DAgger against visual RL](https://arxiv.org/html/2412.13662v1)
- [Insertion with a variable socket position](https://arxiv.org/pdf/1810.01531)
- [Model-based control and RL for tight insertion](https://arxiv.org/pdf/2505.11858)
- [mujoco_wasm](https://github.com/zalo/mujoco_wasm/)
- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground), [its SO-101 pick-cube recipe](https://github.com/google-deepmind/mujoco_playground/tree/main/mujoco_playground/tasks/manipulation/pick_cube/so101/deploy)
- [SO-101 kit cost](https://physicalailab.io/en/so-101-robot-arm-kit-cost-buy-setup/)
