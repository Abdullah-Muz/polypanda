# v4.0 plan: a learned machine-tending skill for machines it has not seen

*Adopted on 2026-11-22, after the v3.2 release. This plan builds on [machine_tending_candidates.md](machine_tending_candidates.md) and takes its medium scope. It adds two things that doc did not have: a learned skill whose job is machines it was not written for, and training that starts from demonstrations.*

*Amended the same day, after Phase 0 and the camera measurement (section 3.1). The plan first bet that realistic sensing error would make the scripted place fail, so that a learned place could win. Both measurements say it does not. On the user's decision, the learned skill's job is now **machine variation that a camera does not reveal**: one learned door-and-load skill for machines it has not seen, with no reprogramming.*

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
| Layout | **A:** machines M1 and M2 take the places of towers B and C; tower A's fixture becomes the outfeed stack |
| Door | **a sliding hood, opened and closed by a learned skill**, with a scripted door beside it |
| The learned skill's job (amended 2026-11-22, user) | **machines it has not seen**: the door and load skill is trained with RL on randomized machine variants and tested on held-out ones, against a script written for one machine and a script that locates the handle with the camera (section 3.2) |
| Two machines | **in parallel**, with different cycle times |
| Clearance of the fixture pocket | **2 mm per side as the bar**; 1 mm reported as a condition |
| Numbering | v4.0 / Milestone 13 |

The candidates doc's other proposals stand as written: machining recolours the part, an unrecoverable machine fault fails the order, and the grasp assist stays for carried parts and is not used on the door handle.

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

## 4. Phases

Each phase has a go/no-go gate and ends with something shown publicly. Durations are rough, and this project's estimates have been optimistic before. Together they come to roughly 5–7 weeks.

| Phase | Work | Gate | Shown publicly | Rough length |
|---|---|---|---|---|
| 0 | Tight-fit feasibility probe: the scripted place puts a cube into a pocket at 2 mm and 1 mm clearance | the contacts are stable and the part seats; otherwise the fixture is loosened or the scope drops to small | the probe's result | days |
| 1 | The machine model with adjustable variants; the scripted door and load (nominal and camera-aware); the variation sweep: both scripts on machines drawn further and further from the nominal one, no training | **the make-or-break experiment:** is there machine variation that the camera-aware script cannot handle? (amended 2026-11-22; the first version asked about sensing error, answered no in section 3.1) | the variation curve: where each script starts to fail | 1–2 weeks |
| 2 | The learned door and load: demonstrations, then PPO on randomized machines; `v4_targets.md` and its held-out variants committed first | the learned skill meets its development bar on held-out variants within the stopping rule | the learned skill on machines it has not seen | within the 4–5 weeks below |
| 3 | Unload, scheduling, recovery; the tending cell with injected faults, run once on a held-out suite; the release | the final run is locked in a commit before it is made | the tending cell and its results | tending in the medium scope: 4–5 weeks (candidates doc) |

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
| Do the machines stand idle? | idle time per machine, split into starved and blocked, for the chosen rule against first-come-first-served |
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
