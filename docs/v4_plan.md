# v4.0 plan: a machine-tending cell that keeps running when things go wrong

*Adopted on 2026-11-22, after the v3.2 release. This plan builds on [machine_tending_candidates.md](machine_tending_candidates.md) and takes its medium scope. It adds three things that doc did not have: sensing with realistic error, training that starts from demonstrations, and a browser demo.*

*Nothing here is a target yet. Targets, evaluation sets and their digests go into `v4_targets.md` before any v4 training, under the project's usual rules: 3 seeds, held-out sets drawn before training, Wilson 95 % intervals, exact McNemar tests on paired cases, a scripted baseline beside every learned result, and every result reported as it comes, misses included.*

## 1. The problem

Machine tending means a robot loads raw parts into a machine, such as a CNC mill, and unloads the finished ones. It is an established use of robots, and demand is growing: one analyst forecasts collaborative robot shipments to grow about 17 % a year from 2025 to 2030 ([Interact Analysis, via Robotics 24/7](https://www.robotics247.com/article/report-17.3-growth-forecast-for-collaborative-robot-shipments-from-2025-30)). Market-size estimates for tending robots differ by a factor of about three between firms, so they are not quoted here ([one of them](https://persistencemarketresearch.com/market-research/machine-tending-robots-market.asp)).

What goes wrong in practice, according to vendor and trade sources:

- **Tending depends on parts being exactly where the program expects them.** A vendor guide names part presentation as a limit: reliable tending needs consistent fixturing and part orientation, and a misaligned part interrupts the workflow ([Standard Bots](https://standardbots.com/guides/guide-to-cobot-machine-tending)).
- **Custom fixturing is costly for small batches.** It is one of the things that makes high-mix, low-volume work inefficient ([NIST](https://www.nist.gov/node/1631246)).
- **Unattended running does not remove faults.** Shops that run overnight plan set responses for misaligned parts, failed grip checks and full output bins ([Mitsubishi Electric](https://us.mitsubishielectric.com/fa/en/resources/blog/assets/lights-out-manufacturing/)), and still need a person who can be called in ([MSC Direct](https://www.mscdirect.com/knowledge-center/articles/definitive-guide-lights-out-manufacturing)).

These are vendor and trade sources, not measurements, so they are directional. No study of why tending projects fail was found.

**What v4.0 sets out to show:** a tending cell that copes with parts that are not exactly where expected, checks its own work, and recovers from faults without a person.

**The limit, stated up front:** this is simulation only. v4.0 can show the approach with honest numbers. It cannot claim to have fixed a factory problem, and nothing here has controlled hardware.

## 2. Decisions (2026-11-22)

The user said to proceed with the candidates doc's recommended picks. They stay revisable until `v4_targets.md` is committed.

| Question in the candidates doc | Decision |
|---|---|
| Scope | **medium** |
| Layout | **A:** machines M1 and M2 take the places of towers B and C; tower A's fixture becomes the outfeed stack |
| Door | **a sliding hood, opened and closed by a learned skill**, with a scripted door beside it |
| Two machines | **in parallel**, with different cycle times |
| Clearance of the fixture pocket | **2 mm per side as the bar**; 1 mm reported as a condition |
| Numbering | v4.0 / Milestone 13 |

The candidates doc's other proposals stand as written: machining recolours the part, an unrecoverable machine fault fails the order, and the grasp assist stays for carried parts and is not used on the door handle.

Added to that scope: the three items of section 3.

## 3. Three additions to the candidates doc

### 3.1 Imperfect sensing

- **Why:** the scripted baselines win in this project because they read the simulator's exact positions. A real robot sees parts through a camera, with error. Work on insertion reports that controllers jam at narrow clearances when the socket's position is uncertain, and that learned or residual policies are the usual remedy ([variable socket position](https://arxiv.org/pdf/1810.01531), [tight insertion under perception noise](https://arxiv.org/pdf/2505.11858)). With realistic pose error, the tight fixture place becomes a task where a learned skill has a real reason to win. Without it, the candidates doc itself expects the scripted door to reach about 100 % and the scripted place to be strong.
- **What:** part and fixture positions come from a simulated camera estimate with a measured error model, not from the simulator. The estimator's error is measured on rendered images of held-out scenes. As candidate C7 proposed, training uses a noise model fitted to that error, because rendering is too slow to put the camera inside training.
- **Feasibility, measured on the development machine** (no NVIDIA GPU; an integrated Radeon 780M; 2026-11-22, the v3.2 cell scene):

  | Measurement | Result |
  |---|---|
  | Offscreen depth rendering, 128 × 128, one process | about 51 frames per second |
  | Offscreen colour rendering, 84 × 84 to 224 × 224, one process | about 7 frames per second |
  | Physics, single thread | about 4,500 steps per second (9 × real time) |

  So training a pose estimator on rendered images is feasible, and training a policy directly from pixels is not. Distilling a state-based policy into a visual one is the common alternative ([state-to-visual DAgger](https://arxiv.org/html/2412.13662v1)); at these frame rates it is out of reach here.
- **Evaluation:** every case is run with exact poses and with estimated poses, paired. Beside it, an error-tolerance curve: success against pose error, for the scripted skill and the learned one.
- **Status change:** this replaces candidate C7's "optional, large scope only" status. Sensing is part of v4.0.

### 3.2 Demonstrations first

- **What:** record the scripted controller doing each new task, train the policy to copy it (behaviour cloning or DAgger, state inputs, on CPU), then refine it with PPO, or with a residual policy on top.
- **Why:**
  - Recent sources describe imitation for initialisation and RL for refinement as the common recipe, and a review states that imitation alone is capped by the quality of its demonstrations ([review](https://arxiv.org/pdf/2607.06706)).
  - One paper reports about 97 % success for a residual RL policy on top of an imitation policy, against about 80 % for imitation alone ([ResiP](https://arxiv.org/html/2407.16677v4)).
  - This project's grasp took 9 training rounds and its stand-up 12. The door has no policy to warm-start from, and the candidates doc names exploration as its likely bottleneck.
- **Caveat:** no paper was found in which the refined policy beats a scripted expert that has perfect information. That is why this addition pairs with imperfect sensing.
- **A proving ground before it is relied on:** one existing skill, with a check committed before the run.

### 3.3 A browser demo

- **What:** a web page that runs the trained policies in the visitor's browser, on MuJoCo's official WebAssembly bindings with onnxruntime-web. Public demos already run trained policies this way ([mujoco_wasm](https://github.com/zalo/mujoco_wasm/), [a walking policy in the browser](https://github.com/ETHRoboticsClub/v1-walk-demo)).
- **Scope:** start with the v3.2 cell or a single skill. The IK, the gates and the observation building must be ported or simplified.
- **Fidelity check:** replay a committed episode in the browser and compare its trajectory with the Python one.
- **Training:** none.

## 4. Phases

Each phase has a go/no-go gate and ends with something shown publicly. Durations are rough, and this project's estimates have been optimistic before. Together they come to roughly 6–9 weeks.

| Phase | Work | Gate | Shown publicly | Rough length |
|---|---|---|---|---|
| 0 | Tight-fit feasibility probe: the scripted place puts a cube into a pocket at 2 mm and 1 mm clearance | the contacts are stable and the part seats; otherwise the fixture is loosened or the scope drops to small | the probe's result | days |
| 1 | Browser demo | the browser replay matches the Python trajectory | the demo link | 1–2 weeks |
| 2 | The machine model and its fixture; the camera estimate and its error model; the error-tolerance experiment; the proving ground of 3.2; then the fixture place | **the make-or-break experiment:** is there a range of realistic pose error where the scripted place fails and a learned one does not? | the error-tolerance curve, whichever way it comes out | 1–2 weeks on top of the tending work |
| 3 | The learned door | the door meets its development bar within the stopping rule | the door skill | within the 4–5 weeks below |
| 4 | Unload, scheduling, recovery; the tending cell with injected faults, run once on a held-out suite; the release | the final run is locked in a commit before it is made | the tending cell and its results | tending in the medium scope: 4–5 weeks (candidates doc) |

**Phase 0 result:**

<!-- PROBE-RESULT -->

- `v4_targets.md` is committed before any training, so before the training in Phase 2.
- Phases 0 and 1 need no training.
- If Phase 2 shows that realistic error does not hurt the scripted place, the place is still built, and the result is reported as it is: no advantage for the learned skill.

## 5. What will be measured

Drafts until `v4_targets.md`.

| Question | Measure |
|---|---|
| How far off can a part be before each approach fails? | success of the fixture place against pose error, scripted and learned, on the same cases |
| Does sensing cost anything? | every case with exact and with estimated poses, paired |
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
| **Work orders in plain English** | quick to build | not a learned skill; adds an outside dependency |
| **Cycle time** | throughput is what a shop buys | partly covered by the idle-time measure |

## 7. Risks

- **Tight-fit contact in the simulator.** Box-in-box contact at 1–2 mm can jam or jitter. Phase 0 measures it before anything is designed around it.
- **The learned skills may still not beat the script under realistic error.** The result is reported either way.
- **Estimates.** Day counts have been optimistic before: skills have taken more rounds than planned.
- **The browser port's fidelity.** The IK, gates and observations are re-implemented, so the demo could drift from the Python cell. The replay check is there to catch that.
- **Simulated images are clean.** A camera estimate in simulation says little about real vision; the error model, not the images, is what carries over.

## 8. Sources

- [Standard Bots: guide to cobot machine tending](https://standardbots.com/guides/guide-to-cobot-machine-tending) (vendor)
- [NIST: cobots and custom fixturing in high-mix, low-volume work](https://www.nist.gov/node/1631246)
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
- [mujoco_wasm](https://github.com/zalo/mujoco_wasm/), [a walking policy in the browser](https://github.com/ETHRoboticsClub/v1-walk-demo)
- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground), [its SO-101 pick-cube recipe](https://github.com/google-deepmind/mujoco_playground/tree/main/mujoco_playground/tasks/manipulation/pick_cube/so101/deploy)
- [SO-101 kit cost](https://physicalailab.io/en/so-101-robot-arm-kit-cost-buy-setup/)
