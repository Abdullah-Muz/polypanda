# v4.1 plan: a robot that runs an electronics test cell

*Drafted on 2026-11-24, after the v4.0 release, for the user's approval. It replaces the v4.1 of [v4_plan.md](v4_plan.md) (a second machine and a flip): the user chose to grow the line into a realistic electronics test cell instead, after research into what arms like ours do in industry (section 9). The second machine and flip, the quality station of the old v4.2, and a learned scheduler stay candidates for later.*

*Nothing here is a target yet. Targets, evaluation sets and their digests go into `v4_1_targets.md` after the scripted baselines are measured and before any v4.1 training, under the project's usual rules: 3 seeds, held-out sets drawn before training, Wilson 95 % intervals, exact McNemar tests on paired cases, a scripted baseline beside every learned result, and every result reported as it comes, misses included. One lesson from v4.0 is written in from the start: **the number of development rounds is fixed here (section 6), and it is not raised later.***

## 1. The scenario

**Industry.** Contract electronics manufacturing: the companies that build and test devices for brands (Foxconn, Jabil, Flex). Their lines are **high-mix**: products change every few weeks, and the test floor is a mixed fleet of testers from different vendors. Loading test stations is the most common cobot job in electronics (Sennheiser's UR5 loads boards into testers, output up 33 %; Universal Robots' guidance for in-circuit testing), and it is still often done by hand: shield-box vendors write that the device is loaded manually and the rest is automated (section 9).

**The part.** A small electronic module, for example an industrial sensor module: a rigid board in a plastic housing, with contact pads underneath that meet the nest's pogo pins. It is **keyed**: it fits its test nest only one way round.

*Amended 2026-11-24, at the start of Phase 2:* the module is **40 × 40 × 60 mm**, the size of the cell's existing parts, not the 50 × 40 × 15 mm first written here. Every pick, place and grasp the cell has (the learned grasp included) works on that size unchanged, and v4.0's keyed rib gives it its one-way fit; it is drawn as a sensor module (dark housing, a board edge, a label). The nest is v4.0's machine pocket made tight with `machine.tight_nest` (clearance 0.25–0.5 mm, a small chamfer, keyed, stiff contacts on both sides via `simulation/contacts.py` and `LineConfig.part_contact`).

**The testers.** RF shield boxes and functional testers on the bench. The module goes into a **nest** with a tight fit (a few tenths of a millimetre) and a small chamfer, the box closes, and a test runs for 30–60 s. Real boxes come in three kinds, and the cell has one of each (all three differ from tester to tester in size, lid travel, latch and nest position, drawn per machine as in v4.0):

| Tester | How it opens | What the robot does |
|---|---|---|
| A: lid | a lid that slides (as v4.0's machine), latched closed | opens and closes it: **the learned door** |
| B: drawer | a drawer that slides **toward the robot** | pulls it out and pushes it home: **the learned door, a new slide direction** |
| C: pneumatic | opens and closes itself on a signal | only loads and unloads |

**The operation.**

```
 conveyor in ──►  [ rail ═════════════════════════════════════ ]
 (untested modules)    the robot drives along the rail
                  [Tester A]   [Tester B]   [Tester C]
                  [scanner + vision]   [pass tray]   [red quality zone: fail bin, rework bench]
```

1. An untested module arrives on the conveyor; the robot picks it and **scans its barcode**.
2. It drives to a free tester, **opens it** (learned door on A and B; C opens itself).
3. It **seats the module in the nest**: tight fit, the nest's position known only through the camera (about 1 mm of error), the module's place in the hand not exact. **Learned insertion: the new skill.**
4. It closes the tester; the test runs. Meanwhile it serves the other testers (a scripted rule: whichever tester finishes first).
5. When the test ends: open, unload, show the module to the **vision station**, then sort: **pass → the pass tray**, **fail → the fail bin** in the red quality zone. Every module carries a hidden truth (about 5 % fail the electrical test, about 3 % have a visible defect), so sorting is scored against ground truth.

*Amended 2026-11-24 (user decisions after the first scripted dev run):* the module is **inspected on the fly**: the robot carries it to the vision camera and stops briefly with the module's front (label and defect face) turned to the camera, without setting it down; and a module the robot cannot get tested (it would not seat after its retries, it was knocked out of its seat before the test started, or its test was aborted) goes to a **rework tray** as **not tested**, counted on its own in the KPIs, not as a false reject and not in the test yield.

## 2. Decisions (user, 2026-11-24)

- **Scenario:** the electronics test cell (option A of the brainstorm), over a machining job shop.
- **Learned skills:** the door (from v4.0, extended to tester lids and drawers) and **precision insertion into the nest** (new). Picking, driving, scanning, inspection and scheduling are scripted in v4.1.
- **Environment:** restyled as an electronics factory (section 5), as a separate theme so the v2/v3 cell still renders as before.
- **No LinkedIn post for now.**

## 3. Phase 0 result: tight nests in this simulator (2026-11-24)

*Measured from `c4a6e1a` (the probe script added in `bc8a668`; `results/v4/nest_insertion_probe.json`).* The scripted cell set a held 4 cm cube into a nest straight down with the unchanged `ScriptedCell.place`: clearance 0.1, 0.25 or 0.5 mm per side, a 0.5 or 1 mm chamfer on the rim, the nest turned 0–2° from where the script believed it was, the place aimed 0–3 mm off, with the cell's default contacts and with stiff ones. 15 trials per condition, 2,160 in all.

- **Contacts.** With the default contact settings the cube and the nest interpenetrate by 0.3–1.0 mm (95th percentile), more than the clearances, so results at this scale would not mean anything. With stiff contacts (solref 0.004, the stiffest stable value at the 2 ms timestep; solimp 0.95/0.99 over 0.5 mm) the depth stays at or below 0.21 mm. **Decision: v4.1's nests and modules use the stiff contacts, and nests have at least 0.25 mm clearance** (0.1 mm is below what this simulator resolves).
- **What a straight-down insertion tolerates: about the chamfer.** Machines seated out of 15, stiff contacts:

  | Aim error | 0 mm | 1 mm | 2 mm | 3 mm |
  |---|---|---|---|---|
  | 0.5 mm chamfer | 13–15 | 1–7 | 0 | 0 |
  | 1 mm chamfer | 15 | 9–13 | 0–2 | 0–2 |

  Jammed parts press on the nest with 15–30 N, which on hardware means bent pins and scratched housings. Even with no aim error the tightest nest seated 10 of 15, because the cube does not sit exactly in the hand.
- **A nest turned 1–2°** mostly self-aligns through the chamfer and the arm's give.

**What it means.** The realistic error in the cell (about 1 mm of camera error, the part's offset in the hand, tester-to-tester variation) is in the range where a straight-down insertion fails most of the time. But industry does not insert tight parts straight down: it touches down, feels the jam and searches (section 4.2). **The learned insertion is compared with that force-guided search script, not with the straight-down one**, and the search script is built and measured before any target is written (Phase 2).

## 4. The learned skills

### 4.1 The door, on testers

v4.0's door works on sliding lids. v4.1 adds:
- **drawers** that slide toward the robot (a new slide direction for the door environment), with a handle on the front instead of on top;
- **tester-shaped lids:** different sizes, handle shapes and latch forces, drawn per tester as v4.0 drew machines;
- **hand-over robustness:** training starts drawn from a wide range of hand poses (heights from a few centimetres to about 35 cm above the handle), so no careful hand-over is needed. T5 showed why (v4.0: 67 % of machines running with the line's hand-over, 95 % with the trained one).

It is retrained from v4.0's round 5, so it keeps what it learned. Its scripted baselines are v4.0's `camera` and `feel_v2`, extended to drawers.

### 4.2 Precision insertion into the nest (new)

- **The task:** the module, held after the pick, goes into a tester's nest. The robot sees a camera estimate of the nest's pose (about 1 mm and 0.5° of error, as measured in v4's pose-error work), its own arm, and an estimate of the force on the hand. It does not see where the module sits in its hand, the nest's exact pose, or the chamfer. Success: seated (on the nest's floor, flat, inside the clearance), released, hand clear, with no force above a damage limit.
- **Scripted baselines:**
  - **straight-down** (what the probe measured);
  - **force-guided search:** touch down; if the module stops short, keep a light downward force and move in a growing spiral until it drops in, then release. This is the standard industrial method (force-controlled search on Flexiv and ABB cobots; the classical stage of NVIDIA and Foxconn's connector work; section 9).
- **Training:** as the door: demonstrations from the search script first, cloned, then PPO; 3 seeds.
- **Where learning can win:** the search script will likely seat nearly every part, but slowly: each spiral costs seconds. A learned insertion can read the first contact and correct in one move. So, as with the door, its likely edge is **time to seat at the same reliability, and less force on the nest**. Industry judges insertion the same way: a success rate held over many trials, and a cycle time (NVIDIA and Foxconn's targets: at least 99.5 % and a time limit; section 9).

## 5. The environment

Restyled as an electronics factory, all of it drawn only (no collisions), so the physics of the cell does not depend on it:

1. **An SMT line behind the cell:** magazine loader, stencil printer, solder-paste inspection, pick-and-place machines, reflow oven, optical inspection, OK/NG unloader, with green boards moving along its conveyor.
2. **Light, colour, floor:** bright, cool, even light from ceiling LED panels; white, light grey and pale blue with ESD blue and green accents; a light ESD floor, green ESD-area boundary lines, yellow walkways.
3. **The testers:** brushed-metal shield boxes and drawers with a gasket line, a small status screen and a **light tower** (green ready, amber testing, red fail) driven by the tester's real state.
4. **The module:** a black housing with a green board edge, gold connector pads and a barcode label.
5. **A live andon / test dashboard screen:** per-tester state, units tested, yield, takt time, updated from the simulation.
6. **Reel storage:** a smart reel tower and glass dry cabinets with green/amber/red interior lights.
7. **A burn-in rack** with rows of blinking unit LEDs.
8. **5S floor colours and a red quality zone** with the fail bin and a rework bench.
9. **ESD details:** EPA signs, a wrist-strap tester at the entrance, ESD smocks, overhead ionizers with task lights above the benches.
10. Optional: a supermarket flow rack with an AMR carrying trays; overhead cable trays and compressed-air drops.

A **look-and-feel still** (a few rendered images) is made for the user's approval before the full build.

## 6. Phases

| Phase | What | Gate | Time |
|---|---|---|---|
| 0 (done) | Tight-nest probe (section 3) | contacts trustworthy at ≥ 0.25 mm with stiff contacts: **passed** | ½ day |
| 1 | This plan → user's approval; look-and-feel still → user's approval | approvals | ½ day |
| 2 | Build: the module, nests and three tester kinds; the drawer in the door environment; the insertion environment; the **straight-down and search scripts, measured**; scanner, vision station, pass tray, quality zone; the cell runner; the electronics theme | both scripted insertions measured on development nests; the cell runs end to end with scripted skills | 2–2½ days |
| 3 | `v4_1_targets.md`: targets and sets, committed before training | user's approval | ½ day |
| 4 | Train: insertion (demonstrations, clone, PPO) and the door on testers. **At most 3 development rounds per skill, fixed now** | dev results | 1½–2 days |
| 5 | Lock, then the one-shot sets (door, insertion) and the cell run, each once | results as they come | ½ day |
| 6 | Video (the storyboard: conveyor, tester, nest close-up with a force trace, test lamp, vision, sorting, a shift time-lapse with the dashboard), write-up, release | user's approval of the clip | ½–1 day |

About **6–7 days**.

## 7. What will be measured

- **Door on testers (one-shot set of unseen testers):** success per seed and per tester kind; time; force; against `camera` and `feel_v2` (extended to drawers). From any hand-over start.
- **Insertion (one-shot set of unseen nests and grips):** success per seed, with the industry-style check: at 99.5 % with 95 % confidence, about 600 trials with no failure; time to seat; peak force on the nest; against straight-down and search.
- **The cell (a set of shifts):** modules tested per hour, tester utilization, modules sorted correctly against their hidden truth, failures by stage; the cell with the learned skills against the all-scripted cell, on the same shifts.

## 8. Not in v4.1, and why

- **Learned picking, inspection and scheduling:** each is a project of its own; v4.1 keeps one new learned skill. A learned scheduler (which tester to serve next) is the strongest candidate for v4.2.
- **A flexible cable or ribbon connector:** deformable cables are slow and fiddly in MuJoCo on CPU.
- **Screwdriving:** the simulator has no threads.
- **A second machine and a flip (the old v4.1), the quality station (the old v4.2):** later, if wanted; the test cell already has inspection and sorting.

## 9. Risks

- **The search script may be hard to beat on reliability.** Then the learned insertion's case is time and force, reported as such, as the door's was.
- **A rigid grasp.** The cell holds parts with a grasp assist (a weld), so the module cannot give in the fingers as on real hardware; jams are harsher than real. Kept for consistency with v4.0, and stated in every result.
- **Contacts at this scale.** Stiff contacts pass at 0.25 mm and wider; anything tighter is not claimed.
- **Scope.** Three tester kinds, a new skill, a retrained door and a restyle are a lot for one release. If time runs short, the order to cut: the pneumatic tester, then the optional environment items, then the drawer.

## 10. Sources

- Electronics test tending: [Assembly Magazine, Sennheiser](https://www.assemblymag.com/articles/99366-cobot-boosts-output-by-33-percent); [Universal Robots, PCBA handling](https://www.universal-robots.com/industries/electronics-and-technology/pcba-handling/); [UR electronics stories](https://www.universal-robots.com/blog/3-cobot-success-stories-in-electronics-manufacturing/).
- Shield boxes and nests: [US8208273](https://patents.google.com/patent/US8208273); [EP1308733](https://data.epo.org/publication-server/rest/v1.2/patents/EP1308733NWA1/document.pdf); [Tescom TC-5901B](https://www.atecorp.com/getmedia/8256741e-5ab2-4b57-9d15-18c0995ab9bd/tescom-5901b_datasheet_1.pdf).
- Force-guided insertion and learning: [Flexiv FPC insertion](https://flexiv.com/blog/1753263715848949760); [NVIDIA and Foxconn](https://developer.nvidia.com/blog/the-machines-that-make-the-machines/); [IndustReal](https://arxiv.org/pdf/2305.17110); [HIL-SERL](https://arxiv.org/pdf/2410.21845); [NIST assembly task boards](https://www.nist.gov/node/1340176).
- Environment: [ANSI/ESD S20.20](https://blog.ansi.org/ansi/ansi-esd-s20-20-2021-protection-electronic-parts/?amp=1); [SMT line layout](https://smtnet.com/library/files/upload/I.C.T%20SMT%20Line%20Solution%20%20V1.0..pdf); [ECD SmartDRY](https://ecd.com/products-solutions/dry-storage/smartdry/); [Essemtec Tower](https://www.electronicspecifier.com/products/test-and-measurement/essemtec-tower-saves-dry-packs-reduces-temperature-load-smd-components/); [Desco overhead ionizers](https://www.digikey.com/en/product-highlight/d/desco/chargebuster-overhead-ionizers); [5S floor marking](https://www.plantservices.com/management/lean-six-sigma/article/11335607/floor-marking-5s-floor-marking-tape-color-standards-take-your-workplace-one-step-closer-to-lean-plant-services).
