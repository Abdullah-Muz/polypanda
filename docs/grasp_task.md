# Grasp task (Milestone 4, in progress)

Steps so far: 4.1 scene, 4.2 pose IK, 4.3 grasp predicate, 4.4 scripted baseline (results at the end). Next: the grasp environment for RL.

## Scene (`polypanda.simulation.scenes`)

`load_pick_scene()` adds one free cube to the Panda scene through `MjSpec`. Menagerie files are not modified.

| Property | Value | Source |
|---|---|---|
| Size | 4 × 4 × 6 cm (half-sizes 0.02, 0.02, 0.03 m) | Menagerie `mjx_single_cube.xml` |
| Mass | 0.096 kg (density 1000 kg/m³) | — |
| Friction | 1.0 sliding, 0.03 torsional, 0.003 rolling | Menagerie |
| Contact `solref` | (0.01, 1): stiffer than default | Menagerie |
| Rest pose at `home` | (0.5, 0, 0.03) m, upright | — |

The robot stands on the ground plane, which serves as the table surface. This is the usual table-mounted setup and the same layout as Menagerie's own pick scene. The cube's 7 free-joint `qpos` entries come after the arm's, so all arm indices and reach code are unchanged. Measured at rest: under 0.5 mm of settling drift and penetration, contact with the floor only.

## Gripper facts (measured or derived from the model)

- TCP frame: **z points out of the gripper** (straight down at `home`) and **the fingers close along y** (world x at `home`). Fully open, the fingers are 8 cm apart.
- The fingertips reach **1.03 cm below the TCP**, so grasping at the cube's mid-height (TCP z = 0.03 m) leaves about 2 cm of clearance above the floor.
- **Grip force is small.** `actuator8` behaves like a 100 N/m spring pulling the finger tendon toward closed (force range ±100 N). Closed on a 4 cm cube, the tendon is 2 cm from its closed target, giving about **2 N of squeeze (≈1 N per finger)**. With friction 1.0 that holds roughly 2 N against the cube's 0.94 N weight, a margin of about 2×. Step 4.4 checks this in simulation.

## Pose IK (`polypanda.control.ik`)

`PoseIK` solves damped least squares on the 6-D TCP pose: position, plus orientation weighted 0.2 m per rad. It uses a private copy of the arm model, so it never touches simulation state.

- `top_down_quat(closing_yaw)` builds the grasp orientation: TCP z = world −z, TCP y = (cos ψ, sin ψ, 0).
- `grasp_closing_yaw(cube_yaw, current)` uses the square cross-section: four closing yaws grasp the same cube, and the one nearest the wrist's current yaw is chosen, so the wrist turns at most 45°.
- `orientation_error` is a world-frame rotation vector, computed with `mju_subQuat` so it stays valid for large errors.
- `PoseIK.action(...)` turns the IK step into a speed-limited, direction-preserving 7-D action, the same scheme as the reach IK baseline.

### Where top-down grasps are feasible

For 300 random cube placements (x 0.25–0.75 m, y ±0.40 m, any yaw), IK was solved from `home` for a pre-grasp pose (TCP 12 cm above the cube centre) and a grasp pose (TCP at the cube centre). A placement counts as feasible if IK converges (under 0.1 mm and 0.001 rad error), every joint stays at least 0.02 rad from its limit, and nothing touches except cube–floor.

- 286 / 300 feasible overall. Every failure was at an extreme: closer than about 0.3 m to the base (the joints hit their limits folding back) or about 0.8 m away (out of reach pointing straight down at floor level).
- Inside x 0.30–0.70 m, y ±0.30 m: **190 / 190 feasible**.

**Cube placement region for the grasp task: x 0.35–0.65 m, y ±0.25 m, any yaw.** This lies inside the fully feasible box with margin, and a test re-checks 25 random placements.

## Can the gripper hold the cube? (physics check)

The arm was teleported to the IK grasp pose around the cube, the gripper closed (control 0), the arm lifted 10 cm over 1 s, and the cube held for 2 s, at four placements:

| Placement | Squeeze after closing, L / R | While holding, L / R | Lifted | Slip in hand | Tilt |
|---|---|---|---|---|---|
| (0.50, 0.00), yaw 0 | 1.02 / 0.98 N | 1.00 / 0.99 N | 9.71 cm | 2.45 mm | 0.7° |
| (0.45, 0.15), yaw 0.6 | 1.01 / 0.99 N | 1.00 / 0.99 N | 9.71 cm | 2.51 mm | 2.3° |
| (0.60, −0.20), yaw −1.0 | **2.09 / 0.00 N** | 1.00 / 0.99 N | 9.45 cm | 4.73 mm | 8.1° |
| (0.40, 0.20), yaw 2.5 | 1.00 / 1.00 N | 1.00 / 1.00 N | 9.69 cm | 2.76 mm | 6.4° |

- The squeeze matches the actuator model exactly: about 1 N per finger.
- The cube sags a few millimetres in the grip and can tilt by several degrees, which is consistent with the roughly 2× friction margin.
- In the third case only one finger was pressing right after closing, because the cube was slightly misaligned and got pushed. This is why the grasp predicate requires **both** fingers.
- MuJoCo combines contact friction as the **maximum** of the two geoms' coefficients. The effective grip friction is therefore max(finger pad, object), so lowering only an object's friction has no effect unless that object's geom has higher `priority`. This matters for friction randomization in Milestone 6.

## Grasp predicate (`polypanda.simulation.grasp`)

`GraspMonitor.update(data)` is called once per policy step. The cube counts as **grasped** only when all of the following hold for 5 consecutive policy steps (0.25 s):

1. **Both** fingers are in contact with the cube (normal force > 0 on each).
2. The cube is at least **2 cm** above its resting height.
3. It is **stable in the hand**: its pose in the TCP frame moves less than **2 mm** and rotates less than **0.05 rad (~3°)** across the window.

The returned `GraspState` also reports per-finger contact and normal force, lift, held steps and slip, for diagnostics and reward terms.

Each test stages a situation physically, and only the real grasp is allowed to pass:

| Situation | Contacts | Lift | Grasped? |
|---|---|---|---|
| Gripper above a resting cube | none | 0 | no |
| Fingers squeezing the cube on the table | both, about 1 N each | < 2 cm | no |
| Cube in the air, away from the gripper | none | 37 cm | no |
| Cube off-centre, so one finger touches first | one | 0 | no (single-finger steps detected) |
| Closed, lifted 10 cm and held (3 placements) | both | > 8 cm | **yes**, after 5 stable steps |
| Gripper opened while lifted | none | falls to < 1 cm | no |
| Cube made slippery after grasping | — | — | no, lost within 5 steps |

## Scripted grasp baseline (`polypanda.control.scripted_grasp`)

A state machine: **APPROACH** (TCP 10 cm above the cube centre, gripper open, closing yaw aligned to the cube) → **DESCEND** (to the cube centre at half joint speed) → **CLOSE** (gripper control 0 for 0.5 s) → **LIFT** (10 cm) → **HOLD** (1 s) → **DONE**.

- The arm moves **only through the 7-D joint-increment action interface**, steered by `PoseIK` with gain 0.2, the same integral-action scheme as the reach IK baseline. There is no teleporting.
- Phases advance on measured TCP pose error (under 4 mm and 0.03 rad). Each moving phase has a 6 s timeout, which is reported as a failure.
- Privileged information: the true cube pose and an exact kinematic model.

### Held-out evaluation (`scripts/evaluate_grasp.py`)

- Placement set `eval_sets/grasp_nominal_v1.json`: 200 placements, uniform over x 0.35–0.65 m, y ±0.25 m, yaw ±π. Drawn from `SeedSequence(20261022, spawn_key=(0x6A5B,))`, digest `ad1eeb22b054`, regenerated exactly by a test. Regions: near (horizontal radius under 0.5 m) 84, far 116.
- **Success** means the verified grasp predicate holds at the **end** of the episode, after the 1 s hold.

Results (commit `2249cd3`, clean):

| Metric | Value |
|---|---|
| Success | **200 / 200 = 100 %** (95 % CI 98.1–100 %); near 84/84, far 116/116 |
| Time to first verified grasp | median 6.95 s, p90 8.50 s |
| Episode length (to DONE) | median 8.43 s |
| Phase durations (median) | approach 3.7 s, descend 2.05 s, close 0.5 s, lift 1.0 s, hold 1.0 s |
| Cube pushed before the fingers closed | 0.00 mm (max 0.00 mm) |
| Cube tilt in the grip | median 4.7°, p95 11.5°, max 14.8° |
| Lift achieved / finger force | 9.3 cm (10 cm commanded) / 1.00 N per finger |
| Episodes with arm–floor contact | 0 % |

**Reading this:**
- With exact knowledge of the cube pose, top-down grasping of this cube is reliable across the whole placement region.
- The only imperfection is **tilt**: with about 2 N of squeeze and a roughly 2× friction margin, the cube pivots up to ~15° in the fingers during the lift, more often for placements close to the robot. It never drops.
- The baseline is deliberately unhurried (half-speed descent, 1 s hold), so a learned policy has room to be faster.
- Watch it: `uv run python scripts/demo_grasp.py --episodes 5`.
