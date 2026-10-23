# Grasp task (Milestone 4, in progress)

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
