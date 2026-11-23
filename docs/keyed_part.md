# The keyed part (v4.0)

*Built 2026-11-23 ([v4_plan.md](v4_plan.md), section 3.4). A line-only option: the fixed v1–v3.2 cell and every default are unchanged.*

**Why.** A plain 4 × 4 × 6 cm block fits the square pocket any way round, so in Phase 0 the lead-in rescued every aiming error. A keyed part fits one way round only, so a part that arrives turned wrong cannot be rescued by the lead-in. Real fixtures are keyed for this reason: to make wrong loading impossible (poka-yoke).

## Design

- **The rib** (`LineConfig.keyed_parts`, `simulation/line.py: add_rib`): a 10 × 8 × 12 mm tab (wide × proud × tall) on the part's local +x face, centred, standing on its bottom. It is a second geom on the part's body. The part's box stays `cube.geom`, so the grasp check, colours and observations are unchanged. The rib has the part's density (1.0 g against 96 g) and moves the centre of mass by 0.24 mm, which is negligible. It sits under the pocket's 15 mm rim, and the fingers close 19 mm or more above the part's bottom, so they never touch it (tested). It is drawn dark so that the part's orientation reads at a glance.
- **Why 8 mm proud, not 5.** Across the rib the part measures 40 + proud mm, against the pocket's 44 mm. With a 5 mm rib the interference was only 1 mm. A wrong part pressed in by the arm then wedged between the walls, held only by MuJoCo's soft contacts, and crept down to the floor within 8 s. Only the orientation check refused it. At 8 mm the interference is 4 mm, and a wrong part stays up on its rib.
- **The slot** (`MachineConfig.keyed`, `key_wall`, `key_clearance`; `machine.py: _add_keyed_wall`): a slot through the full height of one pocket wall (the back wall by default) and its lead-in. It is 14 mm wide (2 mm clearance each side) and 10 mm deep from the pocket face, so a part pushed against that wall still has 2 mm to spare. A 3 mm block closes the slot behind and stands 1 mm proud of the 10 mm wall. The slot's mouth has no chamfer of its own.
- **Seated** (`Machine.in_pocket`, `Machine.key_error`): in a keyed pocket a ribbed part must also point its rib at the slot within 10°. `key_error` reports the angle (0 means the right way round, ±90 or 180 a wrong one) and returns None for a plain part or an unkeyed pocket. A plain part still fits a keyed pocket any way round.
- **Scripted load** (`control/scripted_tending.py`):
  - `load` turns a keyed part modulo 360° rather than 90°, by the shortest wrist turn within joint 7's range (with 0.05 rad to spare).
  - The turn is limited by the lid's opening. The hand (about 200 × 65 mm) passes the 240 × 100 mm opening only lengthwise, so at the machine the wrist can turn the part by about 0° or 180° (± the machine's turn), never 90°. The first build failed exactly here: a 90° turn drove the hand into the roof.
  - `pick_for(part, machine)` therefore chooses the grasp at the pick. Of the four face pairs, and the wrist turns joint 7 allows at both the pick and the machine, it takes the grasp with the least turning in all. At the pick the wrist may turn up to 180°.
  - A part held the wrong way (e.g. by the cell's plain `pick`) is regrasped. `load` sets it down at the machine's stop and picks it again with `pick_for` (tested).
  - With the grasp chosen this way, every rib direction can be loaded for machine turns within about ±27°, which covers the sweep's ±25°. Beyond that the remaining cases need a regrasp; `load` raises if none works.
- **Where the rib faces** (`ScriptedTending.rib_yaw`): read from the simulator. A real cell would read it with the camera, since the rib stands out from the part's side and is visible from above. The belief could carry it with that camera's error. That is not modelled here.

## Probe (`scripts/probe_keyed_insertion.py`, `results/v4/keyed_insertion_probe.json`)

*Measured 2026-11-23 from `87a224c` with a clean tree.*

**Setup.**
- The robot stands at M1's stop. A keyed part waits beside the machine at ±1.5 cm, and its yaw is drawn at random over the full circle, so the rib can start pointing anywhere.
- The script picks the part with `pick_for` and loads it with `load`. The place is aimed at the pocket's centre plus 0–3 mm in a random direction.
- "Wrong" means the script believes the slot is 90, 180 or 270° round from where it really is, so it sets the part in with its rib facing a plain wall.
- 30 trials per condition, the same draws in every condition: 480 trials in all.

| Orientation | 0 mm | 1 mm | 2 mm | 3 mm aiming error |
|---|---|---|---|---|
| right | 30/30 | 30/30 | 30/30 | 30/30 |
| wrong by 90° | 0/30 | 0/30 | 0/30 | 0/30 |
| wrong by 180° | 0/30 | 0/30 | 0/30 | 0/30 |
| wrong by 270° | 0/30 | 0/30 | 0/30 | 0/30 |

- **Right way round:**
  - All 120 parts seat. The rib lands within 3° of the slot.
  - The wrist turned by about 0° (105 trials) or 180° (15).
  - No regrasps and no IK failures.
  - As in Phase 0, aiming error costs force, not success. The median peak force between part and pocket rises from 0 N at 0 mm to 15 N at 3 mm, with a maximum of 60 N. The fingers touched the machine in 1 of 30 trials at 2 mm and 9 of 30 at 3 mm.
- **Wrong way round:**
  - No part seats, by the machine's check or physically: none of the 360 is down on the floor and level.
  - 286 rest on the rib tilted about 19–22°, with the gap under the part's centre about 5 mm.
  - 73 rest level but raised (the rib up on the lead-in).
  - One (wrong by 270°, 3 mm) tipped out over the rim.
  - None crept down within 5 s.
  - The closed-loop descent pressed the refused parts with a median of 23–39 N (up to 57 N).

The expected result held: right ≈ 100 %, wrong ≈ 0 %. Unlike the plain block, an orientation error is not rescued by the lead-in.

## What a learned place would have to do

The learned grasp's observation uses the cube's body pose and `align_sin4`/`align_cos4`, which have a 90° symmetry. It cannot tell which face carries the rib. A learned place into a keyed pocket would need:
- the rib's direction in its observation (a 360° yaw, e.g. sin/cos of the rib's angle to the slot);
- a choice of grasp at the pick that leaves a feasible turn at the machine, as `pick_for` makes. This is a planning step through the wrist's range and the opening, not a reflex;
- the turn itself, made before the hand enters the opening.

Arriving wrong is not recoverable by pushing. The part sits on its rib, and the script's closed-loop descent pressed it there with 20–50 N. So a learned place could add value by noticing a refused part (it stays raised by 5–12 mm) and lifting and turning it, instead of pushing.

## Limitations

- The orientation comes from the simulator. Camera error on the rib's direction is not modelled. A wrong reading of 90° or more is what the "wrong" conditions stand for.
- Soft contacts: wrong parts were checked to stay up for 5 s after release (`seated_after_5s`). The 5 mm rib showed that a small interference can be squeezed through, so the rib was sized to make that physically impossible, not just refused by the check.
- The script does not detect a refused part. It lets go regardless, and the machine simply never counts it as seated, so it will not start.
- One key wall per pocket and one rib per part. The slot's mouth has no chamfer. The `nominal` and `camera` beliefs in `evaluation/tending.py` assume an unkeyed machine. A keyed variation sweep would need beliefs that carry `keyed=True`.
