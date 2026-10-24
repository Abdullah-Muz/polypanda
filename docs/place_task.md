# Place task (Milestone 5)

Set a carried cube down **upright on a target spot**, let go and move the hand away. This is the last skill in the pick-and-place chain; the [supervisor](pick_and_place.md) hands over to it once transport has held the cube 12 cm above the target.

## 5.1: can a tilted cube be set down upright? (physics check)

The weak grip (about 1 N per finger) lets the cube pivot while it is carried: in transport it sits 15–39° off upright in the fingers. A 4 × 4 × 6 cm cube tips over once it leans past atan(2/3) ≈ 34° (the centre of mass passes the bottom edge), so the open question was whether a cube released at that tilt stands up or falls over.

Lowering cubes from real transport handoffs and releasing them just above the table gave the answer by release tilt:

| Tilt in the fingers at release | Toppled |
|---|---|
| 0–25° | 0 % |
| 25–35° | 11 % |
| over 35° | 100 % |

Every toppled cube ended on its side (90°): these are genuine falls, not transient wobbles caught by the classifier. A cube released below about 25° lands on its bottom face and settles upright. **The place skill must therefore correct the cube's tilt before letting go.**

## 5.2: environment (`polypanda.environments.place.PlaceEnv`)

- **Start states** are transport-to-place handoffs: the scripted carry brings a held cube to 12 cm above a target, and the state is captured once transport success has held for 10 consecutive steps (the supervisor's switch point). Targets are uniform in the cube placement region (x 0.35–0.65 m, y ±0.25 m) and at least 10 cm from where the cube was grasped. They are built from the transport pools, so all four grasp controllers are represented (`scripts/make_place_pool.py`).

| Pool | States | Use | Transport successes |
|---|---|---|---|
| `data/handoff/place_train_v1.npz` | 1190 | training starts | 1190 / 1200 |
| `eval_sets/place_nominal_v1.npz` | 198 | held-out evaluation | 198 / 200 |

- **Gated release.** The policy moves the arm; the fingers open by themselves once the cube's bottom is within 1 cm of the table **and** its centre is within 2 cm of the target, the same idea as GraspEnv's closing gate. A policy cannot drop the cube from a height.
- **Observation** (`place-v1-45d`): arm `q`/`dq`/`q_cmd`, TCP pose (6-D rotation), finger gap, gripper open flag, cube relative to the TCP, cube height and tilt, finger contacts, target and target − cube.
- **Reward:** telescoping progress towards the target (20 per m), a capped retreat reward for lifting the hand above the released cube, 0.5 per step while placed, action, action-rate, gripper-tilt and floor penalties, and −5 with termination for a drop before release or a topple (past 45°) after it.

### What counts as placed (`PlacedCheck`)

All of these must hold for 10 consecutive steps (0.5 s):

1. **resting**: the cube's bottom within 3 mm of the table;
2. **upright**: tilted less than 10°;
3. **on target**: centre within 2 cm of the target, horizontally;
4. **still**: every component of the cube's velocity below 0.01 m/s (or rad/s);
5. **released**: no contact between the cube and the hand or fingers.

Tests stage the cases the check must reject: a cube still pinched in the air, and a cube lying on its side on the target (`tests/test_place.py`).

## 5.3: scripted place baseline (`polypanda.control.scripted_place.ScriptedPlace`)

1. **UPRIGHT.** At the pre-place height, rotate the hand so the cube stands upright while its centre is held in place. The required hand orientation follows from the cube's pose in the hand: R_tcp' = Rz(cube yaw) · R_relᵀ, where R_rel is the cube's orientation in the hand frame. This phase ends when the tilt is below 5°, or after 2 s.
2. **LOWER.** Steer the cube's current (privileged) centre, in closed loop, to 3 mm above its resting pose on the target, keeping the upright orientation. The gate opens the fingers.
3. **WAIT** 0.4 s for the cube to settle, then **RETREAT** 10 cm straight up.

Evaluated on all 198 held-out cases at commit `e2c59bc` (clean):

| Metric | Without upright correction (development run) | **With upright correction** |
|---|---|---|
| Verified placement | 90.4 % | **97.5 %** (95 % CI 94.2–98.9 %) |
| Final xy error, median | 6.3 mm | **1.7 mm** (p95 16.7 mm) |

With the correction, the final tilt is 0.0° (median), the median time to a verified placement is 2.05 s (p90 3.90 s), and the arm never touches the floor. By grasp source: PPO seed 0 94 %, seed 1 100 %, seed 2 100 %, scripted 95.9 %. The five failures: 2 toppled, 1 off target, 1 never released (the gate never saw the cube both low and on target) and 1 not settled. Raw outputs: [`results/place/`](../results/place/).

The upright correction also fixed most of the xy error: a tilted cube's centre is not where its footprint lands, so straightening it first makes the target meaningful.

## 5.4: RL for placing (not attempted)

The approved plan made RL for placing conditional on the baseline leaving room to improve. At 97.5 % with five scattered failures it does not. Transport had already shown that a learned residual on a strong scripted controller can match it but not beat it. The effort went into the supervisor instead.
