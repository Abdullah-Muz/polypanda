# Reach task definition

The reach skill moves the TCP (see [environment.md](environment.md#tool-centre-point-tcp)) to a 3-D target position. Orientation is not constrained in v1.

Environment: `polypanda.environments.reach.ReachEnv`, registered as `PolyPanda/Reach-v0`. Configuration is in `configs/reach.yaml`; unknown keys are rejected.

## Interface

**Action:** `Box(-1, 1, (7,), float32)`, sent through the joint-increment adapter ([control_design.md](control_design.md)) at 20 Hz.

**Observation:** schema `reach-v1-27d`, `float32`, world frame (the robot base is at the world origin):

| Slice | Content | Size |
|---|---|---:|
| 0–6 | joint positions `q` (rad) | 7 |
| 7–13 | joint velocities `dq` (rad/s) | 7 |
| 14–20 | commanded joint targets `q_cmd` (rad) | 7 |
| 21–23 | TCP position (m) | 3 |
| 24–26 | `target − TCP` (m) | 3 |

**Reset:** the arm starts at `home`, optionally with uniform joint noise (`start_joint_noise`). Targets are drawn by the sampler below, at least 0.10 m from the start TCP. `reset(options={"target": xyz})` injects a fixed target, which the evaluation set uses.

## Reward

Every term is returned separately in `info["reward_terms"]`.

| Term | Value | Purpose | Exploit risk |
|---|---|---|---|
| progress | `20 · (d_prev − d)` | Dense guidance | Telescopes to `20 · (d_start − d_end)`, so oscillating earns nothing |
| success | `+5` once | Reward finishing | Only paid for a physical success held through the dwell |
| action | `−0.005 · ‖a‖²` | Smooth, economical motion | If too large, standing still becomes optimal |
| contact | `−0.1` per step in contact | Avoid self and floor collisions | None obvious |
| joint_limit | `−0.01` per joint target at a bound | Avoid jamming into the stops | None obvious |
| non_finite | `−10`, then terminate | Numerical blow-up guard | Should never trigger |

Scale check: an efficient reach of 0.36 m earns about `+7 + 5 − 1.4 ≈ +11`. Standing still earns about 0, and a random policy about −2.7 (measured below). Success clearly dominates the do-nothing optimum.

## Termination

- **Success** (`terminated`): TCP within 0.02 m of the target for 5 consecutive policy steps (0.25 s). `info["is_success"]` is set, which SB3 uses for success-rate logging.
- **Time limit** (`truncated`): 200 policy steps (10 s). This is reported as truncation, not termination, so value targets bootstrap through timeouts instead of treating them as terminal states.
- **Non-finite state** (`terminated`): with `info["failure"] = "non_finite"`.

## Random-policy sanity baseline

30 episodes, seeds 0–29, uniform random actions:

| Metric | Value |
|---|---|
| Success | 0 / 30 |
| Return | −2.70 ± 1.50 |
| Mean distance, start → end | 0.378 m → 0.397 m |
| Mean reward terms per episode | progress −0.37, action −2.32, contact 0, joint_limit 0 |
| Single-process throughput | about 1,860 env steps/s (≈46,000 physics steps/s) |

SB3's `check_env` passes with no warnings.

## Targets (`polypanda.environments.targets`)

Targets are **reachable by construction**:

1. Sample a joint configuration uniformly within each joint's range, minus a 0.1 rad margin from each limit.
2. Run forward kinematics and collision detection with the fingers open.
3. Reject the sample if any contact exists (self-collision or floor) or if the TCP lies outside the workspace box.
4. Return the TCP position together with the joint configuration that reached it (the *witness*).

Workspace box (world frame): x ∈ [0.25, 0.75] m, y ∈ [-0.45, 0.45] m, z ∈ [0.10, 0.65] m. This is in front of the robot and above the floor. The TCP at home is (0.5545, 0, 0.5216) m.

Measured with 2000 targets, seed 0 (`uv run python scripts/inspect_targets.py`):

| Quantity | Value |
|---|---|
| Acceptance rate | 9.7 % (about 0.6 ms per target) |
| Distance from the home TCP | median 0.35 m, 5th–95th percentile 0.14–0.52 m |
| Density over a 4×4×4 grid | 7 / 34 / 54 targets per cell (min / median / max) |

![Reach target distribution](figures/reach_targets.png)

### Known bias

Uniform joint sampling is not uniform in space. Many configurations place the TCP close to the robot, and only nearly extended ones reach the far edge. Far, low targets (x > 0.6 m, z < 0.25 m) are about 7× rarer than the densest region, so the policy sees fewer of them during training. Evaluation will therefore report success by region as well as overall. A Cartesian-uniform sampler that uses IK to find a witness can be compared once the IK baseline exists.

### Caveat: witness configurations and gravity

The witness is a kinematic configuration. Commanding it directly through the servos lands a few millimetres away, because gravity sag of about 6.6 mrad at the shoulder becomes millimetres at the TCP. This does not affect learnability, since a policy closes the loop on the measured TCP. It does mean that feeding the witness straight to the servos is not a valid baseline.
