# Reach task definition

The reach skill moves the TCP (see [environment.md](environment.md#tool-centre-point-tcp)) to a 3-D target position. Orientation is not constrained in v1.

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
