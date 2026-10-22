# Results

All numbers come from [`scripts/evaluate.py`](../scripts/evaluate.py) on the held-out set `reach_nominal_v1` (200 targets, digest `5b4bf7e75eec`) under the [evaluation protocol](evaluation_protocol.md). Raw outputs are in [`results/reach/`](../results/reach/).

## Reach: baselines (Milestone 2)

Evaluated at commit `774d38d` (clean tree). MuJoCo 3.16.0, Menagerie `cda5236f`.

| Policy | Success (95 % CI) | Time to success, median / p90 | Held after success | Final distance, median | Path ratio | Action jerk | Contact episodes | Latency, median / p99 |
|---|---|---|---|---|---:|---:|---:|---|
| Random | 0.0 % (0.0–1.9 %) | — | — | 375 mm | 2.50 | 4.67 | 0 % | 10 / 47 µs |
| Jacobian IK (k = 0.2, λ = 0.05) | 100.0 % (98.1–100 %) | 1.65 / 2.35 s | 100 % | < 0.001 mm | 1.00 | 0.015 | 0 % | 70 / 296 µs |

By region (success rate, n): IK reaches 100 % in all four regions (far-high 71, near-high 68, near-low 35, far-low 26). Random reaches 0 % in all four.

### Reading these numbers

- The **random policy** sets the floor. It never succeeds, ends a median 375 mm from the target (about where it started), and wanders along a path 2.5× longer than the straight-line distance.
- The **IK baseline** sets the ceiling for free-space reaching. It succeeds every time, follows a near-straight Cartesian path (ratio 1.00), holds every target, and its integral action removes the servos' gravity sag completely. Its advantage is an exact kinematic model; see the information-asymmetry note in [reach_task.md](reach_task.md#baselines-polypandacontrolclassical_baseline).
- Time to success is mostly set by the 0.3 rad/s joint speed limit of the action interface, not by the controller.
- Latencies are for single-observation `predict()` calls on the development laptop. Both are far below the 50 ms control period.

## Reach: learned policies (Milestone 3)

Not yet trained. PPO and SAC results will be added here under the same protocol.
