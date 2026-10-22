# Control design

## Layers

```
policy / classical baseline ──a ∈ [-1,1]^7──▶ JointDeltaActionAdapter ──q_cmd──▶ Menagerie position servos ──τ──▶ MuJoCo
         ▲                                     (20 Hz)                            (500 Hz PD, per joint)
         └──────────────── observation: q, dq, q_cmd, TCP, target error ◀───────────────┘
```

PolyPanda does **not** add its own torque controller. The Menagerie Panda's actuators already implement `tau = kp (q_cmd - q) - kd dq` (gains in [environment.md](environment.md)). Stacking a second PD loop on top would double-count the feedback. A custom torque controller would be a separate, clearly labelled experiment.

## Action adapter (`polypanda.control.action_adapter`)

| Parameter | Value | Effect |
|---|---|---|
| `max_delta` | 0.015 rad | Largest target change per action, per joint |
| `n_substeps` | 25 | Physics steps per action: 25 × 0.002 s = 0.05 s, i.e. 20 Hz |
| Maximum commanded joint speed | 0.3 rad/s | `max_delta / control_dt` |
| `max_lead` | off (reach) | Optional anti-windup bound on `q_cmd - q` |

Per action: validate (shape `(7,)`, finite), clip `a` to [-1, 1], integrate `q_cmd += max_delta · a`, optionally bound to `q ± max_lead`, clip to the actuator ctrlrange, write `data.ctrl`, step 25 times. The gripper control is never written by the adapter.

### Resets and hand-offs

- `reset_to_measured` (episode start) sets `q_cmd = q`.
- `adopt_ctrl` (switching from one policy to another) keeps the servo commands that are already active. Resetting to `q` at a hand-off would make the targets jump by the gravity offset (about 6.6 mrad at the shoulder), which is a discontinuity the supervisor must avoid.

### Why `q_cmd` is observed

Because actions are integrated, `q_cmd` is hidden state. Without it, the same observed `q` could correspond to very different commanded futures. The gap between `q_cmd` and `q` is not small:

| Joint | Lag at 0.3 rad/s (mrad) | Residual after settling (mrad) |
|---|---:|---:|
| joint1 | 22.7 | 0.00 |
| joint2 | 14.2 | -8.69 (gravity) |
| joint4 | 28.6 | 5.73 (gravity) |
| joint6 | 24.2 | 1.35 |
| joint7 | 24.4 | 0.86 |

A PD servo tracking a ramp of speed `v` lags by `(kd / kp) · v`. Here kd/kp = 0.1 for every joint, so the prediction at 0.3 rad/s is 30 mrad, matching the measurements. At full speed the arm is about two actions (≈0.1 s) behind its commands.

### Target drift and windup

If the arm is blocked (by contact or a joint limit), integration keeps pushing `q_cmd` away from `q`, the servo force grows, and the arm lurches when the block clears. Mitigations:

1. Targets are clipped to the ctrlrange, and `at_limit` reports which joints are saturated, for diagnostics and reward terms.
2. `max_lead` bounds `|q_cmd - q|`. It is off by default for free-space reaching, where the lag above stays under 30 mrad, and it is meant for contact-rich skills.
3. Targets are reset at every episode boundary.

Measurements come from the development laptop. Re-run them after changing gains, `max_delta` or `n_substeps`.
