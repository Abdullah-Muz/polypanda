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

## Reach: PPO (Milestone 3)

Training: `configs/reach_ppo.yaml` on reach task **v2** (`configs/reach.yaml`). 200,704 environment steps per run, 8 `SubprocVecEnv` workers, observation and reward normalization, MLP 128×128 for both policy and value. The best checkpoint is chosen on a separate **validation** target stream, never the test set. Three seeds; each run took 4.6–5.0 min on the development laptop (CPU only).

Evaluated at commit `da4f5e0` (clean). Runs were trained at `dc3604b`, `748ef60` and `da4f5e0`, which differ only in docs and scripts.

| Policy | Success, pooled 600 episodes (95 % CI) | Held after success | Time to success, median / p90 | Final distance, median / worst | Path ratio | Action jerk | Contact episodes | Latency, median / p99 |
|---|---|---|---|---|---:|---:|---:|---|
| PPO v2, mean ± std over 3 seeds | 100 % (99.4–100 %) | 100 % | 1.70 ± 0.09 / 2.34 ± 0.06 s | 1.47 ± 0.25 / 5.8 ± 1.0 mm | 1.076 ± 0.007 | 0.044 ± 0.000 | 0 % | ~430 µs / 1.2–1.8 ms |
| Jacobian IK (for reference) | 100 % (98.1–100 %, 200 episodes) | 100 % | 1.65 / 2.35 s | < 0.001 / < 0.001 mm | 1.00 | 0.015 | 0 % | 70 / 296 µs |

Every seed reaches 100 % in all four regions, including the sparse far-low corner.

![PPO validation learning curves](figures/ppo_reach_learning.png)

*Validation success during training for the three seeds (`scripts/plot_learning.py`). Success means the target was reached and was still held at the end of the 10 s episode. Seeds first reach 100 % at 80k–100k steps, roughly 1.1–1.4 hours of simulated experience.*

### Reading these numbers

- **PPO matches IK on what the task asks for** (success and holding) using only its 27-D observation and about 2.8 hours of simulated practice (200k steps × 0.05 s). IK, by contrast, has an exact kinematic model.
- **IK remains better on quality.** PPO is about 0.05 s slower in median time to success, settles about 1.5 mm off-centre instead of exactly on target, and its actions are about 3× jerkier. The hold reward's gradient inside the 2 cm ball is shallow, so there is little pressure to remove the last millimetre.
- **Inference** is about 6× slower than IK (a 128×128 MLP plus normalization, through SB3's `predict`), but still about 100× faster than the 50 ms control period.

### Pilot on task v1 (diagnostic, superseded)

The first PPO run (seed 0, 200k steps, `configs/reach_v1.yaml`) reached all 200 held-out targets (median 1.73 s) but **held only 14.5 %** and ended a median 38 mm away. The final checkpoint drifted into contact in 11 % of episodes. v1 ended episodes on success, so the policy never experienced post-success states. This is why the task was revised to v2 (see [reach_task.md](reach_task.md#task-versions)). These pilot numbers are reported for transparency; their result files were not archived because the evaluation ran on an uncommitted working tree.

## Reach: SAC vs PPO (Milestone 3)

SAC training: `configs/reach_sac.yaml` on the same task v2, with the same 128×128 networks, the same 200k-step budget, the same validation stream and the same held-out evaluation as PPO. SAC uses 4 envs, one gradient step per collected transition, automatic entropy tuning (SB3 default target entropy −7), observation normalization only, and 1 PyTorch thread. Hyperparameters were fixed **before** seeing results; SAC was not tuned after it under-performed. Trained and evaluated at commit `62e55bb` (clean).

| Policy (3 seeds × 200 targets) | Success, pooled 600 episodes (95 % CI) | Held after success | Time to success, median / p90 | Final distance, median | Path ratio | Action jerk | Training wall clock per seed |
|---|---|---|---|---|---:|---:|---|
| **PPO** | **100 %** (99.4–100 %) | **100 %** | 1.70 ± 0.09 / 2.34 ± 0.06 s | **1.5 ± 0.3 mm** | 1.08 | 0.044 | **~5 min** |
| SAC | 63.2 % (59.2–66.9 %) | 43 ± 19 % | 1.98 ± 0.28 / 4.24 ± 0.16 s | 27 ± 6 mm | 1.16 | **0.032** | ~47 min* |
| Jacobian IK (reference) | 100 % | 100 % | 1.65 / 2.35 s | < 0.001 mm | 1.00 | 0.015 | — |

\* The three SAC seeds ran in parallel. A single SAC run alone measured about 94 steps/s, roughly 35 min for 200k steps.

Per-seed SAC held-out success: 69.0 %, 62.0 %, 58.5 %. Validation success at 200k steps was still changing (56 %, 50 %, 20 %), so SAC had not converged within this budget.

![PPO vs SAC validation learning curves](figures/ppo_vs_sac_reach_learning.png)

### Why SAC lagged here

- **The entropy temperature collapsed early.** In all three seeds, SAC's automatically tuned `ent_coef` fell from 0.89 to about 0.0015 within the first ~36k steps and stayed there. The policy became nearly deterministic and stopped exploring.
- **The hold reward was discovered late.** The hold term only pays inside the 2 cm tolerance ball. In the training rollouts it first appears at 114k–171k steps. Until then SAC's critics had no evidence that the ball was worth entering, and the policy settled 3–5 cm away, collecting only the telescoping progress reward. PPO kept substantial Gaussian exploration noise (std ≈ 0.7 at 200k steps), entered the ball early, and learned to hold by 80k–100k steps.
- **Gradient updates were the bottleneck.** An SAC update (actor, twin critics, entropy temperature) took about 9 ms on this CPU, nearly independent of thread count (1 thread was fastest). With one update per transition, SAC trains at about 94 steps/s, against PPO's ~700 steps/s with batched updates.
- **SAC's actions were smoother** (jerk 0.032 vs 0.044). Its lower final-step entropy shows up as steadier commands, but that does not compensate for the lower success rate.

**Scope of the claim:** with default hyperparameters and an equal 200k-step budget, PPO clearly beat SAC on this task in sample efficiency, final quality and wall clock. This is a statement about this reward structure and configuration, not about SAC in general. A follow-up that keeps exploration alive, such as a higher target entropy or a lower bound on `ent_coef`, would test the explanation above. It would be reported as a separate, labelled experiment, not as a replacement for this comparison.
