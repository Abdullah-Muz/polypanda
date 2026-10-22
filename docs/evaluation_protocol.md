# Evaluation protocol

Every reach policy, whether a baseline or a trained checkpoint, is evaluated the same way:

```bash
uv run python scripts/evaluate.py --policy ik                       # or random
uv run python scripts/evaluate.py --policy ppo:path/to/model.zip --name reach_ppo_seed42
```

## Held-out set: `eval_sets/reach_nominal_v1.json`

- 200 targets from the same reachable-by-construction sampler and workspace box as training ([reach_task.md](reach_task.md)), each at least 0.10 m from the home TCP (distance 0.114–0.601 m, median 0.345 m).
- Drawn from `SeedSequence(20261021, spawn_key=(0xE7A1,))`. Because of the spawn key, no `env.reset(seed=n)` can reproduce this stream, so training and tuning seeds (for example the IK tuning seeds 1000–1049) cannot leak evaluation targets.
- Each target stores its witness configuration and a region label: near or far (horizontal radius < or ≥ 0.5 m) × low or high (z < or ≥ 0.3 m). Counts: far-high 71, near-high 68, near-low 35, far-low 26.
- A SHA-256 digest of the targets (`5b4bf7e75eec`) is checked on load. A test regenerates the set and requires it to be bit-identical. Anything that changes targets requires a new versioned name.
- Start state: `home` keyframe, no joint noise.

## Episode rules

- Same environment and action interface as training, but `success.terminate = false`. Every episode runs the full 200 steps (10 s), so holding after success is measured.
- Policies are queried with `deterministic=True`. No weights are updated during evaluation.
- *Success* means the training success predicate is met at least once: within 2 cm, held for 5 consecutive steps.

## Metrics

| Metric | Definition |
|---|---|
| success rate | fraction of episodes that succeed, with a 95 % Wilson interval |
| time to success | first step where the success predicate holds × 0.05 s (successful episodes only) |
| held | after first success, the TCP never leaves the 2 cm tolerance for the rest of the episode |
| final distance | TCP–target distance at the end of the 10 s horizon |
| path ratio | TCP path length up to success (or horizon) ÷ start distance; 1.0 is a straight line |
| action jerk | mean ‖aₜ − aₜ₋₁‖² up to success (or horizon) |
| contact / limit episodes | fraction of episodes with any contact, or any joint target at a bound |
| return to success | training-style return: rewards summed up to success or horizon |
| inference latency | wall time of `predict()` per step (median, p99), on the development laptop |
| by region | success rate and 95 % CI per region |

## Outputs

`results/reach/<name>.json` holds the summary plus provenance: policy description, set name and digest, git commit with a dirty flag, library versions, and UTC time. `results/reach/<name>_episodes.csv` holds one row per episode. Published numbers must come from a clean commit (`dirty: false`).
