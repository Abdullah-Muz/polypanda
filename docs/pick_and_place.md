# Pick-and-place: supervisor and end-to-end evaluation (Milestone 5)

## 5.5: the supervisor (`polypanda.supervisor.fsm.PickPlaceSupervisor`)

A finite-state machine chains the three skills and handles failures:

```
GRASP --grasped 0.5 s--> TRANSPORT --above target 0.5 s--> PLACE --placed--> SUCCESS
  |                         |                                 |
  +------- failure (knocked over, timeout, lost grip, toppled...)
                            v
        HOME -> cube upright, on the table, within reach?
                  yes: GRASP again (at most 2 retries)   no: FAILURE(stage, reason)
```

- **Each skill runs in its own environment** (`GraspEnv`, `TransportEnv`, `PlaceEnv`), so every skill sees exactly the observations, gates and success checks it was built and evaluated with. Nothing is re-implemented for the chain.
- **Handoffs move the exact physics state** (`mj_getState`/`mj_setState`, positions, velocities, controls and warm start), and the next skill continues from the previous skill's joint commands, so the arm does not jump at a switch.
- **Transitions use the verified predicates**, never elapsed time:
  - GRASP → TRANSPORT when GraspEnv success (verified grasp, cube ≥ 5 cm up) has held for 10 steps;
  - TRANSPORT → PLACE when the cube has been held within 2 cm of the pre-place point (12 cm above the target) for 10 steps;
  - PLACE → SUCCESS when `PlacedCheck` passes (which already requires 0.5 s).
- **Recovery.** On any failure the fingers open and the arm holds still for 0.5 s, the hand lifts 10 cm straight up, and the arm returns home. All of this happens in simulation, so recovery costs real time and can disturb the cube, as it would on hardware. The supervisor then measures the cube: if it is upright (under 10°), resting on the table, and within 3 cm of the placement region, GRASP restarts from home with the cube where it lies. Otherwise the episode fails with the stage and the reason (`cube tipped over`, `cube out of the workspace`, `out of attempts`, `time budget exhausted`).
- **Global time budget:** 90 s for the whole task.

Skill choices follow each skill's evaluation: **grasp** is learned PPO (with the scripted grasp as the comparison), **transport** is the scripted closed-loop carry ([why](transport_task.md)), and **place** is the scripted place with upright correction ([why](place_task.md)).

### A recovery bug found by the evaluation

The first version drove the arm home in joint space the moment it opened the fingers. On the first end-to-end run, many cubes that were upright when a grasp gave up ended on their side. The arm swept back while the fingers were still opening and knocked them over, and those episodes were reported as unrecoverable. Opening first, then lifting clear, then going home fixed it. Of the 39 PPO seed-0 failures, 11 now succeed. The cubes that still end on their side were knocked over by a skill: the grasp tipped them past 60°, or they fell out of the fingers. The same run showed that the retry region was too strict. Two scripted episodes failed with an upright cube resting at x = 0.348 m, 2 mm outside the region cubes start in, which the arm reaches easily. That is why the retry region extends 3 cm beyond the start region. Both fixes are in commit `a8fced5`; all reported results come from that version.

Tests (`tests/test_supervisor.py`) cover:
- a full scripted pick-and-place;
- the exact state carrying over at a handoff;
- re-grasping after a cube is deliberately dropped mid-transport;
- rejecting a cube on its side;
- the time budget.

## 5.6: end-to-end evaluation

Held-out set `pickplace_nominal_v1`: 200 scenes (digest `439a38044438`), each a cube start pose (uniform position in the placement region, any yaw) and a target spot at least 10 cm away (median 21 cm, max 49 cm). It has its own SeedSequence spawn key, separate from every other stream. Every episode starts from the home pose. **Success** means a verified placement within 90 s and at most two re-grasps. Evaluated at commit `a8fced5` (clean) with `scripts/evaluate_pickplace.py`. Raw outputs: [`results/pickplace/`](../results/pickplace/).

| Grasp skill (transport and place scripted) | Success (95 % CI) | First-attempt success | Episodes with a retry → recovered | Time to placed, median / p90 | Final xy error, median |
|---|---|---|---|---|---|
| **Scripted grasp** | **100 %** (98.1–100 %) | 97.5 % | 5 → 5 | 12.4 / 15.8 s | 1.7 mm |
| PPO grasp v1 seed 0 | 86.0 % (80.5–90.1 %) | 75.0 % | 33 → 22 | 10.3 / 28.9 s | 1.8 mm |
| PPO grasp v1 seed 1 | 89.0 % (83.9–92.6 %) | 84.0 % | 20 → 10 | 9.7 / 13.9 s | 1.5 mm |
| PPO grasp v1 seed 2 | 83.0 % (77.2–87.6 %) | 76.0 % | 32 → 14 | 9.3 / 13.2 s | 1.7 mm |
| PPO grasp v1, pooled 600 | **86.0 %** (83.0–88.5 %); 86.0 ± 3.0 per seed | 78.3 % | 85 → 46 | about 9.5 s | 1.7 mm |

(In Milestone 6 the learned grasp was replaced by v2 with a committed grasp yaw: **89.2 %** end to end. See [below](#milestone-6-the-v2-grasp-and-ablations).)

Where the attempts that end an episode fail:

| Grasp skill | Grasp | Transport | Place | Main reasons |
|---|---|---|---|---|
| Scripted | 0 | 0 | 0 | — (the 5 retries were all place timeouts, each recovered) |
| PPO seed 0 | 26 | 1 | 1 | cube pushed out of the workspace 15, knocked over 9 |
| PPO seed 1 | 21 | 0 | 1 | grasp timed out on all 3 attempts 10, pushed out of the workspace 9 |
| PPO seed 2 | 30 | 1 | 3 | knocked over 14, grasp timed out on all 3 attempts 10, pushed out 6 |

Paired on the same 200 scenes, every PPO configuration is significantly worse than all-scripted: exact McNemar p = 7.5 × 10⁻⁹, 4.8 × 10⁻⁷ and 1.2 × 10⁻¹⁰. No scene was solved by a PPO configuration and failed by the scripted one.

### Reading these numbers

- **The full chain works.** With scripted skills the robot picks and places the cube every time on 200 held-out scenes, 1.7 mm from the target (median), in about 12 s. Five place attempts timed out without a verified placement; the supervisor recovered all five by re-grasping and placing again.
- **Grasping is the bottleneck of the learned chain.** Almost every PPO failure is a grasp failure. The policy either pushes the cube out of reach or knocks it over (both unrecoverable), or misses three times in a row. Once the cube is grasped, transport and place almost never fail.
- **Retries are worth about 8 points.** First-attempt success with the PPO grasp is 78.3 %, close to the standalone grasp rate (80.5 %), and recovery lifts it to 86.0 %. Re-grasping helps less than one might expect: the policy is deterministic, so a retry only succeeds if the failed attempt moved the cube to a pose the policy handles better.
- **The learned chain is faster when it works.** The PPO grasp is about 3 s quicker than the deliberately cautious scripted grasp, so successful episodes finish in about 9.5 s instead of 12.4 s. Its p90 times are longer because of retries.
- **Information asymmetry:** the scripted skills use the privileged cube pose; the PPO grasp uses its observation vector. This is the same caveat as for every scripted baseline in the project.

## Milestone 6: the v2 grasp and ablations

### With the v2 grasp (committed yaw)

The grasp policy retrained with a committed grasp yaw ([grasp_task.md](grasp_task.md#task-v2-milestone-62)), same supervisor and scripted transport/place. Evaluated at `78f4c2e` (clean).

| Grasp skill | Success (95 % CI) | First attempt | Retried → recovered | Time to placed, median | Final xy error, median |
|---|---|---|---|---|---|
| PPO grasp v2 seed 0 / 1 / 2 | 88.0 / 90.0 / 89.5 % | 80.0 / 84.0 / 86.0 % | 26 → 16, 14 → 12, 17 → 7 | 10.1 / 9.5 / 9.8 s | 1.5 / 1.4 / 1.5 mm |
| **PPO grasp v2, pooled 600** | **89.2 %** (86.4–91.4 %) | 83.3 % | 57 → 35 | about 9.8 s | 1.5 mm |

Against v1 (86.0 %) the gain is consistent across seeds but not statistically significant: paired exact McNemar p = 0.09, with 66 against 47 discordant scenes. Against all-scripted (100 %) every v2 seed is still significantly worse (p ≤ 1.9 × 10⁻⁶). The failures are again almost all at the grasp stage: 59 of 65.

### Ablations (Milestone 6.3): what each supervisor design choice is worth

`scripts/evaluate_ablations.py` removes one design choice at a time and re-runs the 200 held-out scenes. Each result is paired with the full system on the same scenes. The PPO column pools the three v2 seeds. Evaluated at `d6bc214` (clean); raw outputs in [`results/ablations/`](../results/ablations/).

| System | Scripted grasp | PPO grasp v2 (pooled 600) |
|---|---|---|
| **Full system** | **100 %** | **89.2 %** |
| No retries: failure detection, but give up on the first failure | 97.5 % (p = 0.06) | 83.3 % (−5.9; every seed p ≤ 0.016) |
| Timed switching: skills switch after fixed times, without verification | 84.5 % (p = 9 × 10⁻¹⁰) | **71.2 %** (−18.0; every seed p ≤ 2 × 10⁻⁹) |
| No upright correction before release | 97.0 % (p = 0.03) | 81.5 % (−7.7; every seed p ≤ 0.006) |

- **Verified handoffs matter most.** The timed supervisor gives each skill the 90th percentile of its own successful durations in the full system, a fair, tuned schedule. It still fails every case that runs long. A grasp that is not finished is carried away empty: 100 PPO episodes fail at "transport". A cube still on its way reaches "place" too early: 63 fail there. Without verification there is also nothing to trigger a recovery.
- **Retries are worth about 6 points** with the learned grasp and 2.5 with the scripted one. They are cheap insurance, but they cannot fix a cube pushed out of reach or knocked over.
- **The upright correction matters more for the learned grasp** (−7.7 against −3.0). PPO grasps leave the cube tilted more in the fingers (median 12–15° against 11°), and a cube released past about 34° falls over.
- Across all 2,400 ablated episodes, the ablated system solved only 2 scenes the full system failed (both without the upright correction). Each removed piece only removes capability.

## Watch it

```bash
uv run python scripts/demo_pickplace.py                                  # scripted, 6 scenes
uv run python scripts/demo_pickplace.py --grasp run:runs/grasp_v2yaw_seed1
uv run python scripts/demo_pickplace.py --scenes 62 178                  # place fails, re-grasp
```

The green disc marks the target. The terminal prints each stage as the supervisor switches skills.
