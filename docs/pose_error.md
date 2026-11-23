# How well can a simple camera locate a cube? (v4 sensing)

A measurement for the v4 plan's make-or-break experiment (`docs/v4_plan.md`, 3.1): what pose error does a realistic, simple camera estimate have in this cell? It is not a skill evaluation and not a target. Code: `src/polypanda/perception/`, `scripts/measure_pose_error.py`. Results: `results/v4/pose_error.json` (per condition, with provenance) and `pose_error_rows.csv` (one row per cube, camera and noise level). The run is from commit e2f9f53 (clean tree), 400 test scenes, seed 20261123, about 3 minutes on 4 processes.

**Short answer.** When the estimator returns a pose, its error is about 1–3 mm. The Phase 0 probe found that the scripted place still seats the cube at every aiming error up to 4 mm. So sensing precision alone does not give a learned place a reason to exist. Every estimate more than 4 mm off was a gross failure, usually two neighbouring cubes merged into one cluster (10–35 mm off). Better segmentation would fix those, and a feedback-based place would not help with them.

## Setup

Three depth cameras, added to the cell's `MjSpec` (pinhole model: square pixels, principal point at the image centre, `fx = fy = (H/2) / tan(fovy/2)`):

| Camera | Where | Resolution, vertical FOV | fx (px) | mm per px |
|---|---|---|---|---|
| overhead | fixed, 1.5 m above the bench over the robot base, looking down; sees every station and all loose-cube area | 640 × 576, 65° (time-of-flight narrow mode) | 452 | 3.3 at the table |
| station_b | fixed near the machine at tower B (−90°), 0.6 m away, 21° from vertical | 640 × 480, 50° | 515 | 1.2 at the station |
| hover_b | 0.55 m straight above tower B: stands in for a wrist camera with the hand paused above the part (arm at home only) | 640 × 480, 58° | 433 | 0.6–1.2 at a cube top |

**Estimator** (classical, no learning; `estimator.py`):
1. Back-project the depth image to points.
2. Keep the pixels that are at least 1 cm (or 3σ) nearer than the known scene and higher than 1.5 cm. The known scene is the cell's static geometry plus the robot at its measured joints, rendered without cubes. This removes the table, the fixtures and the arm.
3. Group the foreground pixels into clusters: 8-connected neighbours whose depth step is below 1.5 cm + 3σ√2.
4. For each cluster, find the top-face height h (median of its highest points). The top face is the set of pixels at most 1 mm (or σ/2) below h. Their rays are intersected with the plane z = h, so depth noise moves h but not the face's xy. Yaw (mod 90°) comes from the face's fourth-order moment. A minimum-area rectangle snaps to the pixel grid for cubes turned a few degrees, so it is reported only for comparison. The xy is the centre of the rectangle at that yaw, using the 3–97 % quantiles.
5. Occlusion check: if anything nearer than the top face lies within 2 px of it, the estimate is **withheld**. A controller would look again instead of using it.

**Identity only** comes from a segmentation render, which acts as a perfect labeller saying which cluster is which cube. It never contributes to a pose. Estimator settings were chosen on a separate development seed (seed 1, 40 scenes).

**Test scenes** (400, their own seed): upright cubes on towers A/B/C (0–4 levels each), the buffer slots, the belt's pick point and the open bench (radius 0.30–0.75 m, ±162°). Half the scenes add a second cube 5–8 cm from another cube. Every cube is scaled 0.85–1.15. In 151 scenes the hand is held over a stack (0–8 cm off, 4–20 cm above it); in the rest the arm is at home. In total: 956 tower stacks, 569 buffer, 191 belt, 814 loose and 202 pair cubes.

**Depth noise** (`camera.py: NOISE`): per-pixel Gaussian noise with σ growing with distance squared, plus random dropout and dropout at depth edges. The four levels:
- **none:** the renderer's exact depth.
- **low:** σ = 1 mm·z², 0.5 % dropout, 10 % of edge pixels dropped. This is an assumed good sensor, not a published model.
- **medium:** σ = 1.2 mm + 1.9 mm·(z − 0.4 m)², the Kinect v1 axial model of Nguyen, Izadi & Lovell (2012). That is 3.5 mm at 1.5 m. Plus 2 % dropout and 30 % of edge pixels dropped.
- **high:** σ = 3.6 mm·z², Intel's RealSense D435 RMS formula (z² · subpixel 0.08 / (447 px · 50 mm baseline)). That is 8 mm at 1.5 m. Plus 5 % dropout and 50 % of edge pixels dropped.

**medium+calib** adds an extrinsic calibration error to the medium level: the estimator assumes a camera pose that is off by 1 mm and 0.05°, drawn again for each scene. At 1.5 m, 0.05° is 1.3 mm, and the error scales linearly: 0.1° would be 2.6 mm.

## Results

The xy error is the distance between the estimated and true top-face centre, over accepted estimates (detected and not withheld). Visible means the cube shows at least 25 px. Missed means visible but with no cluster identified as it. Merged means an accepted estimate whose cluster also holds another stack.

| Camera | Noise | Visible | Missed | Withheld | Merged | xy median / p95 / max (mm) | Share > 1 / 2 / 3 / 4 mm | Yaw median / p95 (°) | abs z p95 (mm) |
|---|---|---|---|---|---|---|---|---|---|
| overhead | none | 2402 | 15 | 229 | 11 | 0.25 / 0.81 / 29.4 | 3 / 0.1 / 0.1 / 0.1 % | 0.35 / 1.99 | 0.00 |
| overhead | low | 2402 | 21 | 226 | 17 | 0.56 / 1.21 / 31.8 | 11 / 0.4 / 0.3 / 0.2 % | 1.12 / 3.53 | 0.53 |
| overhead | medium | 2402 | 29 | 223 | 20 | 0.92 / 1.77 / 32.8 | 43 / 2.4 / 0.7 / 0.7 % | 1.28 / 4.26 | 0.82 |
| overhead | high | 2402 | 46 | 209 | 26 | 1.76 / 2.90 / 34.3 | 89 / 36 / 4.1 / 0.8 % | 1.56 / 5.17 | 2.68 |
| overhead | medium+calib | 2402 | 29 | 223 | 20 | 1.50 / 2.89 / 32.3 | 75 / 27 / 3.8 / 0.8 % | 1.29 / 4.22 | 1.34 |
| station_b | none | 808 | 9 | 22 | 3 | 0.31 / 0.56 / 26.2 | 0 / 0.1 / 0.1 / 0.1 % | 0.11 / 0.55 | 0.00 |
| station_b | medium | 808 | 17 | 22 | 10 | 0.55 / 0.92 / 34.1 | 2 / 0.7 / 0.7 / 0.7 % | 0.35 / 1.12 | 0.35 |
| station_b | high | 808 | 17 | 22 | 10 | 0.67 / 1.12 / 33.3 | 12 / 0.5 / 0.5 / 0.5 % | 0.36 / 1.24 | 0.43 |
| station_b | medium+calib | 808 | 17 | 22 | 10 | 1.02 / 1.83 / 35.0 | 52 / 2.0 / 0.7 / 0.7 % | 0.34 / 1.13 | 1.16 |
| hover_b | none | 407 | 0 | 4 | 1 | 0.12 / 0.32 / 2.9 | 0 / 0.2 / 0 / 0 % | 0.05 / 0.58 | 0.00 |
| hover_b | high | 407 | 1 | 4 | 2 | 0.30 / 0.69 / 34.3 | 0 / 0.5 / 0.5 / 0.2 % | 0.28 / 0.90 | 0.17 |
| hover_b | medium+calib | 407 | 1 | 4 | 3 | 0.85 / 1.46 / 34.5 | 38 / 0.7 / 0.7 / 0.5 % | 0.31 / 0.97 | 0.99 |

**Two kinds of error.**
- **Precision.** Leaving out merged clusters, no overhead estimate was ever more than 4 mm off at any noise level. The largest was 3.9 mm at high noise, and 4.2 mm with medium noise plus calibration error. The 99th percentile was 1.2 / 1.6 / 2.1 / 3.3 mm for none / low / medium / high. For the station and hover cameras it stays under 2.4 mm in every condition.
- **Gross errors.** All the remaining errors above 4 mm are merged clusters: 0.5–1.2 % of overhead estimates and up to 1.3 % at the station. They come from two cubes 5–8 cm apart, especially large ones (a 4.6 cm cube 5 cm away leaves a 4 mm gap).

**What makes it worse** (overhead, medium noise unless stated):
- **Distance from the camera.** The share above 2 mm is 2 % near the camera (< 0.45 m from the point below it), 2 % in the middle and 5 % far out (> 0.6 m). At high noise it is 21 %, 34 % and 61 %.
- **Neighbours.** With another cube within 9 cm, 7 % of estimates are above 2 mm and 5 % above 4 mm (merged), against 2 % and 0 % without one. Most missed cubes are these neighbours: 22 of the 29 at medium noise.
- **The arm.** The overhead camera is straight above the base. Even at home, the arm hides the two front buffer slots: 330 of the 2,732 cubes in view were hidden (166 of them with the arm at home), and about 9 % of the visible ones were withheld as partly occluded. In noise-free images the occlusion check caught 193 of 194 cubes the arm partly hid. The one it accepted was 0.2 mm off. A real cell would take its picture with the arm parked clear, or mount the camera off-axis.
- **Tower level** helps slightly: higher tops are nearer the camera (high noise, share above 2 mm: level 1: 37 %, level 4: 21 %).
- **Cube size** barely matters, except that large cubes merge more often.
- **Depth bias.** Noise biases the top height low: −0.24 mm at medium and −1.2 mm at high noise.
- **Duplicates.** In noise-free images the clustering sometimes also splits a side face into a second small cluster, which would be a duplicate detection (85 of 2,387 overhead cubes). This disappears with noise, because the link tolerance grows with σ.

## What it means for the fixture place

Pocket: clearance 1–2 mm per side, 45° lead-in 5 mm wide. The Phase 0 probe found:
- at aiming errors up to 4 mm, the scripted place seats the cube 100 % of the time in both pockets;
- at 6 mm, the 2 mm pocket still seats it 100 %, and the 1 mm pocket 84 % / 70 % (the two wall frictions).

- **Inside the clearance (≤ 1–2 mm):** most estimates. With the overhead camera this holds up to medium noise (p95 1.8 mm, 2.4 % above 2 mm), and with a camera near the station at every noise level (p95 ≤ 1.2 mm, ≤ 1.8 mm with calibration error).
- **Inside the lead-in and the script's 4 mm tolerance:** the rest. This is the overhead camera with a D435-class sensor at 1.5 m (36 % above 2 mm, 4 % above 3 mm) or with a calibration error (27 % above 2 mm). The probe says the script still seats these. Wall forces and rim strikes grow, though, and 43–89 % of overhead estimates are more than 1 mm off.
- **Beyond 4 mm:** only gross failures, about 1 % (merged clusters, 10–35 mm off). That is outside any lead-in. A learned feedback skill would not rescue a pose that points at the wrong place. The fix is the perception: separating merged clusters (for example by size, since one cube's face is 3.4–4.6 cm), or rejecting an estimate whose face is larger than any cube.

So, from these numbers, realistic sensing error does not open a window where the scripted place fails and a learned place succeeds. The window would need errors of about 4–6 mm into a 1 mm pocket, and only the gross failures reach that. Three things could still change this and are not measured here:
- **Yaw.** A cube turned by θ in the pocket needs 20 mm · (cos θ + sin θ − 1) of clearance per side: 1.4 mm at the overhead camera's p95 of 4.3°. A parallel-jaw grasp on flat faces tends to square the cube to the fingers, so the place probably sees the grasp's yaw error rather than the camera's. The probe did not test yaw error.
- **The error that reaches the place.** It is the pocket's position (static: it can be taught once) plus the cube's offset in the hand (the closing axis centres it). Both may be smaller than the camera error.
- **Real sensor effects** (below).

**A wrist camera** close to the part (hover_b, 0.3–0.5 m from a cube top): p95 0.3–0.7 mm and yaw p95 under 1° at every noise level, because σ falls with distance squared. At that range the hand-eye calibration dominates (p95 1.5 mm with 1 mm / 0.05° error). A wrist camera also cannot be hidden by its own arm, but it costs a stop above every part.

## Limits

- **Simulated depth is cleaner than any real sensor:**
  - no multipath, no mixed pixels at edges, and no dropouts on dark or shiny surfaces;
  - noise is independent from pixel to pixel; real stereo noise is spatially correlated and averages down less, so the noise rows are optimistic;
  - no lens distortion, intrinsic error or motion blur (the belt is stopped);
  - the noise levels are assumptions, two of them taken from published models of other sensors.
- **The known-scene subtraction assumes exact knowledge of the cell and the arm.** Identity comes from a perfect segmenter; a real system needs colour, size or tracking for that.
- **Calibration error is one example.** It is the largest real-world term and scales linearly.
- **Only upright cubes were tested,** on a flat bench.
- **A first test run was discarded.** It was the same seed, from commit 7919c32, and put some loose cubes inside the belt rails and tunnel. The scene generator was fixed in e2f9f53. The estimator was not changed between the two runs.
