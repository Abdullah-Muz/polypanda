"""v4 imperfect sensing: a simulated depth camera and a classical cube-pose estimator.

`camera`: pinhole camera setups added to the cell's MjSpec, their intrinsics, and a depth-noise
model. `estimator`: back-projection, known-geometry subtraction, clustering and a per-cluster
top-face pose estimate (no learning). See docs/pose_error.md.
"""
