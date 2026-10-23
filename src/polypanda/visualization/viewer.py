"""Viewer helpers. Rendering settings only; they never affect physics."""

import mujoco


def fast_rendering(model: mujoco.MjModel) -> None:
    """Turn off shadows and floor reflections, the costliest effects on integrated graphics."""
    model.light_castshadow[:] = 0
    model.mat_reflectance[:] = 0.0
