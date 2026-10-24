"""Cube physics variation: mass, contact friction and size, always relative to the nominal model.

Used by the robustness suite (evaluation) and by domain randomization in grasp training. Contact
friction is scaled on the cube, the finger pads and the world geoms together: MuJoCo combines a
contact's friction with max() over its two geoms, so scaling the cube alone would have no effect
below nominal. Size keeps the density, so mass also scales with volume.
"""

import mujoco
import numpy as np

from polypanda.simulation.scenes import CubeIndices


class CubePhysics:
    def __init__(self, model: mujoco.MjModel, cube: CubeIndices) -> None:
        self.model, self.cube = model, cube
        fingers = {model.body(n).id for n in ("left_finger", "right_finger")}
        pads = [g for g in range(model.ngeom) if model.geom_bodyid[g] in fingers]
        self.friction_geoms = [cube.geom, *pads, *np.flatnonzero(model.geom_bodyid == 0)]
        g, b = cube.geom, cube.body
        self.nominal = {
            "friction": model.geom_friction[self.friction_geoms].copy(),
            "size": model.geom_size[g].copy(),
            "rbound": float(model.geom_rbound[g]),
            "aabb": model.geom_aabb[g].copy(),
            "mass": float(model.body_mass[b]),
            "inertia": model.body_inertia[b].copy(),
        }

    def apply(self, mass_scale: float = 1.0, friction_scale: float = 1.0,
              size_scale: float = 1.0) -> None:
        m, n, g, b, s = self.model, self.nominal, self.cube.geom, self.cube.body, size_scale
        m.geom_friction[self.friction_geoms] = n["friction"] * friction_scale
        m.geom_size[g] = n["size"] * s
        m.geom_rbound[g] = n["rbound"] * s
        m.geom_aabb[g] = n["aabb"] * s
        self.cube.half_size[:] = m.geom_size[g]  # shared by the env, its checks and drivers
        mass = n["mass"] * s**3 * mass_scale
        m.body_mass[b] = mass
        m.body_inertia[b] = n["inertia"] * (mass / n["mass"]) * s**2
        # recompute derived constants (subtree masses, inverse weights used by the contact
        # solver) on scratch data, as the compiler would have
        mujoco.mj_setConst(m, mujoco.MjData(m))
