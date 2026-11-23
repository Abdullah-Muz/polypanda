"""v4.0: change a compiled machine into another variant at runtime, without recompiling.

simulation/machine.py builds each variant as its own compiled model. Training the door skill
draws a new machine every episode, and recompiling the scene each time would cost far more than
the episode itself. Everything a door variant changes is a model field, so one compiled machine is
edited in place instead:

- the lid's slide joint: its axis (`jnt_axis`: right, left or out), its range (`jnt_range`: the
  travel), its Coulomb friction (`dof_frictionloss`), damping (`dof_damping`) and closing spring
  (`jnt_stiffness`);
- the handle bar, its two posts and its site (`geom_pos`, `geom_quat`, `geom_size`, and the
  posts' bounding sphere and box `geom_rbound`, `geom_aabb`, which collision detection uses):
  where the handle sits along the bar and along the slide, how high, and which way the bar runs
  (across the slide); and the lid body's bounding-volume hierarchy (`bvh_aabb`), which the
  collision midphase prunes with (refresh_bvh);
- the machine's body (`body_pos`, `body_quat`): where it stands and how it is turned;
- the latch speed and the rest of the runtime's settings: `Machine.cfg`.

`MachineLayout.apply` reproduces exactly what `add_machine` compiles for the same config
(tests/test_door_env.py compares the two, field by field). Not edited: the lid body's inertia
(the handle's 60 g moving by a few centimetres shifts the lid's centre of mass by under 1 cm),
and the body's size, opening, lid size and fixture, which every door variant shares (they are
checked to be unchanged).
"""

from __future__ import annotations

from dataclasses import fields, replace

import mujoco
import numpy as np

from polypanda.simulation.machine import Machine, MachineConfig, _quat_z_to

# what a runtime variant may change (everything else must match the compiled machine)
VARIABLE = ("slide", "travel", "lid_damping", "lid_friction", "lid_spring", "handle_offset",
            "handle_shift", "handle_height", "latch_catch", "latch_speed", "fault_open",
            "cycle_s", "colour")
POST_HALF = 0.006          # m, the posts' square half section (machine._add_lid)
POST_INSET = 0.01          # m, the posts stand this far in from the bar's ends


def needed_opening(cfg: MachineConfig) -> float:
    """How far the lid must slide to uncover the whole opening: the opening's size along the
    slide plus the lid's overlap (as in evaluation/tending.py)."""
    along = cfg.opening[0] if cfg.slide == "out" else cfg.opening[1]
    return float(along + cfg.lid_overlap)


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class MachineLayout:
    """The variable parts of one compiled machine named `name`, and how to set them."""

    def __init__(self, model: mujoco.MjModel, name: str, compiled: MachineConfig) -> None:
        self.name, self.compiled = name, compiled
        j = model.joint(f"{name}_lid")
        self.joint, self.dof = j.id, int(j.dofadr[0])
        self.body = model.body(name).id
        self.lid_body = model.body(f"{name}_lid").id
        self.handle = model.geom(f"{name}_handle").id
        self.posts = [model.geom(f"{name}_handle_post{k}").id for k in (0, 1)]
        self.site = model.site(f"{name}_handle").id

    def check(self, cfg: MachineConfig) -> None:
        """Refuse a variant that changes what only a recompile can change."""
        for f in fields(MachineConfig):
            if f.name not in VARIABLE and getattr(cfg, f.name) != getattr(self.compiled, f.name):
                raise ValueError(f"{f.name} cannot change at runtime "
                                 f"({getattr(self.compiled, f.name)} compiled)")

    def apply(self, model: mujoco.MjModel, cfg: MachineConfig,
              machine: Machine | None = None) -> None:
        """Make the compiled machine the variant `cfg` (call mj_forward afterwards). With
        `machine` (its runtime), its latch and cycle settings follow too."""
        self.check(cfg)
        model.jnt_axis[self.joint] = cfg.slide_axis
        model.jnt_range[self.joint] = [0.0, cfg.travel]
        model.dof_frictionloss[self.dof] = cfg.lid_friction
        model.dof_damping[self.dof] = cfg.lid_damping
        model.jnt_stiffness[self.joint] = cfg.lid_spring
        half = cfg.lid_half
        centre = cfg.handle_centre() - np.array([0.0, 0.0, cfg.height + half[2]])  # lid frame
        bar = cfg.bar_axis
        model.geom_pos[self.handle] = centre
        model.geom_quat[self.handle] = _quat_z_to(bar)
        yaw = float(np.arctan2(bar[1], bar[0]))
        size = np.array([POST_HALF, POST_HALF, cfg.handle_height / 2])
        for s, g in zip((-1, 1), self.posts, strict=True):
            p = centre + bar * s * (cfg.handle_length / 2 - POST_INSET)
            p[2] = half[2] + cfg.handle_height / 2
            model.geom_pos[g] = p
            model.geom_quat[g] = _yaw_quat(yaw)
            model.geom_size[g] = size
            model.geom_rbound[g] = float(np.linalg.norm(size))
            model.geom_aabb[g] = [0.0, 0.0, 0.0, *size]
        model.site_pos[self.site] = centre
        self.refresh_bvh(model)
        if machine is not None:
            machine.cfg = cfg

    def refresh_bvh(self, model: mujoco.MjModel) -> None:
        """Recompute the lid body's bounding-volume hierarchy (`bvh_aabb`) from its geoms as they
        are now. The compiler builds it from the geoms' compiled placements, and collision's
        midphase prunes against it: a handle moved outside its compiled box is passed through
        by a finger (measured before this existed). The boxes are in the body's inertial frame;
        leaves bound one geom, inner nodes their children."""
        b = self.lid_body
        adr, num = int(model.body_bvhadr[b]), int(model.body_bvhnum[b])
        if adr < 0 or num == 0:
            return
        R_i, R_g = np.zeros(9), np.zeros(9)
        mujoco.mju_quat2Mat(R_i, model.body_iquat[b])
        R_i = R_i.reshape(3, 3)
        ipos = model.body_ipos[b]
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)

        def bounds(k: int) -> tuple[np.ndarray, np.ndarray]:
            g = int(model.bvh_nodeid[adr + k])
            if g >= 0:
                mujoco.mju_quat2Mat(R_g, model.geom_quat[g])
                R = R_g.reshape(3, 3)
                box = model.geom_aabb[g]
                corners = model.geom_pos[g] + (box[:3] + signs * box[3:]) @ R.T  # body frame
                local = (corners - ipos) @ R_i                                    # inertial
                lo, hi = local.min(axis=0), local.max(axis=0)
            else:
                (lo1, hi1), (lo2, hi2) = (bounds(int(c)) for c in model.bvh_child[adr + k])
                lo, hi = np.minimum(lo1, lo2), np.maximum(hi1, hi2)
            model.bvh_aabb[adr + k] = [*(lo + hi) / 2, *(hi - lo) / 2]
            return lo, hi

        bounds(0)

    def place(self, model: mujoco.MjModel, xy, yaw: float) -> None:
        """Stand the machine with its pocket's centre at `xy` on the bench, turned by `yaw`
        (rad; its +x points away from a robot at angle yaw + pi from it)."""
        model.body_pos[self.body] = [float(xy[0]), float(xy[1]), 0.0]
        model.body_quat[self.body] = _yaw_quat(yaw)


def variant(base: MachineConfig, **changes) -> MachineConfig:
    """`base` with door-variant fields changed (checked against VARIABLE)."""
    bad = set(changes) - set(VARIABLE)
    if bad:
        raise ValueError(f"not runtime-variable: {sorted(bad)}")
    return replace(base, **changes)
