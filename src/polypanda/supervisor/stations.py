"""Station frames: use a skill trained in front of the robot at any station around it.

The scene is symmetric about the robot's base axis: rotating everything by an angle about that
axis is the same as adding the angle to joint 1. So a policy trained with its work in front of
the robot (base angle 0) can work at a station at angle theta if it sees the scene rotated by
-theta and its hand motions are rotated by +theta:

- observations: world 3-vectors (positions, offsets, the TCP's rotation columns, the cube's up
  axis) and world directions in the table plane (a stand-up's commanded turn) are rotated by
  -theta, and joint 1 (measured and commanded) is reduced by theta; joint velocities,
  hand-frame quantities and scalars are unchanged;
- actions: the env's hand-motion adapter reads them in the station frame
  (CartesianActionAdapter.set_frame); joint-increment actions need no change at all.

StationView wraps a skill env built on the shared scene (e.g. the workcell) and started from the
current state (reset option "continue").
"""

import numpy as np

from polypanda.environments import grasp, place, standup, transport

LAYOUTS = {grasp.GraspEnv: grasp.OBS_LAYOUT, transport.TransportEnv: transport.OBS_LAYOUT,
           place.PlaceEnv: place.OBS_LAYOUT, standup.StandUpV3Env: standup.OBS_LAYOUT_V3}
JOINT_FIELDS = ("q", "q_cmd")
VECTOR_FIELDS = ("tcp", "cube_rel", "goal", "goal_error", "target", "target_error", "cube_up",
                 "tower_rel", "tower_away")
ROT6_FIELDS = ("tcp_rot6",)   # two world 3-vectors (the TCP rotation's first two columns)
PLANAR_FIELDS = ("command",)  # a world direction in the table plane (cos, sin)


def _layout(env) -> tuple:
    for cls, layout in LAYOUTS.items():
        if isinstance(env, cls):
            return layout
    raise TypeError(f"no observation layout known for {type(env).__name__}")


def _slices(layout) -> dict[str, slice]:
    out, start = {}, 0
    for name, n in layout:
        out[name] = slice(start, start + n)
        start += n
    return out


class StationView:
    """A skill env seen from a station at `angle` (rad) about the base axis."""

    def __init__(self, env, angle: float) -> None:
        self.env, self.angle = env, float(angle)
        self.slices = _slices(_layout(env))
        c, s = np.cos(self.angle), np.sin(self.angle)
        self._to_frame = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])  # Rz(-angle)
        if hasattr(env.adapter, "set_frame"):
            env.adapter.set_frame(self.angle)
        if hasattr(env, "obs_transform"):  # env-side scoring (a turn chooser) sees this view too
            env.obs_transform = self.observe

    def __getattr__(self, name):  # everything else (data, model, config, ...) from the env
        return getattr(self.env, name)

    def observe(self, obs: np.ndarray) -> np.ndarray:
        """World-frame observation -> station-frame observation."""
        o = np.array(obs, dtype=np.float64)
        for name, sl in self.slices.items():
            if name in JOINT_FIELDS:
                o[sl.start] -= self.angle
            elif name in VECTOR_FIELDS:
                o[sl] = self._to_frame @ o[sl]
            elif name in ROT6_FIELDS:
                o[sl.start:sl.start + 3] = self._to_frame @ o[sl.start:sl.start + 3]
                o[sl.start + 3:sl.stop] = self._to_frame @ o[sl.start + 3:sl.stop]
            elif name in PLANAR_FIELDS:
                o[sl] = self._to_frame[:2, :2] @ o[sl]
        return o.astype(np.float32)

    def reset(self, **options):
        obs, info = self.env.reset(options={"continue": True, **options})
        return self.observe(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observe(obs), reward, terminated, truncated, info
