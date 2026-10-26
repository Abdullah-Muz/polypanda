"""PlaceEnv: set a carried cube down upright on a target spot, release it and retreat.

Episodes start from transport-to-place handoff states: the cube held above its target spot
(pools built from the transport handoffs, so all grasp sources are represented). The policy moves
the arm; a gate opens the fingers once the cube is low (bottom within `release_height` of the
table) and over the target, mirroring GraspEnv's closing gate. Success is a verified placement
(PlacedCheck): resting on the table, upright, on target, still, and clear of the gripper, at the
end of the episode.
"""

from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from polypanda.config import load_yaml
from polypanda.control.action_adapter import (
    action_limit,
    bound_action,
    make_action_adapter,
)
from polypanda.control.ik import PoseIK
from polypanda.environments.grasp import GRIPPER_CLOSED, GRIPPER_OPEN, gripper_tilt
from polypanda.environments.handoff import HandoffPool, restore_state
from polypanda.environments.reach import ActionConfig
from polypanda.environments.transport import REPO_ROOT
from polypanda.simulation.panda import PandaIndices, tcp_position
from polypanda.simulation.scenes import CubeIndices, cube_position, cube_tilt, load_pick_scene

# v2 (Milestone 8, before any place policy was trained): adds the cube's up axis, so a learned
# policy can tell which way the cube leans in the grip (the scripted place uses its orientation).
OBS_SCHEMA = "place-v2-48d"
OBS_LAYOUT = (
    ("q", 7), ("dq", 7), ("q_cmd", 7),
    ("tcp", 3), ("tcp_rot6", 6), ("finger_gap", 1), ("gripper_open", 1),
    ("cube_rel", 3), ("cube_height", 1), ("cube_tilt", 1), ("cube_up", 3), ("contacts", 2),
    ("target", 3), ("target_error", 3),
)
OBS_DIM = sum(n for _, n in OBS_LAYOUT)


@dataclass(frozen=True)
class PlacedCheckConfig:
    max_rest_height: float = 0.003   # m: cube bottom within 3 mm of the table
    max_tilt: float = 0.1745         # rad (10 deg) from upright
    tolerance: float = 0.02          # m: cube centre to target, horizontally
    max_speed: float = 0.01          # max |qvel| of the cube's free joint (m/s or rad/s)
    window: int = 10                 # consecutive steps (0.5 s) all conditions must hold


@dataclass(frozen=True)
class PlaceGateConfig:
    release_height: float = 0.01     # m: open once the cube bottom is this close to the table
    xy_tol: float = 0.02             # m: and the cube is over the target


@dataclass(frozen=True)
class PlaceRewardConfig:
    progress: float = 20.0           # per metre of cube-to-target distance reduced (telescoping)
    retreat: float = 5.0             # per metre the hand rises above the released cube
    retreat_cap: float = 0.10
    placed: float = 0.5              # per step while the placed check holds
    action: float = 0.005
    action_rate: float = 0.05
    gripper_tilt: float = 0.1
    floor: float = 0.1
    dropped: float = 5.0             # lost before the gate opened; terminates
    toppled: float = 5.0             # cube fell over after release; terminates
    cube_tilt: float = 0.0           # per rad of cube tilt per step while held (learned place)
    centring: float = 0.0            # per metre of cube-to-target xy error, per step after release
    precise: float = 0.0             # extra per step while placed AND within precise_tol of target
    precise_tol: float = 0.01        # m


@dataclass(frozen=True)
class StackConfig:
    # Milestone 9: set the cube down on a tower of `level` cubes (0: the table, as in M8),
    # drawn uniformly from 0..max_level per pool episode. The tower's cubes stand at the target
    # with small random offsets; the pool start is raised by level x cube height (hand and cube
    # together, by IK, so the grip is unchanged). Moving a tower cube more than `disturb_tol`
    # ends the episode (failure `tower_knocked`, penalty reward.toppled); success also needs the
    # tower intact. `disturb_penalty` (v2) charges every metre a tower cube is moved, however
    # little: v1 policies, free to nudge a tower by up to 1 cm, sometimes left a 4-high stack
    # offset by a few millimetres, and in simulation such a stack creeps until it falls
    # (Milestone 9, Layer 2 evaluation). `aim_at_top` (v3): the target is the top tower cube's
    # actual position, not the fixture's centre (v2 aimed every cube at the fixture, so placing
    # errors added up into overhangs); a learned place trained so aims at the stack in the cell
    # too (supervisor.cell).
    max_level: int = 0
    jitter: float = 0.004        # m, each tower cube's offset from the target, per axis
    yaw_jitter: float = 0.26     # rad (15 deg), each tower cube's yaw, either way
    disturb_tol: float = 0.01    # m
    disturb_penalty: float = 0.0  # per metre of the largest tower-cube displacement (telescoping)
    aim_at_top: bool = False


@dataclass(frozen=True)
class PlaceConfig:
    action: ActionConfig = field(default_factory=ActionConfig)
    gate: PlaceGateConfig = field(default_factory=PlaceGateConfig)
    check: PlacedCheckConfig = field(default_factory=PlacedCheckConfig)
    reward: PlaceRewardConfig = field(default_factory=PlaceRewardConfig)
    stack: StackConfig = field(default_factory=StackConfig)
    horizon: int = 120               # policy steps (6 s); no termination on success
    drop_steps: int = 5              # steps without both finger contacts before release = dropped
    topple_tilt: float = 0.785       # rad (45 deg) after release = toppled
    start_pool: str = "data/handoff/place_train_v1.npz"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PlaceConfig":
        return load_yaml(cls, path)


class PlacedCheck:
    """Verified placement: resting, upright, on target, still, clear of the gripper."""

    def __init__(self, model, arm: PandaIndices, cube: CubeIndices, config: PlacedCheckConfig):
        self.model, self.arm, self.cube, self.config = model, arm, cube, config
        self.gripper_bodies = {model.body(n).id for n in ("hand", "left_finger", "right_finger")}
        self.streak = 0

    def reset(self) -> None:
        self.streak = 0

    def gripper_contact(self, data) -> bool:
        for i in range(data.ncon):
            c = data.contact[i]
            pair = {self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]}
            if self.cube.body in pair and pair & self.gripper_bodies:
                return True
        return False

    def update(self, data, target_xy: np.ndarray, rest: float | None = None) -> bool:
        """`rest`: the cube-centre height when set down (default: on the table)."""
        cfg, c = self.config, cube_position(data, self.cube)
        rest = self.cube.half_size[2] if rest is None else rest
        ok = (c[2] - rest < cfg.max_rest_height
              and cube_tilt(data, self.cube) < cfg.max_tilt
              and np.linalg.norm(c[:2] - target_xy) < cfg.tolerance
              and np.abs(data.qvel[self.cube.qvel:self.cube.qvel + 6]).max() < cfg.max_speed
              and not self.gripper_contact(data))
        self.streak = self.streak + 1 if ok else 0
        return self.streak >= cfg.window


class PlaceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: PlaceConfig | None = None, scene=None, cube: str = "cube") -> None:
        """`scene` (model, data): run in an existing scene (e.g. the workcell) on the cube body
        named `cube`, starting from its current state (reset option "continue"); by default the
        env builds its own scene and starts from the handoff pool."""
        self.config = config or PlaceConfig()
        levels = self.config.stack.max_level if scene is None else 0
        self.model, self.data = scene or load_pick_scene(distractors=levels)
        self.arm = PandaIndices.from_model(self.model)
        self.cube = CubeIndices.from_model(self.model, cube)
        self.adapter = make_action_adapter(self.model, self.arm, self.config.action)
        self.check = PlacedCheck(self.model, self.arm, self.cube, self.config.check)
        # Stacking: the pool's states are restored into a plain scene and copied over (the
        # tower cubes come after the arm and the cube in the state vectors).
        self.tower = CubeIndices.distractors(self.model) if levels else []
        self._base = load_pick_scene() if levels else (self.model, self.data)
        self._ik = PoseIK() if levels else None
        self.level = 0
        self._tower_xy = np.zeros((0, 2))
        self._moved = 0.0
        path = Path(self.config.start_pool)
        self.pool = (HandoffPool.load(path if path.exists() else REPO_ROOT / path, self._base[0])
                     if scene is None else None)
        self._finger_qpos = [self.model.joint(f"finger_joint{i}").qposadr[0] for i in (1, 2)]
        limit = action_limit(self.config.action)
        self.action_space = spaces.Box(-limit, limit, shape=(self.adapter.action_dim,),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.target = np.zeros(3)   # cube-centre target: (x, y, resting height)
        self.released = False
        self._steps = self._no_contact = 0
        self._distance = self._rise = 0.0
        self._prev_action = np.zeros(self.adapter.action_dim)
        self._contacts = (False, False)

    @property
    def elapsed_steps(self) -> int:
        return self._steps

    def _finger_contacts(self) -> tuple[bool, bool]:
        left, right = self.model.body("left_finger").id, self.model.body("right_finger").id
        found = {left: False, right: False}
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b = (self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2])
            if self.cube.body in b:
                other = b[1] if b[0] == self.cube.body else b[0]
                if other in found:
                    found[other] = True
        return found[left], found[right]

    def _hand_rise(self) -> float:
        """Height of the TCP above the cube centre, capped (retreat progress after release)."""
        rise = tcp_position(self.data, self.arm)[2] - cube_position(self.data, self.cube)[2]
        return float(min(max(rise, 0.0), self.config.reward.retreat_cap))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        # The target is the cube-centre resting point: (x, y) on the table, or (x, y, z) when
        # the cube is to be set down on something (e.g. a tower: z = level height + half cube).
        self.level = 0
        if options.get("continue"):  # from the scene as it is (a skill inside a larger task)
            self.start_index = -1
            target = np.asarray(options["target"], dtype=float)
        elif "state" in options:
            self.start_index = -1
            restore_state(self.model, self.data, np.asarray(options["state"], dtype=float))
            target = np.asarray(options["target"], dtype=float)
        else:
            self.start_index = int(options.get("state_index",
                                               self.np_random.integers(len(self.pool))))
            target = self.pool.goals[self.start_index][:2]
            if self.tower:  # (no extra draws without stacking: M8 episodes are unchanged)
                level = int(options.get("level", self.np_random.integers(len(self.tower) + 1)))
                self._start_on_tower(self.pool.states[self.start_index], target, level)
                if self.config.stack.aim_at_top and self.level:
                    target = self.data.xpos[self.tower[self.level - 1].body][:2].copy()
                target = np.array([*target, self.cube.half_size[2] * (2 * self.level + 1)])
            else:
                restore_state(self.model, self.data, self.pool.states[self.start_index])
        self.data.ctrl[self.arm.gripper_actuator] = GRIPPER_CLOSED
        self.adapter.adopt_ctrl(self.data)
        rest = float(target[2]) if len(target) > 2 else float(self.cube.half_size[2])
        self.target = np.array([target[0], target[1], rest])
        self.released = False
        self.check.reset()
        self._steps = self._no_contact = 0
        self._moved = 0.0
        self._distance = float(np.linalg.norm(self.target - cube_position(self.data, self.cube)))
        self._rise = 0.0
        self._prev_action = np.zeros(self.adapter.action_dim)
        self._contacts = self._finger_contacts()
        source = self.pool.sources[self.start_index] if self.start_index >= 0 else "given"
        return self._observation(), {"target": self.target.copy(), "start_source": source,
                                     "distance": self._distance, "level": self.level}

    def _start_on_tower(self, state: np.ndarray, xy: np.ndarray, level: int) -> None:
        """A pool start raised by `level` cubes, over a tower of `level` cubes at `xy`."""
        m0, d0 = self._base
        restore_state(m0, d0, state)
        m, d, arm, h = self.model, self.data, self.arm, 2 * self.cube.half_size[2]
        mujoco.mj_resetDataKeyframe(m, d, arm.home_key)  # tower cubes parked
        d.time = d0.time
        d.qpos[:m0.nq], d.qvel[:m0.nv], d.ctrl[:] = d0.qpos, d0.qvel, d0.ctrl
        mujoco.mj_forward(m, d)
        if level:
            q0 = d.qpos[arm.arm_qpos].copy()
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, d.site_xmat[arm.tcp_site])
            q, ok = self._ik.solve(q0, tcp_position(d, arm) + [0.0, 0.0, level * h], quat)
            if not ok:
                level = 0
            else:
                d.qpos[arm.arm_qpos] = q
                d.ctrl[arm.arm_actuators] += q - q0  # same tracking offset (sag) as before
                d.qpos[self.cube.qpos + 2] += level * h
                d.qvel[arm.arm_qvel] = 0.0
                d.qvel[self.cube.qvel:self.cube.qvel + 6] = 0.0
        s = self.config.stack
        for k, t in enumerate(self.tower[:level]):
            x, y = np.asarray(xy) + self.np_random.uniform(-s.jitter, s.jitter, 2)
            yaw = self.np_random.uniform(-s.yaw_jitter, s.yaw_jitter)
            d.qpos[t.qpos:t.qpos + 7] = [x, y, h * (k + 0.5), np.cos(yaw / 2), 0, 0,
                                         np.sin(yaw / 2)]
            d.qvel[t.qvel:t.qvel + 6] = 0.0
        mujoco.mj_forward(m, d)
        self.level = level
        self._tower_pos = np.array([d.xpos[t.body] for t in self.tower[:level]]).reshape(-1, 3)

    def _tower_moved(self) -> float:
        """Largest displacement (m) of a tower cube since the start."""
        if not self.level:
            return 0.0
        now = np.array([self.data.xpos[t.body] for t in self.tower[:self.level]])
        return float(np.linalg.norm(now - self._tower_pos, axis=1).max())

    def step(self, action):
        cfg, r = self.config, self.config.reward
        a, bound_terms = bound_action(action, self.config.action)
        self.adapter.step(self.data, a)
        self._steps += 1
        cube = cube_position(self.data, self.cube)
        lift = cube[2] - self.target[2]  # above its resting height (the table, or a tower top)
        self._contacts = self._finger_contacts()
        both = all(self._contacts)
        xy_err = float(np.linalg.norm(cube[:2] - self.target[:2]))
        if not self.released:
            self._no_contact = 0 if both else self._no_contact + 1
            if lift < cfg.gate.release_height and xy_err < cfg.gate.xy_tol:
                self.released = True
                self.data.ctrl[self.arm.gripper_actuator] = GRIPPER_OPEN
        dropped = (not self.released) and self._no_contact >= cfg.drop_steps
        toppled = self.released and cube_tilt(self.data, self.cube) > cfg.topple_tilt
        moved = self._tower_moved()
        knocked = moved > cfg.stack.disturb_tol
        placed = self.check.update(self.data, self.target[:2], self.target[2]) and not knocked
        distance = float(np.linalg.norm(self.target - cube))
        rise = self._hand_rise() if self.released else 0.0
        R = self.data.site_xmat[self.arm.tcp_site].reshape(3, 3)
        floor = any(
            0 in (self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2])
            and self.cube.body not in (self.model.geom_bodyid[c.geom1],
                                       self.model.geom_bodyid[c.geom2])
            for c in (self.data.contact[i] for i in range(self.data.ncon)))
        terms = {
            "progress": r.progress * (self._distance - distance),
            "retreat": r.retreat * (rise - self._rise),
            "placed": r.placed if placed else 0.0,
            "action": -r.action * float(a @ a),
            "action_rate": -r.action_rate * float(np.sum((a - self._prev_action) ** 2)),
            "gripper_tilt": -r.gripper_tilt * gripper_tilt(R),
            "floor": -r.floor if floor else 0.0,
            "cube_tilt": 0.0 if self.released else -r.cube_tilt * cube_tilt(self.data, self.cube),
            "dropped": -r.dropped if dropped else 0.0,
            "toppled": -r.toppled if toppled else 0.0,
            **({"centring": -r.centring * xy_err if self.released else 0.0}
               if r.centring else {}),
            **({"precise": r.precise if placed and xy_err < r.precise_tol else 0.0}
               if r.precise else {}),
            **({"tower": -r.toppled if knocked else 0.0,
                "tower_disturb": -cfg.stack.disturb_penalty * (moved - self._moved)}
               if self.tower else {}),
            **bound_terms,
        }
        self._distance, self._rise, self._prev_action = distance, rise, a
        self._moved = moved
        terminated = bool(dropped or toppled or knocked)
        truncated = not terminated and self._steps >= cfg.horizon
        failure = ("dropped" if dropped else "toppled" if toppled
                   else "tower_knocked" if knocked else None)
        info = {
            "is_success": placed,
            "failure": failure,
            "level": self.level,
            "tower_moved": moved,
            "released": self.released,
            "distance": distance,
            "xy_error": xy_err,
            "lift": float(lift),
            "cube_tilt": cube_tilt(self.data, self.cube),
            "gripper_contact": self.check.gripper_contact(self.data),
            "floor_contact": floor,
            "reward_terms": terms,
        }
        return self._observation(), float(sum(terms.values())), terminated, truncated, info

    def _observation(self) -> np.ndarray:
        R = self.data.site_xmat[self.arm.tcp_site].reshape(3, 3)
        cube = cube_position(self.data, self.cube)
        tcp = tcp_position(self.data, self.arm)
        return np.concatenate([
            self.data.qpos[self.arm.arm_qpos],
            self.data.qvel[self.arm.arm_qvel],
            self.adapter.q_cmd,
            tcp,
            R[:, :2].T.flatten(),
            [sum(self.data.qpos[adr] for adr in self._finger_qpos), float(self.released)],
            cube - tcp,
            [cube[2] - self.cube.half_size[2], cube_tilt(self.data, self.cube)],
            self.data.xmat[self.cube.body].reshape(3, 3)[:, 2],
            [float(self._contacts[0]), float(self._contacts[1])],
            self.target,
            self.target - cube,
        ]).astype(np.float32)
