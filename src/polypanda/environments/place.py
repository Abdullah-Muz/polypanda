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
import numpy as np
from gymnasium import spaces

from polypanda.config import load_yaml
from polypanda.control.action_adapter import make_action_adapter
from polypanda.environments.grasp import GRIPPER_CLOSED, GRIPPER_OPEN, gripper_tilt
from polypanda.environments.handoff import HandoffPool, restore_state
from polypanda.environments.reach import ActionConfig
from polypanda.environments.transport import REPO_ROOT
from polypanda.simulation.panda import PandaIndices, tcp_position
from polypanda.simulation.scenes import CubeIndices, cube_position, cube_tilt, load_pick_scene

OBS_SCHEMA = "place-v1-45d"
OBS_LAYOUT = (
    ("q", 7), ("dq", 7), ("q_cmd", 7),
    ("tcp", 3), ("tcp_rot6", 6), ("finger_gap", 1), ("gripper_open", 1),
    ("cube_rel", 3), ("cube_height", 1), ("cube_tilt", 1), ("contacts", 2),
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


@dataclass(frozen=True)
class PlaceConfig:
    action: ActionConfig = field(default_factory=ActionConfig)
    gate: PlaceGateConfig = field(default_factory=PlaceGateConfig)
    check: PlacedCheckConfig = field(default_factory=PlacedCheckConfig)
    reward: PlaceRewardConfig = field(default_factory=PlaceRewardConfig)
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

    def update(self, data, target_xy: np.ndarray) -> bool:
        cfg, c = self.config, cube_position(data, self.cube)
        ok = (c[2] - self.cube.half_size[2] < cfg.max_rest_height
              and cube_tilt(data, self.cube) < cfg.max_tilt
              and np.linalg.norm(c[:2] - target_xy) < cfg.tolerance
              and np.abs(data.qvel[self.cube.qvel:self.cube.qvel + 6]).max() < cfg.max_speed
              and not self.gripper_contact(data))
        self.streak = self.streak + 1 if ok else 0
        return self.streak >= cfg.window


class PlaceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: PlaceConfig | None = None) -> None:
        self.config = config or PlaceConfig()
        self.model, self.data = load_pick_scene()
        self.arm = PandaIndices.from_model(self.model)
        self.cube = CubeIndices.from_model(self.model)
        self.adapter = make_action_adapter(self.model, self.arm, self.config.action)
        self.check = PlacedCheck(self.model, self.arm, self.cube, self.config.check)
        path = Path(self.config.start_pool)
        self.pool = HandoffPool.load(path if path.exists() else REPO_ROOT / path, self.model)
        self._finger_qpos = [self.model.joint(f"finger_joint{i}").qposadr[0] for i in (1, 2)]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.adapter.action_dim,),
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
        if "state" in options:
            state, self.start_index = np.asarray(options["state"], dtype=float), -1
            target_xy = np.asarray(options["target"], dtype=float)[:2]
        else:
            self.start_index = int(options.get("state_index",
                                               self.np_random.integers(len(self.pool))))
            state = self.pool.states[self.start_index]
            target_xy = self.pool.goals[self.start_index][:2]
        restore_state(self.model, self.data, state)
        self.data.ctrl[self.arm.gripper_actuator] = GRIPPER_CLOSED
        self.adapter.adopt_ctrl(self.data)
        self.target = np.array([*target_xy, self.cube.half_size[2]])
        self.released = False
        self.check.reset()
        self._steps = self._no_contact = 0
        self._distance = float(np.linalg.norm(self.target - cube_position(self.data, self.cube)))
        self._rise = 0.0
        self._prev_action = np.zeros(self.adapter.action_dim)
        self._contacts = self._finger_contacts()
        source = self.pool.sources[self.start_index] if self.start_index >= 0 else "given"
        return self._observation(), {"target": self.target.copy(), "start_source": source,
                                     "distance": self._distance}

    def step(self, action):
        cfg, r = self.config, self.config.reward
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.adapter.step(self.data, a)
        self._steps += 1
        cube = cube_position(self.data, self.cube)
        lift = cube[2] - self.cube.half_size[2]
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
        placed = self.check.update(self.data, self.target[:2])
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
            "dropped": -r.dropped if dropped else 0.0,
            "toppled": -r.toppled if toppled else 0.0,
        }
        self._distance, self._rise, self._prev_action = distance, rise, a
        terminated = bool(dropped or toppled)
        truncated = not terminated and self._steps >= cfg.horizon
        info = {
            "is_success": placed,
            "failure": "dropped" if dropped else "toppled" if toppled else None,
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
            [float(self._contacts[0]), float(self._contacts[1])],
            self.target,
            self.target - cube,
        ]).astype(np.float32)
