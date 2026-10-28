"""StandUpEnv (v3 step d2): stand a lying cube, held in the hand, upright on the table.

A cube knocked off a tower mostly lies on a side; the any-pose grasp (d1) picks it up gripped
across its 4 cm width, and the cell carries it, hand pointing down, over the spot where it is to
stand. From there this skill turns the hand sideways so the cube's long axis turns vertical,
lowers it, lets go just above the table, and backs the hand away without knocking it over.

Physics as in the workcell, where the skill runs (`gripper`): the cell's grasp assist holds the
cube welded to the hand until the fingers open (the disclosed simplification of
simulation.workcell); the release and the cube settling are physical. A gate opens the fingers
once the cube is upright, low and over the spot, mirroring the grasp's closing gate. The cell's
staging spots lie about 23 cm from a tower, so most episodes put a tower of 1-4 cubes (a fixed
box, observed) at about that distance in a random direction; touching it ends the episode.

Success: the placed check (resting, upright within 10 deg, within check.tolerance of the spot,
still, clear of the gripper) holding at the end of the episode, as for the place skill. The
skill is trained in front of the robot and used at any base angle (supervisor.stations).
"""

from dataclasses import dataclass, field, replace
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from polypanda.config import load_yaml
from polypanda.control.action_adapter import action_limit, bound_action, make_action_adapter
from polypanda.control.ik import PoseIK, top_down_quat
from polypanda.environments.grasp import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    GripperConfig,
    RandomizationConfig,
)
from polypanda.environments.place import PlacedCheck, PlacedCheckConfig
from polypanda.environments.reach import ActionConfig
from polypanda.simulation.assets import panda_spec
from polypanda.simulation.panda import PandaIndices, reset_to_home, tcp_position
from polypanda.simulation.randomization import CubePhysics
from polypanda.simulation.scenes import (
    CubeIndices,
    add_cube,
    cube_position,
    cube_tilt,
    cube_up,
)
from polypanda.simulation.workcell import set_finger_stiffness

OBS_SCHEMA = "standup-v1-52d"
OBS_LAYOUT = (
    ("q", 7), ("dq", 7), ("q_cmd", 7),
    ("tcp", 3), ("tcp_rot6", 6), ("finger_gap", 1), ("released", 1),
    ("cube_rel", 3), ("cube_up", 3), ("cube_height", 1), ("contacts", 2),
    ("target", 3), ("target_error", 3), ("tower_rel", 3), ("tower", 2),
)
OBS_DIM = sum(n for _, n in OBS_LAYOUT)

CUBE_HEIGHT = 0.06       # m, one tower cube
TOWER_HALF = 0.12        # m, half height of the tower box (4 cubes); it sinks into the table
TOWER_PARK = (-0.8, 0.8)  # out of reach, behind the robot


@dataclass(frozen=True)
class StandUpStartConfig:
    radius: tuple[float, float] = (0.32, 0.62)   # m, the spot's distance from the base axis
    angle: tuple[float, float] = (-0.44, 0.44)   # rad (25 deg) about the base axis
    carry_height: float = 0.32                   # m, TCP height (the cell's carry posture)
    grip_offset: float = 0.01                    # m along the cube's long axis, either way
    grip_z: tuple[float, float] = (-0.005, 0.01)  # m, cube centre minus TCP (d1's gate window)
    # Training only: this fraction of episodes start with the hand already turned and low over
    # the spot (reset to helpful states, as the grasp's curriculum starts)
    helpful_prob: float = 0.0


@dataclass(frozen=True)
class TowerConfig:
    prob: float = 0.8                            # episodes with a tower
    distance: tuple[float, float] = (0.20, 0.26)  # m, spot to tower centre
    min_radius: float = 0.30                     # m, tower centre from the base axis
    max_cubes: int = 4                           # tower height: 1..max_cubes cubes


@dataclass(frozen=True)
class StandUpGateConfig:
    # rad (15 deg): the cube's long axis from vertical. The grasp assist's soft weld lets the
    # cube lag a turning hand (it creeps back over seconds); released 1-2 cm up within 15 deg,
    # it rocks back upright (success still needs it within check.max_tilt, 10 deg)
    max_tilt: float = 0.26
    max_drop: float = 0.025      # m: the cube's bottom at most this far above the table
    xy_tol: float = 0.03         # m: the cube's centre to the spot
    dwell_steps: int = 2


@dataclass(frozen=True)
class StandUpRewardConfig:
    progress: float = 20.0       # per m the cube centre nears the release point (held)
    upright: float = 2.0         # per rad the cube's tilt from vertical shrinks (held)
    release_height: float = 0.012  # m, the release point's cube bottom above the table
    retreat: float = 5.0         # per m the hand moves away from the released cube (capped)
    retreat_cap: float = 0.10
    stood: float = 0.5           # per step while the placed check holds
    # The per-step action costs stay below the terminal penalties over a whole episode, even
    # for the untrained (std 1) policy: with action_rate 0.05 a full episode cost -60..-90 and
    # the first runs learned to end episodes early on the tower (-5; stopped at 0.2M steps)
    action: float = 0.001        # x |a|^2 (random 6-D: about 0.005 per step)
    action_rate: float = 0.002   # x |a - a_prev|^2 (random: about 0.02 per step)
    floor: float = 0.1           # per step with the arm or fingers on the table
    toppled: float = 10.0        # the released cube fell over; terminates
    tower: float = 10.0          # the arm or cube touched the tower; terminates
    # v2 (StandUpV2Env): potential-based clearance shaping, clearance x the change of
    # -max(0, 1 - gap / clearance_margin), gap = the arm's and cube's distance to the tower:
    # closing in costs up to `clearance`, backing off earns it back (it cannot accumulate)
    clearance: float = 0.0
    clearance_margin: float = 0.05  # m


@dataclass(frozen=True)
class StandUpConfig:
    action: ActionConfig = field(default_factory=ActionConfig)
    start: StandUpStartConfig = field(default_factory=StandUpStartConfig)
    tower: TowerConfig = field(default_factory=TowerConfig)
    gate: StandUpGateConfig = field(default_factory=StandUpGateConfig)
    check: PlacedCheckConfig = field(default_factory=lambda: PlacedCheckConfig(tolerance=0.03))
    reward: StandUpRewardConfig = field(default_factory=StandUpRewardConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    horizon: int = 300           # policy steps (15 s); no termination on success
    topple_tilt: float = 0.785   # rad (45 deg) after release = toppled

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StandUpConfig":
        return load_yaml(cls, path)

    def for_evaluation(self) -> "StandUpConfig":
        """Every episode starts from the carry posture with the nominal cube."""
        return replace(self, start=replace(self.start, helpful_prob=0.0),
                       randomization=RandomizationConfig())


def load_standup_scene(gripper: GripperConfig) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Panda, one cube (weld `assist` to the hand, inactive), and a tower box (parked)."""
    spec = panda_spec()
    add_cube(spec)
    if gripper.finger_stiffness is not None:
        set_finger_stiffness(spec, gripper.finger_stiffness)
    if gripper.torsional_friction:
        spec.geom("cube").condim = 4
    spec.add_equality(name="assist", type=mujoco.mjtEq.mjEQ_WELD,
                      objtype=mujoco.mjtObj.mjOBJ_BODY, name1="hand", name2="cube", active=False)
    tower = spec.worldbody.add_body(name="tower", pos=[*TOWER_PARK, -TOWER_HALF])
    tower.add_geom(name="tower", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[0.02, 0.02, TOWER_HALF], rgba=[0.55, 0.55, 0.6, 1.0])
    model = spec.compile()
    return model, mujoco.MjData(model)


def _hand_R(direction: float) -> np.ndarray:
    """A hand turned sideways: TCP z horizontal at `direction` (rad), x pointing down."""
    z = np.array([np.cos(direction), np.sin(direction), 0.0])
    y = np.cross([0.0, 0.0, 1.0], z)
    return np.column_stack([np.cross(y, z), y, z])


def _quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


class StandUpEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: StandUpConfig | None = None) -> None:
        self.config = config or StandUpConfig()
        self.model, self.data = load_standup_scene(self.config.gripper)
        m = self.model
        self.arm = PandaIndices.from_model(m)
        self.cube = CubeIndices.from_model(m)
        self.adapter = make_action_adapter(m, self.arm, self.config.action)
        self.check = PlacedCheck(m, self.arm, self.cube, self.config.check)
        self.assist = m.equality("assist").id
        self.hand = m.body("hand").id
        self.tower_body, self.tower_geom = m.body("tower").id, m.geom("tower").id
        self.arm_bodies = {b for b in range(m.nbody) if b not in (0, self.cube.body,
                                                                  self.tower_body)}
        self._finger_qpos = [m.joint(f"finger_joint{i}").qposadr[0] for i in (1, 2)]
        self._fingers = (m.body("left_finger").id, m.body("right_finger").id)
        limit = action_limit(self.config.action)
        self.action_space = spaces.Box(-limit, limit, shape=(self.adapter.action_dim,),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self._ik = PoseIK()
        self._physics = None
        self.spot = np.zeros(2)
        self.tower = None            # (x, y, height) or None
        self.released = False
        self._steps = self._dwell = 0
        self._prev = {}
        self._prev_action = np.zeros(self.adapter.action_dim)
        self._contacts = (False, False)

    @property
    def elapsed_steps(self) -> int:
        return self._steps

    # --- episodes ------------------------------------------------------------------------
    def draw_episode(self, rng: np.random.Generator, helpful: bool = False) -> dict:
        """A start: the spot, the hand's closing yaw, the grip on the cube, and the tower."""
        s, t = self.config.start, self.config.tower
        r, a = rng.uniform(*s.radius), rng.uniform(*s.angle)
        spot = (r * np.cos(a), r * np.sin(a))
        ep = {"spot": spot, "yaw": float(rng.uniform(-np.pi, np.pi)),
              "offset": float(rng.uniform(-s.grip_offset, s.grip_offset)),
              "grip_z": float(rng.uniform(*s.grip_z)), "tower": None, "helpful": helpful}
        if rng.uniform() < t.prob:
            for _ in range(100):  # gives up quietly (no tower) if none fits
                d, b = rng.uniform(*t.distance), rng.uniform(-np.pi, np.pi)
                xy = np.array(spot) + d * np.array([np.cos(b), np.sin(b)])
                if np.linalg.norm(xy) >= t.min_radius:
                    cubes = int(rng.integers(1, t.max_cubes + 1))
                    ep["tower"] = (float(xy[0]), float(xy[1]), cubes)
                    break
        if helpful:
            ep["turn"] = float(rng.uniform(-np.pi, np.pi))   # the turned hand's z direction
            ep["height"] = float(rng.uniform(0.03, 0.12))   # cube bottom above the table
            ep["shift"] = tuple(rng.uniform(-0.02, 0.02, 2).tolist())
        return ep

    def _start(self, ep: dict) -> bool:
        """Set the scene to episode `ep`. False if the start pose is out of reach."""
        m, d, c = self.model, self.data, self.cube
        reset_to_home(m, d, self.arm)
        tower = ep.get("tower")
        if tower:
            x, y, n = tower
            m.body_pos[self.tower_body] = [x, y, n * CUBE_HEIGHT - TOWER_HALF]
            self.tower = np.array([x, y, n * CUBE_HEIGHT])
        else:
            m.body_pos[self.tower_body] = [*TOWER_PARK, -TOWER_HALF]
            self.tower = None
        self.spot = np.asarray(ep["spot"], dtype=float)
        # The grip d1 leaves: hand pointing down, fingers across the cube's width, the cube's
        # long axis horizontal along the hand's x axis
        R0 = np.zeros(9)
        mujoco.mju_quat2Mat(R0, top_down_quat(ep["yaw"]))
        R0 = R0.reshape(3, 3)
        long_axis = R0[:, 0]
        R_cube = np.column_stack([[0.0, 0.0, 1.0], np.cross(long_axis, [0.0, 0.0, 1.0]),
                                  long_axis])
        p_rel = np.array([ep["offset"], 0.0, 0.0]) + R0.T @ [0.0, 0.0, ep["grip_z"]]
        R_rel = R0.T @ R_cube
        if ep.get("helpful"):
            R = _hand_R(ep["turn"])
            centre = np.array([*(self.spot + ep["shift"]), c.half_size[2] + ep["height"]])
            tcp = centre - R @ p_rel
        else:
            R, tcp = R0, np.array([*self.spot, self.config.start.carry_height])
        q, ok = self._ik.solve(d.qpos[self.arm.arm_qpos], tcp, _quat(R))
        if not ok:
            return False
        d.qpos[self.arm.arm_qpos] = q
        d.ctrl[self.arm.arm_actuators] = q
        for adr in self._finger_qpos:
            d.qpos[adr] = c.half_size[0]                 # finger pads on the cube's faces
        d.ctrl[self.arm.gripper_actuator] = GRIPPER_CLOSED
        mujoco.mj_forward(m, d)
        site = d.site_xpos[self.arm.tcp_site].copy()
        R_site = d.site_xmat[self.arm.tcp_site].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (R_site @ R_rel).flatten())
        d.qpos[c.qpos:c.qpos + 7] = [*(site + R_site @ p_rel), *quat]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        self._attach()
        mujoco.mj_step(m, d, nstep=20)                   # fingers settle on the faces
        return True

    def _attach(self) -> None:
        """Grasp assist, as Workcell.attach: weld the cube to the hand where it is now."""
        m, d = self.model, self.data
        R_hand = d.xmat[self.hand].reshape(3, 3)
        inv, rel_quat = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(inv, d.xquat[self.hand])
        mujoco.mju_mulQuat(rel_quat, inv, d.xquat[self.cube.body])
        m.eq_data[self.assist, :3] = 0.0
        m.eq_data[self.assist, 3:6] = R_hand.T @ (d.xpos[self.cube.body] - d.xpos[self.hand])
        m.eq_data[self.assist, 6:10] = rel_quat
        m.eq_data[self.assist, 10] = 1.0
        d.eq_active[self.assist] = 1

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if "episode" in options:                         # explicit (evaluation): nominal cube
            ep = dict(options["episode"])
            if self._physics is not None:
                self._randomize(1.0, 1.0)
            if not self._start(ep):
                raise ValueError(f"start out of reach: {ep}")
        else:
            rnd = self.config.randomization
            if rnd.enabled:
                lo, hi = np.log(rnd.mass_scale)
                self._randomize(float(np.exp(self.np_random.uniform(lo, hi))),
                                float(self.np_random.uniform(*rnd.friction_scale)))
            helpful = self.np_random.uniform() < self.config.start.helpful_prob
            for _ in range(50):
                ep = self.draw_episode(self.np_random, helpful)
                if self._start(ep):
                    break
            else:
                raise RuntimeError("no reachable start in 50 draws")
        self.adapter.adopt_ctrl(self.data)
        self.check.reset()
        self.released = False
        self._steps = self._dwell = 0
        self._prev = self._progress_state()
        self._prev_action = np.zeros(self.adapter.action_dim)
        self._contacts = self._finger_contacts()
        return self._observation(), {"episode": ep, "start": "helpful" if ep.get("helpful")
                                     else "carry", "distance": self._prev["distance"]}

    def _randomize(self, mass: float, friction: float) -> None:
        if self._physics is None:
            self._physics = CubePhysics(self.model, self.cube)
        self._physics.apply(mass_scale=mass, friction_scale=friction)

    # --- measurements --------------------------------------------------------------------
    def _release_point(self) -> np.ndarray:
        r = self.config.reward
        return np.array([*self.spot, self.cube.half_size[2] + r.release_height])

    def _progress_state(self) -> dict:
        cube = cube_position(self.data, self.cube)
        away = np.linalg.norm(tcp_position(self.data, self.arm) - cube)
        return {"distance": float(np.linalg.norm(self._release_point() - cube)),
                "tilt": cube_tilt(self.data, self.cube),
                "away": float(min(away, self.config.reward.retreat_cap))}

    def _finger_contacts(self) -> tuple[bool, bool]:
        found = dict.fromkeys(self._fingers, False)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            b = (self.model.geom_bodyid[con.geom1], self.model.geom_bodyid[con.geom2])
            if self.cube.body in b:
                other = b[1] if b[0] == self.cube.body else b[0]
                if other in found:
                    found[other] = True
        return found[self._fingers[0]], found[self._fingers[1]]

    def _contact_flags(self) -> tuple[bool, bool]:
        """(arm or fingers on the table, anything on the tower)."""
        floor = tower = False
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            b = {self.model.geom_bodyid[con.geom1], self.model.geom_bodyid[con.geom2]}
            if self.tower_body in b:
                tower = True
            elif 0 in b and b & self.arm_bodies:
                floor = True
        return floor, tower

    def step(self, action):
        cfg, r = self.config, self.config.reward
        a, bound_terms = bound_action(action, cfg.action)
        self.adapter.step(self.data, a)
        self._steps += 1
        if not (np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))):
            return (np.zeros(OBS_DIM, dtype=np.float32), -r.toppled, True, False,
                    {"is_success": False, "failure": "non_finite", "reward_terms": {}})
        cube = cube_position(self.data, self.cube)
        tilt = cube_tilt(self.data, self.cube)
        xy_err = float(np.linalg.norm(cube[:2] - self.spot))
        if not self.released:
            g = cfg.gate
            bottom = cube[2] - self.cube.half_size[2]
            ready = tilt < g.max_tilt and -0.003 < bottom < g.max_drop and xy_err < g.xy_tol
            self._dwell = self._dwell + 1 if ready else 0
            if self._dwell >= g.dwell_steps:
                self.released = True
                self.data.eq_active[self.assist] = 0
                self.data.ctrl[self.arm.gripper_actuator] = GRIPPER_OPEN
        self._contacts = self._finger_contacts()
        floor, tower = self._contact_flags()
        tower = tower and self.tower is not None
        toppled = self.released and tilt > cfg.topple_tilt
        stood = self.check.update(self.data, self.spot, float(self.cube.half_size[2]))
        now = self._progress_state()
        held = not self.released
        terms = {
            "progress": r.progress * (self._prev["distance"] - now["distance"]) if held else 0.0,
            "upright": r.upright * (self._prev["tilt"] - now["tilt"]) if held else 0.0,
            "retreat": 0.0 if held else r.retreat * (now["away"] - self._prev["away"]),
            "stood": r.stood if stood else 0.0,
            "action": -r.action * float(a @ a),
            "action_rate": -r.action_rate * float(np.sum((a - self._prev_action) ** 2)),
            "floor": -r.floor if floor else 0.0,
            "toppled": -r.toppled if toppled else 0.0,
            "tower": -r.tower if tower else 0.0,
            **bound_terms,
        }
        if self.released and not self._prev.get("released"):
            terms["retreat"] = 0.0       # (the hand's distance before release is not progress)
        self._prev, self._prev_action = {**now, "released": self.released}, a
        terminated = bool(toppled or tower)
        truncated = not terminated and self._steps >= cfg.horizon
        info = {
            "is_success": stood,
            "failure": "toppled" if toppled else "tower_hit" if tower else None,
            "released": self.released,
            "distance": now["distance"],
            "xy_error": xy_err,
            "cube_tilt": tilt,
            "floor_contact": floor,
            "gripper_contact": self.check.gripper_contact(self.data),
            "reward_terms": terms,
        }
        return self._observation(), float(sum(terms.values())), terminated, truncated, info

    def _observation(self) -> np.ndarray:
        d = self.data
        R = d.site_xmat[self.arm.tcp_site].reshape(3, 3)
        cube = cube_position(d, self.cube)
        tcp = tcp_position(d, self.arm)
        target = np.array([*self.spot, self.cube.half_size[2]])
        if self.tower is None:
            tower_rel, tower = np.zeros(3), [0.0, 0.0]
        else:
            tower_rel, tower = self.tower - tcp, [1.0, float(self.tower[2])]
        return np.concatenate([
            d.qpos[self.arm.arm_qpos],
            d.qvel[self.arm.arm_qvel],
            self.adapter.q_cmd,
            tcp,
            R[:, :2].T.flatten(),
            [sum(d.qpos[adr] for adr in self._finger_qpos), float(self.released)],
            cube - tcp,
            cube_up(d, self.cube),
            [cube[2] - self.cube.half_size[2]],
            [float(self._contacts[0]), float(self._contacts[1])],
            target,
            target - cube,
            tower_rel,
            tower,
        ]).astype(np.float32)


OBS_SCHEMA_V2 = "standup-v2-56d"   # v1 + the tower clearance (distance, and its direction)
OBS_LAYOUT_V2 = OBS_LAYOUT + (("tower_gap", 1), ("tower_away", 3))
OBS_DIM_V2 = sum(n for _, n in OBS_LAYOUT_V2)
GAP_CAP = 0.2   # m: clearances beyond this read as this (and with no tower)


class StandUpV2Env(StandUpEnv):
    """Stand-up v2 (d2 round 3): v1 plus a sense of the tower before touching it.

    v1 policies stood every cube up with no tower near (12/12) but hit the tower in a quarter
    of the episodes with one, mostly mid-turn: they saw only where the tower's top is, and
    the tower penalty came with the contact. v2 observes the clearance, the smallest distance
    from the arm or the cube to the tower (capped at GAP_CAP), and the direction away from it
    (tower point to arm point), as a proximity sensor would, and adds reward.clearance shaping.
    """

    def __init__(self, config: StandUpConfig | None = None) -> None:
        super().__init__(config)
        m = self.model
        self._gap_geoms = [g for g in range(m.ngeom)
                           if (m.geom_bodyid[g] in self.arm_bodies
                               and (m.geom_contype[g] or m.geom_conaffinity[g]))
                           or g == self.cube.geom]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM_V2,),
                                            dtype=np.float32)
        self._fromto = np.zeros(6)
        self._clear = 0.0

    def _tower_gap(self) -> tuple[float, np.ndarray]:
        """The arm's and cube's clearance from the tower (m) and the vector from the tower's
        nearest point to theirs."""
        if self.tower is None:
            return GAP_CAP, np.zeros(3)
        best, away = GAP_CAP, np.zeros(3)
        for g in self._gap_geoms:
            dist = mujoco.mj_geomDistance(self.model, self.data, g, self.tower_geom, GAP_CAP,
                                          self._fromto)
            if dist < best:
                best, away = dist, self._fromto[:3] - self._fromto[3:]
        return max(float(best), 0.0), away

    def _clearance_potential(self, gap: float) -> float:
        r = self.config.reward
        return -max(0.0, 1.0 - gap / r.clearance_margin)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._clear = self._clearance_potential(self._tower_gap()[0])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if info.get("failure") == "non_finite":
            return np.zeros(OBS_DIM_V2, dtype=np.float32), reward, terminated, truncated, info
        clear = self._clearance_potential(self._tower_gap()[0])
        term = self.config.reward.clearance * (clear - self._clear)
        self._clear = clear
        info["reward_terms"]["clearance"] = term
        return obs, reward + term, terminated, truncated, info

    def _observation(self) -> np.ndarray:
        gap, away = self._tower_gap()
        return np.concatenate([super()._observation(), [gap], away]).astype(np.float32)
