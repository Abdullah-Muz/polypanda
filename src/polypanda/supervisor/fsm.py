"""Pick-and-place supervisor: a finite-state machine chaining grasp, transport and place skills.

    GRASP --grasped 0.5 s--> TRANSPORT --above target 0.5 s--> PLACE --placed--> SUCCESS
      |                         |                                 |
      +------- failure (knocked over, timeout, lost grip, toppled...)
                                v
                    HOME -> cube upright, on the table, in the workspace?
                              yes: GRASP again (at most 2 retries)   no: FAILURE(stage, reason)

Each skill runs in its own environment (GraspEnv, TransportEnv, PlaceEnv), so every skill sees
exactly the observations and gating it was built and evaluated with. At each handoff the full
MuJoCo state is transferred bit-exactly (mj_getState / mj_setState) and the next skill continues
from the previous skill's joint commands. For a retry, the fingers open, the hand lifts clear and
the arm drives back to its home pose, all in simulation (so recovery costs real time and can
disturb the cube, as it would on hardware); the cube's resting pose is then measured and the
grasp skill restarts from home with the cube at that pose. Transitions are decided by the same
verified predicates used throughout (grasp predicate, transport success, placed check), never
by elapsed time. A global time budget bounds the whole episode.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from polypanda.control.ik import PoseIK
from polypanda.control.scripted_carry import level_quat
from polypanda.environments.grasp import GRIPPER_OPEN, GraspEnv
from polypanda.environments.handoff import capture_state
from polypanda.environments.place import PlaceEnv
from polypanda.environments.transport import TransportEnv
from polypanda.simulation.panda import tcp_position
from polypanda.simulation.scenes import (
    CUBE_X_RANGE,
    CUBE_Y_RANGE,
    cube_position,
    cube_tilt,
    cube_yaw,
)


class Stage(Enum):
    GRASP = "grasp"
    TRANSPORT = "transport"
    PLACE = "place"
    HOME = "home"


@dataclass(frozen=True)
class SupervisorConfig:
    max_grasp_attempts: int = 3      # first attempt + 2 retries
    handoff_steps: int = 10          # consecutive verified steps before handing over (0.5 s)
    pre_place_height: float = 0.12   # m, cube-centre height above the target before placing
    release_steps: int = 10          # recovery: hold still while the fingers open (0.5 s)
    retreat_height: float = 0.10     # m, then lift the hand straight up before going home
    retreat_steps: int = 40          # policy steps allowed for the lift
    home_tol: float = 0.02           # rad, every arm joint within this of home
    home_timeout: int = 200          # policy steps (10 s), whole recovery
    retry_max_tilt: float = 0.1745   # rad (10 deg): only upright cubes are re-grasped
    retry_max_lift: float = 0.005    # m: and only cubes resting on the table
    retry_margin: float = 0.03       # m beyond the cube placement region, still re-grasped
    time_budget_s: float = 90.0      # global timeout for the whole pick-and-place


@dataclass
class StageRecord:
    stage: str
    attempt: int
    steps: int
    outcome: str        # "ok" or a failure reason


@dataclass
class PickPlaceResult:
    success: bool
    failure_stage: str | None
    failure_reason: str | None
    grasp_attempts: int
    total_s: float
    final_xy_error_mm: float | None
    records: list[StageRecord] = field(default_factory=list)


class PickPlaceSupervisor:
    def __init__(self, grasp_driver, transport_driver, place_driver,
                 config: SupervisorConfig | None = None) -> None:
        self.config = config or SupervisorConfig()
        self.envs = {Stage.GRASP: GraspEnv(), Stage.TRANSPORT: TransportEnv(),
                     Stage.PLACE: PlaceEnv()}
        self.drivers = {Stage.GRASP: grasp_driver, Stage.TRANSPORT: transport_driver,
                        Stage.PLACE: place_driver}
        self.dt = self.envs[Stage.GRASP].adapter.control_dt
        self.budget = round(self.config.time_budget_s / self.dt)
        self.on_step = None  # optional callback(stage, env) after every physics step (demo)
        self._ik = None

    # --- skill execution --------------------------------------------------------------------
    def _run_skill(self, stage: Stage, obs, horizon: int, handoff_steps: int):
        """Step the stage's env with its driver until its success predicate has held for
        `handoff_steps` consecutive steps. Returns (outcome, steps, info)."""
        env, driver = self.envs[stage], self.drivers[stage]
        driver.reset(env)
        streak, info, step = 0, {}, 0
        for step in range(1, horizon + 1):
            obs, _, terminated, truncated, info = env.step(driver(env, obs))
            if self.on_step:
                self.on_step(stage, env)
            streak = streak + 1 if info["is_success"] else 0
            if streak >= handoff_steps:
                return "ok", step, info
            if terminated:
                return info.get("failure") or "terminated", step, info
            if truncated:
                break
        return "timeout", step, info

    def _go_home(self, env, home_q: np.ndarray, horizon: int) -> int:
        """Recovery: open the fingers and hold still, lift the hand straight up so it clears
        the cube, then drive the arm back to its home configuration in joint space."""
        cfg, adapter = self.config, env.adapter
        if self._ik is None:
            self._ik = PoseIK()
        env.data.ctrl[env.arm.gripper_actuator] = GRIPPER_OPEN
        adapter.adopt_ctrl(env.data)
        lift_to = tcp_position(env.data, env.arm) + [0.0, 0.0, cfg.retreat_height]
        quat = level_quat(env.data, env.arm)
        lifting = True
        for step in range(1, horizon + 1):
            if step <= cfg.release_steps:
                action = np.zeros(7)
            else:
                lifting = lifting and (step <= cfg.release_steps + cfg.retreat_steps
                                       and tcp_position(env.data, env.arm)[2] < lift_to[2] - 0.01)
                if lifting:
                    action = self._ik.action(env.data.qpos[env.arm.arm_qpos], lift_to, quat,
                                             0.2, adapter.max_delta)
                else:
                    action = np.clip((home_q - adapter.q_cmd) / adapter.max_delta, -1.0, 1.0)
            adapter.step(env.data, action)
            if self.on_step:
                self.on_step(Stage.HOME, env)
            if (not lifting
                    and np.abs(home_q - env.data.qpos[env.arm.arm_qpos]).max() < cfg.home_tol):
                return step
        return horizon

    def _regraspable(self, env) -> tuple[str | None, tuple[float, float, float]]:
        """None if the cube can be grasped again where it lies, else the reason it cannot."""
        cfg, c = self.config, cube_position(env.data, env.cube)
        pose = (float(c[0]), float(c[1]), cube_yaw(env.data, env.cube))
        m = cfg.retry_margin
        if cube_tilt(env.data, env.cube) >= cfg.retry_max_tilt:
            return "cube tipped over", pose
        if c[2] - env.cube.half_size[2] >= cfg.retry_max_lift:
            return "cube not resting on the table", pose
        if not (CUBE_X_RANGE[0] - m <= c[0] <= CUBE_X_RANGE[1] + m
                and CUBE_Y_RANGE[0] - m <= c[1] <= CUBE_Y_RANGE[1] + m):
            return "cube out of the workspace", pose
        return None, pose

    # --- episode ------------------------------------------------------------------------------
    def run(self, cube_pose, target_xy) -> PickPlaceResult:
        """Pick the cube at (x, y, yaw) and place it upright at target (x, y)."""
        cfg = self.config
        records: list[StageRecord] = []
        used = 0
        pose = tuple(float(v) for v in cube_pose)
        target = np.array([*np.asarray(target_xy, dtype=float)[:2], cfg.pre_place_height])
        grasp_env = self.envs[Stage.GRASP]

        def result(success, stage=None, reason=None, attempts=0, xy_error=None):
            return PickPlaceResult(success, stage.value if stage else None, reason, attempts,
                                   round(used * self.dt, 3), xy_error, records)

        for attempt in range(1, cfg.max_grasp_attempts + 1):
            obs, _ = grasp_env.reset(options={"cube": pose})
            home_q = grasp_env.data.qpos[grasp_env.arm.arm_qpos].copy()
            prev = None
            for stage in (Stage.GRASP, Stage.TRANSPORT, Stage.PLACE):
                env = self.envs[stage]
                if prev is not None:  # hand over the exact physics state
                    state = capture_state(prev.model, prev.data)
                    key = "goal" if stage == Stage.TRANSPORT else "target"
                    obs, _ = env.reset(options={"state": state, key: target})
                horizon = min(env.config.horizon if stage == Stage.PLACE
                              else env.config.episode.horizon, self.budget - used)
                handoff = 1 if stage == Stage.PLACE else cfg.handoff_steps  # check has a window
                outcome, steps, info = self._run_skill(stage, obs, horizon, handoff)
                used += steps
                records.append(StageRecord(stage.value, attempt, steps, outcome))
                if outcome != "ok":
                    failed = stage
                    break
                prev = env
            else:
                return result(True, attempts=attempt, xy_error=info["xy_error"] * 1e3)

            if used >= self.budget:
                return result(False, failed, f"{outcome}; time budget exhausted", attempt)
            steps = self._go_home(env, home_q, min(cfg.home_timeout, self.budget - used))
            used += steps
            records.append(StageRecord(Stage.HOME.value, attempt, steps, "ok"))
            why_not, pose = self._regraspable(env)
            if why_not:
                return result(False, failed, f"{outcome}; {why_not}", attempt)
            if attempt == cfg.max_grasp_attempts:
                return result(False, failed, f"{outcome}; out of attempts", attempt)
            if used >= self.budget:
                return result(False, failed, f"{outcome}; time budget exhausted", attempt)
        raise AssertionError("unreachable")
