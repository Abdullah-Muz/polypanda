"""Learned skills in the Milestone 9 work-order cell.

LearnedCell is the scripted cell controller (control.scripted_cell) with its pick replaced by a
learned grasp policy, trained in front of the robot on the plain table and run here at the
infeed through a station frame (supervisor.stations). The moves between stations stay scripted
(base-only swings in the carry posture).

The learned pick starts as the skill was trained: the arm in its home pose, turned by joint 1 to
face the infeed. The policy runs until its success predicate holds (a verified grasp with the
cube 5 cm up) or its episode horizon ends; the hand then rises to the carry posture. The cell's
grasp assist applies to learned grasps exactly as to scripted ones: only a verified grip is
welded, after the policy has lifted the cube.
"""

from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np

from polypanda.control.ik import closing_yaw
from polypanda.control.scripted_cell import CUBE_HEIGHT, CellControllerConfig, ScriptedCell
from polypanda.environments.grasp import GRIPPER_CLOSED, GRIPPER_OPEN
from polypanda.environments.tasks import TASKS
from polypanda.evaluation.standup import CommandedPolicyDriver
from polypanda.policies.loader import LoadedPolicy, load_run, run_env_config
from polypanda.simulation.disturbances import rest_state
from polypanda.simulation.scenes import CubeIndices
from polypanda.simulation.workcell import Workcell
from polypanda.supervisor.observe import observe
from polypanda.supervisor.stations import StationView

HANDOFF_GAP = 0.09  # m, cube bottom above its resting height where place episodes start


class LearnedSkill:
    """A trained policy and the configuration of the env it was trained in."""

    def __init__(self, run_dir: str | Path, which: str = "best") -> None:
        self.policy: LoadedPolicy = load_run(run_dir, which)
        skill = self.policy.metadata["skill"]
        self.env_cls = TASKS[skill].env_cls
        config = run_env_config(run_dir, skill)
        self.config = config.for_evaluation() if hasattr(config, "for_evaluation") else config
        # a commanded stand-up's turn chooser (scripts/train_turn_chooser.py), if it has one
        path = Path(run_dir) / f"turn_chooser_{self.policy.which}.pt"
        self.chooser = None
        if path.exists():
            from polypanda.training.turn_chooser import TurnChooser
            self.chooser = TurnChooser.load(path)

    def env(self, cell: Workcell, cube: CubeIndices):
        """The skill's env on the cell's scene, for one cube."""
        name = cell.model.body(cube.body).name
        return self.env_cls(self.config, scene=(cell.model, cell.data), cube=name)


class LearnedCell(ScriptedCell):
    def __init__(self, cell: Workcell, grasp: LearnedSkill | None = None,
                 place: LearnedSkill | None = None, config: CellControllerConfig | None = None,
                 on_step: Callable[[], None] | None = None, aim: str = "fixture",
                 standup: LearnedSkill | None = None) -> None:
        super().__init__(cell, config, on_step)
        if aim not in ("fixture", "trained"):
            raise ValueError(f"aim must be 'fixture' or 'trained', not {aim!r}")
        self.grasp, self.placer, self.aim, self.stander = grasp, place, aim, standup
        self.picks: list[dict] = []    # per learned pick: success, steps, failure
        self.places: list[dict] = []   # per learned place: level, success, steps, failure
        self.standups: list[dict] = []  # per learned stand-up: success, steps, failure, command
        self.order_index: int | None = None  # the order being run, for the per-attempt records
        self.knocks: list[dict] = []    # per cube a knock threw off: order, kind, rest state

    def _loose_others(self, cube: CubeIndices) -> int:
        """Loose cubes other than `cube` (supervisor.observe): a rise across an attempt means
        it knocked something down."""
        k = self.cell.cubes.index(cube)
        return sum(j != k for j in observe(self.cell).loose)

    def _situation(self, cube: CubeIndices, xy) -> dict:
        """What a skill attempt starts from (step e, skill outcome rates by situation): where
        the cube comes from, how it rests, where `xy` (the cube, or the spot or tower it goes
        to) is, how far the nearest other stack is from there, and the loose cubes around."""
        state = rest_state(self.cell, cube)
        stack = self.nearest_stack(xy, exclude=cube)
        return {"order": self.order_index, "source": self.source, "pose": state["pose"],
                "radius": float(np.hypot(xy[0], xy[1])),
                "angle": float(np.degrees(np.arctan2(xy[1], xy[0]))),
                "near": (None if stack is None
                         else float(np.hypot(stack[0] - xy[0], stack[1] - xy[1]))),
                "loose_before": self._loose_others(cube), "t0": float(self.d.time)}

    def _finish(self, result: dict, situation: dict, cube: CubeIndices) -> dict:
        result.update(situation, loose_after=self._loose_others(cube), t1=float(self.d.time))
        return result

    def _run_skill(self, skill: LearnedSkill, env, view, obs, after_step=None, act=None) -> dict:
        """Run a policy (or `act()`, a driver's action) until its success predicate holds, it
        fails, or its episode ends."""
        info = {}
        horizon = getattr(env.config, "horizon", None) or env.config.episode.horizon
        for _ in range(horizon):
            self.cell.conveyor.update()
            action = act() if act else skill.policy.predict(obs, deterministic=True)[0]
            obs, _, terminated, truncated, info = view.step(action)
            if after_step:
                after_step()
            if self.on_step:
                self.on_step()
            if info.get("is_success") or terminated or truncated:
                break
        self.q_cmd = env.adapter.q_cmd.copy()
        return {"success": bool(info.get("is_success")), "steps": env.elapsed_steps,
                "failure": info.get("failure")}

    def pick(self, cube: CubeIndices) -> None:
        if self.grasp is None:
            return super().pick(cube)
        c = self.d.xpos[cube.body]
        angle = float(np.arctan2(c[1], c[0]))
        situation = self._situation(cube, c[:2].copy())
        q = self.home.copy()
        q[0] += angle
        self.move_joints(q, rate=1.0)
        self.step(5)  # settle, as the trained episodes start at rest
        env = self.grasp.env(self.cell, cube)
        view = StationView(env, angle)
        obs, _ = view.reset()
        result = self._run_skill(self.grasp, env, view, obs)
        self.grip = GRIPPER_CLOSED if env.closed else GRIPPER_OPEN
        self.picks.append(result)
        if result["success"]:
            self.cell.attach(cube)
        self.to_carry()
        self._finish(result, situation, cube)

    def nearest_stack(self, xy, exclude: CubeIndices) -> tuple[float, float, float] | None:
        """(x, y, top height) of the stack of upright cubes (a tower, a buffered cube) nearest
        to `xy`, the stand-up's "tower"; None if there is none."""
        stacks: list[list] = []
        for c in self.cell.cubes:
            if c is exclude or self.d.xmat[c.body].reshape(3, 3)[2, 2] < 0.9:
                continue                                   # (lying or tilted: not a stack)
            p = self.d.xpos[c.body]
            for s in stacks:
                if np.linalg.norm(p[:2] - s[0]) < 0.03:
                    s[1] = max(s[1], p[2] + c.half_size[2])
                    break
            else:
                stacks.append([p[:2].copy(), p[2] + c.half_size[2]])
        if not stacks:
            return None
        xy, top = min(stacks, key=lambda s: np.linalg.norm(s[0] - np.asarray(xy)))
        return float(xy[0]), float(xy[1]), float(top)

    def stand_up(self, cube: CubeIndices, spot) -> None:
        """Stand a lying cube upright at `spot`: with a learned grasp (d1, which grasps lying
        cubes) it is picked where it lies and lifted to the carry posture; with a learned
        stand-up (d2) it is carried over the spot, as the scripted stand-up does, and the
        stand-up policy (its turn chooser, shield and re-planning, through the spot's station
        frame) turns it upright and lets go; then the hand rises straight up and returns to the
        carry posture. Without either, the scripted stand-up."""
        if self.grasp is None and self.stander is None:
            return super().stand_up(cube, spot)
        situation = self._situation(cube, spot)  # (t0: before the grasp, which logs as a pick)
        if self.grasp is not None:
            c = self.d.xpos[cube.body].copy()
            self.pick(cube)
            if not self.cell.attached(cube):
                return
            yaw = self.carry_yaw(float(np.arctan2(c[1], c[0])))  # where pick's to_carry leaves it
        else:
            held, c, yaw = self.grasp_lying(cube)
            if not held:
                return
        a, rel, R_rel = self.carry_over(cube, spot, c, yaw)
        if self.stander is None:
            return self.stand_held(cube, spot, a, rel, R_rel, c)
        self.step(5)  # settle, as the trained episodes start at rest
        env = self.stander.env(self.cell, cube)
        view = StationView(env, a)
        tower = self.nearest_stack(spot, exclude=cube)
        obs, _ = view.reset(spot=(float(spot[0]), float(spot[1])), tower=tower)
        driver = CommandedPolicyDriver(self.stander.policy, self.stander.chooser or "critic")
        driver.reset(env)
        result = self._run_skill(self.stander, env, view, obs, act=lambda: driver(env, None))
        result.update(command_deg=driver.choice, replans=env._replans,
                      released=bool(env.released))
        self.standups.append(result)
        self.cell.detach(cube)        # (released: already off; not: let go where it is)
        self.grip = GRIPPER_OPEN
        self.step(self.config.grip_steps)
        tcp = self.d.site_xpos[self.arm.tcp_site].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.d.site_xmat[self.arm.tcp_site])
        self.line(tcp, tcp + [0.0, 0.0, 0.12], quat)
        q = self.carry.copy()
        q[0] += a
        self.move_joints(q, rate=1.0)
        self._finish(result, situation, cube)

    @property
    def trained_at_top(self) -> bool:
        """Whether the place policy was trained to aim at the top tower cube (stacking v3)."""
        stack = getattr(getattr(self.placer, "config", None), "stack", None)
        return bool(stack and stack.aim_at_top)

    def top_cube(self, xy, level: int) -> CubeIndices | None:
        """The cube standing at `level` - 1 of the stack at `xy`, if any."""
        z = CUBE_HEIGHT * (level - 0.5)
        for c in self.cell.cubes:
            p = self.d.xpos[c.body]
            if abs(p[2] - z) < 0.01 and np.linalg.norm(p[:2] - xy) < 0.03:
                return c
        return None

    def _aim(self, xy, level: int) -> np.ndarray:
        """Where the learned place should set the cube down: the fixture (or slot) centre.

        `aim="trained"` keeps the target a v3 stacking place was trained with, the top cube's
        actual position. In the cell that lets each policy's small, repeatable placement error
        (a few mm, the same every time) add up level by level into a staircase, which the
        simulator's contact creep then slides further until the tower fails the check or falls
        (docs/workcell.md). Aiming every cube at the fixture keeps the errors from adding up."""
        xy = np.asarray(xy, dtype=float)
        if not level or self.aim == "fixture" or not self.trained_at_top:
            return xy
        top = self.top_cube(xy, level)
        return xy if top is None else self.d.xpos[top.body][:2].copy()

    def place(self, cube: CubeIndices, xy, level: int) -> None:
        """From the carry posture over a tower: lower the cube to where the place skill's
        episodes start (its bottom `handoff_gap` above the stack), then the policy sets it down.
        The grasp assist is released as soon as the policy's gate opens the fingers."""
        if self.placer is None:
            return super().place(cube, xy, level)
        rest = CUBE_HEIGHT * (level + 0.5)
        situation = self._situation(cube, xy)
        xy = self._aim(xy, level)
        tcp = self.d.site_xpos[self.arm.tcp_site]
        offset = tcp[2] - self.d.xpos[cube.body][2]
        angle = float(np.arctan2(xy[1], xy[0]))
        self.vertical(xy, rest + HANDOFF_GAP + offset, self.carry_yaw(angle), track=cube.body)
        env = self.placer.env(self.cell, cube)
        view = StationView(env, angle)
        obs, _ = view.reset(target=(xy[0], xy[1], rest))

        def release():
            if env.released and self.cell.attached(cube):
                self.cell.detach(cube)

        result = self._run_skill(self.placer, env, view, obs, after_step=release)
        result["level"] = level
        self.places.append(result)
        self.cell.detach(cube)
        self.grip = GRIPPER_OPEN
        self.step(self.config.grip_steps)
        R = self.d.site_xmat[self.arm.tcp_site].reshape(3, 3)
        self.vertical(self.d.site_xpos[self.arm.tcp_site][:2].copy(), self.config.carry_height,
                      closing_yaw(R))
        self._finish(result, situation, cube)
