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

import numpy as np

from polypanda.control.ik import closing_yaw
from polypanda.control.scripted_cell import CUBE_HEIGHT, CellControllerConfig, ScriptedCell
from polypanda.environments.grasp import GRIPPER_CLOSED, GRIPPER_OPEN
from polypanda.environments.tasks import TASKS
from polypanda.policies.loader import LoadedPolicy, load_run, run_env_config
from polypanda.simulation.scenes import CubeIndices
from polypanda.simulation.workcell import Workcell
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

    def env(self, cell: Workcell, cube: CubeIndices):
        """The skill's env on the cell's scene, for one cube."""
        name = cell.model.body(cube.body).name
        return self.env_cls(self.config, scene=(cell.model, cell.data), cube=name)


class LearnedCell(ScriptedCell):
    def __init__(self, cell: Workcell, grasp: LearnedSkill | None = None,
                 place: LearnedSkill | None = None, config: CellControllerConfig | None = None,
                 on_step: Callable[[], None] | None = None, aim: str = "config") -> None:
        super().__init__(cell, config, on_step)
        if aim not in ("config", "fixture"):
            raise ValueError(f"aim must be 'config' or 'fixture', not {aim!r}")
        self.grasp, self.placer, self.aim = grasp, place, aim
        self.picks: list[dict] = []    # per learned pick: success, steps, failure
        self.places: list[dict] = []   # per learned place: level, success, steps, failure

    def _run_skill(self, skill: LearnedSkill, env, view, obs, after_step=None) -> dict:
        """Run a policy until its success predicate holds, it fails, or its horizon ends."""
        info = {}
        horizon = getattr(env.config, "horizon", None) or env.config.episode.horizon
        for _ in range(horizon):
            self.cell.conveyor.update()
            action, _ = skill.policy.predict(obs, deterministic=True)
            obs, _, terminated, _, info = view.step(action)
            if after_step:
                after_step()
            if self.on_step:
                self.on_step()
            if info.get("is_success") or terminated:
                break
        self.q_cmd = env.adapter.q_cmd.copy()
        return {"success": bool(info.get("is_success")), "steps": env.elapsed_steps,
                "failure": info.get("failure")}

    def pick(self, cube: CubeIndices) -> None:
        if self.grasp is None:
            return super().pick(cube)
        c = self.d.xpos[cube.body]
        angle = float(np.arctan2(c[1], c[0]))
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

    def _aim(self, xy, level: int) -> np.ndarray:
        """Where the learned place should set the cube down: the fixture (or slot) centre, or,
        for a policy trained to aim at the stack (PlaceConfig.stack.aim_at_top), the top cube's
        actual position. `aim="fixture"` always aims at the fixture: aiming at the cube below
        lets a small one-sided placement error add up level by level (a staircase)."""
        xy = np.asarray(xy, dtype=float)
        stack = getattr(self.placer.config, "stack", None)
        if not level or self.aim == "fixture" or not (stack and stack.aim_at_top):
            return xy
        z = CUBE_HEIGHT * (level - 0.5)
        for c in self.cell.cubes:
            p = self.d.xpos[c.body]
            if abs(p[2] - z) < 0.01 and np.linalg.norm(p[:2] - xy) < 0.03:
                return p[:2].copy()
        return xy

    def place(self, cube: CubeIndices, xy, level: int) -> None:
        """From the carry posture over a tower: lower the cube to where the place skill's
        episodes start (its bottom `handoff_gap` above the stack), then the policy sets it down.
        The grasp assist is released as soon as the policy's gate opens the fingers."""
        if self.placer is None:
            return super().place(cube, xy, level)
        rest = CUBE_HEIGHT * (level + 0.5)
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
