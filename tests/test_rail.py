"""v4: the robot on a linear rail (simulation/rail.py), and controllers that follow its base."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from polypanda.control.scripted_cell import ScriptedCell
from polypanda.simulation.panda import PandaIndices, reset_to_home
from polypanda.simulation.rail import BASE_BODY, RailConfig, RailIndices, add_rail
from polypanda.simulation.scenes import CubeIndices
from polypanda.simulation.workcell import (
    CUBE_NAME,
    Conveyor,
    Workcell,
    WorkcellConfig,
    WorkOrder,
    _yaw_quat,
    build_workcell_spec,
    station_xy,
)
from polypanda.supervisor.stations import StationView

GRASP_RUN = Path(__file__).resolve().parents[1] / "runs" / "grasp_anypose9_seed1"


def rail_cell(rail: RailConfig | None = None) -> Workcell:
    """The cell (no factory hall) with the robot on a rail along x."""
    cfg = WorkcellConfig(dressing=False)
    spec = build_workcell_spec(cfg)
    add_rail(spec, rail or RailConfig(low=-0.6, high=0.6))
    model = spec.compile()
    cell = Workcell(model, mujoco.MjData(model), cfg)
    cell.cubes = [CubeIndices.from_model(model, CUBE_NAME.format(k)) for k in range(cfg.n_cubes)]
    cell.conveyor = Conveyor(cell)
    cell.reset(WorkOrder(recipes=((), (), ()), arrivals=()))
    return cell


def put_cube(cell: Workcell, xy, yaw: float = 0.0) -> CubeIndices:
    cube = cell.cubes[0]
    d = cell.data
    d.qpos[cube.qpos:cube.qpos + 7] = [*xy, cube.half_size[2], *_yaw_quat(yaw)]
    d.qvel[cube.qvel:cube.qvel + 6] = 0.0
    mujoco.mj_forward(cell.model, d)
    return cube


def test_the_fixed_cell_has_no_rail_and_its_base_stays_at_the_origin():
    cell = Workcell.build(WorkcellConfig(dressing=False))
    assert RailIndices.from_model(cell.model) is None
    ctl = ScriptedCell(cell)
    assert ctl.rail is None
    np.testing.assert_array_equal(ctl.base_xy(), [0.0, 0.0])
    with pytest.raises(RuntimeError):
        ctl.drive_to(0.2)


def test_the_home_posture_is_unchanged_on_the_rail():
    fixed = Workcell.build(WorkcellConfig(dressing=False))
    cell = rail_cell()
    for c in (fixed, cell):
        reset_to_home(c.model, c.data, PandaIndices.from_model(c.model))
    a = PandaIndices.from_model(fixed.model)
    b = PandaIndices.from_model(cell.model)
    np.testing.assert_allclose(cell.data.qpos[b.arm_qpos], fixed.data.qpos[a.arm_qpos])
    rail = RailIndices.from_model(cell.model)
    assert cell.data.qpos[rail.joint_qpos] == 0.0
    np.testing.assert_allclose(cell.data.xpos[cell.model.body(BASE_BODY).id], [0.0, 0.0, 0.0])


def test_the_rail_axis_must_be_horizontal():
    with pytest.raises(ValueError):
        add_rail(build_workcell_spec(WorkcellConfig(dressing=False)), RailConfig(axis=(0, 0, 1)))


def test_drive_to_moves_the_base_and_the_arm_rides_along():
    cell = rail_cell()
    ctl = ScriptedCell(cell)
    ctl.step(40)               # the arm settles under gravity first (its servos sag a little)
    tcp0 = cell.data.site_xpos[ctl.arm.tcp_site] - cell.data.xpos[ctl.base_body]
    ctl.drive_to(0.45)
    ctl.step(20)
    np.testing.assert_allclose(ctl.base_xy(), [0.45, 0.0], atol=0.001)
    tcp1 = cell.data.site_xpos[ctl.arm.tcp_site] - cell.data.xpos[ctl.base_body]
    np.testing.assert_allclose(tcp1, tcp0, atol=0.003)      # the arm held its posture
    with pytest.raises(ValueError):
        ctl.drive_to(0.9)                                     # outside the rail's range


def test_angles_and_ik_targets_are_taken_about_the_moving_base():
    cell = rail_cell()
    ctl = ScriptedCell(cell)
    ctl.drive_to(0.3)
    b = ctl.base_xy()
    assert ctl.angle_of(b + [0.0, -0.5]) == pytest.approx(-np.pi / 2, abs=1e-3)
    np.testing.assert_allclose(ctl.staging_xy(0), station_xy(-65.0, 0.36) + b, atol=1e-9)


def _scripted_pick_and_place(base: float) -> np.ndarray:
    """Pick a cube at the station at -90 deg (relative to the base), place it at the one at
    -40 deg; the cube's final position relative to the base."""
    cell = rail_cell()
    ctl = ScriptedCell(cell)
    if base:
        ctl.drive_to(base)
    b = ctl.base_xy()
    cube = put_cube(cell, b + station_xy(-90.0, 0.5), yaw=np.radians(-90.0))
    ctl.step(10)
    ctl.swing_to(np.radians(-90.0))
    ctl.pick(cube)
    assert cell.attached(cube)
    ctl.swing_to(np.radians(-40.0))
    ctl.place(cube, b + station_xy(-40.0, 0.5), 0)
    ctl.step(20)
    return cell.data.xpos[cube.body] - np.array([*b, 0.0])


def test_the_scripted_pick_and_place_does_the_same_from_anywhere_on_the_rail():
    at_origin = _scripted_pick_and_place(0.0)
    moved = _scripted_pick_and_place(0.4)
    np.testing.assert_allclose(at_origin[:2], station_xy(-40.0, 0.5), atol=0.002)
    np.testing.assert_allclose(moved, at_origin, atol=0.002)


def test_station_view_takes_points_relative_to_its_origin_and_offsets_as_they_are():
    from polypanda.environments.grasp import OBS_LAYOUT, GraspEnv
    from polypanda.supervisor.stations import _slices

    class FakeGrasp(GraspEnv):          # just enough of a skill env: the grasp's layout
        def __init__(self):
            self.adapter = object()

    sl = _slices(OBS_LAYOUT)
    n = sum(k for _, k in OBS_LAYOUT)
    obs = np.zeros(n)
    obs[sl["tcp"]] = [0.7, -0.5, 0.3]          # the hand, in the world
    obs[sl["cube_rel"]] = [0.01, 0.02, -0.03]  # an offset
    env = FakeGrasp()
    shifted = StationView(env, 0.0, origin=(0.7, 0.0)).observe(obs)
    np.testing.assert_allclose(shifted[sl["tcp"]], [0.0, -0.5, 0.3], atol=1e-6)
    np.testing.assert_allclose(shifted[sl["cube_rel"]], [0.01, 0.02, -0.03], atol=1e-6)
    fixed = StationView(env, 0.0).observe(obs)                 # the default: the origin
    np.testing.assert_allclose(fixed[sl["tcp"]], [0.7, -0.5, 0.3], atol=1e-6)


@pytest.mark.skipif(not (GRASP_RUN / "best_model.zip").exists(),
                    reason="needs the trained grasp (scripts/fetch_models.sh)")
def test_the_learned_grasp_picks_the_same_cube_from_anywhere_on_the_rail():
    from polypanda.evaluation.cell import skill_from_spec
    from polypanda.supervisor.cell import LearnedCell
    grasp = skill_from_spec(f"run:{GRASP_RUN}", "pick")[0]
    results = []
    for base in (0.0, 0.4):
        cell = rail_cell()
        ctl = LearnedCell(cell, grasp=grasp, aim="fixture")
        if base:
            ctl.drive_to(base)
        b = ctl.base_xy()
        ctl.source = "loose"
        cube = put_cube(cell, b + station_xy(-90.0, 0.36), yaw=np.radians(-80.0))
        ctl.step(10)
        ctl.pick(cube)
        r = ctl.picks[-1]
        results.append((r["success"], r["steps"]))
    assert results[0][0] and results[1][0]
    assert abs(results[0][1] - results[1][1]) <= 3   # the same pick, give or take a few steps
