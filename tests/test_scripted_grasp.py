from pathlib import Path

import numpy as np
import pytest

from polypanda.control.action_adapter import JointDeltaActionAdapter
from polypanda.control.scripted_grasp import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    Phase,
    ScriptedGrasp,
    ScriptedGraspConfig,
)
from polypanda.evaluation.grasp import (
    ScriptedDriver,
    evaluate_grasp_policy,
    evaluate_scripted_grasp,
)
from polypanda.evaluation.sets import GraspEvalSet, generate_grasp_eval_set
from polypanda.simulation.panda import PandaIndices, reset_to_home
from polypanda.simulation.scenes import (
    CUBE_X_RANGE,
    CUBE_Y_RANGE,
    CubeIndices,
    load_pick_scene,
    set_cube_pose,
)

SET_PATH = Path(__file__).parents[1] / "eval_sets" / "grasp_nominal_v1.json"


@pytest.fixture(scope="module")
def eval_set():
    return GraspEvalSet.load(SET_PATH)


def test_committed_grasp_set_regenerates_exactly(eval_set):
    regenerated = generate_grasp_eval_set(eval_set.name, len(eval_set), eval_set.seed)
    np.testing.assert_array_equal(regenerated.placements, eval_set.placements)
    assert regenerated.digest == eval_set.digest and regenerated.regions == eval_set.regions


def test_grasp_set_contents(eval_set):
    p = eval_set.placements
    assert len(eval_set) == 200
    assert np.all((p[:, 0] >= CUBE_X_RANGE[0]) & (p[:, 0] <= CUBE_X_RANGE[1]))
    assert np.all((p[:, 1] >= CUBE_Y_RANGE[0]) & (p[:, 1] <= CUBE_Y_RANGE[1]))
    assert np.all(np.abs(p[:, 2]) <= np.pi)
    assert set(eval_set.regions) == {"near", "far"}


def test_tampered_grasp_set_is_rejected(tmp_path, eval_set):
    original = SET_PATH.read_text()
    text = original.replace(repr(float(eval_set.placements[0][0])), "0.5", 1)
    assert text != original
    bad = tmp_path / "bad.json"
    bad.write_text(text)
    with pytest.raises(ValueError, match="digest"):
        GraspEvalSet.load(bad)


def test_phases_run_in_order_with_matching_gripper_commands():
    model, data = load_pick_scene()
    arm, cube = PandaIndices.from_model(model), CubeIndices.from_model(model)
    adapter = JointDeltaActionAdapter(model, arm)
    ctrl = ScriptedGrasp(model, arm, cube, adapter.max_delta)
    reset_to_home(model, data, arm)
    set_cube_pose(model, data, cube, (0.45, 0.1), 0.8)
    adapter.reset_to_measured(data)
    ctrl.reset(data)
    seen = []
    for _ in range(400):
        action, gripper = ctrl.act(data)
        assert action.shape == (7,) and np.abs(action).max() <= 1.0 + 1e-6
        expected = GRIPPER_OPEN if ctrl.phase <= Phase.DESCEND else GRIPPER_CLOSED
        assert gripper == expected
        data.ctrl[arm.gripper_actuator] = gripper
        adapter.step(data, action)
        if not seen or seen[-1] != ctrl.phase:
            seen.append(ctrl.phase)
        if ctrl.phase == Phase.DONE:
            break
    assert seen == list(Phase) and ctrl.timed_out is None


def test_scripted_grasp_succeeds_on_held_out_subset(eval_set):
    summary, episodes = evaluate_scripted_grasp(eval_set.subset(5))
    assert summary["success_rate"] == 1.0 and summary["failures"] == {}
    for e in episodes:
        assert e.success and e.failure is None
        assert e.nudge_mm < 1.0 and e.floor_contact_steps == 0
        assert e.final_lift_cm > 8.0


def test_scripted_driver_succeeds_under_grasp_env_protocol(eval_set):
    summary, episodes = evaluate_grasp_policy(ScriptedDriver(), eval_set.subset(3))
    assert summary["success_rate"] == 1.0 and summary["failures"] == {}
    for e in episodes:
        assert e.gate_closures == 1 and e.floor_contact_steps == 0
        assert e.final_lift_cm >= 5.0 and e.success_fraction > 0.3


def test_phase_timeout_is_reported_as_failure(eval_set):
    summary, episodes = evaluate_scripted_grasp(eval_set.subset(1),
                                                ScriptedGraspConfig(phase_timeout=5))
    assert summary["success_rate"] == 0.0
    assert episodes[0].failure == "timeout:APPROACH"
