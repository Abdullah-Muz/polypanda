import mujoco
import numpy as np
import pytest

from polypanda.simulation.assets import load_panda_scene
from polypanda.simulation.panda import PandaIndices, reset_to_home, tcp_position


@pytest.fixture
def panda():
    model, data = load_panda_scene()
    idx = PandaIndices.from_model(model)
    reset_to_home(model, data, idx)
    return model, data, idx


def _main_pad_geoms(model):
    """The large fingertip pad box (fingertip_pad_collision_1) on each finger."""
    pads = []
    for finger in ("left_finger", "right_finger"):
        body = model.body(finger).id
        matches = [g for g in range(model.ngeom)
                   if model.geom_bodyid[g] == body
                   and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
                   and np.allclose(model.geom_size[g], [0.0085, 0.004, 0.0085])]
        assert len(matches) == 1, f"expected one main pad on {finger}"
        pads.append(matches[0])
    return pads


def test_tcp_attached_to_hand(panda):
    model, _, idx = panda
    assert model.body(model.site_bodyid[idx.tcp_site]).name == "hand"


@pytest.mark.parametrize("finger_opening", [0.0, 0.02, 0.04])
def test_tcp_is_midway_between_pads(panda, finger_opening):
    model, data, idx = panda
    data.qpos[model.joint("finger_joint1").qposadr[0]] = finger_opening
    data.qpos[model.joint("finger_joint2").qposadr[0]] = finger_opening
    mujoco.mj_forward(model, data)
    left, right = _main_pad_geoms(model)
    midpoint = 0.5 * (data.geom_xpos[left] + data.geom_xpos[right])
    np.testing.assert_allclose(tcp_position(data, idx), midpoint, atol=1e-9)


def test_tcp_rotates_with_base_joint(panda):
    model, data, idx = panda
    j1 = model.joint("joint1").id
    axis_point = data.xanchor[j1].copy()
    before = tcp_position(data, idx) - axis_point

    angle = 0.5
    data.qpos[idx.arm_qpos[0]] += angle
    mujoco.mj_forward(model, data)
    after = tcp_position(data, idx) - axis_point

    # joint1 rotates about world z: height and horizontal radius are preserved.
    assert after[2] == pytest.approx(before[2], abs=1e-9)
    assert np.hypot(*after[:2]) == pytest.approx(np.hypot(*before[:2]), abs=1e-9)
    turned = np.arctan2(after[1], after[0]) - np.arctan2(before[1], before[0])
    assert turned == pytest.approx(angle, abs=1e-9)


def test_tcp_moves_down_when_shoulder_pitches_forward(panda):
    model, data, idx = panda
    before = tcp_position(data, idx)
    data.qpos[idx.arm_qpos[1]] += 0.2
    mujoco.mj_forward(model, data)
    after = tcp_position(data, idx)
    assert after[2] < before[2] and after[0] > before[0]
