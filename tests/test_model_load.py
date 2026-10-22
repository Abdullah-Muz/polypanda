import mujoco
import numpy as np
import pytest

from polypanda.simulation.assets import MENAGERIE_COMMIT, load_panda_scene, menagerie_revision
from polypanda.simulation.panda import PandaIndices


@pytest.fixture(scope="module")
def model():
    return load_panda_scene()[0]


def test_menagerie_is_pinned():
    assert menagerie_revision() == MENAGERIE_COMMIT


def test_actuator_layout(model):
    idx = PandaIndices.from_model(model)
    assert model.nu == 8
    assert len(idx.arm_actuators) == 7
    assert idx.gripper_actuator not in idx.arm_actuators


def test_timestep(model):
    assert model.opt.timestep == pytest.approx(0.002)


def test_arm_actuators_are_position_servos(model):
    idx = PandaIndices.from_model(model)
    for a in idx.arm_actuators:
        kp = model.actuator_gainprm[a, 0]
        # Affine bias of the form -kp*q - kd*qdot makes the actuator a PD position servo.
        assert model.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE
        assert model.actuator_biasprm[a, 1] == pytest.approx(-kp)
        assert model.actuator_biasprm[a, 2] < 0


def test_home_targets_match_home_pose(model):
    idx = PandaIndices.from_model(model)
    key_qpos = model.key_qpos[idx.home_key][idx.arm_qpos]
    key_ctrl = model.key_ctrl[idx.home_key][idx.arm_actuators]
    np.testing.assert_allclose(key_ctrl, key_qpos)
