"""Name-validated index lookup for the Menagerie Panda.

Joints and actuators are resolved by name rather than assumed positions, so a model
update that reorders or adds elements fails loudly instead of silently driving the
wrong joint.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
ARM_ACTUATORS = tuple(f"actuator{i}" for i in range(1, 8))
GRIPPER_ACTUATOR = "actuator8"
HOME_KEY = "home"


def _id(model: mujoco.MjModel, obj: mujoco.mjtObj, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj, name)
    if idx < 0:
        raise KeyError(f"Panda model has no {obj.name} named {name!r}")
    return idx


@dataclass(frozen=True)
class PandaIndices:
    arm_qpos: np.ndarray       # qpos addresses of the 7 arm joints
    arm_qvel: np.ndarray       # qvel (dof) addresses of the 7 arm joints
    arm_actuators: np.ndarray  # actuator ids driving the 7 arm joints
    gripper_actuator: int
    home_key: int

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> "PandaIndices":
        joints = [_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        actuators = [_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_ACTUATORS]
        for act, joint in zip(actuators, joints, strict=True):
            if model.actuator_trntype[act] != mujoco.mjtTrn.mjTRN_JOINT:
                raise ValueError(f"actuator {act} does not drive a joint")
            if model.actuator_trnid[act, 0] != joint:
                raise ValueError(f"actuator {act} drives joint {model.actuator_trnid[act, 0]}, "
                                 f"expected {joint}")
        return cls(
            arm_qpos=np.array([model.jnt_qposadr[j] for j in joints]),
            arm_qvel=np.array([model.jnt_dofadr[j] for j in joints]),
            arm_actuators=np.array(actuators),
            gripper_actuator=_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR),
            home_key=_id(model, mujoco.mjtObj.mjOBJ_KEY, HOME_KEY),
        )


def reset_to_home(model: mujoco.MjModel, data: mujoco.MjData, idx: PandaIndices) -> None:
    """Reset state and controls to the 'home' keyframe and recompute derived quantities."""
    mujoco.mj_resetDataKeyframe(model, data, idx.home_key)
    mujoco.mj_forward(model, data)
