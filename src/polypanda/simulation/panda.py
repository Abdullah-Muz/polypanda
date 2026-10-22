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

# Tool centre point: midway between the two main fingertip pads, on the hand's z axis.
# Finger bodies sit 0.0584 m from the hand origin and the main pad box
# (fingertip_pad_collision_1) is centred 0.0445 m further along each finger.
# Franka documents 0.1034 m from flange to TCP; the 0.5 mm difference is pad vs. nominal.
TCP_SITE = "tcp"
TCP_OFFSET = 0.0584 + 0.0445


def add_tcp_site(spec: mujoco.MjSpec) -> None:
    """Attach the TCP site to the hand. Edits the spec in memory; Menagerie files are untouched."""
    spec.body("hand").add_site(
        name=TCP_SITE, pos=[0.0, 0.0, TCP_OFFSET],
        size=[0.006, 0.006, 0.006], rgba=[1.0, 0.2, 0.2, 1.0], group=4,
    )


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
    tcp_site: int

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
            tcp_site=_id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE),
        )


def tcp_position(data: mujoco.MjData, idx: PandaIndices) -> np.ndarray:
    """TCP position in the world frame (m). Valid after mj_forward or mj_step."""
    return data.site_xpos[idx.tcp_site].copy()


def reset_to_home(model: mujoco.MjModel, data: mujoco.MjData, idx: PandaIndices) -> None:
    """Reset state and controls to the 'home' keyframe and recompute derived quantities."""
    mujoco.mj_resetDataKeyframe(model, data, idx.home_key)
    mujoco.mj_forward(model, data)
