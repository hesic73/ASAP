import mujoco
import numpy as np
from typing import List, Tuple, Dict
from omegaconf import DictConfig


def get_pose(id: str, data: mujoco.MjData) -> np.ndarray:
    """
    Get the pose of a body.
    Returns a numpy array of shape (7,) containing position (xyz) and orientation (wxyz).
    """
    pos = data.body(id).xpos.copy()
    quat_wxyz = data.body(id).xquat.copy()
    return np.concatenate([pos, quat_wxyz])


def set_free_joint_pose(
    joint_name: str,
    pose: np.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData,
):
    """
    Set the pose of a free joint.
    pose: xyz+wxyz
    """
    jid = model.joint(joint_name).id
    qpos_start = model.jnt_qposadr[jid]
    data.qpos[qpos_start: qpos_start + 3] = pose[:3]
    data.qpos[qpos_start + 3: qpos_start + 7] = pose[3:]


def get_ordered_joint_indices(
    model: mujoco.MjModel, policy_joint_order: List[str]
) -> Tuple[List[int], List[int], List[int]]:
    """
    Get ordered joint indices for MuJoCo model based on policy joint order.

    Args:
        model: MuJoCo model
        policy_joint_order: List of joint names in the order expected by the policy

    Returns:
        Tuple containing:
        - mj_qpos_indices: List of position indices in MuJoCo qpos array
        - mj_qvel_indices: List of velocity indices in MuJoCo qvel array  
        - mj_actuator_indices: List of actuator indices in MuJoCo control array
    """
    mj_qpos_indices: List[int] = []
    mj_qvel_indices: List[int] = []
    mj_actuator_indices: List[int] = []

    for joint_name in policy_joint_order:
        try:
            joint_id = model.joint(joint_name).id
        except KeyError:
            available_joints = [mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
            raise ValueError(
                f"Configuration Error: Joint '{joint_name}' "
                f"not found in the MuJoCo model. Available joints: {available_joints}"
            )

        mj_qpos_indices.append(model.jnt_qposadr[joint_id])
        mj_qvel_indices.append(model.jnt_dofadr[joint_id])

        # Find the actuator that targets this joint
        actuator_found_for_joint = False
        for act_id in range(model.nu):
            # Check if actuator targets a joint and if it's the correct joint
            if (
                model.actuator_trntype[act_id] == mujoco.mjtTrn.mjTRN_JOINT
                and model.actuator_trnid[act_id, 0] == joint_id
            ):
                mj_actuator_indices.append(act_id)
                actuator_found_for_joint = True
                break

        if not actuator_found_for_joint:
            all_actuator_targets = [
                f"Actuator '{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)}' targets joint ID {model.actuator_trnid[i, 0]} (name: {model.joint(model.actuator_trnid[i, 0]).name if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT else 'N/A'})"
                for i in range(model.nu)
                if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
            ]
            raise ValueError(
                f"Configuration Error: No actuator found in the MuJoCo model that directly targets "
                f"joint '{joint_name}' (ID: {joint_id}). "
                f"Please ensure your MJCF defines an actuator (e.g. <motor joint='...'/> or <position joint='...'/>) for this joint. "
                f"Details of joint-targeting actuators in model: {all_actuator_targets if all_actuator_targets else 'None'}"
            )

    expected_len = len(policy_joint_order)
    if not (
        len(mj_qpos_indices) == expected_len
        and len(mj_qvel_indices) == expected_len
        and len(mj_actuator_indices) == expected_len
    ):
        # This case should ideally be caught by earlier checks, but it's a safeguard.
        raise RuntimeError(
            "Internal error in get_ordered_joint_indices: "
            "Output list lengths do not match input policy joint order length, despite no explicit errors raised."
        )
    return mj_qpos_indices, mj_qvel_indices, mj_actuator_indices


def initialize_robot_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_config: DictConfig,
    floating_base_joint_name: str,
):
    """
    Initialize the robot to its default state based on the configuration.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        robot_config: Robot configuration containing init_state
        floating_base_joint_name: Name of the floating base joint
    """
    init_state = robot_config.init_state

    # Set floating base pose (position + orientation)
    base_pose = np.concatenate([
        np.array(init_state.pos),  # xyz position
        np.array(init_state.rot)   # xyzw quaternion
    ])
    set_free_joint_pose(floating_base_joint_name, base_pose, model, data)

    # Set floating base velocities
    try:
        base_joint_id = model.joint(floating_base_joint_name).id
        vel_start = model.jnt_dofadr[base_joint_id]
        # Set linear velocity (3 components)
        data.qvel[vel_start:vel_start + 3] = np.array(init_state.lin_vel)
        # Set angular velocity (3 components)
        data.qvel[vel_start + 3:vel_start + 6] = np.array(init_state.ang_vel)
    except KeyError:
        print(
            f"Warning: Floating base joint '{floating_base_joint_name}' not found for velocity initialization")

    # Set joint angles using the joint indices
    dof_names = robot_config.dof_names
    mj_qpos_indices, _, _ = get_ordered_joint_indices(model, dof_names)

    default_joint_angles = init_state.default_joint_angles
    for i, joint_name in enumerate(dof_names):
        if joint_name in default_joint_angles:
            qpos_idx = mj_qpos_indices[i]
            data.qpos[qpos_idx] = default_joint_angles[joint_name]


def pd_control(
    target_q: np.ndarray,
    current_q: np.ndarray,
    kp: np.ndarray,
    target_dq: np.ndarray,
    current_dq: np.ndarray,
    kd: np.ndarray
) -> np.ndarray:
    """
    Compute PD control torques.

    Args:
        target_q: Target joint positions
        current_q: Current joint positions
        kp: Proportional gains
        target_dq: Target joint velocities
        current_dq: Current joint velocities
        kd: Derivative gains

    Returns:
        Control torques
    """
    return kp * (target_q - current_q) + kd * (target_dq - current_dq)
