import mujoco
import numpy as np


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
