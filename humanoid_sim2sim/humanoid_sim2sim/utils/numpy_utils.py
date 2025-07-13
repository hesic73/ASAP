import numpy as np
from scipy.spatial.transform import Rotation as R


def _pose_to_matrix(pos: np.ndarray, rot: R) -> np.ndarray:
    """Converts a (position, Rotation) tuple to a 4x4 homogeneous transformation matrix."""
    matrix = np.eye(4)
    matrix[:3, :3] = rot.as_matrix()
    matrix[:3, 3] = pos
    return matrix


def _matrix_to_pose(matrix: np.ndarray) -> tuple[np.ndarray, R]:
    """Converts a 4x4 homogeneous transformation matrix to a (position, Rotation) tuple."""
    pos = matrix[:3, 3]
    rot = R.from_matrix(matrix[:3, :3])
    return pos, rot


def compute_relative_pose(
    pose_a: np.ndarray,
    pose_b: np.ndarray,
) -> np.ndarray:
    """
    Calculates the relative pose of pose_b with respect to pose_a,
    expressing the result in the coordinate frame of pose_a.
    Both input poses are assumed to be defined as 4x4 homogeneous
    transformation matrices in the same world frame.

    Args:
        pose_a (np.ndarray): The reference pose as a 4x4 homogeneous transformation matrix.
                             The relative pose will be expressed in this frame.
        pose_b (np.ndarray): The target pose as a 4x4 homogeneous transformation matrix.
                             We want to find the pose of this relative to pose_a.

    Returns:
        np.ndarray: The relative pose as a 4x4 homogeneous transformation matrix.
    """

    # Convert input matrices to (position, Rotation) tuples for easier manipulation
    pos1_w, rot1_w = _matrix_to_pose(pose_a)
    pos2_w, rot2_w = _matrix_to_pose(pose_b)

    # --- Calculate Relative Position ---
    # 1. Vector from pose1 origin to pose2 origin (world frame)
    delta_pos_w = pos2_w - pos1_w

    # 2. Inverse orientation of pose1 (world -> pose1 frame)
    rot1_inv = rot1_w.inv()

    # 3. Rotate delta vector into pose1's frame (pose_a's frame)
    rel_pos = rot1_inv.apply(delta_pos_w)

    # --- Calculate Relative Orientation ---
    # Rotation from pose_b frame TO pose_a frame
    # Achieved by: pose_b -> world (rot2_w), then world -> pose_a (rot1_inv)
    # Combined: rot1_inv * rot2_w
    rel_rot = rot1_inv * rot2_w

    # Convert the resulting relative pose back to a 4x4 homogeneous transformation matrix
    return _pose_to_matrix(rel_pos, rel_rot)
