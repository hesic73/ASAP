from __future__ import annotations

import time
from pathlib import Path

import fire
import joblib
import numpy as np
import viser
import yourdfpy
from scipy.spatial.transform import Rotation
from viser.extras import ViserUrdf

# Joint names in the order they appear in the pkl file's pose_aa array.
ASAP_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

# Axis for each joint corresponding to the ASAP_JOINT_ORDER list.
ASAP_JOINT_AXIS = [
    [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0],
    [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0],
    [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1],
    [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0],
]


def main(
    motion_pkl_path: Path,
    urdf_path: Path = Path(
        "humanoidverse/data/robots/g1/g1_29dof_anneal_23dof.urdf"),
) -> None:
    """
    Load G1 motion data from a PKL file and visualize it using Viser.

    Args:
        motion_pkl_path: Path to the input .pkl file containing the motion data.
                         This file can contain single-person or multi-person motion.
        urdf_path: Path to the G1 robot's URDF file.
    """
    urdf = yourdfpy.URDF.load(str(urdf_path))

    # Load data from the .pkl file
    raw_motion_data = joblib.load(motion_pkl_path)

    server = viser.ViserServer(port=8081)
    # Set Z-axis as the up direction.
    server.scene.set_up_direction("+z")

    all_person_data = {}
    asap_axis_np = np.array(ASAP_JOINT_AXIS)
    # Reconstruct data for each person/motion from the pkl format
    for person_id, pkl_data in raw_motion_data.items():
        pose_aa = np.array(pkl_data["pose_aa"])

        # Reconstruct root position
        root_pos = np.array(pkl_data["root_trans_offset"])

        # Reconstruct root quaternion from axis-angle
        root_aa = pose_aa[:, 0, :]
        root_quat = Rotation.from_rotvec(root_aa).as_quat()  # xyzw format

        # Reconstruct joint angles from axis-angle
        joints_aa = pose_aa[:, 1:24, :]
        # Project each axis-angle vector onto its corresponding axis to get the signed angle.
        joints = np.einsum('ijk,jk->ij', joints_aa, asap_axis_np)

        # Store in the format the rest of the script expects
        all_person_data[person_id] = {
            "joints": joints,
            "root_pos": root_pos,
            "root_quat": root_quat,
            "fps": pkl_data["fps"],
        }

    print(f"Found {len(all_person_data)} persons in the file.")

    robot_vis_handles = {}
    max_timesteps = 0
    for person_id, data in all_person_data.items():
        robot_frame = server.scene.add_frame(
            f"/robot_{person_id}", axes_length=0.2, axes_radius=0.01
        )
        urdf_viser = ViserUrdf(
            server, urdf_or_path=urdf, root_node_name=f"/robot_{person_id}"
        )
        robot_vis_handles[person_id] = {
            "frame": robot_frame, "urdf": urdf_viser}

        num_frames = data["joints"].shape[0]
        if num_frames > max_timesteps:
            max_timesteps = num_frames

    # Add a ground plane for better visualization.
    server.scene.add_grid(
        "/ground",
        width=20.0,
        height=20.0,
        width_segments=40,
        height_segments=40,
        plane="xy",  # Set ground plane to xy.
    )

    # Use FPS from the first motion entry as default
    default_fps = next(iter(all_person_data.values()))["fps"]

    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep", min=0, max=max_timesteps - 1, step=1, initial_value=0
        )
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=120, step=1, initial_value=int(default_fps)
        )

    def update_robot_poses(t: int):
        for person_id, data in all_person_data.items():
            handles = robot_vis_handles[person_id]
            if t >= data["joints"].shape[0]:
                handles["frame"].visible = False
                continue

            handles["frame"].visible = True

            root_pos_t = data["root_pos"][t]
            root_quat_xyzw = data["root_quat"][t]
            root_quat_wxyz = np.array(
                [root_quat_xyzw[3], root_quat_xyzw[0],
                    root_quat_xyzw[1], root_quat_xyzw[2]]
            )

            handles["frame"].position = root_pos_t
            handles["frame"].wxyz = root_quat_wxyz

            # Create a dictionary mapping joint names to angles for robust update.
            joint_angles = data["joints"][t]
            joint_cfg = {name: angle for name, angle in zip(
                ASAP_JOINT_ORDER, joint_angles)}
            handles["urdf"].update_cfg(joint_cfg)

    @gui_timestep.on_update
    def _(_) -> None:
        update_robot_poses(gui_timestep.value)

    # Set initial pose at t=0
    update_robot_poses(0)

    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % max_timesteps
        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    fire.Fire(main)
