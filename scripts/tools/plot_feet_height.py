import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from omegaconf import OmegaConf, DictConfig
from pathlib import Path

import fire

from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch


def plot_feet_height(filename: Path,
                     robot_config_path: Path = Path("humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml")):

    if not isinstance(filename, Path):
        filename = Path(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    robot_config = OmegaConf.load(robot_config_path)

    mesh_parser = Humanoid_Batch(robot_config.robot.motion, device=device)

    left_foot_name = "left_ankle_roll_link"
    right_foot_name = "right_ankle_roll_link"

    left_foot_idx = mesh_parser.body_names.index(left_foot_name)
    right_foot_idx = mesh_parser.body_names.index(right_foot_name)

    with open(filename, "rb") as f:
        data = pickle.load(f)
        key = list(data.keys())[0]
        data = data[key]

    fps = int(data['fps'])

    trans = torch.tensor(data['root_trans_offset'],
                         dtype=torch.float32, device=device)
    pose_aa = torch.tensor(data['pose_aa'], dtype=torch.float32, device=device)

    motion = mesh_parser.fk_batch(
        pose_aa[None, ], trans[None, ], return_full=False, dt=1.0 / fps)

    left_foot_height = motion.global_translation[0, :, left_foot_idx, 2].cpu(
    ).numpy()
    right_foot_height = motion.global_translation[0, :, right_foot_idx, 2].cpu(
    ).numpy()

    # Create time axis
    num_frames = len(left_foot_height)
    time_axis = np.arange(num_frames) / fps

    # Set figure size and style
    plt.figure(figsize=(12, 8))
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['lines.linewidth'] = 2

    # Plot foot heights
    plt.plot(time_axis, left_foot_height,
             label="Left Foot Height", color="red", alpha=0.8)
    plt.plot(time_axis, right_foot_height,
             label="Right Foot Height", color="blue", alpha=0.8)

    # Set axis labels and title
    plt.xlabel("Time (seconds)", fontsize=14, fontweight='bold')
    plt.ylabel("Height (meters)", fontsize=14, fontweight='bold')
    plt.title("Foot Height Over Time", fontsize=16, fontweight='bold')

    # Set grid
    plt.grid(True, alpha=0.3, linestyle='--')

    # Set coordinate axis precision and tick density
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))

    # Make y-axis ticks more dense (0.05 interval)
    y_min, y_max = plt.gca().get_ylim()
    y_ticks = np.arange(np.floor(y_min * 20) / 20,
                        np.ceil(y_max * 20) / 20, 0.05)
    plt.gca().set_yticks(y_ticks)

    # Set legend
    plt.legend(fontsize=12, framealpha=0.9)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(filename.with_suffix(".png"), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    fire.Fire(plot_feet_height)
