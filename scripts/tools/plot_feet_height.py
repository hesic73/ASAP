import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from loguru import logger
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

    foot_contacts = torch.tensor(
        data['foot_contacts'], dtype=torch.float32, device=device)  # (N, 2)

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

    # Convert foot contacts to numpy for plotting
    left_foot_contact = foot_contacts[:, 0].cpu().numpy()
    right_foot_contact = foot_contacts[:, 1].cpu().numpy()

    # Set figure size and style
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['lines.linewidth'] = 2

    # Plot foot heights in first subplot
    ax1.plot(time_axis, left_foot_height,
             label="Left Foot Height", color="red", alpha=0.8)
    ax1.plot(time_axis, right_foot_height,
             label="Right Foot Height", color="blue", alpha=0.8)
    ax1.set_ylabel("Height (meters)", fontsize=14, fontweight='bold')
    ax1.set_title("Foot Height Over Time", fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))

    # Make y-axis ticks more dense for height plot
    y_min, y_max = ax1.get_ylim()
    y_ticks = np.arange(np.floor(y_min * 20) / 20,
                        np.ceil(y_max * 20) / 20, 0.05)
    ax1.set_yticks(y_ticks)
    ax1.legend(fontsize=12, framealpha=0.9)

    # Plot foot contacts in second subplot
    ax2.plot(time_axis, left_foot_contact,
             label="Left Foot Contact", color="red", alpha=0.8)
    ax2.plot(time_axis, right_foot_contact,
             label="Right Foot Contact", color="blue", alpha=0.8)
    ax2.set_xlabel("Time (seconds)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Contact", fontsize=14, fontweight='bold')
    ax2.set_title("Foot Contacts Over Time", fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax2.set_ylim(-0.1, 1.1)  # Set y-axis range for contact values (0 or 1)
    ax2.legend(fontsize=12, framealpha=0.9)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(filename.with_suffix(".png"), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved figure to {filename.with_suffix('.png')}")


if __name__ == "__main__":
    fire.Fire(plot_feet_height)
