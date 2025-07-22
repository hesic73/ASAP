import subprocess
import os
import re
import fire
import yaml
from loguru import logger


def run_evaluation(
    log_dir: str,
    n_epochs: int = None,
    use_xvfb: bool = False
):

    # 1. Determine n_epochs if not provided
    if n_epochs is None:
        logger.info(
            f"n_epochs not specified. Searching for the latest checkpoint in {log_dir}...")
        latest_epoch = 0
        found_checkpoint = False
        for filename in os.listdir(log_dir):
            match = re.match(r"model_(\d+)\.pt", filename)
            if match:
                epoch = int(match.group(1))
                if epoch > latest_epoch:
                    latest_epoch = epoch
                found_checkpoint = True

        if not found_checkpoint:
            # As per "let it crash", if no checkpoint is found, it will fail naturally.
            # No explicit error handling here.
            logger.error(
                f"No checkpoint (model_*.pt) found in {log_dir}. Script will terminate.")
            raise FileNotFoundError(
                f"No checkpoint (model_*.pt) found in {log_dir}.")

        n_epochs = latest_epoch
        logger.info(f"Using latest checkpoint: model_{n_epochs}.pt")
    else:
        logger.info(f"Using specified n_epochs: {n_epochs}")

    # 2. Find available GPU ID
    logger.info("Finding available GPU ID...")
    gpu_id_process = subprocess.run(
        ["bash", "scripts/bash/find_gpu.sh"],
        capture_output=True, text=True, check=True
    )
    gpu_id = gpu_id_process.stdout.strip()
    logger.info(f"Using GPU ID: {gpu_id}")

    device = f"cuda:{gpu_id}"

    # 3. Read motion_file path from config.yaml
    config_path = os.path.join(log_dir, "config.yaml")
    logger.info(f"Reading motion_file path from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Assuming the path is robot.motion.motion_file
    motion_file = config['robot']['motion']['motion_file']
    motion_xy_scale = config['robot']['motion']['xy_scale']
    logger.info(f"Found motion_file: {motion_file}")
    logger.info(f"Found motion_xy_scale: {motion_xy_scale}")

    run_name = config['run_name']
    logger.info(f"Found run_name: {run_name}")

    # --- Create video output directories ---
    isaacgym_video_dir = os.path.join(log_dir, "videos", "isaacgym")
    mujoco_video_dir = os.path.join(log_dir, "videos", "mujoco")
    os.makedirs(isaacgym_video_dir, exist_ok=True)
    os.makedirs(mujoco_video_dir, exist_ok=True)

    # 4. Run humanoidverse/eval_offline.py
    logger.info(
        "\nRunning humanoidverse/eval_offline.py for offline evaluation...")
    checkpoint_path = os.path.join(log_dir, f"model_{n_epochs}.pt")
    isaacgym_video_path = os.path.join(
        isaacgym_video_dir, f"{run_name}_{n_epochs}_isaacgym.mp4")

    eval_command = [
        "python", "humanoidverse/eval_offline.py",
        f"+device={device}",
        f"+domain_rand=NO_domain_rand",  # disable domain randomization
        "+opt=my_eval_callbacks",
        f"+checkpoint={checkpoint_path}",
        f"algo.config.eval_callbacks.offline_rendering.config.video_filename={isaacgym_video_path}"
    ]

    if use_xvfb:
        logger.info("Using xvfb...")
        eval_command = ["xvfb-run", "-s",
                        "-screen 0 800x600x24"] + eval_command

    subprocess.run(eval_command, check=True)
    logger.info("Offline evaluation complete.")

    # 5. Run humanoid_sim2sim/run_single_motion.py
    logger.info(
        "\nRunning humanoid_sim2sim/run_single_motion.py for motion simulation and video generation...")
    onnx_path = os.path.join(log_dir, "exported", f"model_{n_epochs}.onnx")
    mujoco_video_path = os.path.join(
        mujoco_video_dir, f"{run_name}_{n_epochs}_mujoco.mp4")

    sim_command = [
        "python", "humanoid_sim2sim/run_single_motion.py",
        f"policy_path={onnx_path}",
        f"video.path={mujoco_video_path}",
        f"robot.motion.motion_file={motion_file}",
        f"robot.motion.xy_scale={motion_xy_scale}"
    ]
    subprocess.run(sim_command, check=True)
    logger.info(
        f"Motion simulation and video generation complete. Video saved to: {mujoco_video_path}")


if __name__ == "__main__":
    fire.Fire(run_evaluation)
