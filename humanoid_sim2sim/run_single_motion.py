import hydra
from omegaconf import DictConfig, OmegaConf
import os
import numpy as np
import math
from typing import Tuple

from rich.progress import track

import onnxruntime

import mujoco
from scipy.spatial.transform import Rotation as R

from loguru import logger

from humanoid_sim2sim.utils.observation_manager import ObservationManager
from humanoid_sim2sim.utils.camera_manager import CameraManager
from humanoid_sim2sim.utils.mujoco_utils import (
    get_ordered_joint_indices,
    initialize_robot_state
)
from humanoid_sim2sim.utils.controller import LowLevelPDController


from humanoid_sim2sim.consts import CONFIG_DIR, ASSETS_DIR


def get_proprio(
    data: mujoco.MjData,
    mj_qpos_indices: np.ndarray,
    mj_qvel_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Retrieve:
      - joint positions
      - joint velocities
      - base angular velocity from IMU
      - gravity vector in base frame
    """
    full_qpos = data.qpos.astype(np.double)
    full_qvel = data.qvel.astype(np.double)

    joint_pos = np.array([full_qpos[idx]
                         for idx in mj_qpos_indices], dtype=np.double)
    joint_vel = np.array([full_qvel[idx]
                         for idx in mj_qvel_indices], dtype=np.double)

    quat_mj_wxyz = data.sensor("imu_quat").data
    quat_scipy_xyzw = quat_mj_wxyz[[1, 2, 3, 0]].astype(np.double)
    base_rotation = R.from_quat(quat_scipy_xyzw)

    base_ang_vel = data.sensor("imu_gyro").data.astype(
        np.double).copy()  # rad/s

    gravity_vector_world = np.array([0.0, 0.0, -1.0], dtype=np.double)
    projected_gravity = base_rotation.apply(
        gravity_vector_world, inverse=True).astype(np.double)

    return joint_pos, joint_vel, base_ang_vel, projected_gravity


def get_proprio_observations(
    data: mujoco.MjData,
    mj_qpos_indices: np.ndarray,
    mj_qvel_indices: np.ndarray,
    last_action: np.ndarray
) -> dict:
    """
    Extract proprioceptive observations from MuJoCo data.
    """
    joint_pos, joint_vel, base_ang_vel, projected_gravity = get_proprio(
        data, mj_qpos_indices, mj_qvel_indices
    )

    return {
        "dof_pos": joint_pos.astype(np.float32),
        "dof_vel": joint_vel.astype(np.float32),
        "base_ang_vel": base_ang_vel.astype(np.float32),
        "projected_gravity": projected_gravity.astype(np.float32),
        "actions": last_action.astype(np.float32)
    }


@hydra.main(version_base="1.1", config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(
        os.path.join(ASSETS_DIR, cfg.robot.asset.xml_path))
    data = mujoco.MjData(model)
    simulation_dt: float = cfg.simulation_dt
    model.opt.timestep = simulation_dt

    logger.info(f"Loaded model: {cfg.robot.asset.xml_path}")
    logger.info(f"Simulation dt: {simulation_dt}")
    logger.info(f"Number of joints: {model.njnt}")
    logger.info(f"Number of actuators: {model.nu}")

    # Get joint indices for policy
    dof_names = cfg.robot.dof_names
    mj_qpos_indices, mj_qvel_indices, mj_actuator_indices = get_ordered_joint_indices(
        model, dof_names
    )

    # Convert to numpy arrays for get_proprio function
    mj_qpos_indices = np.array(mj_qpos_indices)
    mj_qvel_indices = np.array(mj_qvel_indices)

    logger.info(f"Policy controls {len(dof_names)} joints")

    # Get floating base joint name from config (with fallback)
    floating_base_joint_name = getattr(
        cfg.robot, 'floating_base_joint', 'floating_base_joint')

    # Initialize robot state
    initialize_robot_state(model, data, cfg.robot, floating_base_joint_name)
    mujoco.mj_forward(model, data)

    # Initialize PD controller
    pd_controller = LowLevelPDController(model, cfg.robot, control_delay=0)

    # Load ONNX policy
    onnx_model_path = cfg.policy_path
    policy = onnxruntime.InferenceSession(onnx_model_path)
    logger.info(f"Loaded ONNX policy from: {onnx_model_path}")

    # Initialize observation manager
    obs_manager = ObservationManager(cfg.obs)

    # Initialize camera manager if video is enabled
    camera_manager = None
    if cfg.video.enabled:
        camera_manager = CameraManager(model, data, cfg.video)
        logger.info("Video recording enabled")

    # Get base link name from config
    base_link_name = cfg.robot.base_link_name

    # Simulation parameters from config
    total_time = cfg.total_time
    policy_dt = cfg.policy_dt
    motion_length = cfg.motion_length
    simulation_steps_per_policy_step = int(policy_dt / simulation_dt)
    total_policy_steps = int(total_time / policy_dt)

    logger.info(f"Running simulation for {total_time}s")
    logger.info(f"Policy frequency: {1/policy_dt}Hz")
    logger.info(f"Motion cycle length: {motion_length}s")
    logger.info(
        f"Sim steps per policy step: {simulation_steps_per_policy_step}")

    # Simulation variables
    last_action = np.zeros(len(dof_names), dtype=np.float32)

    step: int = 0
    for policy_step in track(range(total_policy_steps), description="Running simulation..."):
        # Calculate motion phase (0-1 cycle)
        ref_motion_phase = np.array(
            [((step + 1) * simulation_dt) / motion_length], dtype=np.float32)

        ref_motion_phase = np.clip(ref_motion_phase, 0.0, 1.0)

        # Get proprioceptive observations
        obs_dict = get_proprio_observations(
            data, mj_qpos_indices, mj_qvel_indices, last_action
        )
        obs_dict["ref_motion_phase"] = ref_motion_phase

        # Update observation manager
        obs_manager.update(obs_dict)

        # Get full observation vector
        full_obs = obs_manager.get()  # (n_obs,)

        policy_input = {policy.get_inputs()[0].name: full_obs[np.newaxis, :]}
        action = policy.run(["action"], policy_input)[0]
        action = action.flatten().astype(np.float32)

        # Apply control for multiple simulation steps (simulation_dt vs policy_dt)
        for _ in range(simulation_steps_per_policy_step):
            pd_controller.apply_control(action, data)
            mujoco.mj_step(model, data)
            step += 1

        # Render frame if video enabled
        if camera_manager is not None:
            camera_manager.render_frame(data)

        # Update for next iteration
        last_action = action.copy()

    # Save video if enabled
    if camera_manager is not None:
        camera_manager.save_video(fps=int(1.0/policy_dt))
        camera_manager.close()
        logger.info(f"Video saved to: {cfg.video.path}")

    logger.info("Simulation completed!")


if __name__ == "__main__":
    main()
