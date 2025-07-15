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

from humanoid_sim2sim.utils.debug_utils import print_actor_obs_v2

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


def get_observations(
    data: mujoco.MjData,
    mj_qpos_indices: np.ndarray,
    mj_qvel_indices: np.ndarray,
    last_action: np.ndarray,
    initial_joint_pos: np.ndarray,
    counter: int,
    simulation_dt: float,
    motion_length: float,
) -> dict:
    joint_pos, joint_vel, base_ang_vel, projected_gravity = get_proprio(
        data, mj_qpos_indices, mj_qvel_indices
    )

    joint_pos = joint_pos - initial_joint_pos

    ref_motion_phase = (counter + 1) * simulation_dt / motion_length
    ref_motion_phase = np.clip(ref_motion_phase, 0, 1)
    ref_motion_phase = np.array([ref_motion_phase], dtype=np.float32)

    return {
        "dof_pos": joint_pos.astype(np.float32),
        "dof_vel": joint_vel.astype(np.float32),
        "base_ang_vel": base_ang_vel.astype(np.float32),
        "projected_gravity": projected_gravity.astype(np.float32),
        "actions": last_action.astype(np.float32),
        "ref_motion_phase": ref_motion_phase,
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
    logger.info(f"qpos indices: {mj_qpos_indices}")
    logger.info(f"qvel indices: {mj_qvel_indices}")
    logger.info(f"actuator indices: {mj_actuator_indices}")

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

    # Get initial joint positions from config for relative position observations
    default_joint_angles = cfg.robot.init_state.default_joint_angles
    initial_joint_pos = np.array([
        default_joint_angles.get(joint_name, 0.0) for joint_name in dof_names
    ], dtype=np.float64)
    action_scale = cfg.robot.control.action_scale

    # Initialize PD controller
    pd_controller = LowLevelPDController(model, cfg.robot)

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
    control_decimation = cfg.control_decimation
    motion_length = cfg.motion_length
    total_policy_steps = int(total_time / simulation_dt / control_decimation)

    logger.info(f"Running simulation for {total_time}s")
    logger.info(f"Policy frequency: {1/simulation_dt/control_decimation}Hz")
    logger.info(f"Motion cycle length: {motion_length}s")

    # Simulation variables
    last_action = np.zeros(len(dof_names), dtype=np.float32)
    target_dof_pos = initial_joint_pos

    step: int = 0

    for _ in track(range(int(total_time / simulation_dt)), description="Running simulation..."):
        pd_controller.apply_control(target_dof_pos, data)
        mujoco.mj_step(model, data)
        step += 1
        if step % control_decimation == 0:
            obs_dict = get_observations(
                data,
                mj_qpos_indices,
                mj_qvel_indices,
                last_action,
                initial_joint_pos,
                step,
                simulation_dt,
                motion_length,
            )
            obs_manager.update(obs_dict)

            full_obs = obs_manager.get()  # (n_obs,)

            policy_input = {
                policy.get_inputs()[0].name: full_obs[np.newaxis, :]}
            action = policy.run(["action"], policy_input)[0]
            action = action.flatten().astype(np.float32)
            action_clip_value = cfg.robot.control.action_clip_value
            action = np.clip(action, -action_clip_value, action_clip_value)

            target_dof_pos = action * action_scale + initial_joint_pos

            last_action = action.copy()
            # Render frame if video enabled
            if camera_manager is not None:
                camera_manager.render_frame(data)

    # Save video if enabled
    if camera_manager is not None:
        logger.info("Saving video...")
        camera_manager.save_video(
            fps=int(1 / simulation_dt / control_decimation))
        camera_manager.close()
        logger.info(f"Video saved to: {cfg.video.path}")

    logger.info("Simulation completed!")


if __name__ == "__main__":
    main()
