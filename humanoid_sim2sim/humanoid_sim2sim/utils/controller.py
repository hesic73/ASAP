import mujoco
import numpy as np
from typing import Dict
from omegaconf import DictConfig

from loguru import logger

from .mujoco_utils import get_ordered_joint_indices, pd_control


class LowLevelPDController:
    """Manages PD control and torque application."""

    def __init__(
        self,
        model: mujoco.MjModel,
        robot_config: DictConfig,
    ):
        """
        Initialize the PD controller.

        Args:
            model: MuJoCo model
            robot_config: Robot configuration containing control parameters
        """
        self.robot_config = robot_config
        control_config = robot_config.control

        # Get joint indices
        dof_names = robot_config.dof_names
        self.mj_qpos_indices, self.mj_qvel_indices, self.mj_actuator_indices = get_ordered_joint_indices(
            model, dof_names
        )

        self.num_joints = len(dof_names)

        # Extract PD gains and convert to numpy arrays
        self.kps = self._extract_gains(control_config.stiffness, dof_names)
        self.kds = self._extract_gains(control_config.damping, dof_names)

        # Extract torque limits and convert to numpy arrays
        self.tau_limits = self._extract_gains(
            control_config.torque_limits, dof_names)

        logger.info(f"KP gains: {self.kps}")
        logger.info(f"KD gains: {self.kds}")
        logger.info(f"Torque limits: {self.tau_limits}")

    def _extract_gains(self, gain_config: DictConfig, dof_names: list) -> np.ndarray:
        """Extract PD gains for each joint based on joint name patterns."""
        gains = np.zeros(self.num_joints)

        for i, joint_name in enumerate(dof_names):
            # Map joint names to gain categories
            if 'hip_yaw' in joint_name:
                gains[i] = gain_config.hip_yaw
            elif 'hip_roll' in joint_name:
                gains[i] = gain_config.hip_roll
            elif 'hip_pitch' in joint_name:
                gains[i] = gain_config.hip_pitch
            elif 'knee' in joint_name:
                gains[i] = gain_config.knee
            elif 'ankle_pitch' in joint_name:
                gains[i] = gain_config.ankle_pitch
            elif 'ankle_roll' in joint_name:
                gains[i] = gain_config.ankle_roll
            elif 'waist_yaw' in joint_name:
                gains[i] = gain_config.waist_yaw
            elif 'waist_roll' in joint_name:
                gains[i] = gain_config.waist_roll
            elif 'waist_pitch' in joint_name:
                gains[i] = gain_config.waist_pitch
            elif 'shoulder_pitch' in joint_name:
                gains[i] = gain_config.shoulder_pitch
            elif 'shoulder_roll' in joint_name:
                gains[i] = gain_config.shoulder_roll
            elif 'shoulder_yaw' in joint_name:
                gains[i] = gain_config.shoulder_yaw
            elif 'elbow' in joint_name:
                gains[i] = gain_config.elbow
            else:
                raise ValueError(
                    f"Unknown joint name '{joint_name}' in gain configuration")

        return gains

    def apply_control(self, target_dof_pos: np.ndarray, data: mujoco.MjData):
        target_dq = np.zeros(self.num_joints, dtype=np.float64)

        # Get current joint states
        current_q_for_pd = data.qpos[self.mj_qpos_indices].astype(
            np.float64).copy()
        current_dq_for_pd = data.qvel[self.mj_qvel_indices].astype(
            np.float64).copy()

        # Compute PD control torques
        tau = pd_control(
            target_dof_pos,
            current_q_for_pd,
            self.kps,
            target_dq,
            current_dq_for_pd,
            self.kds,
        )

        # Clip torques if enabled
        if self.robot_config.control.get('clip_torques', True):
            tau = np.clip(tau, -self.tau_limits, self.tau_limits)

        # Apply torques to actuators
        data.ctrl[self.mj_actuator_indices] = tau
