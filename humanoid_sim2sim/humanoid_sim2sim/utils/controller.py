import mujoco
import numpy as np
from typing import Dict
from omegaconf import DictConfig
from .mujoco_utils import get_ordered_joint_indices, pd_control


class CircularBuffer:
    """A circular buffer for storing action history with delay."""

    def __init__(self, size: int, num_joints: int, dtype=np.float32):
        self.size = size
        self.num_joints = num_joints
        self.buffer = np.zeros((size, num_joints), dtype=dtype)
        self.index = 0
        self.filled = False

    def append(self, action: np.ndarray):
        """Add a new action to the buffer."""
        self.buffer[self.index] = action
        self.index = (self.index + 1) % self.size
        if self.index == 0:
            self.filled = True

    def get_history(self) -> np.ndarray:
        """Get the history in chronological order (oldest first)."""
        if not self.filled:
            # If buffer not filled, return from start to current index
            return self.buffer[:self.index]
        else:
            # If buffer filled, return from current index to end, then start to current index
            return np.concatenate([
                self.buffer[self.index:],
                self.buffer[:self.index]
            ])


class LowLevelPDController:
    """Manages PD control, action delay, and torque application."""

    def __init__(
        self,
        model: mujoco.MjModel,
        robot_config: DictConfig,
        control_delay: int = 0,
    ):
        """
        Initialize the PD controller.

        Args:
            model: MuJoCo model
            robot_config: Robot configuration containing control parameters
            control_delay: Number of timesteps to delay control actions
        """
        self.robot_config = robot_config
        control_config = robot_config.control

        # Get joint indices
        dof_names = robot_config.dof_names
        self.mj_qpos_indices, self.mj_qvel_indices, self.mj_actuator_indices = get_ordered_joint_indices(
            model, dof_names
        )

        self.num_joints = len(dof_names)
        self.action_scale = control_config.action_scale
        self.control_delay = control_delay

        # Extract PD gains and convert to numpy arrays
        self.kps = self._extract_gains(control_config.stiffness, dof_names)
        self.kds = self._extract_gains(control_config.damping, dof_names)

        # Extract torque limits if available
        if hasattr(control_config, 'torque_limits'):
            self.tau_limits = self._extract_gains(
                control_config.torque_limits, dof_names)
        else:
            # Use a default large value if no torque limits specified
            self.tau_limits = np.full(
                self.num_joints, control_config.get('action_clip_value', 100.0))

        # Get initial joint positions for target calculation
        default_joint_angles = robot_config.init_state.default_joint_angles
        self.initial_qpos = np.array([
            default_joint_angles.get(joint_name, 0.0) for joint_name in dof_names
        ], dtype=np.float64)

        # Initialize action delay buffer
        self.action_to_apply_buffer = CircularBuffer(
            control_delay + 1,
            self.num_joints,
            dtype=np.float32,
        )
        # Fill buffer with zeros
        for _ in range(control_delay + 1):
            self.action_to_apply_buffer.append(
                np.zeros(self.num_joints, dtype=np.float32))

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
                # Default gain if no pattern matches
                gains[i] = 50.0
                print(
                    f"Warning: No gain found for joint {joint_name}, using default value 50.0")

        return gains

    def apply_control(self, action: np.ndarray, data: mujoco.MjData):
        """
        Apply PD control based on the given action.

        Args:
            action: Control action from policy
            data: MuJoCo data
        """
        # Add current action to delay buffer
        self.action_to_apply_buffer.append(action)

        # Get the action to apply (delayed action)
        action_history = self.action_to_apply_buffer.get_history()
        action_currently_applied_to_robot = action_history[0]

        # Calculate target joint positions
        target_q = (action_currently_applied_to_robot.astype(np.float64) *
                    self.action_scale + self.initial_qpos)
        target_dq = np.zeros(self.num_joints, dtype=np.float64)

        # Get current joint states
        current_q_for_pd = data.qpos[self.mj_qpos_indices].astype(
            np.float64).copy()
        current_dq_for_pd = data.qvel[self.mj_qvel_indices].astype(
            np.float64).copy()

        # Compute PD control torques
        tau = pd_control(
            target_q,
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

    def reset(self):
        """Reset the controller state (clear action buffer)."""
        for _ in range(self.control_delay + 1):
            self.action_to_apply_buffer.append(
                np.zeros(self.num_joints, dtype=np.float32))
