import glob
import os.path as osp
import numpy as np
import joblib
import torch
import random

from humanoidverse.utils.motion_lib.motion_utils.flags import flags
from enum import Enum
from humanoidverse.utils.motion_lib.skeleton import SkeletonTree
from pathlib import Path
from easydict import EasyDict
from loguru import logger
from rich.progress import track

from typing import Optional, Sequence, Dict, Union, List, Any, Tuple

from humanoidverse.utils.torch_utils import slerp


def to_torch(tensor) -> torch.Tensor:
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)


class MotionLibBaseOptimized():
    """
    Optimized motion library for single unique motion with discrete time steps.
    
    Assumptions:
    1. Only one unique motion
    2. motion_times are discrete with minimum unit of step_dt
    
    This class precomputes motion states for all discrete time steps during loading,
    then applies offset and xy_scale during get_motion_state calls.
    """
    
    def __init__(self, motion_lib_cfg, num_envs: int, device: torch.device):
        self.m_cfg = motion_lib_cfg
        self._sim_fps: float = 1/self.m_cfg.get("step_dt", 1/50)
        self.time_step_size = self.m_cfg.get("step_dt", 1/50)
        
        self.num_envs = num_envs
        self._device = device
        self.mesh_parsers = None
        skeleton_file = Path(self.m_cfg.asset.assetRoot) / \
            self.m_cfg.asset.assetFileName
        self.skeleton_tree = SkeletonTree.from_mjcf(skeleton_file)
        logger.info(f"Loaded skeleton from {skeleton_file}")
        logger.info(
            f"Pre-loading motion data from {self.m_cfg.motion_file} into memory...")
        self._preload_data(self.m_cfg.motion_file)
        self.setup_constants()

    def _preload_data(self, data_path: str):
        """
        Pre-loads motion data from disk into CPU memory at initialization.
        Assumes only one unique motion file.
        """
        if osp.isfile(data_path):
            motion_files = [data_path]
        else:
            motion_files = glob.glob(osp.join(data_path, "*.pkl"))
        
        if len(motion_files) != 1:
            logger.warning(f"Found {len(motion_files)} motion files, but this optimized version expects exactly 1. Using the first one.")

        self._num_unique_motions = 1
        
        # Load the single motion file
        f = motion_files[0]
        data = joblib.load(f)
        key = list(data.keys())[0]
        motion_data = data[key]

        pose_aa = np.asarray(motion_data['pose_aa'], dtype=np.float32)
        # Handle 24->27 joint extension as in original
        if pose_aa.shape[1] == 24:
            pose_aa = np.concatenate(
                [pose_aa, np.zeros((pose_aa.shape[0], 3, 3), dtype=np.float32)], axis=1)

        self._motion_data = {
            'root_trans_offset': np.asarray(motion_data['root_trans_offset'], dtype=np.float32),
            'pose_aa': pose_aa,
            'fps': motion_data['fps'],
            'foot_contacts': np.asarray(motion_data['foot_contacts'], dtype=np.float32)
        }
        
        logger.info(f"Pre-loaded single motion with {pose_aa.shape[0]} frames at {motion_data['fps']} fps.")

    def setup_constants(self):
        self._curr_motion_ids = None
        # Since we only have one motion, these are simplified
        self._termination_history = torch.zeros(1).to(self._device)
        self._success_rate = torch.zeros(1).to(self._device)
        self._sampling_history = torch.zeros(1).to(self._device)
        self._sampling_prob = torch.ones(1).to(self._device)

    def load_motions(self,
                     random_sample: bool = True,
                     start_idx: int = 0,
                     max_len: int = -1,
                     ):
        """
        Load and precompute motion states for all discrete time steps.
        """
        self.num_joints = len(self.skeleton_tree.node_names)
        
        # Process the single motion
        curr_file_data = self._motion_data
        seq_len = curr_file_data['root_trans_offset'].shape[0]
        
        if max_len == -1 or seq_len < max_len:
            start, end = 0, seq_len
        else:
            start = random.randint(0, seq_len - max_len)
            end = start + max_len

        trans = to_torch(curr_file_data['root_trans_offset']).clone()[
            start:end].to(self._device)
        pose_aa = to_torch(
            curr_file_data['pose_aa'][start:end]).clone().to(self._device)
        foot_contacts = to_torch(
            curr_file_data['foot_contacts'][start:end]).clone().to(self._device)
        motion_fps = curr_file_data['fps']
        dt = 1.0 / motion_fps

        if self.mesh_parsers is None:
            logger.error("No mesh parser found")
            return
            
        curr_motion = self.mesh_parsers.fk_batch(
            pose_aa[None, ], trans[None, ], return_full=True, dt=dt)
        curr_motion = EasyDict({k: v.squeeze() if torch.is_tensor(
            v) else v for k, v in curr_motion.items()})
        curr_motion.foot_contacts = foot_contacts

        # Store motion properties
        num_frames = curr_motion.global_rotation.shape[0]
        motion_length = dt * (num_frames - 1)
        
        self._motion_length = motion_length
        self._motion_fps = motion_fps
        self._motion_dt = dt
        self._motion_num_frames = num_frames
        
        # Extract motion data
        self.raw_dvs = curr_motion.dof_vels.float().to(self._device)
        self.raw_gts_t = curr_motion.global_translation_extend.float().to(self._device)
        self.raw_grs_t = curr_motion.global_rotation_extend.float().to(self._device)  
        self.raw_gvs_t = curr_motion.global_velocity_extend.float().to(self._device)
        self.raw_gavs_t = curr_motion.global_angular_velocity_extend.float().to(self._device)
        self.raw_dof_pos = curr_motion.dof_pos.float().to(self._device)
        self.raw_foot_contacts = curr_motion.foot_contacts.float().to(self._device)
        
        # Store initial xy position (first frame all links xy positions)
        self.initial_xy_position = self.raw_gts_t[0, :, :2]  # [27, 2]
        
        # Precompute all discrete time steps
        self._precompute_discrete_states()
        
        logger.info(f"Processed single motion with length {motion_length:.3f}s and {num_frames} frames.")
        logger.info(f"Precomputed {len(self.discrete_times)} discrete time steps with step size {self.time_step_size}s.")

    def _precompute_discrete_states(self):
        """
        Precompute motion states for all discrete time steps.
        """
        # Generate all discrete time points
        max_time = self._motion_length
        num_steps = int(np.ceil(max_time / self.time_step_size)) + 1
        self.discrete_times = torch.arange(num_steps, device=self._device, dtype=torch.float32) * self.time_step_size
        
        # Clip times to motion length
        self.discrete_times = torch.clamp(self.discrete_times, 0, max_time)
        
        # Initialize lists to collect states for all time steps
        dof_pos_list = []
        dof_vel_list = []
        rg_pos_t_list = []
        rg_rot_t_list = []
        body_vel_t_list = []
        body_ang_vel_t_list = []
        foot_contacts_list = []
        
        for i, time_val in enumerate(self.discrete_times):
            # Calculate frame blend parameters
            frame_idx0, frame_idx1, blend = self._calc_frame_blend(
                time_val.unsqueeze(0), 
                torch.tensor([self._motion_length], device=self._device),
                torch.tensor([self._motion_num_frames], device=self._device), 
                torch.tensor([self._motion_dt], device=self._device)
            )
            
            f0 = frame_idx0[0].item()
            f1 = frame_idx1[0].item()
            blend_val = blend[0].item()
            
            # Interpolate all states
            blend_tensor = torch.tensor(blend_val, device=self._device)
            blend_exp = blend_tensor.unsqueeze(-1)
            blend_exp2 = blend_exp.unsqueeze(-1)
            
            # DOF states
            dof_vel = (1.0 - blend_tensor) * self.raw_dvs[f0] + blend_tensor * self.raw_dvs[f1]
            dof_pos = (1.0 - blend_tensor) * self.raw_dof_pos[f0] + blend_tensor * self.raw_dof_pos[f1]
            
            # Global states
            rg_pos_t = (1.0 - blend_exp2) * self.raw_gts_t[f0] + blend_exp2 * self.raw_gts_t[f1]
            rg_rot_t = slerp(self.raw_grs_t[f0].unsqueeze(0), self.raw_grs_t[f1].unsqueeze(0), blend_exp2.unsqueeze(0)).squeeze(0)
            body_vel_t = (1.0 - blend_exp2) * self.raw_gvs_t[f0] + blend_exp2 * self.raw_gvs_t[f1]
            body_ang_vel_t = (1.0 - blend_exp2) * self.raw_gavs_t[f0] + blend_exp2 * self.raw_gavs_t[f1]
            
            # Foot contacts (no interpolation)
            foot_contacts = self.raw_foot_contacts[f0]
            
            # Collect states
            dof_pos_list.append(dof_pos)
            dof_vel_list.append(dof_vel.view(-1))  # Flatten for consistency
            rg_pos_t_list.append(rg_pos_t)
            rg_rot_t_list.append(rg_rot_t)
            body_vel_t_list.append(body_vel_t)
            body_ang_vel_t_list.append(body_ang_vel_t)
            foot_contacts_list.append(foot_contacts)
        
        # Stack all states into tensors with time as first dimension
        self.precomputed_states = {
            'dof_pos': torch.stack(dof_pos_list, dim=0),  # [num_steps, ...]
            'dof_vel': torch.stack(dof_vel_list, dim=0),  # [num_steps, dof_dim]
            'rg_pos_t': torch.stack(rg_pos_t_list, dim=0),  # [num_steps, 27, 3]
            'rg_rot_t': torch.stack(rg_rot_t_list, dim=0),  # [num_steps, 27, 4]
            'body_vel_t': torch.stack(body_vel_t_list, dim=0),  # [num_steps, 27, 3]
            'body_ang_vel_t': torch.stack(body_ang_vel_t_list, dim=0),  # [num_steps, 27, 3]
            'foot_contacts': torch.stack(foot_contacts_list, dim=0),  # [num_steps, 2]
        }

    def get_motion_state(self, motion_ids: torch.Tensor, motion_times: torch.Tensor, offset: Optional[torch.Tensor] = None, xy_scale: Optional[torch.Tensor] = None):
        """
        Get motion state for given times using precomputed discrete states.
        
        Args:
            motion_ids: Tensor of motion IDs (ignored since we only have one motion)
            motion_times: Tensor of motion times 
            offset: Optional offset to apply to positions [batch_size, 3]
            xy_scale: Optional xy scaling factor [batch_size, 2] or [batch_size,]
        """
        batch_size = motion_times.shape[0]
        
        # Convert motion times to discrete indices
        time_indices = torch.round(motion_times / self.time_step_size).long()
        time_indices = torch.clamp(time_indices, 0, len(self.discrete_times) - 1)
        
        # Use batch indexing to get all states at once - much faster than for loop!
        final_dof_pos = self.precomputed_states['dof_pos'][time_indices]
        final_dof_vel = self.precomputed_states['dof_vel'][time_indices]
        final_rg_pos_t = self.precomputed_states['rg_pos_t'][time_indices]
        final_rg_rot_t = self.precomputed_states['rg_rot_t'][time_indices]
        final_body_vel_t = self.precomputed_states['body_vel_t'][time_indices]
        final_body_ang_vel_t = self.precomputed_states['body_ang_vel_t'][time_indices]
        foot_contacts = self.precomputed_states['foot_contacts'][time_indices]
        
        # Apply offset if provided
        if offset is not None:
            final_rg_pos_t = final_rg_pos_t + offset.unsqueeze(1)  # [batch, 1, 3] -> [batch, 27, 3]
        
        # Apply xy scaling if provided
        if xy_scale is not None:
            # Get initial xy positions for all links
            initial_xy = self.initial_xy_position.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 27, 2]
            
            if offset is not None:
                # Apply offset to initial positions for all links
                initial_xy = initial_xy + offset[:, None, :2]
            
            # Handle different xy_scale dimensions
            if xy_scale.dim() == 1:
                xy_scale = xy_scale.unsqueeze(-1).expand(-1, 2)  # [batch,] -> [batch, 2]
            
            # Scale rg_pos_t xy positions: (pos_xy - initial_xy) * scale + initial_xy  
            final_rg_pos_t[..., :2] = (
                final_rg_pos_t[..., :2] - initial_xy) * xy_scale.unsqueeze(1) + initial_xy
            
            # Scale body_vel_t xy for all bodies
            final_body_vel_t[..., :2] *= xy_scale.unsqueeze(1)
        
        return_dict = {
            "dof_pos": final_dof_pos,
            "dof_vel": final_dof_vel,
            "rg_pos_t": final_rg_pos_t,
            "rg_rot_t": final_rg_rot_t,
            "body_vel_t": final_body_vel_t,
            "body_ang_vel_t": final_body_ang_vel_t,
            "foot_contacts": foot_contacts,
        }
        
        return return_dict

    def get_total_length(self) -> float:
        return self._motion_length

    def get_motion_num_steps(self, motion_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        steps = int((self._motion_num_frames * self._sim_fps / self._motion_fps).ceil().item())
        if motion_ids is None:
            return torch.tensor([steps], device=self._device)
        else:
            return torch.full_like(motion_ids, steps, device=self._device)

    def sample_time(self, motion_ids: torch.Tensor, truncate_time: Optional[float] = None):
        n = len(motion_ids)
        phase = torch.rand(n, device=self._device)
        motion_len = self._motion_length
        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time

    def get_motion_length(self, motion_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if motion_ids is None:
            return torch.tensor([self._motion_length], device=self._device)
        else:
            return torch.full_like(motion_ids, self._motion_length, dtype=torch.float32)

    def _calc_frame_blend(self, time: torch.Tensor, len: torch.Tensor, num_frames: torch.Tensor, dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time = time.clone()
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)
        time[time < 0] = 0

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0)

        return frame_idx0, frame_idx1, blend 