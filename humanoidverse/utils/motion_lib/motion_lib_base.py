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


class MotionLibBase():
    def __init__(self, motion_lib_cfg, num_envs: int, device: torch.device):
        self.m_cfg = motion_lib_cfg
        self._sim_fps: float = 1/self.m_cfg.get("step_dt", 1/50)

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
        Pre-loads all motion data from disk into CPU memory at initialization.
        """
        if osp.isfile(data_path):
            motion_files = [data_path]
        else:
            motion_files = glob.glob(osp.join(data_path, "*.pkl"))

        self._motion_data_cache = []
        for f in track(motion_files, description="Pre-loading motions into memory..."):
            data = joblib.load(f)
            key = list(data.keys())[0]
            motion_data = data[key]

            pose_aa = np.asarray(motion_data['pose_aa'], dtype=np.float32)
            # NOTE (hsc): 这里我hardcode一下，如果是(motion_length, 24, 3)，我再额外extend到(motion_length, 27, 3)
            # 在其他地方处理太复杂了。
            if pose_aa.shape[1] == 24:
                pose_aa = np.concatenate(
                    [pose_aa, np.zeros((pose_aa.shape[0], 3, 3), dtype=np.float32)], axis=1)

            self._motion_data_cache.append({
                'root_trans_offset': np.asarray(motion_data['root_trans_offset'], dtype=np.float32),
                'pose_aa': pose_aa,
                'fps': motion_data['fps'],
                'foot_contacts': np.asarray(motion_data['foot_contacts'], dtype=np.float32)
            })

        self._num_unique_motions = len(self._motion_data_cache)
        logger.info(
            f"Pre-loaded {self._num_unique_motions} unique motions into memory.")

    def setup_constants(self):
        self._curr_motion_ids = None
        self._termination_history = torch.zeros(
            self._num_unique_motions).to(self._device)
        self._success_rate = torch.zeros(
            self._num_unique_motions).to(self._device)
        self._sampling_history = torch.zeros(
            self._num_unique_motions).to(self._device)
        self._sampling_prob = torch.ones(self._num_unique_motions).to(
            self._device) / self._num_unique_motions  # For use in sampling batches

    def get_motion_state(self, motion_ids: torch.Tensor, motion_times: torch.Tensor, offset: Optional[torch.Tensor] = None, xy_scale: Optional[torch.Tensor] = None):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(
            motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        local_rot0 = self.dof_pos[f0l]
        local_rot1 = self.dof_pos[f1l]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        blend = blend.unsqueeze(-1)
        blend_exp = blend.unsqueeze(-1)

        dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1
        dof_pos = (1.0 - blend) * local_rot0 + blend * local_rot1

        return_dict = {}

        rg_pos_t0 = self.gts_t[f0l]
        rg_pos_t1 = self.gts_t[f1l]

        rg_rot_t0 = self.grs_t[f0l]
        rg_rot_t1 = self.grs_t[f1l]

        body_vel_t0 = self.gvs_t[f0l]
        body_vel_t1 = self.gvs_t[f1l]

        body_ang_vel_t0 = self.gavs_t[f0l]
        body_ang_vel_t1 = self.gavs_t[f1l]
        if offset is None:
            rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + \
                blend_exp * rg_pos_t1
        else:
            rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + \
                blend_exp * rg_pos_t1 + offset[..., None, :]
        rg_rot_t = slerp(rg_rot_t0, rg_rot_t1, blend_exp)
        body_vel_t = (1.0 - blend_exp) * body_vel_t0 + \
            blend_exp * body_vel_t1
        body_ang_vel_t = (1.0 - blend_exp) * \
            body_ang_vel_t0 + blend_exp * body_ang_vel_t1

        # Extract values before applying scaling
        final_dof_pos = dof_pos.clone()
        final_dof_vel = dof_vel.view(dof_vel.shape[0], -1)
        final_rg_pos_t = rg_pos_t
        final_rg_rot_t = rg_rot_t
        final_body_vel_t = body_vel_t
        final_body_ang_vel_t = body_ang_vel_t

        # Apply xy scaling if provided
        if xy_scale is not None:
            # Get initial xy positions for current motions for all links
            # [batch_size, 27, 2]
            initial_xy = self.initial_xy_positions[motion_ids]
            if offset is not None:
                # Apply offset to initial positions for all links
                # [batch_size, 1, 2] -> [batch_size, 27, 2]
                initial_xy = initial_xy + offset[..., None, :2]

            # Scale rg_pos_t all links xy positions: (pos_xy - initial_xy) * scale + initial_xy
            final_rg_pos_t[..., :2] = (
                final_rg_pos_t[..., :2] - initial_xy) * xy_scale.unsqueeze(-1).unsqueeze(-1) + initial_xy

            # Scale body_vel_t xy for all bodies
            # [batch, 1, 2] to broadcast over bodies
            final_body_vel_t[..., :2] *= xy_scale.unsqueeze(-1).unsqueeze(-2)

        # Get foot contacts (no interpolation needed as per user requirement)
        foot_contacts = self.foot_contacts[f0l]

        return_dict.update({
            "dof_pos": final_dof_pos,
            "dof_vel": final_dof_vel,
            "rg_pos_t": final_rg_pos_t,
            "rg_rot_t": final_rg_rot_t,
            "body_vel_t": final_body_vel_t,
            "body_ang_vel_t": final_body_ang_vel_t,
            "foot_contacts": foot_contacts,
        })
        return return_dict

    def load_motions(self,
                     random_sample: bool = True,
                     start_idx: int = 0,
                     max_len: int = -1,
                     ):
        motions = []
        _motion_lengths = []
        _motion_fps = []
        _motion_dt = []
        _motion_num_frames = []

        total_len = 0.0
        self.num_joints = len(self.skeleton_tree.node_names)
        num_motion_to_load = self.num_envs

        if random_sample:
            sample_idxes = torch.multinomial(
                self._sampling_prob, num_samples=num_motion_to_load, replacement=True).to(self._device)
        else:
            sample_idxes = torch.remainder(torch.arange(
                num_motion_to_load) + start_idx, self._num_unique_motions).to(self._device)

        self._curr_motion_ids = sample_idxes

        logger.info(
            f"Loading {num_motion_to_load} motions from memory cache...")
        logger.info(
            f"Sampling motion indices: {sample_idxes[:5].cpu().numpy()}, ....")

        unique_sample_idxes, inverse_indices = torch.unique(
            sample_idxes, return_inverse=True)
        sampled_motion_data = [self._motion_data_cache[i]
                               for i in unique_sample_idxes.cpu().numpy()]

        processed_motions = []
        for curr_file_data in track(sampled_motion_data, description="Processing unique motions..."):
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

            if self.mesh_parsers is not None:
                curr_motion = self.mesh_parsers.fk_batch(
                    pose_aa[None, ], trans[None, ], return_full=True, dt=dt)
                curr_motion = EasyDict({k: v.squeeze() if torch.is_tensor(
                    v) else v for k, v in curr_motion.items()})
                # Add foot_contacts to the motion data
                curr_motion.foot_contacts = foot_contacts
            else:
                logger.error("No mesh parser found")
                # Handle case where fk is not possible
                continue
            processed_motions.append(curr_motion)

        for i in inverse_indices:
            curr_motion = processed_motions[i]
            num_frames = curr_motion.global_rotation.shape[0]
            # dt is based on the last processed motion, should be fine for now as it's per unique motion
            curr_len = dt * (num_frames - 1)

            # motion_fps is based on the last processed motion, should be fine
            _motion_fps.append(motion_fps)
            # dt is based on the last processed motion, should be fine
            _motion_dt.append(dt)
            _motion_num_frames.append(num_frames)
            motions.append(curr_motion)
            _motion_lengths.append(curr_len)

        self._motion_lengths = torch.tensor(
            _motion_lengths, device=self._device, dtype=torch.float32)
        self._motion_fps = torch.tensor(
            _motion_fps, device=self._device, dtype=torch.float32)

        self._motion_dt = torch.tensor(
            _motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(
            _motion_num_frames, device=self._device)
        self._num_motions = len(motions)

        # (*, 23)
        self.dvs = torch.cat([m.dof_vels for m in motions],
                             dim=0).float().to(self._device)

        # (*, 27, 3)
        self.gts_t = torch.cat(
            [m.global_translation_extend for m in motions], dim=0).float().to(self._device)
        # (*, 27, 4)
        self.grs_t = torch.cat(
            [m.global_rotation_extend for m in motions], dim=0).float().to(self._device)
        # (*, 27, 3)
        self.gvs_t = torch.cat(
            [m.global_velocity_extend for m in motions], dim=0).float().to(self._device)
        # (*, 27, 3)
        self.gavs_t = torch.cat(
            [m.global_angular_velocity_extend for m in motions], dim=0).float().to(self._device)

        self.dof_pos = torch.cat(
            [m.dof_pos for m in motions], dim=0).float().to(self._device)

        # (*, 2)
        self.foot_contacts = torch.cat(
            [m.foot_contacts for m in motions], dim=0).float().to(self._device)

        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        self.motion_ids = torch.arange(
            len(motions), dtype=torch.long, device=self._device)
        self.num_bodies = self.num_joints

        # Store initial xy positions for each motion (first frame all links xy positions)
        initial_xy_positions = []
        for i, motion_start in enumerate(self.length_starts):
            # Get the first frame of each motion for all links
            # [27, 2] - xy coordinates for all links
            initial_links_pos = self.gts_t[motion_start, :, :2]
            initial_xy_positions.append(initial_links_pos)

        self.initial_xy_positions = torch.stack(
            initial_xy_positions, dim=0)  # [num_motions, 27, 2]

        num_motions = self._num_motions
        total_len = self.get_total_length()
        logger.info(
            f"Processed {num_motions:d} motions with a total length of {total_len:.3f}s and {self.gts_t.shape[0]} frames.")

    def get_total_length(self) -> float:
        return self._motion_lengths.sum().item()

    def get_motion_num_steps(self, motion_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if motion_ids is None:
            return (self._motion_num_frames * self._sim_fps / self._motion_fps).ceil().int()
        else:
            return (self._motion_num_frames[motion_ids] * self._sim_fps / self._motion_fps).ceil().int()

    def sample_time(self, motion_ids: torch.Tensor, truncate_time: Optional[float] = None):
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert (truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time.to(self._device)

    def get_motion_length(self, motion_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if motion_ids is None:
            return self._motion_lengths
        else:
            return self._motion_lengths[motion_ids]

    def _calc_frame_blend(self, time: torch.Tensor, len: torch.Tensor, num_frames: torch.Tensor, dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time = time.clone()
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)
        time[time < 0] = 0

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0)

        return frame_idx0, frame_idx1, blend
