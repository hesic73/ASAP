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

from isaac_utils.rotations import (
    quat_angle_axis,
    quat_inverse,
    quat_mul_norm,
    get_euler_xyz,
    normalize_angle,
    slerp,
    quat_to_exp_map,
    quat_to_angle_axis,
    quat_mul,
    quat_conjugate,
)


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
        logger.info(f"Pre-loading motion data from {self.m_cfg.motion_file} into memory...")
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
            
            self._motion_data_cache.append({
                'root_trans_offset': motion_data['root_trans_offset'],
                'pose_aa': motion_data['pose_aa'],
                'fps': motion_data['fps']
            })

        self._num_unique_motions = len(self._motion_data_cache)
        logger.info(f"Pre-loaded {self._num_unique_motions} unique motions into memory.")


    def setup_constants(self):
        # ... (The rest of the original setup_constants method remains unchanged)
        self._curr_motion_ids = None
        self._termination_history = torch.zeros(
            self._num_unique_motions).to(self._device)
        self._success_rate = torch.zeros(
            self._num_unique_motions).to(self._device)
        self._sampling_history = torch.zeros(
            self._num_unique_motions).to(self._device)
        self._sampling_prob = torch.ones(self._num_unique_motions).to(
            self._device) / self._num_unique_motions  # For use in sampling batches

    def get_motion_state(self, motion_ids: torch.Tensor, motion_times: torch.Tensor, offset: Optional[torch.Tensor] = None):
        # This method remains unchanged from the original
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(
            motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        assert "dof_pos" in self.__dict__
        local_rot0 = self.dof_pos[f0l]
        local_rot1 = self.dof_pos[f1l]

        body_vel0 = self.gvs[f0l]
        body_vel1 = self.gvs[f1l]

        body_ang_vel0 = self.gavs[f0l]
        body_ang_vel1 = self.gavs[f1l]

        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        vals = [local_rot0, local_rot1, body_vel0, body_vel1, body_ang_vel0,
                body_ang_vel1, rg_pos0, rg_pos1, dof_vel0, dof_vel1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        if offset is None:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + \
                blend_exp * rg_pos1
        else:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * \
                rg_pos1 + offset[..., None, :]

        body_vel = (1.0 - blend_exp) * body_vel0 + blend_exp * body_vel1
        body_ang_vel = (1.0 - blend_exp) * body_ang_vel0 + \
            blend_exp * body_ang_vel1

        assert "dof_pos" in self.__dict__
        dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1
        dof_pos = (1.0 - blend) * local_rot0 + blend * local_rot1

        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]
        rb_rot = slerp(rb_rot0, rb_rot1, blend_exp)
        return_dict = {}

        if "gts_t" in self.__dict__:
            rg_pos_t0 = self.gts_t[f0l]
            rg_pos_t1 = self.gts_t[f1l]
            
            rg_rot_t0 = self.grs_t[f0l]
            rg_rot_t1 = self.grs_t[f1l]
            
            body_vel_t0 = self.gvs_t[f0l]
            body_vel_t1 = self.gvs_t[f1l]
            
            body_ang_vel_t0 = self.gavs_t[f0l]
            body_ang_vel_t1 = self.gavs_t[f1l]
            if offset is None:
                rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1  
            else:
                rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1 + offset[..., None, :]
            rg_rot_t = slerp(rg_rot_t0, rg_rot_t1, blend_exp)
            body_vel_t = (1.0 - blend_exp) * body_vel_t0 + blend_exp * body_vel_t1
            body_ang_vel_t = (1.0 - blend_exp) * body_ang_vel_t0 + blend_exp * body_ang_vel_t1
        else:
            rg_pos_t = rg_pos
            rg_rot_t = rb_rot
            body_vel_t = body_vel
            body_ang_vel_t = body_ang_vel

        return_dict.update({
            "root_pos": rg_pos[..., 0, :].clone(),
            "root_rot": rb_rot[..., 0, :].clone(),
            "dof_pos": dof_pos.clone(),
            "root_vel": body_vel[..., 0, :].clone(),
            "root_ang_vel": body_ang_vel[..., 0, :].clone(),
            "dof_vel": dof_vel.view(dof_vel.shape[0], -1),
            "rg_pos_t": rg_pos_t,
            "rg_rot_t": rg_rot_t,
            "body_vel_t": body_vel_t,
            "body_ang_vel_t": body_ang_vel_t,
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
        
        logger.info(f"Loading {num_motion_to_load} motions from memory cache...")
        logger.info(f"Sampling motion indices: {sample_idxes[:5].cpu().numpy()}, ....")

        sampled_motion_data = [self._motion_data_cache[i] for i in sample_idxes.cpu().numpy()]
        
        for curr_file_data in track(sampled_motion_data, description="Processing motions..."):
            seq_len = curr_file_data['root_trans_offset'].shape[0]
            if max_len == -1 or seq_len < max_len:
                start, end = 0, seq_len
            else:
                start = random.randint(0, seq_len - max_len)
                end = start + max_len
            
            trans = to_torch(curr_file_data['root_trans_offset']).clone()[start:end]
            pose_aa = to_torch(curr_file_data['pose_aa'][start:end]).clone()
            motion_fps = curr_file_data['fps']
            dt = 1.0 / motion_fps
            
            if self.mesh_parsers is not None:
                curr_motion = self.mesh_parsers.fk_batch(
                    pose_aa[None, ], trans[None, ], return_full=True, dt=dt)
                curr_motion = EasyDict({k: v.squeeze() if torch.is_tensor(
                    v) else v for k, v in curr_motion.items()})
            else:
                logger.error("No mesh parser found")
                # Handle case where fk is not possible
                continue

            num_frames = curr_motion.global_rotation.shape[0]
            curr_len = dt * (num_frames - 1)

            _motion_fps.append(motion_fps)
            _motion_dt.append(dt)
            _motion_num_frames.append(num_frames)
            motions.append(curr_motion)
            _motion_lengths.append(curr_len)

            del curr_motion

        self._motion_lengths = torch.tensor(
            _motion_lengths, device=self._device, dtype=torch.float32)
        self._motion_fps = torch.tensor(
            _motion_fps, device=self._device, dtype=torch.float32)

        self._motion_dt = torch.tensor(
            _motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(
            _motion_num_frames, device=self._device)
        self._num_motions = len(motions)

        # (*, 24, 3)
        self.gts = torch.cat(
            [m.global_translation for m in motions], dim=0).float().to(self._device)
        # (*, 24, 4)
        self.grs = torch.cat(
            [m.global_rotation for m in motions], dim=0).float().to(self._device)
        # (*, 27, 4)
        self.lrs = torch.cat(
            [m.local_rotation for m in motions], dim=0).float().to(self._device)
        # (*, 3)
        self.grvs = torch.cat(
            [m.global_root_velocity for m in motions], dim=0).float().to(self._device)
        # (*, 3)
        self.gravs = torch.cat(
            [m.global_root_angular_velocity for m in motions], dim=0).float().to(self._device)
        # (*, 24, 3)
        self.gavs = torch.cat(
            [m.global_angular_velocity for m in motions], dim=0).float().to(self._device)
        # (*, 24, 3)
        self.gvs = torch.cat(
            [m.global_velocity for m in motions], dim=0).float().to(self._device)
        # (*, 23)
        self.dvs = torch.cat([m.dof_vels for m in motions],
                                dim=0).float().to(self._device)

        if "global_translation_extend" in motions[0].__dict__:
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

        assert "dof_pos" in motions[0].__dict__
        self.dof_pos = torch.cat(
            [m.dof_pos for m in motions], dim=0).float().to(self._device)

        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        self.motion_ids = torch.arange(
            len(motions), dtype=torch.long, device=self._device)
        self.num_bodies = self.num_joints

        num_motions = self._num_motions
        total_len = self.get_total_length()
        logger.info(
            f"Processed {num_motions:d} motions with a total length of {total_len:.3f}s and {self.gts.shape[0]} frames.")


    def get_total_length(self) -> float:
        # This method remains unchanged from the original
        return self._motion_lengths.sum().item()

    def get_motion_num_steps(self, motion_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # This method remains unchanged from the original
        if motion_ids is None:
            return (self._motion_num_frames * self._sim_fps / self._motion_fps).ceil().int()
        else:
            return (self._motion_num_frames[motion_ids] * self._sim_fps / self._motion_fps).ceil().int()

    def sample_time(self, motion_ids: torch.Tensor, truncate_time: Optional[float] = None):
        # This method remains unchanged from the original
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert (truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time.to(self._device)

    def get_motion_length(self, motion_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # This method remains unchanged from the original
        if motion_ids is None:
            return self._motion_lengths
        else:
            return self._motion_lengths[motion_ids]

    def _calc_frame_blend(self, time: torch.Tensor, len: torch.Tensor, num_frames: torch.Tensor, dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # This method remains unchanged from the original
        time = time.clone()
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)
        time[time < 0] = 0

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0)

        return frame_idx0, frame_idx1, blend