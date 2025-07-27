import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
from pathlib import Path

from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.envs.motion_tracking.motion_tracking import LeggedRobotMotionTracking
from humanoidverse.agents.ppo.ppo import PPO
from typing import List, Dict, Any, Optional
import joblib

from termcolor import colored


class SaveMotionCallback(RL_EvalCallback):

    def __init__(self, config: Dict[str, Any], training_loop: PPO):
        super().__init__(config, training_loop)
        self.env: LeggedRobotMotionTracking = training_loop.env
        self.root_trans_offset: List[np.ndarray] = []
        self.pose_aa: List[np.ndarray] = []
        self.foot_contacts: List[np.ndarray] = []

        self.fps = 1 / self.env.dt

        self.num_augment_joint = len(
            self.env.config.robot.motion.extend_config)

        # NOTE (hsc): 这里比较hacky，从save_rendering_dir中提取ckpt_num
        self.ckpt_num = self.env.config.save_rendering_dir.split(
            '/')[-1].split('_')[-1].split('.')[0]

        self.save_motion_dir = Path(
            self.env.config.ckpt_dir) / "motions" / str(self.ckpt_num)

        self.save_motion_dir.mkdir(parents=True, exist_ok=True)

        self.motion_name = self.config.motion_name

        assert self.env.num_envs == 1, "Only support single env for now"

    def on_pre_eval_env_step(self, actor_state: Dict[str, Any]) -> Dict[str, Any]:
        super().on_pre_eval_env_step(actor_state)
        return actor_state

    def on_post_eval_env_step(self, actor_state: Dict[str, Any]) -> Dict[str, Any]:
        super().on_post_eval_env_step(actor_state)

        root_trans = self.env.simulator.robot_root_states[:, 0:3].cpu()
        if self.env.config.simulator.config.name == "isaacgym":
            # xyzw
            root_rot = self.env.simulator.robot_root_states[:, 3:7].cpu()
        elif self.env.config.simulator.config.name == "isaacsim":
            root_rot = self.env.simulator.robot_root_states[:, [
                4, 5, 6, 3]].cpu()  # wxyz to xyzw
        elif self.env.config.simulator.config.name == "genesis":
            # xyzw
            root_rot = self.env.simulator.robot_root_states[:,  3:7].cpu()
        else:
            raise NotImplementedError

        root_rot_vec = torch.from_numpy(
            sRot.from_quat(root_rot.numpy()).as_rotvec()).float()
        dof = self.env.simulator.dof_pos.cpu()
        pose_aa = torch.cat([root_rot_vec[:, None, :], self.env._motion_lib.mesh_parsers.dof_axis *
                            dof[:, :, None], torch.zeros((self.env.num_envs, self.num_augment_joint, 3))], axis=1)

        foot_contacts = (self.env.simulator.contact_forces[:,
                                                           self.env.feet_indices, 2] > 1.0).cpu()

        self.root_trans_offset.append(root_trans)
        self.pose_aa.append(pose_aa)
        self.foot_contacts.append(foot_contacts)

        return actor_state

    def on_post_evaluate_policy(self):
        super().on_post_evaluate_policy()
        self._save_motion()

    def _save_motion(self):

        start_idx = 3
        root_trans_offset = torch.stack(
            self.root_trans_offset[start_idx:]).transpose(0, 1).numpy()
        pose_aa = torch.stack(self.pose_aa[start_idx:]).transpose(0, 1).numpy()
        foot_contacts = torch.stack(
            self.foot_contacts[start_idx:]).transpose(0, 1).numpy()

        dump_data = {}
        dump_data['root_trans_offset'] = root_trans_offset[:, 0]
        dump_data['pose_aa'] = pose_aa[:, 0]
        dump_data['foot_contacts'] = foot_contacts[:, 0]
        dump_data['fps'] = self.fps

        save_path = self.save_motion_dir / \
            f"{self.motion_name}_{self.ckpt_num}.pkl"

        joblib.dump({
            f"{self.motion_name}_{self.ckpt_num}": dump_data,
        }, save_path)

        print(
            colored(f"Saved motion data to {save_path}", 'green'))
