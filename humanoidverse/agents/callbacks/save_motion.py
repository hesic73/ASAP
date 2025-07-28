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

        # NOTE (hsc): This is a bit hacky, extracting ckpt_num from save_rendering_dir
        self.ckpt_num = self.env.config.save_rendering_dir.split(
            '/')[-1].split('_')[-1].split('.')[0]

        self.save_motion_dir = Path(
            self.env.config.ckpt_dir) / "motions" / str(self.ckpt_num)

        self.save_motion_dir.mkdir(parents=True, exist_ok=True)

        self.motion_name = self.config.motion_name

        assert self.env.num_envs == 1, "Only support single env for now"

        # New variables to track episodes
        self.current_episode_idx = 0
        self.prev_episode_length = -1  # Initialize to -1, indicating not started yet
        self.complete_episodes = {}  # Store complete episode data

    def on_pre_eval_env_step(self, actor_state: Dict[str, Any]) -> Dict[str, Any]:
        super().on_pre_eval_env_step(actor_state)
        return actor_state

    def on_post_eval_env_step(self, actor_state: Dict[str, Any]) -> Dict[str, Any]:
        super().on_post_eval_env_step(actor_state)

        # Check if a reset has occurred
        current_episode_length = self.env.episode_length_buf.cpu().item()

        # If episode_length changes from non-zero to 0, or suddenly decreases, it indicates a reset
        if (self.prev_episode_length > 0 and current_episode_length == 0) or \
           (self.prev_episode_length > current_episode_length and current_episode_length < 10):
            # Episode ended, save current collected data (if long enough)
            self._save_current_episode()
            # Reset data collection
            self._reset_data_collection()
            self.current_episode_idx += 1

        self.prev_episode_length = current_episode_length

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
        # Save the last episode (if there's data)
        self._save_current_episode()
        # Save all complete episodes
        self._save_all_episodes()

    def _reset_data_collection(self):
        """Reset data collection lists"""
        self.root_trans_offset = []
        self.pose_aa = []
        self.foot_contacts = []

    def _save_current_episode(self):
        """Save current episode data (if data is long enough)"""
        start_idx = 3  # Skip the first few frames

        if len(self.root_trans_offset) <= start_idx:
            print(colored(
                f"Episode {self.current_episode_idx} too short ({len(self.root_trans_offset)} frames), skipping...", 'yellow'))
            return

        root_trans_offset = torch.stack(
            self.root_trans_offset[start_idx:]).transpose(0, 1).numpy()
        pose_aa = torch.stack(self.pose_aa[start_idx:]).transpose(0, 1).numpy()
        foot_contacts = torch.stack(
            self.foot_contacts[start_idx:]).transpose(0, 1).numpy()

        episode_data = {
            'root_trans_offset': root_trans_offset[:, 0],
            'pose_aa': pose_aa[:, 0],
            'foot_contacts': foot_contacts[:, 0],
            'fps': self.fps
        }

        episode_key = f"{self.motion_name}_{self.ckpt_num}_ep{self.current_episode_idx:03d}"
        self.complete_episodes[episode_key] = episode_data

        print(colored(
            f"Collected complete episode {self.current_episode_idx} with {len(self.root_trans_offset) - start_idx} frames", 'green'))

    def _save_all_episodes(self):
        """Save all complete episodes to file"""
        if not self.complete_episodes:
            print(colored("No complete episodes to save", 'yellow'))
            return

        save_path = self.save_motion_dir / \
            f"{self.motion_name}_{self.ckpt_num}_episodes.pkl"

        joblib.dump(self.complete_episodes, save_path)

        print(colored(
            f"Saved {len(self.complete_episodes)} complete episodes to {save_path}", 'green'))

        # Print information for each episode
        for episode_key, episode_data in self.complete_episodes.items():
            print(colored(
                f"  - {episode_key}: {len(episode_data['root_trans_offset'])} frames", 'cyan'))

    def _save_motion(self):
        # This method is now replaced by _save_all_episodes
        # Keep this method in case it's called elsewhere, but it does nothing now
        print(colored(
            "_save_motion is deprecated, using episode-based saving instead", 'yellow'))
        pass
