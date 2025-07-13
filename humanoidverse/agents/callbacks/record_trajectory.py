import numpy as np
from isaacgym import gymapi
from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from humanoidverse.agents.ppo.ppo import PPO
from typing import List, Dict, Any, Optional


class RecordTrajectoryCallback(RL_EvalCallback):

    def __init__(self, config: Dict[str, Any], training_loop: PPO):
        super().__init__(config, training_loop)
        self.env: LeggedRobotBase = training_loop.env
        self.observations: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []

    def on_pre_eval_env_step(self, actor_state: Dict[str, Any]) -> Dict[str, Any]:
        super().on_pre_eval_env_step(actor_state)
        obs = actor_state["obs"]['actor_obs'].cpu().numpy()
        self.observations.append(obs)
        return actor_state

    def on_post_eval_env_step(self, actor_state: Dict[str, Any]) -> Dict[str, Any]:
        super().on_post_eval_env_step(actor_state)
        action = actor_state['actions'].cpu().numpy()
        self.actions.append(action)
        return actor_state

    def on_post_evaluate_policy(self):
        self._save_trajectory()

    def _save_trajectory(self):
        if not self.observations or not self.actions:
            print("No observations or actions recorded.")
            return

        trajectory_data = {
            "observations": np.array(self.observations),
            "actions": np.array(self.actions)
        }

        # Save the trajectory data to a file
        np.savez(
            self.config.trajectory_filename,
            observations=trajectory_data["observations"],
            actions=trajectory_data["actions"]
        )
        print("Trajectory data saved successfully.")
