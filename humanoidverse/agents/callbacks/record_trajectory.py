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
        # NOTE (hsc): 因为PPO的evaluate_policy莫名在for loop外面额外调用了一次_pre_eval_env_step
        # 我试图理解的一下，也许它希望保持的语义是，obs_t, action_t
        # 但我其实只care obs_t, action_{t+1}
        if "step" in actor_state:
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

        assert len(trajectory_data["observations"]) == len(trajectory_data["actions"]), \
            "Observations and actions must have the same length."

        # Save the trajectory data to a file
        np.savez(
            self.config.trajectory_filename,
            observations=trajectory_data["observations"],
            actions=trajectory_data["actions"]
        )
        print("Trajectory data saved successfully.")
