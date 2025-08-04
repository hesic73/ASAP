from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from humanoidverse.agents.ppo.ppo import PPO
from typing import Dict, Any


class DebugPrintCallback(RL_EvalCallback):
    def __init__(self, config: Dict[str, Any], training_loop: PPO):
        super().__init__(config, training_loop)
        self.env: LeggedRobotBase = training_loop.env

    def on_pre_eval_env_step(self, actor_state: Dict[str, Any]) -> Dict[str, Any]:
        super().on_pre_eval_env_step(actor_state)
        return actor_state

    def on_post_eval_env_step(self, actor_state: Dict[str, Any]) -> Dict[str, Any]:
        super().on_post_eval_env_step(actor_state)
        feet_indices = self.env.feet_indices
        feet_pos = self.env.simulator._rigid_body_pos[:, feet_indices]
        feet_pos = feet_pos.squeeze(0).cpu().numpy()
        print(f"feet_pos: {feet_pos}")
        return actor_state

    def on_post_evaluate_policy(self):
        super().on_post_evaluate_policy()
