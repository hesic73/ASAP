from __future__ import annotations
from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributions import Normal

from .modules import BaseModule

from typing import Dict, Any, Callable


class SACActor(nn.Module):
    def __init__(
        self,
        obs_dim_dict: Dict[str, int],
        module_config_dict: Dict[str, Any],
        num_actions: int,
        init_noise_std: float,
    ):
        super().__init__()

        module_config_dict = self._process_module_config(
            module_config_dict, num_actions
        )

        self.actor_module = BaseModule(obs_dim_dict, module_config_dict)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def _process_module_config(
        self, module_config_dict: Dict[str, Any], num_actions: int
    ) -> Dict[str, Any]:
        for idx, output_dim in enumerate(module_config_dict["output_dim"]):
            if output_dim == "robot_action_dim":
                module_config_dict["output_dim"][idx] = num_actions
        return module_config_dict

    @property
    def actor(self):
        return self.actor_module

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, actor_obs):
        mean = self.actor(actor_obs)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, actor_obs, **kwargs):
        self.update_distribution(actor_obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, actor_obs):
        actions_mean = self.actor(actor_obs)
        return actions_mean

    def to_cpu(self):
        self.actor = deepcopy(self.actor).to("cpu")
        self.std.to("cpu")


class SACCritic(nn.Module):
    def __init__(
        self, obs_dim_dict: Dict[str, int], module_config_dict: Dict[str, Any]
    ):
        super().__init__()
        self.critic_module = BaseModule(obs_dim_dict, module_config_dict)

    @property
    def critic(self):
        return self.critic_module

    def evaluate(self, critic_obs, actions, **kwargs):
        # Concatenate observations and actions
        critic_input = torch.cat([critic_obs, actions], dim=-1)
        value = self.critic(critic_input)
        return value

class DoubleQCritic(torch.nn.Module):
    """Double Q-network for SAC algorithm."""

    def __init__(self, critic_factory: Callable[[], nn.Module], device: torch.device):
        super().__init__()

        # Create critics using the factory function
        self.critic_1 = critic_factory().to(device)
        self.critic_2 = critic_factory().to(device)

        # Create target critics
        self.target_critic_1 = deepcopy(self.critic_1).to(device)
        self.target_critic_2 = deepcopy(self.critic_2).to(device)

        # Freeze target networks
        for param in self.target_critic_1.parameters():
            param.requires_grad = False
        for param in self.target_critic_2.parameters():
            param.requires_grad = False

    def forward(self, obs: torch.Tensor, actions: torch.Tensor):
        """Forward pass through both critics."""
        q1 = self.critic_1.evaluate(obs, actions)
        q2 = self.critic_2.evaluate(obs, actions)
        return q1, q2

    def target_forward(self, obs: torch.Tensor, actions: torch.Tensor):
        """Forward pass through target critics."""
        with torch.no_grad():
            target_q1 = self.target_critic_1.evaluate(obs, actions)
            target_q2 = self.target_critic_2.evaluate(obs, actions)
        return target_q1, target_q2

    def soft_update_targets(self, tau: float):
        """Soft update of target networks."""
        for target_param, param in zip(
            self.target_critic_1.parameters(), self.critic_1.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(
            self.target_critic_2.parameters(), self.critic_2.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
