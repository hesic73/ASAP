from __future__ import annotations
from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributions import Normal

from .modules import BaseModule

from typing import Dict, Any, Callable


class SACTanhActor(nn.Module):
    def __init__(
        self,
        obs_dim_dict: Dict[str, int],
        module_config_dict: Dict[str, Any],
        num_actions: int,
        action_scale: torch.Tensor = None,
        action_bias: torch.Tensor = None,
    ):
        super().__init__()

        # Modify config to output 2 * num_actions (mean and log_std)
        module_config_dict = self._process_module_config(
            module_config_dict, num_actions * 2
        )

        self.actor_module = BaseModule(obs_dim_dict, module_config_dict)
        self.num_actions = num_actions

        # Log std bounds for stability (from CleanRL)
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5

        # Action scaling for bounded action spaces
        if action_scale is not None:
            self.register_buffer("action_scale", action_scale)
            self.register_buffer("action_bias", action_bias)
        else:
            # Default to [-1, 1] if not provided
            self.register_buffer("action_scale", torch.ones(num_actions))
            self.register_buffer("action_bias", torch.zeros(num_actions))

        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def _process_module_config(
        self, module_config_dict: Dict[str, Any], output_dim: int
    ) -> Dict[str, Any]:
        for idx, dim in enumerate(module_config_dict["output_dim"]):
            if dim == "robot_action_dim":
                module_config_dict["output_dim"][idx] = output_dim
        return module_config_dict

    def update_distribution(self, actor_obs: torch.Tensor):
        """Update internal distribution based on current observations."""
        output = self.actor_module(actor_obs)
        mean, log_std = output.chunk(2, dim=-1)

        # Clamp log_std for stability
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * \
            (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        std = log_std.exp()

        self.distribution = Normal(mean, std)

    def act(self, actor_obs: torch.Tensor, **kwargs):
        """Stochastic action sampling with tanh transformation."""
        self.update_distribution(actor_obs)
        # Reparameterization trick
        x_t = self.distribution.rsample()
        # Tanh transformation
        y_t = torch.tanh(x_t)
        # Scale to action space
        action = y_t * self.action_scale + self.action_bias
        return action

    def get_actions_log_prob(self, actions: torch.Tensor):
        """Get log probability of given actions."""
        # Convert actions back to pre-tanh space for log_prob calculation
        y_t = (actions - self.action_bias) / self.action_scale
        # Clamp to avoid numerical issues with atanh
        y_t = torch.clamp(y_t, -0.999, 0.999)
        x_t = torch.atanh(y_t)

        # Get log prob in pre-tanh space
        log_prob = self.distribution.log_prob(x_t)
        # Apply tanh correction
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        return log_prob.sum(dim=-1)

    def act_inference(self, actor_obs: torch.Tensor):
        """Deterministic action for inference."""
        output = self.actor_module(actor_obs)
        mean, _ = output.chunk(2, dim=-1)
        # Use mean action with tanh and scaling
        y_t = torch.tanh(mean)
        action = y_t * self.action_scale + self.action_bias
        return action

    def to_cpu(self):
        self.actor_module = deepcopy(self.actor_module).to("cpu")


class SACCritic(nn.Module):
    def __init__(
        self, obs_dim_dict: Dict[str, int], module_config_dict: Dict[str, Any], num_actions: int
    ):
        super().__init__()

        for idx, input_dim in enumerate(module_config_dict["input_dim"]):
            if input_dim == "robot_action_dim":
                module_config_dict["input_dim"][idx] = num_actions

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

    def __init__(self, critic_factory: Callable[[], nn.Module], device: torch.device, tau: float = 0.005):
        super().__init__()

        self.tau = tau

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

    def soft_update_targets(self):
        """Soft update of target networks."""
        for target_param, param in zip(
            self.target_critic_1.parameters(), self.critic_1.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(
            self.target_critic_2.parameters(), self.critic_2.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)
