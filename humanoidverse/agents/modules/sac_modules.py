from __future__ import annotations
from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.func import vmap, functional_call, stack_module_state

from .modules import BaseModule

from typing import Dict, Any, Callable


def encode_param_name(name: str) -> str:
    """Encode parameter name by replacing '.' with '__' for nn.ParameterDict compatibility."""
    return name.replace('.', '__')


def decode_param_name(name: str) -> str:
    """Decode parameter name by replacing '__' with '.' for functional_call compatibility."""
    return name.replace('__', '.')


class SACActor(nn.Module):
    def freeze_actor(self):
        # Freeze actor_module parameters
        for param in self.actor_module.parameters():
            param.requires_grad = False
        # Freeze std only if not fixed
        if not self.fixed_std:
            self.std.requires_grad = False

    def unfreeze_actor(self):
        # Unfreeze actor_module parameters
        for param in self.actor_module.parameters():
            param.requires_grad = True
        # Unfreeze std only if not fixed
        if not self.fixed_std:
            self.std.requires_grad = True

    def __init__(self,
                 obs_dim_dict: Dict[str, int],
                 module_config_dict: Dict[str, Any],
                 num_actions: int,
                 init_noise_std: float,
                 fixed_std: bool = False,
                 # as in torchrl
                 tanh_loc: bool = False,
                 up_scale: float = 5.0,
                 ):
        super(SACActor, self).__init__()

        module_config_dict = self._process_module_config(
            module_config_dict, num_actions)

        self.actor_module = BaseModule(obs_dim_dict, module_config_dict)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

        self.fixed_std = fixed_std
        if fixed_std:
            self.std.requires_grad = False

        self.distribution = None

        self.tanh_loc = tanh_loc
        self.up_scale = up_scale

        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def _process_module_config(self, module_config_dict, num_actions):
        for idx, output_dim in enumerate(module_config_dict['output_dim']):
            if output_dim == 'robot_action_dim':
                module_config_dict['output_dim'][idx] = num_actions
        return module_config_dict

    @property
    def actor(self):
        return self.actor_module

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

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
        if self.tanh_loc:
            mean = torch.tanh(mean/self.up_scale) * self.up_scale
        self.distribution = Normal(mean, mean*0. + self.std)

    # NOTE (hsc): SAC 是需要action differentiable的
    def act(self, actor_obs, **kwargs):
        self.update_distribution(actor_obs)
        return self.distribution.rsample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, actor_obs):
        actions_mean = self.actor(actor_obs)
        if self.tanh_loc:
            actions_mean = torch.tanh(
                actions_mean/self.up_scale) * self.up_scale
        return actions_mean

    def to_cpu(self):
        self.actor = deepcopy(self.actor).to('cpu')
        self.std.to('cpu')


class SACLogStdActor(nn.Module):
    def __init__(self,
                 obs_dim_dict: Dict[str, int],
                 module_config_dict: Dict[str, Any],
                 num_actions: int,
                 init_noise_std: float,
                 # as in torchrl
                 tanh_loc: bool = False,
                 up_scale: float = 5.0,
                 ):
        super(SACLogStdActor, self).__init__()

        module_config_dict = self._process_module_config(
            module_config_dict, num_actions)

        self.actor_module = BaseModule(obs_dim_dict, module_config_dict)

        # Action noise
        self.log_std = nn.Parameter(
            torch.log(torch.tensor(init_noise_std)) * torch.ones(num_actions))
        # Log std bounds for stability (from CleanRL)
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5
        self.distribution = None

        self.tanh_loc = tanh_loc
        self.up_scale = up_scale

        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def _process_module_config(self, module_config_dict, num_actions):
        for idx, output_dim in enumerate(module_config_dict['output_dim']):
            if output_dim == 'robot_action_dim':
                module_config_dict['output_dim'][idx] = num_actions
        return module_config_dict

    @property
    def actor(self):
        return self.actor_module

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

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
        if self.tanh_loc:
            mean = torch.tanh(mean/self.up_scale) * self.up_scale

        std = torch.clamp(
            self.log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX).exp()
        self.distribution = Normal(mean, std)

    # NOTE (hsc): SAC 的 action 是 reparameterized 的
    def act(self, actor_obs, **kwargs):
        self.update_distribution(actor_obs)
        return self.distribution.rsample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, actor_obs):
        actions_mean = self.actor(actor_obs)
        if self.tanh_loc:
            actions_mean = torch.tanh(
                actions_mean/self.up_scale) * self.up_scale
        return actions_mean

    def to_cpu(self):
        self.actor = deepcopy(self.actor).to('cpu')
        self.log_std.to('cpu')


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

    def evaluate(self, critic_obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # Concatenate observations and actions
        critic_input = torch.cat([critic_obs, actions], dim=-1)
        value = self.critic(critic_input)
        return value

    def forward(self, critic_obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.evaluate(critic_obs, actions)


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


class REDQEnsembleCritic(torch.nn.Module):
    """Randomized Ensembles with Double Q-learning (REDQ) critic ensemble using functional approach."""

    def __init__(self, critic_factory: Callable[[], nn.Module], device: torch.device,
                 num_critics: int = 10, num_sample_critics: int = 2, tau: float = 0.005):
        super().__init__()

        self.num_critics = num_critics
        self.num_sample_critics = num_sample_critics
        self.tau = tau
        self.device = device

        # Create ensemble of critics for stacking
        critics = [critic_factory() for _ in range(num_critics)]

        # Stack module states for functional approach
        self.critic_params_raw, self.critic_buffers = stack_module_state(
            critics)

        # Register parameters with encoded names (replace '.' with '__')
        encoded_params = {
            encode_param_name(name): nn.Parameter(tensor.to(device))
            for name, tensor in self.critic_params_raw.items()
        }
        self.critic_param_dict = nn.ParameterDict(encoded_params)

        # Move buffers to device
        self.critic_buffers_dict = {
            name: tensor.to(device) for name, tensor in self.critic_buffers.items()
        }

        # Create target parameters as copies
        target_encoded_params = {
            f"target_{name}": nn.Parameter(tensor.clone().detach())
            for name, tensor in encoded_params.items()
        }
        self.target_critic_param_dict = nn.ParameterDict(target_encoded_params)

        # Freeze target parameters
        for param in self.target_critic_param_dict.parameters():
            param.requires_grad = False

        # Create target buffers as copies
        self.target_critic_buffers_dict = {
            name: tensor.clone().detach() for name, tensor in self.critic_buffers_dict.items()
        }

        # Store reference critic for functional calls
        self._reference_critic = critics[0]

        # Define vectorized critic function
        def critic_wrapper(params, buffers, obs, actions):
            # Decode parameter names back to original format
            decoded_params = {
                decode_param_name(k): v for k, v in params.items()
            }
            return functional_call(self._reference_critic, (decoded_params, buffers), (obs, actions))

        self._critic_func = vmap(critic_wrapper, in_dims=(0, 0, None, None))

    def forward(self, obs: torch.Tensor, actions: torch.Tensor):
        """Forward pass through all critics."""
        params = {k: v for k, v in self.critic_param_dict.items()}
        buffers = self.critic_buffers_dict

        # Use vectorized function to compute all critic outputs
        batched_out = self._critic_func(params, buffers, obs, actions)
        return batched_out  # Shape: (num_critics, batch_size, 1)

    def target_forward(self, obs: torch.Tensor, actions: torch.Tensor):
        """Forward pass through target critics."""
        with torch.no_grad():
            # Use target parameters (remove 'target_' prefix for decoding)
            target_params = {
                k.replace('target_', ''): v
                for k, v in self.target_critic_param_dict.items()
            }
            target_buffers = self.target_critic_buffers_dict

            # Use vectorized function to compute all target critic outputs
            batched_out = self._critic_func(
                target_params, target_buffers, obs, actions)
            return batched_out  # Shape: (num_critics, batch_size, 1)

    def get_min_target_q(self, obs: torch.Tensor, actions: torch.Tensor):
        """Get minimum Q-value from randomly sampled subset of target critics."""
        with torch.no_grad():
            # Get all target Q values
            # Shape: (num_critics, batch_size, 1)
            all_target_q = self.target_forward(obs, actions)

            # Randomly sample critics for minimum computation
            critic_indices = torch.randperm(self.num_critics, device=self.device)[
                :self.num_sample_critics]

            # Select sampled critics' Q values
            # Shape: (num_sample_critics, batch_size, 1)
            sampled_q_values = all_target_q[critic_indices]

            # Get minimum across sampled critics
            min_q = torch.min(sampled_q_values, dim=0)[
                0]  # Shape: (batch_size, 1)

        return min_q

    def soft_update_targets(self):
        """Soft update of all target networks."""
        with torch.no_grad():
            for name, param in self.critic_param_dict.items():
                target_name = f"target_{name}"
                target_param = self.target_critic_param_dict[target_name]
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
