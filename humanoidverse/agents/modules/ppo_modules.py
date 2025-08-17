from __future__ import annotations
from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributions import Normal
import math

from .modules import BaseModule

from typing import Dict, Any


class PPOActor(nn.Module):
    def __init__(self,
                 obs_dim_dict: Dict[str, int],
                 module_config_dict: Dict[str, Any],
                 num_actions: int,
                 init_noise_std: float,
                 fixed_std: float,
                 # as in torchrl
                 tanh_loc: bool = False,
                 up_scale: float = 5.0,
                 ):
        super(PPOActor, self).__init__()

        module_config_dict = self._process_module_config(
            module_config_dict, num_actions)

        self.actor_module = BaseModule(obs_dim_dict, module_config_dict)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None

        self.fixed_std = fixed_std
        if fixed_std:
            self.std.requires_grad = False

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

    def act(self, actor_obs, **kwargs):
        self.update_distribution(actor_obs)
        return self.distribution.sample()

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


class PPOLogStdActor(nn.Module):
    def __init__(self,
                 obs_dim_dict: Dict[str, int],
                 module_config_dict: Dict[str, Any],
                 num_actions: int,
                 init_noise_std: float,
                 # as in torchrl
                 tanh_loc: bool = False,
                 up_scale: float = 5.0,
                 ):
        super(PPOLogStdActor, self).__init__()

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

    def act(self, actor_obs, **kwargs):
        self.update_distribution(actor_obs)
        return self.distribution.sample()

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


class PPOActorTanh(nn.Module):
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
            module_config_dict, num_actions
        )

        self.actor_module = BaseModule(obs_dim_dict, module_config_dict)
        self.num_actions = num_actions
        # NOTE (hsc): 这个实现参考博添的OmniDrones
        self.log_std = nn.Parameter(torch.zeros(num_actions))

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

    def _process_module_config(self, module_config_dict, num_actions):
        for idx, output_dim in enumerate(module_config_dict['output_dim']):
            if output_dim == 'robot_action_dim':
                module_config_dict['output_dim'][idx] = num_actions
        return module_config_dict

    @property
    def actor(self):
        return self.actor_module

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def gaussian_mean(self):
        """Return the mean of the base Gaussian distribution."""
        if self.distribution is None:
            return None
        return self.distribution.mean

    @property
    def action_mean(self):
        """Return the mean of the transformed (tanh) distribution."""
        if self.distribution is None:
            return None
        # For tanh-transformed distribution, we approximate the mean
        # by transforming the Gaussian mean
        gaussian_mean = self.distribution.mean
        tanh_mean = torch.tanh(gaussian_mean)
        return tanh_mean * self.action_scale + self.action_bias

    @property
    def gaussian_std(self):
        """
        NOTE (hsc): Used when calculating the KL divergence between the old and new policy.
        KL divergence is invariant under parameter transformations. See https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Properties

        """
        return self.distribution.stddev

    def estimate_entropy(self):
        """Estimate the entropy of the transformed (tanh) distribution using sampling."""
        if self.distribution is None:
            return None

        # Sample-based entropy estimation: H(Y) = -E[log p(Y)]
        with torch.no_grad():
            # Sample from the base Gaussian distribution and transform
            gaussian_samples = self.distribution.sample(())
            tanh_samples = torch.tanh(gaussian_samples)
            scaled_samples = tanh_samples * self.action_scale + self.action_bias

            # Compute log probabilities of the transformed samples
            log_probs = self.get_actions_log_prob(scaled_samples)
            # Estimate entropy
            entropy = -log_probs.mean()
        return entropy

    def update_distribution(self, actor_obs: torch.Tensor):
        """Update internal distribution based on current observations."""
        output = self.actor_module(actor_obs)
        mean = output

        # NOTE (hsc): cleanrl的实现是这样的
        # log_std = torch.tanh(log_std)
        # log_std = self.LOG_STD_MIN + 0.5 * \
        #     (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)

        # stable baseline的实现是这样的
        log_std = torch.clamp(
            self.log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)

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

        # NOTE (hsc): 这里cleanrl和stable baseline的实现不一样。
        # cleanrl的实现是：
        # log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        # stable baseline的实现是：
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        return log_prob.sum(dim=-1)

    def act_inference(self, actor_obs: torch.Tensor):
        """Deterministic action for inference."""
        output = self.actor_module(actor_obs)
        mean = output
        # Use mean action with tanh and scaling
        y_t = torch.tanh(mean)
        action = y_t * self.action_scale + self.action_bias
        return action

    def to_cpu(self):
        self.actor_module = deepcopy(self.actor_module).to("cpu")


class PPOCritic(nn.Module):
    def __init__(self,
                 obs_dim_dict,
                 module_config_dict):
        super(PPOCritic, self).__init__()

        self.critic_module = BaseModule(obs_dim_dict, module_config_dict)

    @property
    def critic(self):
        return self.critic_module

    def reset(self, dones=None):
        pass

    def evaluate(self, critic_obs, **kwargs):
        value = self.critic(critic_obs)
        return value

# Deprecated: TODO: Let Wenli Fix this


class PPOActorFixSigma(PPOActor):
    def __init__(self,
                 obs_dim_dict,
                 network_dict,
                 network_load_dict,
                 num_actions,):
        super(PPOActorFixSigma, self).__init__(obs_dim_dict,
                                               network_dict, network_load_dict, num_actions, 0.0)

    def update_distribution(self, obs_dict):
        mean = self.actor(obs_dict)['head']
        self.distribution = mean

    @property
    def action_mean(self):
        return self.distribution

    def get_actions_log_prob(self, actions):
        raise NotImplementedError

    def act(self, obs_dict, **kwargs):
        self.update_distribution(obs_dict)
        return self.distribution
