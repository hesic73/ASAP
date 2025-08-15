import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import time
import statistics

from collections import deque
from loguru import logger
from rich.progress import Progress
from rich.console import Console
from rich.panel import Panel
from rich.live import Live

from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from humanoidverse.agents.base_algo.base_algo import BaseAlgo
from humanoidverse.envs.base_task.base_task import BaseTask
from humanoidverse.agents.modules.sac_modules import SACActor, SACCritic, REDQEnsembleCritic
from humanoidverse.agents.modules.data_utils import ReplayBuffer
from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.utils.average_meters import TensorAverageMeterDict

from hydra.utils import instantiate

from typing import Optional, Dict, List
from itertools import count

from tqdm import trange

console = Console()


def _dict_to_device(data_dict: Dict[str, torch.Tensor], device: torch.device):
    for key in data_dict.keys():
        data_dict[key] = data_dict[key].to(device)
    return data_dict


class SAC(BaseAlgo):

    def __init__(self, env: BaseTask, config: DictConfig, log_dir: str, device: torch.device):
        super().__init__(env, config, device)

        self.log_dir = log_dir
        self.writer = TensorboardSummaryWriter(
            log_dir=self.log_dir, flush_secs=10)

        # Environment config
        self.num_envs = self.env.config.num_envs
        self.algo_obs_dim_dict = self.env.config.robot.algo_obs_dim_dict
        self.num_actions = self.env.config.robot.actions_dim

        self.save_interval = self.config.save_interval

        # SAC specific config - all must be explicitly provided
        self.replay_buffer_size = self.config.replay_buffer_size
        self.batch_size = self.config.batch_size
        self.learning_starts = self.config.learning_starts
        self.target_update_frequency = self.config.target_update_frequency
        self.tau = self.config.tau
        self.gamma = self.config.gamma

        # Ensemble critic config (REDQ-style, degrades to SAC when num_critics=num_sample_critics=2)
        self.num_critics = self.config.num_critics
        self.num_sample_critics = self.config.num_sample_critics

        # Training parameters
        self.samples_per_iter = self.config.samples_per_iter
        self.policy_frequency = self.config.policy_frequency
        self.gradient_steps = self.config.gradient_steps
        self.actor_max_grad_norm = self.config.actor_max_grad_norm
        self.critic_max_grad_norm = self.config.critic_max_grad_norm

        self.replay_buffer_on_device = self.config.replay_buffer_on_device

        # Learning rates
        self.actor_learning_rate = self.config.actor_learning_rate
        self.critic_learning_rate = self.config.critic_learning_rate
        self.alpha_learning_rate = self.config.alpha_learning_rate

        # Training config
        self.num_learning_iterations = self.config.num_learning_iterations

        # Training counters
        self.total_steps = 0
        self.updates = 0
        self.current_learning_iteration = 0

        # Timing variables
        self.start_time = 0
        self.stop_time = 0
        self.collection_time = 0
        self.learn_time = 0
        self.tot_timesteps = 0
        self.tot_time = 0

        # Episode tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.current_episode_reward = torch.zeros(
            self.num_envs, device=self.device)
        self.current_episode_length = torch.zeros(
            self.num_envs, device=self.device)

        # Episode info tracking
        self.ep_infos = []
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.episode_env_tensors = TensorAverageMeterDict()

        # Actor freezing flag
        self.actor_frozen = False

    def setup(self):
        self.actor = SACActor(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config_dict=self.config.module_dict.actor,
            num_actions=self.num_actions,
            init_noise_std=self.config.init_noise_std,
            fixed_std=self.config.fixed_std,
            tanh_loc=self.config.tanh_loc,
            up_scale=self.config.up_scale,
        ).to(self.device)

        def critic_factory():
            return SACCritic(self.algo_obs_dim_dict, self.config.module_dict.critic, self.num_actions)

        # Entropy temperature setup
        self.autotune_alpha = self.config.autotune_alpha

        if self.autotune_alpha:
            raw_target_entropy = self.config.target_entropy
            if isinstance(raw_target_entropy, str):
                if raw_target_entropy.lower() == 'auto':
                    self.target_entropy = -float(self.num_actions)
                else:
                    try:
                        self.target_entropy = float(raw_target_entropy)
                    except Exception as exc:
                        raise ValueError(
                            f"Invalid target_entropy string: {raw_target_entropy}"
                        ) from exc
            elif isinstance(raw_target_entropy, (int, float)):
                self.target_entropy = float(raw_target_entropy)
            else:
                raise TypeError(
                    f"target_entropy must be 'auto' or a number, got {type(raw_target_entropy)}"
                )

            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device)
            logger.info(
                f"Using automatic alpha tuning with target entropy: {self.target_entropy}")
        else:
            self.alpha_value = torch.tensor(
                self.config.alpha, device=self.device)
            logger.info(f"Using fixed alpha value: {self.alpha_value}")

        # Initialize critic ensemble (degrades to SAC when num_critics=num_sample_critics=2)
        self.critic = REDQEnsembleCritic(
            critic_factory=critic_factory,
            device=self.device,
            num_critics=self.num_critics,
            num_sample_critics=self.num_sample_critics,
            tau=self.tau,
        )

        # Initialize target actor network
        self.target_actor = SACActor(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config_dict=self.config.module_dict.actor,
            num_actions=self.num_actions,
            init_noise_std=self.config.init_noise_std,
            fixed_std=self.config.fixed_std,
            tanh_loc=self.config.tanh_loc,
            up_scale=self.config.up_scale,
        ).to(self.device)

        # Hard copy parameters from main actor to target actor
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.actor_learning_rate
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.critic_learning_rate
        )

        if self.autotune_alpha:
            self.alpha_optimizer = optim.Adam(
                [self.log_alpha], lr=self.alpha_learning_rate)

        rb_device = self.device if self.replay_buffer_on_device else torch.device(
            "cpu")
        self.replay_buffer = ReplayBuffer(
            buffer_size=int(self.replay_buffer_size),
            num_envs=self.num_envs,
            device=rb_device,
        )
        for obs_key, obs_shape in self.algo_obs_dim_dict.items():
            self.replay_buffer.register_key(obs_key, obs_shape, is_obs=True)

        self.replay_buffer.register_key("actions", (self.num_actions,))
        self.replay_buffer.register_key("rewards", (1,))
        self.replay_buffer.register_key("dones", (1,), dtype=torch.bool)

        logger.info(f"Replay buffer:\n{self.replay_buffer}")

        # Pre-compute action bounds for action bound loss
        qpos_limits, _, _ = self.env.simulator.get_dof_limits_properties()
        qpos_limits = qpos_limits.to(self.device)  # (num_dof, 2)
        default_dof_pos = self.env.default_dof_pos.to(
            self.device).squeeze(0)  # (num_dof,)

        # Convert to action space bounds
        _action_scale = self.env.action_scale
        raw_action_limits = (
            qpos_limits - default_dof_pos.unsqueeze(-1)) / _action_scale

        # Store action bounds
        self.a_bound_min = raw_action_limits[:, 0]  # (num_dof,)
        self.a_bound_max = raw_action_limits[:, 1]  # (num_dof,)

        # Ensure bounds are finite
        assert torch.all(torch.isfinite(self.a_bound_min)) and torch.all(
            torch.isfinite(self.a_bound_max)), "Actions must be bounded."

        # Action bound loss weight
        self.action_bound_loss_weight = self.config.action_bound_loss_weight
        logger.info(
            f"Action bound loss weight: {self.action_bound_loss_weight}")

    def _get_action_scaling(self):
        qpos_limits, _, _ = self.env.simulator.get_dof_limits_properties()
        qpos_limits = qpos_limits.cpu()  # (num_dof, 2)
        default_dof_pos = self.env.default_dof_pos.cpu().squeeze(0)  # (num_dof,)

        # NOTE (hsc): _compute_torques里，target qpos的计算方式是
        # actions * scale + default_dof_pos
        # 所以raw_action_limits需要减去default_dof_pos，再除以action_scale
        _action_scale = self.env.action_scale
        raw_action_limits = (
            qpos_limits - default_dof_pos.unsqueeze(-1)) / _action_scale

        # action_scale = (
        #     raw_action_limits[:, 1] - raw_action_limits[:, 0]) / 2.0
        # action_bias = (raw_action_limits[:, 1] + raw_action_limits[:, 0]) / 2.0

        # NOTE (hsc): 我非常怀疑这里有问题。因为语义上，action=0对应的是default_dof_pos。而dof limit是不对称的。
        # 所以action_bias会导致它的分布发生偏移。
        action_scale = torch.maximum(
            torch.abs(raw_action_limits[:, 1]), torch.abs(raw_action_limits[:, 0]))
        action_bias = torch.zeros(self.num_actions, device='cpu')
        return action_scale, action_bias

    def _action_bound_loss(self, action_mean):
        """Compute action bound loss to prevent actions from going out of bounds."""
        # Use pre-computed action bounds
        a_bound_min = self.a_bound_min
        a_bound_max = self.a_bound_max

        # Compute violations
        violation_min = torch.minimum(
            action_mean - a_bound_min, torch.zeros_like(action_mean))
        violation_max = torch.maximum(
            action_mean - a_bound_max, torch.zeros_like(action_mean))

        # Sum squared violations
        violation = torch.sum(torch.square(violation_min), dim=-1) + \
            torch.sum(torch.square(violation_max), dim=-1)

        # Return mean violation loss (without weight)
        a_bound_loss = 0.5 * torch.mean(violation)

        return a_bound_loss

    def _soft_update_target_actor(self):
        """Soft update target actor network with moving average."""
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                (1.0 - self.tau) * target_param.data + self.tau * param.data
            )

    def learn(self):
        obs_dict = self.env.reset_all()
        obs_dict = _dict_to_device(obs_dict, self.device)
        self._train_mode()

        # Collect initial samples
        logger.info(f"Collecting {self.learning_starts} initial samples")
        self._collect_initial_samples(obs_dict, self.learning_starts)

        for iteration in range(
            self.current_learning_iteration,
            self.current_learning_iteration + self.num_learning_iterations,
        ):
            self.start_time = time.time()

            # Collect samples_per_iter samples with online training
            obs_dict, loss_dict, info_dict = self._collect_and_train_online(
                obs_dict, self.samples_per_iter)

            # Logging
            log_dict = {
                'it': iteration,
                'loss_dict': loss_dict,
                'alpha_value': self.alpha.item() if isinstance(
                    self.alpha, torch.Tensor) else self.alpha,
                'collection_time': info_dict['collection_time'],
                'learn_time': info_dict['learn_time'],
                'training_steps': info_dict['training_steps'],
                'buffer_size': info_dict['buffer_size'],
                'metrics_dict': info_dict['metrics_dict'],
                'ep_infos': self.ep_infos,
                'rewbuffer': self.rewbuffer,
                'lenbuffer': self.lenbuffer,
                'num_learning_iterations': self.num_learning_iterations
            }
            self._post_epoch_logging(log_dict)

            if iteration % self.save_interval == 0:
                self.save(os.path.join(self.log_dir,
                          'model_{}.pt'.format(iteration)))

            self.ep_infos.clear()

        self.current_learning_iteration += self.num_learning_iterations

    def rollout(self, num_iterations: int):
        """
        Perform rollout to collect episodes data without training.

        Args:
            num_iterations: Number of iterations to collect, each iteration collects samples_per_iter steps
        """

        self.env.set_is_evaluating()

        obs_dict = self.env.reset_all()
        obs_dict = _dict_to_device(obs_dict, self.device)
        self._eval_mode()  # Set to evaluation mode for rollout

        # Clear previous episode data
        self.ep_infos.clear()
        self.rewbuffer.clear()
        self.lenbuffer.clear()
        self.episode_rewards.clear()
        self.episode_lengths.clear()

        # Reset episode tracking
        self.current_episode_reward = torch.zeros(
            self.num_envs, device=self.device)
        self.current_episode_length = torch.zeros(
            self.num_envs, device=self.device)

        logger.info(f"Starting rollout for {num_iterations} iterations")

        for iteration in range(num_iterations):
            # Collect samples_per_iter steps (like _collect_and_train_online but no training)
            sample_count = 0
            while sample_count < self.samples_per_iter:
                obs_dict = self._collect_experience(obs_dict)
                sample_count += self.num_envs

            # Log after each iteration
            self._post_rollout_logging(iteration + 1)

    def _train_mode(self):
        """Set networks to training mode."""
        self.actor.train()
        self.critic.train()
        # Target actor should always be in eval mode
        self.target_actor.eval()

    def _eval_mode(self):
        """Set networks to evaluation mode."""
        self.actor.eval()
        self.critic.eval()
        self.target_actor.eval()

    def freeze_actor(self):
        """Freeze actor parameters to prevent updates during training."""
        logger.info(
            "Freezing actor parameters - only critic will be updated during training")
        self.actor_frozen = True
        # Freeze all actor parameters
        for param in self.actor.parameters():
            param.requires_grad = False

    def unfreeze_actor(self):
        """Unfreeze actor parameters to resume normal training."""
        logger.info(
            "Unfreezing actor parameters - resuming normal actor-critic training")
        self.actor_frozen = False
        # Unfreeze all actor parameters
        for param in self.actor.parameters():
            param.requires_grad = True

    def _collect_experience(self, obs_dict: Dict[str, torch.Tensor]):
        """Collect one step of experience."""
        with torch.no_grad():
            # Select action
            actions = self.actor.act(obs_dict["actor_obs"])

            # Step environment
            actor_state = {"actions": actions}
            next_obs_dict, rewards, dones, infos = self.env.step(actor_state)

            next_obs_dict = _dict_to_device(next_obs_dict, self.device)
            rewards = rewards.to(self.device)
            dones = dones.to(self.device)

            # Store transition in replay buffer
            transition_data = {}
            for obs_key in obs_dict.keys():
                transition_data[obs_key] = obs_dict[obs_key].to(
                    self.replay_buffer.device)
            transition_data["actions"] = actions.to(self.replay_buffer.device)
            transition_data["rewards"] = rewards.to(
                self.replay_buffer.device).unsqueeze(1)
            transition_data["dones"] = dones.to(
                self.replay_buffer.device).unsqueeze(1)

            self.replay_buffer.add(
                transition_data, next_obs_dict=next_obs_dict)

            # Update episode tracking
            self.current_episode_reward += rewards
            self.current_episode_length += 1

            # Track environment tensors
            if "to_log" in infos:
                self.episode_env_tensors.add(infos["to_log"])

            # Handle episode termination
            done_envs = dones.nonzero(as_tuple=False).squeeze(1)
            if len(done_envs) > 0:
                # Track episode info
                if 'episode' in infos:
                    self.ep_infos.append(infos['episode'])

                for env_idx in done_envs:
                    episode_reward = self.current_episode_reward[env_idx].item(
                    )
                    episode_length = self.current_episode_length[env_idx].item(
                    )

                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    self.rewbuffer.append(episode_reward)
                    self.lenbuffer.append(episode_length)

                    self.current_episode_reward[env_idx] = 0
                    self.current_episode_length[env_idx] = 0

            # Update total steps
            self.total_steps += self.num_envs

        return next_obs_dict

    def _collect_initial_samples(self, obs_dict: Dict[str, torch.Tensor], num_samples: int):
        """Collect initial samples to populate replay buffer."""
        sample_count = 0
        with Progress() as progress:
            task = progress.add_task(
                "Collecting initial samples", total=num_samples)
            while sample_count < num_samples:
                obs_dict = self._collect_experience(obs_dict)
                sample_count += self.num_envs
                progress.update(task, advance=self.num_envs)

    def _collect_and_train_online(self, obs_dict: Dict[str, torch.Tensor], num_samples: int):
        """Collect samples and train online (like CleanRL)."""
        sample_count = 0
        loss_dict = {'Critic': [], 'Actor': [], 'Alpha': [],
                     'Actor_Entropy_Term': [], 'Actor_Q_Term': [], 'Action_Bound': []}
        metrics_dict = {}

        collection_time = 0.0
        training_time = 0.0

        collect_start = time.time()
        while sample_count < num_samples:
            obs_dict = self._collect_experience(obs_dict)
            sample_count += self.num_envs

        collection_time += time.time() - collect_start

        # Train (we already collected initial samples in setup)
        train_start = time.time()

        # Perform gradient_steps updates per env step
        for _ in range(self.gradient_steps):
            # Update critics every update
            critic_result = self._update_critics_step()
            loss_dict['Critic'].append(critic_result['critic_loss'])

            # Add critic metrics to metrics_dict
            if 'Critic_Grad_Norm' not in metrics_dict:
                metrics_dict['Critic_Grad_Norm'] = 0
            if 'Mean_Min_Q' not in metrics_dict:
                metrics_dict['Mean_Min_Q'] = 0
            if 'Mean_Target_Q' not in metrics_dict:
                metrics_dict['Mean_Target_Q'] = 0
            if 'Mean_Entropy_Contribution' not in metrics_dict:
                metrics_dict['Mean_Entropy_Contribution'] = 0
            if 'Mean_Log_Prob' not in metrics_dict:
                metrics_dict['Mean_Log_Prob'] = 0

            metrics_dict['Critic_Grad_Norm'] += critic_result['critic_grad_norm']
            metrics_dict['Mean_Min_Q'] += critic_result['mean_min_q']
            metrics_dict['Mean_Target_Q'] += critic_result['mean_target_q']
            metrics_dict['Mean_Entropy_Contribution'] += critic_result['mean_entropy_contribution']
            metrics_dict['Mean_Log_Prob'] += critic_result['mean_log_prob']

            # Update actor and alpha with policy frequency (delayed updates)
            # Skip actor updates if actor is frozen
            if not self.actor_frozen and self.updates % self.policy_frequency == 0:
                # Compensate for delay by doing policy_frequency updates
                for _ in range(self.policy_frequency):
                    actor_result = self._update_actor_and_alpha_step()
                    loss_dict['Actor'].append(actor_result['actor_loss'])
                    loss_dict['Alpha'].append(actor_result['alpha_loss'])
                    loss_dict['Action_Bound'].append(
                        actor_result['action_bound_loss'])

                    # Log actor loss components
                    if 'Actor_Entropy_Term' not in loss_dict:
                        loss_dict['Actor_Entropy_Term'] = []
                    if 'Actor_Q_Term' not in loss_dict:
                        loss_dict['Actor_Q_Term'] = []
                    loss_dict['Actor_Entropy_Term'].append(
                        actor_result['entropy_term'])
                    loss_dict['Actor_Q_Term'].append(
                        actor_result['q_term'])

                    # Add actor grad norm to metrics
                    if 'Actor_Grad_Norm' not in metrics_dict:
                        metrics_dict['Actor_Grad_Norm'] = 0
                    metrics_dict['Actor_Grad_Norm'] += actor_result['actor_grad_norm']

                    # Accumulate metrics
                    for key, value in actor_result['metrics_dict'].items():
                        if key not in metrics_dict:
                            metrics_dict[key] = 0
                        metrics_dict[key] += value

            # Update target networks
            if self.updates % self.target_update_frequency == 0:
                self.critic.soft_update_targets()
                self._soft_update_target_actor()

            self.updates += 1

        training_time += time.time() - train_start

        # Average the losses
        averaged_loss_dict = {}
        for key, values in loss_dict.items():
            if values:
                averaged_loss_dict[key] = sum(values) / len(values)
            else:
                averaged_loss_dict[key] = 0.0

        # Average the metrics
        averaged_metrics_dict = {}
        # Use actor updates as reference
        num_updates = max(1, len(loss_dict['Actor']))
        for key, value in metrics_dict.items():
            averaged_metrics_dict[key] = value / num_updates

        info_dict = {
            'training_steps': (num_samples // self.num_envs) * max(1, self.gradient_steps),
            'buffer_size': self.replay_buffer.size(),
            'collection_time': collection_time,
            'learn_time': training_time,
            'metrics_dict': averaged_metrics_dict,
        }

        return obs_dict, averaged_loss_dict, info_dict

    def _update_critics_step(self):
        """Single critic update step using ensemble (degrades to SAC when num_critics=num_sample_critics=2)."""
        # Sample batch from replay buffer
        current_samples, next_obs_samples = self.replay_buffer.sample(
            self.batch_size)

        # Move to device
        current_samples = _dict_to_device(current_samples, self.device)
        next_obs_samples = _dict_to_device(next_obs_samples, self.device)

        # Extract data
        obs = {k: v for k, v in current_samples.items()
               if k in self.algo_obs_dim_dict}
        actions = current_samples["actions"]
        rewards = current_samples["rewards"]
        dones = current_samples["dones"]
        next_obs = next_obs_samples

        with torch.no_grad():
            # Sample next actions from TARGET ACTOR (key change!)
            next_actions = self.target_actor.act(next_obs["actor_obs"])
            next_log_probs = self.target_actor.get_actions_log_prob(
                next_actions)

            # Use minimum Q from random subset of target critics
            target_q_min = self.critic.get_min_target_q(
                next_obs["critic_obs"], next_actions
            )
            target_q = target_q_min - self.alpha * next_log_probs.unsqueeze(1)

            # Compute TD target
            target_values = rewards + self.gamma * \
                (1 - dones.float()) * target_q

        # Update all critics in ensemble
        # Shape: (num_critics, batch_size, 1)
        current_q_values = self.critic(obs["critic_obs"], actions)

        # Compute losses for all critics
        critic_losses = []
        for i in range(self.critic.num_critics):
            critic_loss = F.mse_loss(current_q_values[i], target_values)
            critic_losses.append(critic_loss)

        # Sum all critic losses
        total_critic_loss = sum(critic_losses)

        # Update critics
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.critic_max_grad_norm)
        self.critic_optimizer.step()

        # Log Q values for monitoring
        with torch.no_grad():
            mean_min_q = torch.min(current_q_values, dim=0)[0].mean().item()
            mean_target_q = target_values.mean().item()

            # Log entropy contribution to target Q
            entropy_contribution = -self.alpha * next_log_probs.unsqueeze(1)
            mean_entropy_contribution = entropy_contribution.mean().item()
            mean_log_prob = next_log_probs.mean().item()

        # Create return dict
        result_dict = {
            'critic_loss': total_critic_loss.item(),
            'critic_grad_norm': critic_grad_norm.item(),
            'mean_min_q': mean_min_q,
            'mean_target_q': mean_target_q,
            'mean_entropy_contribution': mean_entropy_contribution,
            'mean_log_prob': mean_log_prob,
        }

        return result_dict

    def _update_actor_and_alpha_step(self):
        """Single actor and alpha update step using ensemble critic."""
        # Sample batch from replay buffer
        current_samples, _ = self.replay_buffer.sample(self.batch_size)

        # Move to device
        current_samples = _dict_to_device(current_samples, self.device)

        # Extract observations
        obs = {k: v for k, v in current_samples.items()
               if k in self.algo_obs_dim_dict}

        # Sample actions from current policy
        actions = self.actor.act(obs["actor_obs"])
        log_probs = self.actor.get_actions_log_prob(actions)

        # Get Q values from all critics and take minimum
        # Shape: (num_critics, batch_size, 1)
        q_values = self.critic(obs["critic_obs"], actions)
        q_min = torch.min(q_values, dim=0)[0]  # Shape: (batch_size, 1)

        # Actor loss components
        entropy_term = self.alpha * log_probs.unsqueeze(1)
        q_term = q_min
        actor_loss = (entropy_term - q_term).mean()

        # Add action bound loss
        action_bound_loss = self._action_bound_loss(self.actor.action_mean)
        actor_loss += action_bound_loss * self.action_bound_loss_weight

        # Log actor loss components for monitoring
        with torch.no_grad():
            mean_entropy_term = entropy_term.mean().item()
            mean_q_term = q_term.mean().item()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.actor_max_grad_norm)
        self.actor_optimizer.step()

        # Create metrics dict for non-loss metrics
        metrics_dict = {}

        # Add action mean and std metrics
        action_mean_avg = torch.mean(self.actor.action_mean)
        action_std_avg = torch.mean(self.actor.action_std)
        entropy_avg = torch.mean(self.actor.entropy)

        metrics_dict['Action_Mean'] = action_mean_avg.item()
        metrics_dict['Action_Std'] = action_std_avg.item()
        metrics_dict['Entropy'] = entropy_avg.item()

        # Update alpha (temperature parameter) if autotuning
        if self.autotune_alpha:
            alpha_loss = -(
                self.log_alpha * (log_probs + self.target_entropy).detach()
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            result_dict = {
                'actor_loss': actor_loss.item(),
                'alpha_loss': alpha_loss.item(),
                'action_bound_loss': action_bound_loss.item(),
                'actor_grad_norm': actor_grad_norm.item(),
                'entropy_term': mean_entropy_term,
                'q_term': mean_q_term,
                'metrics_dict': metrics_dict,
            }
        else:
            # No alpha loss when using fixed alpha
            result_dict = {
                'actor_loss': actor_loss.item(),
                'alpha_loss': 0.0,
                'action_bound_loss': action_bound_loss.item(),
                'actor_grad_norm': actor_grad_norm.item(),
                'entropy_term': mean_entropy_term,
                'q_term': mean_q_term,
                'metrics_dict': metrics_dict,
            }

        return result_dict

    @property
    def alpha(self):
        """Current value of temperature parameter."""
        if self.autotune_alpha:
            return self.log_alpha.exp()
        else:
            return self.alpha_value

    def save(self, path: str):
        """Save model checkpoint."""
        logger.info(f"Saving checkpoint to {path}")
        save_dict = {
            "actor_model_state_dict": self.actor.state_dict(),
            "target_actor_state_dict": self.target_actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "total_steps": self.total_steps,
            "updates": self.updates,
            "autotune_alpha": self.autotune_alpha,
            "num_critics": self.num_critics,
            "num_sample_critics": self.num_sample_critics,
        }

        if self.autotune_alpha:
            save_dict["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()
            save_dict["log_alpha"] = self.log_alpha
        else:
            save_dict["alpha_value"] = self.alpha_value

        torch.save(save_dict, path)

    def load(self, ckpt_path: str):
        logger.info(f"Loading checkpoint from {ckpt_path}")
        loaded_dict = torch.load(
            ckpt_path, weights_only=True, map_location=self.device)

        self.actor.load_state_dict(loaded_dict["actor_model_state_dict"])
        # Load target actor if it exists in checkpoint, otherwise copy from main actor
        if "target_actor_state_dict" in loaded_dict:
            self.target_actor.load_state_dict(
                loaded_dict["target_actor_state_dict"])
        else:
            # For backward compatibility with old checkpoints
            logger.warning(
                "Target actor state not found in checkpoint, copying from main actor")
            self.target_actor.load_state_dict(self.actor.state_dict())
        self.critic.load_state_dict(loaded_dict["critic_state_dict"])

        self.actor_optimizer.load_state_dict(
            loaded_dict["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(
            loaded_dict["critic_optimizer_state_dict"]
        )

        # Load alpha settings
        if loaded_dict["autotune_alpha"]:
            with torch.no_grad():
                self.log_alpha.data.copy_(
                    loaded_dict["log_alpha"].to(self.device))
            self.alpha_optimizer.load_state_dict(
                loaded_dict["alpha_optimizer_state_dict"])
        else:
            self.alpha_value = torch.as_tensor(
                loaded_dict["alpha_value"], device=self.device)

        self.current_learning_iteration = loaded_dict["iter"]
        self.total_steps = loaded_dict["total_steps"]
        self.updates = loaded_dict["updates"]

        logger.info("Checkpoint loaded successfully")

    def load_actor_only(self, ckpt_path: str):
        logger.info(f"Loading actor checkpoint from {ckpt_path}")
        loaded_dict = torch.load(
            ckpt_path, weights_only=True, map_location=self.device)
        self.actor.load_state_dict(loaded_dict["actor_model_state_dict"])
        # Also update target actor when loading actor only
        if "target_actor_state_dict" in loaded_dict:
            self.target_actor.load_state_dict(
                loaded_dict["target_actor_state_dict"])
        else:
            # For backward compatibility with old checkpoints
            logger.warning(
                "Target actor state not found in checkpoint, copying from main actor")
            self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer.load_state_dict(
            loaded_dict["actor_optimizer_state_dict"])
        logger.info("Actor checkpoint loaded successfully")

    @property
    def inference_model(self) -> Dict[str, nn.Module]:
        """Return models for inference."""
        return {
            "actor": self.actor,
            "target_actor": self.target_actor,
            "critic": self.critic,
        }

    # --- Evaluation ---

    def _create_eval_callbacks(self):
        self.eval_callbacks: List[RL_EvalCallback] = []
        if self.config.eval_callbacks is not None:
            for cb in self.config.eval_callbacks:
                self.eval_callbacks.append(
                    instantiate(
                        self.config.eval_callbacks[cb], training_loop=self)
                )

    def _pre_evaluate_policy(self, reset_env: bool = True):
        self._eval_mode()
        self.env.set_is_evaluating()
        if reset_env:
            _ = self.env.reset_all()

        for c in self.eval_callbacks:
            c.on_pre_evaluate_policy()

    def _post_evaluate_policy(self):
        for c in self.eval_callbacks:
            c.on_post_evaluate_policy()

    def _pre_eval_env_step(self, actor_state: dict):
        self.actor.eval()
        actions = self.actor.act_inference(actor_state["obs"]["actor_obs"])
        actor_state.update({"actions": actions})
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state

    def _post_eval_env_step(self, actor_state):
        for c in self.eval_callbacks:
            actor_state = c.on_post_eval_env_step(actor_state)
        return actor_state

    @torch.no_grad()
    def evaluate_policy(self, max_steps: Optional[int] = None):
        self._create_eval_callbacks()
        self._pre_evaluate_policy()
        actor_state = {"done_indices": [], "stop": False}
        self.eval_policy = self._get_inference_policy()
        obs_dict = self.env.reset_all()
        init_actions = torch.zeros(
            self.env.num_envs, self.num_actions, device=self.device)
        actor_state.update({"obs": obs_dict, "actions": init_actions})
        actor_state = self._pre_eval_env_step(actor_state)

        if max_steps is None:
            it = count(0)
        else:
            it = trange(max_steps)

        for step in it:
            actor_state["step"] = step
            actor_state = self._pre_eval_env_step(actor_state)
            actor_state = self.env_step(actor_state)
            actor_state = self._post_eval_env_step(actor_state)
        self._post_evaluate_policy()

    def _post_epoch_logging(self, log_dict, width=80, pad=35):
        """Comprehensive logging method similar to PPO."""
        self.tot_timesteps += self.samples_per_iter
        self.tot_time += log_dict['collection_time'] + log_dict['learn_time']
        iteration_time = log_dict['collection_time'] + log_dict['learn_time']

        # Episode info logging
        ep_string = ''
        if log_dict['ep_infos']:
            for key in log_dict['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in log_dict['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat(
                        (infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, log_dict['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}}
                    {value:.4f}\n"""

        # Training metrics
        train_log_dict = {}
        fps = int(self.samples_per_iter /
                  (log_dict['collection_time'] + log_dict['learn_time']))
        train_log_dict['fps'] = fps
        train_log_dict['replay_buffer_size'] = self.replay_buffer.size()
        train_log_dict['alpha_value'] = self.alpha.item()

        # Environment metrics
        env_log_dict = self.episode_env_tensors.mean_and_clear()
        env_log_dict = {f"Env/{k}": v for k, v in env_log_dict.items()}

        # Log to TensorBoard
        self._logging_to_writer(log_dict, train_log_dict, env_log_dict)

        # Create console output
        str_header = f" \033[1m SAC Learning iteration {log_dict['it']}/{self.current_learning_iteration + log_dict['num_learning_iterations']} \033[0m "

        if len(log_dict['rewbuffer']) > 0:
            log_string = (f"""{str_header.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {train_log_dict['fps']:.0f} steps/s (Collection: {log_dict['collection_time']:.3f}s, Learning {log_dict['learn_time']:.3f}s)\n"""
                          f"""{'Replay buffer size:':>{pad}} {train_log_dict['replay_buffer_size']}\n"""
                          f"""{'Alpha value:':>{pad}} {train_log_dict['alpha_value']:.4f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(log_dict['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(log_dict['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{str_header.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {train_log_dict['fps']:.0f} steps/s (Collection: {log_dict['collection_time']:.3f}s, Learning {log_dict['learn_time']:.3f}s)\n"""
                          f"""{'Replay buffer size:':>{pad}} {train_log_dict['replay_buffer_size']}\n"""
                          f"""{'Alpha value:':>{pad}} {train_log_dict['alpha_value']:.4f}\n""")

        # Add environment metrics
        env_log_string = ""
        for k, v in env_log_dict.items():
            entry = f"{f'{k}:':>{pad}} {v:.4f}"
            env_log_string += f"{entry}\n"
        log_string += env_log_string
        log_string += ep_string

        # Add loss information
        if log_dict['loss_dict']:
            loss_string = ""
            for loss_name, loss_value in log_dict['loss_dict'].items():
                if loss_name in ['Actor_Entropy_Term', 'Actor_Q_Term']:
                    # Special formatting for actor components
                    component_name = loss_name.replace(
                        'Actor_', '').replace('_', ' ')
                    loss_string += f"""{f'{component_name}:':>{pad}}
                        {loss_value:.4f}\n"""
                else:
                    loss_string += f"""{f'{loss_name} Loss:':>{pad}}
                        {loss_value:.4f}\n"""
            log_string += loss_string

        # Add metrics string
        metrics_log_string = ""
        if 'metrics_dict' in log_dict:
            for k, v in log_dict['metrics_dict'].items():
                if k == 'Action_Std':
                    entry = f"{f'{k}:':>{pad}} {v:.4f}"
                elif k == 'Action_Mean' or k == 'Entropy':
                    entry = f"{f'{k}:':>{pad}} {v:.4f}"
                elif k in ['Critic_Grad_Norm', 'Actor_Grad_Norm']:
                    entry = f"{f'{k}:':>{pad}} {v:.4f}"
                elif k in ['Mean_Q1', 'Mean_Q2', 'Mean_Target_Q', 'Mean_Min_Q']:
                    entry = f"{f'{k}:':>{pad}} {v:.4f}"
                else:
                    entry = f"{f'{k}:':>{pad}} {v:.4f}"
                metrics_log_string += f"{entry}\n"
        log_string += metrics_log_string

        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Total updates:':>{pad}} {self.updates}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (log_dict['it'] + 1) * (log_dict['num_learning_iterations'] - log_dict['it']):.1f}s\n""")
        log_string += f"Logging Directory: {self.log_dir}"

        # Use rich Live to update console
        with Live(Panel(log_string, title="SAC Training Log"), refresh_per_second=4, console=console):
            pass

    def _post_rollout_logging(self, iteration: int, width=80, pad=35):
        """Logging method specifically for rollout results."""
        # Environment metrics
        env_log_dict = self.episode_env_tensors.mean_and_clear()
        env_log_dict = {f"Env/{k}": v for k, v in env_log_dict.items()}

        # Log to TensorBoard
        self._logging_rollout_to_writer(iteration, env_log_dict)

        # Create console output
        str_header = f" \033[1m SAC Rollout Iteration {iteration} \033[0m "

        if len(self.rewbuffer) > 0:
            log_string = (f"""{str_header.center(width, ' ')}\n\n"""
                          f"""{'Total steps:':>{pad}} {self.total_steps}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(self.rewbuffer):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(self.lenbuffer):.2f}\n""")
        else:
            log_string = (f"""{str_header.center(width, ' ')}\n\n"""
                          f"""{'Total steps:':>{pad}} {self.total_steps}\n""")

        # Add environment metrics
        env_log_string = ""
        for k, v in env_log_dict.items():
            entry = f"{f'{k}:':>{pad}} {v:.4f}"
            env_log_string += f"{entry}\n"
        log_string += env_log_string

        # Add episode info
        ep_string = ''
        if self.ep_infos:
            for key in self.ep_infos[0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in self.ep_infos:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat(
                        (infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                ep_string += f"""{f'Mean episode {key}:':>{pad}}
                    {value:.4f}\n"""
        log_string += ep_string

        log_string += (f"""{'-' * width}\n"""
                       f"""{'Iteration completed:':>{pad}} {iteration}\n""")

        console.print(
            Panel(log_string, title="[bold green]SAC Rollout Log", border_style="green"))

    def _logging_to_writer(self, log_dict, train_log_dict, env_log_dict):
        """Log metrics to TensorBoard."""
        # Loss logging
        for loss_key, loss_value in log_dict['loss_dict'].items():
            self.writer.add_scalar(
                f'Loss/{loss_key}', loss_value, log_dict['it'])

        # Learning rates
        self.writer.add_scalar('Loss/actor_learning_rate',
                               self.actor_learning_rate, log_dict['it'])
        self.writer.add_scalar('Loss/critic_learning_rate',
                               self.critic_learning_rate, log_dict['it'])
        self.writer.add_scalar('Loss/alpha_learning_rate',
                               self.alpha_learning_rate, log_dict['it'])

        # SAC specific metrics
        self.writer.add_scalar('Policy/alpha_value',
                               train_log_dict['alpha_value'], log_dict['it'])
        self.writer.add_scalar('Policy/replay_buffer_size',
                               train_log_dict['replay_buffer_size'], log_dict['it'])

        # Additional alpha logging
        if self.autotune_alpha:
            self.writer.add_scalar(
                'Policy/log_alpha', self.log_alpha.item(), log_dict['it'])
            self.writer.add_scalar(
                'Policy/target_entropy', self.target_entropy, log_dict['it'])
            if 'Alpha' in log_dict['loss_dict']:
                self.writer.add_scalar(
                    'Policy/alpha_loss', log_dict['loss_dict']['Alpha'], log_dict['it'])
        else:
            self.writer.add_scalar('Policy/alpha_fixed',
                                   self.alpha_value, log_dict['it'])

        # Performance metrics
        self.writer.add_scalar(
            'Perf/total_fps', train_log_dict['fps'], log_dict['it'])
        self.writer.add_scalar('Perf/collection_time',
                               log_dict['collection_time'], log_dict['it'])
        self.writer.add_scalar('Perf/learning_time',
                               log_dict['learn_time'], log_dict['it'])
        self.writer.add_scalar('Perf/total_updates',
                               self.updates, log_dict['it'])

        # Episode metrics
        if len(log_dict['rewbuffer']) > 0:
            self.writer.add_scalar(
                'Train/mean_reward', statistics.mean(log_dict['rewbuffer']), log_dict['it'])
            self.writer.add_scalar(
                'Train/mean_episode_length', statistics.mean(log_dict['lenbuffer']), log_dict['it'])
            self.writer.add_scalar(
                'Train/mean_reward/time', statistics.mean(log_dict['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time',
                                   statistics.mean(log_dict['lenbuffer']), self.tot_time)

        # Environment metrics
        if len(env_log_dict) > 0:
            for k, v in env_log_dict.items():
                self.writer.add_scalar(k, v, log_dict['it'])

        # Actor loss components
        if 'Actor_Entropy_Term' in log_dict['loss_dict']:
            self.writer.add_scalar(
                'ActorComponents/Entropy_Term', log_dict['loss_dict']['Actor_Entropy_Term'], log_dict['it'])
        if 'Actor_Q_Term' in log_dict['loss_dict']:
            self.writer.add_scalar(
                'ActorComponents/Q_Term', log_dict['loss_dict']['Actor_Q_Term'], log_dict['it'])

        # Metrics logging
        if 'metrics_dict' in log_dict:
            for metric_key, metric_value in log_dict['metrics_dict'].items():
                self.writer.add_scalar(
                    f'Metrics/{metric_key}', metric_value, log_dict['it'])

        if 'alpha_value' in log_dict:
            self.writer.add_scalar(
                'Policy/alpha_value', log_dict['alpha_value'], log_dict['it'])

    def _logging_rollout_to_writer(self, iteration: int, env_log_dict: Dict[str, float]):
        """Log rollout results to TensorBoard."""
        # Log episode rewards and lengths (following training code pattern)
        if len(self.rewbuffer) > 0:
            self.writer.add_scalar(
                'Rollout/mean_reward', statistics.mean(self.rewbuffer), iteration)
            self.writer.add_scalar(
                'Rollout/mean_episode_length', statistics.mean(self.lenbuffer), iteration)

        # Log total steps
        self.writer.add_scalar('Rollout/total_steps',
                               self.total_steps, iteration)

        # Log environment metrics
        if len(env_log_dict) > 0:
            for k, v in env_log_dict.items():
                self.writer.add_scalar(k, v, iteration)

        # Log episode info
        if self.ep_infos:
            for key in self.ep_infos[0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in self.ep_infos:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat(
                        (infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar(f'Episode/{key}', value, iteration)

    def env_step(self, actor_state):
        """Environment step for evaluation."""
        obs_dict, rewards, dones, extras = self.env.step(actor_state)
        actor_state.update(
            {"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras}
        )
        return actor_state

    @torch.no_grad()
    def get_example_obs(self):
        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():
            print(obs_key, sorted(self.env.config.obs.obs_dict[obs_key]))
        # move to cpu
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].cpu()
        return obs_dict

    def _get_inference_policy(self, device: Optional[torch.device] = None):
        self.actor.eval()
        if device is not None:
            self.actor.to(device)
        return self.actor.act_inference
