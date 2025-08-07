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
from humanoidverse.agents.modules.sac_modules import SACActor, SACCritic, DoubleQCritic
from humanoidverse.agents.modules.data_utils import ReplayBuffer
from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.utils.average_meters import TensorAverageMeterDict

from hydra.utils import instantiate

from typing import Optional, Dict, List
from itertools import count

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
        self.learning_frequency = self.config.learning_frequency
        self.target_update_frequency = self.config.target_update_frequency
        self.tau = self.config.tau
        self.gamma = self.config.gamma

        # Off-policy training parameters
        self.samples_per_iter = self.config.samples_per_iter
        self.actor_steps = self.config.actor_steps
        self.critic_steps = self.config.critic_steps

        # Learning rates
        self.actor_learning_rate = self.config.actor_learning_rate
        self.critic_learning_rate = self.config.critic_learning_rate
        self.alpha_learning_rate = self.config.alpha_learning_rate

        # Entropy temperature
        self.target_entropy = self.config.target_entropy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)

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

    def setup(self):
        self.actor = SACActor(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config_dict=self.config.module_dict.actor,
            num_actions=self.num_actions,
            init_noise_std=self.config.init_noise_std,
        ).to(self.device)

        def critic_factory():
            return SACCritic(self.algo_obs_dim_dict, self.config.module_dict.critic, self.num_actions)

        self.double_q_critic = DoubleQCritic(
            critic_factory=critic_factory,
            device=self.device,
        )

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.actor_learning_rate
        )
        self.critic_optimizer = optim.Adam(
            self.double_q_critic.parameters(), lr=self.critic_learning_rate
        )
        self.alpha_optimizer = optim.Adam(
            [self.log_alpha], lr=self.alpha_learning_rate)

        self.replay_buffer = ReplayBuffer(
            buffer_size=int(self.replay_buffer_size),
            num_envs=self.num_envs,
            device="cpu",
        )
        for obs_key, obs_shape in self.algo_obs_dim_dict.items():
            self.replay_buffer.register_key(obs_key, obs_shape, is_obs=True)

        self.replay_buffer.register_key("actions", (self.num_actions,))
        self.replay_buffer.register_key("rewards", (1,))
        self.replay_buffer.register_key("dones", (1,), dtype=torch.bool)

        logger.info(f"Replay buffer:\n{self.replay_buffer}")

    def learn(self):
        obs_dict = self.env.reset_all()
        obs_dict = _dict_to_device(obs_dict, self.device)
        self._train_mode()

        # Collect initial samples if needed
        if self.replay_buffer.size() < self.learning_starts:
            logger.info(f"Collecting {self.learning_starts} initial samples")
            self._collect_initial_samples(obs_dict)

        for iteration in range(
            self.current_learning_iteration,
            self.current_learning_iteration + self.num_learning_iterations,
        ):
            self.start_time = time.time()
            
            # Collect samples_per_iter samples
            obs_dict = self._collect_samples(obs_dict, self.samples_per_iter)
            
            self.stop_time = time.time()
            self.collection_time = self.stop_time - self.start_time
            self.start_time = self.stop_time

            # Perform multiple training steps
            loss_dict = self._training_phase()
            
            self.stop_time = time.time()
            self.learn_time = self.stop_time - self.start_time

            # Logging
            log_dict = {
                'it': iteration,
                'loss_dict': loss_dict,
                'collection_time': self.collection_time,
                'learn_time': self.learn_time,
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

    def _train_mode(self):
        """Set networks to training mode."""
        self.actor.train()
        self.double_q_critic.train()

    def _eval_mode(self):
        """Set networks to evaluation mode."""
        self.actor.eval()
        self.double_q_critic.eval()

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
                transition_data[obs_key] = obs_dict[obs_key].cpu()
            transition_data["actions"] = actions.cpu()
            transition_data["rewards"] = rewards.cpu().unsqueeze(1)
            transition_data["dones"] = dones.cpu().unsqueeze(1)

            self.replay_buffer.add(transition_data)

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
                    episode_reward = self.current_episode_reward[env_idx].item()
                    episode_length = self.current_episode_length[env_idx].item()
                    
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    self.rewbuffer.append(episode_reward)
                    self.lenbuffer.append(episode_length)
                    
                    self.current_episode_reward[env_idx] = 0
                    self.current_episode_length[env_idx] = 0

            # Update total steps
            self.total_steps += self.num_envs

        return next_obs_dict

    def _collect_initial_samples(self, obs_dict):
        """Collect initial samples to populate replay buffer."""
        sample_count = 0
        with Progress() as progress:
            task = progress.add_task(
                "Collecting initial samples", total=self.learning_starts)
            while sample_count < self.learning_starts:
                obs_dict = self._collect_experience(obs_dict)
                sample_count += self.num_envs
                progress.update(task, advance=self.num_envs)

    def _collect_samples(self, obs_dict, num_samples):
        """Collect num_samples samples from environment."""
        sample_count = 0
        while sample_count < num_samples:
            obs_dict = self._collect_experience(obs_dict)
            sample_count += self.num_envs
        return obs_dict

    def _training_phase(self):
        """Perform multiple training steps for both critic and actor."""
        # Update critics for critic_steps
        critic_losses = []
        for _ in range(self.critic_steps):
            critic_loss_1, critic_loss_2 = self._update_critics_step()
            critic_losses.append((critic_loss_1, critic_loss_2))

        # Update actor and alpha for actor_steps
        actor_losses = []
        alpha_losses = []
        for _ in range(self.actor_steps):
            actor_loss, alpha_loss = self._update_actor_and_alpha_step()
            actor_losses.append(actor_loss)
            alpha_losses.append(alpha_loss)

            # Update target networks
            if self.updates % self.target_update_frequency == 0:
                self.double_q_critic.soft_update_targets(self.tau)

            self.updates += 1

        # Calculate average losses
        loss_dict = {}
        if critic_losses:
            avg_critic_loss_1 = sum(
                [c[0] for c in critic_losses]) / len(critic_losses)
            avg_critic_loss_2 = sum(
                [c[1] for c in critic_losses]) / len(critic_losses)
            loss_dict['Critic_Q1'] = avg_critic_loss_1
            loss_dict['Critic_Q2'] = avg_critic_loss_2


        if actor_losses:
            avg_actor_loss = sum(actor_losses) / len(actor_losses)
            avg_alpha_loss = sum(alpha_losses) / len(alpha_losses)
            loss_dict['Actor'] = avg_actor_loss
            loss_dict['Alpha'] = avg_alpha_loss
            loss_dict['Alpha_Value'] = self.alpha.item()
        
        return loss_dict

    def _update_critics_step(self):
        """Single critic update step."""
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
            # Sample next actions from current policy
            next_actions = self.actor.act(next_obs["actor_obs"])
            next_log_probs = self.actor.get_actions_log_prob(next_actions)

            # Compute target Q values
            target_q1, target_q2 = self.double_q_critic.target_forward(
                next_obs["critic_obs"], next_actions
            )
            target_q = torch.min(
                target_q1, target_q2
            ) - self.alpha * next_log_probs.unsqueeze(1)

            # Compute target values
            target_values = rewards + self.gamma * \
                (1 - dones.float()) * target_q

        # Current Q values
        current_q1, current_q2 = self.double_q_critic(
            obs["critic_obs"], actions)

        # Critic losses
        critic_loss_1 = F.mse_loss(current_q1, target_values)
        critic_loss_2 = F.mse_loss(current_q2, target_values)

        # Combined critic loss
        critic_loss = critic_loss_1 + critic_loss_2

        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss_1.item(), critic_loss_2.item()

    def _update_actor_and_alpha_step(self):
        """Single actor and alpha update step."""
        # Sample batch from replay buffer
        current_samples, next_obs_samples = self.replay_buffer.sample(
            self.batch_size)

        # Move to device
        current_samples = _dict_to_device(current_samples, self.device)

        # Extract observations
        obs = {k: v for k, v in current_samples.items()
               if k in self.algo_obs_dim_dict}

        # Sample actions from current policy
        actions = self.actor.act(obs["actor_obs"])
        log_probs = self.actor.get_actions_log_prob(actions)

        # Q values for sampled actions
        q1, q2 = self.double_q_critic(obs["critic_obs"], actions)
        q_min = torch.min(q1, q2)

        # Actor loss
        actor_loss = (self.alpha * log_probs.unsqueeze(1) - q_min).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha (temperature parameter)
        alpha_loss = -(
            self.log_alpha * (log_probs + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return actor_loss.item(), alpha_loss.item()

    @property
    def alpha(self):
        """Current value of temperature parameter."""
        return self.log_alpha.exp()

    def save(self, path: str):
        """Save model checkpoint."""
        logger.info(f"Saving checkpoint to {path}")
        torch.save(
            {
                "actor_model_state_dict": self.actor.state_dict(),
                "double_q_critic_state_dict": self.double_q_critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
                "log_alpha": self.log_alpha,
                "iter": self.current_learning_iteration,
                "total_steps": self.total_steps,
                "updates": self.updates,
            },
            path,
        )

    def load(self, ckpt_path: str):
        logger.info(f"Loading checkpoint from {ckpt_path}")
        loaded_dict = torch.load(ckpt_path, map_location=self.device)

        self.actor.load_state_dict(loaded_dict["actor_model_state_dict"])
        self.double_q_critic.load_state_dict(
            loaded_dict["double_q_critic_state_dict"])

        self.actor_optimizer.load_state_dict(
            loaded_dict["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(
            loaded_dict["critic_optimizer_state_dict"]
        )
        self.alpha_optimizer.load_state_dict(
            loaded_dict["alpha_optimizer_state_dict"])

        self.log_alpha = loaded_dict["log_alpha"]
        self.current_learning_iteration = loaded_dict["iter"]
        self.total_steps = loaded_dict["total_steps"]
        self.updates = loaded_dict["updates"]

        logger.info("Checkpoint loaded successfully")

    @property
    def inference_model(self) -> Dict[str, nn.Module]:
        """Return models for inference."""
        return {
            "actor": self.actor,
            "double_q_critic": self.double_q_critic,
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
        actor_state = self._create_actor_state()
        self.eval_policy = self._get_inference_policy()
        obs_dict = self.env.reset_all()
        init_actions = torch.zeros(
            self.env.num_envs, self.num_actions, device=self.device)
        actor_state.update({"obs": obs_dict, "actions": init_actions})
        actor_state = self._pre_eval_env_step(actor_state)

        if max_steps is None:
            it = count(0)
        else:
            from tqdm import trange

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
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

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
                loss_string += f"""{f'{loss_name} Loss:':>{pad}} {loss_value:.4f}\n"""
            log_string += loss_string

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

    def _logging_to_writer(self, log_dict, train_log_dict, env_log_dict):
        """Log metrics to TensorBoard."""
        # Loss logging
        for loss_key, loss_value in log_dict['loss_dict'].items():
            self.writer.add_scalar(f'Loss/{loss_key}', loss_value, log_dict['it'])
        
        # Learning rates
        self.writer.add_scalar('Loss/actor_learning_rate', self.actor_learning_rate, log_dict['it'])
        self.writer.add_scalar('Loss/critic_learning_rate', self.critic_learning_rate, log_dict['it'])
        self.writer.add_scalar('Loss/alpha_learning_rate', self.alpha_learning_rate, log_dict['it'])
        
        # SAC specific metrics
        self.writer.add_scalar('Policy/alpha_value', train_log_dict['alpha_value'], log_dict['it'])
        self.writer.add_scalar('Policy/replay_buffer_size', train_log_dict['replay_buffer_size'], log_dict['it'])
        
        # Performance metrics
        self.writer.add_scalar('Perf/total_fps', train_log_dict['fps'], log_dict['it'])
        self.writer.add_scalar('Perf/collection_time', log_dict['collection_time'], log_dict['it'])
        self.writer.add_scalar('Perf/learning_time', log_dict['learn_time'], log_dict['it'])
        self.writer.add_scalar('Perf/total_updates', self.updates, log_dict['it'])
        
        # Episode metrics
        if len(log_dict['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(log_dict['rewbuffer']), log_dict['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(log_dict['lenbuffer']), log_dict['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(log_dict['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(log_dict['lenbuffer']), self.tot_time)
        
        # Environment metrics
        if len(env_log_dict) > 0:
            for k, v in env_log_dict.items():
                self.writer.add_scalar(k, v, log_dict['it'])
    
    def env_step(self, actor_state):
        """Environment step for evaluation."""
        obs_dict, rewards, dones, extras = self.env.step(actor_state)
        actor_state.update(
            {"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras}
        )
        return actor_state
