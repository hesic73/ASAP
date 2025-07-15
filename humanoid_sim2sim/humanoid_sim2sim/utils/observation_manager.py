import numpy as np
from omegaconf import DictConfig
from typing import Dict


class ObservationManager:
    """
    Manages observations for a single environment based on a configuration object.
    It handles scaling, history management, and proper flattening of observation vectors.
    """

    def __init__(self, config: DictConfig):
        """
        Initializes the ObservationManager.

        Args:
            config (DictConfig): The configuration object containing observation specs.
        """
        self.config = config

        # --- 将 obs_dims 列表转换为字典，方便查询 ---
        self.obs_dims_map = {
            key: val for item in self.config.obs_dims for key, val in item.items()
        }

        # --- Actor observation configuration ---
        self.actor_obs_keys = sorted(self.config.obs_dict.actor_obs)

        # --- History configuration for the actor ---
        self.history_spec = self.config.obs_auxiliary.get(
            'history_actor', {})
        self.history_keys = sorted(self.history_spec.keys())

        self.history_buffer: Dict[str, np.ndarray] = {}
        for key in self.history_keys:
            length = self.history_spec[key]
            dim = self.obs_dims_map[key]
            self.history_buffer[key] = np.zeros(
                (length, dim), dtype=np.float32)

        # --- Store current and previous raw observations ---
        self.current_raw_obs: Dict[str, np.ndarray] = {}
        # 新增：用于暂存上一个时间步的观测，以实现历史记录的延迟更新
        self.last_timestep_obs: Dict[str, np.ndarray] = {}

    def reset(self):
        """
        Resets all history buffers and observation dictionaries to zero/empty.
        """
        for key in self.history_buffer:
            self.history_buffer[key].fill(0.0)
        self.current_raw_obs.clear()
        # 新增：同时清空上一步的观测记录
        self.last_timestep_obs.clear()

    def update(self, raw_obs: Dict[str, np.ndarray]):
        """
        Updates the manager with new raw observations and updates the history
        using the observations from the *previous* timestep.

        Args:
            raw_obs (Dict[str, np.ndarray]): A dictionary of the latest raw observations.
        """
        # 仅当 last_timestep_obs 不为空时（即至少已经 update 过一次），才更新 history
        if self.last_timestep_obs:
            # 使用上一个时间步的数据来更新 history buffer
            # 注意：history buffer存储的是scaled过的observation
            for key in self.history_keys:
                if key in self.last_timestep_obs:
                    buffer = self.history_buffer[key]
                    buffer[1:] = buffer[:-1]
                    # 应用scale后存储到history buffer
                    raw_value = self.last_timestep_obs[key]
                    scale = self.config.obs_scales[key]
                    scaled_value = raw_value * scale
                    buffer[0] = scaled_value

        # 存储当前时间步的观测值，供 get() 方法立即使用
        self.current_raw_obs = raw_obs
        # 暂存当前时间步的观测值，供下一个 update 周期使用
        self.last_timestep_obs = raw_obs

    def get(self) -> np.ndarray:
        """
        Constructs and returns the final, scaled, and flattened actor observation vector.
        The order of components is determined by sorting their string keys.

        Returns:
            np.ndarray: The complete observation vector for the actor.
        """
        obs_parts = []

        for key in self.actor_obs_keys:
            if key == "history_actor":
                history_component_parts = []
                for hist_key in self.history_keys:
                    hist_len = self.history_spec[hist_key]
                    # History buffer中已经存储了scaled过的observation
                    flat_history = self.history_buffer[hist_key][:hist_len].flatten(
                    )
                    history_component_parts.append(flat_history)

                full_history_vec = np.concatenate(history_component_parts)
                # 只对history整体应用history_actor的scale（通常是1.0）
                scaled_history = full_history_vec * self.config.obs_scales.history_actor
                obs_parts.append(scaled_history)
            else:
                # 非历史部分总是使用最新的观测值并应用相应的scale
                raw_value = self.current_raw_obs[key]
                scale = self.config.obs_scales[key]
                scaled_value = raw_value * scale
                obs_parts.append(scaled_value)

        full_obs = np.concatenate(obs_parts).astype(np.float32)
        full_obs = np.clip(full_obs, -self.config.clip_obs,
                           self.config.clip_obs)
        return full_obs
