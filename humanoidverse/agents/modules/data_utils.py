import torch
from torch import nn, Tensor
from prettytable import PrettyTable

from typing import Dict, List, Tuple, Union


def compute_returns(self, rewards, values, dones, last_values, gamma, lam):
    advantage = 0
    returns = torch.zeros_like(values)
    for step in reversed(range(self.num_transitions_per_env)):
        if step == self.num_transitions_per_env - 1:
            next_values = last_values
        else:
            next_values = values[step + 1]
        next_is_not_terminal = 1.0 - dones[step].float()
        delta = (
            rewards[step] + next_is_not_terminal *
            gamma * next_values - values[step]
        )
        advantage = delta + next_is_not_terminal * gamma * lam * advantage
        returns[step] = advantage + values[step]

    # Compute and normalize the advantages
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


class RolloutStorage(nn.Module):
    def __init__(self, num_envs, num_transitions_per_env, device="cpu"):
        super().__init__()

        self.device = device

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        # self.saved_hidden_states_a = None
        # self.saved_hidden_states_c = None

        self.step = 0
        self.stored_keys = list()

    def register_key(self, key: str, shape=(), dtype=torch.float):
        # This class was partially copied from https://github.com/NVlabs/ProtoMotions/blob/94059259ba2b596bf908828cc04e8fc6ff901114/phys_anim/agents/utils/data_utils.py
        assert not hasattr(self, key), key
        assert isinstance(shape, (list, tuple)
                          ), "shape must be a list or tuple"
        buffer = torch.zeros(
            (self.num_transitions_per_env, self.num_envs) + shape,
            dtype=dtype,
            device=self.device,
        )
        self.register_buffer(key, buffer, persistent=False)
        self.stored_keys.append(key)

    def increment_step(self):
        self.step += 1

    def update_key(self, key: str, data: Tensor):
        # This class was partially copied from https://github.com/NVlabs/ProtoMotions/blob/94059259ba2b596bf908828cc04e8fc6ff901114/phys_anim/agents/utils/data_utils.py
        assert not data.requires_grad
        assert self.step < self.num_transitions_per_env, "Rollout buffer overflow"
        getattr(self, key)[self.step].copy_(data)

    def batch_update_data(self, key: str, data: Tensor):
        # This class was partially copied from https://github.com/NVlabs/ProtoMotions/blob/94059259ba2b596bf908828cc04e8fc6ff901114/phys_anim/agents/utils/data_utils.py
        assert not data.requires_grad
        getattr(self, key)[:] = data
        # self.store_dict[key] += self.total_sum()

    def _save_hidden_states(self, hidden_states):
        assert NotImplementedError
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = (
            hidden_states[0]
            if isinstance(hidden_states[0], tuple)
            else (hidden_states[0],)
        )
        hid_c = (
            hidden_states[1]
            if isinstance(hidden_states[1], tuple)
            else (hidden_states[1],)
        )

        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(
                    self.observations.shape[0], *
                    hid_a[i].shape, device=self.device
                )
                for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(
                    self.observations.shape[0], *
                    hid_c[i].shape, device=self.device
                )
                for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def get_statistics(self):
        raise NotImplementedError
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (
                flat_dones.new_tensor([-1], dtype=torch.int64),
                flat_dones.nonzero(as_tuple=False)[:, 0],
            )
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def query_key(self, key: str):
        assert hasattr(self, key), key
        return getattr(self, key)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, requires_grad=False, device=self.device
        )

        _buffer_dict = {
            key: getattr(self, key)[:].flatten(0, 1) for key in self.stored_keys
        }

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                _batch_buffer_dict = {
                    key: _buffer_dict[key][batch_idx] for key in self.stored_keys
                }
                yield _batch_buffer_dict


# Reference:
# - https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl_utils/buffers.py
class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        num_envs: int,
        device: torch.device = torch.device("cpu"),
    ):
        self._buffer_size = max(buffer_size // num_envs, 1)
        self.num_envs = num_envs
        self.device = device

        self._buffer_dict: Dict[str, Tensor] = {}
        self._buffer_dict_shape: Dict[str, Tuple[int, ...]] = {}
        self._buffer_dict_is_obs: Dict[str, bool] = {}

        self._pos = 0
        self._full = False

    def register_key(
        self,
        key: str,
        shape: Union[Tuple[int, ...], List[int], int],
        dtype: torch.dtype = torch.float,
        is_obs: bool = False,
    ):
        assert key not in self._buffer_dict, f"Key {key} already registered"
        if isinstance(shape, int):
            shape = (shape,)
        assert isinstance(shape, (list, tuple)
                          ), "shape must be a list or tuple"
        self._buffer_dict[key] = torch.zeros(
            (self._buffer_size, self.num_envs) + shape,
            dtype=dtype,
            device=self.device,
        )
        self._buffer_dict_shape[key] = shape
        self._buffer_dict_is_obs[key] = is_obs

    def size(self) -> int:
        if self._full:
            return self._buffer_size
        return self._pos

    def reset(self):
        self._pos = 0
        self._full = False

    def add(self, data_dict: Dict[str, Tensor], next_obs_dict:Dict[str, Tensor]):
        for key, data in data_dict.items():
            assert key in self._buffer_dict, f"Key {key} not registered"
            expected_shape = (self.num_envs, *self._buffer_dict_shape[key])
            assert data.shape == expected_shape, (
                f"{key} data shape {data.shape} does not match buffer shape. Expected {expected_shape}"
            )
            self._buffer_dict[key][self._pos] = data

        # Write next observations into the next index for obs keys
        if next_obs_dict is not None:
            next_index = (self._pos + 1) % self._buffer_size
            for key, is_obs in self._buffer_dict_is_obs.items():
                if is_obs:
                    assert key in next_obs_dict, f"Missing next_obs for key {key}"
                    next_data = next_obs_dict[key]
                    expected_shape = (self.num_envs, *self._buffer_dict_shape[key])
                    assert next_data.shape == expected_shape, (
                        f"next {key} data shape {next_data.shape} does not match buffer shape. Expected {expected_shape}"
                    )
                    self._buffer_dict[key][next_index] = next_data
        self._pos += 1
        if self._pos == self._buffer_size:
            self._full = True
            self._pos = 0

    def sample(self, batch_size: int) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        if self._full:
            batch_inds = (
                torch.randint(1, self._buffer_size,
                              (batch_size,), device=self.device)
                + self._pos
            ) % self._buffer_size
        else:
            batch_inds = torch.randint(
                0, self._pos, (batch_size,), device=self.device)

        env_indices = torch.randint(
            0, self.num_envs, (batch_size,), device=self.device)

        current_samples = {
            key: self._buffer_dict[key][batch_inds, env_indices]
            for key in self._buffer_dict.keys()
        }

        next_indices = (batch_inds + 1) % self._buffer_size

        next_obs_samples = {
            key: self._buffer_dict[key][next_indices, env_indices]
            for key, is_obs in self._buffer_dict_is_obs.items()
            if is_obs
        }

        return current_samples, next_obs_samples

    def __str__(self):
        table = PrettyTable()
        table.field_names = ["Key", "Shape", "Is Obs"]
        for key, shape, is_obs in zip(self._buffer_dict.keys(), self._buffer_dict_shape.values(), self._buffer_dict_is_obs.values()):
            table.add_row([key, shape, is_obs])
        return table.get_string()
