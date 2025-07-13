import torch
import numpy as np
import random
import os

from typing import Tuple


@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def torch_rand_float(lower, upper, shape: Tuple[int], device: torch.device) -> torch.Tensor:
    return (upper - lower) * torch.rand(*shape, device=device) + lower


@torch.jit.script
def copysign(a: float, b: torch.Tensor) -> torch.Tensor:
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)
