import math

import torch
from einops import rearrange


def top_k(logits: torch.Tensor, thres: float = 0.9) -> torch.Tensor:
    k = max(1, math.ceil((1 - thres) * logits.shape[-1]))
    values, indices = logits.topk(k, dim=-1)
    masked = torch.full_like(logits, float("-inf"))
    masked.scatter_(2, indices, values)
    return masked


def log(t: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    return torch.log(t + eps)


def gumbel_noise(t: torch.Tensor) -> torch.Tensor:
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t: torch.Tensor, temperature: float = 1.0, dim: int = -1) -> torch.Tensor:
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


def cosine_mask_prob(step: torch.Tensor) -> torch.Tensor:
    return torch.cos(step * math.pi / 2)


def choose_next_mask(scores: torch.Tensor, next_mask_num: int) -> torch.Tensor:
    next_mask_num = max(0, int(next_mask_num))
    if next_mask_num == 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    indices = scores.topk(next_mask_num, dim=-1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter(1, indices, True)

