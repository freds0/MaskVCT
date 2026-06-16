import math

import torch
import torch.nn as nn


class SinusoidalPitchEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("pitch embedding dimension must be even")
        self.dim = dim

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        if f0.dim() != 2:
            raise ValueError("expected pitch tensor with shape (batch, time)")

        eps = 1.0
        half = self.dim // 2
        device = f0.device

        log_f0 = torch.log(f0.clamp_min(0.0) + eps)
        scales = torch.exp(
            torch.arange(half, device=device, dtype=log_f0.dtype)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        angles = log_f0.unsqueeze(-1) * scales.view(1, 1, -1)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


def resample_pitch_to_50hz(f0: torch.Tensor, src_hz: int, target_hz: int = 50) -> torch.Tensor:
    if src_hz == target_hz:
        return f0

    ratio = src_hz / target_hz
    target_len = max(1, int(round(f0.shape[-1] / ratio)))
    return torch.nn.functional.interpolate(
        f0.unsqueeze(1), size=target_len, mode="linear", align_corners=False
    ).squeeze(1)

