"""
Conditioning helpers for MaskVCT.

The paper specifies SylBoost, continuous linguistic features, pitch conditioning,
and prompt-based speaker conditioning. The local codebase currently implements
these through the MaskVCT model and the feature pipeline, while retaining these
helpers as a future extension point.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class MaskVCTConditionBatch:
    prompt_tokens: torch.Tensor
    source_tokens: torch.Tensor
    ling_disc_prompt: Optional[torch.Tensor] = None
    ling_disc_source: Optional[torch.Tensor] = None
    ling_cont_prompt: Optional[torch.Tensor] = None
    ling_cont_source: Optional[torch.Tensor] = None
    pitch_prompt: Optional[torch.Tensor] = None
    pitch_source: Optional[torch.Tensor] = None


def split_prompt_source(tensor: torch.Tensor, prompt_len: int):
    return tensor[:, :prompt_len], tensor[:, prompt_len:]

