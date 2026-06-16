"""
Training entry points for MaskVCT.

TODO(MaskVCT): Paper does not fully specify this behavior for the local repo,
so the actual trainer should be added once the dataset pipeline is finalized.
"""

from dataclasses import dataclass


@dataclass
class MaskVCTTrainingConfig:
    steps: int = 250_000
    batch_size: int = 168
    learning_rate: float = 2e-4
    dropout: float = 0.05
    layer_drop: float = 0.05

