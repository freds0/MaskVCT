"""
Paper-aligned preprocessing hooks for MaskVCT.

TODO(MaskVCT): Paper does not fully specify this behavior for the local repo,
so these helpers are intentionally light-weight placeholders.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MaskVCTPreprocessConfig:
    prompt_seconds: float = 3.0
    source_seconds: float = 10.24
    sample_rate: int = 24000


def phase_aug(waveform: np.ndarray) -> np.ndarray:
    return waveform


def pitch_shift_clean_speech(waveform: np.ndarray, semitones: float = 0.0) -> np.ndarray:
    return waveform


def select_prompt_and_source(
    waveform: np.ndarray,
    prompt_seconds: float = 3.0,
    source_seconds: float = 10.24,
    sample_rate: int = 24000,
) -> tuple[np.ndarray, np.ndarray]:
    prompt_len = int(round(prompt_seconds * sample_rate))
    source_len = int(round(source_seconds * sample_rate))
    prompt = waveform[:prompt_len]
    source = waveform[prompt_len : prompt_len + source_len]
    return prompt, source

