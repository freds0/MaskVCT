"""
Dataset helpers for MaskVCT.

The paper uses several corpora and a prompt/source pairing strategy. The local
repository does not yet contain a dedicated MaskVCT dataset pipeline, so this
module only provides a small placeholder API.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MaskVCTSample:
    source_path: str
    prompt_path: str
    source_speaker_id: Optional[str] = None
    prompt_speaker_id: Optional[str] = None

