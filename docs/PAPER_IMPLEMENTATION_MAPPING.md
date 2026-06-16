# MaskVCT Paper Implementation Mapping

Primary source: `2509.17143v2.pdf`

This document tracks implementation requirements extracted from the paper and maps them to the intended code locations in the repository. Status values:

- `pending`: not implemented yet
- `in_progress`: partial implementation exists
- `complete`: implemented and validated

## Implementation Checklist

### Architecture
- Build a single-stage MaskVCT model for VC rather than the two-stage MaskGCT TTS pipeline.
- Use DAC acoustic tokens at 16 kHz with 9 RVQ codebooks.
- Use SylBoost 8.33 Hz linguistic features as the coarse content representation.
- Support both continuous and quantized linguistic conditioning paths.
- Support pitch conditioning with a log-scale sinusoidal embedding at 50 Hz.
- Implement prompt-speech conditioning with a 3-second speaker prompt.
- Use a 16-layer Transformer encoder with 16 heads, hidden size 1024, FFN size 4096, Pre-LN, RoPE, ReLU FFNs, and per-codebook heads.

### Conditioning and CFG
- Implement four training/inference condition modes:
  - all-conditioned
  - speaker+linguistic-conditioned
  - linguistic-conditioned
  - no-conditioned
- Implement triple CFG with separate weights for:
  - all conditions
  - speaker guidance
  - linguistic guidance
- Expose paper-matched presets:
  - `MaskVCT-All`: continuous linguistic features, pitch-conditioned
  - `MaskVCT-Spk`: quantized linguistic features, no pitch conditioning

### Training
- Train on 16 kHz DAC tokens.
- Apply PhaseAug before DAC encoding.
- Use 50% pitch-shifted perturbed speech and 50% clean speech.
- Train with 250k steps, AdamW, batch size 168, learning rate 2e-4, layer drop 5%, dropout 5%.
- Apply SpecAugment-style masking of 10% of the channel dimension on the combined embeddings.
- Use 3-second prompt speech and 10.24-second source windows.

### Inference
- Implement iterative masked decoding with codebook-wise schedules.
- Use the paper’s default inference settings:
  - `N = 64`
  - top-k = 35
  - top-p = 0.9
- Support the paper’s two main modes and the reported CFG coefficients.

### Evaluation
- Add documentation for LibriTTS-R and L2-ARCTIC evaluation setup.
- Track the objective metrics used in the paper:
  - WER/CER
  - S-SIM
  - FPC
  - UTMOS
  - Q-MOS
  - A-SIM
  - AS-MOS

## Mapping Table

| Paper Section | Page | Extracted Requirement | Corresponding Implementation File | Status | Notes / TODOs |
| --- | ---: | --- | --- | --- | --- |
| Abstract / Sec. 1 | 1 | Zero-shot VC with multi-factor controllability via multiple CFGs | `Amphion/models/tts/maskvct/README.md` | in_progress | High-level design goal and user-facing mode names |
| 2.1 Linguistic Conditioning | 2 | Use SylBoost 8.33 Hz discrete tokens and a continuous FFN-LN-FFN path | `Amphion/models/tts/maskvct/sylboost.py` | in_progress | SylBoost is integrated as a checkpoint-driven reader; fallback remains for missing checkpoints |
| 2.1 Linguistic Conditioning | 2 | Train with balanced sampling for continuous vs. quantized linguistic features | `Amphion/models/tts/maskvct/training.py` | pending | `TODO(MaskVCT): Paper does not fully specify this behavior.` |
| 2.2 Pitch Conditioning | 2 | Log-scale sinusoidal pitch embedding with 50 Hz Praat extraction | `Amphion/models/tts/maskvct/pitch.py` | in_progress | `TODO(MaskVCT): Paper does not fully specify this behavior.` |
| 2.3 Speaker Conditioning | 2 | 3-second speaker prompt appended to source conditions | `Amphion/models/tts/maskvct/inference.py` | pending | Prompt concatenation must preserve the source/prompt ordering in the paper |
| 2.4 Masked Codec LM | 2 | Masked token reconstruction over RVQ acoustic tokens with cosine masking schedule | `Amphion/models/tts/maskvct/model.py` | in_progress | Reuse the MaskGCT masking pattern only where it matches the paper |
| 2.4 Masked Codec LM | 2 | Iterative unmasking from all-masked state over `N` steps | `Amphion/models/tts/maskvct/sampling.py` | pending | Inference sampling schedule must be explicit and configurable |
| 2.5 Multiple CFGs | 2 | Triple CFG with `ωall`, `ωspk`, `ωling` | `Amphion/models/tts/maskvct/cfg.py` | in_progress | Need separate logits for all-conditioned, spk+ling, ling-only, and null |
| 2.6 Architecture Details | 3 | 16-layer Transformer encoder, 16 heads, hidden 1024, FFN 4096, Pre-LN, RoPE, ReLU | `Amphion/models/tts/maskvct/model.py` | in_progress | MaskGCT’s Llama-based decoder is not the paper’s architecture |
| 3.2 Training | 3 | DAC 16 kHz, 9 codebooks, PhaseAug, pitch-shift perturbation, 50% sampling each | `Amphion/models/tts/maskvct/acoustic_codec.py` | in_progress | DAC backend now supports the local 16 kHz / 24 kHz / 44.1 kHz weights; Amphion remains available as fallback. `TODO(MaskVCT): Paper does not fully specify this behavior.` |
| 3.2 Training | 3 | Train from scratch 250k steps, AdamW, batch 168, lr 2e-4, layer drop 5%, dropout 5% | `Amphion/models/tts/maskvct/config/maskvct.json` | pending | Hyperparameter defaults should match the paper |
| 3.2 Training | 3 | SpecAugment on combined embeddings by masking 10% of channel dimension | `Amphion/models/tts/maskvct/model.py` | pending | Likely model-side regularization rather than data-side augmentation |
| 3.3 Dataset | 3 | Train on LibriTTS-R, MLS-en, VCTK, LibriHeavy-Large, HiFi-TTS, LJSpeech, RAVDESS | `Amphion/models/tts/maskvct/data.py` | pending | Need dataset manifests or loaders that reflect the paper mix |
| 3.3 Dataset | 3 | Evaluation on LibriTTS-R test-clean and L2-ARCTIC | `docs/PAPER_IMPLEMENTATION_MAPPING.md` | pending | Also document evaluation scripts and subset selection |
| 3.4 Inference Setup | 3 | Default `N=64`, top-k=35, top-p=0.9, per-codebook step schedule | `Amphion/models/tts/maskvct/maskvct_inference.py` | in_progress | Paper reports `[40,16,2,1,1,1,1,1,1]` for the main 9-codebook setting |
| 3.4 Inference Setup | 3 | MaskVCT-All weights `(1.5, 1.0, 1.0)` | `Amphion/models/tts/maskvct/cfg.py` | in_progress | Pitch-conditioned, continuous linguistic features |
| 3.4 Inference Setup | 3 | MaskVCT-Spk weights `(0, 2.0, 0.5)` | `Amphion/models/tts/maskvct/cfg.py` | in_progress | Quantized linguistic features, no pitch conditioning |
| 3.4 Inference Setup | 3 | Accent conversion uses MaskVCT-Spk with `(0, 2.5, 0.5)` | `Amphion/models/tts/maskvct/cfg.py` | in_progress | Ambient noise caveat from the paper should be documented |
| 3.5 Metrics | 4 | Objective metrics WER/CER/S-SIM/FPC/UTMOS/Q-MOS | `docs/PAPER_IMPLEMENTATION_MAPPING.md` | pending | Needed for evaluation harness and result tracking |
| 3.5 Metrics | 4 | Accent metrics A-SIM and AS-MOS | `docs/PAPER_IMPLEMENTATION_MAPPING.md` | pending | Needed for the L2-ARCTIC evaluation notes |
| 4 Results | 4 | Report both objective and subjective quality trade-offs for MaskVCT-All/Spk | `docs/PAPER_IMPLEMENTATION_MAPPING.md` | pending | Useful for README summary once implementation exists |
| 5 Conclusion | 4 | Future work: learned quantized representation to reduce misreadings | `Amphion/models/tts/maskvct/README.md` | in_progress | Keep as a documented limitation, not an implementation requirement |

## Notes

- MaskGCT is the local reference implementation, but it is not a direct architectural match for MaskVCT.
- Wherever the paper is explicit, its requirements take priority over existing MaskGCT behavior.
- Wherever the paper is ambiguous or incomplete, the implementation should include `TODO(MaskVCT): Paper does not fully specify this behavior.` and use the smallest reasonable approximation.
