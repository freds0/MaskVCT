# MaskVCT

This directory contains the first-pass implementation scaffold for the MaskVCT paper.

## Status

The repository now includes:

- a new `MaskVCT` model skeleton
- pitch helpers
- triple-CFG presets
- an inference wrapper
- a paper-to-code mapping document at `docs/PAPER_IMPLEMENTATION_MAPPING.md`

## Current approximations

The local Amphion tree now includes a SylBoost reader adapted from `SyllableLM`, but it is still checkpoint-driven. Configure the following fields in `config/maskvct.json` to activate it:

- `ling_tokenizer.sylboost.checkpoint`
- `ling_tokenizer.sylboost.kmeans`
- `ling_tokenizer.sylboost.agglom`
- `ling_tokenizer.sylboost.model_key`

If those fields are left empty, the pipeline falls back to the existing `RepCodec` approximation for the discrete linguistic path.

Current approximations that still remain:

- DAC integration is now available and supports the local 16 kHz, 24 kHz, and 44.1 kHz variants from `descript-audio-codec`
- Amphion codec remains available as a fallback backend in `config/maskvct.json`
- Use `codec.backend = "dac"` for the paper-aligned path or `codec.backend = "amphion"` to keep the legacy codec path
- For DAC, set `codec.dac.model_type` to `16khz`, `24khz`, or `44khz` and `codec.dac.n_quantizers` to the number of RVQ layers you want to use
- `librosa.pyin` as the local pitch extractor approximation

These are intentionally marked in code with `TODO(MaskVCT)` where the paper is more specific than the local codebase.
