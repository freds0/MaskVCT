# MaskVCT

This repository contains a paper-traceable implementation scaffold for MaskVCT, with the local Amphion codebase kept as the reference integration layer and a DAC backend added from `descript-audio-codec`.

## What Was Implemented

- A custom MaskVCT model in `Amphion/models/tts/maskvct/model.py`
- Triple CFG presets for the paper modes in `Amphion/models/tts/maskvct/cfg.py`
- Pitch embedding utilities in `Amphion/models/tts/maskvct/pitch.py`
- A SylBoost reader vendored from `SyllableLM` in `Amphion/models/tts/maskvct/sylboost.py`
- A dual acoustic backend in `Amphion/models/tts/maskvct/acoustic_codec.py`
  - `dac` backend using the local `descript-audio-codec` repository
  - `amphion` backend kept as fallback
- An inference pipeline in `Amphion/models/tts/maskvct/maskvct_utils.py`
- A CLI entrypoint in `Amphion/models/tts/maskvct/maskvct_inference.py`
- A smoke-test training script for LJSpeech in `scripts/train_maskvct_ljspeech_smoke.py`
- A paper-to-code mapping document in `docs/PAPER_IMPLEMENTATION_MAPPING.md`

## Main Adaptations

The paper is the primary specification, but the local repository needed a few engineering decisions:

- `SylBoost` is used through the public `SylBoostFeatureReader` API from `SyllableLM`, not as a new tokenization model.
- The paper specifies DAC at 16 kHz, but the local DAC repo also exposes 24 kHz and 44.1 kHz variants. The MaskVCT config now supports all three via `codec.backend = "dac"` and `codec.dac.model_type = "16khz" | "24khz" | "44khz"`.
- The Amphion codec path is preserved under `codec.backend = "amphion"` for compatibility and fallback.
- Pitch extraction is approximated with `librosa.pyin` at 50 Hz until a Praat-based extractor is wired in.
- The current training script is a smoke test, not the full paper training recipe. It uses LJSpeech pairs and repeats/pads clips because LJSpeech utterances are shorter than the paper's 3 s prompt + 10.24 s source setup.
- The semantic feature path still uses `facebook/w2v-bert-2.0` plus the local normalization stats from `Amphion/models/tts/maskgct/ckpt/wav2vec2bert_stats.pt`.

## Repository Layout

- `Amphion/models/tts/maskvct/`
  - model, sampling, conditioning, pitch, codec backends, inference helpers
- `descript-audio-codec/`
  - local DAC implementation with 16 kHz / 24 kHz / 44.1 kHz support
- `SyllableLM/`
  - SylBoost source used as the linguistic-unit extractor
- `checkpoints/`
  - optional local storage for downloaded auxiliary checkpoints

## Auxiliary Checkpoints

If you want local copies of the auxiliary checkpoints, save them in `./checkpoints/`.

Recommended files:

- DAC 16 kHz weights from `descript-audio-codec`
- DAC 24 kHz weights if you want to experiment with the non-paper backend
- SylBoost 8.33 Hz model, KMeans, and agglomerative clustering arrays from `SyllableLM`

The DAC repo supports these variants natively:

- `python3 -m dac download --model_type 16khz`
- `python3 -m dac download --model_type 24khz`
- `python3 -m dac download --model_type 44khz`

For MaskVCT, the paper-aligned configuration is:

- `codec.backend = "dac"`
- `codec.dac.model_type = "16khz"`
- `codec.dac.n_quantizers = 9`
- the default config points to the local checkpoints in `./checkpoints/`

Current local checkpoint files:

- `checkpoints/dac_16khz_weights.pth`
- `checkpoints/sylboost_833hz.pth`
- `checkpoints/sylboost_833hz_kmeans.npy`
- `checkpoints/sylboost_833hz_agglom.npy`

## Installation

Create or activate the conda environment and install the repository requirements:

```bash
conda activate maskvct
python -m pip install -r requirements.txt
```

The requirements file installs the CPU PyTorch wheel by default. If you have a CUDA environment, adjust the PyTorch installation line as needed.

I validated the environment with:

```bash
conda run -n maskvct python -m pip install -r requirements.txt
```

## LJSpeech Smoke Test

The LJSpeech dataset is single-speaker and its clips are only 1 to 10 seconds long, so the smoke test pairs two utterances and repeats/pads them to satisfy the prompt/source windows.

Run a short training test:

```bash
conda activate maskvct
python scripts/train_maskvct_ljspeech_smoke.py \
  --dataset /home/fred/Projetos/DATASETS/LJSpeech-1.1 \
  --steps 1
```

Paper-aligned windows are:

- prompt: 3.0 seconds
- source: 10.24 seconds

For a faster local smoke test, you can reduce both values, for example:

```bash
python scripts/train_maskvct_ljspeech_smoke.py \
  --dataset /home/fred/Projetos/DATASETS/LJSpeech-1.1 \
  --steps 1 \
  --prompt-seconds 1.0 \
  --source-seconds 2.0
```

You can enable logging with TensorBoard, Weights & Biases, or both:

```bash
python scripts/train_maskvct_ljspeech_smoke.py \
  --dataset /home/fred/Projetos/DATASETS/LJSpeech-1.1 \
  --steps 1 \
  --loggers tensorboard wandb \
  --wandb-mode offline
```

Useful logging flags:

- `--loggers tensorboard`
- `--loggers wandb`
- `--loggers tensorboard wandb`
- `--log-dir ./runs/maskvct_ljspeech_smoke`
- `--wandb-project maskvct`
- `--wandb-run-name <name>`
- `--wandb-mode offline|online|disabled`

Validated smoke test output:

- `device=cpu`
- `dataset_size=13100`
- `codec_backend=dac`
- `codec_sample_rate=16000`
- `step=0`
- `loss=7.1370110511779785`
- `prompt_len=25`
- `source_len=50`

## Generic Training Script

For multi-dataset training, use the config-driven trainer:

```bash
python scripts/train_maskvct.py --config configs/train_maskvct.example.json
```

The config file separates:

- `maskvct_config`: path to the model/config bundle
- `training`: learning rate, batch size, steps, prompt/source lengths, accumulation, checkpointing
- `logging`: TensorBoard and W&B settings
- `datasets`: one or more dataset entries with per-dataset folders and weights

Supported dataset kinds:

- `ljspeech`: expects `<root>/metadata.csv` and `<root>/wavs/*.wav`
- `manifest` or `pair_manifest`: expects a JSONL or CSV manifest with `prompt_path` and `source_path`

Example multi-dataset structure:

```json
{
  "datasets": [
    {"name": "ljspeech", "kind": "ljspeech", "root": "/data/LJSpeech-1.1", "weight": 1.0},
    {"name": "custom_pairs", "kind": "manifest", "manifest": "/data/custom_pairs.jsonl", "weight": 0.5}
  ]
}
```

Useful overrides:

- `--batch-size 4`
- `--device cuda` or `--device cpu`
- `--steps 1000`
- `--lr 2e-4`
- `--prompt-seconds 3.0`
- `--source-seconds 10.24`
- `--loggers tensorboard wandb`
- `--mode-cycle all spk_ling`

Batching note:

- `batch_size` is a real minibatch size now, not gradient accumulation.
- All items in the same minibatch must resolve to the same prompt/source window lengths.
- If you mix datasets with different prompt/source lengths, keep their window parameters aligned or use separate runs.

Device note:

- The example config defaults to `training.device = "cpu"` so it works on CPU-only environments.
- To force GPU, set `training.device = "cuda"` and use a CUDA-enabled PyTorch build.
- You can also override from the CLI with `--device cuda`.

Periodic sample export is configured in the JSON file:

- `training.sample_every`: export prompt/source audio every N steps
- `training.sample_dir`: directory where the exported `.wav` files are written

When enabled, the trainer:

- saves `prompt` and `source` as audio in TensorBoard
- logs `wandb.Audio` entries when W&B is enabled
- writes a pair of `.wav` files per sample step under `samples/`

## Inference

The current CLI is:

```bash
python -m Amphion.models.tts.maskvct.maskvct_inference \
  --config ./Amphion/models/tts/maskvct/config/maskvct.json \
  --source /path/to/source.wav \
  --prompt /path/to/prompt.wav \
  --output generated_maskvct.wav \
  --mode all
```

Supported modes:

- `all` for MaskVCT-All
- `spk` for MaskVCT-Spk
- `accent` for the accent-conversion preset

The output sample rate follows the selected acoustic backend:

- DAC 16 kHz -> output written at 16000 Hz
- DAC 24 kHz -> output written at 24000 Hz
- DAC 44.1 kHz -> output written at 44100 Hz
- Amphion fallback -> output written at 24000 Hz

## Paper-to-Code Notes

See `docs/PAPER_IMPLEMENTATION_MAPPING.md` for the current traceability table.

Notable current approximations:

- pitch extraction still uses `librosa.pyin`
- the training script is a minimal smoke test
- the paper's full dataset mix and evaluation harness are not yet implemented

## What Matches the Paper vs. What Is an Approximation

This section separates the parts that follow the paper closely from the parts that are engineering approximations in the current repository.

### Implemented Literally From the Paper

- `SylBoost 8.33 Hz` as the discrete linguistic representation path.
- `DAC 16 kHz` as the default paper-aligned acoustic codec backend.
- 9 DAC codebooks for acoustic tokenization.
- 50 Hz pitch extraction target.
- log-scale sinusoidal pitch embedding.
- triple CFG modes:
  - `MaskVCT-All`
  - `MaskVCT-Spk`
  - `MaskVCT-Accent`
- masked codebook prediction with iterative unmasking.
- separate heads per acoustic codebook layer.
- 3-second prompt and 10.24-second source windows in the smoke test setup.

### Engineering Approximations

- `facebook/w2v-bert-2.0` is used as the continuous linguistic feature source because the paper does not specify a concrete continuous encoder implementation.
- `librosa.pyin` is used for pitch extraction instead of a Praat binding in the code path.
- The full paper training recipe is not yet implemented; the current trainer is a smoke test on LJSpeech.
- The paper does not define a public API for SylBoost extraction, so the code vendors `SylBoostFeatureReader` from `SyllableLM` and wraps it for MaskVCT.
- The paper does not describe how to handle invalid mask-layer sampling weights for the 9-codebook configuration; the code clamps them to keep training stable.
- The DAC backend preserves the Amphion codec path as a fallback for compatibility, even though the paper only specifies DAC.

## Practical Notes

- Run commands from the repository root so the local `Amphion` package is importable.
- The first run will download `facebook/w2v-bert-2.0` through Hugging Face unless it is already cached.
- If the DAC backend cannot be initialized, the code falls back to the Amphion codec and emits a warning.
