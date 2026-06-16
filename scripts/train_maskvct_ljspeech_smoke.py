from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency.
    wandb = None

REPO_ROOT = Path(__file__).resolve().parents[1]
AMPHION_ROOT = REPO_ROOT / "Amphion"
for path in (REPO_ROOT, AMPHION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Amphion.models.tts.maskvct.maskvct_utils import (
    _match_length,
    extract_pitch_50hz,
    load_maskvct_bundle,
)


@dataclass
class LJSpeechPair:
    prompt_path: Path
    source_path: Path


class LJSpeechPairDataset:
    def __init__(self, root: str, seed: int = 0):
        self.root = Path(root)
        metadata_path = self.root / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)

        self.wav_dir = self.root / "wavs"
        self.entries = []
        with metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                wav_id = line.split("|", 1)[0]
                self.entries.append(self.wav_dir / f"{wav_id}.wav")

        if len(self.entries) < 2:
            raise ValueError("LJSpeech metadata must contain at least two utterances")

        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int) -> LJSpeechPair:
        prompt_idx = idx % len(self.entries)
        source_idx = self.rng.randrange(len(self.entries))
        if source_idx == prompt_idx:
            source_idx = (source_idx + 1) % len(self.entries)
        return LJSpeechPair(
            prompt_path=self.entries[prompt_idx],
            source_path=self.entries[source_idx],
        )


class RunLogger:
    def __init__(
        self,
        loggers,
        log_dir: Path,
        project: str,
        run_name: str,
        config: dict,
        wandb_mode: str,
    ):
        self.tensorboard = SummaryWriter(log_dir=str(log_dir)) if "tensorboard" in loggers else None
        self.wandb_run = None
        if "wandb" in loggers:
            if wandb is None:
                raise ImportError("wandb is not installed but was requested via --loggers")
            self.wandb_run = wandb.init(
                project=project,
                name=run_name,
                dir=str(log_dir),
                mode=wandb_mode,
                config=config,
                reinit="finish_previous",
            )

    def log(self, step: int, metrics: dict, text: dict | None = None):
        if self.tensorboard is not None:
            for key, value in metrics.items():
                self.tensorboard.add_scalar(key, value, step)
            if text:
                for key, value in text.items():
                    self.tensorboard.add_text(key, value, step)
        if self.wandb_run is not None:
            payload = dict(metrics)
            if text:
                payload.update(text)
            self.wandb_run.log(payload, step=step)

    def close(self):
        if self.tensorboard is not None:
            self.tensorboard.flush()
            self.tensorboard.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()


def _load_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return audio.astype(np.float32)


def _fit_duration(audio: np.ndarray, sample_rate: int, seconds: float) -> np.ndarray:
    target = int(round(sample_rate * seconds))
    if target <= 0:
        raise ValueError("seconds must be positive")
    if audio.shape[0] == target:
        return audio
    if audio.shape[0] > target:
        return audio[:target]
    repeat = int(np.ceil(target / max(1, audio.shape[0])))
    tiled = np.tile(audio, repeat)
    return tiled[:target]


def _build_sample(bundle, pair: LJSpeechPair, prompt_seconds: float, source_seconds: float):
    prompt_sr = 16000
    acoustic_sr = bundle.features.acoustic_codec.sample_rate

    prompt_16k = _fit_duration(_load_audio(pair.prompt_path, prompt_sr), prompt_sr, prompt_seconds)
    source_16k = _fit_duration(_load_audio(pair.source_path, prompt_sr), prompt_sr, source_seconds)
    prompt_acoustic = _fit_duration(_load_audio(pair.prompt_path, acoustic_sr), acoustic_sr, prompt_seconds)
    source_acoustic = _fit_duration(_load_audio(pair.source_path, acoustic_sr), acoustic_sr, source_seconds)

    ling_mode = "cont"
    ling_prompt = bundle.features.extract_ling_cont(prompt_16k)[0]
    ling_source = bundle.features.extract_ling_cont(source_16k)[0]
    pitch_prompt = extract_pitch_50hz(prompt_16k, sr=prompt_sr)
    pitch_source = extract_pitch_50hz(source_16k, sr=prompt_sr)
    acoustic_prompt = bundle.features.extract_acoustic_code(prompt_acoustic)
    acoustic_source = bundle.features.extract_acoustic_code(source_acoustic)

    prompt_len = int(acoustic_prompt.shape[1])
    source_len = int(acoustic_source.shape[1])

    full = torch.zeros(
        1,
        prompt_len + source_len,
        acoustic_source.shape[-1],
        dtype=torch.long,
        device=bundle.device,
    )
    full[:, :prompt_len, :] = acoustic_prompt.to(bundle.device)
    full[:, prompt_len:, :] = acoustic_source.to(bundle.device)

    ling_prompt_t = torch.as_tensor(_match_length(ling_prompt, prompt_len), device=bundle.device).unsqueeze(0)
    ling_source_t = torch.as_tensor(_match_length(ling_source, source_len), device=bundle.device).unsqueeze(0)
    pitch_prompt_t = torch.as_tensor(_match_length(pitch_prompt, prompt_len), device=bundle.device).unsqueeze(0)
    pitch_source_t = torch.as_tensor(_match_length(pitch_source, source_len), device=bundle.device).unsqueeze(0)

    return {
        "acoustic_tokens": full,
        "prompt_len": prompt_len,
        "ling_cont_prompt": ling_prompt_t,
        "ling_cont_source": ling_source_t,
        "pitch_prompt": pitch_prompt_t,
        "pitch_source": pitch_source_t,
        "mode": ling_mode,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./Amphion/models/tts/maskvct/config/maskvct.json")
    parser.add_argument("--dataset", default="/home/fred/Projetos/DATASETS/LJSpeech-1.1")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--prompt-seconds", type=float, default=3.0)
    parser.add_argument("--source-seconds", type=float, default=10.24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--loggers",
        nargs="+",
        default=["tensorboard"],
        choices=["tensorboard", "wandb"],
        help="Enable one or more experiment loggers.",
    )
    parser.add_argument("--log-dir", default="./runs/maskvct_ljspeech_smoke")
    parser.add_argument("--wandb-project", default="maskvct")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-mode",
        default="offline",
        choices=["online", "offline", "disabled"],
        help="W&B run mode. Offline avoids requiring an API key.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_maskvct_bundle(args.config, device)
    bundle.model.train()
    for module in [bundle.features.semantic_model, bundle.features.ling_tokenizer]:
        if hasattr(module, "eval"):
            module.eval()

    dataset = LJSpeechPairDataset(args.dataset, seed=args.seed)
    optimizer = torch.optim.AdamW(bundle.model.parameters(), lr=args.lr)
    run_name = args.wandb_run_name or f"maskvct-smoke-seed{args.seed}"
    logger = RunLogger(
        loggers=args.loggers,
        log_dir=Path(args.log_dir),
        project=args.wandb_project,
        run_name=run_name,
        config=vars(args),
        wandb_mode=args.wandb_mode,
    )

    print(f"device={device}")
    print(f"dataset_size={len(dataset)}")
    print(f"codec_backend={bundle.features.acoustic_codec.backend_name}")
    print(f"codec_sample_rate={bundle.output_sample_rate}")

    try:
        for step in range(args.steps):
            pair = dataset[step]
            batch = _build_sample(bundle, pair, args.prompt_seconds, args.source_seconds)

            if step % 2 == 0:
                output = bundle.model(
                    acoustic_tokens=batch["acoustic_tokens"],
                    prompt_len=batch["prompt_len"],
                    ling_cont_prompt=batch["ling_cont_prompt"],
                    ling_cont_source=batch["ling_cont_source"],
                    pitch_prompt=batch["pitch_prompt"],
                    pitch_source=batch["pitch_source"],
                    mode="all",
                )
                mode = "all"
            else:
                output = bundle.model(
                    acoustic_tokens=batch["acoustic_tokens"],
                    prompt_len=batch["prompt_len"],
                    ling_disc_prompt=bundle.features.extract_ling_disc(
                        _fit_duration(_load_audio(pair.prompt_path, 16000), 16000, args.prompt_seconds)
                    ).to(device).unsqueeze(0),
                    ling_disc_source=bundle.features.extract_ling_disc(
                        _fit_duration(_load_audio(pair.source_path, 16000), 16000, args.source_seconds)
                    ).to(device).unsqueeze(0),
                    mode="spk_ling",
                )
                mode = "spk_ling"

            logits = output.logits.reshape(-1, output.logits.shape[-1])
            target = output.target.reshape(-1)
            loss = F.cross_entropy(logits, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            metrics = {
                "loss": float(loss.detach().cpu()),
                "prompt_len": batch["prompt_len"],
                "source_len": int(batch["acoustic_tokens"].shape[1] - batch["prompt_len"]),
            }
            logger.log(
                step,
                metrics=metrics,
                text={
                    "pair/prompt": pair.prompt_path.name,
                    "pair/source": pair.source_path.name,
                    "mode": mode,
                },
            )
            print(
                json.dumps(
                    {
                        "step": step,
                        **metrics,
                        "pair": {
                            "prompt": str(pair.prompt_path.name),
                            "source": str(pair.source_path.name),
                        },
                    }
                )
            )
    finally:
        logger.close()


if __name__ == "__main__":
    main()
