from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
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
class PairSample:
    prompt_path: Path
    source_path: Path
    dataset_name: str
    prompt_seconds: Optional[float] = None
    source_seconds: Optional[float] = None


@dataclass
class AudioSample:
    prompt: np.ndarray
    source: np.ndarray
    sample_rate: int
    dataset_name: str
    prompt_path: Path
    source_path: Path


@dataclass
class BatchSample:
    acoustic_tokens: torch.Tensor
    prompt_len: int
    ling_cont_prompt: torch.Tensor
    ling_cont_source: torch.Tensor
    ling_disc_prompt: torch.Tensor
    ling_disc_source: torch.Tensor
    pitch_prompt: torch.Tensor
    pitch_source: torch.Tensor
    prompt_audio: np.ndarray
    source_audio: np.ndarray
    sample_rate: int
    dataset_name: str
    prompt_name: str
    source_name: str


class DatasetSource:
    name: str
    weight: float

    def sample(self, rng: random.Random) -> PairSample:
        raise NotImplementedError


class LJSpeechSource(DatasetSource):
    def __init__(
        self,
        name: str,
        root: str,
        weight: float = 1.0,
        prompt_seconds: Optional[float] = None,
        source_seconds: Optional[float] = None,
    ):
        self.name = name
        self.root = Path(root)
        self.weight = float(weight)
        self.prompt_seconds = prompt_seconds
        self.source_seconds = source_seconds
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
            raise ValueError(f"{metadata_path} must reference at least two utterances")

    def sample(self, rng: random.Random) -> PairSample:
        prompt_idx = rng.randrange(len(self.entries))
        source_idx = rng.randrange(len(self.entries))
        if source_idx == prompt_idx:
            source_idx = (source_idx + 1) % len(self.entries)
        return PairSample(
            prompt_path=self.entries[prompt_idx],
            source_path=self.entries[source_idx],
            dataset_name=self.name,
            prompt_seconds=self.prompt_seconds,
            source_seconds=self.source_seconds,
        )


class PairManifestSource(DatasetSource):
    def __init__(
        self,
        name: str,
        manifest: str,
        weight: float = 1.0,
        prompt_seconds: Optional[float] = None,
        source_seconds: Optional[float] = None,
    ):
        self.name = name
        self.manifest = Path(manifest)
        self.weight = float(weight)
        self.prompt_seconds = prompt_seconds
        self.source_seconds = source_seconds
        if not self.manifest.exists():
            raise FileNotFoundError(self.manifest)
        self.entries = self._load_entries(self.manifest)
        if not self.entries:
            raise ValueError(f"{self.manifest} does not contain any pairs")

    @staticmethod
    def _load_entries(path: Path):
        if path.suffix.lower() == ".jsonl":
            rows = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    rows.append(row)
            return rows

        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as f:
                return list(csv.DictReader(f))

        raise ValueError(f"unsupported manifest format: {path.suffix}")

    def sample(self, rng: random.Random) -> PairSample:
        row = self.entries[rng.randrange(len(self.entries))]
        prompt_path = row["prompt_path"]
        source_path = row["source_path"]
        prompt_seconds = row.get("prompt_seconds", self.prompt_seconds)
        source_seconds = row.get("source_seconds", self.source_seconds)
        return PairSample(
            prompt_path=Path(prompt_path),
            source_path=Path(source_path),
            dataset_name=self.name,
            prompt_seconds=float(prompt_seconds) if prompt_seconds is not None else None,
            source_seconds=float(source_seconds) if source_seconds is not None else None,
        )


class DatasetMix:
    def __init__(self, sources: Iterable[DatasetSource]):
        self.sources = list(sources)
        if not self.sources:
            raise ValueError("at least one dataset source is required")
        self.weights = [source.weight for source in self.sources]

    def sample(self, rng: random.Random) -> PairSample:
        source = rng.choices(self.sources, weights=self.weights, k=1)[0]
        return source.sample(rng)


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

    def log_audio_pair(
        self,
        step: int,
        prompt_audio: np.ndarray,
        source_audio: np.ndarray,
        sample_rate: int,
        dataset_name: str,
        prompt_name: str,
        source_name: str,
    ):
        if self.tensorboard is not None:
            self.tensorboard.add_audio("samples/prompt", prompt_audio, step, sample_rate=sample_rate)
            self.tensorboard.add_audio("samples/source", source_audio, step, sample_rate=sample_rate)
        if self.wandb_run is not None:
            self.wandb_run.log(
                {
                    "samples/prompt": wandb.Audio(prompt_audio, sample_rate=sample_rate, caption=f"{dataset_name}:{prompt_name}"),
                    "samples/source": wandb.Audio(source_audio, sample_rate=sample_rate, caption=f"{dataset_name}:{source_name}"),
                },
                step=step,
            )

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


def _build_single_sample(bundle, pair: PairSample, default_prompt_seconds: float, default_source_seconds: float):
    prompt_sr = 16000
    acoustic_sr = bundle.features.acoustic_codec.sample_rate
    prompt_seconds = pair.prompt_seconds if pair.prompt_seconds is not None else default_prompt_seconds
    source_seconds = pair.source_seconds if pair.source_seconds is not None else default_source_seconds

    prompt_16k = _fit_duration(_load_audio(pair.prompt_path, prompt_sr), prompt_sr, prompt_seconds)
    source_16k = _fit_duration(_load_audio(pair.source_path, prompt_sr), prompt_sr, source_seconds)
    prompt_acoustic = _fit_duration(_load_audio(pair.prompt_path, acoustic_sr), acoustic_sr, prompt_seconds)
    source_acoustic = _fit_duration(_load_audio(pair.source_path, acoustic_sr), acoustic_sr, source_seconds)

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

    ling_disc_prompt = torch.as_tensor(
        _match_length(bundle.features.extract_ling_disc(prompt_16k), prompt_len, kind="nearest"),
        device=bundle.device,
    ).long().unsqueeze(0)
    ling_disc_source = torch.as_tensor(
        _match_length(bundle.features.extract_ling_disc(source_16k), source_len, kind="nearest"),
        device=bundle.device,
    ).long().unsqueeze(0)

    return {
        "acoustic_tokens": full,
        "prompt_len": prompt_len,
        "ling_cont_prompt": ling_prompt_t,
        "ling_cont_source": ling_source_t,
        "ling_disc_prompt": ling_disc_prompt,
        "ling_disc_source": ling_disc_source,
        "pitch_prompt": pitch_prompt_t,
        "pitch_source": pitch_source_t,
        "prompt_audio": prompt_16k,
        "source_audio": source_16k,
        "sample_rate": prompt_sr,
    }


def _build_batch(bundle, pairs: list[PairSample], default_prompt_seconds: float, default_source_seconds: float) -> BatchSample:
    samples = [_build_single_sample(bundle, pair, default_prompt_seconds, default_source_seconds) for pair in pairs]
    prompt_lens = {sample["prompt_len"] for sample in samples}
    total_lens = {sample["acoustic_tokens"].shape[1] for sample in samples}
    if len(prompt_lens) != 1:
        raise ValueError(f"all samples in a batch must share the same prompt_len, got {sorted(prompt_lens)}")
    if len(total_lens) != 1:
        raise ValueError(f"all samples in a batch must share the same total length, got {sorted(total_lens)}")

    return BatchSample(
        acoustic_tokens=torch.cat([sample["acoustic_tokens"] for sample in samples], dim=0),
        prompt_len=samples[0]["prompt_len"],
        ling_cont_prompt=torch.cat([sample["ling_cont_prompt"] for sample in samples], dim=0),
        ling_cont_source=torch.cat([sample["ling_cont_source"] for sample in samples], dim=0),
        ling_disc_prompt=torch.cat([sample["ling_disc_prompt"] for sample in samples], dim=0),
        ling_disc_source=torch.cat([sample["ling_disc_source"] for sample in samples], dim=0),
        pitch_prompt=torch.cat([sample["pitch_prompt"] for sample in samples], dim=0),
        pitch_source=torch.cat([sample["pitch_source"] for sample in samples], dim=0),
        prompt_audio=samples[0]["prompt_audio"],
        source_audio=samples[0]["source_audio"],
        sample_rate=samples[0]["sample_rate"],
        dataset_name=pairs[0].dataset_name,
        prompt_name=pairs[0].prompt_path.name,
        source_name=pairs[0].source_path.name,
    )


def _load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_sources(dataset_cfgs: list[dict]) -> DatasetMix:
    sources: list[DatasetSource] = []
    for dataset_cfg in dataset_cfgs:
        kind = dataset_cfg.get("kind", "ljspeech")
        name = dataset_cfg["name"]
        weight = float(dataset_cfg.get("weight", 1.0))
        prompt_seconds = dataset_cfg.get("prompt_seconds")
        source_seconds = dataset_cfg.get("source_seconds")
        if kind == "ljspeech":
            sources.append(
                LJSpeechSource(
                    name=name,
                    root=dataset_cfg["root"],
                    weight=weight,
                    prompt_seconds=prompt_seconds,
                    source_seconds=source_seconds,
                )
            )
        elif kind in {"manifest", "pair_manifest"}:
            sources.append(
                PairManifestSource(
                    name=name,
                    manifest=dataset_cfg["manifest"],
                    weight=weight,
                    prompt_seconds=prompt_seconds,
                    source_seconds=source_seconds,
                )
            )
        else:
            raise ValueError(f"unsupported dataset kind: {kind}")
    return DatasetMix(sources)


def _select_mode(step: int, mode_cycle: list[str]) -> str:
    if not mode_cycle:
        raise ValueError("mode_cycle cannot be empty")
    return mode_cycle[step % len(mode_cycle)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a JSON training config.")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--prompt-seconds", type=float, default=None)
    parser.add_argument("--source-seconds", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None, help='Override the training device, e.g. "cpu" or "cuda".')
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--loggers", nargs="+", default=None, choices=["tensorboard", "wandb"])
    parser.add_argument("--mode-cycle", nargs="+", default=None)
    args = parser.parse_args()

    config = _load_config(args.config)
    training_cfg = dict(config.get("training", {}))
    logging_cfg = dict(config.get("logging", {}))
    dataset_cfgs = list(config.get("datasets", []))

    if args.steps is not None:
        training_cfg["steps"] = args.steps
    if args.lr is not None:
        training_cfg["lr"] = args.lr
    if args.prompt_seconds is not None:
        training_cfg["prompt_seconds"] = args.prompt_seconds
    if args.source_seconds is not None:
        training_cfg["source_seconds"] = args.source_seconds
    if args.seed is not None:
        training_cfg["seed"] = args.seed
    if args.batch_size is not None:
        training_cfg["batch_size"] = args.batch_size
    if args.device is not None:
        training_cfg["device"] = args.device
    if args.output_dir is not None:
        training_cfg["output_dir"] = args.output_dir
    if args.mode_cycle is not None:
        training_cfg["mode_cycle"] = args.mode_cycle

    if args.log_dir is not None:
        logging_cfg["log_dir"] = args.log_dir
    if args.wandb_project is not None:
        logging_cfg["wandb_project"] = args.wandb_project
    if args.wandb_run_name is not None:
        logging_cfg["wandb_run_name"] = args.wandb_run_name
    if args.wandb_mode is not None:
        logging_cfg["wandb_mode"] = args.wandb_mode
    if args.loggers is not None:
        logging_cfg["loggers"] = args.loggers

    if not dataset_cfgs:
        raise ValueError("config must contain at least one dataset entry")

    seed = int(training_cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    device_name = training_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch build does not have CUDA enabled. "
            'Set "training.device" to "cpu" or install a CUDA-enabled PyTorch build.'
        )
    device = torch.device(device_name)
    bundle = load_maskvct_bundle(config.get("maskvct_config", "./Amphion/models/tts/maskvct/config/maskvct.json"), device)
    bundle.model.train()
    for module in [bundle.features.semantic_model, bundle.features.ling_tokenizer]:
        if hasattr(module, "eval"):
            module.eval()

    dataset_mix = _build_sources(dataset_cfgs)
    optimizer = torch.optim.AdamW(bundle.model.parameters(), lr=float(training_cfg.get("lr", 2e-4)))

    output_dir = Path(training_cfg.get("output_dir", "./checkpoints/maskvct_train"))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(logging_cfg.get("log_dir", "./runs/maskvct_train"))
    log_dir.mkdir(parents=True, exist_ok=True)

    run_name = logging_cfg.get("wandb_run_name") or output_dir.name
    logger = RunLogger(
        loggers=logging_cfg.get("loggers", ["tensorboard"]),
        log_dir=log_dir,
        project=logging_cfg.get("wandb_project", "maskvct"),
        run_name=run_name,
        config=config,
        wandb_mode=logging_cfg.get("wandb_mode", "offline"),
    )

    steps = int(training_cfg.get("steps", 250_000))
    batch_size = int(training_cfg.get("batch_size", 1))
    grad_accum_steps = int(training_cfg.get("gradient_accumulation_steps", 1))
    max_grad_norm = training_cfg.get("max_grad_norm")
    save_every = int(training_cfg.get("save_every", 1000))
    sample_every = int(training_cfg.get("sample_every", 0))
    sample_dir = Path(training_cfg.get("sample_dir", "./samples"))
    default_prompt_seconds = float(training_cfg.get("prompt_seconds", 3.0))
    default_source_seconds = float(training_cfg.get("source_seconds", 10.24))
    mode_cycle = list(training_cfg.get("mode_cycle", ["all", "spk_ling"]))
    sample_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}")
    print(f"datasets={[cfg['name'] for cfg in dataset_cfgs]}")
    print(f"dataset_weights={[cfg.get('weight', 1.0) for cfg in dataset_cfgs]}")
    print(f"batch_size={batch_size}")
    print(f"codec_backend={bundle.features.acoustic_codec.backend_name}")
    print(f"codec_sample_rate={bundle.output_sample_rate}")
    print(f"output_dir={output_dir}")
    print(f"log_dir={log_dir}")

    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    try:
        for step in range(steps):
            pairs = [dataset_mix.sample(rng) for _ in range(batch_size)]
            batch = _build_batch(bundle, pairs, default_prompt_seconds, default_source_seconds)
            mode = _select_mode(step, mode_cycle)

            if mode == "all":
                output = bundle.model(
                    acoustic_tokens=batch.acoustic_tokens,
                    prompt_len=batch.prompt_len,
                    ling_cont_prompt=batch.ling_cont_prompt,
                    ling_cont_source=batch.ling_cont_source,
                    pitch_prompt=batch.pitch_prompt,
                    pitch_source=batch.pitch_source,
                    mode="all",
                )
            elif mode in {"spk_ling", "spk"}:
                output = bundle.model(
                    acoustic_tokens=batch.acoustic_tokens,
                    prompt_len=batch.prompt_len,
                    ling_disc_prompt=batch.ling_disc_prompt,
                    ling_disc_source=batch.ling_disc_source,
                    mode="spk_ling",
                )
            else:
                raise ValueError(f"unknown mode in mode_cycle: {mode}")

            logits = output.logits.reshape(-1, output.logits.shape[-1])
            target = output.target.reshape(-1)
            loss = F.cross_entropy(logits, target) / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(bundle.model.parameters(), float(max_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            metrics = {
                "loss": float((loss * grad_accum_steps).detach().cpu()),
                "prompt_len": batch.prompt_len,
                "source_len": int(batch.acoustic_tokens.shape[1] - batch.prompt_len),
                "batch_size": batch_size,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "optimizer_step": optimizer_step,
            }
            logger.log(
                step,
                metrics=metrics,
                text={
                    "dataset": batch.dataset_name,
                    "mode": mode,
                    "pair/prompt": batch.prompt_name,
                    "pair/source": batch.source_name,
                },
            )

            if sample_every > 0 and (step % sample_every == 0):
                sample_prefix = f"{step:08d}_{batch.dataset_name}"
                prompt_sample_path = sample_dir / f"{sample_prefix}_prompt.wav"
                source_sample_path = sample_dir / f"{sample_prefix}_source.wav"
                sf.write(prompt_sample_path, batch.prompt_audio, batch.sample_rate)
                sf.write(source_sample_path, batch.source_audio, batch.sample_rate)
                logger.log_audio_pair(
                    step,
                    prompt_audio=batch.prompt_audio,
                    source_audio=batch.source_audio,
                    sample_rate=batch.sample_rate,
                    dataset_name=batch.dataset_name,
                    prompt_name=batch.prompt_name,
                    source_name=batch.source_name,
                )
                logger.log(
                    step,
                    metrics={"sample_step": 1},
                    text={
                        "samples/prompt_path": str(prompt_sample_path),
                        "samples/source_path": str(source_sample_path),
                    },
                )

            print(
                json.dumps(
                    {
                        "step": step,
                        **metrics,
                        "dataset": batch.dataset_name,
                        "pair": {
                            "prompt": str(batch.prompt_name),
                            "source": str(batch.source_name),
                        },
                    }
                )
            )

            if save_every > 0 and optimizer_step > 0 and optimizer_step % save_every == 0:
                ckpt_path = output_dir / f"maskvct_step_{optimizer_step:08d}.pt"
                torch.save(
                    {
                        "step": step,
                        "optimizer_step": optimizer_step,
                        "model": bundle.model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": config,
                    },
                    ckpt_path,
                )
        final_path = output_dir / "maskvct_final.pt"
        torch.save(
            {
                "step": steps,
                "optimizer_step": optimizer_step,
                "model": bundle.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
            },
            final_path,
        )
        print(f"saved_checkpoint={final_path}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
