"""Acoustic codec backends for MaskVCT.

This module keeps the local Amphion codec path available while adding a DAC
backend that can load the local `descript-audio-codec` repository.
"""

from __future__ import annotations

import sys
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from ...codec.amphion_codec.codec import CodecDecoder, CodecEncoder


def _namespace(value: Any):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_namespace(v) for v in value]
    return value


def _default_dac_repo_path() -> Path:
    return Path(__file__).resolve().parents[4] / "descript-audio-codec"


class AcousticCodecBackend(ABC):
    backend_name: str
    sample_rate: int
    n_quantizers: int

    @abstractmethod
    def encode_codes(self, waveform: np.ndarray) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def decode_codes(self, codes: torch.Tensor) -> np.ndarray:
        raise NotImplementedError


class AmphionAcousticCodec(AcousticCodecBackend):
    backend_name = "amphion"

    def __init__(self, cfg, device):
        cfg = _namespace(cfg)
        self.device = device
        self.sample_rate = int(getattr(cfg, "sample_rate", 24000))
        self.n_quantizers = int(getattr(cfg.decoder, "num_quantizers", 9))

        self.codec_encoder = CodecEncoder(cfg=cfg.encoder)
        self.codec_decoder = CodecDecoder(cfg=cfg.decoder)
        self.codec_encoder.eval()
        self.codec_decoder.eval()
        self.codec_encoder.to(device)
        self.codec_decoder.to(device)

    @torch.no_grad()
    def encode_codes(self, waveform: np.ndarray) -> torch.Tensor:
        audio = torch.tensor(waveform, dtype=torch.float32, device=self.device)
        audio = audio.unsqueeze(0).unsqueeze(0)
        vq_emb = self.codec_encoder(audio)
        _, vq, _, _, _ = self.codec_decoder.quantizer(vq_emb)
        return vq.permute(1, 2, 0).contiguous()

    @torch.no_grad()
    def decode_codes(self, codes: torch.Tensor) -> np.ndarray:
        if codes.dim() == 3:
            codes = codes[0]
        codes = codes.long().to(self.device)
        vq_emb = self.codec_decoder.vq2emb(
            codes.permute(1, 0).unsqueeze(1), n_quantizers=codes.shape[-1]
        )
        recovered_audio = self.codec_decoder(vq_emb)
        return recovered_audio[0][0].detach().cpu().numpy()


class DACAcousticCodec(AcousticCodecBackend):
    backend_name = "dac"

    def __init__(self, cfg, device):
        cfg = _namespace(cfg)
        self.device = device

        repo_path = Path(getattr(cfg, "repo_path", "") or _default_dac_repo_path())
        if repo_path.exists() and str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        try:
            import dac  # type: ignore
            from dac.utils import load_model  # type: ignore
        except ImportError as exc:  # pragma: no cover - import path depends on local env
            raise ImportError(
                "DAC backend requested but the local descript-audio-codec package "
                "could not be imported."
            ) from exc

        self._dac = dac
        self._load_model = load_model
        self.model_type = str(getattr(cfg, "model_type", "16khz"))
        self.model_bitrate = str(getattr(cfg, "model_bitrate", "8kbps"))
        self.model_tag = str(getattr(cfg, "model_tag", "latest"))
        self.weights_path = str(getattr(cfg, "weights_path", ""))
        self.n_quantizers = int(getattr(cfg, "n_quantizers", 9))

        default_sample_rates = {
            "16khz": 16000,
            "24khz": 24000,
            "44khz": 44100,
        }
        self.sample_rate = int(
            getattr(
                cfg,
                "sample_rate",
                default_sample_rates.get(self.model_type, 16000),
            )
        )

        self.model = self._load_model(
            model_type=self.model_type,
            model_bitrate=self.model_bitrate,
            tag=self.model_tag,
            load_path=self.weights_path or None,
        )
        self.model.to(device)
        self.model.eval()

        if self.n_quantizers > self.model.n_codebooks:
            raise ValueError(
                f"DAC backend requested {self.n_quantizers} quantizers, "
                f"but the selected model only exposes {self.model.n_codebooks}."
            )

    @torch.no_grad()
    def encode_codes(self, waveform: np.ndarray) -> torch.Tensor:
        audio = torch.tensor(waveform, dtype=torch.float32, device=self.device)
        audio = audio.unsqueeze(0).unsqueeze(0)
        audio = self.model.preprocess(audio, self.sample_rate)
        _, codes, _, _, _ = self.model.encode(audio, n_quantizers=self.n_quantizers)
        return codes.permute(0, 2, 1).contiguous()

    @torch.no_grad()
    def decode_codes(self, codes: torch.Tensor) -> np.ndarray:
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        if codes.shape[1] == self.n_quantizers and codes.shape[-1] != self.n_quantizers:
            # Already in B x N x T format.
            codes_bt = codes.long()
        else:
            # Convert from B x T x N to B x N x T.
            codes_bt = codes.permute(0, 2, 1).long()
        codes_bt = codes_bt.to(self.device)
        z_q, _, _ = self.model.quantizer.from_codes(codes_bt)
        recovered_audio = self.model.decode(z_q)
        return recovered_audio[0, 0].detach().cpu().numpy()


def build_acoustic_codec(cfg, device):
    cfg = _namespace(cfg)
    backend = str(getattr(cfg, "backend", "amphion")).lower()

    if backend == "dac":
        try:
            return DACAcousticCodec(cfg.dac, device)
        except Exception as exc:  # pragma: no cover - exercised when DAC deps/checkpoints are missing
            warnings.warn(
                f"DAC backend could not be initialized, falling back to Amphion: {exc}",
                RuntimeWarning,
            )

    if backend != "amphion":
        raise ValueError(f"Unknown acoustic codec backend: {backend}")

    return AmphionAcousticCodec(cfg, device)
