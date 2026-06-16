import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import librosa
import numpy as np
import safetensors
import torch
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

from ...codec.kmeans.repcodec_model import RepCodec

from .acoustic_codec import AcousticCodecBackend, build_acoustic_codec
from .cfg import MASKVCT_ALL, MASKVCT_SPK, MASKVCT_SPK_ACCENT, MaskVCTGuidance
from .model import MaskVCT
from .sylboost import SylBoostFeatureReader


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_namespace(v) for v in value]
    return value


def build_maskvct_model(cfg, device):
    cfg = _namespace(cfg)
    model = MaskVCT(
        num_quantizers=cfg.model.num_quantizers,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        ffn_size=cfg.model.ffn_size,
        codebook_size=cfg.model.codebook_size,
        ling_codebook_size=cfg.model.ling_codebook_size,
        cont_ling_dim=cfg.model.cont_ling_dim,
        pitch_dim=cfg.model.pitch_dim,
        dropout=cfg.model.dropout,
        layer_drop=cfg.model.layer_drop,
    )
    model.eval()
    model.to(device)
    return model


def build_semantic_model(device):
    semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    semantic_model.eval()
    semantic_model.to(device)
    stats_path = (
        Path(__file__).resolve().parents[1]
        / "maskgct"
        / "ckpt"
        / "wav2vec2bert_stats.pt"
    )
    stats = torch.load(stats_path, map_location=device)
    semantic_mean = stats["mean"].to(device)
    semantic_std = torch.sqrt(stats["var"]).to(device)
    return semantic_model, semantic_mean, semantic_std


def build_ling_tokenizer(cfg, device):
    cfg = _namespace(cfg)
    sylboost_cfg = getattr(cfg, "sylboost", None)
    if sylboost_cfg is not None and all(
        getattr(sylboost_cfg, key, None)
        for key in ("checkpoint", "kmeans", "agglom", "model_key")
    ):
        return SylBoostFeatureReader(
            sylboost_checkpoint=sylboost_cfg.checkpoint,
            kmeans_centroids_path=sylboost_cfg.kmeans,
            agglom_indices_path=sylboost_cfg.agglom,
            model_key=sylboost_cfg.model_key,
            device=device,
        )

    # Fallback used when SylBoost checkpoints are unavailable.
    tokenizer = RepCodec(cfg=cfg)
    tokenizer.eval()
    tokenizer.to(device)
    return tokenizer


def extract_pitch_50hz(waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    # TODO(MaskVCT): The paper specifies Praat pitch extraction at 50 Hz.
    # librosa.pyin is the closest lightweight local approximation.
    f0, _, _ = librosa.pyin(
        waveform.astype(np.float32),
        fmin=50.0,
        fmax=1100.0,
        sr=sr,
        frame_length=1024,
        hop_length=max(1, int(round(sr / 50))),
    )
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
    return f0


def _match_length(x, target_len: int, kind: str = "linear"):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        if x.shape[-1] == target_len or x.shape[0] == target_len:
            return x
        mode = "nearest" if kind == "nearest" else "linear"
        if x.dim() == 1:
            tensor = x[None, None, :].float()
            if mode == "nearest":
                y = torch.nn.functional.interpolate(tensor, size=target_len, mode=mode)
            else:
                y = torch.nn.functional.interpolate(
                    tensor, size=target_len, mode=mode, align_corners=False
                )
            return y[0, 0]
        if x.dim() == 2:
            tensor = x.transpose(0, 1)[None, :, :].float()
            if mode == "nearest":
                y = torch.nn.functional.interpolate(tensor, size=target_len, mode=mode)
            else:
                y = torch.nn.functional.interpolate(
                    tensor, size=target_len, mode=mode, align_corners=False
                )
            return y[0].transpose(0, 1)
        raise ValueError("unsupported tensor rank for length matching")
    arr = np.asarray(x)
    if arr.shape[0] == target_len:
        return arr
    mode = "nearest" if kind == "nearest" else "linear"
    tensor = torch.tensor(arr)
    if tensor.dim() == 1:
        tensor = tensor[None, None, :].float()
        y = (
            torch.nn.functional.interpolate(tensor, size=target_len, mode=mode)
            if mode == "nearest"
            else torch.nn.functional.interpolate(
                tensor, size=target_len, mode=mode, align_corners=False
            )
        )
        return y[0, 0].cpu().numpy()
    if tensor.dim() == 2:
        tensor = tensor.transpose(0, 1)[None, :, :].float()
        y = (
            torch.nn.functional.interpolate(tensor, size=target_len, mode=mode)
            if mode == "nearest"
            else torch.nn.functional.interpolate(
                tensor, size=target_len, mode=mode, align_corners=False
            )
        )
        return y[0].transpose(0, 1).cpu().numpy()
    raise ValueError("unsupported array rank for length matching")


class MaskVCTFeaturePipeline:
    def __init__(
        self,
        semantic_model,
        semantic_mean,
        semantic_std,
        ling_tokenizer,
        acoustic_codec: AcousticCodecBackend,
        device,
    ):
        self.processor = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )
        self.semantic_model = semantic_model
        self.semantic_mean = semantic_mean
        self.semantic_std = semantic_std
        self.ling_tokenizer = ling_tokenizer
        self.acoustic_codec = acoustic_codec
        self.device = device

    @torch.no_grad()
    def extract_w2vbert_features(self, speech_16k: np.ndarray):
        inputs = self.processor(speech_16k, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]
        feat = (feat - self.semantic_mean) / self.semantic_std
        return feat

    @torch.no_grad()
    def extract_ling_cont(self, speech_16k: np.ndarray):
        return self.extract_w2vbert_features(speech_16k)

    @torch.no_grad()
    def extract_ling_disc(self, speech_16k: np.ndarray):
        if hasattr(self.ling_tokenizer, "encode"):
            return self.ling_tokenizer.encode(speech_16k)
        feat = self.extract_w2vbert_features(speech_16k)
        codes, _ = self.ling_tokenizer.quantize(feat)
        return codes

    @torch.no_grad()
    def extract_acoustic_code(self, speech: np.ndarray):
        return self.acoustic_codec.encode_codes(speech)

    @torch.no_grad()
    def decode_acoustic_code(self, code: torch.Tensor) -> np.ndarray:
        return self.acoustic_codec.decode_codes(code)


class MaskVCTInferencePipeline:
    def __init__(
        self,
        model: MaskVCT,
        feature_pipeline: MaskVCTFeaturePipeline,
        device,
    ):
        self.model = model
        self.features = feature_pipeline
        self.device = device
        self.output_sample_rate = feature_pipeline.acoustic_codec.sample_rate

    def _mode_to_guidance(self, mode: str) -> MaskVCTGuidance:
        if mode == "all":
            return MASKVCT_ALL
        if mode == "spk":
            return MASKVCT_SPK
        if mode == "accent":
            return MASKVCT_SPK_ACCENT
        raise ValueError(f"unknown mode: {mode}")

    @torch.no_grad()
    def convert(
        self,
        source_wav_path: str,
        prompt_wav_path: str,
        mode: str = "all",
        n_timesteps: Optional[list] = None,
    ) -> np.ndarray:
        source_16k, _ = librosa.load(source_wav_path, sr=16000)
        prompt_16k, _ = librosa.load(prompt_wav_path, sr=16000)
        acoustic_sr = self.features.acoustic_codec.sample_rate
        source_acoustic, _ = librosa.load(source_wav_path, sr=acoustic_sr)
        prompt_acoustic, _ = librosa.load(prompt_wav_path, sr=acoustic_sr)

        ling_mode = "cont" if mode == "all" else "disc"
        if ling_mode == "cont":
            ling_source = self.features.extract_ling_cont(source_16k)
            ling_prompt = self.features.extract_ling_cont(prompt_16k)
        else:
            ling_source = self.features.extract_ling_disc(source_16k)
            ling_prompt = self.features.extract_ling_disc(prompt_16k)

        pitch_source = extract_pitch_50hz(source_16k)
        pitch_prompt = extract_pitch_50hz(prompt_16k)
        acoustic_source = self.features.extract_acoustic_code(source_acoustic)
        acoustic_prompt = self.features.extract_acoustic_code(prompt_acoustic)

        prompt_len = acoustic_prompt.shape[1]
        source_len = acoustic_source.shape[1]

        full = torch.zeros(
            1,
            prompt_len + source_len,
            acoustic_source.shape[-1],
            dtype=torch.long,
            device=self.device,
        )
        full[:, :prompt_len, :] = acoustic_prompt.to(self.device)

        if ling_mode == "cont":
            ling_prompt_t = torch.tensor(_match_length(ling_prompt, prompt_len), device=self.device).unsqueeze(0)
            ling_source_t = torch.tensor(_match_length(ling_source, source_len), device=self.device).unsqueeze(0)
        else:
            ling_prompt_t = torch.tensor(_match_length(ling_prompt, prompt_len), device=self.device).unsqueeze(0)
            ling_source_t = torch.tensor(_match_length(ling_source, source_len), device=self.device).unsqueeze(0)
        pitch_prompt_t = torch.tensor(_match_length(pitch_prompt, prompt_len), device=self.device).unsqueeze(0)
        pitch_source_t = torch.tensor(_match_length(pitch_source, source_len), device=self.device).unsqueeze(0)

        generated = self.model.reverse_diffusion(
            acoustic_tokens=full,
            prompt_len=prompt_len,
            ling_disc_prompt=ling_prompt_t if ling_mode == "disc" else None,
            ling_disc_source=ling_source_t if ling_mode == "disc" else None,
            ling_cont_prompt=ling_prompt_t if ling_mode == "cont" else None,
            ling_cont_source=ling_source_t if ling_mode == "cont" else None,
            pitch_prompt=pitch_prompt_t if mode == "all" else None,
            pitch_source=pitch_source_t if mode == "all" else None,
            guidance=self._mode_to_guidance(mode),
            n_timesteps=n_timesteps,
            mode="all" if mode == "all" else "spk_ling",
        )

        full[:, prompt_len:, :] = generated.to(self.device)
        return self.features.decode_acoustic_code(full[0].cpu())


def load_maskvct_bundle(
    cfg_path: str,
    device,
):
    cfg = load_config(cfg_path)
    semantic_model, semantic_mean, semantic_std = build_semantic_model(device)
    ling_tokenizer = build_ling_tokenizer(cfg["ling_tokenizer"], device)
    acoustic_codec = build_acoustic_codec(cfg["codec"], device)
    model = build_maskvct_model(cfg, device)
    return MaskVCTInferencePipeline(
        model=model,
        feature_pipeline=MaskVCTFeaturePipeline(
            semantic_model=semantic_model,
            semantic_mean=semantic_mean,
            semantic_std=semantic_std,
            ling_tokenizer=ling_tokenizer,
            acoustic_codec=acoustic_codec,
            device=device,
        ),
        device=device,
    )
