"""
SylBoost integration for MaskVCT.

This module vendors the SylBoost extraction logic from the local SyllableLM
repository and adapts it for MaskVCT use. The original code is conceptually
the same as SyllableLM/extract_units.py, but here it is:

- device-aware instead of hard-coded to CUDA
- usable as a reusable tokenizer/reader
- able to emit syllable-level token ids for conditioning
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def _ensure_syllablelm_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    syllablelm_root = repo_root / "SyllableLM"
    if syllablelm_root.exists() and str(syllablelm_root) not in sys.path:
        sys.path.insert(0, str(syllablelm_root))


_ensure_syllablelm_on_path()

from syllablelm.data2vec.data.modality import Modality  # type: ignore  # noqa: E402
from syllablelm.data2vec.models.data2vec2 import Data2VecMultiModel  # type: ignore  # noqa: E402


THRESHOLD = 1 / 0.10 / 50.0
FULL_MODELS_DICT = {
    "8.33Hz": {"delta": 0.0033, "quantile": 0.75},
    "6.25Hz": {"delta": 0.0028, "quantile": 0.75},
    "5.0Hz": {"delta": 0.0019, "quantile": 0.75},
}

d2v2_config = SimpleNamespace(
    **{
        "_name": "data2vec_multi",
        "loss_beta": 0.0,
        "loss_scale": None,
        "depth": 8,
        "start_drop_path_rate": 0.0,
        "end_drop_path_rate": 0.0,
        "num_heads": 12,
        "norm_eps": 1e-05,
        "norm_affine": True,
        "encoder_dropout": 0.1,
        "post_mlp_drop": 0.1,
        "attention_dropout": 0.1,
        "activation_dropout": 0.0,
        "dropout_input": 0.0,
        "layerdrop": 0.05,
        "embed_dim": 768,
        "mlp_ratio": 4.0,
        "layer_norm_first": False,
        "average_top_k_layers": 8,
        "end_of_block_targets": False,
        "clone_batch": 8,
        "layer_norm_target_layer": False,
        "batch_norm_target_layer": False,
        "instance_norm_target_layer": True,
        "instance_norm_targets": False,
        "layer_norm_targets": False,
        "ema_decay": 0.999,
        "ema_same_dtype": True,
        "log_norms": True,
        "ema_end_decay": 0.99999,
        "ema_anneal_end_step": 75000,
        "ema_encoder_only": False,
        "max_update": 400000,
        "modalities": SimpleNamespace(
            **{
                "_name": None,
                "audio": SimpleNamespace(
                    **{
                        "type": Modality.AUDIO,
                        "prenet_depth": 4,
                        "prenet_layerdrop": 0.05,
                        "prenet_dropout": 0.1,
                        "start_drop_path_rate": 0.0,
                        "end_drop_path_rate": 0.0,
                        "num_extra_tokens": 0,
                        "init_extra_token_zero": True,
                        "mask_noise_std": 0.01,
                        "mask_prob_min": None,
                        "mask_prob": 0.5,
                        "inverse_mask": False,
                        "mask_prob_adjust": 0.05,
                        "keep_masked_pct": 0.0,
                        "mask_length": 5,
                        "add_masks": False,
                        "remove_masks": False,
                        "mask_dropout": 0.0,
                        "encoder_zero_mask": True,
                        "mask_channel_prob": 0.0,
                        "mask_channel_length": 64,
                        "ema_local_encoder": False,
                        "local_grad_mult": 1.0,
                        "use_alibi_encoder": True,
                        "alibi_scale": 1.0,
                        "learned_alibi": False,
                        "alibi_max_pos": None,
                        "learned_alibi_scale": True,
                        "learned_alibi_scale_per_head": True,
                        "learned_alibi_scale_per_layer": False,
                        "num_alibi_heads": 12,
                        "model_depth": 8,
                        "decoder": SimpleNamespace(
                            **{
                                "decoder_dim": 384,
                                "decoder_groups": 16,
                                "decoder_kernel": 7,
                                "decoder_layers": 4,
                                "input_dropout": 0.1,
                                "add_positions_masked": False,
                                "add_positions_all": False,
                                "decoder_residual": True,
                                "projection_layers": 1,
                                "projection_ratio": 2.0,
                                "channel_mult": [1, 0.5, 0.25, 0.25, 0.25],
                                "decoder_transformer_layers": 4,
                            }
                        ),
                        "extractor_mode": "layer_norm",
                        "feature_encoder_spec": "[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
                        "conv_pos_width": 95,
                        "conv_pos_groups": 16,
                        "conv_pos_depth": 5,
                        "conv_pos_pre_ln": False,
                    }
                ),
            }
        ),
        "shared_decoder": None,
        "min_target_var": 0.1,
        "min_pred_var": 0.01,
        "supported_modality": Modality.AUDIO,
        "mae_init": False,
        "seed": 1,
        "skip_ema": False,
        "cls_loss": 0.0,
        "recon_loss": 0.0,
        "d2v_loss": 1.0,
        "decoder_group": False,
    }
)


class ApplyKmeans:
    def __init__(self, km_path: str, device: torch.device):
        self.cluster_centers = np.load(km_path)
        self.C_np = self.cluster_centers.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)
        self.C = torch.from_numpy(self.C_np).to(device)
        self.Cnorm = torch.from_numpy(self.Cnorm_np).to(device)

    def __call__(self, x: torch.Tensor):
        dist = x.pow(2).sum(-1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
        return dist.argmin(dim=-1).cpu()


@torch.inference_mode()
def efficient_extraction_dp_helper(x, threshold=THRESHOLD, s=35, min_hop=3):
    b, n, d = x.shape
    dists = x.new_full((b, s + 1, n + s), 16384)
    rolled = torch.stack([torch.roll(x, shifts=-i, dims=-2) for i in range(s)]).transpose(0, 1)
    rolled_prepend = x[:, :s].unsqueeze(2).repeat(1, 1, s - 1, 1)
    arranged = torch.cat([rolled_prepend, rolled], dim=2)
    len_indices = torch.arange(s, device=x.device) + 1
    dots = arranged.pow(2).mean(dim=-1).cumsum(dim=-2)
    middle = -1 / len_indices.view(1, -1, 1) * arranged.cumsum(dim=-3).pow(2).mean(dim=-1)
    outs = dots + middle
    outs = torch.cat([outs[:, i : i + 1].roll(shifts=-(s - i - 1), dims=2) for i in range(s)], dim=1)
    dists[:, 1:, s:] = outs[:, :, : -(s - 1)]
    dists += dists.new_full(dists.shape, 16384).tril(s - 2)
    dists = dists.clamp(max=16384)
    m = int(threshold * n)
    total_dists = x.new_full((b, n + 2), 16384)
    total_dists[:, 0] = 0
    back = x.new_zeros((b, n + 1, m + 1), dtype=int)
    magic_mask = torch.tensor(
        [[(j + 1 - k if j + 1 >= k else n + 1) for j in range(n)] for k in range(min_hop, s + 1)],
        device=x.device,
    ).unsqueeze(0).expand(b, s + 1 - min_hop, n)

    for j in range(1, m + 1):
        cur_min = torch.min(
            total_dists.unsqueeze(1).expand(b, s + 1 - min_hop, n + 2).gather(2, magic_mask)
            + dists[:, min_hop:, s : n + s],
            dim=1,
        )
        total_dists[:, 1:-1] = cur_min.values
        back[:, 1 : 1 + n, j] = cur_min.indices + min_hop

    return dists, back


def get_quantile_borders_helper(dists, back, n=None, s=None, num_units=None, delta=None, quantile=None):
    min_, max_ = num_units // 3, num_units
    best_m = min_

    while min_ <= max_:
        mid_ = (min_ + max_) // 2
        q = n
        j = mid_
        costs = []
        while q > 0:
            costs.append(dists[back[q, j], q - 1 + s] / back[q, j])
            q = q - back[q, j]
            j = j - 1
        quantile_cost = np.quantile(costs, quantile)

        if quantile_cost > delta:
            min_ = mid_ + 1
            best_m = mid_
        else:
            max_ = mid_ - 1

    q = n
    j = best_m
    borders = [q]
    while q > 0:
        q = q - back[q, j]
        borders.append(q)
        j = j - 1
    borders.reverse()
    return borders


@torch.no_grad()
def efficient_extraction(embeddings, threshold=THRESHOLD, s=35, min_hop=3, deltas=None, quantiles=None):
    b, n, d = embeddings.shape
    x = embeddings.float()
    m = int(threshold * n)
    s = min(n, s)
    dists, back = efficient_extraction_dp_helper(x, threshold=threshold, s=s, min_hop=min_hop)
    back = back.cpu().numpy()
    dists = dists.cpu().numpy()
    batch_outs = [
        [
            get_quantile_borders_helper(d_, b_, n=n, s=s, num_units=m, delta=delta, quantile=quantile)
            for d_, b_ in zip(dists, back)
        ]
        for delta, quantile in zip(deltas, quantiles)
    ]
    return batch_outs


class SylBoostFeatureReader:
    def __init__(
        self,
        sylboost_checkpoint: str,
        kmeans_centroids_path: str,
        agglom_indices_path: str,
        model_key: str,
        device: torch.device,
    ):
        self.device = device
        d2v2_model = Data2VecMultiModel(d2v2_config, [Modality.AUDIO])
        d2v2_model = d2v2_model.to(device).eval()
        if device.type == "cuda":
            d2v2_model = d2v2_model.half()
        # PyTorch 2.6 defaults to weights_only=True, but this checkpoint stores
        # the original bundle dict with numpy objects.
        state_dict = torch.load(
            sylboost_checkpoint,
            map_location=device,
            weights_only=False,
        )
        d2v2_model.load_state_dict(
            {k[len("model.") :]: v for k, v in state_dict["model_seg"].items()}
        )
        self.d2v2_model = d2v2_model
        self.kmeans_centroids = ApplyKmeans(kmeans_centroids_path, device=device)
        self.agglom = np.load(agglom_indices_path)
        self.model_key = model_key
        if model_key not in FULL_MODELS_DICT:
            raise ValueError(f"unknown SylBoost model key: {model_key}")
        self.delta = FULL_MODELS_DICT[model_key]["delta"]
        self.quantile = FULL_MODELS_DICT[model_key]["quantile"]

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        if self.device.type == "cuda":
            x = x.half()
        features = self.d2v2_model(
            x,
            mode=None,
            mask=False,
            features_only=True,
            remove_extra_tokens=True,
            out_layer=-2,
        )["x"]
        result = {"features": features, "clusters_with_times": []}
        mincut = efficient_extraction(
            features, deltas=[self.delta], quantiles=[self.quantile]
        )[0]

        for feats, mincut_boundaries in zip(features, mincut):
            mincut_boundaries = np.array(mincut_boundaries)
            meaned_features = torch.stack(
                [
                    feats[mincut_boundaries[idx] + 1 : mincut_boundaries[idx + 1] - 1].mean(dim=0)
                    for idx in range(len(mincut_boundaries) - 1)
                ]
            )
            meaned_features = (
                meaned_features - meaned_features.mean(dim=-1, keepdim=True)
            ) / meaned_features.std(dim=-1, keepdim=True)
            clusters = self.agglom[self.kmeans_centroids(meaned_features.float()).numpy()].reshape(-1)
            not_repeat_mask = ~np.insert((clusters[1:] == clusters[:-1]), 0, 0)
            not_repeat_mask_end = ~np.insert(
                (clusters[1:] == clusters[:-1]), clusters.shape[0] - 1, 0
            )
            clusters_with_times = np.stack(
                [
                    clusters[not_repeat_mask],
                    mincut_boundaries[:-1][not_repeat_mask],
                    mincut_boundaries[1:][not_repeat_mask_end],
                ]
            )
            result["clusters_with_times"].append(clusters_with_times)
        return result

    @torch.no_grad()
    def encode(self, waveform: np.ndarray) -> torch.LongTensor:
        result = self.forward(torch.tensor(waveform))
        clusters = result["clusters_with_times"][0][0].astype(np.int64)
        return torch.from_numpy(clusters)
