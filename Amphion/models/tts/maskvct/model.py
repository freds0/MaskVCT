import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .cfg import MaskVCTGuidance
from .pitch import SinusoidalPitchEmbedding
from .sampling import cosine_mask_prob, gumbel_noise, gumbel_sample, top_k


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("rotary embedding dimension must be even")
        self.dim = dim
        self.base = base

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        half = self.dim // 2
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, half, device=device, dtype=dtype) / half)
        )
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        return freqs.cos(), freqs.sin()


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    cos = torch.repeat_interleave(cos, 2, dim=-1)[None, None, :, :]
    sin = torch.repeat_interleave(sin, 2, dim=-1)[None, None, :, :]
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class MaskVCTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ffn_size: int, dropout: float):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_size, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        residual = x
        x = self.norm1(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.num_heads)

        cos, sin = self.rotary(x.shape[1], x.device, x.dtype)
        q, k = apply_rotary(q, k, cos, sin)

        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_mask = attention_mask[:, None, None, :].to(dtype=attn.dtype)
            attn = attn.masked_fill(~attn_mask.bool(), torch.finfo(attn.dtype).min)
        attn = attn.softmax(dim=-1)
        attn_out = torch.matmul(attn, v)
        attn_out = rearrange(attn_out, "b h t d -> b t (h d)")
        x = residual + self.dropout(self.out_proj(attn_out))

        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))
        return x


@dataclass
class MaskVCTOutput:
    logits: torch.Tensor
    mask: torch.Tensor
    target: torch.Tensor
    mode: str


class MaskVCT(nn.Module):
    def __init__(
        self,
        num_quantizers: int = 9,
        hidden_size: int = 1024,
        num_layers: int = 16,
        num_heads: int = 16,
        ffn_size: int = 4096,
        codebook_size: int = 1024,
        ling_codebook_size: int = 8192,
        cont_ling_dim: int = 1024,
        pitch_dim: int = 1024,
        dropout: float = 0.05,
        layer_drop: float = 0.05,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_size = ffn_size
        self.codebook_size = codebook_size
        self.ling_codebook_size = ling_codebook_size
        self.cont_ling_dim = cont_ling_dim
        self.pitch_dim = pitch_dim
        self.dropout = dropout
        self.layer_drop = layer_drop

        self.acoustic_embs = nn.ModuleList(
            [nn.Embedding(codebook_size, hidden_size) for _ in range(num_quantizers)]
        )
        self.acoustic_heads = nn.ModuleList(
            [nn.Linear(hidden_size, codebook_size) for _ in range(num_quantizers)]
        )
        self.layer_emb = nn.Embedding(num_quantizers, hidden_size)
        self.mask_emb = nn.Parameter(torch.zeros(hidden_size))

        self.prompt_mix = nn.Linear(hidden_size, hidden_size)
        self.ling_disc_emb = nn.Embedding(ling_codebook_size, hidden_size)
        self.ling_cont_proj = nn.Sequential(
            nn.Linear(cont_ling_dim, hidden_size * 4),
            nn.LayerNorm(hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.pitch_emb = SinusoidalPitchEmbedding(pitch_dim)
        self.pitch_proj = nn.Linear(pitch_dim, hidden_size)
        self.ctx_norm = nn.LayerNorm(hidden_size)

        self.blocks = nn.ModuleList(
            [
                MaskVCTBlock(hidden_size, num_heads, ffn_size, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_size)

    def _sum_acoustic_embeddings(self, tokens: torch.Tensor, start_codebook: int = 0) -> torch.Tensor:
        out = torch.zeros(tokens.shape[0], tokens.shape[1], self.hidden_size, device=tokens.device, dtype=self.mask_emb.dtype)
        for idx in range(start_codebook, tokens.shape[-1]):
            out = out + self.acoustic_embs[idx](tokens[:, :, idx])
        return out

    def _prompt_and_source_embeddings(
        self,
        prompt_tokens: torch.Tensor,
        source_tokens: torch.Tensor,
        ling_disc_prompt: Optional[torch.Tensor] = None,
        ling_disc_source: Optional[torch.Tensor] = None,
        ling_cont_prompt: Optional[torch.Tensor] = None,
        ling_cont_source: Optional[torch.Tensor] = None,
        pitch_prompt: Optional[torch.Tensor] = None,
        pitch_source: Optional[torch.Tensor] = None,
        mode: str = "all",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt = self._sum_acoustic_embeddings(prompt_tokens)
        source = self._sum_acoustic_embeddings(source_tokens)

        if mode in {"all", "spk_ling"}:
            if ling_cont_prompt is not None or ling_cont_source is not None:
                prompt = prompt + self.ling_cont_proj(ling_cont_prompt) if ling_cont_prompt is not None else prompt
                source = source + self.ling_cont_proj(ling_cont_source) if ling_cont_source is not None else source
            elif ling_disc_prompt is not None or ling_disc_source is not None:
                prompt = prompt + self.ling_disc_emb(ling_disc_prompt) if ling_disc_prompt is not None else prompt
                source = source + self.ling_disc_emb(ling_disc_source) if ling_disc_source is not None else source
            if mode == "all":
                if pitch_prompt is not None:
                    prompt = prompt + self.pitch_proj(self.pitch_emb(pitch_prompt))
                if pitch_source is not None:
                    source = source + self.pitch_proj(self.pitch_emb(pitch_source))
        elif mode == "ling":
            if ling_cont_prompt is not None or ling_cont_source is not None:
                prompt = self.ling_cont_proj(ling_cont_prompt) if ling_cont_prompt is not None else prompt
                source = self.ling_cont_proj(ling_cont_source) if ling_cont_source is not None else source
            elif ling_disc_prompt is not None or ling_disc_source is not None:
                prompt = self.ling_disc_emb(ling_disc_prompt) if ling_disc_prompt is not None else prompt
                source = self.ling_disc_emb(ling_disc_source) if ling_disc_source is not None else source
        elif mode == "none":
            pass
        else:
            raise ValueError(f"unknown mode: {mode}")

        prompt = self.prompt_mix(prompt)
        source = self.prompt_mix(source)
        return self.ctx_norm(prompt), self.ctx_norm(source)

    def _encode(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for block in self.blocks:
            if self.training and self.layer_drop > 0 and torch.rand(()) < self.layer_drop:
                continue
            x = block(x, attention_mask=attention_mask)
        return self.final_norm(x)

    def _build_masked_acoustic_input(
        self,
        acoustic_tokens: torch.Tensor,
        prompt_len: int,
        mask_layer: int,
        mask_prob: torch.Tensor,
        current_mask: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, total_len, _ = acoustic_tokens.shape
        hidden = torch.zeros(batch, total_len, self.hidden_size, device=acoustic_tokens.device, dtype=self.mask_emb.dtype)
        prompt_mask = prompt_mask if prompt_mask is not None else torch.ones(batch, prompt_len, device=acoustic_tokens.device, dtype=torch.bool)
        prompt_mask = prompt_mask.bool()
        source_mask = torch.ones(batch, total_len - prompt_len, device=acoustic_tokens.device, dtype=torch.bool)

        # Prompt tokens are always visible.
        hidden[:, :prompt_len, :] = self._sum_acoustic_embeddings(acoustic_tokens[:, :prompt_len, :], start_codebook=0)
        for idx in range(self.num_quantizers):
            if idx < mask_layer:
                hidden[:, prompt_len:, :] += self.acoustic_embs[idx](acoustic_tokens[:, prompt_len:, idx])
            elif idx == mask_layer:
                if current_mask is None:
                    mask = torch.bernoulli(
                        torch.ones(batch, total_len - prompt_len, device=acoustic_tokens.device) * mask_prob[:, None]
                    ).bool()
                else:
                    mask = current_mask.bool()
                mask = mask & source_mask
                for b in range(batch):
                    if mask[b].sum() == 0:
                        mask[b, 0] = True
                current = self.acoustic_embs[idx](acoustic_tokens[:, prompt_len:, idx])
                hidden[:, prompt_len:, :] += torch.where(
                    mask[..., None],
                    self.mask_emb[None, None, :],
                    current,
                )
            else:
                hidden[:, prompt_len:, :] += self.mask_emb[None, None, :]

        attn_mask = torch.cat([prompt_mask, source_mask], dim=1)
        return hidden, attn_mask

    def _mask_layer_distribution(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # p(q) = 1 - q(q+1)/(2(Q+1)) in the paper; we normalize by sampling weights.
        layers = torch.arange(self.num_quantizers, device=device, dtype=torch.float32)
        weights = 1.0 - (layers * (layers + 1.0)) / (2.0 * (self.num_quantizers + 1.0))
        # TODO(MaskVCT): The paper does not fully specify how to resolve negative
        # weights for the final quantizers when Q=9. Clamp to keep sampling valid.
        weights = weights.clamp_min(1e-6)
        weights = weights / weights.sum()
        return torch.multinomial(weights, batch_size, replacement=True)

    def _condition_modes(
        self,
        prompt_tokens: torch.Tensor,
        source_tokens: torch.Tensor,
        ling_disc_prompt: Optional[torch.Tensor],
        ling_disc_source: Optional[torch.Tensor],
        ling_cont_prompt: Optional[torch.Tensor],
        ling_cont_source: Optional[torch.Tensor],
        pitch_prompt: Optional[torch.Tensor],
        pitch_source: Optional[torch.Tensor],
        mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt_ctx, source_ctx = self._prompt_and_source_embeddings(
            prompt_tokens=prompt_tokens,
            source_tokens=source_tokens,
            ling_disc_prompt=ling_disc_prompt,
            ling_disc_source=ling_disc_source,
            ling_cont_prompt=ling_cont_prompt,
            ling_cont_source=ling_cont_source,
            pitch_prompt=pitch_prompt,
            pitch_source=pitch_source,
            mode=mode,
        )
        return torch.cat([prompt_ctx, source_ctx], dim=1), torch.ones(
            prompt_ctx.shape[0], prompt_ctx.shape[1] + source_ctx.shape[1], device=prompt_ctx.device, dtype=torch.bool
        )

    def forward(
        self,
        acoustic_tokens: torch.Tensor,
        prompt_len: int,
        ling_disc_prompt: Optional[torch.Tensor] = None,
        ling_disc_source: Optional[torch.Tensor] = None,
        ling_cont_prompt: Optional[torch.Tensor] = None,
        ling_cont_source: Optional[torch.Tensor] = None,
        pitch_prompt: Optional[torch.Tensor] = None,
        pitch_source: Optional[torch.Tensor] = None,
        mode: str = "all",
        mask_layer: Optional[torch.Tensor] = None,
        mask_prob: Optional[torch.Tensor] = None,
    ) -> MaskVCTOutput:
        batch, total_len, _ = acoustic_tokens.shape
        source_len = total_len - prompt_len
        if mask_layer is None:
            mask_layer = self._mask_layer_distribution(batch, acoustic_tokens.device)
        if mask_prob is None:
            t = torch.rand(batch, device=acoustic_tokens.device).clamp_(1e-5, 1.0)
            mask_prob = cosine_mask_prob(t).clamp_min(0.2)

        hidden, attn_mask = self._build_masked_acoustic_input(
            acoustic_tokens=acoustic_tokens,
            prompt_len=prompt_len,
            mask_layer=int(mask_layer[0].item()),
            mask_prob=mask_prob,
        )

        context, context_mask = self._condition_modes(
            prompt_tokens=acoustic_tokens[:, :prompt_len, :],
            source_tokens=acoustic_tokens[:, prompt_len:, :],
            ling_disc_prompt=ling_disc_prompt[:, :prompt_len] if ling_disc_prompt is not None else None,
            ling_disc_source=ling_disc_source[:, :source_len] if ling_disc_source is not None else None,
            ling_cont_prompt=ling_cont_prompt[:, :prompt_len, :] if ling_cont_prompt is not None else None,
            ling_cont_source=ling_cont_source[:, :source_len, :] if ling_cont_source is not None else None,
            pitch_prompt=pitch_prompt[:, :prompt_len] if pitch_prompt is not None else None,
            pitch_source=pitch_source[:, :source_len] if pitch_source is not None else None,
            mode=mode,
        )
        x = hidden + context
        x = self._encode(x, attention_mask=attn_mask & context_mask)
        logits = self.acoustic_heads[int(mask_layer[0].item())](x[:, prompt_len:, :])

        target = acoustic_tokens[:, prompt_len:, int(mask_layer[0].item())]
        masked_positions = torch.zeros(batch, source_len, device=acoustic_tokens.device, dtype=torch.bool)
        masked_positions[:] = True
        return MaskVCTOutput(logits=logits, mask=masked_positions, target=target, mode=mode)

    def _guided_logits(
        self,
        acoustic_tokens: torch.Tensor,
        prompt_len: int,
        current_mask: Optional[torch.Tensor] = None,
        ling_disc_prompt: Optional[torch.Tensor] = None,
        ling_disc_source: Optional[torch.Tensor] = None,
        ling_cont_prompt: Optional[torch.Tensor] = None,
        ling_cont_source: Optional[torch.Tensor] = None,
        pitch_prompt: Optional[torch.Tensor] = None,
        pitch_source: Optional[torch.Tensor] = None,
        mask_layer: int = 0,
        mode: str = "all",
    ) -> torch.Tensor:
        hidden, attn_mask = self._build_masked_acoustic_input(
            acoustic_tokens=acoustic_tokens,
            prompt_len=prompt_len,
            mask_layer=mask_layer,
            mask_prob=torch.ones(acoustic_tokens.shape[0], device=acoustic_tokens.device),
            current_mask=current_mask,
        )
        context, context_mask = self._condition_modes(
            prompt_tokens=acoustic_tokens[:, :prompt_len, :],
            source_tokens=acoustic_tokens[:, prompt_len:, :],
            ling_disc_prompt=ling_disc_prompt[:, :prompt_len] if ling_disc_prompt is not None else None,
            ling_disc_source=ling_disc_source[:, : acoustic_tokens.shape[1] - prompt_len] if ling_disc_source is not None else None,
            ling_cont_prompt=ling_cont_prompt[:, :prompt_len, :] if ling_cont_prompt is not None else None,
            ling_cont_source=ling_cont_source[:, : acoustic_tokens.shape[1] - prompt_len, :] if ling_cont_source is not None else None,
            pitch_prompt=pitch_prompt[:, :prompt_len] if pitch_prompt is not None else None,
            pitch_source=pitch_source[:, : acoustic_tokens.shape[1] - prompt_len] if pitch_source is not None else None,
            mode=mode,
        )
        x = hidden + context
        x = self._encode(x, attention_mask=attn_mask & context_mask)
        return self.acoustic_heads[mask_layer](x[:, prompt_len:, :])

    @torch.no_grad()
    def reverse_diffusion(
        self,
        acoustic_tokens: torch.Tensor,
        prompt_len: int,
        ling_disc_prompt: Optional[torch.Tensor] = None,
        ling_disc_source: Optional[torch.Tensor] = None,
        ling_cont_prompt: Optional[torch.Tensor] = None,
        ling_cont_source: Optional[torch.Tensor] = None,
        pitch_prompt: Optional[torch.Tensor] = None,
        pitch_source: Optional[torch.Tensor] = None,
        guidance: MaskVCTGuidance = MaskVCTGuidance(1.5, 1.0, 1.0),
        n_timesteps: Optional[Sequence[int]] = None,
        temp: float = 1.0,
        filter_thres: float = 0.9,
        mode: str = "all",
    ) -> torch.Tensor:
        if n_timesteps is None:
            n_timesteps = [64] * self.num_quantizers
        if len(n_timesteps) != self.num_quantizers:
            raise ValueError("n_timesteps must provide one schedule entry per codebook")

        batch, total_len, _ = acoustic_tokens.shape
        source_len = total_len - prompt_len
        device = acoustic_tokens.device
        state = acoustic_tokens.clone()
        state[:, prompt_len:, :] = 0
        seq = torch.zeros(batch, source_len, self.num_quantizers, dtype=torch.long, device=device)

        for layer in range(self.num_quantizers):
            steps = int(n_timesteps[layer])
            if steps < 1:
                steps = 1
            mask = torch.ones(batch, source_len, 1, device=device, dtype=torch.bool)
            choice_temp = 1.0
            start_temp = temp
            start_choice_temp = choice_temp
            h = 1.0 / steps
            t_list = [1.0 - i * h for i in range(steps)] + [0.0]

            for i in range(steps):
                current = state.clone()
                current[:, prompt_len:, layer] = seq[:, :, layer]

                logits_full = self._guided_logits(
                    acoustic_tokens=current,
                    prompt_len=prompt_len,
                    current_mask=mask.squeeze(-1),
                    ling_disc_prompt=ling_disc_prompt,
                    ling_disc_source=ling_disc_source,
                    ling_cont_prompt=ling_cont_prompt,
                    ling_cont_source=ling_cont_source,
                    pitch_prompt=pitch_prompt,
                    pitch_source=pitch_source,
                    mask_layer=layer,
                    mode="all",
                )
                if guidance.omega_all == guidance.omega_spk == guidance.omega_ling == 0:
                    logits = logits_full
                else:
                    logits_ling = self._guided_logits(
                        acoustic_tokens=current,
                        prompt_len=prompt_len,
                        current_mask=mask.squeeze(-1),
                        ling_disc_prompt=ling_disc_prompt,
                        ling_disc_source=ling_disc_source,
                        ling_cont_prompt=ling_cont_prompt,
                        ling_cont_source=ling_cont_source,
                        pitch_prompt=pitch_prompt,
                        pitch_source=pitch_source,
                        mask_layer=layer,
                        mode="ling",
                    )
                    logits_spk = self._guided_logits(
                        acoustic_tokens=current,
                        prompt_len=prompt_len,
                        current_mask=mask.squeeze(-1),
                        ling_disc_prompt=ling_disc_prompt,
                        ling_disc_source=ling_disc_source,
                        ling_cont_prompt=ling_cont_prompt,
                        ling_cont_source=ling_cont_source,
                        pitch_prompt=pitch_prompt,
                        pitch_source=pitch_source,
                        mask_layer=layer,
                        mode="spk_ling",
                    )
                    logits_none = self._guided_logits(
                        acoustic_tokens=current,
                        prompt_len=prompt_len,
                        current_mask=mask.squeeze(-1),
                        ling_disc_prompt=ling_disc_prompt,
                        ling_disc_source=ling_disc_source,
                        ling_cont_prompt=ling_cont_prompt,
                        ling_cont_source=ling_cont_source,
                        pitch_prompt=pitch_prompt,
                        pitch_source=pitch_source,
                        mask_layer=layer,
                        mode="none",
                    )
                    logits = logits_ling
                    logits = logits + guidance.omega_all * (logits_full - logits_ling)
                    logits = logits + guidance.omega_spk * (logits_spk - logits_ling)
                    logits = logits + guidance.omega_ling * (logits_ling - logits_none)

                logits = top_k(logits, filter_thres)
                annealing_scale = t_list[i]
                choice_temp = start_choice_temp * annealing_scale
                temp = start_temp * annealing_scale

                if i == steps - 1:
                    sampled = logits.argmax(dim=-1) if steps > 1 else gumbel_sample(logits, temperature=max(temp, 1e-3))
                else:
                    sampled = gumbel_sample(logits, temperature=max(temp, 1e-3))

                seq[:, :, layer] = torch.where(mask.squeeze(-1), sampled, seq[:, :, layer])
                state[:, prompt_len:, layer] = seq[:, :, layer]
                scores = logits.softmax(dim=-1).gather(2, sampled.unsqueeze(-1)).squeeze(-1)
                scores = choice_temp * gumbel_noise(scores) + scores
                scores = 1 - scores
                next_mask_num = int((cosine_mask_prob(torch.full((batch,), t_list[i + 1], device=device)) * source_len).long()[0].item())
                if next_mask_num == 0:
                    break
                scores = scores.masked_fill(~mask.squeeze(-1), -torch.finfo(scores.dtype).max)
                next_mask = torch.zeros_like(scores, dtype=torch.bool).scatter(1, scores.topk(next_mask_num, dim=-1).indices, True)
                seq[:, :, layer] = seq[:, :, layer].masked_fill(next_mask, 0)
                state[:, prompt_len:, layer] = seq[:, :, layer]
                mask = next_mask.unsqueeze(-1)

        return seq
