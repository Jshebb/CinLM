from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cinlm.config import CinLMConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(x.dtype)


def precompute_rope_cache(
    max_seq_len: int,
    head_dim: int,
    theta: float = 100000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # (T, head_dim / 2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings.

    Args:
        x: (B, H, T, D)
        cos, sin: (T, D/2)
    """
    x_even = x[..., 0::2].float()
    x_odd = x[..., 1::2].float()
    cos_b = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D/2)
    sin_b = sin.unsqueeze(0).unsqueeze(0)
    out_even = x_even * cos_b - x_odd * sin_b
    out_odd = x_even * sin_b + x_odd * cos_b
    out = torch.stack((out_even, out_odd), dim=-1).flatten(-2)
    return out.to(x.dtype)


def build_hybrid_causal_mask(
    q_len: int,
    kv_len: int,
    start_pos: int,
    window_size: Optional[int],
    device: torch.device,
) -> torch.Tensor:
    """Return boolean mask of shape (q_len, kv_len) where True means ALLOWED to attend."""
    q_pos = torch.arange(start_pos, start_pos + q_len, device=device).unsqueeze(1)
    k_pos = torch.arange(0, kv_len, device=device).unsqueeze(0)
    mask = k_pos <= q_pos
    if window_size is not None:
        mask = mask & ((q_pos - k_pos) < window_size)
    return mask


class HybridAttention(nn.Module):
    """Causal Grouped-Query Attention supporting Global or Sliding Window Attention (SWA),
    QK-RMSNorm, RoPE, exact KV-cache streaming, head-level causal ablation, and attention weight capture.
    """

    def __init__(self, config: CinLMConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_global = config.is_global_layer(layer_idx)
        self.window_size: Optional[int] = None if self.is_global else config.swa_window_size

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.use_qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.use_qk_norm else None
        self.dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_attn_weights: bool = False,
        ablate_heads: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        cos_slice = rope_cos[start_pos : start_pos + seq_len]
        sin_slice = rope_sin[start_pos : start_pos + seq_len]
        q = apply_rope(q, cos_slice, sin_slice)
        k = apply_rope(k, cos_slice, sin_slice)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)

        # Keep full prefix in cache for exact parity; callers can trim SWA cache beyond window if desired
        new_cache = (k, v)

        if self.n_rep > 1:
            k_exp = k.repeat_interleave(self.n_rep, dim=1)
            v_exp = v.repeat_interleave(self.n_rep, dim=1)
        else:
            k_exp, v_exp = k, v

        kv_len = k_exp.shape[2]
        attn_weights_out: Optional[torch.Tensor] = None

        if return_attn_weights:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q.float(), k_exp.float().transpose(-2, -1)) * scale
            mask = build_hybrid_causal_mask(seq_len, kv_len, start_pos, self.window_size, x.device)
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            probs = F.softmax(scores, dim=-1)
            attn_weights_out = probs.detach()
            y = torch.matmul(probs.to(v_exp.dtype), v_exp)
        else:
            if self.is_global and kv_cache is None:
                # Fast path: PyTorch FlashAttention / SDPA causal kernel
                y = F.scaled_dot_product_attention(
                    q,
                    k_exp,
                    v_exp,
                    is_causal=True,
                    dropout_p=self.dropout if self.training else 0.0,
                )
            else:
                mask = build_hybrid_causal_mask(seq_len, kv_len, start_pos, self.window_size, x.device)
                y = F.scaled_dot_product_attention(
                    q,
                    k_exp,
                    v_exp,
                    attn_mask=mask.unsqueeze(0).unsqueeze(0),
                    dropout_p=self.dropout if self.training else 0.0,
                )

        if ablate_heads:
            # Zero out specific attention heads before o_proj for causal interpretability tests
            y = y.clone()
            for head_idx in ablate_heads:
                if 0 <= head_idx < self.n_heads:
                    y[:, head_idx, :, :] = 0.0

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        out = self.o_proj(y)
        return out, new_cache, attn_weights_out
