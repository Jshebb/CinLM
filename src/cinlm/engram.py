from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cinlm.attention import RMSNorm
from cinlm.config import CinLMConfig


@dataclass
class EngramState:
    """Minimal causal state for exact streaming parity in EngramMemory."""

    prev_tokens: torch.Tensor  # (B, max_n - 1)
    prev_conv_inputs: torch.Tensor  # (B, d_model, conv_kernel - 1)


class EngramMemory(nn.Module):
    """Conditional Memory via Scalable N-gram Lookup (inspired by DeepSeek Engram, arXiv:2601.07372).

    Properties:
    - Strictly causal: n-gram at position t depends ONLY on tokens <= t, and depthwise conv is left-padded.
    - O(1) deterministic multi-head hashed table lookups (friendly to host-RAM prefetching).
    - Explicit scalar context-aware gate alpha_t in (0, 1) for every token, enabling direct
      interpretability and causal ablation by Frontier LLMs.
    """

    # Distinct large odd multipliers per hash head for collision-resistant multiplicative hashing
    _HASH_PRIMES: Tuple[int, ...] = (
        1000003,
        1000033,
        1000037,
        1000039,
        1000081,
        1000099,
        1000117,
        1000121,
        1000133,
        1000151,
        1000159,
        1000171,
        1000183,
        1000187,
        1000193,
        1000199,
    )

    def __init__(self, config: CinLMConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        self.ngram_orders = config.engram_ngram_orders
        self.max_n = max(self.ngram_orders)
        self.heads_per_order = config.engram_heads_per_order
        self.head_dim = config.engram_head_dim
        self.total_heads = len(self.ngram_orders) * self.heads_per_order
        self.engram_dim = self.total_heads * self.head_dim
        self.conv_kernel = config.engram_conv_kernel

        self.table_sizes = list(config.engram_table_sizes[: self.total_heads])
        self.tables = nn.ModuleList(
            [nn.Embedding(size, self.head_dim) for size in self.table_sizes]
        )
        for tbl in self.tables:
            nn.init.normal_(tbl.weight, mean=0.0, std=config.init_std)

        # Register per-head deterministic hash multipliers (shape: total_heads, max_n)
        multipliers = []
        for h_idx in range(self.total_heads):
            row = []
            for pos in range(self.max_n):
                p_idx = (h_idx * 3 + pos * 5 + layer_idx * 7) % len(self._HASH_PRIMES)
                row.append(self._HASH_PRIMES[p_idx] + 2 * (h_idx + 1) * (pos + 1))
            multipliers.append(row)
        self.register_buffer(
            "hash_multipliers",
            torch.tensor(multipliers, dtype=torch.long),
            persistent=True,
        )

        # Context-aware gating projections (key & value from retrieved n-gram vector)
        self.w_k = nn.Linear(self.engram_dim, self.d_model, bias=False)
        self.w_v = nn.Linear(self.engram_dim, self.d_model, bias=False)
        self.h_norm = RMSNorm(self.d_model, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.d_model, eps=config.rms_norm_eps)
        self.conv_norm = RMSNorm(self.d_model, eps=config.rms_norm_eps)

        # Short causal depthwise convolution (left-padded manually for exact causality & streaming)
        self.conv = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            kernel_size=self.conv_kernel,
            groups=self.d_model,
            bias=False,
        )
        # Initialize conv close to zero so Engram starts as a gentle residual perturbation
        nn.init.zeros_(self.conv.weight)

    def _compute_ngram_hashes(
        self,
        input_ids: torch.Tensor,
        prev_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Compute strictly causal n-gram hash indices for every head.

        Args:
            input_ids: (B, T)
            prev_tokens: Optional (B, max_n - 1) preceding tokens for streaming decode.
        Returns:
            list of (B, T) index tensors (one per head), and updated prev_tokens (B, max_n - 1).
        """
        bsz, seq_len = input_ids.shape
        pad_len = self.max_n - 1
        if prev_tokens is None:
            prefix = torch.zeros((bsz, pad_len), dtype=torch.long, device=input_ids.device)
        else:
            prefix = prev_tokens.to(device=input_ids.device, dtype=torch.long)

        full_ids = torch.cat([prefix, input_ids.long()], dim=1)  # (B, pad_len + T)
        new_prev_tokens = full_ids[:, -pad_len:].detach() if pad_len > 0 else full_ids[:, :0]

        head_indices: List[torch.Tensor] = []
        head_cursor = 0
        for order in self.ngram_orders:
            # Gather causal suffix (x_{t-order+1}, ..., x_t)
            # Token at offset o in [0, order-1] is at slice [pad_len - o : pad_len - o + T]
            for _ in range(self.heads_per_order):
                mod = self.table_sizes[head_cursor]
                h = torch.zeros((bsz, seq_len), dtype=torch.long, device=input_ids.device)
                for offset in range(order):
                    tok_slice = full_ids[:, pad_len - offset : pad_len - offset + seq_len]
                    mult = self.hash_multipliers[head_cursor, offset]
                    # XOR + multiplicative mixing, always non-negative before modulo
                    h = (h * mult + (tok_slice + 1) * (mult ^ (offset + 17))) & 0x7FFFFFFFFFFFFFFF
                head_indices.append(h % mod)
                head_cursor += 1

        return head_indices, new_prev_tokens

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        state: Optional[EngramState] = None,
        force_gate: Optional[float] = None,
        return_telemetry: bool = False,
    ) -> Tuple[torch.Tensor, EngramState, Optional[Dict[str, torch.Tensor]]]:
        bsz, seq_len, _ = hidden_states.shape
        prev_tokens = state.prev_tokens if state is not None else None
        head_indices, new_prev_tokens = self._compute_ngram_hashes(input_ids, prev_tokens)

        retrieved_parts = [tbl(idx) for tbl, idx in zip(self.tables, head_indices)]
        e_t = torch.cat(retrieved_parts, dim=-1).to(hidden_states.dtype)  # (B, T, engram_dim)

        k_t = self.w_k(e_t)
        v_t = self.w_v(e_t)

        # Scalar context-aware gate alpha_t in (0, 1)
        if force_gate is not None:
            alpha = torch.full(
                (bsz, seq_len, 1),
                float(force_gate),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        else:
            h_n = self.h_norm(hidden_states).float()
            k_n = self.k_norm(k_t).float()
            dot = (h_n * k_n).sum(dim=-1, keepdim=True) / (self.d_model ** 0.5)
            alpha = torch.sigmoid(dot).to(hidden_states.dtype)

        gated_v = alpha * v_t  # (B, T, d_model)

        # Strictly causal depthwise Conv1d with exact streaming state
        conv_in = self.conv_norm(gated_v).transpose(1, 2)  # (B, d_model, T)
        pad_c = self.conv_kernel - 1
        if pad_c > 0:
            if state is not None:
                conv_prefix = state.prev_conv_inputs.to(device=conv_in.device, dtype=conv_in.dtype)
            else:
                conv_prefix = torch.zeros(
                    (bsz, self.d_model, pad_c), dtype=conv_in.dtype, device=conv_in.device
                )
            conv_padded = torch.cat([conv_prefix, conv_in], dim=2)
            new_conv_inputs = conv_padded[:, :, -pad_c:].detach()
        else:
            conv_padded = conv_in
            new_conv_inputs = conv_in[:, :, :0].detach()

        conv_out = F.silu(self.conv(conv_padded)).transpose(1, 2)
        out = gated_v + conv_out

        new_state = EngramState(prev_tokens=new_prev_tokens, prev_conv_inputs=new_conv_inputs)
        telemetry: Optional[Dict[str, torch.Tensor]] = None
        if return_telemetry:
            telemetry = {
                "gate_alpha": alpha.squeeze(-1).detach(),  # (B, T)
                "output_norm": out.float().norm(dim=-1).detach(),  # (B, T)
            }
        return out, new_state, telemetry
