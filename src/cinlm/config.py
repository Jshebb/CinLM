from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple


@dataclass
class CinLMConfig:
    """Configuration for CinLM — Hybrid Attention + Engram + Sparse ReLU² MoE.

    Designed around four principles:
    1. Iso-active parameter comparability (~100M active across ablation ladder).
    2. Hybrid Attention (Sliding Window + periodic Global Attention) with GQA, RoPE & QK-Norm.
    3. Explicit static memory via Engram (multi-head n-gram hashing + scalar context gate).
    4. Native interpretability for Frontier LLMs (single additive residual stream,
       ReLU² natural activation sparsity, explicit router/gate/attention telemetry).
    """

    name: str = "cinlm-260m-a100m"
    vocab_size: int = 32768
    d_model: int = 640
    n_layers: int = 16
    n_heads: int = 10
    n_kv_heads: int = 2
    max_seq_len: int = 2048
    rope_theta: float = 100000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    dropout: float = 0.0

    # Hybrid Attention schedule: every `global_attn_every` layer (0-indexed: (i+1) % k == 0)
    # uses Global Causal Attention; the rest use Sliding Window Attention (SWA).
    # Set global_attn_every=1 for pure Global Attention baseline.
    global_attn_every: int = 4
    swa_window_size: int = 256
    use_qk_norm: bool = True

    # Feed-Forward / MoE (always ReLU² for high quality + natural activation sparsity)
    use_moe: bool = True
    d_ff_dense: int = 3072  # Used when use_moe=False (iso-active with 1 shared + 2 routed * 1024)
    d_shared_expert: int = 1024
    d_routed_expert: int = 1024
    n_routed_experts: int = 8
    moe_top_k: int = 2
    moe_bias_update_rate: float = 1e-3  # Auxiliary-loss-free load balancing (DeepSeek-V3/V4 style)

    # Engram Conditional Memory (arXiv:2601.07372)
    use_engram: bool = True
    engram_layers: Tuple[int, ...] = (1, 5)
    engram_ngram_orders: Tuple[int, ...] = (2, 3)
    engram_heads_per_order: int = 4
    engram_head_dim: int = 64
    engram_table_sizes: Tuple[int, ...] = (32771, 32779, 32783, 32789, 32797, 32801, 32803, 32831)
    engram_conv_kernel: int = 4

    # Looped Core / Adaptive Recursive Depth (Ouro / Parcae style, function-preserving at R=1)
    # Prelude = layers[0:loop_start_layer], Core = layers[loop_start_layer:loop_end_layer], Coda = layers[loop_end_layer:]
    loop_core_steps: int = 1
    loop_start_layer: int = 4
    loop_end_layer: int = 12
    loop_max_steps: int = 4

    # Multi-Token Prediction auxiliary heads (DeepSeek-V3 style, 0 = disabled, zero inference overhead)
    mtp_num_heads: int = 0
    mtp_loss_weight: float = 0.3

    # Initialization
    init_std: float = 0.02

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.global_attn_every >= 1, "global_attn_every must be >= 1"
        assert self.loop_core_steps >= 1, "loop_core_steps must be >= 1"
        assert self.loop_max_steps >= self.loop_core_steps, "loop_max_steps must be >= loop_core_steps"
        if self.loop_core_steps > 1:
            assert 0 <= self.loop_start_layer < self.loop_end_layer <= self.n_layers, (
                f"Invalid loop core slice [{self.loop_start_layer}, {self.loop_end_layer}) for n_layers={self.n_layers}"
            )
        assert self.mtp_num_heads >= 0, "mtp_num_heads must be >= 0"
        if self.use_moe:
            assert 1 <= self.moe_top_k <= self.n_routed_experts
        if self.use_engram:
            total_heads = len(self.engram_ngram_orders) * self.engram_heads_per_order
            assert len(self.engram_table_sizes) >= total_heads, (
                f"Need at least {total_heads} prime table sizes for Engram, got {len(self.engram_table_sizes)}"
            )
            for layer_idx in self.engram_layers:
                assert 0 <= layer_idx < self.n_layers, f"Invalid engram layer {layer_idx}"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def is_global_layer(self, layer_idx: int) -> bool:
        return (layer_idx + 1) % self.global_attn_every == 0

    def is_engram_layer(self, layer_idx: int) -> bool:
        return self.use_engram and (layer_idx in self.engram_layers)

    def estimate_parameters(self) -> Dict[str, int]:
        """Analytical parameter breakdown: total vs active-per-token."""
        emb = self.vocab_size * self.d_model
        lm_head = 0 if self.tie_word_embeddings else self.vocab_size * self.d_model

        # Per attention layer: Q, K, V, O + RMSNorms
        kv_dim = self.n_kv_heads * self.head_dim
        attn_per_layer = (
            self.d_model * self.d_model  # Q
            + self.d_model * kv_dim  # K
            + self.d_model * kv_dim  # V
            + self.d_model * self.d_model  # O
            + self.d_model  # pre-attn norm
            + (2 * self.head_dim if self.use_qk_norm else 0)
        )
        attn_total = attn_per_layer * self.n_layers

        # Per FFN/MoE layer (ReLU² uses 2 linear projections: up + down)
        if self.use_moe:
            shared = 2 * self.d_model * self.d_shared_expert
            one_routed = 2 * self.d_model * self.d_routed_expert
            router = self.d_model * self.n_routed_experts
            ffn_total_per_layer = shared + self.n_routed_experts * one_routed + router + self.d_model
            ffn_active_per_layer = shared + self.moe_top_k * one_routed + router + self.d_model
        else:
            ffn_total_per_layer = 2 * self.d_model * self.d_ff_dense + self.d_model
            ffn_active_per_layer = ffn_total_per_layer

        ffn_total = ffn_total_per_layer * self.n_layers
        ffn_active = ffn_active_per_layer * self.n_layers

        # Engram layers
        engram_table_total = 0
        engram_proj_active = 0
        if self.use_engram:
            total_heads = len(self.engram_ngram_orders) * self.engram_heads_per_order
            engram_dim = total_heads * self.engram_head_dim
            per_layer_tables = sum(self.engram_table_sizes[:total_heads]) * self.engram_head_dim
            # Projections: W_k (engram_dim -> d_model), W_v (engram_dim -> d_model),
            # norms (h_norm, k_norm, conv_norm = 3 * d_model), depthwise conv (d_model * kernel)
            per_layer_proj = (
                2 * engram_dim * self.d_model
                + 3 * self.d_model
                + self.d_model * self.engram_conv_kernel
            )
            engram_table_total = per_layer_tables * len(self.engram_layers)
            engram_proj_active = per_layer_proj * len(self.engram_layers)

        final_norm = self.d_model
        loop_params = (
            self.loop_max_steps * self.d_model + self.loop_max_steps
            if self.loop_core_steps > 1
            else 0
        )
        # Per MTP head: 3 RMSNorms (3 * d_model) + Linear(2*d_model -> d_model)
        mtp_params = (
            self.mtp_num_heads * (2 * self.d_model * self.d_model + 3 * self.d_model)
            if self.mtp_num_heads > 0
            else 0
        )
        total = (
            emb
            + lm_head
            + attn_total
            + ffn_total
            + engram_table_total
            + engram_proj_active
            + final_norm
            + loop_params
            + mtp_params
        )
        # Engram lookups are O(1) sparse row reads (not dense matmul weights), so active compute
        # excludes the unaccessed table rows. MTP heads are training-only (zero inference cost).
        active = emb + lm_head + attn_total + ffn_active + engram_proj_active + final_norm + loop_params

        return {
            "total": total,
            "active_per_token": active,
            "embeddings": emb + lm_head,
            "attention": attn_total,
            "ffn_total": ffn_total,
            "ffn_active": ffn_active,
            "engram_tables": engram_table_total,
            "engram_projections": engram_proj_active,
            "loop_params": loop_params,
            "mtp_params": mtp_params,
        }


PresetName = Literal[
    "smoke",
    "dense_global_100m",
    "dense_hybrid_100m",
    "hybrid_engram_100m",
    "full_260m_a100m",
]


def get_preset_config(preset: PresetName) -> CinLMConfig:
    """Return scientifically matched presets for the CinLM ablation ladder."""
    if preset == "smoke":
        return CinLMConfig(
            name="cinlm-smoke",
            vocab_size=512,
            d_model=128,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=128,
            global_attn_every=2,
            swa_window_size=16,
            use_moe=True,
            d_shared_expert=128,
            d_routed_expert=128,
            n_routed_experts=4,
            moe_top_k=2,
            use_engram=True,
            engram_layers=(1,),
            engram_ngram_orders=(2, 3),
            engram_heads_per_order=2,
            engram_head_dim=32,
            engram_table_sizes=(257, 263, 269, 271),
        )
    if preset == "dense_global_100m":
        return CinLMConfig(
            name="cinlm-dense-global-100m",
            global_attn_every=1,
            use_moe=False,
            use_engram=False,
        )
    if preset == "dense_hybrid_100m":
        return CinLMConfig(
            name="cinlm-dense-hybrid-100m",
            global_attn_every=4,
            swa_window_size=256,
            use_moe=False,
            use_engram=False,
        )
    if preset == "hybrid_engram_100m":
        return CinLMConfig(
            name="cinlm-hybrid-engram-134m-a101m",
            global_attn_every=4,
            swa_window_size=256,
            use_moe=False,
            use_engram=True,
        )
    if preset == "full_260m_a100m":
        return CinLMConfig(
            name="cinlm-full-260m-a100m",
            global_attn_every=4,
            swa_window_size=256,
            use_moe=True,
            use_engram=True,
        )
    raise ValueError(f"Unknown preset: {preset}")
