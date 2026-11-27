from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cinlm.config import CinLMConfig


class ReLU2MLP(nn.Module):
    """Squared-ReLU MLP: W_down(relu(W_up(x))^2).

    Why ReLU² for CinLM:
    - Competitive quality with SwiGLU at equal parameter count (2 matrices instead of 3).
    - Naturally induces ~80-90% activation sparsity, reducing polysemanticity and making
      neurons directly inspectable by Frontier LLMs and downstream SAEs/Transcoders.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        return_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        act = F.relu(self.w_up(x)).square()
        out = self.w_down(act)
        stats: Optional[Dict[str, torch.Tensor]] = None
        if return_stats:
            zero_frac = (act == 0).float().mean(dim=-1).detach()  # (...)
            stats = {"relu2_zero_frac": zero_frac}
        return out, stats


class SparseReLU2MoE(nn.Module):
    """Fine-grained Sparse ReLU² MoE with Shared Expert + Auxiliary-Loss-Free Load Balancing.

    Inspired by DeepSeek-V3/V4:
    - 1 always-active Shared Expert captures common syntax/grammar.
    - N fine-grained Routed Experts with top-k sigmoid routing (normalized over top-k).
    - True sparse token dispatch: each expert runs ONLY on tokens routed to it.
    - Auxiliary-loss-free load balancing via dynamic per-expert routing bias updated outside autograd.
    """

    def __init__(self, config: CinLMConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        self.n_experts = config.n_routed_experts
        self.top_k = config.moe_top_k
        self.bias_update_rate = config.moe_bias_update_rate

        self.shared_expert = ReLU2MLP(config.d_model, config.d_shared_expert)
        self.experts = nn.ModuleList(
            [ReLU2MLP(config.d_model, config.d_routed_expert) for _ in range(self.n_experts)]
        )
        self.router = nn.Linear(config.d_model, self.n_experts, bias=False)

        # Auxiliary-loss-free balancing state (not trained by gradient descent)
        self.register_buffer("expert_bias", torch.zeros(self.n_experts, dtype=torch.float32))
        self.register_buffer("_last_expert_counts", torch.zeros(self.n_experts, dtype=torch.float32), persistent=False)

    @torch.no_grad()
    def update_router_bias(self) -> None:
        """DeepSeek-style auxiliary-loss-free bias update based on last forward load."""
        counts = self._last_expert_counts
        total = counts.sum()
        if total <= 0:
            return
        avg = total / self.n_experts
        # Increase bias for underloaded experts, decrease for overloaded ones
        delta = torch.sign(avg - counts) * self.bias_update_rate
        self.expert_bias.add_(delta)
        # Keep mean centered at 0 to avoid drift
        self.expert_bias.sub_(self.expert_bias.mean())

    def forward(
        self,
        x: torch.Tensor,
        ablate_experts: Optional[Sequence[int]] = None,
        ablate_shared: bool = False,
        return_telemetry: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        bsz, seq_len, d_model = x.shape
        n_tokens = bsz * seq_len
        x_flat = x.reshape(n_tokens, d_model)

        # 1. Shared expert (always active unless ablated)
        if ablate_shared:
            shared_out = torch.zeros_like(x_flat)
            shared_zero_frac = torch.ones((bsz, seq_len), device=x.device)
        else:
            shared_out, shared_stats = self.shared_expert(x_flat, return_stats=return_telemetry)
            shared_zero_frac = (
                shared_stats["relu2_zero_frac"].view(bsz, seq_len)
                if shared_stats is not None
                else None
            )

        # 2. Router scores (sigmoid affinity + dynamic bias ONLY for top-k selection)
        router_logits = self.router(x_flat).float()  # (N, E)
        affinities = torch.sigmoid(router_logits)  # (N, E)
        selection_scores = affinities + self.expert_bias.unsqueeze(0)

        _, topk_idx = torch.topk(selection_scores, k=self.top_k, dim=-1)  # (N, K)
        # Gather unbiased affinities and normalize over top-k
        topk_weights = torch.gather(affinities, dim=-1, index=topk_idx)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)
        topk_weights = topk_weights.to(x.dtype)

        # Track load counts for auxiliary-loss-free balancing
        with torch.no_grad():
            counts = torch.bincount(topk_idx.reshape(-1), minlength=self.n_experts).float()
            self._last_expert_counts.copy_(counts)

        # 3. True sparse token dispatch per expert (single argsort + contiguous slices, 1 sync/layer)
        routed_out = torch.zeros_like(x_flat)
        ablate_set = set(ablate_experts) if ablate_experts else set()

        flat_expert_ids = topk_idx.reshape(-1)  # (N * K,)
        flat_token_ids = torch.arange(n_tokens, device=x.device).repeat_interleave(self.top_k)
        flat_weights = topk_weights.reshape(-1)  # (N * K,)

        sort_idx = torch.argsort(flat_expert_ids, stable=False)
        sorted_tok_ids = flat_token_ids.index_select(0, sort_idx)
        sorted_weights = flat_weights.index_select(0, sort_idx).unsqueeze(-1)
        sorted_x = x_flat.index_select(0, sorted_tok_ids)

        counts_list = counts.to(dtype=torch.long).tolist()
        routed_zero_sum = torch.zeros(n_tokens, device=x.device, dtype=torch.float32) if return_telemetry else None

        offset = 0
        for exp_id, cnt in enumerate(counts_list):
            if cnt == 0:
                continue
            end = offset + cnt
            if exp_id not in ablate_set:
                exp_in = sorted_x[offset:end]
                exp_out, exp_stats = self.experts[exp_id](exp_in, return_stats=return_telemetry)
                tok_ids = sorted_tok_ids[offset:end]
                routed_out.index_add_(0, tok_ids, exp_out * sorted_weights[offset:end])
                if return_telemetry and exp_stats is not None and routed_zero_sum is not None:
                    routed_zero_sum.index_add_(0, tok_ids, exp_stats["relu2_zero_frac"] / self.top_k)
            offset = end

        out = (shared_out + routed_out).view(bsz, seq_len, d_model)

        telemetry: Optional[Dict[str, torch.Tensor]] = None
        if return_telemetry:
            telemetry = {
                "topk_experts": topk_idx.view(bsz, seq_len, self.top_k).detach(),
                "topk_weights": topk_weights.view(bsz, seq_len, self.top_k).detach(),
                "expert_counts": counts.detach(),
                "shared_out": shared_out.view(bsz, seq_len, d_model).detach(),
                "routed_out": routed_out.view(bsz, seq_len, d_model).detach(),
                "shared_relu2_zero_frac": shared_zero_frac.detach() if shared_zero_frac is not None else torch.zeros((bsz, seq_len), device=x.device),
                "routed_relu2_zero_frac": routed_zero_sum.view(bsz, seq_len).detach() if routed_zero_sum is not None else torch.zeros((bsz, seq_len), device=x.device),
            }
        return out, telemetry
